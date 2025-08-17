
from __future__ import annotations
import os
import math
import random
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =============== ユーティリティ ===============
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: [N,2], b:[M,2] -> [N,M]"""
    return torch.cdist(a, b, p=2)


# =============== データ抽出 ===============
class NuScenesPedSeqs(Dataset):
    """nuScenes mini から歩行者の連続軌跡を切り出して学習用シーケンスを作る。
    - 各インスタンス（歩行者）ごとに連続 T_obs + T_pred のウィンドウを抽出
    - 位置はシーン座標 (x,y) [メートル]
    - 返り値: past_abs [T_obs,2], future_abs [T_pred,2]
    """
    def __init__(self, dataroot: str, version: str = 'v1.0-mini', T_obs: int = 8, T_pred: int = 12,
                 split: str = 'train', split_ratio: Tuple[float,float,float] = (0.7,0.15,0.15)):
        super().__init__()
        from nuscenes.nuscenes import NuScenes
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_total = T_obs + T_pred
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        # 全シーンを時系列で走査し、歩行者 instance ごとの連続位置列を作る
        all_trajs: List[np.ndarray] = []  # [L,2]
        for scene in nusc.scene:
            token = scene['first_sample_token']
            # instance_token -> list of positions
            tracks: Dict[str, List[List[float]]] = {}
            timestamps: Dict[str, List[int]] = {}
            while token:
                sample = nusc.get('sample', token)
                next_token = sample['next']
                for ann_token in sample['anns']:
                    ann = nusc.get('sample_annotation', ann_token)
                    cat = ann['category_name']
                    if not cat.startswith('human.pedestrian'):
                        continue
                    iid = ann['instance_token']
                    pos = ann['translation'][:2]  # [x,y]
                    if iid not in tracks:
                        tracks[iid] = []
                        timestamps[iid] = []
                    tracks[iid].append(pos)
                    timestamps[iid].append(sample['timestamp'])
                token = next_token
                if token == '':
                    break
            # 連続なフレームのみ抽出（タイムスタンプの欠落を除外）
            for iid, xy_list in tracks.items():
                ts = np.array(timestamps[iid])
                xy = np.array(xy_list, dtype=np.float32)
                # 欠損で飛んでいる箇所で切る
                if len(ts) < self.T_total:
                    continue
                # だいたい 0.5s (2Hz) 間隔。差が一定かどうかで連続区間を検出
                dt = np.diff(ts)
                # 許容: モードの±10% 以内
                if len(dt) == 0:
                    continue
                mode_dt = np.bincount((dt // 1e5).astype(int)).argmax() * 1e5
                breaks = np.where(np.abs(dt - mode_dt) > 0.2 * mode_dt)[0]
                start = 0
                breaks = list(breaks) + [len(xy) - 1]
                for br in breaks:
                    end = br + 1
                    seg = xy[start:end]
                    if len(seg) >= self.T_total:
                        all_trajs.append(seg)
                    start = end
        # ウィンドウスライス
        windows: List[Tuple[np.ndarray, np.ndarray]] = []
        for traj in all_trajs:
            L = len(traj)
            for s in range(0, L - self.T_total + 1):
                past = traj[s : s + self.T_obs]
                future = traj[s + self.T_obs : s + self.T_total]
                windows.append((past, future))
        # シャッフル & スプリット
        rng = np.random.RandomState(42)
        idx = np.arange(len(windows))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        if split == 'train':
            sel = idx[:n_train]
        elif split == 'val':
            sel = idx[n_train:n_train + n_val]
        else:
            sel = idx[n_train + n_val:]
        self.samples = [windows[i] for i in sel]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        past_abs, future_abs = self.samples[i]
        # 原点を観測開始点に合わせた相対座標も計算
        origin = past_abs[0]
        past_rel = past_abs - origin
        future_rel = future_abs - origin
        return {
            'past_abs': torch.from_numpy(past_abs).float(),      # [T_obs,2]
            'future_abs': torch.from_numpy(future_abs).float(),  # [T_pred,2]
            'past_rel': torch.from_numpy(past_rel).float(),
            'future_rel': torch.from_numpy(future_rel).float(),
            'origin': torch.from_numpy(origin).float(),
        }


# =============== モデル ===============
class TrajLSTM(nn.Module):
    """Encoder-Decoder LSTM（相対変位をオートレグレッシブに予測）"""
    def __init__(self, d_model: int = 128, num_layers: int = 2):
        super().__init__()
        self.enc = nn.LSTM(input_size=2, hidden_size=d_model, num_layers=num_layers, batch_first=True)
        self.dec = nn.LSTM(input_size=2, hidden_size=d_model, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(d_model, 2)

    def forward(self, past_rel: torch.Tensor, T_pred: int, teacher_forcing: bool = True, future_rel: torch.Tensor | None = None):
        """
        past_rel: [B, T_obs, 2]
        future_rel: [B, T_pred, 2] (教師強制用)
        return: future_rel_pred [B,T_pred,2]
        """
        B = past_rel.size(0)
        enc_out, (h, c) = self.enc(past_rel)
        # デコーダ初期入力は最後の相対ステップ（またはゼロ）
        y = past_rel[:, -1, :].unsqueeze(1)  # [B,1,2]
        preds = []
        for t in range(T_pred):
            dec_out, (h, c) = self.dec(y, (h, c))
            step = self.out(dec_out)  # [B,1,2] 相対変位（次ステップとの差分）
            preds.append(step)
            if teacher_forcing and future_rel is not None:
                y = future_rel[:, t:t+1, :]
            else:
                y = step
        return torch.cat(preds, dim=1)


# =============== 評価指標 ===============
@torch.no_grad()
def compute_metrics(future_abs_pred: torch.Tensor, future_abs_gt: torch.Tensor) -> Dict[str, float]:
    """future_abs_*: [B,T_pred,2]"""
    diff = future_abs_pred - future_abs_gt
    dist = diff.norm(dim=-1)  # [B,T]
    ade = dist.mean().item()
    fde = dist[:, -1].mean().item()
    mr2 = (dist[:, -1] > 2.0).float().mean().item()
    return {'ADE': ade, 'FDE': fde, 'MR@2m': mr2}


# =============== 学習ループ ===============
def train_one_epoch(model, loader, opt, device, T_pred, clip=1.0):
    model.train()
    total_loss = 0.0
    crit = nn.L1Loss()  # L1 は外れ値にややロバスト
    for batch in loader:
        past_rel = batch['past_rel'].to(device)
        future_rel = batch['future_rel'].to(device)
        origin = batch['origin'].to(device)
        past_abs = batch['past_abs'].to(device)
        future_abs = batch['future_abs'].to(device)

        opt.zero_grad()
        pred_rel = model(past_rel, T_pred=T_pred, teacher_forcing=True, future_rel=future_rel)
        # 相対 → 絶対（原点 + 累積和）
        pred_abs = origin.unsqueeze(1) + torch.cumsum(pred_rel, dim=1)
        loss = crit(pred_abs, future_abs)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += loss.item() * past_rel.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, T_pred):
    model.eval()
    total = 0
    mets = {'ADE': 0.0, 'FDE': 0.0, 'MR@2m': 0.0}
    for batch in loader:
        past_rel = batch['past_rel'].to(device)
        future_rel = batch['future_rel'].to(device)
        origin = batch['origin'].to(device)
        future_abs = batch['future_abs'].to(device)
        pred_rel = model(past_rel, T_pred=T_pred, teacher_forcing=False)
        pred_abs = origin.unsqueeze(1) + torch.cumsum(pred_rel, dim=1)
        m = compute_metrics(pred_abs, future_abs)
        bs = past_rel.size(0)
        for k in mets:
            mets[k] += m[k] * bs
        total += bs
    for k in mets:
        mets[k] /= total
    return mets


# =============== メイン ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--obs', type=int, default=8)
    parser.add_argument('--pred', type=int, default=12)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str, default='./checkpoint_lstm.pt')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データセット
    train_ds = NuScenesPedSeqs(args.dataroot, args.version, args.obs, args.pred, 'train')
    val_ds = NuScenesPedSeqs(args.dataroot, args.version, args.obs, args.pred, 'val')
    test_ds = NuScenesPedSeqs(args.dataroot, args.version, args.obs, args.pred, 'test')

    if len(train_ds) == 0:
        print('データが見つかりません。--dataroot と nuScenes mini の配置を確認してください。')
        return

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    # モデル
    model = TrajLSTM(d_model=128, num_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_ld, opt, device, T_pred=args.pred)
        val_m = evaluate(model, val_ld, device, T_pred=args.pred)
        print(f"[Epoch {ep:02d}] train L1: {tr_loss:.4f} | val ADE: {val_m['ADE']:.3f} FDE: {val_m['FDE']:.3f} MR@2m: {val_m['MR@2m']:.3f}")
        if val_m['ADE'] < best_val:
            best_val = val_m['ADE']
            torch.save({'model': model.state_dict(), 'args': vars(args)}, args.save)
            print(f"  -> checkpoint saved to {args.save}")

    # テスト評価
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_m = evaluate(model, test_ld, device, T_pred=args.pred)
    print(f"[Test] ADE: {test_m['ADE']:.3f} FDE: {test_m['FDE']:.3f} MR@2m: {test_m['MR@2m']:.3f}")


if __name__ == '__main__':
    main()
