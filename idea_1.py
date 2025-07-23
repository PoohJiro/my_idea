import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nuscenes.nuscenes import NuScenes

# --------------------- 環境特徴量抽出 ---------------------
def extract_environment_features(lidar_points, trajectories):
    B, N_ped, T, _ = trajectories.shape
    static_density = torch.full((B, N_ped, 1), fill_value=lidar_points.size(1)/1000.0)
    nearest_dist = torch.rand(B, N_ped, 1) * 10
    pedestrian_density = torch.full((B, N_ped, 1), fill_value=N_ped/10.0)
    interaction_strength = torch.rand(B, N_ped, 1)
    return static_density, nearest_dist, pedestrian_density, interaction_strength

# --------------------- GATモジュール ---------------------
class SimpleGATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.attn_fc = nn.Linear(out_features*2, 1, bias=False)
    def forward(self, h):
        B, N, F_in = h.size()
        Wh = self.fc(h)
        Wh_repeat_i = Wh.unsqueeze(2).repeat(1,1,N,1)
        Wh_repeat_j = Wh.unsqueeze(1).repeat(1,N,1,1)
        e = self.attn_fc(torch.cat([Wh_repeat_i, Wh_repeat_j], dim=-1)).squeeze(-1)
        alpha = F.softmax(F.leaky_relu(e), dim=-1)
        h_prime = torch.bmm(alpha, Wh)
        return h_prime

# --------------------- Beam Model ---------------------
class BeamModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, features):
        return self.fc(features)

# --------------------- 環境適応型LSTM ---------------------
class EnvAdaptiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# --------------------- 統合モデル ---------------------
class TrajectoryPredictor(nn.Module):
    def __init__(self, gat_in_dim, gat_out_dim, beam_in_dim, beam_hidden_dim, lstm_input_dim, lstm_hidden_dim):
        super().__init__()
        self.gat = SimpleGATLayer(gat_in_dim, gat_out_dim)
        self.beam = BeamModel(beam_in_dim, beam_hidden_dim)
        self.env_lstm = EnvAdaptiveLSTM(lstm_input_dim, lstm_hidden_dim)
    def forward(self, traj_hist, env_feats):
        B, N, T, _ = traj_hist.shape
        gat_input = traj_hist[:, :, -1, :]
        gat_out = self.gat(gat_input)
        beam_out = self.beam(env_feats)
        first_pred = gat_out[:, :, :2] + beam_out
        pred_confidence = torch.sigmoid(-torch.norm(beam_out, dim=-1, keepdim=True))
        correction_needed = (pred_confidence < 0.5).float()
        lstm_in = torch.cat([traj_hist.reshape(B*N, T, 2), env_feats.unsqueeze(2).repeat(1,1,T,1).reshape(B*N, T, -1)], dim=-1)
        correction = self.env_lstm(lstm_in).view(B, N, 2)
        final_pred = first_pred + correction * correction_needed
        return final_pred, pred_confidence.squeeze(-1)

# --------------------- nuScenes_miniから歩行者軌跡抽出 ---------------------
def get_pedestrian_trajectories(nusc, max_timesteps=8):
    trajectories = []
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample_token = first_sample_token
        ped_positions = dict()
        for _ in range(max_timesteps):
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
            if sample_token == '':
                break
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if ann['category_name'] == 'human.pedestrian.adult':
                    ped_token = ann['instance_token']
                    pos = ann['translation'][:2]
                    if ped_token not in ped_positions:
                        ped_positions[ped_token] = []
                    ped_positions[ped_token].append(pos)
        for traj in ped_positions.values():
            if len(traj) == max_timesteps:
                trajectories.append(traj)
    return np.array(trajectories)

# --------------------- ADE/FDE計算 ---------------------
def calculate_ade_fde(pred, gt):
    diff = pred - gt
    dist = torch.norm(diff, dim=-1)
    ade = dist.mean(dim=-1).mean().item()
    fde = dist[:, :, -1].mean(dim=-1).item()
    return ade, fde

# --------------------- メイン ---------------------
def main():
    data_root = "./data"  # または絶対パスを指定（例: /content/data）
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)
    trajectories_np = get_pedestrian_trajectories(nusc, max_timesteps=8)
    if len(trajectories_np) == 0:
        print("歩行者軌跡が取得できませんでした")
        return
    print(f"抽出された軌跡数: {len(trajectories_np)}")
    B = 1
    N_ped, T, _ = trajectories_np.shape
    lidar_points = torch.randn(B, 1000, 3)
    trajectories = torch.tensor(trajectories_np, dtype=torch.float32).unsqueeze(0)
    static_density, nearest_dist, pedestrian_density, interaction_strength = extract_environment_features(lidar_points, trajectories)
    env_feats = torch.cat([static_density, nearest_dist, pedestrian_density, interaction_strength], dim=-1)
    model = TrajectoryPredictor(
        gat_in_dim=2, gat_out_dim=4,
        beam_in_dim=4, beam_hidden_dim=16,
        lstm_input_dim=2+4, lstm_hidden_dim=32
    )
    model.eval()
    with torch.no_grad():
        pred_traj, pred_conf = model(trajectories, env_feats)
    # ここでは単ステップ予測のため、GTの次時刻を使ってADE,FDE計算
    gt_next_pos = trajectories[:, :, -1, :]
    ade = torch.norm(pred_traj.squeeze(0) - gt_next_pos.squeeze(0), dim=-1).mean().item()
    fde = ade  # 単ステップなら同じ
    print(f"ADE (1-step): {ade:.4f}, FDE (1-step): {fde:.4f}")

if __name__ == "__main__":
    main()