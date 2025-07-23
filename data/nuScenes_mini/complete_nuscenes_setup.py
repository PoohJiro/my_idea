# complete_nuscenes_setup.py
# nuScenes_miniを使用するための完全なセットアップスクリプト

import os
import json
import numpy as np
import pickle
import shutil
from typing import Dict, List, Tuple
import argparse

def setup_directories():
    """必要なディレクトリを作成"""
    directories = [
        './datasets/',
        './datasets/nuscenes_mini/',
        './datasets/nuscenes_mini/train/',
        './datasets/nuscenes_mini/val/',
        './datasets/nuscenes_mini/test/',
        './checkpoint/',
        './results/'
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def convert_nuscenes_raw_to_eth(nuscenes_path: str, output_path: str):
    """
    nuScenes rawデータ（JSON形式）をETH形式に変換
    """
    try:
        from nuscenes.nuscenes import NuScenes
        print("Using nuScenes official API for conversion...")
        
        # nuScenesデータを読み込み
        nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path, verbose=True)
        
        # 全シーンを取得
        all_scenes = nusc.scene
        n_scenes = len(all_scenes)
        
        # データ分割（70% train, 20% val, 10% test）
        train_end = int(n_scenes * 0.7)
        val_end = int(n_scenes * 0.9)
        
        train_scenes = all_scenes[:train_end]
        val_scenes = all_scenes[train_end:val_end]
        test_scenes = all_scenes[val_end:]
        
        def process_scenes(scenes: List, split_name: str):
            """シーンを処理してETH形式に変換"""
            total_pedestrians = 0
            total_frames = 0
            
            for scene_idx, scene in enumerate(scenes):
                scene_token = scene['token']
                scene_name = scene['name']
                
                # シーンの最初のサンプルを取得
                sample_token = scene['first_sample_token']
                
                trajectories = {}  # instance_token -> [(frame, x, y), ...]
                frame_id = 0
                
                while sample_token != '':
                    sample = nusc.get('sample', sample_token)
                    
                    # アノテーションを取得
                    for ann_token in sample['anns']:
                        ann = nusc.get('sample_annotation', ann_token)
                        
                        # 歩行者のみを対象
                        if ann['category_name'].startswith('human.pedestrian'):
                            instance_token = ann['instance_token']
                            
                            # 位置情報を取得（グローバル座標）
                            translation = ann['translation']
                            x, y = translation[0], translation[1]  # x, y座標（メートル）
                            
                            # 軌跡データに追加
                            if instance_token not in trajectories:
                                trajectories[instance_token] = []
                            trajectories[instance_token].append((frame_id, x, y))
                    
                    # 次のサンプルへ
                    sample_token = sample['next']
                    frame_id += 1
                
                # ETH形式でファイルに保存
                if trajectories:  # 歩行者がいる場合のみ
                    output_file = f"{output_path}/{split_name}/{scene_name}.txt"
                    with open(output_file, 'w') as f:
                        # person_id を連番に変換
                        person_mapping = {token: idx for idx, token in enumerate(trajectories.keys())}
                        
                        # フレーム順にソートして出力
                        all_data = []
                        for person_token, traj in trajectories.items():
                            person_id = person_mapping[person_token]
                            for frame, x, y in traj:
                                all_data.append((frame, person_id, x, y))
                        
                        # フレーム、人物ID順でソート
                        all_data.sort(key=lambda x: (x[0], x[1]))
                        
                        for frame, person_id, x, y in all_data:
                            f.write(f"{frame} {person_id} {x:.6f} {y:.6f}\n")
                    
                    total_pedestrians += len(trajectories)
                    total_frames += frame_id
                    print(f"Processed {split_name}/{scene_name}: {len(trajectories)} pedestrians, {frame_id} frames")
            
            print(f"{split_name} split: {len(scenes)} scenes, {total_pedestrians} total pedestrians, {total_frames} total frames")
        
        # 各分割を処理
        process_scenes(train_scenes, 'train')
        process_scenes(val_scenes, 'val')
        process_scenes(test_scenes, 'test')
        
        print(f"nuScenes raw data conversion completed! Data saved to {output_path}")
        return True
        
    except ImportError:
        print("nuScenes API not available. Please install: pip install nuscenes-devkit")
        return False
    except Exception as e:
        print(f"Error in raw data conversion: {e}")
        return False

def convert_processed_data_to_eth(input_path: str, output_path: str):
    """
    前処理済みのnuScenesデータをETH形式に変換
    """
    print("Converting processed nuScenes data to ETH format...")
    
    # 入力ファイルを探す
    input_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv') or file.endswith('.json'):
                input_files.append(os.path.join(root, file))
    
    if not input_files:
        print(f"No data files found in {input_path}")
        return False
    
    print(f"Found {len(input_files)} data files")
    
    # データ分割
    n_files = len(input_files)
    train_end = int(n_files * 0.7)
    val_end = int(n_files * 0.9)
    
    train_files = input_files[:train_end]
    val_files = input_files[train_end:val_end]
    test_files = input_files[val_end:]
    
    def copy_and_convert_files(files: List[str], split_name: str):
        """ファイルをコピーして必要に応じて形式変換"""
        for i, file_path in enumerate(files):
            filename = os.path.basename(file_path)
            output_file = f"{output_path}/{split_name}/scene_{i:03d}.txt"
            
            try:
                if file_path.endswith('.txt'):
                    # すでにテキスト形式の場合はコピー
                    shutil.copy(file_path, output_file)
                elif file_path.endswith('.csv'):
                    # CSV形式の場合は変換
                    convert_csv_to_eth(file_path, output_file)
                elif file_path.endswith('.json'):
                    # JSON形式の場合は変換
                    convert_json_to_eth(file_path, output_file)
                
                print(f"Converted {filename} -> {split_name}/scene_{i:03d}.txt")
            except Exception as e:
                print(f"Error converting {filename}: {e}")
    
    copy_and_convert_files(train_files, 'train')
    copy_and_convert_files(val_files, 'val')
    copy_and_convert_files(test_files, 'test')
    
    print(f"Processed data conversion completed!")
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")
    return True

def convert_csv_to_eth(csv_path: str, output_path: str):
    """CSV形式をETH形式に変換"""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
        
        # 列名を推測して変換
        if 'frame' in df.columns and 'person_id' in df.columns:
            # 既にETH形式に近い場合
            with open(output_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{int(row['frame'])} {int(row['person_id'])} {row['x']:.6f} {row['y']:.6f}\n")
        else:
            # 一般的な形式を仮定
            frame_col = df.columns[0]  # 最初の列をフレームとして仮定
            person_col = df.columns[1]  # 2番目の列を人物IDとして仮定
            x_col = df.columns[2]  # 3番目の列をxとして仮定
            y_col = df.columns[3]  # 4番目の列をyとして仮定
            
            with open(output_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{int(row[frame_col])} {int(row[person_col])} {row[x_col]:.6f} {row[y_col]:.6f}\n")
    
    except Exception as e:
        print(f"Error converting CSV {csv_path}: {e}")

def convert_json_to_eth(json_path: str, output_path: str):
    """JSON形式をETH形式に変換"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # JSON構造を解析して変換（構造に応じて調整が必要）
        with open(output_path, 'w') as f:
            if isinstance(data, list):
                for item in data:
                    if all(key in item for key in ['frame', 'person_id', 'x', 'y']):
                        f.write(f"{item['frame']} {item['person_id']} {item['x']:.6f} {item['y']:.6f}\n")
            elif isinstance(data, dict):
                # 辞書形式の場合の処理（具体的な構造に応じて調整）
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if all(k in item for k in ['frame', 'person_id', 'x', 'y']):
                                f.write(f"{item['frame']} {item['person_id']} {item['x']:.6f} {item['y']:.6f}\n")
    
    except Exception as e:
        print(f"Error converting JSON {json_path}: {e}")

def create_dummy_data():
    """
    テスト用のダミーデータを作成
    """
    print("Creating dummy nuScenes_mini data for testing...")
    
    output_path = './datasets/nuscenes_mini/'
    
    # 各分割用のダミーデータを作成
    for split in ['train', 'val', 'test']:
        n_files = {'train': 5, 'val': 2, 'test': 1}[split]
        
        for i in range(n_files):
            filename = f"{output_path}/{split}/scene_{i:03d}.txt"
            
            with open(filename, 'w') as f:
                # ランダムな軌道データを生成
                np.random.seed(i + ord(split[0]))  # 再現可能な乱数
                
                n_pedestrians = np.random.randint(2, 6)  # 2-5人の歩行者
                n_frames = np.random.randint(50, 100)    # 50-100フレーム
                
                for person_id in range(n_pedestrians):
                    # 初期位置
                    start_x = np.random.uniform(-10, 10)
                    start_y = np.random.uniform(-10, 10)
                    
                    # 速度（ランダムウォーク）
                    vx = np.random.uniform(-0.5, 0.5)
                    vy = np.random.uniform(-0.5, 0.5)
                    
                    x, y = start_x, start_y
                    
                    for frame in range(n_frames):
                        # 位置を更新（ノイズ付きランダムウォーク）
                        x += vx + np.random.normal(0, 0.1)
                        y += vy + np.random.normal(0, 0.1)
                        
                        # 速度をわずかに変更
                        vx += np.random.normal(0, 0.05)
                        vy += np.random.normal(0, 0.05)
                        
                        # 速度制限
                        vx = np.clip(vx, -1.0, 1.0)
                        vy = np.clip(vy, -1.0, 1.0)
                        
                        f.write(f"{frame} {person_id} {x:.6f} {y:.6f}\n")
            
            print(f"Created dummy data: {filename}")
    
    print("Dummy data creation completed!")

def verify_data_format(data_path: str):
    """
    変換されたデータの形式を確認
    """
    print(f"Verifying data format in {data_path}...")
    
    for split in ['train', 'val', 'test']:
        split_path = f"{data_path}/{split}/"
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            continue
        
        files = [f for f in os.listdir(split_path) if f.endswith('.txt')]
        print(f"{split}: {len(files)} files")
        
        if files:
            # 最初のファイルをサンプルとして確認
            sample_file = f"{split_path}/{files[0]}"
            try:
                with open(sample_file, 'r') as f:
                    lines = f.readlines()
                
                print(f"  Sample file: {files[0]}")
                print(f"  Lines: {len(lines)}")
                
                if lines:
                    # 最初の数行を表示
                    print("  Sample data:")
                    for i, line in enumerate(lines[:3]):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            print(f"    Frame: {parts[0]}, Person: {parts[1]}, X: {parts[2]}, Y: {parts[3]}")
                        else:
                            print(f"    Invalid format: {line.strip()}")
                        
                        if i >= 2:  # 最初の3行だけ表示
                            break
            
            except Exception as e:
                print(f"  Error reading {sample_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Setup nuScenes_mini for Social-STGCNN')
    parser.add_argument('--mode', choices=['raw', 'processed', 'dummy'], default='dummy',
                       help='Conversion mode: raw (nuScenes raw data), processed (pre-processed data), dummy (create test data)')
    parser.add_argument('--input_path', type=str, default='./nuscenes_mini/',
                       help='Input path for nuScenes data')
    parser.add_argument('--output_path', type=str, default='./datasets/nuscenes_mini/',
                       help='Output path for converted data')
    
    args = parser.parse_args()
    
    print("="*50)
    print("nuScenes_mini Setup for Social-STGCNN")
    print("="*50)
    
    # ディレクトリ作成
    setup_directories()
    
    # データ変換
    success = False
    
    if args.mode == 'raw':
        print(f"Converting nuScenes raw data from {args.input_path}...")
        success = convert_nuscenes_raw_to_eth(args.input_path, args.output_path)
    
    elif args.mode == 'processed':
        print(f"Converting processed data from {args.input_path}...")
        success = convert_processed_data_to_eth(args.input_path, args.output_path)
    
    elif args.mode == 'dummy':
        print("Creating dummy data for testing...")
        create_dummy_data()
        success = True
    
    if success:
        # データ形式確認
        verify_data_format(args.output_path)
        
        print("\n" + "="*50)
        print("Setup completed successfully!")
        print("You can now run training with:")
        print("python train.py --dataset nuscenes_mini")
        print("="*50)
    else:
        print("\nSetup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()