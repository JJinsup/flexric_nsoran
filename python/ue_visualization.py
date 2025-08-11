#!/usr/bin/env python3
"""
UE Trajectory Comparison Visualization: Actual vs Predicted Trajectories (Simple Version)
- ue_position.txt: Actual positions 
- lstm_trajectory.txt: Predicted positions
- 색상 구분이 잘 되도록 개선
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")



def fast_trajectory_similarity(actual_traj, predicted_traj, num_points=50):
    """
    빠르고 정확한 궤적 유사도 평가
    - 시간 무시, 순서만 고려
    - 정규화된 길이로 리샘플링 후 비교
    """
    if len(actual_traj) < 2 or len(predicted_traj) < 2:
        return None
    
    try:
        # 1. 좌표만 추출 (시간 무시)
        actual_points = actual_traj[['x', 'y']].values
        pred_points = predicted_traj[['x', 'y']].values
        
        # 2. 궤적 길이 정규화 (같은 개수 점으로 리샘플링)
        def resample_trajectory(points, n_samples):
            """궤적을 n_samples 개수로 균등하게 리샘플링"""
            if len(points) <= n_samples:
                return points
            
            # 0부터 len-1까지를 n_samples 개로 균등 분할
            indices = np.linspace(0, len(points) - 1, n_samples)
            resampled = []
            
            for idx in indices:
                if idx == int(idx):  # 정수 인덱스
                    resampled.append(points[int(idx)])
                else:  # 보간 필요
                    lower_idx = int(np.floor(idx))
                    upper_idx = int(np.ceil(idx))
                    weight = idx - lower_idx
                    
                    interpolated = (1 - weight) * points[lower_idx] + weight * points[upper_idx]
                    resampled.append(interpolated)
            
            return np.array(resampled)
        
        # 3. 두 궤적을 같은 길이로 리샘플링
        actual_resampled = resample_trajectory(actual_points, num_points)
        pred_resampled = resample_trajectory(pred_points, num_points)
        
        # 4. 벡터화된 거리 계산 (순서대로 1:1 매칭)
        distances = np.sqrt(np.sum((actual_resampled - pred_resampled)**2, axis=1))
        
        # 5. 통계 계산
        return {
            'avg_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances),
            'median_distance': np.median(distances),
            'trajectory_points': num_points
        }
        
    except Exception as e:
        return {'error': str(e), 'avg_distance': float('inf')}

def analyze_trajectory_similarity_optimized(actual_df, predicted_df):
    """최적화된 궤적 유사도 분석"""
    print("\n" + "="*80)
    print("🚀 FAST TRAJECTORY SIMILARITY ANALYSIS")
    print("="*80)

    # 공통 UE 찾기
    common_ues = sorted(set(actual_df['id'].unique()) & set(predicted_df['id'].unique()))
    print(f"📊 Found {len(common_ues)} common UEs")
    
    # 결과 저장
    similarity_results = {}
    
    # 각 UE 처리
    for i, ue_id in enumerate(common_ues):
        print(f"Processing UE {ue_id:2d} ({i+1:2d}/{len(common_ues)})...", end=" ")
        
        # 궤적 데이터 추출
        actual_traj = actual_df[actual_df['id'] == ue_id].copy()
        predicted_traj = predicted_df[predicted_df['id'] == ue_id].copy()
        
        if len(actual_traj) < 2 or len(predicted_traj) < 2:
            print("⚠️  Insufficient points")
            continue
        
        # 🔥 빠른 유사도 계산
        similarity = fast_trajectory_similarity(actual_traj, predicted_traj)
        
        if similarity and 'error' not in similarity:
            similarity_results[ue_id] = similarity
            print(f"✅ Avg: {similarity['avg_distance']:.1f}m, Max: {similarity['max_distance']:.1f}m")
        else:
            print(f"❌ Error: {similarity.get('error', 'Unknown')}")
    
    # 전체 통계
    if similarity_results:
        print(f"\n" + "="*80)
        print("📊 OVERALL STATISTICS")
        print("="*80)
        
        all_avg_dist = [r['avg_distance'] for r in similarity_results.values()]
        all_max_dist = [r['max_distance'] for r in similarity_results.values()]
        all_std_dist = [r['std_distance'] for r in similarity_results.values()]
        
        print(f"📊 Average Distance Error:")
        print(f"   Mean: {np.mean(all_avg_dist):.2f}m ± {np.std(all_avg_dist):.2f}m")
        print(f"   Range: {np.min(all_avg_dist):.2f}m - {np.max(all_avg_dist):.2f}m")
        
        print(f"📊 Maximum Distance Error:")
        print(f"   Mean: {np.mean(all_max_dist):.2f}m ± {np.std(all_max_dist):.2f}m")
        print(f"   Range: {np.min(all_max_dist):.2f}m - {np.max(all_max_dist):.2f}m")
        
        # 성능 순위
        best_ue = min(similarity_results.keys(), key=lambda x: similarity_results[x]['avg_distance'])
        worst_ue = max(similarity_results.keys(), key=lambda x: similarity_results[x]['avg_distance'])
        
        print(f"\n🏆 Best Performance:  UE {best_ue}")
        print(f"   Avg: {similarity_results[best_ue]['avg_distance']:.2f}m")
        print(f"   Max: {similarity_results[best_ue]['max_distance']:.2f}m")
        
        print(f"💥 Worst Performance: UE {worst_ue}")
        print(f"   Avg: {similarity_results[worst_ue]['avg_distance']:.2f}m") 
        print(f"   Max: {similarity_results[worst_ue]['max_distance']:.2f}m")
        
        print("="*80)
        
        return similarity_results
    else:
        print("❌ No valid similarity results")
        return {}

def calculate_path_length(trajectory):
    """궤적의 총 길이 계산"""
    if len(trajectory) < 2:
        return 0
    
    total_length = 0
    for i in range(1, len(trajectory)):
        dx = trajectory['x'].iloc[i] - trajectory['x'].iloc[i-1]
        dy = trajectory['y'].iloc[i] - trajectory['y'].iloc[i-1]
        total_length += np.sqrt(dx**2 + dy**2)
    
    return total_length

def load_and_process_data():
    """Load data and preprocess for trajectory generation"""
    print("📊 Loading data...")
    
    # Actual positions (ue_position.txt)
    actual_df = pd.read_csv('ue_position_3gpp1.txt')
    print(f"✅ Actual positions: {actual_df.shape}")
    
    # Predicted positions (lstm_trajectory.txt)
    predicted_df = pd.read_csv('lstm_trajectory_3gpp1.txt')
    print(f"✅ Predicted positions: {predicted_df.shape}")
    
    # Unify column names (imsi -> id)
    if 'imsi' in predicted_df.columns:
        predicted_df = predicted_df.rename(columns={'imsi': 'id'})
    
    # Sort by time
    actual_df = actual_df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    predicted_df = predicted_df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    print(f"📊 Actual UE IDs: {sorted(actual_df['id'].unique())}")
    print(f"📊 Predicted UE IDs: {sorted(predicted_df['id'].unique())}")
    
    return actual_df, predicted_df

def get_distinct_colors(n):
    """구분이 잘 되는 색상 생성"""
    if n <= 7:
        # 기본 7가지 뚜렷한 색상
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', '#00FFFF']
        return colors[:n]
    elif n <= 12:
        # 12가지 색상 조합
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', 
                 '#FFFF00', '#FF8080', '#8080FF', '#80FF80', '#FF8040', '#4080FF']
        return colors[:n]
    else:
        # 많은 수를 위한 HSV 색상환
        hues = np.linspace(0, 360, n, endpoint=False)
        colors = []
        for i, hue in enumerate(hues):
            # 채도와 명도를 조절해서 구분이 잘 되도록
            saturation = 0.8 if i % 2 == 0 else 1.0
            value = 0.9 if i % 3 == 0 else 0.7
            
            # HSV to RGB 변환
            h = hue / 60.0
            c = value * saturation
            x = c * (1 - abs((h % 2) - 1))
            m = value - c
            
            if 0 <= h < 1:
                r, g, b = c, x, 0
            elif 1 <= h < 2:
                r, g, b = x, c, 0
            elif 2 <= h < 3:
                r, g, b = 0, c, x
            elif 3 <= h < 4:
                r, g, b = 0, x, c
            elif 4 <= h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append(f'#{int((r+m)*255):02x}{int((g+m)*255):02x}{int((b+m)*255):02x}')
        
        return colors

def plot_trajectories(actual_df, predicted_df):
    """Plot actual vs predicted trajectory comparison with better colors"""
    
    actual_df, predicted_df = load_and_process_data()
    
    # Select only common UE IDs
    common_ues = sorted(set(actual_df['id'].unique()) & set(predicted_df['id'].unique()))
    total_ues = min(len(common_ues), 28)  # Max 28 UEs
    common_ues = common_ues[:total_ues]
    
    print(f"🎯 Visualizing {total_ues} UE trajectory comparisons...")
    
    # 🔥 개선된 색상 설정
    colors = get_distinct_colors(7)  # 7개씩 그룹이므로 7가지 색상
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle('UE Trajectory Comparison: Actual vs Predicted', 
                 fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    # Divide into 4 groups of 7 each
    ues_per_plot = 7
    
    for plot_idx in range(4):
        ax = axes[plot_idx]
        
        # Select UEs for current plot
        start_idx = plot_idx * ues_per_plot
        end_idx = min(start_idx + ues_per_plot, len(common_ues))
        current_ues = common_ues[start_idx:end_idx]
        
        print(f"  🛣️ Group {plot_idx+1}: UE {current_ues}")
        
        # Plot actual vs predicted trajectories for each UE
        for i, ue_id in enumerate(current_ues):
            color = colors[i % len(colors)]
            
            # Actual trajectory data
            actual_data = actual_df[actual_df['id'] == ue_id].copy()
            if len(actual_data) > 1:
                ax.scatter(actual_data['x'], actual_data['y'], 
                            color='white', s=80, alpha=0.9, 
                            label=f'UE {ue_id} Actual',
                            marker='o', edgecolors=color, linewidths=3)
                
                # Start and end points
                ax.scatter(actual_data['x'].iloc[0], actual_data['y'].iloc[0], 
                          color=color, s=150, marker='s', edgecolors='black', 
                          linewidth=2, alpha=1.0, zorder=10)
                ax.scatter(actual_data['x'].iloc[-1], actual_data['y'].iloc[-1], 
                          color=color, s=200, marker='*', edgecolors='black', 
                          linewidth=2, alpha=1.0, zorder=10)
            
            # Predicted trajectory data
            predicted_data = predicted_df[predicted_df['id'] == ue_id].copy()
            if len(predicted_data) > 1:
                ax.scatter(predicted_data['x'], predicted_data['y'], 
                        color=color, s=80, alpha=0.8, 
                        label=f'UE {ue_id} Predicted',
                        marker='^', edgecolors='black', linewidths=0.5)
                ax.scatter(predicted_data['x'].iloc[0], predicted_data['y'].iloc[0], 
                        color=color, s=150, marker='s', edgecolors='white', 
                        linewidth=2, alpha=1.0, zorder=12)  # 시작점
                ax.scatter(predicted_data['x'].iloc[-1], predicted_data['y'].iloc[-1], 
                        color=color, s=200, marker='*', edgecolors='white', 
                        linewidth=2, alpha=1.0, zorder=12)  # 끝점
        
        # Axis settings
        ax.set_xlabel('X Coordinate (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate (m)', fontsize=14, fontweight='bold')
        ax.set_title(f'Group {plot_idx+1}: UE {start_idx+1}-{end_idx} Trajectories', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.4, linewidth=1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        
        # Add enhanced symbol legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='black', label='Start Point', 
                      markersize=10, linestyle='None', markerfacecolor='lightgray', 
                      markeredgewidth=2),
            plt.Line2D([0], [0], marker='*', color='black', label='End Point', 
                      markersize=12, linestyle='None', markerfacecolor='lightgray',
                      markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='gray', label='Actual Points', 
                       markersize=8, linestyle='None', markerfacecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], marker='^', color='gray', label='Predicted Points', 
                      markersize=8, linestyle='None', markerfacecolor='gray', markeredgewidth=2)
        ]
        
        # Combine legends
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles + legend_elements
        all_labels = labels + [elem.get_label() for elem in legend_elements]
        ax.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ue_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Saved: ue_trajectory_comparison.png")

def main():
    """Main execution"""
    print("🚀 Starting UE trajectory visualization!")
    
    try:
        # 데이터 로딩
        actual_df, predicted_df = load_and_process_data()

        # 1. 궤적 유사도 분석 (터미널 출력)
        analyze_trajectory_similarity_optimized(actual_df, predicted_df)
        
        # 2. 궤적 비교 시각화
        plot_trajectories(actual_df, predicted_df)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
