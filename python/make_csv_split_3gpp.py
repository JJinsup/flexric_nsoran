from pathlib import Path
import pandas as pd
import numpy as np
import re

# ——— 설정 ———
root = Path("/home/delivery/flexric_oran/dev/data")
scenario_dirs = sorted(root.glob("data_ten*"))
out_dir = root / "output"
out_dir.mkdir(exist_ok=True)

# 🔥 시나리오별 저장을 위한 디렉토리 생성
scenario_out_dir = out_dir / "scenarios"
scenario_out_dir.mkdir(exist_ok=True)

cell_ids = [2,3,4,5,6,7,8]

# 결과를 누적할 리스트
all_results = []

for base_dir in scenario_dirs:
    print(f"\n=== Processing {base_dir.name} ===")

    # 1) cu-cp 로그 7개 합치기 & cu-cp의 t0 계산
    dfs = []
    t0_cucp = None
    for cid in cell_ids:
        fp = base_dir / f"cu-cp-cell-{cid}.txt"
        tmp = pd.read_csv(fp)
        dfs.append(tmp)
        mt = tmp["timestamp"].min()
        t0_cucp = mt if t0_cucp is None else min(t0_cucp, mt)
    df = pd.concat(dfs, ignore_index=True)

    # 2) cu-cp의 상대 timestamp 만들기
    df.sort_values("timestamp", inplace=True)
    df["relative_timestamp"] = (df["timestamp"].astype(int) - t0_cucp)

    # 2.5) 7개 이웃셀 중 유효한 상위 3개 선택하여 재배치
    print(f"Total rows before neighbor selection: {len(df)}")
    
    # 새로운 neighbor 컬럼들 초기화
    for i in range(1, 4):
        df[f"selected_neigh_id_{i}"] = np.nan
        df[f"selected_neigh_sinr_{i}"] = np.nan
    
    # 각 행에 대해 유효한 neighbor 3개 선택
    valid_row_indices = []
    
    for idx in df.index:
        row = df.loc[idx]
        valid_neighbors = []
        
        # neighbor 1~7 중 유효한 것들 찾기
        for i in range(1, 8):
            cell_id_col = f"L3 neigh Id {i} (cellId)"
            sinr_col = f"L3 neigh SINR 3gpp {i} (convertedSinr)"
            
            if (cell_id_col in df.columns and sinr_col in df.columns and
                pd.notna(row[cell_id_col]) and pd.notna(row[sinr_col]) and
                row[cell_id_col] > 0 ):  # 양수 cell ID만 허용
                
                valid_neighbors.append({
                    'cell_id': row[cell_id_col],
                    'sinr': row[sinr_col]
                })
        
        # 유효한 neighbor가 3개 이상인 경우만 사용
        if len(valid_neighbors) >= 3:
            # 상위 3개 선택 (이미 SINR 순으로 정렬되어 있음)
            top_3 = valid_neighbors[:3]
            
            # 선택된 정보 저장
            for j, neighbor in enumerate(top_3, 1):
                df.loc[idx, f"selected_neigh_id_{j}"] = neighbor['cell_id']
                df.loc[idx, f"selected_neigh_sinr_{j}"] = neighbor['sinr']
            
            valid_row_indices.append(idx)
    
    print(f"Rows with at least 3 valid neighbors: {len(valid_row_indices)}")
    
    # 유효한 행들만 선택
    df = df.loc[valid_row_indices].copy()
    
    # 기존 neighbor 컬럼을 선택된 것들로 교체
    for i in range(1, 4):
        df[f"L3 neigh Id {i} (cellId)"] = df[f"selected_neigh_id_{i}"]
        df[f"L3 neigh SINR 3gpp {i} (convertedSinr)"] = df[f"selected_neigh_sinr_{i}"]
    
    # 임시 컬럼 제거
    df = df.drop(columns=[f"selected_neigh_id_{i}" for i in range(1, 4)] + 
                         [f"selected_neigh_sinr_{i}" for i in range(1, 4)])
    
    print(f"Final rows after neighbor selection: {len(df)}")

    # 3) ue_position.txt 처리
    trace = pd.read_csv(base_dir / "ue_position.txt")
    trace = trace[["timestamp", "id", "x", "y"]]
    trace.rename(columns={"id": "UE (imsi)"}, inplace=True)
    
    # ue_position의 t0 계산 및 relative_timestamp 생성
    t0_uepos = trace["timestamp"].min()
    trace["relative_timestamp"] = trace["timestamp"].astype(int) - t0_uepos
    trace = trace[["relative_timestamp", "UE (imsi)", "x", "y"]]

    # 4) 병합
    df = df.merge(
        trace,
        on=["relative_timestamp","UE (imsi)"],
        how="left"
    )

    # 5) 이동평균 윈도우 크기 계산 (5초)
    dt = df["relative_timestamp"].diff().loc[lambda x: x>0].min()
    window_size = max(1, int(5000 / dt))
    print(f" → window_size = {window_size}")

    # 6) SINR 컬럼 이동평균 적용 (serving + 이웃 1~3만)
    sinr_cols = ["L3 serving SINR 3gpp"] + [
        f"L3 neigh SINR 3gpp {i} (convertedSinr)" for i in range(1,4)
    ]

    print(f"Rows before serving cell filter: {len(df)}")

    # serving cell SINR이 유효하지 않은 행 제거
    df = df[df["L3 serving SINR 3gpp"].notna()]

    print(f"Rows after serving cell filter: {len(df)}")

    # 이동평균 적용
    for col in sinr_cols:
        df[f"{col}_ma"] = (
            df
            .groupby("UE (imsi)")[col]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )

    # 7) gNB 좌표 하드코딩 매핑
    coords = {
        2: (800, 800),   # 기지국 2번의 x, y 좌표
        3: (1300, 800),   # 기지국 3번의 x, y 좌표  
        4: (1050, 1233),   # 기지국 4번의 x, y 좌표
        5: (550, 1233),   # 기지국 5번의 x, y 좌표
        6: (300, 800),  # 기지국 6번의 x, y 좌표
        7: (550, 366), # 기지국 7번의 x, y 좌표
        8: (1050, 366)  # 기지국 8번의 x, y 좌표
    }
    # 8) serving 및 neighbor 좌표 매핑
    # Serving cell 좌표
    df[["serving_x","serving_y"]] = (
        df["L3 serving Id(m_cellId)"]
        .map(lambda i: coords.get(i, (np.nan, np.nan)))
        .apply(pd.Series)
    )
    df["serving_x"] = df["serving_x"].apply(lambda x: int(x) if pd.notna(x) else x)
    df["serving_y"] = df["serving_y"].apply(lambda x: int(x) if pd.notna(x) else x)

    # Neighbor cells 좌표
    for i in range(1, 4):  # neighbor 1, 2, 3
        col_id = f"L3 neigh Id {i} (cellId)"
        if col_id in df.columns:
            df[[f"neighbor{i}_x", f"neighbor{i}_y"]] = (
                df[col_id]
                .map(lambda cell_id: coords.get(cell_id, (np.nan, np.nan)) if pd.notna(cell_id) else (np.nan, np.nan))
                .apply(pd.Series)
            )
            df[f"neighbor{i}_x"] = df[f"neighbor{i}_x"].apply(lambda x: int(x) if pd.notna(x) else x)
            df[f"neighbor{i}_y"] = df[f"neighbor{i}_y"].apply(lambda x: int(x) if pd.notna(x) else x)

    # 9) UE 좌표 컬럼 이름 변경
    df.rename(columns={"x":"UE_x", "y":"UE_y"}, inplace=True)

    # 10) 최종 컬럼 선택
    final_cols = [
        "relative_timestamp",
        "UE (imsi)",
        "serving_x", "serving_y",
        "L3 serving SINR 3gpp_ma",
        "L3 neigh SINR 3gpp 1 (convertedSinr)_ma",
        "L3 neigh SINR 3gpp 2 (convertedSinr)_ma",
        "L3 neigh SINR 3gpp 3 (convertedSinr)_ma",
        "UE_x", "UE_y"
    ]
    available = [c for c in final_cols if c in df.columns]
    final = df[available].copy()

    # 11) 컬럼명 정리
    final.rename(columns={
        "UE (imsi)": "imsi"
    }, inplace=True)

    # 12) 결측치 처리
    final.dropna(subset=["UE_x","UE_y"], inplace=True)
    final.fillna(0, inplace=True)
    
    # SINR 컬럼들 소수점 1자리로 반올림
    sinr_ma_cols = [f"{col}_ma" for col in sinr_cols]
    for col in sinr_ma_cols:
        if col in final.columns:
            final[col] = final[col].round(1)

    # UE 좌표 소수점 1자리로 반올림
    final["UE_x"] = final["UE_x"].round(1)
    final["UE_y"] = final["UE_y"].round(1)
    final["relative_timestamp"] = final["relative_timestamp"] // 100

    # 🔥 시나리오별 개별 저장
    scenario_name = base_dir.name
    scenario_csv_path = scenario_out_dir / f"{scenario_name}.csv"
    final.to_csv(scenario_csv_path, index=False)
    print(f"✅ Saved individual scenario: {scenario_csv_path}")
    print(f"   Shape: {final.shape}")
    
    # 리스트에 추가 (전체 결합용)
    all_results.append(final)

# ——— 모든 시나리오 합치기 & 저장 ———
combined = pd.concat(all_results, ignore_index=True)
out_path = out_dir / "training_ten.csv"
combined.to_csv(out_path, index=False)
print(f"\n✅ All scenarios merged and saved to {out_path}")
print("Combined shape:", combined.shape)

# 🔥 시나리오별 파일 목록 출력
print(f"\n📁 Individual scenario files saved in: {scenario_out_dir}")
for scenario_file in sorted(scenario_out_dir.glob("*.csv")):
    df_info = pd.read_csv(scenario_file)
    print(f"   - {scenario_file.name}: {df_info.shape[0]} rows")
