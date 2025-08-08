#!/usr/bin/env python3
"""
🧠 시계열 SINR 데이터를 활용한 정확한 위치 예측

주요 특징:
- Trilateration 기반 3D 위치 추정
- 4개 기지국의 SINR 데이터 활용
- 3GPP Path Loss 모델 적용
- 모든 UE 동시 위치 추정
"""

import numpy as np
import pandas as pd
import signal
import sys
import time
from scipy.optimize import least_squares
import math
from collections import defaultdict, deque

# utils에서 실시간 처리에 필요한 클래스들 import
from utils_tri import (
    SINRMeasurement, 
    DataParser,
    setup_logging, 
    create_socket_receiver, 
    start_receiver_thread
)
import queue

CONFIG = {
    'output_file': 'trilateration_trajectory.txt',
}

class TriLocalizer:
    
    def __init__(self, output_file: str = "trilateration_trajectory.txt"):

        # 출력 파일 설정
        self.output_file = output_file
        self.file_handle = self._init_output_file()
        self.stats = {'processed': 0, 'estimated': 0, 'start_time': None}  
        # 🔥 추가: 물리 파라미터들
        self.tx_power_dbm = 30.0      # ns3에서 확인한 값
        self.frequency_hz = 3.5e9     # 3.5 GHz
        self.noise_power_dbm = -96.0
        self.h_bs = 3.0               # 기지국 높이 3m
        self.h_ut = 0.0               # UE 높이 0m
        self.h_e = 1.0                # ns-3에서 사용되는 E 높이 (1m)

    def _binary_search_los_pl2(self, target_path_loss: float) -> float:
        """PL2 공식의 이진 탐색 역산"""
        low, high = 50.0, 2000.0  # PL2는 breakpoint 이후부터
        
        for _ in range(20):  # 충분히 정확
            mid = (low + high) / 2
            calc_pl, calc_type = self.calculate_3gpp_path_loss(mid)
            
            if abs(calc_pl - target_path_loss) < 0.1:
                return mid
            elif calc_pl < target_path_loss:
                low = mid
            else:
                high = mid
        
        return mid

    def convert_3gpp_sinr_to_db(self, sinr_3gpp: float) -> float:
        """ns-3와 정확히 동일한 3GPP SINR 변환"""
        return (sinr_3gpp * 63.0 / 127.0) - 23.0

    def calculate_3gpp_path_loss(self, distance_3d: float) -> tuple:
        """LOS와 NLOS Path Loss와 어느 것이 사용됐는지 반환"""
        frequency_ghz = self.frequency_hz / 1e9
        distance_2d = math.sqrt(max(0, distance_3d**2 - (self.h_bs - self.h_ut)**2))
        
        # Breakpoint 계산
        distanceBp = 4 * (self.h_bs - self.h_e) * (self.h_ut - self.h_e) * self.frequency_hz / 3e8
        
        # LOS Path Loss
        if distance_2d <= distanceBp:
            pl_los = 32.4 + 21.0 * math.log10(distance_3d) + 20.0 * math.log10(frequency_ghz)
        else:
            pl_los = (32.4 + 40.0 * math.log10(distance_3d) + 20.0 * math.log10(frequency_ghz) 
                    - 9.5 * math.log10(distanceBp**2 + (self.h_bs - self.h_ut)**2))
        
        # NLOS Path Loss
        pl_nlos = (22.4 + 35.3 * math.log10(distance_3d) + 21.3 * math.log10(frequency_ghz) 
                - 0.3 * (self.h_ut - 1.5))
        
        # 더 큰 값과 타입 반환
        if pl_los >= pl_nlos:
            return pl_los, 0  # 0 = LOS
        else:
            return pl_nlos, 1  # 1 = NLOS

    def inverse_3gpp_los_path_loss(self, path_loss_db: float, loss_type: int) -> float:
        """Path Loss 타입에 따라 역산"""
        frequency_ghz = self.frequency_hz / 1e9
        
        if loss_type == 0:  # LOS
            # 먼저 PL1 시도
            log_term = (path_loss_db - 32.4 - 20.0 * math.log10(frequency_ghz)) / 21.0
            distance_3d = 10 ** log_term
            distance_2d = math.sqrt(max(0, distance_3d**2 - (self.h_bs - self.h_ut)**2))
            
            distanceBp = 4 * (self.h_bs - self.h_e) * (self.h_ut - self.h_e) * self.frequency_hz / 3e8
            if distance_2d <= distanceBp:
                return np.clip(distance_3d, 1.0, 5000.0)
            else:
                # PL2는 복잡하니 이진탐색
                return self._binary_search_los_pl2(path_loss_db)
        
        else:  # NLOS
            log_term = (path_loss_db - 22.4 - 21.3 * math.log10(frequency_ghz) + 0.3 * (self.h_ut - 1.5)) / 35.3
            distance_3d = 10 ** log_term
            return np.clip(distance_3d, 1.0, 5000.0)

    def sinr_to_distance(self, sinr_3gpp: float) -> float:
        """3D 거리 직접 반환 (논문 방식)"""
        actual_sinr_db = self.convert_3gpp_sinr_to_db(sinr_3gpp)
        rx_power_dbm = actual_sinr_db + self.noise_power_dbm
        path_loss_db = self.tx_power_dbm - rx_power_dbm
        
        # 3D 거리 직접 계산 (2D 변환 없이)
        for loss_type in [0, 1]:
            try:
                distance_3d = self.inverse_3gpp_los_path_loss(path_loss_db, loss_type)
                calc_pl, calc_type = self.calculate_3gpp_path_loss(distance_3d)
                if abs(calc_pl - path_loss_db) < 1.0 and calc_type == loss_type:
                    return distance_3d  # 3D 거리 그대로 반환
            except:
                continue
        
        return 100.0
    
    def trilateration_4points(self, base_stations: list, distances: list) -> tuple:
        """논문의 3D trilateration 방법"""
        
        # 3D 좌표 설정 (기지국 높이 포함)
        bs_3d = [(bs[0], bs[1], self.h_bs) for bs in base_stations]
        
        def equations(pos):
            x, y, z = pos
            errors = []
            for i in range(4):  # 4개 기지국
                bx, by, bz = bs_3d[i]
                # 논문 식 (2): 3D 유클리드 거리
                calculated_dist = math.sqrt((x - bx)**2 + (y - by)**2 + (z - bz)**2)
                errors.append(calculated_dist - distances[i])
            return errors
        
        # 초기값: (중심점, UE 높이)
        center_x = sum(bs[0] for bs in base_stations) / 4
        center_y = sum(bs[1] for bs in base_stations) / 4
        initial_guess = (center_x, center_y, self.h_ut)  # z좌표 포함
        
        # 논문의 비선형 연립방정식 해결
        result = least_squares(equations, initial_guess, method='lm')
        
        if result.success:
            return result.x[0], result.x[1]  # x, y만 반환 (z는 고정)
        else:
            return center_x, center_y

    def process_measurement(self, measurement: SINRMeasurement) -> bool:
        # 1. 기지국 좌표와 SINR (이미 간섭 반영됨)
        base_stations = [
            (measurement.serving_cell_x, measurement.serving_cell_y),
            (measurement.neighbor1_x, measurement.neighbor1_y),
            (measurement.neighbor2_x, measurement.neighbor2_y),
            (measurement.neighbor3_x, measurement.neighbor3_y)
        ]
        sinr_values = [
            measurement.serving_cell_sinr,
            measurement.neighbor1_sinr,
            measurement.neighbor2_sinr,
            measurement.neighbor3_sinr
        ]
        try:
            # 2. 각 SINR → 거리 직접 변환
            distances = [self.sinr_to_distance(sinr) for sinr in sinr_values]
            
            # 3. Trilateration
            estimated_x, estimated_y = self.trilateration_4points(base_stations, distances)
            
            # 🔥 추가: 예측값 저장
            self.file_handle.write(f"{measurement.timestamp},{measurement.ue_id},"
                                f"{estimated_x:.2f},{estimated_y:.2f}\n") 
            return True
        except Exception as e:
            return False
    
    def _init_output_file(self):
        """출력 파일 초기화"""
        file_handle = open(self.output_file, 'w', buffering=1)
        file_handle.write("timestamp,imsi,x,y\n")
        file_handle.flush()
        print(f"📝 Output file initialized: {self.output_file}")
        return file_handle
    
    def start_data_collection(self, data_queue: queue.Queue):
        """🔥 실시간 데이터 수집 및 처리"""
        self.running = True
        self.stats['start_time'] = time.time()
        print(f"🚀 Starting real-time Trilateration for ALL UEs")
        
        while self.running:
            try:
                line = data_queue.get(timeout=0.1)
                print(f"📥 Received: {line[:100]}...")  # 디버깅 추가
                measurement = DataParser.parse_sinr_line(line)
                
                if measurement:  # 모든 UE 처리
                    print(f"✅ Parsed UE_{measurement.ue_id}")  # 추가!
                    self.stats['processed'] += 1
                    
                    # Trilateration 위치 추정
                    success = self.process_measurement(measurement)
                    if success:
                        self.stats['estimated'] += 1
                        print(f"🎯 UE_{measurement.ue_id} position estimated")  # 추가!

                    # 주기적 상태 출력
                    if self.stats['processed'] % 50 == 0:
                        self._print_status()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Real-time processing error: {e}")

    def _print_status(self):
        """성능 통계 출력"""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        processing_rate = self.stats['processed'] / elapsed if elapsed > 0 else 0
        success_rate = (self.stats['estimated'] / self.stats['processed']) * 100 if self.stats['processed'] > 0 else 0
        
        print(f"📊 Trilateration | "
            f"Rate:{processing_rate:.1f}/s | "
            f"Success:{success_rate:.1f}% | "
            f"Processed:{self.stats['processed']} | "
            f"Estimated:{self.stats['estimated']}")

    def stop(self):
        """정지 및 파일 안전 종료"""
        print(f"\n🛑 Stopping Trilateration...")
        self.running = False
        self.close()
    
    def close(self):
        """파일 안전하게 종료"""
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            print(f"✅ Output file closed: {self.output_file} ({self.stats['estimated']} total positions)")

def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    
    print("=" * 60)
    print("📐 Real-time Trilateration-Based UE Localization")
    print("🔄 4-Point SINR to Position Estimation")
    print("=" * 60)
    
    # Trilateration 위치 추정기 초기화
    try:
        localizer = TriLocalizer(output_file=CONFIG['output_file'])
    except Exception as e:
        print(f"❌ Trilateration 초기화 실패: {e}")
        return
    
    # 실시간 데이터 수신 설정
    data_queue = queue.Queue()
    receiver = create_socket_receiver(data_queue)
    receiver_thread = start_receiver_thread(receiver)
    
    # 종료 신호 처리
    def signal_handler(sig, frame):
        print(f"\n🛑 종료 신호 수신 ({sig})")
        localizer.stop()
        receiver.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Trilateration started for ALL UEs!")
    print("💡 C xApp을 실행해서 SINR 데이터를 전송하세요")
    print("🛑 종료하려면 Ctrl+C를 누르세요")
    print("-" * 60)
    
    try:
        # 📐 실시간 Trilateration 시작
        localizer.start_data_collection(data_queue)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 정리 중...")
        localizer.stop()
        receiver.stop()
        print("✅ Trilateration 완료!")

if __name__ == "__main__":
    
    main()
