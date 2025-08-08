#!/usr/bin/env python3
"""
Localization Utils - 진짜 공통 함수들만!
DNN, RNN, LSTM, 사변측량 모든 방법이 공통으로 사용하는 핵심 기능들

핵심 기능:
1. 실시간 데이터 수신 (Unix Domain Socket)
2. 데이터 파싱 및 구조화
3. 고성능 궤적 저장 (txt)
4. 기본 실시간 처리 프레임워크
5. 시각화 및 성능 평가
"""

import socket
import threading
import queue
import time
import signal
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# 설정 상수
SOCKET_PATH = "/tmp/sinr_localization.sock"
BUFFER_SIZE = 4096
MAX_QUEUE_SIZE = 1000

@dataclass
class SINRMeasurement:
    """🔥 Trilateration용 확장된 SINR 측정 데이터"""
    # 기본 정보
    timestamp: int              # relative_timestamp
    ue_id: int                  # imsi
    
    # Serving cell 정보
    serving_cell_id: int        # L3 serving Id(m_cellId)
    serving_cell_x: int         # serving_x
    serving_cell_y: int         # serving_y
    serving_cell_sinr: float    # L3 serving SINR 3gpp_ma
    
    # Neighbor 1 정보
    neighbor1_id: int           # L3 neigh Id 1 (cellId)
    neighbor1_x: int            # neighbor1_x
    neighbor1_y: int            # neighbor1_y
    neighbor1_sinr: float       # L3 neigh SINR 3gpp 1 (convertedSinr)_ma
    
    # Neighbor 2 정보
    neighbor2_id: int           # L3 neigh Id 2 (cellId)
    neighbor2_x: int            # neighbor2_x
    neighbor2_y: int            # neighbor2_y
    neighbor2_sinr: float       # L3 neigh SINR 3gpp 2 (convertedSinr)_ma
    
    # Neighbor 3 정보
    neighbor3_id: int           # L3 neigh Id 3 (cellId)
    neighbor3_x: int            # neighbor3_x
    neighbor3_y: int            # neighbor3_y
    neighbor3_sinr: float       # L3 neigh SINR 3gpp 3 (convertedSinr)_ma

class SocketReceiver:
    """Unix Domain Socket 실시간 데이터 수신기"""
    
    def __init__(self, socket_path: str, data_queue: queue.Queue):
        self.socket_path = socket_path
        self.data_queue = data_queue
        self.running = False
        self.sock = None
        self.stats = defaultdict(int)
        
    def start(self):
        """소켓 서버 시작"""
        self.running = True
        
        # 기존 소켓 파일 삭제
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass
            
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(1)
        
        print(f"🔌 Socket ready: {self.socket_path}")
        
        while self.running:
            try:
                conn, addr = self.sock.accept()
                print("✅ xApp connected")
                self._handle_connection(conn)
            except OSError:
                if self.running:
                    logging.error("Socket accept error")
                break
                
    def _handle_connection(self, conn):
        """클라이언트 연결 처리"""
        buffer = ""
        
        try:
            while self.running:
                data = conn.recv(BUFFER_SIZE).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # 마지막 불완전한 라인 보관
                
                for line in lines[:-1]:
                    if line.strip():
                        try:
                            self.data_queue.put_nowait(line.strip())
                            self.stats['received'] += 1
                        except queue.Full:
                            # Queue 오버플로우 시 오래된 데이터 제거
                            try:
                                self.data_queue.get_nowait()
                                self.data_queue.put_nowait(line.strip())
                                self.stats['dropped'] += 1
                            except queue.Empty:
                                pass
                                
        except Exception as e:
            logging.error(f"Connection error: {e}")
        finally:
            conn.close()
            print("🔌 xApp disconnected")
            
    def stop(self):
        """소켓 서버 중지"""
        self.running = False
        if self.sock:
            self.sock.close()
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass

class DataParser:
    """데이터 파싱 클래스 - 학습 코드 버전"""
    
    @staticmethod
    def parse_sinr_line(line: str) -> Optional[SINRMeasurement]:
        """🔥 C xApp의 18개 컬럼 출력 파싱 (Trilateration용)"""
        try:
            parts = line.strip().split(',')
            if len(parts) != 18:  # 18개 컬럼으로 수정
                return None
                
            return SINRMeasurement(
                # 기본 정보
                timestamp=int(float(parts[0])),           # relative_timestamp
                ue_id=int(parts[1]),                 # imsi
                
                # Serving cell 정보  
                serving_cell_id=int(parts[2]),            # L3 serving Id(m_cellId)
                serving_cell_x=int(parts[3]),             # serving_x
                serving_cell_y=int(parts[4]),             # serving_y
                serving_cell_sinr=float(parts[5]),        # L3 serving SINR 3gpp_ma
                
                # Neighbor 1 정보
                neighbor1_id=int(parts[6]),               # L3 neigh Id 1 (cellId)
                neighbor1_x=int(parts[7]),                # neighbor1_x
                neighbor1_y=int(parts[8]),                # neighbor1_y
                neighbor1_sinr=float(parts[9]),           # L3 neigh SINR 3gpp 1
                
                # Neighbor 2 정보
                neighbor2_id=int(parts[10]),              # L3 neigh Id 2 (cellId)
                neighbor2_x=int(parts[11]),               # neighbor2_x
                neighbor2_y=int(parts[12]),               # neighbor2_y
                neighbor2_sinr=float(parts[13]),          # L3 neigh SINR 3gpp 2
                
                # Neighbor 3 정보
                neighbor3_id=int(parts[14]),              # L3 neigh Id 3 (cellId)
                neighbor3_x=int(parts[15]),               # neighbor3_x
                neighbor3_y=int(parts[16]),               # neighbor3_y
                neighbor3_sinr=float(parts[17]),          # L3 neigh SINR 3gpp 3
                
            )
        except (ValueError, IndexError) as e:
            logging.error(f"Parse error: {e} for line: {line}")
            return None
    
    @staticmethod
    def measurements_to_dataframe(measurements: List[SINRMeasurement]) -> pd.DataFrame:
        """🔥 수정된 DataFrame 변환"""
        data = []
        for m in measurements:
            data.append({
                'timestamp': m.timestamp,
                'ue_id': m.ue_id,
                'serving_cell_x': m.serving_cell_x,
                'serving_cell_y': m.serving_cell_y,
                'serving_cell_sinr': m.serving_cell_sinr,
                'neighbor1_sinr': m.neighbor1_sinr,
                'neighbor2_sinr': m.neighbor2_sinr,
                'neighbor3_sinr': m.neighbor3_sinr
            })
        return pd.DataFrame(data)

class TrajectoryTracker:
    """고성능 UE 궤적 저장 (실시간 txt 저장)"""
    
    def __init__(self, ue_id: int, auto_save: bool = True, buffer_size: int = 50):
        self.ue_id = ue_id  # IMSI 역할
        self.trajectory = []  # 메모리 버퍼 (최근 위치 추적용)
        self.auto_save = auto_save
        self.buffer_size = buffer_size
        self.position_count = 0
        self.last_position = None
        
        # 고성능 파일 처리 (append 모드)
        self.filename = f"ue_{self.ue_id}_trajectory.txt"
        self.file_handle = None
        
        if auto_save:
            self._init_file()
    
    def _init_file(self):
        """파일 초기화 및 헤더 작성"""
        self.file_handle = open(self.filename, 'w', buffering=1)  # 라인 버퍼링
        # 헤더 작성: timestamp,imsi,x,y
        self.file_handle.write("timestamp,imsi,x,y\n")
        self.file_handle.flush()
        print(f"📝 Trajectory file initialized: {self.filename}")
    
    def add_position(self, timestamp: int, x: float, y: float):
        """새로운 위치를 즉시 파일에 저장 (최고 성능)"""
        # 메모리 버퍼에도 저장 (최근 위치 추적용)
        position = (timestamp, self.ue_id, x, y)
        self.trajectory.append(position)
        self.last_position = (x, y)
        self.position_count += 1
        
        # 즉시 파일에 기록 (append 모드로 최고 성능)
        if self.file_handle:
            self.file_handle.write(f"{timestamp},{self.ue_id},{x:.6f},{y:.6f}\n")
            
            # 주기적으로 flush (안전성과 성능 균형)
            if self.position_count % self.buffer_size == 0:
                self.file_handle.flush()
                print(f"💾 {self.position_count} positions saved to {self.filename}")
        
        # 메모리 절약: 최근 1000개만 유지
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-500:]
    
    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """현재 위치 반환"""
        return self.last_position
    
    def get_trajectory_length(self) -> int:
        """저장된 궤적 길이 반환"""
        return self.position_count
    
    def get_recent_positions(self, count: int = 10) -> List[Tuple[float, float]]:
        """최근 N개 위치 반환 (메모리에서)"""
        recent = self.trajectory[-count:] if len(self.trajectory) >= count else self.trajectory
        return [(pos[2], pos[3]) for pos in recent]  # x, y만 추출
    
    def close(self):
        """파일 안전하게 종료"""
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            print(f"✅ Trajectory file closed: {self.filename} ({self.position_count} total positions)")

class RealtimeLocalizer:
    """실시간 위치 추정 기본 클래스 (서브클래스에서 상속)"""
    
    def __init__(self, ue_id: int):
        self.ue_id = ue_id
        self.trajectory_tracker = TrajectoryTracker(ue_id, auto_save=True)
        self.running = False
        self.stats = {'processed': 0, 'estimated': 0, 'start_time': None}
        
    def start_data_collection(self, data_queue: queue.Queue):
        """실시간 데이터 수집 및 즉시 처리"""
        self.running = True
        self.stats['start_time'] = time.time()
        print(f"🚀 Starting real-time localization for UE {self.ue_id}")
        print(f"📝 Saving to: ue_{self.ue_id}_trajectory.txt")
        
        while self.running:
            try:
                line = data_queue.get(timeout=0.1)
                measurement = DataParser.parse_sinr_line(line)
                
                if measurement and measurement.ue_id == self.ue_id:
                    self.stats['processed'] += 1
                    
                    # 서브클래스에서 구현할 위치 추정 로직
                    self.process_measurement(measurement)
                    
                    # 주기적 상태 출력
                    if self.stats['processed'] % 50 == 0:
                        self._print_status()
                        
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Real-time processing error: {e}")
    
    def process_measurement(self, measurement: SINRMeasurement):
        """측정값 처리 (서브클래스에서 오버라이드)"""
        pass
    
    def save_position(self, timestamp: int, x: float, y: float):
        """추정된 위치를 저장"""
        self.trajectory_tracker.add_position(timestamp, x, y)
        self.stats['estimated'] += 1
        
        # 실시간 출력 (10개마다)
        if self.stats['estimated'] % 10 == 0:
            print(f"🎯 UE_{self.ue_id}: ({x:.2f}, {y:.2f}) | Total: {self.stats['estimated']}")
    
    def _print_status(self):
        """성능 통계 출력"""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        processing_rate = self.stats['processed'] / elapsed if elapsed > 0 else 0
        success_rate = (self.stats['estimated'] / self.stats['processed']) * 100 if self.stats['processed'] > 0 else 0
        trajectory_length = self.trajectory_tracker.get_trajectory_length()
        current_pos = self.trajectory_tracker.get_current_position()
        
        print(f"📊 UE_{self.ue_id} | "
              f"Rate:{processing_rate:.1f}/s | "
              f"Success:{success_rate:.1f}% | "
              f"Saved:{trajectory_length} | "
              f"Current:{current_pos}")
    
    def stop(self):
        """정지 및 파일 안전 종료"""
        print(f"\n🛑 Stopping UE_{self.ue_id} localization...")
        self.running = False
        
        # 최종 통계
        self._print_status()
        
        # 파일 안전하게 종료
        self.trajectory_tracker.close()
        
        print(f"✅ UE_{self.ue_id} localization completed")

class Visualizer:
    """데이터 시각화 클래스"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def plot_trajectory_from_file(self, filename: str, figsize: Tuple[int, int] = (12, 8)):
        """저장된 궤적 파일에서 시각화"""
        try:
            # txt 파일 읽기
            df = pd.read_csv(filename)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 궤적 그리기
            ax.plot(df['x'], df['y'], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            ax.scatter(df['x'], df['y'], c='blue', s=10, alpha=0.6)
            
            # 시작점과 끝점 표시
            if len(df) > 0:
                ax.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, marker='o', label='Start')
                ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=100, marker='s', label='End')
            
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_title(f'UE Trajectory ({len(df)} points)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig, ax
            
        except Exception as e:
            print(f"Error plotting trajectory: {e}")
            return None, None
    
    def plot_sinr_timeline(self, measurements: List[SINRMeasurement], 
                          figsize: Tuple[int, int] = (14, 6)):
        """SINR 시계열 시각화"""
        fig, ax = plt.subplots(figsize=figsize)
        
        timestamps = [m.timestamp for m in measurements]
        serving_sinr = [m.serving_cell_sinr for m in measurements]
        neighbor1_sinr = [m.neighbor1_sinr if m.neighbor1_sinr > 0 else None for m in measurements]
        neighbor2_sinr = [m.neighbor2_sinr if m.neighbor2_sinr > 0 else None for m in measurements]
        neighbor3_sinr = [m.neighbor3_sinr if m.neighbor3_sinr > 0 else None for m in measurements]
        
        ax.plot(timestamps, serving_sinr, label='Serving Cell', linewidth=2)
        ax.plot(timestamps, neighbor1_sinr, label='Neighbor 1', alpha=0.7)
        ax.plot(timestamps, neighbor2_sinr, label='Neighbor 2', alpha=0.7)
        ax.plot(timestamps, neighbor3_sinr, label='Neighbor 3', alpha=0.7)
        
        ax.set_xlabel('Timestamp (ms)')
        ax.set_ylabel('SINR (dB)')
        ax.set_title('SINR Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax

class PerformanceEvaluator:
    """성능 평가 메트릭"""
    
    @staticmethod
    def calculate_positioning_error(true_positions: List[Tuple[float, float]], 
                                  estimated_positions: List[Tuple[float, float]]) -> Dict[str, float]:
        """위치 추정 오차 계산"""
        if len(true_positions) != len(estimated_positions):
            raise ValueError("Position lists must have same length")
        
        errors = []
        for (true_x, true_y), (est_x, est_y) in zip(true_positions, estimated_positions):
            error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
            errors.append(error)
        
        return {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'rmse': np.sqrt(np.mean(np.array(errors)**2))
        }
    
    @staticmethod
    def load_trajectory_from_file(filename: str) -> List[Tuple[float, float]]:
        """저장된 궤적 파일에서 위치 리스트 로드"""
        df = pd.read_csv(filename)
        return list(zip(df['x'], df['y']))

# 편의 함수들
def setup_logging(level=logging.INFO):
    """로깅 설정"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_socket_receiver(data_queue: queue.Queue) -> SocketReceiver:
    """소켓 수신기 생성"""
    return SocketReceiver(SOCKET_PATH, data_queue)

def start_receiver_thread(receiver: SocketReceiver) -> threading.Thread:
    """수신기 스레드 시작"""
    thread = threading.Thread(target=receiver.start, name="SocketReceiver")
    thread.daemon = True
    thread.start()
    return thread

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Testing Localization Utils...")
    
    # 데이터 파서 테스트
    test_line = "1640000000,1,2,25.5,3,22.1,4,20.8,5,18.7,800.0,800.0"
    measurement = DataParser.parse_sinr_line(test_line)
    print(f"📊 Parsed measurement: UE {measurement.ue_id} at ({measurement.serving_cell_x}, {measurement.serving_cell_y})")
    
    print("✅ Clean utils module ready for all localization methods!")
