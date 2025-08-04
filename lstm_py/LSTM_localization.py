#!/usr/bin/env python3
"""
PyTorch LSTM 기반 실시간 모든 UE 위치 추정
🧠 시계열 SINR 데이터를 활용한 정확한 위치 예측

주요 특징:
- Bidirectional LSTM 모델 사용
- 시계열 패턴 학습 (LOOKBACK=10)
- 실시간 처리 및 즉시 저장
- 모든 UE 동시 위치 추정
"""

import numpy as np
import torch
import torch.nn as nn
import joblib
import queue
import signal
import sys
import time
from collections import defaultdict, deque
from sklearn.preprocessing import MinMaxScaler

# utils.py에서 필요한 클래스들 import
from utils import (
    SINRMeasurement, 
    DataParser,
    setup_logging, 
    create_socket_receiver, 
    start_receiver_thread
)

CONFIG = {
    'model_path': 'model/lstm_positioning.pth',           
    'scaler_x_path': 'model/lstm_x.pkl',      
    'scaler_y_path': 'model/lstm_y.pkl',      
    'output_file': 'lstm_trajectory.txt',
    'lookback': 10,  # 시계열 윈도우 크기
    'input_size': 12,  # 입력 피처 개수
    'hidden_size': 64,
    'output_size': 2,   # x, y 좌표
    # 🔥 추가: 물리적 제약 파라미터
    'max_speed': 5.0,      # 최대 속도 (m/s)
    'time_interval': 0.1   # 시간 간격 (s)
}

class UELocalizationLSTM(nn.Module):
    """PyTorch LSTM 위치 추정 모델 (학습 코드와 동일한 구조)"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=2):
        super(UELocalizationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # *2 for bidirectional
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시점 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # Dense layers
        out = torch.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TimeSeriesBuffer:
    """LSTM용 시계열 데이터 버퍼"""
    
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)
        
    def add_features(self, features: np.ndarray):
        """새로운 특성을 버퍼에 추가"""
        self.buffer.append(features.copy())
        
    def is_ready(self) -> bool:
        """시퀀스 예측 준비 여부"""
        return len(self.buffer) == self.max_length
    
    def get_sequence(self) -> np.ndarray:
        """시계열 시퀀스 반환 [sequence_length, features]"""
        return np.array(list(self.buffer))
    
    def get_length(self) -> int:
        """현재 버퍼 길이"""
        return len(self.buffer)

class LSTMLocalizer:
    """LSTM 기반 실시간 위치 추정기"""
    
    def __init__(self, model_path: str, scaler_x_path: str, scaler_y_path: str, output_file: str = "lstm_trajectory.txt"):
        print(f"🧠 Initializing LSTM Localizer for ALL UEs...")
        
        # Device 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # LSTM 모델 로드
        self.model = self._load_lstm_model(model_path)
        
        # Scaler 로드
        self.scaler_x = self._load_scaler(scaler_x_path, "Input")
        self.scaler_y = self._load_scaler(scaler_y_path, "Output")
        
        # 출력 파일 설정
        self.output_file = output_file
        self.file_handle = self._init_output_file()
        
        # UE별 시계열 버퍼 (LOOKBACK 개의 과거 데이터 저장)
        self.ue_buffers = defaultdict(lambda: TimeSeriesBuffer(CONFIG['lookback']))
        
        # UE별 첫 timestamp 추적 (상대 시간 계산용)
        self.ue_first_timestamps = {}
        # 🔥 추가: CSV 배치 코드와 동일한 timestamp 처리를 위한 변수들
        self.burst_group_mapping = {}      # burst_group -> sequence_timestamp 매핑
        self.next_sequence_timestamp = 0.0 # 다음 할당할 sequence timestamp
        self.sequence_interval = 100.0     # 100ms 간격
        # 🔥 추가: UE별 이전 위치 저장 (물리적 제약용)
        self.ue_previous_positions = {}

        # 통계 및 상태
        self.stats = {'processed': 0, 'estimated': 0, 'start_time': None}
        self.running = False

        # 🔥 추가: 카테고리 매핑 로드
        try:
            self.category_mappings = joblib.load("category_mappings.pkl")
            print(f"✅ Category mappings loaded:")
            for cat, mapping in self.category_mappings.items():
                print(f"  {cat}: {mapping}")
        except FileNotFoundError:
            print("❌ category_mappings.pkl not found! Using default mapping")
            self.category_mappings = None

        print(f"🎯 LSTM Localizer ready for ALL UEs")
        print(f"💾 Output file: {self.output_file}")
        print(f"🔢 Sequence length: {CONFIG['lookback']}")

    def _apply_category_mapping(self, value, category_name):
        """카테고리 값을 학습 시와 동일하게 매핑"""
        if self.category_mappings and category_name in self.category_mappings:
            mapping = self.category_mappings[category_name]
            return mapping.get(value, 0)  # 없으면 0으로 기본값
        return value  # 매핑 정보 없으면 원본 사용    
    
    def _init_output_file(self):
        """출력 파일 초기화"""
        file_handle = open(self.output_file, 'w', buffering=1)
        file_handle.write("timestamp,imsi,x,y\n")
        file_handle.flush()
        print(f"📝 Output file initialized: {self.output_file}")
        return file_handle
    
    def _load_lstm_model(self, model_path: str):
        """PyTorch LSTM 모델 로드"""
        try:
            # 모델 상태 딕셔너리 로드
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 모델 생성
            model = UELocalizationLSTM(
                input_size=CONFIG['input_size'],
                hidden_size=CONFIG['hidden_size'],
                output_size=CONFIG['output_size']
            ).to(self.device)
            
            # 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # 추론 모드
            
            print(f"✅ LSTM model loaded from {model_path}")
            print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            sys.exit(1)
    
    def _load_scaler(self, scaler_path: str, scaler_type: str):
        """Scaler 파일 로드"""
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ {scaler_type} scaler loaded from {scaler_path}")
            return scaler
        except FileNotFoundError:
            print(f"❌ {scaler_type} scaler file not found: {scaler_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading {scaler_type} scaler: {e}")
            sys.exit(1)
    
    def _extract_features(self, measurement: SINRMeasurement) -> np.ndarray:
        """🔥 간소화: sequence_timestamp 직접 사용"""
        
        # 1. Burst group → sequence timestamp
        timestamp_str = str(int(measurement.timestamp))
        burst_group = timestamp_str[:10]
        
        if burst_group not in self.burst_group_mapping:
            self.burst_group_mapping[burst_group] = self.next_sequence_timestamp
            print(f"🕒 New burst group '{burst_group}' → {self.next_sequence_timestamp}ms")
            self.next_sequence_timestamp += 100.0
        
        sequence_timestamp = self.burst_group_mapping[burst_group]
        
        # 🔥 relative_timestamp 계산 제거! 바로 sequence_timestamp 사용
        ue_id_mapped = self._apply_category_mapping(measurement.ue_id, "ueImsiComplete")
        serving_cell_mapped = self._apply_category_mapping(measurement.serving_cell_id, "L3 serving Id(m_cellId)")
        neigh1_mapped = self._apply_category_mapping(measurement.neighbor1_id, "L3 neigh Id 1 (cellId)")
        neigh2_mapped = self._apply_category_mapping(measurement.neighbor2_id, "L3 neigh Id 2 (cellId)")
        neigh3_mapped = self._apply_category_mapping(measurement.neighbor3_id, "L3 neigh Id 3 (cellId)")
        
        features = [
            sequence_timestamp,  # 🔥 바로 사용! relative 계산 안 함
            ue_id_mapped,
            serving_cell_mapped,
            measurement.serving_cell_sinr,
            neigh1_mapped,
            measurement.neighbor1_sinr,
            neigh2_mapped,
            measurement.neighbor2_sinr,
            neigh3_mapped,
            measurement.neighbor3_sinr,
            measurement.serving_cell_x,
            measurement.serving_cell_y
        ]
        
        return np.array(features, dtype=np.float32)
    
    def apply_movement_constraint(self, pred_pos: np.ndarray, prev_pos: np.ndarray, 
                            max_speed: float = 5.0, dt: float = 0.1) -> np.ndarray:

        max_distance = max_speed * dt  # 최대 이동거리 (예: 5m/s * 0.1s = 0.5m)
        
        # 이동 벡터 계산
        movement = pred_pos - prev_pos
        distance = np.linalg.norm(movement)
        
        # 최대 이동거리 초과 시 제한
        if distance > max_distance:
            # 방향은 유지하되 거리만 제한
            movement = movement * (max_distance / distance)
            constrained_pos = prev_pos + movement
            
            print(f"⚠️  Movement constrained: {distance:.2f}m → {max_distance:.2f}m")
            return constrained_pos
        
        return pred_pos

    def process_measurement(self, measurement: SINRMeasurement):
        """🧠 LSTM으로 위치 추정 (물리적 제약 적용)"""
        try:
            # 1. 특성 추출
            features = self._extract_features(measurement)
            
            # 2. UE별 시계열 버퍼에 추가
            buffer = self.ue_buffers[measurement.ue_id]
            buffer.add_features(features)
            
            # 3. 충분한 시계열 데이터가 있을 때만 예측
            if buffer.is_ready():
                # 시계열 배열 생성 [LOOKBACK, Features]
                sequence = buffer.get_sequence()
                
                # 입력 정규화
                sequence_scaled = self.scaler_x.transform(sequence.reshape(-1, CONFIG['input_size'])).reshape(sequence.shape)
                
                # PyTorch 텐서로 변환 [1, LOOKBACK, Features]
                sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
                
                # LSTM 예측
                with torch.no_grad():
                    prediction_scaled = self.model(sequence_tensor)
                
                # CPU로 이동 및 numpy 변환
                prediction_scaled_np = prediction_scaled.cpu().numpy()
                
                # 출력 역정규화
                raw_prediction = self.scaler_y.inverse_transform(prediction_scaled_np)[0]
                
                # 🔥 추가: 물리적 제약 조건 적용
                if measurement.ue_id in self.ue_previous_positions:
                    prev_pos = self.ue_previous_positions[measurement.ue_id]
                    constrained_prediction = self.apply_movement_constraint(raw_prediction, prev_pos)
                else:
                    # 첫 번째 예측이면 제약 없이 사용
                    constrained_prediction = raw_prediction
                
                # 🔥 추가: 이전 위치 업데이트
                self.ue_previous_positions[measurement.ue_id] = constrained_prediction.copy()
                
                # x, y 좌표 추출
                x, y = constrained_prediction

                # 파일에 즉시 저장
                self.file_handle.write(f"{measurement.timestamp},{measurement.ue_id},{x:.6f},{y:.6f}\n")
                
                self.stats['estimated'] += 1
                
                # 주기적 flush 및 디버깅
                if self.stats['estimated'] % 50 == 0:
                    self.file_handle.flush()
                    print(f"💾 {self.stats['estimated']} positions saved to {self.output_file}")
                
                if self.stats['estimated'] % 10 == 0:
                    print(f"🧠 UE_{measurement.ue_id}: ({x:.2f},{y:.2f}) | "
                        f"SeqTime:{self.current_sequence_timestamp}ms | "  # 🔥 저장된 값 사용
                        f"SINR: [{measurement.serving_cell_sinr:.1f}, {measurement.neighbor1_sinr:.1f}] | "
                        f"Buffer: {buffer.get_length()}/{CONFIG['lookback']}")
                
            else:
                # 버퍼 채우는 중
                if buffer.get_length() % 5 == 0:
                    print(f"📊 UE_{measurement.ue_id}: Buffering {buffer.get_length()}/{CONFIG['lookback']}...")
                    
        except Exception as e:
            print(f"❌ LSTM prediction error for UE {measurement.ue_id}: {e}")
    
    def start_data_collection(self, data_queue: queue.Queue):
        """🔥 실시간 데이터 수집 및 모든 UE 처리"""
        self.running = True
        self.stats['start_time'] = time.time()
        print(f"🚀 Starting real-time LSTM localization for ALL UEs")
        print(f"⏱️  Sequence buffer size: {CONFIG['lookback']}")
        
        while self.running:
            try:
                line = data_queue.get(timeout=0.1)
                measurement = DataParser.parse_sinr_line(line)
                
                if measurement:  # 모든 UE 처리
                    self.stats['processed'] += 1
                    
                    # 모든 UE에 대해 LSTM 위치 추정
                    self.process_measurement(measurement)
                    
                    # 주기적 상태 출력
                    if self.stats['processed'] % 100 == 0:
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
        unique_ues = len(self.ue_first_timestamps)
        active_buffers = sum(1 for buffer in self.ue_buffers.values() if buffer.is_ready())
        # 🔥 추가: timestamp 매핑 상태
        burst_groups = len(self.burst_group_mapping)

        print(f"📊 LSTM ALL UEs | "
              f"Rate:{processing_rate:.1f}/s | "
              f"Success:{success_rate:.1f}% | "
              f"Processed:{self.stats['processed']} | "
              f"Estimated:{self.stats['estimated']} | "
              f"UEs:{unique_ues} | "
              f"Ready:{active_buffers}")
    
    def stop(self):
        """정지 및 파일 안전 종료"""
        print(f"\n🛑 Stopping LSTM localization...")
        self.running = False
        
        # 최종 통계
        self._print_status()
        print(f"\n📈 Final UE Buffer Status:")
        for ue_id, buffer in self.ue_buffers.items():
            status = buffer.get_length()
            ready = "✅ Ready" if buffer.is_ready() else "⏳ Buffering"
            print(f"   UE_{ue_id}: {status}/{CONFIG['lookback']} measurements - {ready}")
        
        # 파일 안전하게 종료
        if self.file_handle:
            self.file_handle.flush()
            self.file_handle.close()
            print(f"✅ Output file closed: {self.output_file} ({self.stats['estimated']} total positions)")
        
        print(f"✅ LSTM localization completed")

def main():
    """메인 실행 함수"""
    # 로깅 설정
    setup_logging()
    
    print("=" * 60)
    print("🧠 PyTorch LSTM-Based Real-time ALL UEs Localization")
    print("🔄 Time-series SINR pattern learning")
    print("=" * 60)
    print(f"🎯 Target: ALL UEs (no filtering)")
    print(f"🧠 Model: {CONFIG['model_path']}")
    print(f"📊 Input scaler: {CONFIG['scaler_x_path']}")
    print(f"📊 Output scaler: {CONFIG['scaler_y_path']}")
    print(f"💾 Output file: {CONFIG['output_file']}")
    print(f"🔢 Sequence length: {CONFIG['lookback']} timesteps")
    print("=" * 60)
    
    # LSTM 위치 추정기 초기화
    try:
        localizer = LSTMLocalizer(
            model_path=CONFIG['model_path'],
            scaler_x_path=CONFIG['scaler_x_path'],
            scaler_y_path=CONFIG['scaler_y_path'],
            output_file=CONFIG['output_file']
        )
    except Exception as e:
        print(f"❌ LSTM 초기화 실패: {e}")
        return
    
    # 실시간 데이터 수신 설정 (utils.py 활용)
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
    
    print("🚀 LSTM localization started for ALL UEs!")
    print("💡 C xApp을 실행해서 SINR 데이터를 전송하세요")
    print("🔄 시계열 패턴 학습으로 정확한 위치 예측!")
    print(f"⏱️  {CONFIG['lookback']}개 측정값이 모이면 예측 시작")
    print("🛑 종료하려면 Ctrl+C를 누르세요")
    print("-" * 60)
    
    try:
        # 🧠 실시간 LSTM 위치 추정 시작
        localizer.start_data_collection(data_queue)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 정리 중...")
        localizer.stop()
        receiver.stop()
        print("✅ LSTM localization 완료!")

if __name__ == "__main__":
    print("🔍 Checking required files...")
    
    required_files = [
        CONFIG['model_path'],
        CONFIG['scaler_x_path'], 
        CONFIG['scaler_y_path']
    ]
    
    missing_files = []
    for file_path in required_files:
        try:
            with open(file_path, 'rb'):
                pass
            print(f"  ✅ {file_path}")
        except FileNotFoundError:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}")
    
    if missing_files:
        print(f"\n❌ 필수 파일이 없습니다:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\n💡 먼저 LSTM 모델을 학습시켜주세요!")
        sys.exit(1)
    
    print("✅ All files found! Starting LSTM localization...\n")
    main()
