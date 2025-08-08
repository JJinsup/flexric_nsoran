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
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F  # ← 이게 없음!

# utils.py에서 필요한 클래스들 import
from utils_Standard import (
    SINRMeasurement, 
    DataParser,
    setup_logging, 
    create_socket_receiver, 
    start_receiver_thread
)

CONFIG = {
    'model_path': 'model/lstm_3gpp.pth',           
    'scaler_x_path': 'model/lstm_3gpp_x.pkl',      # 🔥 StandardScaler
    'scaler_y_path': 'model/lstm_3gpp_y.pkl',      # 🔥 StandardScaler
    'output_file': 'lstm_trajectory_3gpp.txt',
    'lookback': 10,
    'input_size': 7,      
    'hidden_size': 128,
    'num_layers': 3,      
    'output_size': 2,
    'max_speed': 5.0,
    'time_interval': 0.1
}

class UELocalizationLSTM(nn.Module):
    """학습 코드와 정확히 동일한 LSTM 구조"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(UELocalizationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0  
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = F.relu(self.fc1(last_output))  
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
                output_size=CONFIG['output_size'],
                num_layers=CONFIG['num_layers']
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
        """학습 데이터와 동일한 6개 features 추출 (UE ID는 feature에서 제외)"""
                
        # 2. 학습과 완전히 동일한 7개 features만 추출
        features = [
            measurement.timestamp,                  # relative_timestamp
            measurement.serving_cell_x,          # serving_x
            measurement.serving_cell_y,          # serving_y  
            measurement.serving_cell_sinr,       # L3 serving SINR 3gpp_ma
            measurement.neighbor1_sinr,          # L3 neigh SINR 3gpp 1 (convertedSinr)_ma
            measurement.neighbor2_sinr,          # L3 neigh SINR 3gpp 2 (convertedSinr)_ma
            measurement.neighbor3_sinr           # L3 neigh SINR 3gpp 3 (convertedSinr)_ma
        ]
        
        # 🔍 디버깅용 (처음 몇 번만 출력)
        if measurement.ue_id not in getattr(self, '_debug_printed', set()):
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = set()
            
            print(f"🧠 UE_{measurement.ue_id} features: "
                f"ts={measurement.timestamp:.1f}, "
                f"pos=({measurement.serving_cell_x},{measurement.serving_cell_y}), "
                f"sinr=[{measurement.serving_cell_sinr:.1f}, "
                f"{measurement.neighbor1_sinr:.1f}, "
                f"{measurement.neighbor2_sinr:.1f}, "
                f"{measurement.neighbor3_sinr:.1f}]")
            
            self._debug_printed.add(measurement.ue_id)
        
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
            
            print(f"⚠️  UE_{ue_id} Movement constrained: {distance:.2f}m → {max_distance:.2f}m")
            return constrained_pos
        
        return pred_pos

    def process_measurement(self, measurement: SINRMeasurement):
        """🧠 LSTM으로 위치 추정 (UE ID는 버퍼 키로만 사용)"""
        try:
            # 1. 특성 추출 (6개 features, UE ID 제외)
            features = self._extract_features(measurement)
            
            # 2. UE ID를 키로 사용해서 UE별 시계열 버퍼에 추가
            buffer = self.ue_buffers[measurement.ue_id]  # ← UE ID는 여기서만 사용
            buffer.add_features(features)
            
            # 3. 충분한 시계열 데이터가 있을 때만 예측
            if buffer.is_ready():
                # 시계열 배열 생성 [LOOKBACK, 6 Features]
                sequence = buffer.get_sequence()
                
                # 입력 정규화 (6개 features)
                sequence_scaled = self.scaler_x.transform(
                    sequence.reshape(-1, 7)  # ✅ 7개로 수정
                ).reshape(sequence.shape)
                
                # PyTorch 텐서로 변환 [1, LOOKBACK, 6]
                sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
                
                # LSTM 예측
                with torch.no_grad():
                    prediction_scaled = self.model(sequence_tensor)
                
                # CPU로 이동 및 numpy 변환
                prediction_scaled_np = prediction_scaled.cpu().numpy()
                
                # 출력 역정규화
                raw_prediction = self.scaler_y.inverse_transform(prediction_scaled_np)[0]
                
                # 🔥 물리적 제약 조건 적용
                if measurement.ue_id in self.ue_previous_positions:
                    prev_pos = self.ue_previous_positions[measurement.ue_id]
                    constrained_prediction = self.apply_movement_constraint(raw_prediction, prev_pos)
                else:
                    constrained_prediction = raw_prediction
                
                # 이전 위치 업데이트
                self.ue_previous_positions[measurement.ue_id] = constrained_prediction.copy()
                
                # x, y 좌표 추출
                x, y = constrained_prediction

                # 파일에 즉시 저장 (UE ID 포함)
                self.file_handle.write(f"{measurement.timestamp},{measurement.ue_id},{x:.6f},{y:.6f}\n")
                
                self.stats['estimated'] += 1
                
                # 주기적 출력에서도 UE ID 사용
                if self.stats['estimated'] % 10 == 0:
                    print(f"🧠 UE_{measurement.ue_id}: ({x:.2f},{y:.2f}) | "
                        f"RelTime:{measurement.timestamp:.1f}ms | "
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
        
        # ✅ 존재하는 속성으로 수정
        unique_ues = len(self.ue_buffers)  # ue_buffers는 존재함
        active_buffers = sum(1 for buffer in self.ue_buffers.values() if buffer.is_ready())

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
