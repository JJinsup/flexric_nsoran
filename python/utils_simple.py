#!/usr/bin/env python3
"""
최소화된 Localization Utils - 필수 기능만!
당신이 import하는 5개 함수/클래스만 포함
"""

import socket
import threading
import queue
import time
import os
import logging
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass

# 설정 상수
SOCKET_PATH = "/tmp/sinr_localization.sock"
BUFFER_SIZE = 4096

@dataclass
class SINRMeasurement:
    """🔥 C xApp 8개 컬럼 출력에 맞춘 데이터 클래스"""
    timestamp: int              # relative_timestamp
    ue_id: int                  # imsi  
    serving_cell_x: int        # serving_x
    serving_cell_y: int        # serving_y
    serving_cell_sinr: float   # L3 serving SINR 3gpp_ma
    neighbor1_sinr: float      # L3 neigh SINR 3gpp 1_ma
    neighbor2_sinr: float      # L3 neigh SINR 3gpp 2_ma
    neighbor3_sinr: float      # L3 neigh SINR 3gpp 3_ma

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
    """데이터 파싱 클래스"""
    
    @staticmethod
    def parse_sinr_line(line: str) -> Optional[SINRMeasurement]:
        """🔥 C xApp의 8개 컬럼 출력 파싱"""
        try:
            parts = line.strip().split(',')
            if len(parts) != 8:
                return None
                
            return SINRMeasurement(
                timestamp=int(float(parts[0])),           # relative_timestamp
                ue_id=int(parts[1]),                      # imsi
                serving_cell_x=int(parts[2]),            # serving_x
                serving_cell_y=int(parts[3]),            # serving_y
                serving_cell_sinr=float(parts[4]),       # L3 serving SINR 3gpp_ma
                neighbor1_sinr=float(parts[5]),          # L3 neigh SINR 3gpp 1_ma
                neighbor2_sinr=float(parts[6]),          # L3 neigh SINR 3gpp 2_ma
                neighbor3_sinr=float(parts[7])           # L3 neigh SINR 3gpp 3_ma
            )
        except (ValueError, IndexError) as e:
            logging.error(f"Parse error: {e} for line: {line}")
            return None

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
    # 간단 테스트
    print("🧪 Testing Minimal Utils...")
    test_line = "1640000000,1,100,200,-85.2,-92.1,-88.7,-95.3"
    measurement = DataParser.parse_sinr_line(test_line)
    print(f"📊 Parsed: UE {measurement.ue_id} at ({measurement.serving_cell_x}, {measurement.serving_cell_y})")
    print("✅ Minimal utils ready!")
