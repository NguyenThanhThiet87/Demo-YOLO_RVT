from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
import time
import sys
import cv2
import numpy as np
from collections import deque
import torch
from Model_YOLO_RVT.PredictModel import predict_license_plate, load_model_for_prediction
from decord import VideoReader, cpu

BEST_CONFIDENCE_THRESHOLD = 0.8
MAX_FRAME_HISTORY = 20
FRAME_SIZE = (480, 480)
class Result:
    def __init__(self, pixmap, license_plate, confidence, timestamp):
        self.pixmap = pixmap
        self.license_plate = license_plate
        self.confidence = confidence
        self.timestamp = timestamp
    
class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('untitled.ui', self)

    def setFPS(self, fps):
        self.lblFPS.setText(f"FPS: {fps:.0f}")
    def setTimeFrame(self, time_frame):
        self.lblFPS_Frame.setText(f"Time Frame: {time_frame:.2f} ms")
    def setTimeModel(self, time_model):
        self.lblFPS_Model.setText(f"Time Model: {time_model:.2f} ms")
    def setTimeSystem(self, time_system):
        self.lblFPS_System.setText(f"Time System: {time_system:.2f} ms")

    def setImgKq(self, img):
        self.imgBienSo.setPixmap(img)
    
    def setLabelBienSo(self, text):
        self.lblKQBienSo.setText(text)
    
    def setLabelTime(self, text):
        self.lblTime.setText(text)

    def setEventButtonCamera(self, func):
        self.btnCamera.clicked.connect(func)

    def setEventButtonVideo(self, func):
        self.btnVideo.clicked.connect(func)

    def start(self):
        self.show()

class System:
    def __init__(self, path_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model_for_prediction(
            checkpoint_path=path_model,
            yolo_base_model_path='Model_YOLO_RVT\\yolov11s-pytorch-default-v1\\best.pt'
        ).to(self.device).eval()

        if self.device.type == 'cuda':
            self.model = self.model.half()
            torch.backends.cudnn.benchmark = True 
            torch.backends.cuda.matmul.allow_tf32 = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.window = UI()
        
        # Video (Decord)
        self.vr = None
        self.frame_idx = 0
        
        # Camera (OpenCV)
        self.cap = None
        
        # Mode: 'video' hoặc 'camera' hoặc None
        self.mode = None
        
        self.timer = QTimer(self.window)
        self.timer.timeout.connect(self.updateFrame)
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        self.history = deque(maxlen=MAX_FRAME_HISTORY)
        self.previousLP = Result(None, None, 0.0, 0.0)

    def handleVideo(self):
        # Nếu đang chạy thì dừng
        if self.vr is not None or self.cap is not None:
            self.stopAll()
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Chọn video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not file_path:
            return
            
        print(f"Selected video: {file_path}")
        self.vr = VideoReader(file_path, ctx=cpu(0), num_threads=4)
        self.frame_idx = 0
        self.mode = 'video'
        print("Đang dùng Decord CPU để decode video")
        self.timer.start(0)

    def handleOpenCamera(self):
        self.stopAll()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        if not self.cap.isOpened():
            print("Không mở được camera")
            self.cap = None
            return
        self.mode = 'camera'
        print("Camera đã mở")
        self.timer.start(0)

    def handleCloseCamera(self):
        self.stopAll()

    def stopAll(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.vr = None
        self.frame_idx = 0
        self.mode = None
        self.window.lblScreen.clear()

    def updateFrame(self):
        # Bắt đầu đo thời gian tổng
        t_system_start = time.perf_counter()
        
        frame = None
        
        # Đo thời gian đọc frame
        t_frame_start = time.perf_counter()

        # Đọc frame theo mode
        if self.mode == 'video' and self.vr is not None:
            if self.frame_idx >= len(self.vr):
                print("Video kết thúc")
                self.stopAll()
                return
            frame = self.vr[self.frame_idx].asnumpy()  # RGB format
            self.frame_idx += 1
            
        elif self.mode == 'camera' and self.cap is not None:
            if not self.cap.isOpened():
                self.stopAll()
                return
            ret, frame = self.cap.read()
            if not ret:
                print("Camera lỗi")
                self.stopAll()
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return
        t_frame_end = time.perf_counter()
        time_frame = (t_frame_end - t_frame_start) * 1000  # ms
 
        # Tính FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.window.setFPS(fps)
            self.frame_count = 0
            self.last_fps_update = current_time

        # Convert frame to QImage
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Đo thời gian mô hình
        t_model_start = time.perf_counter()

        # Xử lý nhận diện
        license_plate, conf = predict_license_plate(self.model, frame, size=FRAME_SIZE)[0:2]
        
        t_model_end = time.perf_counter()
        time_model = (t_model_end - t_model_start) * 1000  # ms

        # print(f"Detected: {license_plate} conf: {conf:.2f}")
        
        if len(self.history) < MAX_FRAME_HISTORY and conf > BEST_CONFIDENCE_THRESHOLD:
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.window.imgBienSo.width(),
                self.window.imgBienSo.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.history.append(Result(pixmap, license_plate, conf, time.time()))
        elif len(self.history) > 0:
            best = max(self.history, key=lambda x: x.confidence)
            # print(f"Best: {best.license_plate} conf: {best.confidence:.2f}")
            if best.license_plate != self.previousLP.license_plate:
                self.previousLP = best
                self.displayResult(best.pixmap, best.license_plate, time.time() - best.timestamp)
            self.history.clear()

        # Hiển thị frame
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.window.lblScreen.width(),
            self.window.lblScreen.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.window.lblScreen.setPixmap(pixmap)

         # Đo thời gian tổng
        t_system_end = time.perf_counter()
        time_system_ms = (t_system_end - t_system_start) * 1000
        self.window.setTimeSystem(time_system_ms)
        self.window.setTimeFrame(time_frame)
        self.window.setTimeModel(time_model)
        print(f"Time Frame: {time_frame:.2f} ms, Time Model: {time_model:.2f} ms, Time System: {time_system_ms:.2f} ms")


    def displayResult(self, pixmap, license_plate, process_time):
        self.window.setLabelBienSo(f"{license_plate}")
        self.window.setLabelTime(f"{process_time:.2f}")
        self.window.setImgKq(pixmap)

    def start(self):
        self.window.setEventButtonCamera(self.handleOpenCamera)
        self.window.setEventButtonVideo(self.handleVideo)
        self.window.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    system = System('Model_YOLO_RVT\\final_yolo_rvit_model_E2E_92.pth')
    system.start()
    sys.exit(app.exec_())