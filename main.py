from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic
<<<<<<< HEAD
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage
=======
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
import time
import sys
import cv2
import numpy as np
from collections import deque
import torch
from Model_YOLO_RVT.PredictModel import predict_license_plate, load_model_for_prediction
from decord import VideoReader, cpu

BEST_CONFIDENCE_THRESHOLD = 0.8
<<<<<<< HEAD
MAX_FRAME_HISTORY = 20
FRAME_SIZE = (480, 480)
=======
MAX_FRAME_HISTORY = 30
FRAME_SIZE = (640, 640)

>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
class Result:
    def __init__(self, pixmap, license_plate, confidence, timestamp):
        self.pixmap = pixmap
        self.license_plate = license_plate
        self.confidence = confidence
        self.timestamp = timestamp
<<<<<<< HEAD
    
=======

class CameraThread(QThread):
    """Thread riêng để đọc camera, tránh blocking main thread"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
    
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm latency
        
        self.running = True
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame_rgb)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def stop(self):
        self.running = False
        self.wait()

>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('untitled.ui', self)

    def setFPS(self, fps):
        self.lblFPS.setText(f"FPS: {fps:.0f}")
<<<<<<< HEAD
    def setTimeFrame(self, time_frame):
        self.lblFPS_Frame.setText(f"Time Frame: {time_frame:.2f} ms")
    def setTimeModel(self, time_model):
        self.lblFPS_Model.setText(f"Time Model: {time_model:.2f} ms")
    def setTimeSystem(self, time_system):
        self.lblFPS_System.setText(f"Time System: {time_system:.2f} ms")
=======
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf

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
        
<<<<<<< HEAD
        # Camera (OpenCV)
        self.cap = None
=======
        # Camera Thread
        self.camera_thread = None
        self.latest_camera_frame = None
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
        
        # Mode: 'video' hoặc 'camera' hoặc None
        self.mode = None
        
        self.timer = QTimer(self.window)
        self.timer.timeout.connect(self.updateFrame)
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        self.history = deque(maxlen=MAX_FRAME_HISTORY)
        self.previousLP = Result(None, None, 0.0, 0.0)

    def handleVideo(self):
<<<<<<< HEAD
        # Nếu đang chạy thì dừng
        if self.vr is not None or self.cap is not None:
=======
        if self.vr is not None or self.camera_thread is not None:
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
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

<<<<<<< HEAD
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
=======
    def onCameraFrame(self, frame):
        """Callback khi camera thread gửi frame mới"""
        self.latest_camera_frame = frame

    def handleOpenCamera(self):
        self.stopAll()
        
        self.camera_thread = CameraThread(0)
        self.camera_thread.frame_ready.connect(self.onCameraFrame)
        self.camera_thread.start()
        
        self.mode = 'camera'
        self.latest_camera_frame = None
        print("Camera đã mở (threaded)")
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
        self.timer.start(0)

    def handleCloseCamera(self):
        self.stopAll()

    def stopAll(self):
        self.timer.stop()
<<<<<<< HEAD
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
=======
        
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.vr = None
        self.frame_idx = 0
        self.mode = None
        self.latest_camera_frame = None
        self.window.lblScreen.clear()

    def updateFrame(self):
        frame = None
        
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
        if self.mode == 'video' and self.vr is not None:
            if self.frame_idx >= len(self.vr):
                print("Video kết thúc")
                self.stopAll()
                return
<<<<<<< HEAD
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
 
=======
            frame = self.vr[self.frame_idx].asnumpy()
            self.frame_idx += 1
            
        elif self.mode == 'camera':
            if self.latest_camera_frame is None:
                return  # Chưa có frame mới
            frame = self.latest_camera_frame
        else:
            return

>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
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
        
<<<<<<< HEAD
        # Đo thời gian mô hình
        t_model_start = time.perf_counter()

        # Xử lý nhận diện
        license_plate, conf = predict_license_plate(self.model, frame, size=FRAME_SIZE)[0:2]
        
        t_model_end = time.perf_counter()
        time_model = (t_model_end - t_model_start) * 1000  # ms

=======
        # Xử lý nhận diện
        license_plate, conf = predict_license_plate(self.model, frame, size=FRAME_SIZE)[0:2]
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
        print(f"Detected: {license_plate} conf: {conf:.2f}")
        
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
            print(f"Best: {best.license_plate} conf: {best.confidence:.2f}")
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

<<<<<<< HEAD
         # Đo thời gian tổng
        t_system_end = time.perf_counter()
        time_system_ms = (t_system_end - t_system_start) * 1000
        self.window.setTimeSystem(time_system_ms)
        self.window.setTimeFrame(time_frame)
        self.window.setTimeModel(time_model)
        print(f"Time Frame: {time_frame:.2f} ms, Time Model: {time_model:.2f} ms, Time System: {time_system_ms:.2f} ms")


=======
>>>>>>> 0183415f39c61c8ba1f0da357ffc677d07af9faf
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