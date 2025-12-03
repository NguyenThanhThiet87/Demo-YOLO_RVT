import cv2
import threading
import queue
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import sys
from collections import deque
import torch
from Model_YOLO_RVT.PredictModel import predict_license_plate, load_model_for_prediction
from UI import UI


class Result:
    def __init__(self, pixmap, license_plate, confidence, timestamp):
        self.pixmap = pixmap
        self.license_plate = license_plate
        self.confidence = confidence
        self.timestamp = timestamp

YOLO_PATH = 'Model_YOLO_RVT\\yolov11s-pytorch-default-v1\\best.pt'
BEST_CONFIDENCE_THRESHOLD = 0.8
MAX_FRAME_HISTORY = 20
IMG_SIZE_MODEL = (480, 480)
FRAME_SIZE = (640, 480)

class System:
    def __init__(self, path_model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model_for_prediction(
            checkpoint_path=path_model,
            yolo_base_model_path=YOLO_PATH
        ).to(self.device).eval()

        if self.device.type == 'cuda':
            self.model = self.model.half()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.window = UI()

        # Queue và Event cho threading 
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        # Camera (OpenCV)
        self.video_path = None
        self.cap = None
        # Threads
        self.thread_read = None
        self.thread_ai = None

        self.history = deque(maxlen=MAX_FRAME_HISTORY)
        self.previousLP = Result(None, None, 0.0, 0.0)

    def handleVideo(self):
        # Nếu đang chạy thì dừng
        self.stopAll()

        file_path, _ = QFileDialog.getOpenFileName(
            self.window, "Chọn video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not file_path:
            return

        self.video_path = file_path

        print(f"Selected video: {file_path}")

        # Tạo và chạy threads
        self.thread_read = threading.Thread(target=self.thread_read_video, daemon=True)
        self.thread_ai = threading.Thread(target=self.thread_process_ai, daemon=True)
        self.thread_read.start()
        self.thread_ai.start()

    def handleOpenCamera(self):
        self.stopAll()

        # Tạo và chạy threads
        self.thread_read = threading.Thread(target=self.thread_read_camera, daemon=True)
        self.thread_ai = threading.Thread(target=self.thread_process_ai, daemon=True)
        self.thread_read.start()
        self.thread_ai.start()

    def stopAll(self):
        # Báo hiệu dừng threads
        self.stop_event.set()

        # Đợi threads kết thúc
        if self.thread_read is not None and self.thread_read.is_alive():
            self.thread_read.join(timeout=2)
        if self.thread_ai is not None and self.thread_ai.is_alive():
            self.thread_ai.join(timeout=2)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.window.lblScreen.clear()

    def thread_read_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Thread Video] Cannot open video: {self.video_path}")
            self.stopAll()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Thread Video] Video loaded: {total_frames} frames, {fps:.2f} FPS")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

            # Vứt bỏ frame cũ nếu queue đầy
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            # Nhét frame mới vào queue
            self.frame_queue.put(frame)

        cap.release()


    def thread_read_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[Thread Camera] Mất kết nối Camera.")
                self.stopAll()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB

            # Vứt bỏ frame cũ nếu queue đầy
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            # Nhét frame mới vào queue
            self.frame_queue.put(frame)

        cap.release()


    def thread_process_ai(self):
        curr_fps = 0

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            t0 = time.perf_counter() # Bắt đầu đo thời gian dự đoán
            # Predict
            license_plate, conf = predict_license_plate(self.model, frame, size=IMG_SIZE_MODEL)[0:2]
            t1 = time.perf_counter() # Kết thúc đo thời gian dự đoán

            # Xử lý kết quả
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
                if best.license_plate != self.previousLP.license_plate:
                    self.previousLP = best
                    self.displayResult(best.pixmap, best.license_plate, time.time() - best.timestamp)
                self.history.clear()
            t2 = time.perf_counter() # Kết thúc đo thời gian xử lý kết quả

            # Hiển thị frame lên màn hình
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.window.lblScreen.width(),
                self.window.lblScreen.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.window.lblScreen.setPixmap(pixmap)
            t3 = time.perf_counter() # Kết thúc đo thời gian hiển thị

            # In thời gian chi tiết
            time_model = (t1 - t0) * 1000
            time_post = (t2 - t1) * 1000
            time_display = (t3 - t2) * 1000
            time_total = (t3 - t0) * 1000

            print(f"Model: {time_model:.1f}ms | Post: {time_post:.1f}ms | Display: {time_display:.1f}ms | Total: {time_total:.1f}ms")

            # Tính FPS
            if time_total > 0:
                instant_fps = 1000.0 / time_total
                curr_fps = 0.9 * curr_fps + 0.1 * instant_fps

            self.window.setFPS(curr_fps)

        print("[Thread AI] Stopped.")

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