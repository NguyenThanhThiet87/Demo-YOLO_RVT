from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import uic
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

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