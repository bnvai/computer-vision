import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PySide6.QtGui import QPixmap, QImage

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt OpenCV App")
        self.setGeometry(200, 200, 600, 500)

        self.label = QLabel("No Image")
        self.btn_load = QPushButton("Load Image")
        self.btn_gray = QPushButton("Convert Gray")

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_gray)
        self.setLayout(layout)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_gray.clicked.connect(self.convert_gray)

        self.cv_img = None

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if not path:
            return

        self.cv_img = cv2.imread(path)
        self.show_image(self.cv_img)

    def convert_gray(self):
        if self.cv_img is None:
            return

        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        self.show_image(gray)

    def show_image(self, img):
        if len(img.shape) == 2:
            height, width = img.shape
            bytes_per_line = width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())