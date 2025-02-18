from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from main import VoiceAssistant
import sys

class WorkerThread(QThread):
    finished = pyqtSignal()

    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant

    def run(self):
        self.assistant.run_voice_assistant()
        self.finished.emit()


class VoiceAssistantGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.assistant = VoiceAssistant()
        self.worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Voice Assistant")
        self.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout()

        self.label = QLabel("Hold the button to speak", self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.button = QPushButton("Hold to Speak", self)
        layout.addWidget(self.button)

        # Use only one signal for pressing and holding
        self.button.pressed.connect(self.handle_button_press)

        self.setLayout(layout)

    def handle_button_press(self):
        self.label.setText("Listening...")
        self.button.setEnabled(False)  # Disable button while processing

        self.worker = WorkerThread(self.assistant)
        self.worker.finished.connect(self.handle_finished)
        self.worker.start()

    def handle_finished(self):
        self.label.setText("Hold the button to speak")
        self.button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = VoiceAssistantGUI()
    gui.show()
    sys.exit(app.exec_())
