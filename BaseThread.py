import threading
import time
from PyQt5.QtCore import QThread, pyqtSignal, QTimer


def Default_Progress(update_train_signal):
    # 默认进度到99就停止更新
    for idx in range(0, 100):
        update_train_signal.emit(idx)
        print("[BaseThread] Emit value: {} done.".format(idx))
        time.sleep(0.05)


class BaseThread(QThread):
    # 用来更新进度取值为[0,100], 如果线程结束则为100
    update_train_signal = pyqtSignal(int)
    update_predict_signal = pyqtSignal(int)
    update_feature_signal = pyqtSignal(int)
    update_input_signal = pyqtSignal(int)
    # 异步日志输出
    update_runlog_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.started = False
        self.progress = 0
        self.class_name = "BaseThread"
        self.default_progress = None
        return

    def GetClassName(self):
        print("[BaseThread] This is BaseThread")
        return self.class_name

    def GetProgress(self):
        return self.progress

    def SetProgress(self, value):
        self.progress = value

    def SetThreadStarted(self, status):
        self.started = status

    def isStarted(self):
        return self.started

    def SetRunLog(self, log):
        self.update_runlog_signal.emit(log)

    # 启动一个线程更新线程进度条
    def DefaultProgressUpdate(self):
        self.default_progress = threading.Thread(target=Default_Progress, args=(self.update_train_signal,))
        self.default_progress.start()
