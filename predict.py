import os
import threading
import time

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from BaseThread import BaseThread


class Predict(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.output_path = None
        self.progress = 0
        self.model = None
        self.plot_instance = None
        self.datap = None
        self.y_sort = None
        self.xpred = None
        self.n = None
        self.m = None
        self.x_pred_std = None
        self.vary = None
        self.meany = None
        self.varx = None
        self.meanx = None
        self.x_train = None
        self.y_train = None
        self.X = None
        self.Y = None
        self.Z = None
        self.model_type = model_type
        self.started = False
        print("[Predict] Init model: {} done.".format(self.model_type))
        return

    def get_model_type(self):
        return self.model_type

    def set_output_path(self, outputPath):
        self.output_path = os.path.realpath(outputPath)

    def set_normalize_data(self, data_train, data_pred):
        if data_pred is None:
            super().SetRunLog("[Predict] Invalid predict data.")
            return False
        if data_train is None and self.model_type == "NN":
            super().SetRunLog("[Predict] Invalid train data.")
            return False
        (m, n) = np.shape(data_pred)
        if n != 5:
            super().SetRunLog("[Predict] Invalid predict data.")
            return False

        x = np.arange(233, 433, 5)  # 温度范围,hf2
        y = np.arange(0, 8, 0.2)  # 应变范围,hf2
        self.X, self.Y = np.meshgrid(x, y)
        self.datap = np.zeros((len(x) * len(y), 7))
        self.m = len(x)
        self.n = len(y)
        dimension = self.m * self.n
        self.datap[:, 0:5] = np.tile(data_pred, (dimension, 1))
        self.datap[:, 5] = self.X.reshape(dimension, order='F')
        self.datap[:, 6] = self.Y.reshape(dimension, order='F')

        # 归一化
        if self.model_type == "NN":
            (m, n) = np.shape(data_train)
            if n < 8:
                super().SetRunLog("[Predict] Invalid input data.")
                return False
            x_train, y_train = data_train[:, 0:7], data_train[:, 7]
            standarscaler = StandardScaler()
            standarscaler.fit(x_train)
            self.meanx = standarscaler.mean_
            self.varx = standarscaler.scale_

            self.x_train = standarscaler.transform(x_train)
            self.x_pred_std = standarscaler.transform(self.datap)

            standarscaler.fit(y_train.reshape(-1, 1))
            self.meany = standarscaler.mean_
            self.vary = standarscaler.scale_
        super().SetRunLog("[Predict] Set data and normalize done.")
        print("[Predict] Set data and normalize done.")
        return True

    def set_model(self, model):
        self.model = model

    def predict(self):
        print("[Predict] Enter: {}".format(self.model_type))
        super().SetRunLog("[Predict] Predict Enter.")
        # 新数据预测，5D物理特征(多次预测取均值，消除噪声影响)
        print("[Predict] dim: {}, {}".format(self.m, self.n))
        y_pred = np.zeros((self.m * self.n, 50))
        if self.model_type == "NN":
            self.xpred = torch.Tensor(self.x_pred_std)
        else:
            print("[Predict] 000000")
            self.xpred = self.datap
        print("[PredictNN] Predict 00.")
        super().update_predict_signal.emit(10)
        for t in range(50):
            y_prediction = None
            if self.model_type == "NN":
                y_prediction = self.model(self.xpred)
                y_prediction = y_prediction.detach().numpy()
                y_prediction = y_prediction * self.vary + self.meany
            else:
                y_prediction = self.model.predict(self.xpred)
            y_pred[:, t] = np.squeeze(y_prediction)
        print("[PredictNN] Predict 01.")
        super().update_predict_signal.emit(30)
        y_mean = np.average(y_pred, axis=1)
        y_std = np.std(y_pred, axis=1)
        # print("[PredictNN] Predict 1.")
        self.y_sort = y_mean.reshape(self.m * self.n, order='F')
        y_sort_re = y_std.reshape(self.m * self.n, order='F')
        print(self.y_sort.shape)
        b = np.sort(self.y_sort)  # 从小到大排序，b为排序结果
        ind = np.argsort(self.y_sort)
        print("[PredictNN] Predict 2.")
        temp_max = self.datap[ind[-1], 5]
        stress_max = self.datap[ind[-1], 6]
        delta_t = b[-1]
        err = y_sort_re[ind[-1]]
        super().SetRunLog('最优绝热温度及对应的温度和应变：{:6f}, {}, {:6f}, {:6f}'.format(delta_t, temp_max, stress_max, err))
        super().update_predict_signal.emit(50)
        temp_1 = self.datap[ind[-2], 5]
        stress_1 = self.datap[ind[-2], 6]
        delta_t1 = b[-2]
        super().SetRunLog('次优绝热温度及对应的温度和应变：{:6f}, {}, {:6f}'.format(delta_t1, temp_1, stress_1))

        temp_2 = self.datap[ind[-3], 5]
        stress_2 = self.datap[ind[-3], 6]
        delta_t2 = b[-3]
        super().SetRunLog('第三绝热温度及对应的温度和应变：{:6f}, {}, {:6f}'.format(delta_t2, temp_2, stress_2))

        temp_3 = self.datap[ind[-4], 5]
        stress_3 = self.datap[ind[-4], 6]
        delta_t3 = b[-4]
        super().SetRunLog('第四绝热温度及对应的温度和应变：{:6f}, {}, {:6f}'.format(delta_t3, temp_3, stress_3))

        temp_4 = self.datap[ind[-5], 5]
        stress_4 = self.datap[ind[-5], 6]
        delta_t4 = b[-5]
        super().SetRunLog('第五绝热温度及对应的温度和应变：{:6f}, {}, {:6f}'.format(delta_t4, temp_4, stress_4))
        super().update_predict_signal.emit(80)
        super().SetRunLog("[Predict] Predict done.")

    def plot_3d_surface(self):
        figure = plt.figure()
        ax = Axes3D(figure)
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        np.random.seed(0)

        # 3D surface
        self.Z = self.y_sort.reshape(self.n, self.m, order='F')
        # row和cloum_stride为横竖方向的绘图采样步长，越小绘图越精细
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='rainbow')

        # 设置灰色显示区为白色
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.xaxis._axinfo["grid"].update({'linewidth': 1, 'linestyle': 'dotted'})
        ax.yaxis._axinfo["grid"].update({'linewidth': 1, 'linestyle': 'dotted'})
        ax.zaxis._axinfo["grid"].update({'linewidth': 1, 'linestyle': 'dotted'})

        ax.view_init(elev=25, azim=39)

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('$\mathregular{\epsilon}$ (%)')
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('$\mathregular{\Delta}$ T', rotation=360)
        # plt.show()
        plt.tight_layout()
        save_path_name = os.path.join(self.output_path, "predict.png")
        plt.savefig(save_path_name, bbox_inches='tight', dpi=300, pad_inches=0.0)
        print("[Predict] Save 3d surface done.")

    def GetResults(self):
        self.Z = self.y_sort.reshape(self.n, self.m, order='F')
        return self.X, self.Y, self.Z

    def isStarted(self):
        return super().isStarted()

    def GetProgress(self):
        return super().GetProgress()

    def run_predict(self):
        super().SetRunLog("[Predict] Thread: {} start running.".format(self.class_name))
        super().SetThreadStarted(True)
        self.predict()
        # 离线绘制3d表面
        # self.plot_3d_surface()
        super().update_predict_signal.emit(100)
