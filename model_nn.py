import os.path
import threading
import time

import torch
import numpy as np
from BaseThread import BaseThread
from sklearn.preprocessing import StandardScaler
from QChartPlotImpl import QChartViewPlot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TrainNN(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.mae = None
        self.rmse = None
        self.r2 = 0.0
        self.model_nn = None
        self.train_result = False
        self.model_name = None
        self.save_path = None
        self.data_train = None
        self.plot_instance = None
        self.var_y = None
        self.mean_y = None
        self.var_x = None
        self.mean_x = None
        self.Y = None
        # Batch Size, Input Neurons, Hidden Neurons, Output Neurons
        self.N, self.D_in, self.H1, self.H2, self.D_out = 10, 7, 32, 16, 1
        self.iter_count = 4000
        self.learning_rate = 5e-4
        self.model_type = model_type
        self.class_name = "TrainNN"
        print("[TrainNN] ModelNN init done.")
        return

    def GetModelType(self):
        return self.model_type

    def GetClassName(self):
        return self.class_name

    def Normalize(self, train_x, train_y):
        # 归一化
        standardscaler = StandardScaler()
        standardscaler.fit(train_x)
        self.mean_x = standardscaler.mean_
        self.var_x = standardscaler.scale_
        train_x_norm = standardscaler.transform(train_x)

        standardscaler.fit(train_y.reshape(-1, 1))
        self.mean_y = standardscaler.mean_
        self.var_y = standardscaler.scale_
        train_y_norm = standardscaler.transform(train_y.reshape(-1, 1))
        super().SetRunLog("[TrainNN] Normalization Done.")
        return train_x_norm, train_y_norm

    def SetSuperParams(self, iterCount=4000, lrn_Rate=5e-4):
        self.iter_count = iterCount
        self.learning_rate = lrn_Rate
        super().SetRunLog("[TrainNN] iter count: {}, learning rate: {}".format(iterCount, lrn_Rate))
        return

    def InitModelNN(self):
        model_nn = None
        # Linear (Input -> Hidden), ReLU (Non-linearity), Linear (Hidden-> Output)
        model_nn = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(self.H1, self.H2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H2, self.D_out),
        )
        super().SetRunLog("[TrainNN] NN model init done.")
        return model_nn

    def SetPlotObject(self, plotObject: QChartViewPlot):
        super().SetRunLog("[TrainNN] Set plot object enter.")
        if plotObject is None:
            super().SetRunLog("[TrainNN] Invalid plot object.")
            return
        self.plot_instance = plotObject
        self.plot_instance.add_xy_axis()
        self.plot_instance.add_series("NN Training Loss")
        self.plot_instance.set_xrange(0, self.iter_count/100)
        self.plot_instance.set_yrange(0, 100)
        super().SetRunLog("[TrainNN] Set plot object done.")
        return

    def DrawLossCurve(self, data):
        self.plot_instance.handle_update(data)
        return

    def SetTrainingData(self, data_train, save_path, model_name):
        super().SetRunLog("[TrainNN] Set train data enter.")
        (m, n) = np.shape(data_train)
        if n < 8:
            super().SetRunLog("[TrainNN] Invalid input data.")
            return False
        self.data_train = data_train
        if os.path.exists(save_path):
            self.save_path = save_path
        else:
            self.save_path = "./"
        self.save_path = os.path.realpath(self.save_path)
        self.model_name = model_name
        return True

    def GetTrainModelName(self):
        return self.model_name

    def GetTrainingResult(self):
        return self.train_result, self.model_nn, self.rmse, self.mae, self.r2

    def run(self):
        super().SetThreadStarted(True)
        super().SetRunLog("[TrainNN] run thread enter.")
        train_x, train_y = self.data_train[:, 0:7], self.data_train[:, 7]
        Y = train_y
        print("TrainNN debug 0")
        train_x_norm, train_y_norm = self.Normalize(train_x, train_y)
        np.random.seed(0)
        xap = abs(train_x_norm + 0.2 * np.random.normal(size=train_x_norm.shape))
        # 转化为tensor
        x = torch.Tensor(train_x_norm)
        xp = torch.Tensor(xap)
        y = torch.Tensor(train_y_norm)
        y = y.unsqueeze(0)
        self.model_nn = self.InitModelNN()
        # Define the loss function: Mean Squared Error
        # The sum of the squares of the differences between prediction and ground truth
        loss_fn = torch.nn.MSELoss(reduction='mean')
        # The optimizer does a lot of the work of actually calculating gradients and
        # applying backpropagation through the network to update weights
        optimizer = torch.optim.Adam(self.model_nn.parameters(), lr=self.learning_rate, weight_decay=0.001)
        print("TrainNN, debug 1")
        loss_t = []
        loss_val = []
        self.plot_instance.set_xrange(0, self.iter_count / 100)
        for t in range(self.iter_count):
            # Forward pass: compute predicted y by passing x to the model.
            y_predict = self.model_nn(x)
            y_pp = self.model_nn(xp)

            # Compute loss and print it periodically
            se_loss = loss_fn(y_predict, y)

            l1_regularization, l2_regularization = \
                torch.tensor([0], dtype=torch.float32), torch.tensor([0], dtype=torch.float32)  # 定义L1及L2正则化损失
            for param in self.model_nn.parameters():
                l1_regularization += torch.norm(param, 1)  # L1正则化
                l2_regularization += torch.norm(param, 2)  # L2 正则化

            loss = se_loss + 0.02 * l2_regularization + 0.1 * torch.sum(torch.abs(y_pp - y_predict)) / \
                   torch.sum(torch.abs(xp - x))
            if t % 100 == 0:
                loss_value = loss.item()
                self.DrawLossCurve(loss_value)
                loss_t.append(loss_value)
            # Update the network weights using gradient of the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录进度
            progress = int(t * 100 / self.iter_count)
            super().update_train_signal.emit(progress)

        # TODO: 校验输出目录
        model_pn = '{}/{}.pkl'.format(self.save_path, self.model_name)
        torch.save(self.model_nn.state_dict(), model_pn)
        super().SetRunLog("[TrainNN] Model saved done: {}.".format(model_pn))
        self.train_result = True
        # 计算R^2
        y_pred_train = self.model_nn(x)
        y_pred_train = y_pred_train.detach().numpy()
        y_pred_train = y_pred_train * self.var_y + self.mean_y
        y_pred_train = np.squeeze(y_pred_train)
        self.r2 = r2_score(Y, y_pred_train)
        mse_nn = mean_squared_error(Y, y_pred_train)
        self.rmse = np.sqrt(mse_nn)
        self.mae = mean_absolute_error(Y, y_pred_train)
        # 发送信号给界面线程
        super().update_train_signal.emit(100)
        super().SetRunLog("[TrainNN] Train done.")
