import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from BaseThread import BaseThread


# catboost
class TrainCatboost(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.output_model_id = None
        self.output_path = None
        self.model = None
        self.r2_test = None
        self.r2_train = None
        self.mae = None
        self.trmse = None
        self.rmse = None
        self.param_grid = {
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
        }
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.model_type = model_type
        print("[Train] Model CAT init done.")

    def GetModelType(self):
        return self.model_type

    def GetModelId(self):
        return self.output_model_id

    def SetTrainingData(self, data, model_path, model_id):
        (m, n) = np.shape(data)
        if m < 230 or n < 8:
            print("[ModelSet] Invalid input.")
            return False
        x_train = data[0:230, 0:7]
        y_train = data[0:230, 7]
        x = x_train
        y = y_train
        # testing data （Ta2和Ta4）
        x_test = data[230:, 0:7]
        y_test = data[230:, 7]
        self.SetTrainInputData(x, y, x_test, y_test)
        self.output_path = model_path
        self.output_model_id = model_id
        print("[Train] SetTrainingData enter.")
        return True

    def SetTrainInputData(self, x, y, x_test, y_test):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

    def GetTrainingResults(self):
        return self.model, self.trmse, self.mae, self.r2_train, self.r2_test

    def train(self):
        print("[Train] train enter.")
        super().DefaultProgressUpdate()
        estimator = CatBoostRegressor(iterations=1000, posterior_sampling=True, verbose=False)
        self.model = GridSearchCV(estimator, self.param_grid, cv=3)
        self.model.fit(self.x, self.y)
        y_hat_catboost = self.model.predict(self.x_test)
        # np.savetxt("test_phy_cat.csv",y_hat_catboost,delimiter=',', fmt='%.5f')
        # print(y_hat_catboost)
        y_pred_cat = self.model.predict(self.x)
        # np.savetxt("train_phy_cat.csv",y_pred_cat,delimiter=',', fmt='%.5f')
        mse_cat = mean_squared_error(self.y_test, y_hat_catboost)
        self.rmse = np.sqrt(mse_cat)
        tmse_cat = mean_squared_error(self.y, y_pred_cat)
        self.trmse = np.sqrt(tmse_cat)
        self.mae = mean_absolute_error(self.y_test, y_hat_catboost)
        self.r2_train = r2_score(self.y, y_pred_cat)
        self.r2_test = r2_score(self.y_test, y_hat_catboost)
        super().SetRunLog("catBoost-testing: MSE={}  tRMSE={}  RMSE={}  MAE={}  R2_train={} R2_test={}"
                          .format(mse_cat, self.trmse, self.rmse, self.mae, self.r2_train, self.r2_test))

    def run(self):
        self.train()
        super().update_train_signal.emit(100)
        super().SetRunLog("[TrainCatboost] Train done.")


# GBR
class TrainGBR(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.output_model_id = None
        self.output_path = None
        self.r2_test = None
        self.r2_train = None
        self.mae = None
        self.trmse = None
        self.rmse = None
        self.model = None
        self.param_grid = {'learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2],
                           'max_depth': [2, 3, 4, 5, 6],
                           'n_estimators': [50, 100, 150, 200],
                           }
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.model_type = model_type
        print("[Train] Model GBR init done.")

    def GetModelType(self):
        return self.model_type

    def GetModelId(self):
        return self.output_model_id

    def SetTrainingData(self, data, model_path, model_id):
        (m, n) = np.shape(data)
        if m < 230 or n < 8:
            print("[ModelSet] Invalid input.")
            return False
        x_train = data[0:230, 0:7]
        y_train = data[0:230, 7]
        x = x_train
        y = y_train
        # testing data （Ta2和Ta4）
        x_test = data[230:, 0:7]
        y_test = data[230:, 7]
        self.SetTrainInputData(x, y, x_test, y_test)
        self.output_path = model_path
        self.output_model_id = model_id
        print("[Train] SetTrainingData enter.")
        return True

    def SetTrainInputData(self, x, y, x_test, y_test):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

    def GetTrainingResults(self):
        return self.model, self.trmse, self.mae, self.r2_train, self.r2_test

    def train(self):
        print("[Train] train enter.")
        super().DefaultProgressUpdate()
        estimator = GradientBoostingRegressor(loss='squared_error', subsample=0.85)
        self.model = GridSearchCV(estimator, self.param_grid, cv=3)
        self.model.fit(self.x, self.y)
        y_hat_gbr = self.model.predict(self.x_test)
        # np.savetxt("test_phy_gbr.csv",y_hat_gbr,delimiter=',', fmt='%.5f')
        y_pred = self.model.predict(self.x)
        # np.savetxt("train_phy_gbr.csv",y_pred,delimiter=',', fmt='%.5f')
        mse_gbr = mean_squared_error(self.y_test, y_hat_gbr)
        self.rmse = np.sqrt(mse_gbr)
        tmse_gbr = mean_squared_error(self.y, y_pred)
        self.trmse = np.sqrt(tmse_gbr)
        self.mae = mean_absolute_error(self.y_test, y_hat_gbr)
        self.r2_train = r2_score(self.y, y_pred)
        self.r2_test = r2_score(self.y_test, y_hat_gbr)
        super().SetRunLog("GBR-testing: MSE={}  tRMSE={}  RMSE={}  MAE={}  R2_train={}  R2_test={}"
                          .format(mse_gbr, self.trmse, self.rmse, self.mae, self.r2_train, self.r2_test))

    def run(self):
        self.train()
        super().update_train_signal.emit(100)
        super().SetRunLog("[TrainGBR] Train done.")


# XGBoost
class TrainXGBR(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.output_model_id = None
        self.output_path = None
        self.param_grid = {'max_depth': [2, 3, 4, 5, 6],
                           'n_estimators': [50, 100, 150, 200],
                           'objective': ['reg:squarederror', 'reg:squaredlogerror'],
                           }
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.r2_test = None
        self.r2_train = None
        self.mae = None
        self.trmse = None
        self.rmse = None
        self.model = None
        self.model_type = model_type
        print("[Train] Model XGBR init done.")

    def GetModelType(self):
        return self.model_type

    def GetModelId(self):
        return self.output_model_id

    def SetTrainingData(self, data, model_path, model_id):
        (m, n) = np.shape(data)
        if m < 230 or n < 8:
            print("[ModelSet] Invalid input.")
            return False
        x_train = data[0:230, 0:7]
        y_train = data[0:230, 7]
        x = x_train
        y = y_train
        # testing data （Ta2和Ta4）
        x_test = data[230:, 0:7]
        y_test = data[230:, 7]
        self.SetTrainInputData(x, y, x_test, y_test)
        self.output_path = model_path
        self.output_model_id = model_id
        print("[Train] SetTrainingData enter.")
        return True

    def SetTrainInputData(self, x, y, x_test, y_test):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

    def GetTrainingResults(self):
        return self.model, self.trmse, self.mae, self.r2_train, self.r2_test

    def train(self):
        print("[Train] train enter.")
        super().DefaultProgressUpdate()
        estimator = xgb.XGBRegressor(booster='gbtree')
        self.model = GridSearchCV(estimator, self.param_grid, cv=3)
        self.model.fit(self.x, self.y)
        y_hat_xgbr = self.model.predict(self.x_test)
        y_train_xgbr = self.model.predict(self.x)
        # np.savetxt("test_phy_xgbr.csv",y_hat_xgbr,delimiter=',', fmt='%.5f')
        # np.savetxt("train_phy_xgbr.csv",y_train_xgbr,delimiter=',', fmt='%.5f')
        mse_xgbr = mean_squared_error(self.y_test, y_hat_xgbr)
        self.rmse = np.sqrt(mse_xgbr)
        tmse_xgbr = mean_squared_error(self.y, y_train_xgbr)
        self.trmse = np.sqrt(tmse_xgbr)
        self.mae = mean_absolute_error(self.y_test, y_hat_xgbr)
        self.r2_train = r2_score(self.y, y_train_xgbr)
        self.r2_test = r2_score(self.y_test, y_hat_xgbr)
        super().SetRunLog("XGBR-testing: MSE={}  tRMSE={}  RMSE={}  MAE={}  R2_train={}  R2_test={}"
                          .format(mse_xgbr, self.trmse, self.rmse, self.mae, self.r2_train, self.r2_test))

    def run(self):
        self.train()
        super().update_train_signal.emit(100)
        super().SetRunLog("[TrainXGBR] Train done.")


# random forest
class TrainRF(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.output_model_id = None
        self.output_path = None
        self.param_grid = {'max_depth': [2, 3, 4, 5, 6],
                           'n_estimators': [50, 100, 150, 200],
                           'criterion': ['squared_error', 'absolute_error'],
                           }
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.r2_test = None
        self.r2_train = None
        self.mae = None
        self.trmse = None
        self.rmse = None
        self.model = None
        self.model_type = model_type
        print("[Train] Model RF init done.")

    def GetModelType(self):
        return self.model_type

    def GetModelId(self):
        return self.output_model_id

    def SetTrainingData(self, data, model_path, model_id):
        (m, n) = np.shape(data)
        if m < 230 or n < 8:
            print("[ModelSet] Invalid input.")
            return False
        x_train = data[0:230, 0:7]
        y_train = data[0:230, 7]
        x = x_train
        y = y_train
        # testing data （Ta2和Ta4）
        x_test = data[230:, 0:7]
        y_test = data[230:, 7]
        self.SetTrainInputData(x, y, x_test, y_test)
        self.output_path = model_path
        self.output_model_id = model_id
        print("[Train] Model RF set training data done.")
        return True

    def SetTrainInputData(self, x, y, x_test, y_test):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

    def GetTrainingResults(self):
        return self.model, self.trmse, self.mae, self.r2_train, self.r2_test

    def train(self):
        print("[Train] train enter.")
        super().DefaultProgressUpdate()
        estimator = RandomForestRegressor()
        self.model = GridSearchCV(estimator, self.param_grid, cv=3)
        self.model.fit(self.x, self.y)
        y_hat_rf = self.model.predict(self.x_test)
        y_train_rf = self.model.predict(self.x)
        # np.savetxt("test_phy_rf.csv",y_hat_rf,delimiter=',', fmt='%.5f')
        # np.savetxt("train_phy_rf.csv",y_train_rf,delimiter=',', fmt='%.5f')
        mse_rf = mean_squared_error(self.y_test, y_hat_rf)
        self.rmse = np.sqrt(mse_rf)
        tmse_rf = mean_squared_error(self.y, y_train_rf)
        self.trmse = np.sqrt(tmse_rf)
        self.mae = mean_absolute_error(self.y_test, y_hat_rf)
        self.r2_train = r2_score(self.y, y_train_rf)
        self.r2_test = r2_score(self.y_test, y_hat_rf)
        super().SetRunLog("random forest-testing: MSE={}  tRMSE={}  RMSE={}  MAE={}  R2_train={}  R2_test={}"
                          .format(mse_rf, self.trmse, self.rmse, self.mae, self.r2_train, self.r2_test))

    def run(self):
        self.train()
        super().update_train_signal.emit(100)
        super().SetRunLog("[TrainRF] Train done.")


# AdaBoostRegressor
class TrainAdaboost(BaseThread):
    def __init__(self, model_type):
        super().__init__()
        self.output_model_id = None
        self.output_path = None
        self.param_grid = {'n_estimators': [50, 100, 150, 200],
                           'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
                           'loss': ['linear', 'square', 'exponential'],
                           }
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None
        self.r2_test = None
        self.r2_train = None
        self.mae = None
        self.trmse = None
        self.rmse = None
        self.model = None
        self.model_type = model_type
        print("[Train] Model Adaboost init done.")

    def GetModelType(self):
        return self.model_type

    def GetModelId(self):
        return self.output_model_id

    def SetTrainingData(self, data, model_path, model_id):
        (m, n) = np.shape(data)
        if m < 230 or n < 8:
            print("[ModelSet] Invalid input.")
            return False
        x_train = data[0:230, 0:7]
        y_train = data[0:230, 7]
        x = x_train
        y = y_train
        # testing data （Ta2和Ta4）
        x_test = data[230:, 0:7]
        y_test = data[230:, 7]
        self.SetTrainInputData(x, y, x_test, y_test)
        self.output_path = model_path
        self.output_model_id = model_id
        print("[Train] SetTrainingData enter.")
        return True

    def SetTrainInputData(self, x, y, x_test, y_test):
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

    def GetTrainingResults(self):
        return self.model, self.trmse, self.mae, self.r2_train, self.r2_test

    def train(self):
        print("[Train] train enter.")
        super().DefaultProgressUpdate()
        estimator = AdaBoostRegressor()
        self.model = GridSearchCV(estimator, self.param_grid, cv=3)
        self.model.fit(self.x, self.y)
        y_hat_ada = self.model.predict(self.x_test)
        y_train_boost = self.model.predict(self.x)
        # np.savetxt("test_phy_ada.csv",y_hat_ada,delimiter=',', fmt='%.5f')
        # np.savetxt("train_phy_ada.csv",y_train_boost,delimiter=',', fmt='%.5f')
        mse_ada = mean_squared_error(self.y_test, y_hat_ada)
        self.rmse = np.sqrt(mse_ada)
        tmse_ada = mean_squared_error(self.y, y_train_boost)
        self.trmse = np.sqrt(tmse_ada)
        self.mae = mean_absolute_error(self.y_test, y_hat_ada)
        self.r2_train = r2_score(self.y, y_train_boost)
        self.r2_test = r2_score(self.y_test, y_hat_ada)
        super().SetRunLog("AdaBoostRegressor-testing: MSE={}  tRMSE={}  RMSE={}  MAE={}  R2_train={}  R2_test={}"
                          .format(mse_ada, self.trmse, self.rmse, self.mae, self.r2_train, self.r2_test))

    def run(self):
        self.train()
        super().update_train_signal.emit(100)
        super().SetRunLog("[TrainAdaboost] Train done.")
