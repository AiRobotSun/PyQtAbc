import os
import threading
import logging
import datetime

from PyQt5.QtChart import QChartView
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QVBoxLayout

from BaseThread import BaseThread
from MachineLeaningTool import Ui_MachineLearningPlatform
from QChartPlotImpl import QChartViewPlot
from predict import Predict
from model_nn import TrainNN
from model_set import TrainRF, TrainGBR, TrainXGBR, TrainAdaboost, TrainCatboost
from DataLoaderSet import DataLoaderSet
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class MachineLearningToolImpl(QMainWindow, Ui_MachineLearningPlatform):
    def __init__(self, parent=None):
        super(MachineLearningToolImpl, self).__init__(parent)
        self.canvas = None
        self.figure = None
        self.predictor = None
        self.trainer = None
        self.inputPath = None
        self.predictPath = None
        self.modelPath = None
        self.featurePath = None
        self.outputPath = "./"  # 默认输出为当前目录
        self.inputFileDict = {}  # key为数据id，value为文件或文件夹路径
        self.raw_list = []
        self.feature_list = []
        self.model_list = []
        self.data_loader = DataLoaderSet()
        self.data_map = {}  # key为数据id，value为读取后的数据
        self.setupUi(self)
        self.pushButton_output_path.clicked.connect(self.SelectOutputPath)
        self.pushButton_input_path.clicked.connect(self.SelectInputPath)
        self.pushButton_input_file.clicked.connect(self.SelectInputFile)
        # self.pushButton_feature_auxpath.clicked.connect(self.SelectAuxPath)
        self.pushButton_output_open.clicked.connect(self.OpenOutputPath)
        self.pushButton_input_confirm.clicked.connect(self.ConfirmInput)
        # self.pushButton_feature_extract.clicked.connect(self.StartFeatureExtract)
        self.pushButton_train_start.clicked.connect(self.StartTrain)
        self.pushButton_predict_start.clicked.connect(self.StartPredict)
        # 初始化下拉框
        # 输入数据类型
        # self.comboBox_input_datatype.addItems(["IMG_RAW", "CSV_RAW", "XLSX_RAW",
        #                                        "XLSX_FEAT", "CSV_FEAT", "PKL_MOD-NN", "PTH_MOD-RF"])
        self.comboBox_input_datatype.addItems(["XLSX_FEAT", "CSV_FEAT", "PKL_MOD-NN", "PTH_MOD-RF"])
        self.comboBox_input_datatype.setCurrentIndex(0)     # 默认显示XLSX_FEAT
        self.comboBox_input_datatype.currentIndexChanged.connect(self.SelectInputDataType)
        # 特征类型
        """self.comboBox_feature_select.addItems(["WST", "VAE"])
        self.comboBox_feature_select.setCurrentIndex(0)
        self.comboBox_feature_select.currentIndexChanged.connect(self.SelectFeatureType)"""
        # 特征提取输入文件槽
        # self.comboBox_feature_file.currentIndexChanged.connect(self.SelectFeatureExtractInputFile)
        # 模型训练输入特征
        self.comboBox_train_feature.currentIndexChanged.connect(self.SelectTrainFeatureFile)
        # 模型类型
        self.modelList = ["RF", "GBR", "XGBR", "AB", "CAT", "NN"]
        self.comboBox_train_model.addItems(self.modelList)
        self.comboBox_train_model.setCurrentIndex(5)
        self.comboBox_train_model.currentIndexChanged.connect(self.SelectTrainModelType)
        self.comboBox_predict_modelType.addItems(self.modelList)
        self.comboBox_predict_modelType.setCurrentIndex(5)
        self.comboBox_predict_modelType.currentIndexChanged.connect(self.SelectPredictModelType)
        self.comboBox_predict_model.addItem("")
        self.comboBox_predict_model.setCurrentIndex(5)
        self.comboBox_predict_model.currentIndexChanged.connect(self.SelectPredictModelType)
        # 预测输入特征文件选择
        self.comboBox_predict_model.currentIndexChanged.connect(self.SelectPredictFeatureFile)
        # 初始化特征提取
        # 初始化训练
        # 初始化预测
        # 初始化绘图实例
        self.plot_qchart = None

        # 初始化线程状态监视线程
        self.lock = threading.Lock()
        self.thread_type_map = {}

        # 屏蔽部分功能
        self.lineEdit_sample_num.setEnabled(False)
        self.lineEdit_e_max.setEnabled(False)
        self.lineEdit_e_min.setEnabled(False)
        self.lineEdit_t_max.setEnabled(False)
        self.lineEdit_t_min.setEnabled(False)
        self.checkBox_feature_random.setEnabled(False)
        self.label_e_max.setEnabled(False)
        self.label_e_min.setEnabled(False)
        self.label_t_max.setEnabled(False)
        self.label_t_min.setEnabled(False)
        self.label_sample_num.setEnabled(False)
        # 训练模型除了NN其它的默认超参数不支持设置
        if self.comboBox_train_model.currentText() != "NN":
            self.lineEdit_learning_rate.setEnabled(False)
            self.lineEdit_iter_count.setEnabled(False)

        # 运行日志初始化
        logging.basicConfig(filename="run.log", level=logging.INFO)
        logging.info("--------------- Start Running Machine Learning Model Tool ------------------")
        logging.info("--------------- Time: {} -----------------".format(datetime.datetime.now()))
        self.UpdateRunlog("[UI] UI Init Done.")

    def ShowWarningInfo(self, message):
        QMessageBox.warning(self, "提示", message)

    def SelectPredictFeatureFile(self):
        self.UpdateRunlog("[UI][ModelPredict] Using feature file: {}".format(self.comboBox_predict_model.currentText()))
        return

    def SelectTrainFeatureFile(self):
        self.UpdateRunlog("[UI][ModelTraining] Using feature file: {}".format(self.comboBox_train_feature.currentText()))
        return

    def SelectFeatureExtractInputFile(self):
        self.UpdateRunlog("[UI][FeatureExtract] Using input file: {}".format(self.comboBox_feature_file.currentText()))
        return

    def SelectPredictModelType(self):
        self.UpdateRunlog("[UI][Predict] Predict Model type changed to {}".format(self.comboBox_predict_modelType.currentText()))
        if self.comboBox_predict_modelType.currentText() != "NN":
            self.comboBox_predict_normfeature.setEnabled(False)
        else:
            self.comboBox_predict_normfeature.setEnabled(True)

    def SelectTrainModelType(self):
        self.UpdateRunlog("[UI][Train] Train Model type changed to {}".format(self.comboBox_train_model.currentText()))
        if self.comboBox_train_model.currentText() != "NN":
            self.lineEdit_learning_rate.setEnabled(False)
            self.lineEdit_iter_count.setEnabled(False)
        else:
            self.lineEdit_learning_rate.setEnabled(True)
            self.lineEdit_iter_count.setEnabled(True)
        # 清空原有信息
        self.ClearTrainInfo()

    def SelectFeatureType(self):
        self.UpdateRunlog("[UI][FeatureExtract] Feature type changed to {}".format(self.comboBox_feature_select.currentText()))
        return

    def SelectInputDataType(self):
        self.UpdateRunlog("[UI][InputConfig] Input type changed to {}".format(self.comboBox_input_datatype.currentText()))
        return

    def MakeFolder(self, folderName):
        newPath = os.path.join(self.outputPath, folderName)
        if not os.path.exists(newPath):
            os.mkdir(newPath)
        return newPath

    def SelectOutputPath(self):
        self.UpdateRunlog("[UI] Select output path...")
        self.outputPath = QFileDialog.getExistingDirectory(self, "选择输出文件夹", os.getcwd())
        if self.outputPath is None or self.outputPath == "":
            self.outputPath = os.path.abspath("./")
        self.lineEdit_output_path.setText(self.outputPath)
        logging.info("Select output done: {}".format(self.outputPath))
        # 创建输出目录
        self.featurePath = self.MakeFolder("features")
        self.modelPath = self.MakeFolder("models")
        self.predictPath = self.MakeFolder("predicts")
        self.UpdateRunlog("[UI][OutputPath] features: {}, models: {}, predicts: {}".format(
            self.featurePath, self.modelPath, self.predictPath))

    def UpdateInputFileList(self):
        return

    def GetInputFileInfo(self):
        inputFileType = self.comboBox_input_datatype.currentText()
        self.UpdateRunlog("[UI] Get inputFileType: {}".format(inputFileType))
        elements = inputFileType.split("_")
        fileType = elements[0]
        fileClass = elements[1]
        return fileType, fileClass

    def SelectInputFile(self):
        prefix, postfix = self.GetInputFileInfo()
        file_type = "All Files(*)"
        if prefix == "CSV":
            file_type = "*.csv"
        elif prefix == "XLSX":
            file_type = "*.xlsx"
        elif prefix == "PKL":
            file_type = "*.pkl"
        elif prefix == "PTH":
            file_type = "*.pth"
        self.UpdateRunlog("[UI] Get file info: {}, {}, type:{}".format(prefix, postfix, file_type))
        file_pn, _ = QFileDialog.getOpenFileName(None, "打开文件", "", file_type)
        if file_pn is None:
            self.ShowWarningInfo("Invalid file.")
        file_name = file_pn.split("/")[-1].split(".")[0]
        key = "{}_{}_{}_{}".format(prefix, postfix, "file", file_name)
        self.UpdateRunlog("[UI] Add New folder: {} {}".format(key, file_pn))
        self.textEdit_input_datalist.append("{} {}".format(key, file_pn))

    def SelectInputPath(self):
        # 首先获取文件类型
        inputFileType = self.comboBox_input_datatype.currentText()
        inputPath = QFileDialog.getExistingDirectory(self, "选择输入文件夹", os.getcwd())
        # 不同类型文件放到字典中，key为[类型_文件夹], value为目录的完整路径
        folderName = inputPath.split("/")[-1]
        key = "{}_{}_{}".format(inputFileType, "folder", folderName)
        self.UpdateRunlog("[UI] Add New folder: {} {}".format(key, inputPath))
        self.textEdit_input_datalist.append("{} {}".format(key, inputPath))

    def SelectAuxPath(self):
        return

    def OpenOutputPath(self):
        self.UpdateRunlog("[UI] Open output path: {}.".format(self.outputPath))
        abs_path = os.path.relpath(self.outputPath)
        os.startfile(abs_path)
        self.UpdateRunlog("[UI] Open output path done.")

    def GetInputInfoFromId(self, input_id):
        if input_id is None:
            return
        # input_id 格式： {文件格式}_{用途}_{文件/文件夹}_{文件/文件夹名字}
        input_ext = input_id.split("_")[0]
        input_type = input_id.split("_")[1]
        input_class = input_id.split("_")[2]
        self.UpdateRunlog("[UI] Get input type: {} {} {}".format(input_ext, input_type, input_class))
        return input_ext, input_type, input_class

    def UpdateRawList(self):
        return

    def UpdateFeatList(self):
        return

    def UpdateModelList(self):
        return

    def GetIndexOfList(self, list, target):
        index = 0
        for item in list:
            if target == item:
                return index
            index += 1
        return index - 1

    def UpdateComboBox(self):
        current_index = 0
        text_train = self.comboBox_train_feature.currentText()
        # self.comboBox_feature_file.clear()
        self.comboBox_train_feature.clear()
        self.comboBox_predict_feature.clear()
        self.comboBox_predict_normfeature.clear()
        self.comboBox_predict_model.clear()
        # 特征提取
        current_index = len(self.raw_list) - 1
        # self.comboBox_feature_file.addItems(self.raw_list)
        # self.comboBox_feature_file.setCurrentIndex(current_index)
        # 训练数据
        current_index = len(self.feature_list) - 1
        print("UI Debug: {} ".format(text_train))
        if text_train != "":
            current_index = self.GetIndexOfList(self.feature_list, text_train)
        self.comboBox_train_feature.addItems(self.feature_list)
        self.comboBox_train_feature.setCurrentIndex(current_index)
        # 预测数据
        self.comboBox_predict_feature.addItems(self.feature_list)
        self.comboBox_predict_feature.setCurrentIndex(current_index)
        norm_feature_list = self.feature_list.copy()
        norm_feature_list.insert(0, '')
        self.comboBox_predict_normfeature.addItems(norm_feature_list)
        # 模型
        current_index = len(self.model_list) - 1
        self.comboBox_predict_model.addItems(self.model_list)
        self.comboBox_predict_model.setCurrentIndex(current_index)

    def GetDefaultDataInRunPath(self):
        self.feature_list.clear()
        self.model_list.clear()
        self.raw_list.clear()

    def LoadRawData(self):
        id_count = 0
        for input_id in self.raw_list:
            if input_id not in self.inputFileDict:
                self.UpdateRunlog("[UI] [ERROR] Invalid input: {}".format(input_id))
                continue
            data_path = self.inputFileDict[input_id]
            input_ext, input_type, input_class = self.GetInputInfoFromId(input_id)
            # 读取文件
            data = None
            if input_class == "file":
                if input_ext == "XLSX":
                    data = self.data_loader.LoadExcelData(data_path)
                elif input_ext == "CSV":
                    data = self.data_loader.LoadCsvData(data_path)
                else:
                    self.UpdateRunlog("[UI] [ERROR] Invalid format: {}".format(input_id))
                    continue
            # 读取目录中文件
            if input_class == "folder":
                if input_ext == "XLSX":
                    data = self.data_loader.LoadExcelsFromPath(data_path)
                elif input_ext == "CSV":
                    data = self.data_loader.LoadCsvsFromPath(data_path)
                elif input_ext == "IMG":
                    data = self.data_loader.LoadImagesFromPath(data_path)
                else:
                    self.UpdateRunlog("[UI] [ERROR] Invalid format: {}".format(input_id))
                    continue
            # 将数据放到data map中
            if data is not None:
                self.data_map[input_id] = data
                id_count += 1
        self.UpdateRunlog("[UI] Load all raw data done, num: {}.".format(id_count))

    def LoadFeature(self):
        feature_count = 0
        for input_id in self.feature_list:
            if input_id not in self.inputFileDict:
                self.UpdateRunlog("[UI] [ERROR] Invalid input: {}".format(input_id))
                continue
            data_path = self.inputFileDict[input_id]
            input_ext, input_type, input_class = self.GetInputInfoFromId(input_id)
            if input_class != "file":
                self.UpdateRunlog("[UI] [ERROR] Invalid input for feature: {}".format(input_id))
                continue
            data = None
            if input_ext == "XLSX":
                data = self.data_loader.LoadExcelData(data_path)
            elif input_ext == "CSV":
                data = self.data_loader.LoadCsvData(data_path)
            else:
                self.UpdateRunlog("[UI] [ERROR] Invalid feature format: {}".format(input_id))
                continue
            # 将数据放到data map中
            if data is not None:
                self.UpdateRunlog("[UI] [LoadInput] Success: {}".format(input_id))
                self.data_map[input_id] = data
                feature_count += 1
        self.UpdateRunlog("[UI] Load all feature data done, num: {}.".format(feature_count))

    def LoadModel(self):
        model_count = 0
        for input_id in self.model_list:
            if input_id not in self.inputFileDict:
                self.UpdateRunlog("[UI] [ERROR] Invalid input: {}".format(input_id))
                continue
            data_path = self.inputFileDict[input_id]
            input_ext, input_type, input_class = self.GetInputInfoFromId(input_id)
            if input_class != "file":
                self.UpdateRunlog("[UI] [ERROR] Invalid input for feature: {}".format(input_id))
                continue
            model_type = input_type.split("-")[1]
            model = None
            model = self.data_loader.LoadModel(model_type, data_path)
            if model is not None:
                self.data_map[input_id] = model
                model_count += 1
        self.UpdateRunlog("[UI] Load all models done, num: {}.".format(model_count))

    def LoadAllInputData(self):
        self.UpdateRunlog("[UI] Load all data, raw: {}, feature: {}, model: {}"
              .format(len(self.raw_list), len(self.feature_list), len(self.model_list)))
        if len(self.raw_list) != 0:
            self.LoadRawData()
        if len(self.feature_list) != 0:
            self.LoadFeature()
        if len(self.model_list) != 0:
            self.LoadModel()
        self.UpdateRunlog("[UI] [INFO] Load all input done.")

    def ConfirmInput(self):
        self.pushButton_input_confirm.setEnabled(False)
        self.GetDefaultDataInRunPath()
        # 根据textEdit表刷新输入文件列表
        files = self.textEdit_input_datalist.toPlainText()
        files = files.split("\n")
        self.UpdateRunlog("[UI] textEdit,len:{}, content: \n{}".format(len(files), files))
        if len(files) == 0 or len(files[0]) == 0:
            self.ShowWarningInfo("No input file.")
            self.pushButton_input_confirm.setEnabled(True)
            return
        for line in files:
            elements = line.split(" ")
            typeName = elements[0]
            filePath = elements[1]
            self.inputFileDict[typeName] = filePath  # 此处可能是文件，或者路径
            input_ext, input_type, input_class = self.GetInputInfoFromId(typeName)
            input_type = input_type.split("-")[0]  # model的类型格式为MOD-{算法模型}，如MOD-NN, MOD-RF
            if input_type == "RAW":
                self.raw_list.append(typeName)
            elif input_type == "FEAT":
                self.feature_list.append(typeName)
            elif input_type == "MOD":
                self.model_list.append(typeName)

        # 调用数据加载模块加载数据
        self.LoadAllInputData()
        # 将输入文件列表刷新到comboBox中，后续可以选择
        self.UpdateComboBox()
        self.pushButton_input_confirm.setEnabled(True)

    def StartFeatureExtract(self):
        return

    def ClearTrainInfo(self):
        self.lineEdit_learning_rate.clear()
        self.lineEdit_train_R2_train.clear()
        self.lineEdit_train_R2_test.clear()
        self.lineEdit_train_RMSE.clear()
        self.lineEdit_train_MAE.clear()
        self.lineEdit_iter_count.clear()
        self.progressBar_train_progress.setValue(0)

    def StartTrain(self):
        print("[Train] Enter.")
        # 校验，仅支持一个train任务
        if self.trainer is not None:
            if self.trainer.isRunning():
                self.ShowWarningInfo("Only one training task is supported.")
                return

        # 多次启动训练
        if self.trainer is not None:
            del self.trainer
        if self.plot_qchart is not None:
            del self.plot_qchart
        self.trainer = None
        self.plot_qchart = None
        self.ClearTrainInfo()
        # 准备数据
        model_type = self.comboBox_train_model.currentText()
        print("[Train] model_type is: {}.".format(model_type))
        if model_type == "NN":
            # 创建模型实例
            self.UpdateRunlog("[UI] [Train] Using NN trainer 1.")
            self.trainer = TrainNN(model_type)
            self.trainer.update_runlog_signal.connect(self.UpdateRunlog)
            # 初始化绘制区域
            self.plot_qchart = QChartViewPlot()
            self.graphicsView_plot.setChart(self.plot_qchart)
            self.graphicsView_plot.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
            self.graphicsView_plot.setRubberBand(QChartView.RectangleRubberBand)
            self.trainer.SetPlotObject(self.plot_qchart)
            self.UpdateRunlog("[UI] [Train] Using NN trainer 2.")
        elif model_type == "RF":
            self.trainer = TrainRF(model_type)
        elif model_type == "GBR":
            self.trainer = TrainGBR(model_type)
        elif model_type == "XGBR":
            self.trainer = TrainXGBR(model_type)
        elif model_type == "AB":
            self.trainer = TrainAdaboost(model_type)
        elif model_type == "CAT":
            self.trainer = TrainCatboost(model_type)
        else:
            self.ShowWarningInfo("Invalid Model Type!")
            return
        if model_type != "NN":
            self.trainer.update_runlog_signal.connect(self.UpdateRunlog)
        print("[Train] Model instance done.")

        # 准备输入数据
        train_feature_id = self.comboBox_train_feature.currentText()
        print("[Train] feature id is: {}".format(train_feature_id))
        if train_feature_id not in self.data_map:
            self.ShowWarningInfo("Invalid feature id.")
            return
        train_data = self.data_map[train_feature_id]
        self.UpdateRunlog("[UI] [Train] using feature: {}".format(train_feature_id))
        # NN需要设定超参数，其它模型超参数自动搜索
        model_id = ""
        if model_type == "NN":
            # 获取训练超参数
            lr = self.lineEdit_learning_rate.text()
            if lr == "":
                lr = "0.0005"
            learning_rate = float(lr)
            ic = self.lineEdit_iter_count.text()
            if ic == "":
                ic = "4000"
            iter_count = int(ic)
            if learning_rate == 0:
                learning_rate = 5e-4
            if iter_count == 0:
                iter_count = 4000
            self.trainer.SetSuperParams(iter_count, learning_rate)
            model_id = "MOD-{}_{}_{}-{}".format(model_type, train_feature_id, iter_count, learning_rate)
        else:
            model_id = "MOD-{}_{}".format(model_type, train_feature_id)

        # 检查结果保存目录
        if self.modelPath is None:
            self.modelPath = "./models"
            if not os.path.exists(self.modelPath):
                os.mkdir(self.modelPath)
        if not self.trainer.SetTrainingData(train_data, self.modelPath, model_id):
            self.ShowWarningInfo("Invalid Training Data!")
            return
        self.trainer.update_train_signal.connect(self.UpdateTrainProgress)
        # 开始训练
        self.pushButton_train_start.setEnabled(False)
        self.trainer.start()
        print("[Train] Model train start done.")
        self.UpdateRunlog("[UI] [Train] Start train thread done.")

    def StartPredict(self):
        # 校验，仅支持一个predict任务
        if self.predictor is not None:
            if self.predictor.isRunning():
                self.ShowWarningInfo("Only one predict task is supported.")
                return
        # 多次启动训练
        if self.predictor is not None:
            del self.predictor
        self.UpdateRunlog("[UI] [Predict] Predictor Checked done.")
        # 获取输入信息
        model_type = self.comboBox_predict_modelType.currentText()
        self.predictor = None
        self.predictor = Predict(model_type)
        self.UpdateRunlog("[UI] [Predict] Using {} predictor.".format(model_type))

        model_id = self.comboBox_predict_model.currentText()
        predict_feature_id = self.comboBox_predict_feature.currentText()
        predict_normfeature_id = self.comboBox_predict_normfeature.currentText()
        self.UpdateRunlog("[UI] [Predict] Inputs: model_id={}, predict_feature={}, norm_feature={}"
                          .format(model_id, predict_feature_id, predict_normfeature_id))
        # 选择特征文件
        norm_data = None
        if model_type == "NN":
            if predict_normfeature_id not in self.data_map:
                self.ShowWarningInfo("No feature [{}] for normalization.".format(predict_normfeature_id))
                return
            norm_data = self.data_map[predict_normfeature_id]
        predict_data = None
        if predict_feature_id not in self.data_map:
            self.ShowWarningInfo("No predict feature [{}].".format(predict_feature_id))
            return
        predict_data = self.data_map[predict_feature_id]
        # 检查结果保存目录
        if self.predictPath is None:
            self.predictPath = "./predicts"
            if not os.path.exists(self.predictPath):
                os.mkdir(self.predictPath)
        self.predictor.set_output_path(self.predictPath)
        self.predictor.update_runlog_signal.connect(self.UpdateRunlog)
        if not self.predictor.set_normalize_data(norm_data, predict_data):
            self.ShowWarningInfo("Invalid Predict Data!")
            return
        # 选择模型
        if model_id not in self.data_map:
            self.ShowWarningInfo("No model [{}] found.".format(model_id))
            return
        model = self.data_map[model_id]
        self.predictor.set_model(model)
        self.predictor.update_predict_signal.connect(self.UpdatePredictProgress)
        print("[UI][Predict] All inputs are ready.")
        self.pushButton_predict_start.setEnabled(False)
        self.predictor.run_predict()
        self.UpdateRunlog("[UI] [Predict] Start predict done.")

    # value为运行进度，int
    def UpdateTrainProgress(self, value):
        print("[UI] [Signal] Receive train progress signal: {}".format(value))
        self.progressBar_train_progress.setValue(value)
        # 如果进度为100，则保存模型，销毁线程
        if value == 100:
            if self.trainer.GetModelType() == "NN":
                success, model_result, rmse, mae, r2 = self.trainer.GetTrainingResult()
                model_id = self.trainer.GetTrainModelName()
                if success:
                    self.data_map[model_id] = model_result
                    # 多次训练去重
                    if model_id in self.model_list:
                        self.model_list.remove(model_id)
                        print("[UI][Signal] Update model: {}".format(model_id))
                    self.UpdateComboBox()
                    # NN的test置灰
                    self.lineEdit_train_R2_test.setEnabled(False)
                    self.label_train_R2_test.setEnabled(False)
                    # 取保留小数点6位显示
                    self.lineEdit_train_R2_train.setText("{:6f}".format(r2))
                    self.lineEdit_train_R2_train.show()
                    self.lineEdit_train_MAE.setText("{:6f}".format(mae))
                    self.lineEdit_train_MAE.show()
                    self.lineEdit_train_RMSE.setText("{:6f}".format(rmse))
                    self.lineEdit_train_RMSE.show()
            else:
                model_result, rmse, mae, r2_train, r2_test = self.trainer.GetTrainingResults()
                model_id = self.trainer.GetModelId()
                self.data_map[model_id] = model_result
                # 多次训练去重
                if model_id in self.model_list:
                    self.model_list.remove(model_id)
                    print("[UI][Signal] Update model: {}".format(model_id))
                self.model_list.append(model_id)
                self.UpdateComboBox()
                print("[UI][Signal] Save model done.")
                # NN的test置灰
                self.lineEdit_train_R2_test.setEnabled(True)
                self.label_train_R2_test.setEnabled(True)
                # 取保留小数点6位显示
                self.lineEdit_train_R2_train.setText("{:6f}".format(r2_train))
                self.lineEdit_train_R2_train.show()
                self.lineEdit_train_R2_test.setText("{:6f}".format(r2_test))
                self.lineEdit_train_R2_test.show()
                self.lineEdit_train_MAE.setText("{:6f}".format(mae))
                self.lineEdit_train_MAE.show()
                self.lineEdit_train_RMSE.setText("{:6f}".format(rmse))
                self.lineEdit_train_RMSE.show()
            print("[UI][Signal] done done done.")
            self.pushButton_train_start.setEnabled(True)
            self.UpdateRunlog("[UI] [Signal] Train thread processing done.")

    def plot_3d_surface(self, X, Y, Z, figure, model_type):
        self.UpdateRunlog("[UI] [Signal] Plot 3d surface start.")
        ax = Axes3D(figure)
        ax.cla()
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        np.random.seed(0)
        # 3D surface
        # row和cloum_stride为横竖方向的绘图采样步长，越小绘图越精细
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        # 设置灰色显示区为白色
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.xaxis._axinfo["grid"].update({'linewidth': 1, 'linestyle': 'dotted'})
        ax.yaxis._axinfo["grid"].update({'linewidth': 1, 'linestyle': 'dotted'})
        ax.zaxis._axinfo["grid"].update({'linewidth': 1, 'linestyle': 'dotted'})
        ax.view_init(elev=25, azim=39)
        # 设置x,y方向的名称和单位
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('$\mathregular{\epsilon}$ (%)')
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel('$\mathregular{\Delta}$ T', rotation=360)
        # 显示和保存
        # plt.tight_layout()
        save_pn = os.path.join(os.path.realpath(self.predictPath), "predict_{}.png".format(model_type))
        plt.savefig(save_pn, bbox_inches='tight', dpi=300, pad_inches=0.0)
        # 使用离线保存数据显示
        # pix = QPixmap(os.path.join(self.predictPath, 'predict.png'))
        # self.label_predict_plot.setPixmap(pix)
        # self.label_predict_plot.show()
        # self.UpdateRunlog("[UI] [Signal] Plot 3d surface done.")

    # value为运行进度，int
    def UpdatePredictProgress(self, value):
        print("[Signal] Receive predict progress signal: {}".format(value))
        # 清理线程
        if value == 100:
            # 创建画布
            figure = plt.figure()
            figure.clf()
            canvas = FigureCanvas(figure)
            # 把画布放进widget组件
            vLayout = QVBoxLayout()
            vLayout.addWidget(canvas)
            self.widget_plot_predict.setLayout(vLayout)
            self.widget_plot_predict.setObjectName("predict_3d_surface")
            X, Y, Z = self.predictor.GetResults()
            model_type = self.predictor.get_model_type()
            self.plot_3d_surface(X, Y, Z, figure, model_type)
            canvas.draw()
            canvas.flush_events()
            print("[Signal] Plot result image done.")
            self.progressBar_predict_progress.setValue(value)
            self.pushButton_predict_start.setEnabled(True)
        else:
            self.progressBar_predict_progress.setValue(value)

    # value为运行进度，int
    def UpdateFeatureProgress(self, value):
        print("[Signal] Receive feature extract progress signal: {}".format(value))
        self.progressBar_feature_progress.setValue(value)
        # 清理线程
        if value == 100:
            # TODO: Save features
            # TODO: Save done
            pass

    # value为运行日志，str
    def UpdateRunlog(self, value):
        run_log = "[{}]: {}".format(datetime.datetime.now(), value)
        self.textBrowser_runlog.append(run_log)
        # print("[UI] Update run log done.")
