import logging
import datetime
import os
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtChart import QChartView, QLegend
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QTableWidgetItem, QHeaderView, QGraphicsLayout

import DataInputOutput
import FeatureExtractSet
from FeatureTechs import Ui_FeatureEngineering
from QChartPlotImpl import QChartViewPlot
from FeatureExtractSet import FeatureExtractor


class FeatureEngineeringImpl(QMainWindow, Ui_FeatureEngineering):
    def __init__(self, parent=None):
        super(FeatureEngineeringImpl, self).__init__(parent)
        self.inputLoader = None
        self.featurePathImg = "./features_img"
        self.featurePathExcel = "./features_xlsx"
        self.canvas = None
        self.figure = None
        self.inputPath = None
        self.outputPath = "./"  # 默认输出为当前目录
        self.inputDataExcel = {}
        self.inputDataImage = {}
        self.extractor = None
        self.usingWavelet = False
        self.setupUi(self)

        # 初始化特征提取方法
        self.comboBox_feature_select.addItems(["WST", "VAE"])

        # 初始化界面操作
        self.pushButton_output_path.clicked.connect(self.SelectOutputPath)
        self.pushButton_input_path.clicked.connect(self.SelectInputPath)
        self.pushButton_input_open.clicked.connect(self.OpenInputPath)
        self.pushButton_output_open.clicked.connect(self.OpenOutputPath)
        self.pushButton_input_clear.clicked.connect(self.ClearInputData)
        self.comboBox_feature_select.currentIndexChanged.connect(self.SelectFeatureType)
        self.pushButton_feature_extract.clicked.connect(self.FeatureExtract)
        self.checkBox_wavelet.clicked.connect(self.SetWaveletTransform)
        # self.pushButton_input_preview.clicked.connect(self.InputPreview)
        # 支持鼠标点击切换预览
        self.tableWidget_input_list.cellClicked.connect(self.InputPreview)
        # 支持键盘上下移动切换预览
        self.tableWidget_input_list.currentCellChanged.connect(self.CellChangedProcess)
        # 初始化绘制区域
        self.plot_qtChart = QChartViewPlot()
        self.graphicsView_input_preview.setChart(self.plot_qtChart)
        self.graphicsView_input_preview.setRenderHint(QPainter.Antialiasing)             # 抗锯齿
        self.graphicsView_input_preview.setRubberBand(QChartView.RectangleRubberBand)
        self.plot_qtChart.add_xy_axis()
        self.plot_qtChart.set_xrange(0, 100)
        self.plot_qtChart.set_yrange(0, 100)
        self.plot_qtChart.legend().setAlignment(Qt.AlignTop)
        # self.plot_qtChart.legend().setLayout(QLegend.TopToBottom)

        # 运行日志初始化
        logging.basicConfig(filename="run.log", level=logging.INFO)
        logging.info("--------------- Start Running Machine Learning Model Tool ------------------")
        logging.info("--------------- Time: {} -----------------".format(datetime.datetime.now()))
        self.UpdateRunlog("[UI] Init done.")

    # value为运行日志，str
    def UpdateRunlog(self, value):
        run_log = "[{}]: {}".format(datetime.datetime.now(), value)
        self.textBrowser_runlog.append(run_log)

    def ShowWarningInfo(self, message):
        QMessageBox.warning(self, "提示", message)

    def SetWaveletTransform(self):
        if self.checkBox_wavelet.isChecked():
            self.usingWavelet = True
            self.label_input_preview_image.setVisible(True)
            self.graphicsView_input_preview.setVisible(False)
            self.UpdateRunlog("[UI] Using wavelet transforming.")
        else:
            self.usingWavelet = False
            self.label_input_preview_image.setVisible(False)
            self.graphicsView_input_preview.setVisible(True)
            self.UpdateRunlog("[UI] No preprocessing.")
        # 更新绘制区域
        row = self.tableWidget_input_list.currentRow()
        col = self.tableWidget_input_list.currentColumn()
        self.InputPreview(row, col)

    def SelectFeatureType(self):
        self.UpdateRunlog("[UI] Feature type changed to {}".format(self.comboBox_feature_select.currentText()))
        self.progressBar_feature_progress.setValue(0)
        return

    def OpenPathInWindows(self, path):
        abs_path = os.path.relpath(path)
        os.startfile(abs_path)

    def OpenInputPath(self):
        if self.inputPath is None or self.inputPath == "":
            self.ShowWarningInfo("Invalid Input Path.")
            return
        self.OpenPathInWindows(self.inputPath)
        self.UpdateRunlog("[UI] Open input path done.")

    def OpenOutputPath(self):
        if self.outputPath is None or self.outputPath == "":
            self.ShowWarningInfo("Invalid Output Path.")
            return
        self.OpenPathInWindows(self.outputPath)
        self.UpdateRunlog("[UI] Open output path done.")

    def ClearInputData(self):
        # 清空输入相关显示内容
        self.inputPath = None
        self.lineEdit_input_path.clear()
        self.lineEdit_input_path.setText("点击加载数据")
        self.inputDataImage = {}
        self.inputDataExcel = {}
        self.tableWidget_input_list.clear()
        self.tableWidget_input_list.setRowCount(0)
        self.tableWidget_input_list.setColumnCount(0)
        self.plot_qtChart.clearAllSeries()
        self.plot_qtChart.set_xrange(0, 100)
        self.plot_qtChart.set_yrange(0, 100)
        self.label_input_preview_image.clear()
        self.progressBar_feature_progress.setValue(0)
        self.UpdateRunlog("[UI] Clear input done.")

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
        self.featurePathExcel = self.MakeFolder("features_xlsx")
        self.featurePathImg = self.MakeFolder("features_img")
        self.UpdateRunlog("[UI][OutputPath] xlsx: {}, Image: {}.".format(
            self.featurePathExcel, self.featurePathImg))

    def SelectInputPath(self):
        self.UpdateRunlog("[UI] Select output path...")
        self.inputPath = QFileDialog.getExistingDirectory(self, "选择输入文件夹", os.getcwd())
        if self.outputPath is None or self.outputPath == "":
            self.ShowWarningInfo("Please select valid folder.")
            return
        self.lineEdit_input_path.setText(self.inputPath)
        self.UpdateRunlog("[UI] Selected input folder: {}".format(self.inputPath))
        # 启动线程进行数据加载
        if self.inputLoader is not None:
            if self.inputLoader.isRunning():
                self.ShowWarningInfo("Loading task is running...")
                return
        # 多次加载数据
        if self.inputLoader is not None:
            del self.inputLoader
        self.inputLoader = None
        self.inputLoader = DataInputOutput.InputDataLoader(self.inputPath)
        self.inputLoader.update_runlog_signal.connect(self.UpdateRunlog)
        self.inputLoader.update_input_signal.connect(self.UpdateLoadingProgress)
        # 按钮锁定
        self.pushButton_input_path.setText("加载中...")
        self.pushButton_input_path.setEnabled(False)
        self.inputLoader.start()
        print("[UI] Input data loader start done.")
        self.UpdateRunlog("[UI] Start input loading thread done.")

    # 将加载的excel和图像数据在表格中显示
    def RefreshInputTable(self):
        # 先清空列表
        self.tableWidget_input_list.clear()
        # 配置table widget属性
        rowNum = len(self.inputDataImage) + len(self.inputDataExcel)
        colNum = 1
        self.tableWidget_input_list.setRowCount(rowNum)
        self.tableWidget_input_list.setColumnCount(colNum)
        self.tableWidget_input_list.setColumnWidth(0, 100)
        # 设置表头
        self.tableWidget_input_list.setHorizontalHeaderLabels(["文件名"])
        # 动态调整行宽和列宽
        self.tableWidget_input_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.tableWidget_input_list.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 设置表格为只读模式
        self.tableWidget_input_list.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        count = 0
        # 先显示excel文件列表
        if self.inputDataExcel is not None:
            excel_list = list(self.inputDataExcel.keys())
            excel_list.sort()
            for fileName in excel_list:
                self.tableWidget_input_list.setItem(count, 0, QTableWidgetItem("{}".format(fileName)))
                count += 1
        # 显示image文件列表
        if self.inputDataImage is not None:
            image_list = list(self.inputDataImage.keys())
            image_list.sort()
            for fileName in image_list:
                self.tableWidget_input_list.setItem(count, 0, QTableWidgetItem("{}".format(fileName)))
                count += 1
        # 刷新
        self.tableWidget_input_list.show()
        print("[Debug] Fresh table done.")

    def UpdateLoadingProgress(self, value):
        print("[UI] [Signal] Receive input loading progress: {}".format(value))
        self.progressBar_feature_progress.setValue(value)
        if value == 100:
            # 获取结果并保存到原字典中
            success, excelData, imgData = self.inputLoader.GetLoadedResults()
            if success:
                excelCount = 0
                for excelName in excelData:
                    if excelName not in self.inputDataExcel:
                        self.inputDataExcel[excelName] = excelData[excelName]
                        excelCount += 1
                self.UpdateRunlog("[UI] {} excels added.".format(excelCount))
                imageCount = 0
                for imageName in imgData:
                    if imageName not in self.inputDataImage:
                        self.inputDataImage[imageName] = imgData[imageName]
                        imageCount += 1
                self.UpdateRunlog("[UI] {} images added.".format(imageCount))
                print("[Debug] excelCount: {}, imageCount: {}".format(excelCount, imageCount))
                # 刷新输入列表
                self.RefreshInputTable()
            # 按钮解锁
            self.pushButton_input_path.setEnabled(True)
            self.pushButton_input_path.setText("加载数据")
            # 默认设置当前为[0,0]，并进行显示
            self.tableWidget_input_list.setCurrentCell(0, 0)
            self.UpdateRunlog("[UI] Default: show table item[R{} C{}]".format(0, 0))
            self.InputPreview(0, 0)

    def PlotExcel(self, data, curveName):
        # 校验data
        if data is None:
            self.UpdateRunlog("[UI][Preview] Invalid Excel File.")
            return
        (m, n) = np.shape(data)
        if n < 2:
            self.UpdateRunlog("[UI][Preview] Invalid Excel File.")
            return
        # 获取x, y数据
        dataX0 = data[:round(m / 2), 0].tolist()
        dataX1 = data[round(m / 2):, 0].tolist()
        dataX = dataX0 + dataX1
        dataY0 = data[:round(m / 2), 1].tolist()
        dataY1 = data[round(m / 2):, 1].tolist()
        dataY = dataY0 + dataY1
        print("Get x y data done.")
        # 更新坐标轴
        minX = min(dataX)
        maxX = max(dataX)
        minY = min(dataY)
        maxY = max(dataY)
        # minX = minX - (maxX - minX) * 0.05
        # maxX = maxX + (maxX - minX) * 0.05
        minY = minY - (maxY - minY) * 0.05
        maxY = maxY + (maxY - minY) * 0.05
        self.plot_qtChart.set_xrange(minX, maxX)
        self.plot_qtChart.set_yrange(minY, maxY)
        print("Set range done.")
        # 获取curve序列并赋值
        sIdx0 = self.plot_qtChart.add_multiple_series("+" + curveName, QPen(Qt.red))
        sIdx1 = self.plot_qtChart.add_multiple_series("-" + curveName, QPen(Qt.blue))
        self.plot_qtChart.draw(dataX0, dataY0, sIdx1)
        self.plot_qtChart.draw(dataX1, dataY1, sIdx0)

    def DrawImage(self, data):
        #
        pass

    def CellChangedProcess(self, current_row, current_col, pre_row, pre_col):
        self.InputPreview(current_row, current_col)

    def ShowImageInLabel(self, imagePathName):
        pixmap = QPixmap(imagePathName)
        self.label_input_preview_image.setPixmap(pixmap)
        self.label_input_preview_image.setScaledContents(True)
        self.label_input_preview_image.show()

    def InputPreview(self, row, col):
        # 从表格中获取点击的单元格的值，如果没有点击则查看显示第一个单元格
        print("[UI] show table item[R{} C{}].".format(row, col))
        item = self.tableWidget_input_list.item(row, col)
        itemType = "xlsx"
        if item is not None:
            fileName = item.text()
            print("Clicked {}".format(fileName))
            itemType = fileName.split(".")[-1]
            itemName = fileName.split(".")[0]
            # 清空之前绘制的结果
            self.plot_qtChart.clearAllSeries()
            self.label_input_preview_image.clear()
            if itemType == "xlsx":
                # 用表格中的数据绘制曲线
                self.label_input_preview_image.setVisible(False)
                self.graphicsView_input_preview.setVisible(True)
                # 如果用频域，则显示图像
                if self.usingWavelet:
                    self.label_input_preview_image.setVisible(True)
                    self.graphicsView_input_preview.setVisible(False)
                # 获取数据
                if fileName in self.inputDataExcel:
                    data = self.inputDataExcel[fileName]
                    # 更新绘制曲线
                    if self.usingWavelet:
                        # 先变换到频域，然后显示频域图像
                        saveFigName = self.featurePathImg + "/{}.png".format(itemName)
                        self.UpdateRunlog("[Debug] Save image to {}.".format(saveFigName))
                        FeatureExtractSet.WaveletTransformSSWT(data, saveFigName)
                        self.UpdateRunlog("[Debug] Save image to {} Done.".format(saveFigName))
                        # 读取图像然后显示到QLabel中
                        self.ShowImageInLabel(saveFigName)
                    else:
                        self.PlotExcel(data, fileName)
                        self.graphicsView_input_preview.show()
                        self.UpdateRunlog("Plot curve {} done.".format(fileName))
            else:
                # 绘制图像到qlabel中
                self.label_input_preview_image.setVisible(True)
                self.graphicsView_input_preview.setVisible(False)
                # 获取数据
                if fileName in self.inputDataImage:
                    data = self.inputDataImage[fileName]
                    # 更新绘制图像
                    self.DrawImage(data)
                self.label_input_preview_image.show()

    def SSWTTransforming(self):
        # 进行SSWT变换获取所有图像
        index = 0
        total = len(self.inputDataExcel)
        excel_list = list(self.inputDataExcel.keys())
        for excel_file in excel_list:
            fileName = excel_file.split(".")[0]
            imageName = "{}.png".format(fileName)
            filePathName = self.featurePathImg + "/{}".format(imageName)
            print("[Feature] Start SSWT: {}.".format(filePathName))
            FeatureExtractSet.WaveletTransformSSWT(self.inputDataExcel[excel_file], filePathName)
            index += 1
            progress = int(index * 100 / total)
            self.progressBar_feature_progress.setValue(progress)
        print("[Feature] SSWT convert done.")

    def FeatureExtract(self):
        # 先获取提取特征的方法类型
        featureType = self.comboBox_feature_select.currentText()
        if self.extractor is not None:
            del self.extractor
            self.extractor = None
            print("[UI] Fresh feature extractor.")
        self.extractor = FeatureExtractor()
        self.extractor.SetFeatureExtractType(featureType)
        self.extractor.update_runlog_signal.connect(self.UpdateRunlog)
        self.extractor.update_feature_signal.connect(self.UpdateExtractingProgress)
        # 不同类型处理不同的类型输入数据，如离散小波处理的是excel
        inputData = list(self.inputDataExcel.values())
        self.extractor.SetBatchInputData(inputData)
        if featureType == "VAE":
            inputDataName = list(self.inputDataExcel.keys())
            self.extractor.SetTrainDataFileNames(inputDataName)
            self.SSWTTransforming()
            self.extractor.SetVAETrainDataPath(self.featurePathImg)
        else:
            self.UpdateRunlog("[UI] Feature type is invalid.")
        # 异步执行批量特征提取
        self.pushButton_feature_extract.setEnabled(False)
        self.extractor.start()
        self.UpdateRunlog("[UI] Start feature extracting thread done.")
        pass

    def UpdateExtractingProgress(self, value):
        print("[UI] [Signal] Feature extracting progress: {}".format(value))
        self.progressBar_feature_progress.setValue(value)
        if value == 100:
            success, results = self.extractor.GetExtractResults()
            if not success:
                print("Get result failed.")
                self.pushButton_feature_extract.setEnabled(True)
                return
            # 将结果保存到特定目录中
            print("Start to export data.")
            feature_type = self.extractor.GetFeatureType()
            if feature_type == "WST":
                ret = DataInputOutput.ExportDataToExcel(self.featurePathExcel, "DSC", results)
                if not ret:
                    self.UpdateRunlog("[UI] Export results to {} failed.".format(self.featurePathExcel))
            elif feature_type == "VAE":
                # VAE feature
                print("Results len: {}".format(len(results)))
                ret = DataInputOutput.ExportDataToExcel(self.featurePathExcel, "feature_vae_5D", results)
                if not ret:
                    self.UpdateRunlog("[UI] Export results to {} failed.".format(self.featurePathExcel))
            self.pushButton_feature_extract.setEnabled(True)
            print("Feature Extract done.")
