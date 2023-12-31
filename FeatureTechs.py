# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FeatureTechs.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FeatureEngineering(object):
    def setupUi(self, FeatureEngineering):
        FeatureEngineering.setObjectName("FeatureEngineering")
        FeatureEngineering.resize(1500, 893)
        FeatureEngineering.setMinimumSize(QtCore.QSize(1500, 893))
        FeatureEngineering.setMaximumSize(QtCore.QSize(1500, 893))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        FeatureEngineering.setFont(font)
        FeatureEngineering.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(FeatureEngineering)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_output_path = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_output_path.setGeometry(QtCore.QRect(360, 150, 91, 31))
        self.pushButton_output_path.setObjectName("pushButton_output_path")
        self.lineEdit_output_path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_output_path.setGeometry(QtCore.QRect(130, 150, 221, 31))
        self.lineEdit_output_path.setReadOnly(True)
        self.lineEdit_output_path.setObjectName("lineEdit_output_path")
        self.label_output_path = QtWidgets.QLabel(self.centralwidget)
        self.label_output_path.setGeometry(QtCore.QRect(30, 150, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_output_path.setFont(font)
        self.label_output_path.setTextFormat(QtCore.Qt.AutoText)
        self.label_output_path.setObjectName("label_output_path")
        self.pushButton_output_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_output_open.setGeometry(QtCore.QRect(470, 150, 131, 31))
        self.pushButton_output_open.setObjectName("pushButton_output_open")
        self.label_runlog = QtWidgets.QLabel(self.centralwidget)
        self.label_runlog.setGeometry(QtCore.QRect(570, 740, 41, 61))
        self.label_runlog.setWordWrap(True)
        self.label_runlog.setObjectName("label_runlog")
        self.label_studio = QtWidgets.QLabel(self.centralwidget)
        self.label_studio.setGeometry(QtCore.QRect(420, 60, 181, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.label_studio.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_studio.setFont(font)
        self.label_studio.setObjectName("label_studio")
        self.label_studio_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_studio_2.setGeometry(QtCore.QRect(430, 20, 141, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.PlaceholderText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 128))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.PlaceholderText, brush)
        self.label_studio_2.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_studio_2.setFont(font)
        self.label_studio_2.setObjectName("label_studio_2")
        self.label_logo = QtWidgets.QLabel(self.centralwidget)
        self.label_logo.setGeometry(QtCore.QRect(30, 10, 350, 100))
        self.label_logo.setStyleSheet("background-color:rgb(17,64,108)")
        self.label_logo.setText("")
        self.label_logo.setPixmap(QtGui.QPixmap("logo.png"))
        self.label_logo.setObjectName("label_logo")
        self.label_soft_title = QtWidgets.QLabel(self.centralwidget)
        self.label_soft_title.setGeometry(QtCore.QRect(760, 20, 611, 81))
        font = QtGui.QFont()
        font.setFamily("方正粗黑宋简体")
        font.setPointSize(28)
        self.label_soft_title.setFont(font)
        self.label_soft_title.setObjectName("label_soft_title")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(370, 10, 261, 100))
        self.label_2.setStyleSheet("background-color:rgb(17,64,108)")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.textBrowser_runlog = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser_runlog.setGeometry(QtCore.QRect(620, 700, 861, 151))
        self.textBrowser_runlog.setObjectName("textBrowser_runlog")
        self.pushButton_input_path = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_input_path.setGeometry(QtCore.QRect(360, 210, 91, 31))
        self.pushButton_input_path.setObjectName("pushButton_input_path")
        self.label_input_path = QtWidgets.QLabel(self.centralwidget)
        self.label_input_path.setGeometry(QtCore.QRect(30, 210, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_input_path.setFont(font)
        self.label_input_path.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_path.setObjectName("label_input_path")
        self.lineEdit_input_path = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_input_path.setGeometry(QtCore.QRect(130, 210, 221, 31))
        self.lineEdit_input_path.setReadOnly(True)
        self.lineEdit_input_path.setObjectName("lineEdit_input_path")
        self.graphicsView_input_preview = QChartView(self.centralwidget)
        self.graphicsView_input_preview.setGeometry(QtCore.QRect(620, 150, 861, 531))
        self.graphicsView_input_preview.setObjectName("graphicsView_input_preview")
        self.pushButton_input_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_input_open.setGeometry(QtCore.QRect(470, 210, 131, 31))
        self.pushButton_input_open.setObjectName("pushButton_input_open")
        self.tableWidget_input_list = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget_input_list.setGeometry(QtCore.QRect(40, 280, 431, 391))
        self.tableWidget_input_list.setEditTriggers(QtWidgets.QAbstractItemView.AnyKeyPressed|QtWidgets.QAbstractItemView.CurrentChanged|QtWidgets.QAbstractItemView.DoubleClicked|QtWidgets.QAbstractItemView.EditKeyPressed)
        self.tableWidget_input_list.setObjectName("tableWidget_input_list")
        self.tableWidget_input_list.setColumnCount(0)
        self.tableWidget_input_list.setRowCount(0)
        self.progressBar_feature_progress = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_feature_progress.setGeometry(QtCore.QRect(40, 800, 511, 21))
        self.progressBar_feature_progress.setProperty("value", 0)
        self.progressBar_feature_progress.setObjectName("progressBar_feature_progress")
        self.pushButton_feature_extract = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_feature_extract.setGeometry(QtCore.QRect(270, 690, 201, 71))
        self.pushButton_feature_extract.setObjectName("pushButton_feature_extract")
        self.comboBox_feature_select = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_feature_select.setGeometry(QtCore.QRect(60, 720, 171, 31))
        self.comboBox_feature_select.setObjectName("comboBox_feature_select")
        self.label_feature_type = QtWidgets.QLabel(self.centralwidget)
        self.label_feature_type.setGeometry(QtCore.QRect(60, 690, 91, 31))
        self.label_feature_type.setObjectName("label_feature_type")
        self.label_input_list = QtWidgets.QLabel(self.centralwidget)
        self.label_input_list.setGeometry(QtCore.QRect(40, 250, 131, 31))
        self.label_input_list.setObjectName("label_input_list")
        self.label_input_preview_image = QtWidgets.QLabel(self.centralwidget)
        self.label_input_preview_image.setGeometry(QtCore.QRect(740, 150, 651, 501))
        self.label_input_preview_image.setText("")
        self.label_input_preview_image.setObjectName("label_input_preview_image")
        self.pushButton_input_clear = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_input_clear.setGeometry(QtCore.QRect(480, 380, 121, 51))
        self.pushButton_input_clear.setObjectName("pushButton_input_clear")
        self.checkBox_wavelet = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_wavelet.setGeometry(QtCore.QRect(490, 310, 121, 31))
        self.checkBox_wavelet.setObjectName("checkBox_wavelet")
        self.label_logo.raise_()
        self.label_2.raise_()
        self.label_runlog.raise_()
        self.label_studio.raise_()
        self.label_studio_2.raise_()
        self.label_soft_title.raise_()
        self.pushButton_output_open.raise_()
        self.lineEdit_output_path.raise_()
        self.label_output_path.raise_()
        self.pushButton_output_path.raise_()
        self.textBrowser_runlog.raise_()
        self.pushButton_input_path.raise_()
        self.label_input_path.raise_()
        self.lineEdit_input_path.raise_()
        self.graphicsView_input_preview.raise_()
        self.pushButton_input_open.raise_()
        self.tableWidget_input_list.raise_()
        self.progressBar_feature_progress.raise_()
        self.pushButton_feature_extract.raise_()
        self.comboBox_feature_select.raise_()
        self.label_feature_type.raise_()
        self.label_input_list.raise_()
        self.label_input_preview_image.raise_()
        self.pushButton_input_clear.raise_()
        self.checkBox_wavelet.raise_()
        FeatureEngineering.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(FeatureEngineering)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1500, 29))
        self.menubar.setObjectName("menubar")
        FeatureEngineering.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(FeatureEngineering)
        self.statusbar.setObjectName("statusbar")
        FeatureEngineering.setStatusBar(self.statusbar)

        self.retranslateUi(FeatureEngineering)
        QtCore.QMetaObject.connectSlotsByName(FeatureEngineering)

    def retranslateUi(self, FeatureEngineering):
        _translate = QtCore.QCoreApplication.translate
        FeatureEngineering.setWindowTitle(_translate("FeatureEngineering", "Materials Informatics Lab"))
        self.pushButton_output_path.setText(_translate("FeatureEngineering", "选择目录"))
        self.lineEdit_output_path.setToolTip(_translate("FeatureEngineering", "该目录为所有运行中间结果及最终结果和日志的存放目录，目录下会自动生成操作对应的文件和文件夹"))
        self.lineEdit_output_path.setText(_translate("FeatureEngineering", "选择输出目录"))
        self.label_output_path.setText(_translate("FeatureEngineering", "输出目录"))
        self.pushButton_output_open.setText(_translate("FeatureEngineering", "打开输出目录"))
        self.label_runlog.setText(_translate("FeatureEngineering", "运行日志"))
        self.label_studio.setText(_translate("FeatureEngineering", "国家重点实验室"))
        self.label_studio_2.setText(_translate("FeatureEngineering", "金属材料强度"))
        self.label_soft_title.setText(_translate("FeatureEngineering", "Materials Informatics Lab"))
        self.pushButton_input_path.setToolTip(_translate("FeatureEngineering", "点击选择目录并加载数据"))
        self.pushButton_input_path.setText(_translate("FeatureEngineering", "加载数据"))
        self.label_input_path.setText(_translate("FeatureEngineering", "输入目录"))
        self.lineEdit_input_path.setToolTip(_translate("FeatureEngineering", "该目录为所有运行中间结果及最终结果和日志的存放目录，目录下会自动生成操作对应的文件和文件夹"))
        self.lineEdit_input_path.setText(_translate("FeatureEngineering", "点击加载数据"))
        self.pushButton_input_open.setText(_translate("FeatureEngineering", "打开输入目录"))
        self.pushButton_feature_extract.setText(_translate("FeatureEngineering", "开始提取"))
        self.label_feature_type.setText(_translate("FeatureEngineering", "特征类型"))
        self.label_input_list.setText(_translate("FeatureEngineering", "输入文件列表"))
        self.pushButton_input_clear.setToolTip(_translate("FeatureEngineering", "清空所有输入数据"))
        self.pushButton_input_clear.setText(_translate("FeatureEngineering", "清空输入"))
        self.checkBox_wavelet.setText(_translate("FeatureEngineering", "小波频域"))
from PyQt5.QtChart import QChartView
