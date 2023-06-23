# 小波散射变换特征提取
import glob
import os.path

from PIL import Image
import numpy as np
from kymatio.numpy import Scattering1D, Scattering2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft
from BaseThread import BaseThread
from VaeResnet import VaeModel


def PlotAndSaveFig(Wx, figName):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='jet')
    # plt.show(block=False)
    plt.xticks([])  # 去x坐标刻度
    plt.yticks([])  # 去y坐标刻度
    plt.axis('off')  # 去坐标轴
    plt.savefig(figName, bbox_inches='tight', pad_inches=0.0)
    plt.close()


# 小波变换到频域，返回频域图像数据
def WaveletTransformSSWT(data, figName):
    # STFT + SSQ STFT #
    Tsxo, Sxo, *_ = ssq_stft(data[:, 1])
    PlotAndSaveFig(np.flipud(Sxo), figName)


class Extractor:
    def __init__(self):
        pass

    def get_feature(self, data):
        pass


class ScatterTransform(Extractor):
    def __init__(self):
        super().__init__()
        self.J = 2  # 小波分解层数
        self.Q = 3  # 滤波器组数

    def set_feature_params(self, J, Q):
        self.J = J
        self.Q = Q

    # 定义函数进行不变散射小波变换
    def scattering_transform(self, data):
        scattering = Scattering1D(shape=data.shape, J=self.J, Q=self.Q)  # numpy
        # 进行不变散射小波变换
        coefficients = scattering(data)  # numpy
        return coefficients

    def get_feature(self, data):
        scattering_coefficients = self.scattering_transform(data[:, 1])
        # coefficients为从DSC数据中提取的特征
        coefficients = np.mean(scattering_coefficients, axis=1)
        return coefficients


class FeatureExtractVAE(Extractor):
    def __init__(self):
        super().__init__()
        self.output_path = None
        self.progress_signal = None
        self.train_data_path = None
        self.feature_list = None
        self.vae_model = VaeModel()

    def get_feature(self, data):
        pass

    def set_output_path(self, output_path):
        if output_path is not None:
            self.output_path = output_path
            self.vae_model.set_output_model_path(output_path)
            print("[VAE] Set vae model save path to {}.".format(output_path))

    def set_train_data_path(self, inputPath):
        self.train_data_path = inputPath

    def set_train_dataloader_path(self, dataloader_path):
        if dataloader_path is not None:
            print("[VAE] Dataloader input data path is: {}.".format(dataloader_path))
            self.vae_model.set_train_input(dataloader_path)

    def set_progress_observer(self, signalObj):
        self.progress_signal = signalObj
        self.vae_model.set_progress_updater(signalObj, 10)

    def resize_input_image(self):
        # 遍历输入目录然后resize图像到特定目录
        if self.train_data_path is None:
            print("Invalid input image path.")
            return False
        # 创建Resize之后的存放目录，即训练的输入目录
        # 创建train目录
        trainPath = os.path.join(self.train_data_path, "train")
        if not os.path.exists(trainPath):
            os.mkdir(trainPath)
        trainSetPath = os.path.join(trainPath, "0")
        if not os.path.exists(trainSetPath):
            os.mkdir(trainSetPath)
        testSetPath = os.path.join(trainPath, "1")
        if not os.path.exists(testSetPath):
            os.mkdir(testSetPath)
        imageList = glob.glob(self.train_data_path + "/*.png", recursive=False)
        total = len(imageList)
        counter = 0
        for imagePN in imageList:
            image = Image.open(imagePN)
            image_resized = image.resize((224, 224))
            imageName = imagePN.split("\\")[-1]
            imageSavePN = os.path.join(trainSetPath, imageName)
            print("Save resized image to {}.".format(imageSavePN))
            image_resized.save(imageSavePN)
            counter += 1
            # 固定resize之后进度完成25%
            progress = int(counter * 15 / total)
            self.progress_signal.emit(progress)
        print("Images resized done.")

    def train_and_extract(self):
        # VAE train
        self.vae_model.train()
        # 获取特征
        feature_tensor = self.vae_model.get_train_data_features()
        # print("[VAE] Feature length: {}".format(feature_tensor.size()))
        # tensor转list
        feature_list = feature_tensor.tolist()
        print("[VAE] Tensor to list length: {}".format(len(feature_list)))
        return feature_list


# 异步提取特征
class FeatureExtractor(BaseThread):
    def __init__(self):
        super().__init__()
        self.inputDataPath = None
        self.batchDataNames = None
        self.batchData = None
        self.data = None
        self.featureType = None
        self.extractor = None
        self.batchResults = []
        self.success = True
        self.SetRunLog("[Feature] Init extractor done.")

    def SetFeatureExtractType(self, featureType):
        self.featureType = featureType
        if self.extractor is not None:
            del self.extractor
        self.extractor = None
        if featureType == "WST":
            self.extractor = ScatterTransform()
        elif featureType == "VAE":
            self.extractor = FeatureExtractVAE()
            print("[Feature] Model download done.")
            self.extractor.set_progress_observer(super().update_feature_signal)
        else:
            self.SetRunLog("[Feature] Invalid feature extractor.")
        print("[Feature] Set feature type to: {}".format(self.featureType))

    def GetFeatureType(self):
        return self.featureType

    # batchData为data的list
    def SetBatchInputData(self, batchData):
        self.batchData = batchData
        self.SetRunLog("[Feature] Set input data done.")

    def SetTrainDataFileNames(self, dataNames):
        self.batchDataNames = dataNames
        self.SetRunLog("[Feature] Set input data name done.")
        print("[Feature] Set input data name done.")

    def SetVAETrainDataPath(self, dataPath):
        if self.extractor is not None:
            self.inputDataPath = dataPath
            self.extractor.set_train_data_path(dataPath)
            self.extractor.set_output_path(dataPath)
            print("Set input data path to: {}.".format(dataPath))

    def GetExtractResults(self):
        print("Get result enter.")
        return self.success, self.batchResults

    def run(self):
        self.batchResults.clear()
        self.success = False
        if self.batchData is None:
            self.SetRunLog("[Feature] Invalid input data")
            return
        if self.featureType == "VAE":
            # 先进行图像大小归一化成224*224
            self.extractor.resize_input_image()
            # 加载数据
            print(self.inputDataPath)
            dataloaderPath = self.inputDataPath + "/train/"
            self.extractor.set_train_dataloader_path(dataloaderPath)
            # 训练VAE模型
            self.batchResults = self.extractor.train_and_extract()
        else:
            total = len(self.batchData)
            counter = 0
            for data in self.batchData:
                result = self.extractor.get_feature(data)
                if result is not None:
                    self.batchResults.append(result)
                counter += 1
                progress = int(counter * 100 / total)
                if progress < 100:
                    super().update_feature_signal.emit(progress)
        self.success = True
        print("[Feature] Extracting Done, vector length: {}".format(self.batchResults))
        super().update_feature_signal.emit(100)
