import glob
import os.path

import numpy
import pandas as pd

from BaseThread import BaseThread


class InputDataLoader(BaseThread):
    def __init__(self, path):
        super().__init__()
        self.success = True
        self.inputPath = path
        # 文件数据以字典的形式存储key: 文件名， value: 加载的数据
        self.inputDataExl = {}
        self.inputDataImg = {}

    def GetLoadedResults(self):
        return self.success, self.inputDataExl, self.inputDataImg

    def LoadExcelData(self, data_path_name):
        print("[DataLoaderSet] Load excel file: {}".format(data_path_name))
        df = pd.read_excel(data_path_name, sheet_name=0)
        data = df.values
        return data

    def LoadInputExcelData(self):
        if not os.path.exists(self.inputPath):
            self.success = False
            return False
        # 遍历输入目录获取所有的xlsx文件
        target_pn = os.path.join(self.inputPath, "*.xlsx")
        excelList = glob.glob(target_pn, recursive=False)  # 不进行recursive
        total = len(excelList)
        counter = 1
        for filePN in excelList:
            # 获取文件名作为Key,包括扩展名
            fileName = filePN.split("\\")[-1]
            data = self.LoadExcelData(filePN)
            if data is not None:
                if fileName not in self.inputDataExl:
                    self.inputDataExl[fileName] = data
            progress = int(counter * 100 / total)
            counter += 1
            super().update_input_signal.emit(progress)
        print("[Debug] Loaded {} excel files success.".format(len(self.inputDataExl)))
        return True

    def LoadInputImage(self, imagePN):
        data = []
        return data

    def LoadInputImageData(self):
        if not os.path.exists(self.inputPath):
            self.success = False
            return False
        # 遍历输入目录获取所有的png，jpg文件
        target_pn_png = os.path.join(self.inputPath, "*.png")
        pngList = glob.glob(target_pn_png, recursive=False)  # 不进行recursive
        target_pn_jpg = os.path.join(self.inputPath, "*.jpg")
        jpgList = glob.glob(target_pn_jpg, recursive=False)  # 不进行recursive
        total = len(pngList) + len(jpgList)
        counter = 1
        for filePN in pngList:
            # 获取文件名作为Key,包括扩展名
            fileName = filePN.split("\\")[-1]
            data = self.LoadInputImage(filePN)
            if data is not None:
                if fileName not in self.inputDataImg:
                    self.inputDataImg[fileName] = data
            progress = int(counter * 100 / total)
            counter += 1
            super().update_input_signal.emit(progress)

        for filePN in jpgList:
            # 获取文件名作为Key,包括扩展名
            fileName = filePN.split("\\")[-1]
            data = self.LoadInputImage(filePN)
            if data is not None:
                if fileName not in self.inputDataImg:
                    self.inputDataImg[fileName] = data
            progress = int(counter * 100 / total)
            counter += 1
            super().update_input_signal.emit(progress)
        print("[Debug] {} images have been loaded.".format(len(self.inputDataImg)))
        return True

    def run(self) -> None:
        if not self.LoadInputExcelData():
            super().SetRunLog("[LoadInput] Load excel data failed.")
            super().update_input_signal.emit(100)
            return
        if not self.LoadInputImageData():
            super().SetRunLog("[LoadInput] Load image data failed.")
            super().update_input_signal.emit(100)
            return
        self.success = True


def ExportDataToExcel(outputPath, fileName, dataList):
    if len(dataList) == 0:
        return False
    if outputPath is None:
        outputPath = "./"
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    if fileName is None:
        fileName = "Default"
    savePathName = outputPath + "/{}.xlsx".format(fileName)
    df = pd.DataFrame(dataList)
    df.to_excel(savePathName, index=False, header=False)
    print("[Output] Export data done: {}.".format(savePathName))
    return True
