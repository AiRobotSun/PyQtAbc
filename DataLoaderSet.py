import os
import glob
import pandas as pd
import torch


class DataLoaderSet:
    def __init__(self):
        return

    def LoadExcelData(self, data_path_name):
        print("[DataLoaderSet] Load excel file: {}".format(data_path_name))
        df = pd.read_excel(data_path_name, sheet_name=0)
        data = df.values
        return data

    def LoadCsvData(self, data_path_name):
        return None

    def LoadExcelsFromPath(self, data_path):
        excel_list = []
        # 遍历文件目录找到后缀.xlsx文件
        find_desp = "{}/*.xlsx"
        excels = glob.glob(find_desp, recursive=True)
        for excel in excels:
            data = self.LoadExcelData(excel)
            excel_list.append(data)
        return excel_list

    def LoadCsvsFromPath(self, data_path):
        csv_list = []
        return csv_list

    def LoadImagesFromPath(self, data_path):
        image_list = []
        return image_list

    def InitModel(self, model_type):
        model = None
        if model_type == "NN":
            N, D_in, H1, H2, D_out = 10, 7, 32, 16, 1
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                torch.nn.Linear(H1, H2),
                torch.nn.ReLU(),
                torch.nn.Linear(H2, D_out),
            )

        return model

    def LoadModel(self, model_type, model_path):
        print("[DataLoaderSet] Load model[{}] file: {}".format(model_type, model_path))
        model = self.InitModel(model_type)
        model.load_state_dict(torch.load(model_path))
        return model
