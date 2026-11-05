import os
import torch
import pandas as pd
import numpy as np
import rasterio
import rasterio.windows
from torch.utils.data import Dataset
import warnings

def carregar_dataframe_starcop(caminho_csv, diretorio_imagens):
    df = pd.read_csv(caminho_csv)
    df["window"] = df.apply(
        lambda row: rasterio.windows.Window(
            col_off=row.window_col_off, 
            row_off=row.window_row_off,
            width=row.window_width, 
            height=row.window_height
        ), axis=1
    )
    if "folder" in df.columns:
        df["folder"] = df["folder"].apply(
            lambda x: os.path.join(diretorio_imagens, os.path.basename(os.path.normpath(x)))
        )
    elif "id" in df.columns:
        df["folder"] = df["id"].apply(lambda x: os.path.join(diretorio_imagens, str(x)))

    linhas_validas = []
    for idx, row in df.iterrows():
        caminho_teste = os.path.join(row['folder'], "mag1c.tif")
        linhas_validas.append(os.path.exists(caminho_teste))
            
    df = df[linhas_validas].reset_index(drop=True)
    return df

class STARCOPDataset(Dataset):
    def __init__(self, dataframe, input_products, output_products, weight_loss=None, modo_treino=True):
        self.dataframe = dataframe
        self.input_products = input_products
        self.output_products = output_products
        self.weight_loss = weight_loss
        self.modo_treino = modo_treino

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        data_iter = self.dataframe.iloc[idx]
        product_folder = data_iter.folder
        window = data_iter.window

        out_dict = {}
        names_outputs = ["input", "output"]
        output_products = [self.input_products, self.output_products]
        
        if self.weight_loss is not None:
            names_outputs.append("weight_loss")
            output_products.append([self.weight_loss])

        for io_name, products in zip(names_outputs, output_products):
            tensors = []
            for key_name in products:
                path = os.path.join(product_folder, f"{key_name}.tif")
                with rasterio.open(path) as src:
                    tensors.append(torch.from_numpy(src.read(window=window)))
            
            if len(tensors) > 1:
                tensor_concat = torch.cat(tensors, dim=0).float()
            elif len(tensors) == 1:
                tensor_concat = tensors[0].float()

            if self.modo_treino:
                _, h, w = tensor_concat.shape
                patches = []
                for y in range(0, h - 128 + 1, 64):
                    for x in range(0, w - 128 + 1, 64):
                        patches.append(tensor_concat[:, y:y+128, x:x+128])
                
                out_dict[io_name] = torch.stack(patches)
            else:
                out_dict[io_name] = tensor_concat

        return out_dict


BAND_NORMALIZATION = {
 'TOA_AVIRIS_550nm': {'offset': 0, 'factor': 60, 'clip': (0, 2)},
 'TOA_AVIRIS_640nm': {'offset': 0, 'factor': 60, 'clip': (0, 2)},
 'TOA_AVIRIS_460nm': {'offset': 0, 'factor': 60, 'clip': (0, 2)},
 'mag1c': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},
}

class DataNormalizer(torch.nn.Module):
    def __init__(self, input_products):
        super().__init__()
        offsets, factors, clip_mins, clip_maxs = [], [], [], []
        
        for p in input_products:
            if p not in BAND_NORMALIZATION:
                warnings.warn(f"Sem normalização para {p}. Usando default.")
                offsets.append(0); factors.append(1)
                clip_mins.append(-10); clip_maxs.append(10)
            else:
                cfg = BAND_NORMALIZATION[p]
                offsets.append(cfg['offset']); factors.append(cfg['factor'])
                clip_mins.append(cfg['clip'][0]); clip_maxs.append(cfg['clip'][1])

        self.offsets = torch.nn.Parameter(torch.tensor(offsets).view(-1, 1, 1).float(), requires_grad=False)
        self.factors = torch.nn.Parameter(torch.tensor(factors).view(-1, 1, 1).float(), requires_grad=False)
        self.clip_min = torch.nn.Parameter(torch.tensor(clip_mins).view(-1, 1, 1).float(), requires_grad=False)
        self.clip_max = torch.nn.Parameter(torch.tensor(clip_maxs).view(-1, 1, 1).float(), requires_grad=False)

    def forward(self, x):
        return torch.clamp((x - self.offsets) / self.factors, self.clip_min, self.clip_max)

