import os
import torch
from tqdm import tqdm
# Importa as classes originais do seu DataLoader.py
from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer

def construir_cache_dataset():
    print("novo gerador de cache")
    CAMINHO_CSV = "./Datasets/STARCOP_train/train.csv"
    DIRETORIO_DADOS = "./Datasets/STARCOP_train"
    DIRETORIO_CACHE = "./Datasets/STARCOP_cache_128" # Nova pasta onde os tensores ficarão
    
    PRODUTOS_ENTRADA = [
        "TOA_AVIRIS_460nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_640nm",
        "TOA_AVIRIS_2004nm", "TOA_AVIRIS_2109nm", "TOA_AVIRIS_2310nm", 
        "TOA_AVIRIS_2350nm", "TOA_AVIRIS_2360nm", "mag1c"
    ]
    PRODUTO_SAIDA = ["labelbinary"]
    
    os.makedirs(DIRETORIO_CACHE, exist_ok=True)
    
    # Carrega o dataset original (lento, baseado em GeoTIFF)
    df_train = carregar_dataframe_starcop(CAMINHO_CSV, DIRETORIO_DADOS)
    dataset = STARCOPDataset(df_train, PRODUTOS_ENTRADA, PRODUTO_SAIDA, weight_loss="weight_mag1c")
    normalizador = DataNormalizer(PRODUTOS_ENTRADA) 
    
    stride = 64 # Sobreposição
    window_size = 128 # Tamanho da janela
    patch_id = 0
    
    for idx in tqdm(range(len(dataset)), desc="Gerando Cache 128x128"):
        batch = dataset[idx] 
        inputs = batch["input"]
        outputs = batch["output"]
        pesos_loss = batch["weight_loss"]
        
        _, h, w = inputs.shape
        
        # Extrai os recortes e salva no disco
        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                caminho_salvamento = os.path.join(DIRETORIO_CACHE, f"patch_{patch_id}.pt")
                
                if os.path.exists(caminho_salvamento):
                    patch_id += 1
                    continue

                patch_input = inputs[:, y:y+window_size, x:x+window_size]
                patch_output = outputs[:, y:y+window_size, x:x+window_size]
                patch_peso = pesos_loss[:, y:y+window_size, x:x+window_size]
                
                dados_patch = {
                    "input": normalizador(patch_input.unsqueeze(0)).squeeze(0),
                    "output": patch_output,
                    "weight_loss": patch_peso
                }
                
                torch.save(dados_patch, caminho_salvamento)
                patch_id += 1

if __name__ == "__main__":
    construir_cache_dataset()