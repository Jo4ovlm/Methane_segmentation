import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from kornia.morphology import erosion, dilation
import kornia.augmentation as K

from Utils.DataLoader import carregar_dataframe_starcop, STARCOPDataset, DataNormalizer

def binary_opening(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    eroded = torch.clamp(erosion(x.float(), kernel), 0, 1) > 0
    return torch.clamp(dilation(eroded.float(), kernel), 0, 1) > 0

def calcular_f1_score(previsao_logits, gabarito, threshold=0.0):
    device = previsao_logits.device
    previsao_binaria = (previsao_logits > threshold).float()
    
    kernel_cruz = torch.tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]).float().to(device)
    
    previsao_limpa = binary_opening(previsao_binaria, kernel_cruz).float()
    
    intersecao = (previsao_limpa * gabarito).sum(dim=(2, 3))
    soma_areas = previsao_limpa.sum(dim=(2, 3)) + gabarito.sum(dim=(2, 3))
    f1 = (2 * intersecao + 1e-6) / (soma_areas + 1e-6)
    
    # Retorna soma absoluta e contagem para cálculo fora do loop de validação
    return f1.sum().item(), f1.numel() 

def treinar_modelo(modelo_escolhido, nome_modelo_salvar, starting_point, produtos_entrada):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Iniciando Treinamento: {nome_modelo_salvar} ---")

    CAMINHO_CSV = "Datasets/STARCOP_train/train.csv"
    DIRETORIO_DADOS = "Datasets/STARCOP_train"
    
    df_train = carregar_dataframe_starcop(CAMINHO_CSV, DIRETORIO_DADOS)
    
    # Dataset lendo GeoTIFFs grandes e cortando na RAM
    dataset_treino = STARCOPDataset(df_train, produtos_entrada, ["labelbinary"], weight_loss="weight_mag1c")
    normalizador = DataNormalizer(produtos_entrada).to(device)
    
    dataloader = DataLoader(
        dataset_treino, 
        batch_size=4,   
        shuffle=True,   
        num_workers=4,  
        pin_memory=True
    )

    modelo = modelo_escolhido(in_channels=len(produtos_entrada), out_channels=1).to(device)
    
    caminho_salvamento = f"Modelos_treinados/{nome_modelo_salvar}.pth"
    os.makedirs("Modelos_treinados", exist_ok=True)
    
    if os.path.exists(caminho_salvamento):
        print("Carregando pesos...")
        modelo.load_state_dict(torch.load(caminho_salvamento, map_location=device, weights_only=True))
    
    optimizer = optim.Adam(modelo.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    scaler = torch.amp.GradScaler('cuda') 

    # --- PIPELINE DE DATA AUGMENTATION (GPU) ---
    # Aplica a mesma transformação na imagem, na máscara e nos pesos
    augmentacoes = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90.0, p=0.5),
        data_keys=["input", "mask", "mask"], 
        same_on_batch=False
    )

    epocas = 200
    paciencia_maxima = 10 
    melhor_f1 = starting_point
    paciencia_atual = 0

    for epoca in range(epocas):
        modelo.train()
        loss_acumulada = 0.0
        
        loop_treino = tqdm(dataloader, desc=f"Época {epoca+1}/{epocas} [Treino]")
        for batch in loop_treino:
            # 1. Achata as 49 imagens de cada batch (ex: 4 * 49 = 196)
            b, p, c, h_dim, w_dim = batch["input"].shape
            
            inputs = batch["input"].view(b * p, c, h_dim, w_dim).to(device, non_blocking=True)
            targets = batch["output"].view(b * p, 1, h_dim, w_dim).to(device, non_blocking=True)
            pesos_loss = batch["weight_loss"].view(b * p, 1, h_dim, w_dim).to(device, non_blocking=True)

            # 2. Embaralha os recortes diretamente na GPU (quebra a sequência de pixels adjacentes)
            indices_shuffled = torch.randperm(b * p, device=device)
            inputs = inputs[indices_shuffled]
            targets = targets[indices_shuffled]
            pesos_loss = pesos_loss[indices_shuffled]

            # 3. Data Augmentation
            inputs, targets, pesos_loss = augmentacoes(inputs, targets, pesos_loss)

            # 4. Normalização e Forward/Backward Pass (AMP)
            inputs = normalizador(inputs)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                previsoes = modelo(inputs)
                loss = (criterion(previsoes, targets) * pesos_loss).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_acumulada += loss.item()
            loop_treino.set_postfix(Loss=f"{loss.item():.4f}")

        media_loss = loss_acumulada / len(dataloader)
        
        # --- VALIDAÇÃO (Sem Augmentation e com F1-Score) ---
        modelo.eval()
        f1_total = 0.0
        num_amostras = 0
        lotes_avaliacao = 100 # Avalia em uma amostra de lotes para não onerar o tempo total
        
        with torch.no_grad():
            loop_val = tqdm(dataloader, total=lotes_avaliacao, desc=f"Época {epoca+1}/{epocas} [Validação]")
            for i, batch in enumerate(loop_val):
                if i >= lotes_avaliacao:
                    break
                    
                b, p, c, h_dim, w_dim = batch["input"].shape
                inputs = batch["input"].view(b * p, c, h_dim, w_dim).to(device, non_blocking=True)
                targets = batch["output"].view(b * p, 1, h_dim, w_dim).to(device, non_blocking=True)
                
                inputs = normalizador(inputs)
                
                with torch.amp.autocast('cuda'):
                    previsoes = modelo(inputs)
                    
                soma_f1_batch, qtd_amostras = calcular_f1_score(previsoes, targets)
                f1_total += soma_f1_batch
                num_amostras += qtd_amostras
                
        media_f1 = f1_total / num_amostras
        print(f"Resumo da Época {epoca+1} -> Loss Média: {media_loss:.4f} | F1-Score (Amostra): {media_f1:.4f}")
        
        if media_f1 > melhor_f1:
            melhor_f1 = media_f1
            paciencia_atual = 0
            torch.save(modelo.state_dict(), caminho_salvamento)
            print(f" -> Novo recorde F1: {melhor_f1:.4f}. Modelo salvo!")
        else:
            paciencia_atual += 1
            if paciencia_atual >= paciencia_maxima:
                print(f"=== Early Stopping! Melhor F1-Score: {melhor_f1:.4f} ===")
                break