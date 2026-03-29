import torch
import numpy as np




def calculate_metrics(pred, target):
    """计算 PCC, COD (R2) 和 RMSE"""
    # 处理 NaN
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        # 保持返回三个值: PCC, COD, RMSE
        return 0.0, 0.0, 0.0

    # 移除 NaN 对应的预测值和真实值
    pred = pred[mask]
    target = target[mask]
    n = len(pred)

    if n < 2:
        return 0.0, 0.0, 0.0

    # --- 1. PCC (皮尔逊相关系数) ---
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    # 分母加 1e-8 防止除以 0
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)

    # --- 2. COD (决定系数 R2) 和 SSR ---
    # Sum of Squares of Residuals (SSR): Sum_i (y_i - y_hat_i)^2
    residuals = target - pred
    SSR = torch.sum(residuals ** 2)

    # Total Sum of Squares (SST): Sum_i (y_i - mean(y))^2
    mean_target = torch.mean(target)
    SST = torch.sum((target - mean_target) ** 2)

    if SST < 1e-8:
        cod = 1.0 if SSR < 1e-8 else 0.0
    else:
        cod = 1.0 - (SSR / SST)

    # --- 3. RMSE (均方根误差) ---
    # RMSE = sqrt(SSR / n)
    rmse = torch.sqrt(SSR / n)

    # 返回 PCC, COD, RMSE
    return pcc.item(), cod.item(), rmse.item()




def calculate_pcc_cod(pred, target):
    label_score = target
    sys_score = pred
    device_cpu = torch.device('cpu')

    nan_mask = torch.isnan(label_score)
    label_score[nan_mask] = 0.0
    sys_score[nan_mask] = 0.0
    label_sys_corr = torch.tensor(torch.zeros(1, sys_score.shape[1]), device=device_cpu)
    label_sys_cod = torch.tensor(torch.zeros(1, label_sys_corr.shape[1]), device=device_cpu)

    for i in range(label_score.shape[1]):
        if np.corrcoef(label_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1] < 0:
            sys_score[1:, i] = -sys_score[1:, i]
        pcc = np.corrcoef(label_score[1:, i].detach().cpu().numpy(), sys_score[1:, i].detach().cpu().numpy())[0, 1]
        label_sys_corr[0, i] = pcc

    for i in range(label_sys_corr.shape[1]):
        # 获取第 i 个任务的真实值和预测值 (NumPy 格式)
        y_true = label_score[1:, i].detach().cpu().numpy()
        y_pred = sys_score[1:, i].detach().cpu().numpy()
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if valid_mask.sum() == 0:
            label_sys_cod[0, i] = 0.0
            continue

        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        n = len(y_true)

        if n < 2:
            label_sys_cod[0, i] = 0.0
            continue
        SSR = np.sum((y_true - y_pred) ** 2)
        SST = np.sum((y_true - np.mean(y_true)) ** 2)
        if SST < 1e-8:
            cod = 1.0 if SSR < 1e-8 else 0.0
        else:
            cod = 1.0 - (SSR / SST)
        label_sys_cod[0, i] = cod

    return label_sys_corr, label_sys_cod, sys_score[1:,:], label_score[1:,:]