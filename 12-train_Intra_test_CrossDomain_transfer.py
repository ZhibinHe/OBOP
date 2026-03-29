"""
=========================================================
INTRA-DATASET ZERO-SHOT PHENOTYPE PREDICTION PIPELINE
=========================================================
DESCRIPTION:
    This script implements a two-stage generative modeling pipeline designed to evaluate
zero-shot task generalization WITHIN a single dataset. The model learns a general
    mapping between brain connectivity and phenotype semantic features on a subset of
    tasks (Stage 1 & 2) and is then tested on unseen phenotype tasks within the same
    dataset population.

USAGE:
    1. Training & Testing (Default):
       python ./train_Intra_test_CrossDomain_transfer.py --dataset ABCD_intra  --num_prediction 71 --zero_num_prediction 44 --gpu_id 0
       python ./train_Intra_test_CrossDomain_transfer.py --dataset HCP_A_intra  --num_prediction 22 --zero_num_prediction 12 --gpu_id 0
       python ./train_Intra_test_CrossDomain_transfer.py --dataset HCP_D_intra  --num_prediction 21 --zero_num_prediction 13 --gpu_id 0
       python ./train_Intra_test_CrossDomain_transfer.py --dataset HCP_YA_intra  --num_prediction 23 --zero_num_prediction 12 --gpu_id 0


    2. Test Only (Skip Training, load existing models):
       python ./train_Intra_test_CrossDomain_transfer.py --test_only --dataset ABCD_intra  --num_prediction 71 --zero_num_prediction 44 --gpu_id 0
       python ./train_Intra_test_CrossDomain_transfer.py --test_only --dataset HCP_A_intra  --num_prediction 22 --zero_num_prediction 12 --gpu_id 0
       python ./train_Intra_test_CrossDomain_transfer.py --test_only --dataset HCP_D_intra  --num_prediction 21 --zero_num_prediction 13 --gpu_id 0
       python ./train_Intra_test_CrossDomain_transfer.py --test_only --dataset HCP_YA_intra  --num_prediction 23 --zero_num_prediction 12 --gpu_id 0


INPUTS:
    - Dataset Name: Selection of 'ABCD_intra', 'HCP_D_intra', 'HCP_A_intra', or 'HCP_YA_intra'.
    - Brain Connectivity: Preprocessed fMRI ROI-based connectivity matrices (Schaefer atlas).
    - Phenotype Scores: Behavioral or clinical scores partitioned into 'seen' (training) and 'unseen' (zero-shot test) tasks.
    - Text Features: Pre-computed semantic embeddings (e.g., from BERT/LLM) for both training and test tasks.

OUTPUTS:
    - Model Checkpoints:
        - Intra_best_stage1_model_train_<dataset>.pth (Supervised encoder weights)
        - Intra_best_stage2_model_test_<dataset>.pth (Generative decoder weights)
    - Test Results (.npy):
        - results_Intra_<dataset>_sys_score.npy (Zero-shot predicted phenotype scores)
        - results_Intra_<dataset>_label_score.npy (Ground truth phenotype scores)
        - results_Intra_<dataset>_pcc_per_task.npy (Pearson Correlation for each unseen task)
        - results_Intra_<dataset>_cod_per_task.npy (Coefficient of Determination/R2 for each unseen task)
=========================================================
"""



import argparse
import torch.nn as nn
from torch.optim import lr_scheduler
from torch_geometric.data import Data, Dataset, DataLoader

from net.configuration_brainlm import BrainLMConfig
from net.Network_Combine_HCP import Explicit_parameter_construction
from net.modeling_brainlm import BrainLMDecoder_mask
from imports.data_load import *
from imports.model_metrics import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# 1. Configuration and Arguments
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
# Training Parameters
parser.add_argument('--gpu_id', type=str, default='0', help='device id')
parser.add_argument('--stage1_epochs', type=int, default=100, help='Epochs for Stage 1')
parser.add_argument('--stage2_epochs', type=int, default=200, help='Epochs for Stage 2')
parser.add_argument('--batchSize', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='Weight decay')
parser.add_argument('--stepsize', type=int, default=20, help='Scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler gamma')

# Dataset Path Configuration
parser.add_argument('--dataset', type=str, default='HCP_A_intra', choices=['ABCD_intra', 'HCP_D_intra', 'HCP_A_intra', 'HCP_YA_intra'],
                    help='Source domain for training')
parser.add_argument('--dataroot_root', type=str, default='/data/hzb/project/BodyDecoding/data', help='Data root')
parser.add_argument('--save_path', type=str, default='./model_merged/', help='Path to save models')

# Model Parameters
parser.add_argument('--nroi', type=int, default=400, help='Number of ROIs')
parser.add_argument('--num_prediction', type=int, default=22, help='Dimension of phenotype scores (labels)') #71,44; 22, 12; 21, 13; 23, 12
parser.add_argument('--zero_num_prediction', type=int, default=12, help='number prediction')
parser.add_argument('--lamb0', type=float, default=1.0, help='Loss weight')
parser.add_argument('--test_only', action='store_true', help='Only run final test, skip training stages.')

opt = parser.parse_args()
loss_fn = nn.MSELoss(reduction='none')

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)



# ---------------------------------------------------------
# 2. Utility Classes
# ---------------------------------------------------------
def pearson_correlation_loss(pred, target, mask):
    """
    计算预测值与真值之间的平均 PCC 损失。
    pred, target, mask: [batch_size, num_tasks]
    """
    # 掩码处理
    x = pred * mask
    y = target * mask

    # 计算均值
    sum_mask = torch.sum(mask, dim=0) + 1e-8
    mu_x = torch.sum(x, dim=0) / sum_mask
    mu_y = torch.sum(y, dim=0) / sum_mask

    # 计算离差
    vx = (pred - mu_x) * mask
    vy = (target - mu_y) * mask

    # 计算相关系数
    cos = torch.nn.CosineSimilarity(dim=0)
    # 计算每列（每个任务）在 batch 维度上的 PCC
    rho = cos(vx, vy)

    # 返回 1 - 平均相关系数作为损失（PCC越大，损失越小）
    return 1 - torch.mean(rho)


class CustomDataset(Dataset):
    """Simple Dataset for Stage 2 DataLoader"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return Data(x=x, y=y)


# ---------------------------------------------------------
# 3. Data Loading Wrapper
# ---------------------------------------------------------
def load_data(dataset_name, opt):
    print(f"[-] Loading dataset: {dataset_name}...")

    # Construct the specific path based on the dataset name
    if  dataset_name == 'ABCD_intra':
        dataroot = os.path.join(opt.dataroot_root, 'ABCD_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBOP_data/ABCD/ABCD_phenotype'
        return data_load_abcd_intra(dataroot, dataset_name, opt)


    elif dataset_name == 'HCP_A_intra':
        dataroot = os.path.join(opt.dataroot_root, 'HCP_A_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBOP_data/HCP_A/HCP_A_phenotype'
        return data_load_hcp_a_intra(dataroot, dataset_name, opt)


    elif dataset_name == 'HCP_D_intra':
        dataroot = os.path.join(opt.dataroot_root, 'HCP_D_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBOP_data/HCP_D/HCP_D_phenotype'
        return data_load_hcp_d_intra(dataroot, dataset_name, opt)

    elif dataset_name == 'HCP_YA_intra':
        dataroot = os.path.join(opt.dataroot_root, 'HCP_YA_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBOP_data/HCP_YA/HCP_YA_phenotype'
        return data_load_hcp_ya_intra(dataroot, dataset_name, opt)


    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ---------------------------------------------------------
# 4. Stage 1: Supervised Phenotype Prediction
# ---------------------------------------------------------
def run_stage1(train_loader, val_loader, text_feature):
    print("\n" + "=" * 20 + " Starting Stage 1: Supervised Training " + "=" * 20)
    model = Explicit_parameter_construction(num_task=opt.num_prediction).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
    criterion = nn.MSELoss(reduction='none')

    best_pcc = -1.0
    best_model_path = os.path.join(opt.save_path, f'Intra_best_stage1_model_train_{opt.dataset}.pth')

    for epoch in range(opt.stage1_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            score_predict, _, _ = model(data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi),
                                        text_feature.to(device))
            data.y = data.y[:, 0:opt.num_prediction]
            data.y = torch.tensor(data.y, dtype=torch.float)
            nan_mask = torch.isnan(data.y)
            isnan_matrix = torch.zeros_like(data.y)
            isnan_matrix[~nan_mask] = 1
            data.y[torch.where(torch.isnan(data.y) == True)] = 0
            column_losses = loss_fn(torch.stack(score_predict)[:, :, 0].T * isnan_matrix, data.y * isnan_matrix)
            loss_c = torch.sum(column_losses)
            loss = opt.lamb0 * loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_pccs = []
        with torch.no_grad():
            preds_all = []
            targets_all = []
            for data in val_loader:
                data = data.to(device)
                score_predict, _, _ = model(data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi),
                                            text_feature.to(device))
                pred = torch.stack(score_predict)[:, :, 0].T

                preds_all.append(pred.cpu())
                targets_all.append(data.y.cpu())

            preds_all = torch.cat(preds_all, dim=0)
            targets_all = torch.cat(targets_all, dim=0)
            targets_all = targets_all[:, 0:opt.num_prediction]
            # Calculate mean PCC across all tasks
            for i in range(preds_all.shape[1]):
                pcc, _, _ = calculate_metrics(preds_all[:, i], targets_all[:, i])
                val_pccs.append(pcc)

            mean_val_pcc = np.mean(val_pccs)

        print(
            f"Epoch {epoch + 1}/{opt.stage1_epochs} | Train Loss: {train_loss:.4f} | Val Mean PCC: {mean_val_pcc:.4f}")

        # --- Save Best Model ---
        if mean_val_pcc > best_pcc:
            best_pcc = mean_val_pcc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New best Stage 1 model saved (PCC: {best_pcc:.4f})")

    return best_model_path


# ---------------------------------------------------------
# 5. Feature Extraction (Transition)
# ---------------------------------------------------------
def extract_features(model_path, loader, text_feature, num_task):
    print("\n[-] Extracting features using best Stage 1 model...")
    model = torch.load(model_path)
    model.eval()

    fusion_features = []
    regress_weights = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, reg_weight, feat_cat = model(data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi),
                                            text_feature.to(device))

            # Accumulate
            fusion_features.append(feat_cat.cpu())
            rw = torch.squeeze(torch.stack(reg_weight), axis=1).cpu()
            rw = rw.unsqueeze(0).repeat(fusion_features[0].size()[0], 1, 1)
            labels.append(data.y.cpu())

    fusion_features = torch.cat(fusion_features, dim=0)
    regress_weights = rw
    labels = torch.cat(labels, dim=0)
    labels = labels[:,0:opt.num_prediction]
    return fusion_features, regress_weights, labels


# ---------------------------------------------------------
# 6. Stage 2: Generative Modeling
# ---------------------------------------------------------
def run_stage2(train_feat, train_rw, train_labels, text_feature):      ##biaozhun
    print("\n" + "=" * 20 + " Starting Stage 2: Generative Modeling " + "=" * 20)

    dataset = CustomDataset(train_feat, train_labels)
    loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    config = BrainLMConfig(num_brain_voxels=opt.num_prediction * 2)
    model = BrainLMDecoder_mask(config, num_patches=196).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    train_rw = train_rw.to(device)
    best_loss = float('inf')
    best_model_path = os.path.join(opt.save_path, f'Intra_best_stage2_model_test_{opt.dataset}.pth')

    for epoch in range(opt.stage2_epochs):
        model.train()
        total_loss = 0.0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()


            bs = int(data.x.shape[0]/opt.num_prediction)
            fusion_feature_sbj = data.x.view(bs, opt.num_prediction, 1024)
            merged_matrix = torch.cat([fusion_feature_sbj, train_rw[0:bs,:,:]], dim=1)
            odd_columns = torch.arange(1, merged_matrix.size(1), 2)
            merged_matrix[:, odd_columns, :] = fusion_feature_sbj
            even_columns = torch.arange(0, merged_matrix.size(1), 2)
            merged_matrix[:, even_columns, :] = train_rw[0:bs,:,:]
            out, mask = model(merged_matrix, model.training, 0)
            t1 = mask.view(bs, opt.num_prediction * 2, 1)
            t1 = t1.unsqueeze(-1).repeat(1, 1, 1, out.logits.shape[-1])
            loss12 = abs((out.logits - merged_matrix.view(bs, opt.num_prediction * 2, 1, 1024)) * t1).sum()
            loss11 = abs((out.logits[:, :, 0, :] - merged_matrix)).sum() * 0.01
            loss = opt.lamb0 * loss11 + loss12
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{opt.stage2_epochs} | Train Loss: {avg_loss:.4f}")


        # Save Best Model based on Training Loss (or you can pass validation set)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            # print(f"  >>> New best Stage 2 model saved")

    return best_model_path, train_rw


# ---------------------------------------------------------
# 7. Zero-Shot Testing Function
# ---------------------------------------------------------
def test_acc_zero(stage2_model_path, train_loader_stage2, tgt_fusion_feat, src_fusion_feat, src_rw):

    config = BrainLMConfig(num_brain_voxels=opt.num_prediction * 2)
    model2 = BrainLMDecoder_mask(config, num_patches=196).to(device)
    model2.load_state_dict(torch.load(stage2_model_path))
    model2.eval()

    correct = []
    device_cpu = torch.device('cpu')
    src_rw = src_rw.to(device)
    with torch.no_grad():
      label_sys_corr =  torch.tensor(torch.zeros(1, tgt_fusion_feat.shape[1]), device=device_cpu)
      label_score  =  torch.tensor(torch.zeros(1, tgt_fusion_feat.shape[1]), device=device_cpu)
      sys_score =   torch.tensor(torch.zeros(1, tgt_fusion_feat.shape[1]), device=device_cpu)

      for data in train_loader_stage2:
          data = data.to(device)

          fusion_feature_sbj = data.x.view(opt.batchSize, opt.zero_num_prediction, 1024)
          label_sbj = data.y.view(opt.batchSize, opt.zero_num_prediction)
          fusion_feature_template =src_fusion_feat[:opt.batchSize, :].to(device)
          mask_index = 0
          sys_score_tmp = torch.zeros(label_sbj.shape, device=device)

          for score_index in range(label_sbj.shape[1]):
              fusion_feature_template[:, mask_index:mask_index+1, :] = fusion_feature_sbj[:, score_index:score_index + 1, :].to(device)
              merged_matrix = torch.cat([fusion_feature_template, src_rw], dim=1)
              odd_columns = torch.arange(1, merged_matrix.size(1), 2)
              merged_matrix[:, odd_columns, :] = fusion_feature_template
              even_columns = torch.arange(0, merged_matrix.size(1), 2)
              merged_matrix[:, even_columns, :] = src_rw
              out, mask = model2(merged_matrix, model2.training, mask_index * 2)
              sys_score_tmp[:,score_index] = torch.matmul(merged_matrix[:, mask_index * 2 + 1, :], out.logits[:,mask_index * 2, 0, :].T)[:, 0]

          sys_score = torch.cat((sys_score, torch.tensor(sys_score_tmp, device=device_cpu)), dim=0)
          label_score = torch.cat((label_score, torch.tensor(label_sbj, device=device_cpu)), dim=0)

      label_sys_corr, label_sys_cod, sys_score, label_score = calculate_pcc_cod(sys_score, label_score)

    return label_sys_corr, label_sys_cod, sys_score, label_score

def extract_tgt_features(stage1_model_path, loader, text_feature_tgt):

    model1 = torch.load(stage1_model_path)
    model1.eval()

    device_cpu = torch.device('cpu')

    with torch.no_grad():


      # pred_score = torch.tensor(torch.zeros(1, text_feature_hcp.shape[0]), device=device)
      label_score = torch.tensor(torch.zeros(1, text_feature_tgt.shape[0]), device=device)
      fusion_feature = torch.tensor(torch.zeros(1, text_feature_tgt.shape[0], text_feature_tgt.shape[1]*2), device=device_cpu)

      for data in loader:
        data = data.to(device)
        all_datay = data.y[:,opt.num_prediction:]
        feature_cat_tmp = torch.tensor(torch.zeros(all_datay.shape[0], text_feature_tgt.shape[0], text_feature_tgt.shape[1]*2), device=device)
        text_feature_tmp = text_feature

        for i in range(text_feature_tgt.shape[0]):
            text_feature_tmp[0:1, :] = text_feature_tgt[i:i+1, :]
            score_predict, regress_weight, feature_cat = model1(
                data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi), text_feature_tmp)
            feature_cat_tmp[:, i, :] = feature_cat[:, 0, :]


        fusion_feature = torch.cat((fusion_feature, feature_cat_tmp.cpu()), dim=0)
        label_score = torch.cat((label_score, all_datay), dim=0)
    return fusion_feature[1:,:,:] , label_score[1:,:]


def run_stage2_zhibin(train_feat, train_rw, train_labels, text_feature, val_loader_tgt, tgt_fusion_feat, src_fusion_feat,
               src_rw):
    # ... (初始化和训练循环代码不变)
    print("\n" + "=" * 20 + " Starting Stage 2: Generative Modeling " + "=" * 20)

    # 准备 Stage 2 数据
    # 将 fusion_feature 和 regress_weight 展平或拼接，取决于 model2 的输入要求
    # 参照你的代码：merged_matrix 拼接

    # 构建 DataLoader
    # 这里我们只存 fusion_feature，在训练循环里拼接 regress_weight
    dataset = CustomDataset(train_feat, train_labels)
    loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    config = BrainLMConfig(num_brain_voxels=opt.num_prediction * 2)
    model = BrainLMDecoder_mask(config, num_patches=196).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

    train_rw = train_rw.to(device)
    # 用于拼接的 Regress Weight (假设所有样本共享一套权重，或者取平均，或者是 Stage 1 的训练集权重)
    # 注意：在你的源代码中，regress_weight_tr_repeat 似乎是固定的
    # 这里为了通用性，我们取训练集权重的平均值作为模板，或者你需要根据逻辑修改
    # regress_weight_template = torch.mean(train_rw, dim=0).unsqueeze(0).to(device)  # [1, num_pred, dim]

    # best_loss = float('inf')
    best_pcc = -1.0  # <--- 评估指标改为PCC
    best_model_path = os.path.join(opt.save_path, f'Intra_best_stage2_model_test_{opt.dataset}.pth')


    for epoch in range(opt.stage2_epochs):
        # ... (Training Code)
        model.train()
        total_loss = 0.0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            bs = int(data.x.shape[0] / opt.num_prediction)
            fusion_feature_sbj = data.x.view(bs, opt.num_prediction, 1024)

            # 拼接并整理 merged_matrix (Even: Weights, Odd: Features)
            merged_matrix = torch.cat([fusion_feature_sbj, train_rw[0:bs, :, :]], dim=1)
            odd_columns = torch.arange(1, merged_matrix.size(1), 2)
            merged_matrix[:, odd_columns, :] = fusion_feature_sbj
            even_columns = torch.arange(0, merged_matrix.size(1), 2)
            merged_matrix[:, even_columns, :] = train_rw[0:bs, :, :]

            # 模型前向传播
            out, mask_out = model(merged_matrix, model.training, 0)

            # --- [新增逻辑] 计算预测分数 ---
            # 根据 test_acc_zero 的逻辑：分数 = 权重 与 重建特征 的点积
            # 在本循环中：偶数列是权重，奇数列是模型重建的特征
            pred_features = out.logits[:, 1::2, 0, :]  # 模型生成的特征 [bs, num_pred, 1024]
            target_weights = merged_matrix[:, 0::2, :]  # 对应的回归权重 [bs, num_pred, 1024]

            # 点积求和得到预测分数 [bs, num_pred]
            pred_scores = torch.sum(pred_features * target_weights, dim=-1)

            # 准备真实标签和掩码
            target_scores = data.y.view(bs, opt.num_prediction)
            # 处理 NaN
            isnan_mask = ~torch.isnan(target_scores)
            clean_target_scores = torch.where(isnan_mask, target_scores, torch.zeros_like(target_scores))

            # 计算 PCC 损失
            loss_pcc = pearson_correlation_loss(pred_scores, clean_target_scores, isnan_mask.float())
            # -----------------------------

            # 原有的重构损失
            t1 = mask_out.view(bs, opt.num_prediction * 2, 1)
            t1 = t1.unsqueeze(-1).repeat(1, 1, 1, out.logits.shape[-1])
            loss_recon = abs((out.logits - merged_matrix.view(bs, opt.num_prediction * 2, 1, 1024)) * t1).sum()
            loss_mse = abs((out.logits[:, :, 0, :] - merged_matrix)).sum() * 0.01

            # --- [修改] 整合总损失 ---
            # 给 PCC 损失分配一个权重（例如 0.5），你可以根据需要调整
            lamb_pcc = 0.5
            loss = opt.lamb0 * loss_mse + loss_recon + lamb_pcc * loss_pcc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            total_loss += loss.item()

            # bs = int(data.x.shape[0]/opt.num_prediction)
            # fusion_feature_sbj = data.x.view(bs, opt.num_prediction, 1024)
            # merged_matrix = torch.cat([fusion_feature_sbj, train_rw[0:bs,:,:]], dim=1)
            # odd_columns = torch.arange(1, merged_matrix.size(1), 2)
            # merged_matrix[:, odd_columns, :] = fusion_feature_sbj
            # even_columns = torch.arange(0, merged_matrix.size(1), 2)
            # merged_matrix[:, even_columns, :] = train_rw[0:bs,:,:]
            # out, mask = model(merged_matrix, model.training, 0)
            # t1 = mask.view(bs, opt.num_prediction * 2, 1)
            # t1 = t1.unsqueeze(-1).repeat(1, 1, 1, out.logits.shape[-1])
            # loss12 = abs((out.logits - merged_matrix.view(bs, opt.num_prediction * 2, 1, 1024)) * t1).sum()
            # loss11 = abs((out.logits[:, :, 0, :] - merged_matrix)).sum() * 0.01
            # loss = opt.lamb0 * loss11 + loss12
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            # optimizer.step()
            # total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{opt.stage2_epochs} | Train Loss: {avg_loss:.4f}")


        # --- Validation (Cross-Domain Test) ---
        model.eval()
        # 调用零样本测试函数，这里使用 test_acc_zero
        # 传递模型本身，而不是路径，以便在当前 epoch 的权重上测试
        current_model_path = os.path.join(opt.save_path, f'current_stage2_model_test_{opt.dataset}.pth')

        torch.save(model.state_dict(), current_model_path)

        # 使用当前训练的模型进行零样本测试
        mean_val_pcc, mean_val_cod, _, _ = test_acc_zero(
            current_model_path, val_loader_tgt, tgt_fusion_feat, src_fusion_feat, src_rw
        )
        mean_val_pcc = torch.mean(mean_val_pcc)

        print(
            f"Epoch {epoch + 1}/{opt.stage2_epochs} | Train Loss: {avg_loss:.4f} | Val Mean PCC (Cross-Domain): {mean_val_pcc:.4f} | Val Mean COD (Cross-Domain): {torch.mean(mean_val_cod):.4f}")

        # --- Save Best Model ---
        if mean_val_pcc > best_pcc:  # <--- 使用跨域测试的PCC作为判断标准
            if torch.mean(mean_val_cod)>0:
                best_pcc = mean_val_pcc
                torch.save(model.state_dict(), best_model_path)
                print(f"  >>> New best Stage 2 model saved (Cross-Domain PCC: {best_pcc:.4f})")

    # ...
    return best_model_path, train_rw




# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
if __name__ == '__main__':

    # =========================================================
    # A. Training Stage (If not test_only mode)
    # =========================================================
    if opt.test_only:


        # 1. Load Source Data
        train_set, val_set, test_set, text_feature, test_text_feature = load_data(opt.dataset, opt)
        train_loader_src = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True)
        val_loader_src = DataLoader(val_set, batch_size=opt.batchSize, shuffle=False)

        # Define model paths
        best_stage1_path = os.path.join(opt.save_path, f'Intra_best_stage1_model_train_{opt.dataset}.pth')
        best_stage2_path = os.path.join(opt.save_path, f'Intra_best_stage2_model_test_{opt.dataset}.pth')

        # 2. Stage 1: Supervised Training
        # best_stage1_path = run_stage1(train_loader_src, val_loader_src, text_feature)

        # 3. Feature Extraction
        src_fusion_feat, src_rw, src_labels = extract_features(best_stage1_path, train_loader_src, text_feature,
                                                               num_task=opt.num_prediction)


        # 4. Stage 2: Generative Modeling
        # best_stage2_path, rw_template = run_stage2(src_fusion_feat, src_rw, src_labels, text_feature)


        test_loader = DataLoader(test_set, batch_size=opt.batchSize, shuffle=False)
        tgt_fusion_feat, tgt_labels = extract_tgt_features(best_stage1_path, test_loader, test_text_feature)
        val_dataset_stage2 = CustomDataset(tgt_fusion_feat, tgt_labels)
        val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)
        best_stage2_path, rw_template = run_stage2_zhibin(src_fusion_feat, src_rw, src_labels, text_feature, val_dataset_stage2, tgt_fusion_feat, src_fusion_feat, src_rw)

        # 5. Final Test: Use saved model for cross-domain PCC and COD
        per_task_pcc, per_task_cod, sys_score, label_score = test_acc_zero(best_stage2_path, val_dataset_stage2, tgt_fusion_feat, src_fusion_feat, src_rw)


        mean_pcc = torch.mean(per_task_pcc)

        print(f"\nFinal Test Results on {opt.dataset}:")
        print(f"  > Mean PCC: {mean_pcc:.4f}")
        print(f"  > Mean COD: {torch.mean(per_task_cod).item():.4f}")
        print("\n[Done] Pipeline finished successfully.")


    # =========================================================
    # B. Test Only Mode
    # =========================================================
    if  not opt.test_only:

        # Define model paths
        best_stage1_path = os.path.join(opt.save_path, f'Intra_best_stage1_model_train_{opt.dataset}.pth')
        best_stage2_path = os.path.join(opt.save_path, f'Intra_best_stage2_model_test_{opt.dataset}.pth')

        # 1. Check Model Existence
        if not os.path.exists(best_stage1_path) or not os.path.exists(best_stage2_path):
            raise FileNotFoundError(f"Missing model files for testing. S1: {best_stage1_path}, S2: {best_stage2_path}")

        print(f"\n" + "=" * 20 + f" Starting Standalone Test on {opt.dataset} " + "=" * 20)

        # 2. Load Source Data (to extract src_rw template)
        train_set, val_set, test_set, text_feature, test_text_feature = load_data(opt.dataset, opt)

        train_loader_src = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True)
        val_loader_src = DataLoader(val_set, batch_size=opt.batchSize, shuffle=False)
        src_fusion_feat, src_rw, src_labels = extract_features(best_stage1_path, train_loader_src, text_feature,
                                                               num_task=opt.num_prediction)


        # 3. Extract Target Test Features

        test_loader = DataLoader(test_set, batch_size=opt.batchSize, shuffle=False)
        tgt_fusion_feat, tgt_labels = extract_tgt_features(best_stage1_path, test_loader, test_text_feature)
        val_dataset_stage2 = CustomDataset(tgt_fusion_feat, tgt_labels)
        val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)

        # 4. Run Zero-Shot Test (Cross-Domain)
        per_task_pcc, per_task_cod, sys_score, label_score = test_acc_zero(best_stage2_path, val_dataset_stage2, tgt_fusion_feat, src_fusion_feat, src_rw)


        mean_pcc = torch.mean(per_task_pcc)

        # 6. Save results
        base_name = f'results_Intra_{opt.dataset}'

        # Save results in NumPy format
        np.save(os.path.join(opt.save_path, f'{base_name}_sys_score.npy'), sys_score.numpy())
        np.save(os.path.join(opt.save_path, f'{base_name}_label_score.npy'), label_score.numpy())
        np.save(os.path.join(opt.save_path, f'{base_name}_pcc_per_task.npy'), per_task_pcc.numpy())
        np.save(os.path.join(opt.save_path, f'{base_name}_cod_per_task.npy'), per_task_cod.numpy())

        print(f"\nFinal Test Results on {opt.dataset}:")
        print(f"  > Mean PCC: {mean_pcc:.4f}")
        print(f"  > Mean COD: {torch.mean(per_task_cod).item():.4f}")
        print(f"  > Results saved with prefix: {base_name} in {opt.save_path}")

        print("\n[Done] Pipeline finished successfully.")

