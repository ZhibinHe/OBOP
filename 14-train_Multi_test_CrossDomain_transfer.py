
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
parser.add_argument('--stage1_epochs', type=int, default=5, help='Epochs for Stage 1')
parser.add_argument('--stage2_epochs', type=int, default=200, help='Epochs for Stage 2')
parser.add_argument('--batchSize', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='Weight decay')
parser.add_argument('--stepsize', type=int, default=20, help='Scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='Scheduler gamma')

# Dataset Path Configuration
parser.add_argument('--source_dataset', type=str, default='ABCD,HCP_A,HCP_D',
                    help='Source domain for training. Can be single (e.g., "ABCD") or multiple (e.g., "ABCD,HCP_D,HCP_YA")')
parser.add_argument('--target_dataset', type=str, default='HCP_YA', choices=['HCP_D', 'HCP_A', 'HCP_YA', 'ABCD'],
                    help='Target domain for testing')
parser.add_argument('--dataroot_root', type=str, default='/data/hzb/project/BodyDecoding/data', help='Data root')
parser.add_argument('--save_path', type=str, default='./model_merged/', help='Path to save models')

# Model Parameters
parser.add_argument('--nroi', type=int, default=400, help='Number of ROIs')
parser.add_argument('--num_prediction', type=int, default=183, help='Dimension of phenotype scores (labels)') # 183, 184, 184
parser.add_argument('--zero_num_prediction', type=int, default=35, help='number prediction')  #34, 34, 35
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


def data_load_combined(source_list, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    train_datasets_list = []
    val_datasets_list = []
    text_features_list = []

    raw_data_info = []

    total_tasks = 0

    print(f"[-] Starting Combined Data Loading for: {source_list}")

    for ds_name in source_list:
        if ds_name == 'ABCD':
            path = os.path.join(opt.dataroot_root, 'ABCD_pcp/Schaefer/filt_noglobal')
            opt.phenoroot = '/data/hzb/project/OBOP_data/ABCD/ABCD_phenotype'
            tr, val, _, txt = data_load_abcd(path, ds_name, opt, device)
        elif ds_name == 'HCP_A':
            path = os.path.join(opt.dataroot_root, 'HCP_A_pcp/Schaefer/filt_noglobal')
            opt.phenoroot = '/data/hzb/project/OBOP_data/HCP_A/HCP_A_phenotype'
            tr, val, _, txt = data_load_hcp_a(path, ds_name, opt, device)
        elif ds_name == 'HCP_D':
            path = os.path.join(opt.dataroot_root, 'HCP_D_pcp/Schaefer/filt_noglobal')
            opt.phenoroot = '/data/hzb/project/OBOP_data/HCP_D/HCP_D_phenotype'
            tr, val, _, txt = data_load_hcp_d(path, ds_name, opt, device)
        elif ds_name == 'HCP_YA':
            path = os.path.join(opt.dataroot_root, 'HCP_YA_pcp/Schaefer/filt_noglobal')
            opt.phenoroot = '/data/hzb/project/OBOP_data/HCP_YA/HCP_YA_phenotype'
            tr, val, _, txt = data_load_hcp_ya(path, ds_name, opt, device)
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

        num_tasks = txt.shape[0]
        text_features_list.append(txt)

        raw_data_info.append({
            'name': ds_name,
            'train': tr,
            'val': val,
            'n_tasks': num_tasks
        })

        total_tasks += num_tasks

    combined_text_feature = torch.cat(text_features_list, dim=0)

    print(f"[-] Combined Text Feature Shape: {combined_text_feature.shape}")
    print(f"[-] Total combined tasks: {total_tasks}")


    current_offset = 0
    final_train_list = []
    final_val_list = []

    for info in raw_data_info:
        n_tasks = info['n_tasks']
        start_idx = current_offset
        end_idx = current_offset + n_tasks

        print(f"    Processing {info['name']}: Tasks [{start_idx}:{end_idx}]")

        for data in info['train']:
            original_y = data.y.view(-1)  #

            new_y = torch.full((total_tasks,), float('nan'), device=original_y.device)

            new_y[start_idx:end_idx] = original_y

            data.y = new_y.unsqueeze(0)
            final_train_list.append(data)

        # 处理验证集 (逻辑同上)
        for data in info['val']:
            original_y = data.y.view(-1)
            new_y = torch.full((total_tasks,), float('nan'), device=original_y.device)
            new_y[start_idx:end_idx] = original_y
            data.y = new_y.unsqueeze(0)
            final_val_list.append(data)

        current_offset += n_tasks


    return final_train_list, final_val_list, None, combined_text_feature


def load_data(dataset_name_or_list, opt):
    if ',' in dataset_name_or_list:
        datasets = dataset_name_or_list.split(',')
        return data_load_combined(datasets, opt, device)

    print(f"[-] Loading dataset: {dataset_name_or_list}...")

    dataset_name = dataset_name_or_list
    if dataset_name == 'HCP_A':
        dataroot = os.path.join(opt.dataroot_root, 'HCP_A_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBIP_data/HCP_A/HCP_A_phenotype'
        return data_load_hcp_a(dataroot, dataset_name, opt)

    elif dataset_name == 'ABCD':
        dataroot = os.path.join(opt.dataroot_root, 'ABCD_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBIP_data/ABCD/ABCD_phenotype'
        return data_load_abcd(dataroot, dataset_name, opt)

    elif dataset_name == 'HCP_D':
        dataroot = os.path.join(opt.dataroot_root, 'HCP_D_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBIP_data/HCP_D/HCP_D_phenotype'
        return data_load_hcp_d(dataroot, dataset_name, opt)

    elif dataset_name == 'HCP_YA':
        dataroot = os.path.join(opt.dataroot_root, 'HCP_YA_pcp/Schaefer/filt_noglobal')
        opt.phenoroot = '/data/hzb/project/OBIP_data/HCP_YA/HCP_YA_phenotype'
        return data_load_hcp_ya(dataroot, dataset_name, opt)

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
    best_model_path = os.path.join(opt.save_path, f'best_stage1_model_train_{opt.source_dataset}.pth')

    for epoch in range(opt.stage1_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            score_predict, _, _ = model(data.x.view(int(data.x.shape[0] / opt.nroi), opt.nroi, opt.nroi),
                                        text_feature.to(device))
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
    model = Explicit_parameter_construction(num_task=opt.num_prediction).to(device)
    model.load_state_dict(torch.load(model_path))
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
    best_model_path = os.path.join(opt.save_path, f'best_stage2_model_test_{opt.target_dataset}_from_{opt.source_dataset}.pth')

    for epoch in range(opt.stage2_epochs):
        model.train()
        total_loss = 0.0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            bs = int(data.x.shape[0] / opt.num_prediction)
            valid_task_mask = ~torch.isnan(data.y.view(bs, opt.num_prediction))  # [bs, num_prediction]
            valid_task_mask = valid_task_mask.float().to(device)

            mask_interleaved = torch.zeros(bs, opt.num_prediction * 2, device=device)
            mask_interleaved[:, 0::2] = 1.0
            mask_interleaved[:, 1::2] = valid_task_mask
            mask_interleaved = mask_interleaved.unsqueeze(-1).unsqueeze(-1)
            fusion_feature_sbj = data.x.view(bs, opt.num_prediction, 1024)
            merged_matrix = torch.cat([fusion_feature_sbj, train_rw[0:bs, :, :]], dim=1)
            odd_columns = torch.arange(1, merged_matrix.size(1), 2)
            merged_matrix[:, odd_columns, :] = fusion_feature_sbj
            even_columns = torch.arange(0, merged_matrix.size(1), 2)
            merged_matrix[:, even_columns, :] = train_rw[0:bs, :, :]
            out, mask = model(merged_matrix, model.training, 0)
            t1 = mask.view(bs, opt.num_prediction * 2, 1)
            t1 = t1.unsqueeze(-1).repeat(1, 1, 1, out.logits.shape[-1])
            recon_diff = abs((out.logits - merged_matrix.view(bs, opt.num_prediction * 2, 1, 1024)))
            valid_count = mask_interleaved.sum()
            if valid_count < 1: valid_count = 1.0  # 防止除
            loss12 = (recon_diff * t1 * mask_interleaved).sum() / valid_count
            loss11_diff = abs((out.logits[:, :, 0, :] - merged_matrix))
            loss11 = (loss11_diff * mask_interleaved.squeeze(-1)).sum() * 0.01 / valid_count
            loss = opt.lamb0 * loss11 + loss12
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataset)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)

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

    model1 = Explicit_parameter_construction(num_task=opt.num_prediction).to(device)
    model1.load_state_dict(torch.load(stage1_model_path))
    model1.eval()

    device_cpu = torch.device('cpu')

    with torch.no_grad():


      # pred_score = torch.tensor(torch.zeros(1, text_feature_hcp.shape[0]), device=device)
      label_score = torch.tensor(torch.zeros(1, text_feature_tgt.shape[0]), device=device)
      fusion_feature = torch.tensor(torch.zeros(1, text_feature_tgt.shape[0], text_feature_tgt.shape[1]*2), device=device_cpu)

      for data in loader:
        data = data.to(device)
        all_datay = data.y
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


# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
if __name__ == '__main__':

    # =========================================================
    # A. Training Stage (If not test_only mode)
    # =========================================================
    if not opt.test_only:
        # 1. Load Source Data
        train_set_src, val_set_src, _, text_feature = load_data(opt.source_dataset, opt)

        opt.num_prediction = text_feature.shape[0]
        print(f"[-] Auto-updated num_prediction to {opt.num_prediction} based on source datasets.")

        train_loader_src = DataLoader(train_set_src, batch_size=opt.batchSize,shuffle=True)
        val_loader_src = DataLoader(val_set_src, batch_size=opt.batchSize, shuffle=False)

        # Define model paths (Naming convention updated to reflect combined sources)
        # source_name_clean = opt.source_dataset.replace(',', '_')
        best_stage1_path = os.path.join(opt.save_path, f'best_stage1_model_train_{opt.source_dataset}.pth')
        best_stage2_path = os.path.join(opt.save_path, f'best_stage2_model_test_{opt.target_dataset}_from_{opt.source_dataset}.pth')


###################
        best_stage1_path = run_stage1(train_loader_src, val_loader_src, text_feature)
#############################
        src_fusion_feat, src_rw, src_labels = extract_features(best_stage1_path, train_loader_src, text_feature,num_task=opt.num_prediction)

        best_stage2_path, rw_template = run_stage2(src_fusion_feat, src_rw, src_labels, text_feature)

        train_set_tgt, val_set_tgt, test_set_tgt, text_feature_tgt = load_data(opt.target_dataset, opt)
        test_loader_tgt = DataLoader(train_set_tgt, batch_size=opt.batchSize, shuffle=False)

        # Extract features using the model trained on 183 tasks
        tgt_fusion_feat, tgt_labels = extract_tgt_features(best_stage1_path, test_loader_tgt, text_feature_tgt)
        val_dataset_stage2 = CustomDataset(tgt_fusion_feat, tgt_labels)
        val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)

        # 6. Final Test
        per_task_pcc, per_task_cod, sys_score, label_score = test_acc_zero(best_stage2_path, val_dataset_stage2,tgt_fusion_feat, src_fusion_feat, src_rw)

        mean_pcc = torch.mean(per_task_pcc)

        print(f"\nFinal Test Results on {opt.target_dataset}:")
        print(f"  > Mean PCC: {mean_pcc:.4f}")
        print(f"  > Mean COD: {torch.mean(per_task_cod).item():.4f}")



    # B. Test Only Mode
    # =========================================================
    if  opt.test_only:

        # source_name_clean = opt.source_dataset.replace(',', '_')
        best_stage1_path = os.path.join(opt.save_path, f'best_stage1_model_train_{opt.source_dataset}.pth')
        best_stage2_path = os.path.join(opt.save_path, f'best_stage2_model_test_{opt.target_dataset}_from_{opt.source_dataset}.pth')

        train_set_src, val_set_src, _, text_feature = load_data(opt.source_dataset, opt)

        opt.num_prediction = text_feature.shape[0]
        print(f"[-] Auto-updated num_prediction to {opt.num_prediction} based on source datasets.")

        train_loader_src = DataLoader(train_set_src, batch_size=opt.batchSize,shuffle=False)
        val_loader_src = DataLoader(val_set_src, batch_size=opt.batchSize, shuffle=False)



        # 1. Check Model Existence
        if not os.path.exists(best_stage1_path) or not os.path.exists(best_stage2_path):
            raise FileNotFoundError(f"Missing model files for testing. S1: {best_stage1_path}, S2: {best_stage2_path}")

        print(f"\n" + "=" * 20 + f" Starting Standalone Test on {opt.target_dataset} " + "=" * 20)

        # 2. Load Source Data (to extract src_rw template)

        src_fusion_feat, src_rw, src_labels = extract_features(best_stage1_path, train_loader_src, text_feature,
                                                               num_task=opt.num_prediction)

        # 3. Load Target TEST Dataset

        train_set_tgt, val_set_tgt, test_set_tgt, text_feature_tgt = load_data(opt.target_dataset, opt)
        test_loader_tgt = DataLoader(train_set_tgt, batch_size=opt.batchSize, shuffle=False)

        # Extract features using the model trained on 183 tasks
        tgt_fusion_feat, tgt_labels = extract_tgt_features(best_stage1_path, test_loader_tgt, text_feature_tgt)
        val_dataset_stage2 = CustomDataset(tgt_fusion_feat, tgt_labels)
        val_dataset_stage2 = DataLoader(val_dataset_stage2, batch_size=opt.batchSize, shuffle=False, drop_last=True)



        # 5. Run Zero-Shot Test (Cross-Domain)
        per_task_pcc, per_task_cod, sys_score, label_score = test_acc_zero(best_stage2_path, val_dataset_stage2, tgt_fusion_feat, src_fusion_feat, src_rw)


        mean_pcc = torch.mean(per_task_pcc)

        # 6. Save results
        base_name = f'results_{opt.source_dataset}_to_{opt.target_dataset}'

        # Save results in NumPy format
        np.save(os.path.join(opt.save_path, f'{base_name}_sys_score.npy'), sys_score.numpy())
        np.save(os.path.join(opt.save_path, f'{base_name}_label_score.npy'), label_score.numpy())
        np.save(os.path.join(opt.save_path, f'{base_name}_pcc_per_task.npy'), per_task_pcc.numpy())
        np.save(os.path.join(opt.save_path, f'{base_name}_cod_per_task.npy'), per_task_cod.numpy())

        print(f"\nFinal Test Results on {opt.target_dataset}:")
        print(f"  > Mean PCC: {mean_pcc:.4f}")
        print(f"  > Mean COD: {torch.mean(per_task_cod).item():.4f}")
        print(f"  > Results saved with prefix: {base_name} in {opt.save_path}")

        print("\n[Done] Pipeline finished successfully.")







