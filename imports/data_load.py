from imports.ABIDEDataset import HCPfmriScoreDataset_sbjnum, HCPT1wScoreDataset_sbjnum, HCPAfmriScoreDataset_sbjnum, HCPAT1wScoreDataset_sbjnum
from imports.utils import train_val_test_split_hcp
import pandas as pd
from torch_geometric.data import InMemoryDataset,Data
import os.path as osp
from os import listdir
import os
import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_array
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
from imports.gdc import GDC


#########################################
def data_load_abcd(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/ABCD_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    dataset = FmriScoreDataset_sbjnum(path, name)

    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Phenotype_Value.csv'))  # pheno_value

    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Phenotype_Description.csv'))  # description


    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)


    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)


    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]


    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname


    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1

def data_load_hcp_a(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    dataset = FmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Value.csv'))  # pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Description.csv'))  # description
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1

def data_load_hcp_d(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = FmriScoreDataset_sbjnum(path, name)

    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Value.csv'))  # pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Description.csv'))  # description

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1

def data_load_hcp_ya(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    dataset = FmriScoreDataset_sbjnum(path, name)

    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Value.csv'))  # pheno_value

    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Description.csv'))  # description



    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    csv_indices = np.where(select_fname == csv_fname_values.astype(str)[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]


    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]
    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]

    return train_dataset, val_dataset, test_dataset, text_feature

##Intra

def data_load_abcd_intra(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/ABCD_Train_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/ABCD_Test_Phenotype_Description.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Train_Phenotype_Description.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Test_Phenotype_Description.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)



    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)


    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])


    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]





    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature

def data_load_hcp_a_intra(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Train_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Test_Phenotype_Description.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Train_Phenotype_Description.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Test_Phenotype_Description.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature

def data_load_hcp_d_intra(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Train_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Test_Phenotype_Description.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]

    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Train_Phenotype_Description.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Test_Phenotype_Description.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname


    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature #, dataset, train_index1, val_index1

def data_load_hcp_ya_intra(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Train_Phenotype_Description.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Test_Phenotype_Description.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]

    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Train_Phenotype_Description.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Test_Phenotype_Description.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values
    dataset.data.sbj_fname = [int(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]


    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature

##Intra_SL

def data_load_abcd_intra_SL(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/ABCD_Train_Phenotype_Description_SL.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/ABCD_Test_Phenotype_Description_SL.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Train_Phenotype_Description_SL.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/ABCD_Test_Phenotype_Description_SL.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)



    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)


    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])


    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[te_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature

def data_load_hcp_a_intra_SL(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Train_Phenotype_Description_SL.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Test_Phenotype_Description_SL.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]


    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Train_Phenotype_Description_SL.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Test_Phenotype_Description_SL.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature

def data_load_hcp_d_intra_SL(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Train_Phenotype_Description_SL.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Test_Phenotype_Description_SL.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]

    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Train_Phenotype_Description_SL.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Test_Phenotype_Description_SL.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname


    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature #, dataset, train_index1, val_index1

def data_load_hcp_ya_intra_SL(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Train_Phenotype_Description_SL.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]

    test_text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Test_Phenotype_Description_SL.pt'), map_location=device)
    text_text_feature = test_text_feature[0:test_text_feature.shape[0], :]

    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Value.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Train_Phenotype_Description_SL.csv')) # description
    testcsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Test_Phenotype_Description_SL.csv')) # description
    all_csvdata = pd.concat([traincsvdata['var_name'], testcsvdata['var_name']], ignore_index=True)

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(all_csvdata.shape[0]):
        select_score[i] = csvdata[all_csvdata[i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values
    dataset.data.sbj_fname = [int(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]


    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (y_arr.shape[0], 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (y_arr.shape[0], 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[:, np.newaxis])

    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature, test_text_feature

##Seamantic Sensitivity
def data_load_hcp_a_ss(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    if  opt.semantic_sensitivity == 1:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Description_1.pt'), map_location=device)
    elif opt.semantic_sensitivity == 2:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Description_2.pt'),map_location=device)
    elif opt.semantic_sensitivity == 3:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Description_3.pt'),map_location=device)


    text_feature = text_feature[0:text_feature.shape[0], :]

    dataset = FmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Value.csv'))  # pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_A_Phenotype_Description.csv'))  # description
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))


    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1

def data_load_hcp_d_ss(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    if  opt.semantic_sensitivity == 1:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Description_1.pt'), map_location=device)
    elif opt.semantic_sensitivity == 2:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Description_2.pt'),map_location=device)
    elif opt.semantic_sensitivity == 3:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Description_3.pt'),map_location=device)


    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = FmriScoreDataset_sbjnum(path, name)

    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Value.csv'))  # pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_D_Phenotype_Description.csv'))  # description

    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]


    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]


    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1

def data_load_hcp_ya_ss(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    if  opt.semantic_sensitivity == 1:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Description_1.pt'), map_location=device)
    elif opt.semantic_sensitivity == 2:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Description_2.pt'),map_location=device)
    elif opt.semantic_sensitivity == 3:
        text_feature = torch.load(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Description_3.pt'),map_location=device)

    text_feature = text_feature[0:text_feature.shape[0], :]

    dataset = FmriScoreDataset_sbjnum(path, name)

    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Value.csv'))  # pheno_value

    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/HCP_YA_Phenotype_Description.csv'))  # description



    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)
    csv_indices = np.where(select_fname == csv_fname_values.astype(str)[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]


    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (int(dataset.data.x.size(0)/400), 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]
    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0])
    tr_index = np.concatenate((tr_index, te_index))
    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]

    return train_dataset, val_dataset, test_dataset, text_feature




#########################################
#########################################
class FmriScoreDataset_sbjnum(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(FmriScoreDataset_sbjnum, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        #########################THIS##########################
        self.data, self.slices = read_datafmri_sbjnum_score(self.raw_dir)
        ##########################################################

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

def read_datafmri_sbjnum_score(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list,att_list = [], [], []
    sbj_fname = []
    edge_sbj_list = []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_fmri_data_score_sbjnum, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))
        sbj_fname.append(res[j][5])
        t = res[j][0]
        t[:] = j
        edge_sbj_list.append(t)

    edge_att_arr = np.concatenate(edge_att_list)
    edge_sbj_arr = np.concatenate(edge_sbj_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    pseudo_arr = np.concatenate(pseudo, axis=0)

###########################
    # y_list = [sub_list for sub_list in y_list if sub_list]
    for i in range(len(y_list)):
        if not y_list[i]:
            y_list[i] = [0]
    ###########################

    y_arr = np.stack(y_list)
    y_arr = (y_arr - np.mean(y_arr)) / np.std(y_arr)
    # y_arr = (y_arr - np.min(y_arr)) / (np.max(y_arr) - np.min(y_arr))

    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    edge_sbj_torch = torch.from_numpy(edge_sbj_arr.reshape(len(edge_sbj_arr), 1)).float()

    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr)  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    pseudo_torch = torch.from_numpy(pseudo_arr).float()

    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch,sbj_fname=sbj_fname, edge_sbj_torch=edge_sbj_torch  )

    # data = Data(x=att_list, edge_index=edge_index_list, y=y_torch, edge_attr=edge_att_list, pos = pseudo,sbj_fname=sbj_fname  )


    data, slices = split(data, batch_torch)

    return data, slices

def read_sigle_fmri_data_score_sbjnum(data_dir,filename,use_gdc =False):
    temp = dd.io.load(osp.join(data_dir, filename))
    # temp['corr'] = temp['corr'][1:, 1:]
    # temp['pcorr'] = temp['pcorr'][1:, 1:]


    # label_all = np.genfromtxt(osp.join(data_dir[:-17], 'interview_age.txt'))
    # ProcSpeed_AgeAdj.txt     ReadEng_Unadj.txt PMAT24_A_CR.txt
    # read edge and edge attribute
    pcorr = temp['pcorr'][()]
    pcorr[pcorr < 0] = 0
    positive_values = pcorr[pcorr > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    pcorr[pcorr < threshold] = 0



    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    # A = nx.to_scipy_sparse_array(G)
    A = nx.to_scipy_sparse_matrix(G)

    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = temp['corr'][()]
    att[att < 0] = 0
    positive_values = att[att > 0]
    num_elements = int(positive_values.size * 0.11)
    sorted_values = np.sort(positive_values)
    threshold = sorted_values[-num_elements]
    att[att < threshold] = 0


    # num_elements = int(att.size * 0.15)
    # flattened_matrix = att.flatten()
    # sorted_values = np.sort(flattened_matrix)
    # top_values = sorted_values[-num_elements:]
    # mask = np.isin(att, top_values)
    # att_1 = np.where(mask, att, 0)
    # att = att_1
    label = 1 #label_all[np.where(label_all==int(filename[3:-3]))[0],1]     # hcp D

    # label = label_all[np.where(label_all==int(filename[3:-4]))[0],1]     # hcp A

    # label = label_all[np.where(label_all==int(filename[:-8]))[0],1]     # which?

    # label = label_all[np.where(label_all==int(filename[:-3]))[0],1]




    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).long()
    sbj_fname = filename[:-3]# classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att, sbj_fname = sbj_fname)

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att.data.numpy(),edge_index.data.numpy(),att,label,num_nodes, sbj_fname

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices



#########################################
#########################################
#########################################


def data_load_abcd_fmri_delete_high(path, name, opt, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    fold = opt.fold


    text_feature = torch.load(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.pt'), map_location=device)
    text_feature = text_feature[0:text_feature.shape[0], :]
    dataset = HCPfmriScoreDataset_sbjnum(path, name)
    csvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Value_115.csv'))  #pheno_value
    traincsvdata = pd.read_csv(os.path.join(opt.phenoroot + '/1223_ABCD_Phenotype_Description_115.csv')) # description

    if opt.test_dataset=='HCP_A':
        test_text_feature = torch.load(os.path.join('/data/hzb/project/BodyDecoding_data/HCP_A/HCP_A_Phenotype/1223_HCP_A_Phenotype_Description_34.pt'), map_location=device)
    if opt.test_dataset=='HCP_D':
        test_text_feature = torch.load(os.path.join('/data/hzb/project/BodyDecoding_data/HCP_D/HCP_D_Phenotype/0109_HCP_D_Phenotype_Description_34.pt'),map_location=device)
    if opt.test_dataset=='HCP_YA':
        test_text_feature = torch.load(os.path.join('/data/hzb/project/BodyDecoding_data/HCP/HCP_Phenotype/1223_HCP_YA_Phenotype_Description_35.pt'),map_location=device)

    correlation_matrix = np.corrcoef(text_feature.cpu().numpy(), test_text_feature.cpu().numpy())  # 计算相关性矩阵
    row_correlation  = correlation_matrix[:text_feature.shape[0], text_feature.shape[0]:]
    row_correlation = torch.from_numpy(row_correlation)
    row_correlation = row_correlation.to(device)
    row_correlation_max =row_correlation.max(dim=1).values


#########################
    # num_top_percent = int(50 * 0.01 * row_correlation_max.size(0))
    #
    # top_values, top_indices = torch.topk(row_correlation_max, num_top_percent)
    # row_correlation2 = row_correlation
    # mask = torch.ones(row_correlation2.size(0), dtype=torch.bool)
    # mask[top_indices] = False
    # new_tensor = row_correlation2[mask]
    # new_tensor.max(dim=0).values

    ############################
    num_top_percent = int(opt.delete_percent*0.01 * row_correlation_max.size(0))

    top_values, top_indices = torch.topk(row_correlation_max, num_top_percent)
    top_indices = top_indices+1

    csvdata.iloc[:, top_indices.cpu().numpy()] = np.nan

    # csvdata_head = csvdata.columns.tolist()
    select_score = np.zeros((csvdata.shape[1] - 1, csvdata.shape[0]))

    for i in range(traincsvdata['var_name'].shape[0]):
        select_score[i] = csvdata[traincsvdata['var_name'][i]]

    # non_nan_columns = np.where(~np.isnan(select_score).any(axis=0))[0]
    # select_score_1 = select_score[:, non_nan_columns]
    # csv_fname_values = csvdata["Subject"][non_nan_columns].values

    select_score_1 = select_score
    csv_fname_values = csvdata["NAME"].values

    dataset.data.sbj_fname = [(x) for x in dataset.data.sbj_fname]
    dataset_sbj_fname = np.array(dataset.data.sbj_fname, ndmin=1)
    select_fname = np.intersect1d(dataset_sbj_fname, csv_fname_values)

     # save dataset_sbj_fname as .csv



    ###score

    csv_indices = np.where(select_fname == csv_fname_values[:, np.newaxis])
    select_score_2 = select_score_1[:, csv_indices[0]]

    #
    for i in range(select_score_2.shape[0]):
        select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]] = (select_score_2[i, np.where(
            ~np.isnan(select_score_2[i, :]))[0]] - np.mean(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])) / np.std(
            select_score_2[i, np.where(~np.isnan(select_score_2[i, :]))[0]])

    y_arr = select_score_2
    y_arr = y_arr.T
    y_torch = torch.from_numpy(y_arr)

    # (select_score_2[6,:] - np.mean(select_score_2[6,:])) / np.std(select_score_2[6,:])    select_score_2[6,np.where(~np.isnan(select_score_2[6,:]))[0]]

    ###other
    dataset_indices = np.where(select_fname == dataset_sbj_fname[:, np.newaxis])

    dataset_x = np.reshape(dataset.data.x, (9438, 400, 400))
    dataset_x = dataset_x[dataset_indices[0], :, :]
    dataset_x = np.reshape(dataset_x, (dataset_x.shape[0] * dataset_x.shape[1], 400))

    dataset_pos = np.reshape(dataset.data.pos, (9438, 400, 400))
    dataset_pos = dataset_pos[dataset_indices[0], :, :]
    dataset_pos = np.reshape(dataset_pos, (dataset_pos.shape[0] * dataset_pos.shape[1], 400))

    dataset_indices_torch = torch.from_numpy(dataset_indices[0])

    # att_indices = np.where(dataset_indices_torch == dataset.data.edge_sbj_torch[ :, np.newaxis])

    att_indices = np.arange(0, dataset.data.edge_sbj_torch[:, np.newaxis].shape[0])
    dataset_edge_sbj_torch = dataset.data.edge_sbj_torch[att_indices[0], :]
    dataset_edge_index = dataset.data.edge_index[:, att_indices[0]]
    dataset_edge_attr = dataset.data.edge_attr[att_indices[0], :]

    #########################

    dataset.data.x = dataset_x
    dataset.data.edge_index = dataset_edge_index
    dataset.data.edge_attr = dataset_edge_attr
    dataset.data.y = y_torch
    dataset.data.pos = dataset_pos
    dataset.data.edge_sbj_torch = dataset_edge_sbj_torch
    dataset.data.sbj_fname = select_fname

    del dataset.data.edge_index
    del dataset.data.edge_attr
    del dataset.data.pos
    del dataset.data.edge_sbj_torch
    del dataset.data.sbj_fname
    # torch.where(dataset.data.edge_sbj_torch!=dataset_indices)
    #
    # dataset.data.sbj_fname[0]
    # csvdata["Emotion_Task_Acc"][non_nan_index]
    # csvdata["Subject"]

    # dataset.data.y = dataset.data.y.squeeze()

    dataset.data.x[dataset.data.x == float('inf')] = 0

    tr_index, val_index, te_index = train_val_test_split_hcp(n_sub=dataset.data.y.size()[0], fold=fold)
    tr_index = np.concatenate((tr_index, te_index))

    # tr_index = tr_index[:-(tr_index.shape[0]%10)]
    # val_index = val_index[:-(val_index.shape[0]%10)]

    train_dataset = dataset[tr_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[val_index]



    return train_dataset, val_dataset, test_dataset, text_feature #, dataset, train_index1, val_index1
