import os
import argparse
import torch
import json
import warnings
from utils import DTADataset, GraphDataset, custom_collate, model_evaluate, load_pickle, proGraph
from models import MultiDTA
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import r2_score
from math import sqrt
from scipy import stats
from lifelines.utils import concordance_index
import random
from datetime import datetime
import csv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

def cindex_score(y, p):
    sum_m = 0
    pair = 0
    print(f"len(y) {len(y)}")
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    sum_m += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    if pair != 0:
        return sum_m / pair
    else:
        return 0
    
def regression_scores(label, pred, is_valid=True):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    mse = ((label - pred)**2).mean(axis=0)
    rmse = sqrt(mse)
    if is_valid:
        ci = -1
    else:
        ci = concordance_index(label, pred)
    r2 = r2_score(label, pred)
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    return round(mse, 6), round(rmse, 6), round(ci, 6), round(r2, 6), round(pearson, 6), round(spearman, 6)
    
def test(model, dataloader, pro_graph, device, is_valid=True):
    model.eval()
    preds = []
    labels = []
    # print('Make prediction for {} samples...'.format(len(dataloader.dataset)))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            drug_data = batch_data.to(device)
            affinity = drug_data.y.to(device)
            pred = model(drug_data, pro_graph)
            # print(pred)
            preds += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += affinity.cpu().numpy().reshape(-1).tolist() 
    preds = np.array(preds)
    labels = np.array(labels)
    mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(labels, preds, is_valid)
    return mse_value, rmse_value, ci, r2, pearson_value, spearman_value
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=1000)               
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--fold', type=int, help='Fold of 5-CV', default=5)
    parser.add_argument('--running_set', type=str, help='new_pair', default='new_pair')
    parser.add_argument('--max_patience', type=int, help='early stop patience', default=100)
    parser.add_argument('--seed', type=float, help='random seed', default=0)
    args, _ = parser.parse_known_args()
    # args = parser.parse_args()

    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    dataset = args.dataset
    cuda_name = f'cuda:{args.cuda}'
    TRAIN_BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LR = args.lr
    Model = MultiDTA
    kfold = args.fold
    max_patience = args.max_patience

    model_name = Model.__name__

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print('batch size', TRAIN_BATCH_SIZE)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print('running set', args.running_set)
    # print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    dataset_path = 'data_split/' + args.dataset + '/' + args.running_set + '/'
    # print(dataset_path)
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print(device)
    print("load drug substructure graph")
    drug_graphs_dict = load_pickle(f'data/{dataset}/sub_mol_data.pkl')
    # print(drug_graphs_dict['Cc1[nH]nc2ccc(-c3cncc(OCC(N)Cc4ccccc4)c3)cc12'])
    
    print("load pretrained feature ...")
    mol2vec_dict = load_pickle(f'pretrained-featrue/{dataset}/{dataset}_drug_pretrain.pkl')
    # print(mol2vec_dict.keys())
    protvec_dict = load_pickle(f'pretrained-featrue/{dataset}/{dataset}_esm_pretrain.pkl')
    
    fold_metrics = {'mse':[], 'rmse':[], 'ci':[], 'r2':[], 'pearson':[], 'spearman':[]}
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    for fold_i in range(kfold):  
        print("Data preparation in progress for the {} dataset...".format(args.dataset))
        train_target_graphs_dict = load_pickle(f'{dataset_path}fold_{fold_i}_train_prot_data.pkl')
        valid_target_graphs_dict = load_pickle(f'{dataset_path}pro_data.pkl')
        test_target_graphs_dict = load_pickle(f'{dataset_path}fold_{fold_i}_test_prot_data.pkl')
        train_pro_graph = proGraph(train_target_graphs_dict, protvec_dict, TRAIN_BATCH_SIZE, device) 
        valid_pro_graph = proGraph(valid_target_graphs_dict, protvec_dict, TRAIN_BATCH_SIZE, device) 
        test_pro_graph = proGraph(test_target_graphs_dict, protvec_dict, TRAIN_BATCH_SIZE, device) 
        pro_num = len(train_pro_graph[-1]) + len(valid_pro_graph[-1]) + len(test_pro_graph[-1])
        df_train = pd.read_csv(f'{dataset_path}' + f'fold_{fold_i}_train.csv')
        df_valid = pd.read_csv(f'{dataset_path}' + f'fold_{fold_i}_valid.csv')
        df_test = pd.read_csv(f'{dataset_path}' + f'fold_{fold_i}_test.csv') 
        train_smiles,train_seq,train_label = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']),list(df_train['affinity'])
        valid_smiles,valid_seq,valid_label = list(df_valid['compound_iso_smiles']), list(df_valid['target_sequence']),list(df_valid['affinity'])
        test_smiles,test_seq,test_label = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']),list(df_test['affinity'])
        # print(protvec_dict.keys())
        
        print("load data finished ...")
        train_dataset = DTADataset(smile_list=train_smiles, seq_list=train_seq, label_list=train_label, mol_data=drug_graphs_dict, pro_data=train_target_graphs_dict, drug_pretrain_dict=mol2vec_dict)
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate,num_workers=4)
        valid_dataset = DTADataset(smile_list=valid_smiles, seq_list=valid_seq, label_list=valid_label, mol_data=drug_graphs_dict, pro_data=valid_target_graphs_dict, drug_pretrain_dict=mol2vec_dict)
        valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate,num_workers=4)
        test_dataset = DTADataset(smile_list=test_smiles, seq_list=test_seq, label_list=test_label, mol_data=drug_graphs_dict, pro_data=test_target_graphs_dict, drug_pretrain_dict=mol2vec_dict)
        test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=custom_collate,num_workers=4)
        
        print("load model ...")
        model = Model(n_output=1, output_dim=128, num_features_xd=130, num_features_pro=33, device=device, pro_num=pro_num, batch_size=TRAIN_BATCH_SIZE)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), LR, betas=(0.9, 0.999))
        criterion = F.mse_loss
        
        print("start training ...")
        train_log = []     
        best_valid_mse = 100
        patience = 0
        # model_file_name = f'./results/{dataset}/'  + f'train_{model_name}.model'
        result_file_name = f'./results/{dataset}/' + f'train_{model_name}.csv'
        # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # print(current_time)
        model_fromTrain = f'./savemodel/{dataset}-{args.running_set}-fold{fold_i}-{current_time}.pth'
        for epoch in range(1, NUM_EPOCHS + 1):
            # print('Training on {} samples...'.format(len(train_loader.dataset)))
            model.train()
            pred = []
            label = []  
            # total_loss = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                drug_data = batch_data.to(device)
                affinity = drug_data.y.to(device)
                # print(drug_data.seq_num)
                predictions = model(drug_data, train_pro_graph)
                pred = pred + predictions.cpu().detach().numpy().reshape(-1).tolist()
                label = label + affinity.cpu().detach().numpy().reshape(-1).tolist() 
                loss = criterion(predictions.squeeze(), affinity)
                loss.backward()
                optimizer.step()
                # total_loss = total_loss + loss.item()
                optimizer.zero_grad()
            pred = np.array(pred)
            label= np.array(label)
            mse_value, rmse_value, ci, r2, pearson_value, spearman_value = regression_scores(pred, label)
            train_log.append([mse_value, rmse_value, ci, r2, pearson_value, spearman_value])
            print(f'Traing Log at fold: {fold_i}, epoch: {epoch}, mse: {mse_value}, rmse: {rmse_value}, r2: {r2}, dataset: {dataset}')
        
            # valid
            mse, rmse, ci, r2, pearson, spearman = test(model, valid_loader, valid_pro_graph, device, is_valid=True)
            print(f'Valid at fold:{fold_i}, mse:{mse}')
            # Early stop        
            if mse < best_valid_mse :
                patience = 0
                best_valid_mse = mse
                # save model
                torch.save(model.state_dict(), model_fromTrain)
                print(f'Update best_mse, Valid at fold: {fold_i}, epoch: {epoch}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}')
            else:
                patience += 1
                if patience > max_patience:
                    print(f'Traing stop at epoch-{epoch}, model save at-{model_fromTrain}')
                    break   
                
        log_dir = f"./log/{current_time}-{dataset}-{args.running_set}-fold{fold_i}.csv"
        with open(log_dir, "w+")as f:
            writer = csv.writer(f)
            writer.writerow(["mse", "rmse",  "ci", "r2", 'pearson', 'spearman'])
            for r in train_log:
                writer.writerow(r)
        print(f'Save log over at {log_dir}')
            
        # Test    
        predModel = Model(n_output=1, output_dim=128, num_features_xd=130, num_features_pro=33, device=device, pro_num=pro_num, batch_size=TRAIN_BATCH_SIZE)
        predModel.load_state_dict(torch.load(model_fromTrain))
        predModel = predModel.to(device)    
        mse, rmse, ci, r2, pearson, spearman = test(predModel, test_loader, test_pro_graph, device, is_valid=False)
        print(f'Test at fold: {fold_i}, mse: {mse}, rmse: {rmse}, ci: {ci}, r2: {r2}, pearson: {pearson}, spearman: {spearman}\n')
        fold_metrics['mse'].append(mse)
        fold_metrics['rmse'].append(rmse)
        fold_metrics['ci'].append(ci)
        fold_metrics['r2'].append(r2)
        fold_metrics['pearson'].append(pearson)
        fold_metrics['spearman'].append(spearman)
        
    # save training log
    fold_test_metrics = pd.DataFrame(fold_metrics)    
    fold_test_metrics.to_csv(f'./log/Test-{dataset}-{args.running_set}-{current_time}.csv', index=False)    
    mean_values = fold_test_metrics.mean()
    variance_values = fold_test_metrics.var()    
    print(f"Dataset-{dataset}-{args.running_set}")
    print(f"Mean Values:{pd.concat([mean_values, variance_values], axis=1)}") 
        




