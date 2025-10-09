import torch
import numpy as np

from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
import pickle

def load_pickle(dir):
    with open(dir, 'rb+') as f:
        return pickle.load(f)

def matrix_pad(arr, max_len):   
    dim = arr.shape[-1]
    len = arr.shape[0]
    if len < max_len:            
        new_arr = np.zeros((max_len, dim))
        vec_mask = np.zeros((max_len))                            
        new_arr[:len] = arr
        vec_mask[:len] = 1
        return new_arr, vec_mask
    else:
        new_arr = arr[:max_len]
        vec_mask = np.ones((max_len))  
        return new_arr, vec_mask

class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', graphs_dict=None, seq_pretrain_dict = None, dttype=None,):
        super(GraphDataset, self).__init__(root)
        self.dttype = dttype
        self.process(graphs_dict, seq_pretrain_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
    
    def _process_pretrain(self, seq_pretrain_dict):
        protein_max = 1022
        protvec_dim = 1280
        target_batch_size = len(seq_pretrain_dict['length_dict'])
        target_id = list(seq_pretrain_dict['length_dict'].keys())
        all_prot_mat = torch.zeros((target_batch_size, protein_max, protvec_dim), dtype=torch.float32)
        
        all_prot_mat_dict = {}
        for i, id in enumerate(target_id):
            prot_mat = seq_pretrain_dict["mat_dict"][id]
            prot_mat_pad, prot_mask = matrix_pad(prot_mat, protein_max)
            
            all_prot_mat[i] = torch.from_numpy(prot_mat_pad)
            all_prot_mat_dict[id] = all_prot_mat[i]
        # print(all_prot_mat.shape)                    # 379, 1022, 1280 
        
        return all_prot_mat_dict
        
    def process(self, graphs_dict, seq_pretrain_dict):
        data_list = []
        # count = 0
        all_prot_mat_dict = self._process_pretrain(seq_pretrain_dict)
        for key in graphs_dict:
            # print(key)
            size, features, edge_index, target_id, _= graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index), target_embedding=torch.FloatTensor(all_prot_mat_dict[target_id]).unsqueeze(0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            # count += 1
            data_list.append(GCNData)
        # print('data数据', len(data_list), data_list[0])
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def proGraph(graph_data, seq_pretrain_dict, batch_size, device):
    proGraph_dataset = GraphDataset(graphs_dict=graph_data, seq_pretrain_dict=seq_pretrain_dict, dttype= 'pro')
    proGraph_loader = DataLoader(proGraph_dataset, batch_size=len(graph_data), shuffle=False)
    pro_graph = None
    for batchid, batch in enumerate(proGraph_loader):
        pro_graph = batch.x.to(device),batch.edge_index.to(device), batch.batch.to(device), batch.target_embedding.to(device)
    return pro_graph

class DTADataset(Dataset):          
    def __init__(self, smile_list, seq_list, label_list, mol_data = None, pro_data=None, drug_pretrain_dict = None):
        super(DTADataset,self).__init__()
        self.smile_list = smile_list
        self.seq_list = seq_list
        self.label_list = label_list
        self.smile_graph = mol_data
        self.target_graph = pro_data
        self.all_drug_mat_dict = self._process(drug_pretrain_dict)
    
    def _process(self, drug_pretrain_dict):
        mol2vec_dim = 300
        substructure_max_len = 100
        
        drug_batch_size = len(drug_pretrain_dict['length_dict'])
        drug_id = list(drug_pretrain_dict['length_dict'].keys())
        
        all_drug_mat = torch.zeros((drug_batch_size, substructure_max_len, mol2vec_dim), dtype=torch.float32)
        
        all_drug_mat_dict = {}
        
        for i, id in enumerate(drug_id):
            drug_mat = drug_pretrain_dict["mat_dict"][id]
            drug_mat_pad, drug_mask = matrix_pad(drug_mat, substructure_max_len) 
            
            all_drug_mat[i] = torch.from_numpy(drug_mat_pad)
            all_drug_mat_dict[id] = all_drug_mat[i]
            
        # print(all_drug_mat.shape)                   # 68, 100, 300
        
        return all_drug_mat_dict

    def len(self):
        return len(self.smile_list)

    def get(self, index):
        smile = self.smile_list[index]
        seq = self.seq_list[index]
        label =self.label_list[index]
        drug_clique_size, drug_features, drug_edge_index, edge_attr, _, drug_id = self.smile_graph[smile]
        seq_num = self.target_graph[seq][-1]
        # print(seq_num)

        # Wrapping graph data into the Data format supported by PyG (PyTorch Geometric).
        GCNData_smile = DATA.Data(x=torch.Tensor(np.array(drug_features)), edge_index=torch.LongTensor(drug_edge_index).transpose(1, 0), edge_attr=torch.BoolTensor(edge_attr), 
                                  y=torch.FloatTensor([label]), drug_embedding=torch.FloatTensor(self.all_drug_mat_dict[str(drug_id)]).unsqueeze(0), seq_num=torch.LongTensor([seq_num]))
        GCNData_smile.__setitem__('drug_clique_size', torch.LongTensor([drug_clique_size]))
        
        return GCNData_smile


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


def denseAffinityRefine(adj, k):
    refine_adj = np.zeros_like(adj)
    indexs1 = np.tile(np.expand_dims(np.arange(adj.shape[0]), 0), (k, 1)).transpose()
    indexs2 = np.argpartition(adj, -k, 1)[:, -k:]
    refine_adj[indexs1, indexs2] = adj[indexs1, indexs2]
    return refine_adj

def custom_collate(data):
    drug_batch = Batch.from_data_list( [item[0] for item in data])
    seq_batch = Batch.from_data_list([item[1] for item in data])
    return drug_batch,seq_batch


def get_mse(Y, P):
    Y = np.array(Y)
    P = np.array(P)
    return np.average((Y - P) ** 2)


def get_rm2(Y, P):
    r2 = r_squared_error(Y, P)
    r02 = squared_error_zero(Y, P)
    return r2 * (1 - np.sqrt(np.absolute(r2 ** 2 - r02 ** 2)))


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = sum((y_obs - y_obs_mean) * (y_pred - y_pred_mean)) ** 2
    y_obs_sq = sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / sum(y_pred ** 2)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = sum((y_obs - k * y_pred) ** 2)
    down = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (upp / down)


def model_evaluate(Y, P):

    return (get_mse(Y, P),
            get_rm2(Y, P))
