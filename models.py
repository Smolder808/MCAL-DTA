import torch
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool as gep
from torch_geometric.nn import global_max_pool as gmp
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self.MLP_v = nn.Linear(hidden_dim, hidden_dim)
        self.MLP_k = nn.Linear(hidden_dim, hidden_dim)
        self.MLP_q = nn.Linear(hidden_dim, hidden_dim)
        self.MLP_fusion = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.nhead = n_heads

        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hidden_dim / self.nhead)
    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.MLP_v(v).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.MLP_k(k).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.MLP_q(q).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        atted_feature = self.attention(v, k, q, mask)
        atted_feature = atted_feature.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_dim
        )

        atted_feature = self.MLP_fusion(atted_feature)

        return atted_feature

    def attention(self, v, k, q, mask):
        dim_k = q.size(-1)

        att_scores = torch.matmul(
            q, k.transpose(-2, -1)
        ) / math.sqrt(dim_k)

        if mask is not None:
            att_scores = att_scores.masked_fill(mask, -1e9)

        att_map = F.softmax(att_scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, v)

class MHA_Layer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(MHA_Layer, self).__init__()
        self.mhatt = MultiHeadAttention(hidden_dim, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y1, y2, y_mask=None):
        x = self.norm1(x+self.dropout1(
            self.mhatt(y1, y2, x, y_mask)
        ))

        return x

class Target_Encoder(nn.Module):
    def __init__(self, max_len, input_dim, device='cuda', hidden_dim=128):
        super(Target_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.drop = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0,2,1)
              
        for i, conv in enumerate(self.convs):
            conved = conv(self.drop(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved+h_map)* self.scale
            h_map = conved
        
        pool_map = self.max_pool(h_map).squeeze(-1)  # b,d
        h_map = h_map.permute(0,2,1)
        h_map = self.ln(h_map)    # b, len, d
        return pool_map, h_map
    
class Drug_Encoder(nn.Module):
    def __init__(self, max_len, input_dim, device='cuda', hidden_dim=128):
        super(Drug_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.drop = nn.Dropout(0.1)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 6)
        self.fc3 = nn.Linear(self.hidden_dim * 6, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.conv1 = nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)
        # self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)
        # self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        h_map = self.fc1(feat_map)
        h_map = h_map.permute(0,2,1)
        
        d_conved1 = (h_map + F.glu(self.drop(self.conv1(h_map)), dim=1)) * self.scale
        d_conved1_pool = self.max_pool(d_conved1).squeeze(-1)
        
        d_conved2 = (d_conved1 + F.glu(self.drop(self.conv1(d_conved1)), dim=1)) * self.scale
        d_conved2_pool = self.max_pool(d_conved2).squeeze(-1)
        
        d_conved3 = (d_conved2 + F.glu(self.drop(self.conv1(d_conved2)), dim=1)) * self.scale
        d_conved3_pool = self.max_pool(d_conved3).squeeze(-1)
        
        drug_embed = torch.cat((d_conved1_pool, d_conved2_pool, d_conved3_pool), 1)
        # print(drug_embed.shape)
        drug_embed = self.fc2(drug_embed)
        drug_embed = self.fc3(drug_embed)
        
        return drug_embed

class MultiDTA(nn.Module):
    def __init__(self, n_output, output_dim, num_features_xd, num_features_pro, device, pro_num, batch_size):
        super(MultiDTA, self).__init__()
        
        self.pro_num = pro_num
        self.output_dim = output_dim
        self.n_output = n_output
        self.mol2vec_dim = 300 
        self.protvec_dim = 1280 
        self.drug_max_len = 100
        self.prot_max_len = 1022

        self.drug_embed = Drug_Encoder(self.drug_max_len, self.mol2vec_dim, device)  # b, 100, 128
        self.prot_embed = Target_Encoder(self.prot_max_len, self.protvec_dim, device)  # b, 1022, 128
            
        # # GCN encoder used for extracting drug features.
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, output_dim)
        self.molFC1 = nn.Linear(output_dim + num_features_xd * 6, 512)
        self.molFC2 = nn.Linear(512, output_dim)
        
        # GCN encoder used for extracting protein features.
        self.proGconv1 = GCNConv(num_features_pro, output_dim)
        self.proGconv2 = GCNConv(output_dim, output_dim)
        self.proGconv3 = GCNConv(output_dim, output_dim)
        self.proFC1 = nn.Linear(output_dim, 1024)
        self.proFC2 = nn.Linear(1024, output_dim)
        
        # Batch normalization for drug
        self.bn1_dx1 = nn.BatchNorm1d(num_features_xd * 2)
        self.bn2_dx2 = nn.BatchNorm1d(num_features_xd * 4)
        self.bn3_dx3 = nn.BatchNorm1d(output_dim)
        
        # Batch normalization for protein
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)

        # classifier
        self.fc1 = nn.Linear(6 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(6 * output_dim, 1024)
        self.out = nn.Linear(512, self.n_output)
    
        self.mha_d_p = MHA_Layer(output_dim*2, 8, 0.1)
        self.mha_p_d = MHA_Layer(output_dim*2, 8, 0.1)

    def forward(self, mol_data, pro_data):
        d_x, d_edge_index, d_batch, d_edge_attr, d_smile_embedding, seq_num = mol_data.x, mol_data.edge_index, mol_data.batch, mol_data.edge_attr, mol_data.drug_embedding, mol_data.seq_num
        p_x, p_edge_index, p_batch, p_seq_embedding = pro_data
        # print(p_x.shape, p_edge_index.shape, p_batch.shape)

        # Extracting drug features
        d_x_1 = self.bn1_dx1(self.relu(self.molGconv1(d_x, d_edge_index)))
        d_x_2 = self.bn2_dx2(self.relu(self.molGconv2(d_x_1, d_edge_index)))
        d_x_3 = self.bn3_dx3(self.relu(self.molGconv3(d_x_2, d_edge_index)))
        d_x_concat = torch.cat([d_x_1,d_x_2,d_x_3],dim=1)
        dx_pool = gmp(d_x_concat, d_batch)
        d_x = self.dropout2(self.relu(self.molFC1(dx_pool)))
        d_x = self.dropout2(self.molFC2(d_x))
        # print(d_x.shape)


        # Extracting protein structural features from protein graphs.
        p_x = self.bn1(self.relu(self.proGconv1(p_x, p_edge_index)))
        # print(p_x.shape)
        p_x = self.bn2(self.relu(self.proGconv2(p_x, p_edge_index)))
        p_x = self.bn3(self.relu(self.proGconv3(p_x, p_edge_index)))
        p_x = gep(p_x, p_batch)
        p_x = self.dropout2(self.relu(self.proFC1(p_x)))
        p_x = self.dropout2(self.proFC2(p_x))
        p_x = p_x[seq_num]
        
        # Extracting seq features
        drug_embed = self.drug_embed(d_smile_embedding)
        # print(drug_embed.shape)
        prot_embed, _ = self.prot_embed(p_seq_embedding)  # 100 -> 128    
        prot_embed = prot_embed[seq_num] 

        enhance_graph_p = p_x * (F.softmax(p_x * prot_embed))
        # print(atten_graph_xt.shape)                            # 128
        enhance_seq_p = prot_embed * (F.softmax(p_x * prot_embed))
        # print(atten_conv_xt.shape)
        enhance_graph_d = d_x * (F.softmax(d_x * drug_embed))
        enhance_seq_d = drug_embed * (F.softmax(d_x * drug_embed))
        
        drug_embedding = torch.cat((d_x, drug_embed), 1)
        prot_embedding = torch.cat((p_x, prot_embed), 1)

        att_prot_feature = self.mha_p_d(prot_embedding.unsqueeze(1), drug_embedding.unsqueeze(1), drug_embedding.unsqueeze(1), None).squeeze(1)
        att_drug_feature = self.mha_d_p(drug_embedding.unsqueeze(1), prot_embedding.unsqueeze(1), prot_embedding.unsqueeze(1), None).squeeze(1)
        # print(final_target_embed.shape)
        
        # combination
        xc1 = torch.cat((drug_embedding, prot_embedding, enhance_graph_p, enhance_graph_d), 1)
        xc2 =  torch.cat((att_prot_feature, att_drug_feature, enhance_seq_p, enhance_seq_d), 1)   
                        
        # classifier
        xc1 = self.relu(self.fc1(xc1))
        xc2 = self.relu(self.fc3(xc2))
        xc = self.dropout1(self.relu(self.fc2(xc1 + xc2)))
        embedding = xc
        out = self.out(xc)

        return out#, embedding