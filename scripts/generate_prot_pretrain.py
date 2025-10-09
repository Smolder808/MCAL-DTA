import torch
import esm
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

torch.set_num_threads(2)

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results 
print("success")

def get_esm_pretrain(model, df_dir, db_name, sep=' ', header=None, col_names=['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label']):
    # This function code come from LLMDTA:  https://github.com/Chris-Tang6/LLMDTA
    df = pd.read_csv(df_dir, sep=sep)
    df.columns = col_names
    df.drop_duplicates(subset='prot_id', inplace=True)
    prot_ids = df['prot_id'].tolist()
    prot_seqs = df['prot_seq'].tolist()
    data = []
    prot_size = len(prot_ids)
    for i in range(prot_size):
        seq_len = min(len(prot_seqs[i]),1022)
        data.append((prot_ids[i], prot_seqs[i][:seq_len]))
    
    emb_dict = {}
    emb_mat_dict = {}
    length_target = {}

    for d in tqdm(data):
        prot_id = d[0]
        batch_labels, batch_strs, batch_tokens = batch_converter([d])
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33].numpy()

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

        emb_dict[prot_id] = sequence_representations[0]
        emb_mat_dict[prot_id] = token_representations[0]
        length_target[prot_id] = len(d[1])
    
    dump_data = {
        "dataset": db_name,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_target
    }    
    with open(f'./{db_name}_esm_pretrain.pkl', 'wb+') as f:
        pickle.dump(dump_data, f)  
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'kiba', help='dataset name',choices=['davis','kiba'])
    args = parser.parse_args()
    db_name = args.dataset
    df_dir = f'../data/{args.dataset}/{args.dataset}_prots.csv'
    col_names = ['prot_id', 'prot_seq']
    get_esm_pretrain(model, df_dir, db_name, sep=',', header=0, col_names=col_names)