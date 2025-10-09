import numpy as np 
import pandas as pd 
import pickle
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
from tqdm import tqdm

def get_mol2vec(mol2vec_dir, df_dir, db_name, sep=' ', header=None, col_names=['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label'], embedding_dimension=300, is_debug=False):
    # This function code come from LLMDTA:  https://github.com/Chris-Tang6/LLMDTA
    mol2vec_model = word2vec.Word2Vec.load(mol2vec_dir)
    
    df = pd.read_csv(df_dir, header=header, sep=sep)
    df.columns = col_names
    df.drop_duplicates(subset='drug_id', inplace=True)    
    drug_ids = df['drug_id'].tolist()
    drug_seqs = df['drug_smile'].tolist()
    
    emb_dict = {}
    emb_mat_dict = {}
    length_dict = {}
    
    percent_unknown = []
    bad_mol = 0
    
    # get pretrain feature
    for idx in tqdm(range(len(drug_ids))):
        flag = 0
        mol_miss_words = 0
        
        drug_id = str(drug_ids[idx])
        molecule = Chem.MolFromSmiles(drug_seqs[idx])
        length_dict
        try:
            # Get fingerprint from molecule
            sub_structures = mol2alt_sentence(molecule, 2)
        except Exception as e: 
            if is_debug: 
                print (e)
            percent_unknown.append(100)
            continue    
                
        emb_mat = np.zeros((len(sub_structures), embedding_dimension))
        length_dict[drug_id] = len(sub_structures)
                
        for i, sub in enumerate(sub_structures):
            # Check to see if substructure exists
            try:
                emb_dict[drug_id] = emb_dict.get(drug_id, np.zeros(embedding_dimension)) + mol2vec_model.wv[sub]  
                emb_mat[i] = mol2vec_model.wv[sub]  
            # If not, replace with UNK (unknown)
            except Exception as e:
                if is_debug : 
                    print ("Sub structure not found")
                    print (e)
                emb_dict[drug_id] = emb_dict.get(drug_id, np.zeros(embedding_dimension)) + mol2vec_model.wv['UNK']
                emb_mat[i] = mol2vec_model.wv['UNK']                
                flag = 1
                mol_miss_words = mol_miss_words + 1        
        emb_mat_dict[drug_id] = emb_mat
        
        percent_unknown.append((mol_miss_words / len(sub_structures)) * 100)
        if flag == 1:
            bad_mol = bad_mol + 1
            
    print(f'All Bad Mol: {bad_mol}, Avg Miss Rate: {sum(percent_unknown)/len(percent_unknown)}%')
        
    dump_data = {
        "dataset": db_name,
        "vec_dict": emb_dict,
        "mat_dict": emb_mat_dict,
        "length_dict": length_dict
    }    
    with open(f'./{db_name}_drug_pretrain.pkl', 'wb+') as f:
        pickle.dump(dump_data, f)

if __name__ == '__main__':
    mol2vec_dir = '../models/model_300dim.pkl'
    db_name = 'kiba_train'
    df_dir = '../data/kiba/kiba_iso_drug.csv'

    get_mol2vec(mol2vec_dir, df_dir, db_name, sep=',', header=0, col_names=['drug_id', 'drug_smile'])