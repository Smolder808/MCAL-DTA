import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.create_graph_data import construct_prot_graph_fold
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--seed', type=int, default=72)         # 72
    parser.add_argument('--fold', type=int, default=5)
    parser.add_argument('--running_set', type=str, help='warm; new_drug; new_prot; new_pair', default='warm')
    args, _ = parser.parse_known_args()
    # args = parser.parse_args()

    dataset = args.dataset
    print("Dataset:", dataset)

    random_seed = args.seed
    print(f'random seed:', random_seed)
    print(f'running set:', args.running_set)
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    dataset_path = 'data/' + args.dataset + '/'
    df_train = pd.read_csv(f'{dataset_path}' + 'train.csv')
    # print(df_train.shape)
    df_test = pd.read_csv(f'{dataset_path}' + 'test.csv') 
    # print(df_test.shape)
    df_all = df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    # print(df_all.shape)
    df_train_pairs = df_train.loc[:,['compound_iso_smiles', 'target_sequence', 'affinity']]
    # print(df_train_pairs)
    df_train_pairs = df_train_pairs.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_all_pairs = df_all.loc[:,['compound_iso_smiles', 'target_sequence', 'affinity']]
    df_all_pairs = df_all_pairs.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # print(df_all_pairs)
    # print(df_all_pairs.shape)
    df_all_drugs = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_iso_drug.csv')
    df_all_drugs = df_all_drugs.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    all_drug_ids, all_compound_iso_smiles = list(df_all_drugs['drug_id']), list(df_all_drugs['drug_iso_seq'])
    df_all_prots = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_prots.csv')
    df_all_prots = df_all_prots.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    all_prot_ids, all_prot_seq = list(df_all_prots['prot_id']), list(df_all_prots['prot_seq'])
    # print(df_all_prots.shape)
    
    k = args.fold
    if args.running_set == 'warm':
        path = f'./data_split/{dataset}/{args.running_set}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Create path {path}')
        fold_size = len(df_train_pairs) // k
        for i in range(k):
            print(i)
            valid_start = i * fold_size
            # print(valid_start)
            if i != k - 1 and i != 0:
                valid_end = (i + 1) * fold_size
                validset = df_train_pairs[valid_start:valid_end]
                print(validset.shape)
                trainset = pd.concat([df_train_pairs[0:valid_start], df_train_pairs[valid_end:]])
                print(trainset.shape)
            elif i == 0:
                valid_end = fold_size
                validset = df_train_pairs[valid_start:valid_end]
                print(validset.shape)
                trainset = df_train_pairs[valid_end:]
                print(trainset.shape)
            else:
                validset = df_train_pairs[valid_start:]
                print(validset.shape)
                trainset = df_train_pairs[0:valid_start]
                print(trainset.shape)
            
            # split training-set and valid-set
            print(f'train:{len(trainset)}, valid:{len(validset)}')
            trainset.to_csv(f'{path}/fold_{i}_train.csv', index=False, header=True) 
            validset.to_csv(f'{path}/fold_{i}_valid.csv', index=False, header=True)
            
    elif args.running_set == 'new_drug':
        path = f'./data_split/{dataset}/{args.running_set}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Create path {path}')
        drugs_num = len(df_all_drugs)
        # print(drugs_num)
        fold_size = drugs_num // k
        for i in range(k):
            # print(i)
            test_start = i * fold_size
            if i == k-1:
                test_end = drugs_num
            else:
                test_end = (i + 1) * fold_size      
            
            drugs_smiles = all_compound_iso_smiles[test_start:test_end]
            # print(len(drugs_smiles))
            testset = df_all_pairs[df_all_pairs['compound_iso_smiles'].isin(drugs_smiles)]
            # print(testset.shape)
            tvset = df_all_pairs[~df_all_pairs['compound_iso_smiles'].isin(drugs_smiles)]

            trainset, validset = train_test_split(tvset, test_size=0.2, random_state=0)

            print(f'train:{len(trainset)}, valid:{len(validset)}, test:{len(testset)}')
            trainset.to_csv(f'{path}/fold_{i}_train.csv', index=False, header=True) 
            validset.to_csv(f'{path}/fold_{i}_valid.csv', index=False, header=True)
            testset.to_csv(f'{path}/fold_{i}_test.csv', index=False, header=True)
            
    elif args.running_set == 'new_prot':
        path = f'./data_split/{dataset}/{args.running_set}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Create path {path}')
        prots_num = len(df_all_prots)
        fold_size = prots_num // k
        for i in range(k):
            # print(i)
            test_start = i * fold_size
            if i == k-1:
                test_end = prots_num
            else:
                test_end = (i + 1) * fold_size      
            test_prot = all_prot_seq[test_start:test_end]
            testset = df_all_pairs[df_all_pairs['target_sequence'].isin(test_prot)]
            # print(testset.shape)
            tvset = df_all_pairs[~df_all_pairs['target_sequence'].isin(test_prot)]
            trainset, validset = train_test_split(tvset, test_size=0.2, random_state=0)
            # print(trainset)
            train_prot = trainset['target_sequence'].drop_duplicates()
            # print(train_prot.shape)
            valid_prot = validset['target_sequence'].drop_duplicates()
            
            train_prot_id = df_all_prots[df_all_prots['prot_seq'].isin(train_prot)]['prot_id'].drop_duplicates()
            train_prot_seq = df_all_prots[df_all_prots['prot_id'].isin(train_prot_id)]['prot_seq'].drop_duplicates()
            
            valid_prot_id = df_all_prots[df_all_prots['prot_seq'].isin(valid_prot)]['prot_id'].drop_duplicates()
            valid_prot_seq = df_all_prots[df_all_prots['prot_id'].isin(valid_prot_id)]['prot_seq'].drop_duplicates()
            
            test_prot_id = df_all_prots[df_all_prots['prot_seq'].isin(test_prot)]['prot_id'].drop_duplicates()
            test_prot_seq = df_all_prots[df_all_prots['prot_id'].isin(test_prot_id)]['prot_seq'].drop_duplicates()
            
            train_prot_id_seq = pd.concat([train_prot_id.reset_index(drop=True), train_prot_seq.reset_index(drop=True)], axis=1)
            valid_prot_id_seq = pd.concat([valid_prot_id.reset_index(drop=True), valid_prot_seq.reset_index(drop=True)], axis=1)
            test_prot_id_seq = pd.concat([test_prot_id.reset_index(drop=True), test_prot_seq.reset_index(drop=True)], axis=1)
            
            train_prot_id_seq['fasta_format'] = '>' + train_prot_id_seq['prot_seq'] + '\t' + train_prot_id_seq['prot_id']
            train_prot_id_seq['fasta_format'].to_csv(f'{path}/fold_{i}_train_prot_dict.txt', index=False, header=False)
            construct_prot_graph_fold(path, 'train', i, dataset)
            
            valid_prot_id_seq['fasta_format'] = '>' + valid_prot_id_seq['prot_seq'] + '\t' + valid_prot_id_seq['prot_id']
            valid_prot_id_seq['fasta_format'].to_csv(f'{path}/fold_{i}_valid_prot_dict.txt', index=False, header=False)
            construct_prot_graph_fold(path, 'valid', i, dataset)
            
            test_prot_id_seq['fasta_format'] = '>' + test_prot_id_seq['prot_seq'] + '\t' + test_prot_id_seq['prot_id']
            test_prot_id_seq['fasta_format'].to_csv(f'{path}/fold_{i}_test_prot_dict.txt', index=False, header=False)
            construct_prot_graph_fold(path, 'test', i, dataset)
            
            # print(trian_prot_id_seq)
            print(f'train_prot:{len(train_prot)}, valid_prot:{len(valid_prot)}, test_prot:{len(test_prot)}')

            print(f'train:{len(trainset)}, valid:{len(validset)}, test:{len(testset)}')
            trainset.to_csv(f'{path}/fold_{i}_train.csv', index=False, header=True) 
            validset.to_csv(f'{path}/fold_{i}_valid.csv', index=False, header=True)
            testset.to_csv(f'{path}/fold_{i}_test.csv', index=False, header=True)
            
    elif args.running_set == 'new_pair':
        path = f'./data_split/{dataset}/{args.running_set}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f'Create path {path}')
        for seed_i in range(k):    
            drugs_smiles = df_all_drugs.sample(frac=0.4, random_state=seed_i).reset_index(drop=True)
            prots_seq = df_all_prots.sample(frac=0.4, random_state=seed_i).reset_index(drop=True)
            # print(drugs_smiles.shape, prots_seq.shape)
            
            test_prot = prots_seq['prot_seq'].drop_duplicates()    
            test_prot_id = df_all_prots[df_all_prots['prot_seq'].isin(test_prot)]['prot_id'].drop_duplicates()
            # print(test_prot_id)
            test_prot_seq = df_all_prots[df_all_prots['prot_id'].isin(test_prot_id)]['prot_seq'].drop_duplicates()

            testset = df_all_pairs[(df_all_pairs['compound_iso_smiles'].isin(drugs_smiles['drug_iso_seq'])) & (df_all_pairs['target_sequence'].isin(prots_seq['prot_seq']))]
            trainset = df_all_pairs[(~df_all_pairs['compound_iso_smiles'].isin(drugs_smiles['drug_iso_seq'])) & (~df_all_pairs['target_sequence'].isin(prots_seq['prot_seq']))]  
            train_prot = trainset['target_sequence'].drop_duplicates()
            
            train_prot_id = df_all_prots[df_all_prots['prot_seq'].isin(train_prot)]['prot_id'].drop_duplicates()
            train_prot_seq = df_all_prots[df_all_prots['prot_id'].isin(train_prot_id)]['prot_seq'].drop_duplicates()
            
            train_prot_id_seq = pd.concat([train_prot_id.reset_index(drop=True), train_prot_seq.reset_index(drop=True)], axis=1)
            test_prot_id_seq = pd.concat([test_prot_id.reset_index(drop=True), test_prot_seq.reset_index(drop=True)], axis=1)
            # print(test_prot_id_seq)
            
            train_prot_id_seq['fasta_format'] = '>' + train_prot_id_seq['prot_seq'] + '\t' + train_prot_id_seq['prot_id']
            train_prot_id_seq['fasta_format'].to_csv(f'{path}/fold_{seed_i}_train_prot_dict.txt', index=False, header=False)
            construct_prot_graph_fold(path, 'train', seed_i, dataset)
            
            test_prot_id_seq['fasta_format'] = '>' + test_prot_id_seq['prot_seq'] + '\t' + test_prot_id_seq['prot_id']
            test_prot_id_seq['fasta_format'].to_csv(f'{path}/fold_{seed_i}_test_prot_dict.txt', index=False, header=False)
            construct_prot_graph_fold(path, 'test', seed_i, dataset)
            
            merged_df = pd.merge(testset, trainset, on=['compound_iso_smiles', 'target_sequence'], how='outer', indicator=True)
            validset = df_all_pairs[~df_all_pairs.index.isin(merged_df.index)]
            valid_prot = validset['target_sequence'].drop_duplicates()
            
            print(len(train_prot), len(test_prot), len(valid_prot))
            # print(train_prot.shape,valid_prot.shape)
            # print(len(testset), len(trainset), len(validset))  
            
            print(f'train:{len(trainset)}, valid:{len(validset)}, test:{len(testset)}')
            trainset.to_csv(f'{path}/fold_{seed_i}_train.csv', index=False, header=True) 
            validset.to_csv(f'{path}/fold_{seed_i}_valid.csv', index=False, header=True)
            testset.to_csv(f'{path}/fold_{seed_i}_test.csv', index=False, header=True)


