from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS,AllChem,RDConfig,FragmentCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
import numpy as np
import networkx as nx
import pickle
import pandas as pd
from collections import Counter
import molvs as mv
import argparse
import os
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

# Note: All codes about substructure graph construction come from CGPDTA: https://github.com/fanqiejun479/CGPDTA 
#       We do some modification for our work
# Note: All codes about protein graph construction come from HiSIF-DTA: https://github.com/bixiangpeng/HiSIF-DTA
#       We do some modification for our work


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def mol_standard(mol):
    STANDARDIZER = mv.Standardizer()

    mol = STANDARDIZER.charge_parent(mol, skip_standardize=True)
    # Return the charge parent of a given molecule.
    # The charge parent is the uncharged version of the fragment parent.

    mol = STANDARDIZER.isotope_parent(mol, skip_standardize=True)  # 同位素
    # Return the isotope parent of a given molecule.
    # The isotope parent has all atoms replaced with the most abundant isotope for that element.

    mol = STANDARDIZER.stereo_parent(mol, skip_standardize=True)
    # Return the stereo parent of a given molecule.
    # The stereo parent has all stereochemistry information removed from tetrahedral centers and double bonds.

    mol = STANDARDIZER.standardize(mol)
    smi = Chem.MolToSmiles(mol)
    return smi

def cli_atom_features(atom): # 获取原子特征共101维，包括
    return one_of_k_encoding_unk(
        atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +one_of_k_encoding(
        atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetImplicitValence(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetFormalCharge(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + [atom.GetIsAromatic()] +[atom.IsInRing()]

def cli_features(mol,cli_list,nei_list): # 获取子结构特征共130维，其中原子特征101，子结构特征29维
    # 获取分子的所有原子特征
    features = []
    for atom in mol.GetAtoms():
        feature = cli_atom_features(atom)
        for i in range(len(feature)):
            feature[i] = feature[i] / sum(feature)
        features.append(feature)  # 各特征在总特征中的贡献占比

    # get pharmacophore vector according to smiles
    fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    keys_list = list(fdef.GetFeatureDefs().keys())

    mol_feats = fdef.GetFeaturesForMol(mol)

    # 获取分子的药效团特征并记录对应的原子序号
    buff = [0] * 27
    aloc_list = [[] for i in range(27)]
    for feat in mol_feats:
        # print(feat.GetFamily(), feat.GetType(), feat.GetAtomIds())
        # print(pos.x, pos.y, pos.z)
        feat_FT = feat.GetFamily() + '.' + feat.GetType()
        if feat_FT in keys_list:
            index = keys_list.index(feat_FT)
            buff[index] = buff[index] + 1
            aloc_list[index].append(list(feat.GetAtomIds()))  # 记录每个特征对应的原子序号

    # 获取子结构的连接键数
    nei_list = [i for j in nei_list for i in j]
    nei_dict = dict(Counter(nei_list))
    # 加上含金属离子的分子中分散的子结构连接键数
    for i in range(len(nei_list)):
        if i not in nei_dict.keys():
            nei_dict[i] = 0

    # 根据子结构分类获取每个子结构对应的特征
    struc_buff = []
    for j in range(len(cli_list)):
        cli = cli_list[j]
        cli_buff = [0] * 27
        cli.sort()

        # 计算子结构中的原子特征
        catom_feture = []
        atom_feature = features[0]
        for index in range(1, len(cli)):
            atom_feature2 = features[cli[index]]
            atom_feature = [atom_feature[i] + atom_feature2[i] for i in range(len(atom_feature))]  # 原子特征每维相加
        for i in range(len(atom_feature)):
            catom_feture.append(atom_feature[i] / sum(atom_feature))

        # 计算子结构的药效团信息
        for i in range(len(aloc_list)):
            for sub_cli in aloc_list[i]:
                sub_cli.sort()
                if list(set(cli) & set(sub_cli)) == sub_cli:  # 若子结构中有提供该特征的原子或副子结构，则记录
                    cli_buff[i] += 1

        ## 获取分子中所有的环 GetSymmSSSR(m)
        ssr = Chem.GetSymmSSSR(mol)
        ring_num = 0
        for ring in ssr:
            ring = list(ring)
            ring.sort()
            if list(set(cli) & set(ring)) == ring:  # 若子结构中有环结构，则记录
                ring_num += 1
        cli_buff.append(ring_num)  # 添加子结构环信息
        if j not in nei_dict.keys():
            cli_buff.append(0)
        else:
            cli_buff.append(nei_dict[j])  # 添加子结构连接键信息
        struc_buff.append(catom_feture + cli_buff)
    # struc_buff = np.array(struc_buff)

    return struc_buff

def atom_build_cli(mol,atom_list):
    # 零散原子对应的键
    bond_sup = []
    for atom in atom_list:
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() == atom and bond.GetEndAtomIdx() in atom_list:
                if [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] not in bond_sup:
                    bond_sup.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            elif bond.GetEndAtomIdx() == atom and bond.GetBeginAtomIdx() in atom_list:
                if [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] not in bond_sup:
                    bond_sup.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    # 零散原子构建子结构
    cli_sup = []
    index = 0
    for bond in bond_sup:
        if index == 0:
            cli_sup.append(bond)
            index += 1
        uniq = 0
        for i in range(len(cli_sup)):
            if bond[0] in cli_sup[i] and bond[1] in cli_sup[i]:
                uniq = 1
                continue
            elif bond[0] in cli_sup[i]:
                cli_sup[i].append(bond[1])
                uniq = 1
            elif bond[1] in cli_sup[i]:
                cli_sup[i].append(bond[0])
                uniq = 1
        if uniq == 0:
            cli_sup.append(bond)
    return cli_sup

def clique_extact(mol,type):
    # 获取rdkit官能团库，包含39种官能团
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(1, 6, fName)
    num = fparams.GetNumFuncGroups()
    func_g = []
    e_li = [5, 10, 11, 26, 27, 35, 37, 38]
    for i in range(num):
        try:
            if i in e_li:
                continue
            else:
                m = fparams.GetFuncGroup(i)
                s = mol_standard(m)
                if re.search(r'\*', s) is not None:
                    s = s.replace('*', '')
                func_g.append(s)
        except:
            continue

    # 通过BRICS算法拆分子结构
    if type == 'BRICS':
        fragments = BRICS.BRICSDecompose(mol) # smiles格式
        # print(fragments)

    # 拆分Murcko骨架
    if type == 'Murcko':
        core = MurckoScaffold.GetScaffoldForMol(mol)
        fragments = [Chem.MolToSmiles(core)]
        if fragments == []:
            fragments.append(Chem.MolToSmiles(mol))

    # 获取环中的原子序号
    atom_ring = []
    for atom in mol.GetAtoms():
        if atom.IsInRing() == True:
            atom_ring.append(atom.GetIdx())

    # 进行子结构搜索,记录子结构对应的原子序号，针对BRICS结果进行一些处理
    BRICS_id = ['[1*]','[2*]','[3*]','[4*]','[5*]','[6*]','[7*]','[8*]','[9*]','[10*]','[11*]','[12*]','[13*]','[14*]','[15*]','[16*]']
    cli_list =[]
    for smi in fragments:
        try:
            if type == 'BRICS':
                for id in BRICS_id:
                    if re.search(id, smi) is not None:
                        smi = smi.replace(id,'') #断点替换,[12*]S(=O)(=O)c1cc([16*])c(O)cc1O  S(=O)(=O)c1ccc(O)cc1O
                        if re.search('()',smi) is not None:
                            smi = smi.replace('()','')
            pattern = Chem.MolFromSmiles(smi, sanitize=False)
            results = mol.GetSubstructMatches(pattern)

            # 在子结构拆分的基础上提取官能团
            fg_loc = []
            for fg in func_g:
                sub_pattern = Chem.MolFromSmiles(fg, sanitize=False)
                sub_results = pattern.GetSubstructMatches(sub_pattern)
                # 将子结构中的子结构序号转化为原分子中的子结构序号
                for loc in results:
                    for sub_loc in sub_results:
                        tem_loc =[]
                        for i in sub_loc:
                            tem_loc.append(loc[i])
                        # 排除破坏环结果的官能团切割
                        if len(set(atom_ring).intersection(set(tem_loc))) != 0:
                            fg_loc.append(loc)
                        elif tem_loc not in fg_loc:
                            fg_loc.append(tem_loc)
            if fg_loc == []:
                for loc in results:
                    if list(loc) not in cli_list:
                        cli_list.append(list(loc))
            else:
                for loc in fg_loc:
                    if list(loc) not in cli_list:
                        cli_list.append(list(loc))
        except:
            pass
            continue
    # 结构去重，去除A为B子集的子结构
    loc_list = []
    for i in range(len(cli_list)):
        for j in range(i + 1, len(cli_list)):
            if list(set(list(cli_list[i])) & set(list(cli_list[j]))) == list(cli_list[i]) \
                    or list(set(list(cli_list[i])) & set(list(cli_list[j]))) == list(cli_list[j]):
                if len(cli_list[i]) >= len(cli_list[j]):
                    loc_list.append(j)
                else:
                    loc_list.append(i)
    filter1 = []  # 删除含重复原子的结构
    for i in range(len(cli_list)):
        if i not in loc_list:
            filter1.append(cli_list[i])
    # 去除元素部分重叠的较短子结构，记录未重叠元素
    loc_list = []
    atom_list = []
    for i in range(len(filter1)):
        for j in range(i + 1, len(filter1)):
            if list(set(list(filter1[i])) & set(list(filter1[j]))) != []:
                if len(filter1[i]) >= len(filter1[j]):
                    loc_list.append(j)
                    diff = list(set(list(filter1[j])).difference(set(list(filter1[i]))))
                    for a in diff:
                        if a not in atom_list:
                            atom_list.append(a)
                else:
                    loc_list.append(i)
                    diff = list(set(list(filter1[i])).difference(set(list(filter1[j]))))
                    for a in diff:
                        if a not in atom_list:
                            atom_list.append(a)
    filter2 = []  # 删除含部分重复原子的结构
    for i in range(len(filter1)):
        if i not in loc_list:
            filter2.append(filter1[i])
    atom_fil = [i for j in filter2 for i in j]
    atom_list = list(set(atom_list).difference(set(atom_fil)))
    # 加入分子中的零散原子
    atom_fil2 = []
    for atom in mol.GetAtoms():
        atom_fil2.append(atom.GetIdx())
    miss_list = list(set(atom_fil2).difference(set(atom_fil)))
    atom_list = atom_list + miss_list
    filter3 = []
    for cli in filter2:
        if len(cli) == 1:
            atom_list.append(cli[0])
        else:
            filter3.append(cli)
    atom_list.sort()
    cli_sup = atom_build_cli(mol,atom_list)

    atom_sup = [i for j in cli_sup for i in j]
    diff = list(set(atom_list).difference(set(atom_sup)))
    for a in diff:
        cli_sup.append([a])

    cli_list = filter3+cli_sup
    return cli_list

def get_edge_attr(mol,edge_list):#键类型，立体化学类型，是否为共轭键
    edge_attr = []
    for i in edge_list:
        bond = mol.GetBondWithIdx(i)
        edge_attr.append(
            one_of_k_encoding(bond.GetBondType().name, ['UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
                                                        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF',
                                                        'FIVEANDAHALF','AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE',
                                                        'DATIVE','DATIVEL','DATIVER','OTHER','ZERO']) +
            one_of_k_encoding(bond.GetStereo().name,['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS']) +
            [bond.GetIsConjugated()]
        )
    # edge_attr = np.array(edge_attr)

    return edge_attr

def neighbor_extact(mol,cli_list):
    a_edges = []
    edge_list = []
    # 通过键建立邻接列表
    for bond in mol.GetBonds():
        index = 0
        for cli in cli_list:
            if bond.GetBeginAtomIdx() in cli and bond.GetEndAtomIdx() in cli:
                index = 1
        if index == 0:
            if [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] not in a_edges:
                a_edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                edge_list.append(bond.GetIdx())
    edge_attr = get_edge_attr(mol,edge_list)
    # 构建子结构邻接表
    nei_list = []
    for j in range(len(a_edges)):
        for i in range(len(cli_list)):
            if a_edges[j][0] in cli_list[i]:
                b_loc = i
            if a_edges[j][1] in cli_list[i]:
                e_loc = i
        nei_list.append([b_loc,e_loc])
    return nei_list,edge_attr

def smile_to_clique_graph(smile, c_type, drug_id):
    mol = Chem.MolFromSmiles(smile)
    if mol == None:
        print(f"{smile} is none")
    clique_list = clique_extact(mol, c_type)
    if len(clique_list) <= 1:# 无法提取子结构的分子使用原子级替代
        c_size = mol.GetNumAtoms()
        clique_list = [[i] for j in clique_list for i in j]
        edge_index = []
        edge_list = []
        for bond in mol.GetBonds():
            edge_list.append(bond.GetIdx())
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_attr = get_edge_attr(mol,edge_list)
        features = cli_features(mol,clique_list,edge_index)
    else:
        n_atoms = mol.GetNumAtoms()
        n_atoms_2 = [i for j in clique_list for i in j]
        g_num = len(list(set(n_atoms_2)))
        if n_atoms != g_num:
            raise Exception('注意！！！原分子共有{}个原子，拆分后还剩{}个原子'.format(n_atoms, g_num))
        c_size = len(clique_list)
        edge_index,edge_attr = neighbor_extact(mol,clique_list)
        features = cli_features(mol,clique_list,edge_index)

    return c_size, features, edge_index, edge_attr, clique_list, drug_id

def smile_to_graph(smile,pdb):
    mol = Chem.MolFromSmiles(smile)
    if mol == None:
        mol = Chem.MolFromMol2File('data/PDBBind/raw/' + pdb + '/' + pdb + '_ligand.mol2')
    c_size = mol.GetNumAtoms()
    atom_list = []
    for atom in mol.GetAtoms():
        atom_list.append([atom.GetIdx()])
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()  # 生成有向图
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    _,edge_attr = neighbor_extact(mol,atom_list)
    features = cli_features(mol, atom_list, edge_index)

    return c_size, features, edge_index,edge_attr


def dic_normalize(dic):
    """
    Feature normalization
    :param dic: A dict for describing residue feature
    :return: Normalizied feature
    """
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

# predefined resdiue type space
pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']
# predefined aliphatic resdiue
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
# predefined aromatic resdiue
pro_res_aromatic_table = ['F', 'W', 'Y']
# predefined polar neutral resdiue
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
# predefined acidic resdiue
pro_res_acidic_charged_table = ['D', 'E']
# predefined basic resdiue
pro_res_basic_charged_table = ['H', 'K', 'R']

# predefined resdiue weigh space
res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

#Dissociation constant for the –COOH group
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
#Dissociation constant for the –NH3 group
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
#Dissociation constant for any other group in the molecule
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
#pH at the isoelectric point
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

#Hydrophobicity of the residue(pH = 2)
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

#Hydrophobicity of the residue (pH = 7)
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

#normalization
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    # Generating initial residual descriptors based on atomic properties.
    if residue not in pro_res_table:
        residue = 'X'
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    """
    Generate feature descriptors for all residues in one protein
    :param pro_seq: protein sequence
    :return: the initial residue feature descriptors (33 dim) for pro_seq
    """
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding_unk(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def seq_to_graph(num, pro_id, seq, db):
    """
    Convert a protein sequence into topo graph (i.e., feature matrix and adjacency matrix )
    :param pro_id: the UniProt id of protein
    :param seq: protein sequence
    :param db: dataset name
    :return: nodenum, feature matrix, and adjacency matrix(sparse).
    """
    sparse_edge_matrix = np.load(f'../data/{db}/contact_map/{pro_id}.npy') # reading contact map
    edge_index = np.argwhere(sparse_edge_matrix == 1).transpose(1, 0)

    c_size = len(seq)
    features = seq_feature(seq)
    return c_size, features, edge_index, pro_id, num

def seq_to_graph_fold(num, pro_id, seq, db):
    """
    Convert a protein sequence into topo graph (i.e., feature matrix and adjacency matrix )
    :param pro_id: the UniProt id of protein
    :param seq: protein sequence
    :param db: dataset name
    :return: nodenum, feature matrix, and adjacency matrix(sparse).
    """
    sparse_edge_matrix = np.load(f'./data/{db}/contact_map/{pro_id}.npy') # reading contact map
    edge_index = np.argwhere(sparse_edge_matrix == 1).transpose(1, 0)

    c_size = len(seq)
    features = seq_feature(seq)
    return c_size, features, edge_index, pro_id, num

def construct_prot_graph_fold(path, opt, fold, dataset):
    seq_dict = {}
    with open(f'{path}/fold_{fold}_{opt}_prot_dict.txt', 'r') as file:
        for line in file.readlines():
            line = line.lstrip('>').strip().split('\t')
            seq_dict[line[1]] = line[0]
    seq_graph = {}
    num = 0
    for pro_id, seq in seq_dict.items():
        g = seq_to_graph_fold(num, pro_id, seq, dataset)
        # print(g)
        seq_graph[seq] = g
        num = num +1
    with open(f'{path}/fold_{fold}_{opt}_prot_data.pkl', 'wb') as seq_file:
        pickle.dump(seq_graph, seq_file)
    print('protein graph is constructed successfully!')
    


def construct_graph(args):
    """
    Construct topological graph for protein (pro_data) and drug (mol_data), respectively.
    No value is returned, but the corresponding graph data file is eventually generated.
    :param dataset: dataset name
    :return: None
    """
    print('Construct graph for', args.dataset)
    # 1. generate drug substructure graph dict.
    clique_type = 'BRICS'
    df = pd.read_csv(f'../data/{args.dataset}/{args.dataset}_iso_drug.csv')
    drug_ids, compound_iso_smiles = list(df['drug_id']), list(df['drug_iso_seq'])
    smile_graph = {}
    for drug_id, smile in tqdm(zip(drug_ids, compound_iso_smiles), desc='generating graph dict...',
                        total=len(list(compound_iso_smiles))):
        g = smile_to_clique_graph(smile, clique_type, drug_id)
        smile_graph[smile] = g
    print('drug graph is constructed successfully!')

    # 2.generate protein graph dict.
    seq_dict = {}
    with open(f'../data/{args.dataset}/{args.dataset}_dict.txt', 'r') as file:
        for line in file.readlines():
            line = line.lstrip('>').strip().split('\t')
            seq_dict[line[1]] = line[0]
    seq_graph = {}
    num = 0
    for pro_id, seq in seq_dict.items():
        g = seq_to_graph(num, pro_id, seq, args.dataset)
        # print(g)
        seq_graph[seq] = g
        num = num +1
    print('protein graph is constructed successfully!')


    # 3. Serialized graph data
    with open(f'../data/{args.dataset}/sub_mol_data.pkl', 'wb') as smile_file:
        pickle.dump(smile_graph, smile_file)
    with open(f'../data/{args.dataset}/pro_data.pkl', 'wb') as seq_file:
        pickle.dump(seq_graph, seq_file)

def main(args):
    construct_graph(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default= 'kiba', help='dataset name',choices=['davis','kiba'])
    args = parser.parse_args()
    main(args)
