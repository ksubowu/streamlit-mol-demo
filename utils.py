import copy
import json
import math
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdDepictor
import matplotlib.pyplot as plt
import re
##################### MolScribe#################################################################################### 
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

     
COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}

#helper function
def view_box_center(bond_bbox,heavy_centers):
    fig, ax = plt.subplots(figsize=(10, 10))
    # 绘制矩形框 (boxes)
    for box in bond_bbox:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

    # 绘制圆形 (centers)
    for center in heavy_centers:
        x, y = center
        circle = Circle((x, y), radius=5, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(circle)

    # 设置坐标轴范围（根据数据自动调整）
    x_min = min(bond_bbox[:, 0].min(), heavy_centers[:, 0].min()) - 10
    x_max = max(bond_bbox[:, 2].max(), heavy_centers[:, 0].max()) + 10
    y_min = min(bond_bbox[:, 1].min(), heavy_centers[:, 1].min()) - 10
    y_max = max(bond_bbox[:, 3].max(), heavy_centers[:, 1].max()) + 10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 设置标题和标签
    ax.set_title("Boxes and Centers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # 显示图像
    plt.gca().set_aspect('equal', adjustable='box')  # 保持比例
    plt.grid(True, linestyle='--', alpha=0.7)

def molIDX(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)  #映射
        # print(i)
    return mol

def molIDX_del(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(0)  #映射
        print(i)
    return mol
from det_engine import ABBREVIATIONS



def Val_extract_atom_info(error_message):
    """
    从错误信息中提取 atomid, atomType 和 valence。
    :param error_message: 错误信息字符串
    :return: (atomid, atomType, valence) 元组
    """
    # 定义正则表达式来提取原子信息
    pattern = r"Explicit valence for atom # (\d+) (\w), (\d+)"
    pattern2 =r"Explicit valence for atom # (\d+) (\w) "
    # print(type(error_message))
    if not isinstance(error_message, type('strs')):
        error_message=str(error_message)
    match = re.search(pattern, error_message)
    match2 = re.search(pattern2, error_message)
    if match:
        # 提取 atomid, atomType 和 valence
        atomid = int(match.group(1))  # 原子索引
        atomType = match.group(2)     # 原子类型
        valence = int(match.group(3)) # 当前价态
        return atomid, atomType, valence
    elif match2:
        atomid = int(match2.group(1))  # 原子索引
        atomType = match2.group(2)     # 原子类型
        # valence = int(match2.group(3)) # 当前价态
        return atomid, atomType, None
        
    else:
        raise ValueError("无法从错误信息中提取原子信息")
    
def calculate_charge_adjustment(atom_symbol, current_valence):
    """
    计算需要调整的电荷，根据反馈的原子符号和当前价态。
    :param atom_symbol: 原子符号（如 "C"）
    :param current_valence: 当前价态（如 5）
    :return: 需要添加的电荷数（正数表示负电荷，负数表示正电荷）
    """
    if atom_symbol not in VALENCES:
        raise ValueError(f"未知的原子符号: {atom_symbol}")

    # 查找该元素的最大价态
    max_valence = max(VALENCES[atom_symbol])
    if current_valence is None:
        current_valence=max_valence
    # 如果当前价态大于最大允许价态，需要调整电荷
    if current_valence > max_valence:
        # 需要添加的负电荷数
        charge_adjustment = current_valence - max_valence
        return charge_adjustment 
    else:
        # 当前价态已经符合最大允许价态，不需要调整
        return 0

from rdkit.Chem import rdchem, RWMol, CombineMols

def expandABB(mol,ABBREVIATIONS, placeholder_atoms):
    mols = [mol]
    # **第三步: 替换 * 并合并官能团**
    # 逆序遍历 placeholder_atoms，确保删除后不会影响后续索引
    for idx in sorted(placeholder_atoms.keys(), reverse=True):
        group = placeholder_atoms[idx]  # 获取官能团名称
        # print(idx, group)
        submol = Chem.MolFromSmiles(ABBREVIATIONS[group].smiles)  # 获取官能团的子分子
        submol_rw = RWMol(submol)  # 让 submol 变成可编辑的 RWMol
        anchor_atom_idx = 0  # 选择 `submol` 的第一个原子作为连接点 as defined in ABBREVIATIONS
        # **1. 复制主分子**
        new_mol = RWMol(mol)
        # **2. 计算 `*` 在 `new_mol` 中的索引**
        placeholder_idx = idx
        # **3. 记录 `*` 原子的邻居**
        neighbors = [nb.GetIdx() for nb in new_mol.GetAtomWithIdx(placeholder_idx).GetNeighbors()]
        # **4. 断开 `*` 的所有键**
        bonds_to_remove = []  # 记录要断开的键
        for bond in new_mol.GetBonds():
            if bond.GetBeginAtomIdx() == placeholder_idx or bond.GetEndAtomIdx() == placeholder_idx:
                bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in bonds_to_remove:
            new_mol.RemoveBond(bond[0], bond[1])
        # **5. 删除 `*` 原子**
        new_mol.RemoveAtom(placeholder_idx)
        # **6. 重新计算 `neighbors`（删除后索引变化）**
        new_neighbors = []
        for neighbor in neighbors:
            if neighbor < placeholder_idx:
                new_neighbors.append(neighbor)
            else:
                new_neighbors.append(neighbor - 1)  # 因为删除了一个原子，所有索引 -1
        # **7. 合并 `submol`**
        new_mol = RWMol(CombineMols(new_mol, submol_rw))

        # **8. 计算 `submol` 的第一个原子在合并后的位置**
        new_anchor_idx = new_mol.GetNumAtoms() - len(submol_rw.GetAtoms()) + anchor_atom_idx

        # **9. 重新连接官能团**
        for neighbor in new_neighbors:
            # print(neighbor, new_anchor_idx, "!!")
            new_mol.AddBond(neighbor, new_anchor_idx, Chem.BondType.SINGLE)
            a1=new_mol.GetAtomWithIdx(neighbor)
            a2=new_mol.GetAtomWithIdx(new_anchor_idx)
            a1.SetNumRadicalElectrons(0)
            a2.SetNumRadicalElectrons(0)## 将自由基电子数设为 0,as has added new bond
        # **10. 更新主分子**
        mol = new_mol
        mols.append(mol)
    # # 遍历分子中的每个原子
    # for atom in mols[-1].GetAtoms(): NOTE considering original image has the RadicalElectrons
    #     atom_idx = atom.GetIdx()  # 原子索引
    #     radical_electrons = atom.GetNumRadicalElectrons()  # 自由基电子数
    #     if radical_electrons > 0:
    #         # print(f"原子 {atom_idx} 存在自由基，自由基电子数: {radical_electrons}\n current NumExplicitHs: {atom.GetNumExplicitHs()}")
    #         # 消除自由基：通过添加氢原子调整价态
    #         atom.SetNumRadicalElectrons(0)  # 将自由基电子数设为 0,as has added bond
    #         # atom.SetNumExplicitHs(atom.GetNumExplicitHs() + radical_electrons) 
    Chem.SanitizeMol(mols[-1])
    # 输出修改后的分子 SMILES
    modified_smiles = Chem.MolToSmiles(mols[-1])
    # print(f"修改后的分子 SMILES: {modified_smiles}")            
    return mols[-1], modified_smiles

################################################################################################################################################################
def output_to_smiles(output,idx_to_labels,bond_labels,result):#this will output * without abbre version
    #only output smiles with *
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2

    center_coords = torch.stack((x_center, y_center), dim=1)

    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    

    atoms_list, bonds_list,charge = bbox_to_graph_with_charge(output,
                                                idx_to_labels=idx_to_labels,
                                                bond_labels=bond_labels,
                                                result=result)
    smiles, mol= mol_from_graph_with_chiral(atoms_list, bonds_list,charge)
    abc=[atoms_list, bonds_list,charge ]
    
    if isinstance(smiles, type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    elif isinstance(atoms_list,type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    # else:
    #     smiles, mol=smiles_mol
    return abc,smiles,mol,output


def output_to_smiles2(output,idx_to_labels,bond_labels,result):#this will output * without abbre version
    #only output smiles with *
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2

    center_coords = torch.stack((x_center, y_center), dim=1)

    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    

    atoms_list, bonds_list,charge = bbox_to_graph_with_charge(output,
                                                idx_to_labels=idx_to_labels,
                                                bond_labels=bond_labels,
                                                result=result)
    smiles, mol= mol_from_graph_with_chiral(atoms_list, bonds_list,charge)
    abc=[atoms_list, bonds_list,charge ]
    if isinstance(smiles, type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    elif isinstance(atoms_list,type(None)):
        print(f"get atoms_list problems")
        # smiles, mol=None,None
    # else:
    #     smiles, mol=smiles_mol
    return abc,smiles,mol,output



def bbox_to_graph(output, idx_to_labels, bond_labels,result):
    
    # calculate atoms mask (pred classes that are atoms/bonds)
    atoms_mask = np.array([True if ins not in bond_labels else False for ins in output['pred_classes']])

    # get atom list
    atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]

    # if len(result) !=0 and 'other' in atoms_list:
    #     new_list = []
    #     replace_index = 0
    #     for item in atoms_list:
    #         if item == 'other':
    #             new_list.append(result[replace_index % len(result)])
    #             replace_index += 1
    #         else:
    #             new_list.append(item)
    #     atoms_list = new_list

    atoms_list = pd.DataFrame({'atom': atoms_list,
                            'x':    output['bbox_centers'][atoms_mask, 0],
                            'y':    output['bbox_centers'][atoms_mask, 1]})

    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            if row.atom[-2] != '-':#assume charge value -9~9
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bonds_list = []

    # get bonds
    for bbox, bond_type, score in zip(output['bbox'][np.logical_not(atoms_mask)],
                                    output['pred_classes'][np.logical_not(atoms_mask)],
                                    output['scores'][np.logical_not(atoms_mask)]):
         
        # if idx_to_labels[bond_type] == 'SINGLE':
        if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            _margin = 5
        else:
            _margin = 8

        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        #NOTE  this proces may lead self-bonding for one atom
        if begin_idx != end_idx:# avoid self-bond
            bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], idx_to_labels[bond_type], score))
        else:
            continue
    # return atoms_list.atom.values.tolist(), bonds_list
    return atoms_list, bonds_list


def calculate_distance(coord1, coord2):
    # Calculate Euclidean distance between two coordinates
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def assemble_atoms_with_charges(atom_list, charge_list):
    used_charge_indices=set()
    atom_list = atom_list.reset_index(drop=True)
    # atom_list['atom'] = atom_list['atom'] + '0'
    kdt = cKDTree(atom_list[['x','y']])
    for i, charge in charge_list.iterrows():
        if i in used_charge_indices:
            continue
        charge_=charge['charge']
        # if charge_=='1':charge_='+'
        dist, idx_atom=kdt.query([charge_list.x[i],charge_list.y[i]], k=1)
        # atom_str=atom_list.loc[idx_atom,'atom'] 
        if idx_atom not in atom_list.index:
            print(f"Warning: idx_atom {idx_atom} is out of range for atom_list.")
            continue  # 跳过当前循环迭代
        atom_str = atom_list.iloc[idx_atom]['atom']
        if atom_str=='*':
            atom_=atom_str + charge_
        else:
            try:
                atom_ = re.findall(r'[A-Za-z*]+', atom_str)[0] + charge_
            except Exception as e:
                print(atom_str,charge_,charge_list)
                print(f"@assemble_atoms_with_charges\n {e}\n{atom_list}")
                atom_=atom_str + charge_
        atom_list.loc[idx_atom,'atom']=atom_

    return atom_list



def assemble_atoms_with_charges2(atom_list, charge_list, max_distance=10):
    used_charge_indices = set()

    for idx, atom in atom_list.iterrows():
        atom_coord = atom['x'],atom['y']
        atom_label = atom['atom']
        closest_charge = None
        min_distance = float('inf')

        for i, charge in charge_list.iterrows():
            if i in used_charge_indices:
                continue

            charge_coord = charge['x'],charge['y']
            charge_label = charge['charge']

            distance = calculate_distance(atom_coord, charge_coord)
            #NOTE how t determin this max_distance, dependent on image size??
            if distance <= max_distance and distance < min_distance:
                closest_charge = charge
                min_distance = distance

        
        if closest_charge is not None:
            if closest_charge['charge'] == '1':
                charge_ = '+'
            else:
                charge_ = closest_charge['charge']
            atom_ = atom['atom'] + charge_

            # atom['atom'] = atom_
            atom_list.loc[idx,'atom'] = atom_
            used_charge_indices.add(tuple(charge))

        else:
            # atom['atom'] = atom['atom'] + '0'
            atom_list.loc[idx,'atom'] = atom['atom'] + '0'

    return atom_list



def bbox_to_graph_with_charge(output, idx_to_labels, bond_labels,result):
    
    bond_labels_pre=bond_labels
    # charge_labels = [18,19,20,21,22]#make influence
    atoms_mask = np.array([True if ins not in bond_labels and ins not in charge_labels else False for ins in output['pred_classes']])

    try:
        # print(atoms_mask.shape)
        # print(output['pred_classes'].shape)
        atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]
        if isinstance(atoms_list, pd.Series) and atoms_list.empty:
            return None, None, None
        else:
            atoms_list = pd.DataFrame({'atom': atoms_list,
                                    'x':    output['bbox_centers'][atoms_mask, 0],
                                    'y':    output['bbox_centers'][atoms_mask, 1],
                                    'bbox':  output['bbox'][atoms_mask].tolist() ,#need this for */other converting
                                    'scores': output['scores'][atoms_mask].tolist(),
                                    })
    except Exception as e:
        print(output['pred_classes'][atoms_mask].dtype,output['pred_classes'][atoms_mask])#int64 [ 1  1  1  1  1  2  1 29]
        print(e)
        print(idx_to_labels)
        # print(output['pred_classes'][atoms_mask],"output['pred_classes'][atoms_mask]")
    
        
        # confict_atompaire=[]
        # # 如果你想计算所有边界框之间的IOU，考虑2个原子box 重叠 是否要删掉一个？？ TODO gmy version most box larger then normal mix the rules
        # for i in range(len(atoms_list)):
        #     for j in range(i + 1, len(atoms_list)):
        #         iou_value = calculate_iou(atoms_list.bbox[i], atoms_list.bbox[j])
        #         if iou_value !=0:
        #             # print(f"IOU between box {i} and box {j}: {iou_value}")
        #             if i !=j : confict_atompaire.append([i,j])
        # if len(confict_atompaire)>0:
        #     need_del=[]
        #     for i,j in confict_atompaire:
        #         ij_lab=[atoms_list.loc[i].atom,atoms_list.loc[j].atom ]
        #         ij_score=[atoms_list.loc[i].scores,atoms_list.loc[j].scores]
        #         # print(ij_lab,ij_score)
        #         if ij_lab==['C','N'] or ij_lab==['N','C']:
        #             if atoms_list.loc[i].atom =='C':
        #                 need_del.append(i)
        #             else:
        #                 need_del.append(j)
                # elif atoms_list.loc[i].scores> atoms_list.loc[j].scores:
                #         need_del.append(j)
                # elif atoms_list.loc[j].scores> atoms_list.loc[i].scores:
                #         need_del.append(i)  
            # print(need_del)          
            # atoms_list= atoms_list.drop(need_del)

    charge_mask = np.array([True if ins in charge_labels else False for ins in output['pred_classes']])
    charge_list = [idx_to_labels[a] for a in output['pred_classes'][charge_mask]]
    charge_list = pd.DataFrame({'charge': charge_list,
                        'x':    output['bbox_centers'][charge_mask, 0],
                        'y':    output['bbox_centers'][charge_mask, 1],
                        'scores':    output['scores'][charge_mask],
                        
                        })
    
    # print(charge_list,'\n@bbox_to_graph_with_charge')
    try:
        atoms_list['atom'] = atoms_list['atom']+'0'#add 0 
    except Exception as e:
        print(e)
        print(atoms_list['atom'],'atoms_list["atom"] @@ adding 0 ')
        

    if len(charge_list) > 0:
        atoms_list = assemble_atoms_with_charges(atoms_list,charge_list)
    # else:#Note Most mols are not formal charged 
        # atoms_list['atom'] = atoms_list['atom']+'0'
    # print(atoms_list,"after @@assemble_atoms_with_charges ")
    
    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            try:
                if row.atom[-2] != '-':#assume charge value -9~9
                    overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            except Exception as e:
                print(row.atom,"@rin case atoms with sign gets detected two times")
                print(e)
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bonds_list = []
    # get bonds
    # bond_mask=np.logical_not(np.logical_not(atoms_mask) | np.logical_not(charge_mask))
    bond_mask=np.logical_not(atoms_mask) & np.logical_not(charge_mask)
    for bbox, bond_type, score in zip(output['bbox'][bond_mask],  #NOTE also including the charge part
                                    output['pred_classes'][bond_mask],
                                    output['scores'][bond_mask]):
         
        # if idx_to_labels[bond_type] == 'SINGLE':
        if len(idx_to_labels)==23:
            if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                _margin = 5
            else:
                _margin = 8
        elif len(idx_to_labels)==30:
            _margin=0#ad this version bond dynamicaly changed
        elif len(idx_to_labels)==24:
            _margin=0#ad this version bond dynamicaly changed
        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        #NOTE  this proces may lead self-bonding for one atom
        if begin_idx != end_idx: 
            if bond_type in bond_labels:# avoid self-bond
                bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], idx_to_labels[bond_type], score))
            else:
                print(f'this box may be charges box not bonds {[bbox, bond_type, score ]}')
        else:
            continue
    # return atoms_list.atom.values.tolist(), bonds_list
    # print(f"@box2graph: atom,bond nums:: {len(atoms_list)}, {len(bonds_list)}")
    return atoms_list, bonds_list,charge_list#dataframe, list

def parse_atom(node):
    s10 = [str(x) for x in range(10)]
    # Determine atom and formal charge
    if 'other' in node:
        a = '*'
        if '-' in node or '+' in node:
            fc = -1 if node[-1] == '-' else 1
        else:
            fc = int(node[-2:]) if node[-2:] in s10 else 0
    elif node[-1] in s10:
        if '-' in node or '+' in node:
            fc = -1 if node[-1] == '-' else 1
            a = node[:-1]
        else:
            a = node[:-1]
            fc = int(node[-1])
    elif node[-1] == '+':
        a = node[:-1]
        fc = 1
    elif node[-1] == '-':
        a = node[:-1]
        fc = -1
    else:
        a = node
        fc = 0
    return a, fc

#from engine

def iou_(box1, box2):
    """
    计算两个框的 IoU（Intersection over Union）。
    参数:
        box1, box2: [x1, y1, x2, y2] 格式的框坐标
        
    返回:
        float: IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def calculate_iou(bbox1, bbox2):
    # 提取坐标
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    
    # 计算交集坐标
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    
    # 计算交集面积
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height
    
    # 计算两个框的面积
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    # 判断关系并记录
    result = []
    if iou == 0:
        result.append("无重叠")
    elif iou > 0:
        result.append("有重叠")
        if iou == 1:
            result.append("完全重合")
        elif inter_area == area2:
            result.append("bbox1 包含 bbox2")
        elif inter_area == area1:
            result.append("bbox2 包含 bbox1")
    
    return iou, result, inter_area, union_area

def adjust_bbox1(large_bbox, small_bbox, bond_bbox):
    # 假设调整策略：扣除小的 atom bbox 和 bond box 的区域
    # 这里简单假设从较大 bbox 中移除小的区域，可能需要根据具体需求调整
    x_min_l, y_min_l, x_max_l, y_max_l = large_bbox
    x_min_s, y_min_s, x_max_s, y_max_s = small_bbox
    x_min_b, y_min_b, x_max_b, y_max_b = bond_bbox
    scaled_box= max([x_min_l,x_min_s,x_min_b]),max([y_min_l,y_min_s,y_min_b]),x_max_l, y_max_l
    return large_bbox
    # 示例调整：如果小的 bbox 和 bond box 在较大 bbox 内，缩小较大 bbox
    # if x_min_s > x_min_l and y_min_s > y_min_l:
    #     return [x_min_l, y_min_l, x_min_s, y_min_s]  # 示例：保留左上部分
    # return large_bbox  # 默认不调整


def nms_per_class(labels, boxes, scores, iou_thresh=0.5):
    """
    对每个类别应用 NMS，保留得分最高的框。
    参数:
        labels: numpy array，类别标签
        boxes: numpy array，框坐标 [x1, y1, x2, y2]
        scores: numpy array，得分
        iou_thresh: float，IoU 阈值
    返回:
        dict: 筛选后的输出
    """
    # 按类别分组
    unique_labels = np.unique(labels)
    kept_indices = []
    for label in unique_labels:
        # 筛选当前类别的框
        class_mask = labels == label
        class_indices = np.where(class_mask)[0]
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # 按得分从高到低排序
        order = np.argsort(class_scores)[::-1]
        class_boxes = class_boxes[order]
        class_scores = class_scores[order]
        class_indices = class_indices[order]
        
        # NMS
        keep = []
        while len(class_scores) > 0:
            # 保留得分最高的框
            keep.append(class_indices[0])
            if len(class_scores) == 1:
                break
            
            # 计算当前框与其他框的 IoU
            ious = np.array([calculate_iou(class_boxes[0], box) for box in class_boxes[1:]])
            # 保留 IoU 低于阈值的框
            keep_mask = ious < iou_thresh
            class_boxes = class_boxes[1:][keep_mask]
            class_scores = class_scores[1:][keep_mask]
            class_indices = class_indices[1:][keep_mask]
        
        kept_indices.extend(keep)
    
    # 根据保留的索引更新输出
    kept_indices = np.array(kept_indices)
    return {
        'labels': labels[kept_indices],
        'boxes': boxes[kept_indices],
        'scores': scores[kept_indices]
    }

