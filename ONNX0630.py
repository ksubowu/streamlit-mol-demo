#!/usr/bin/env python
# coding: utf-8
import os,sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--tuning', '-t', type=str,)# default='/nfs_home/bowen/model_checkpoint/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth')
parser.add_argument('--test-only',default=True,)
parser.add_argument('--amp', default=False,)
parser.add_argument('--dataname', '-da', type=str, default=None)
parser.add_argument('--gpuid', '-gi', type=str, default=None)
parser.add_argument('--number', '-n', type=str, default=None)
args, unknown = parser.parse_known_args()#in jupyter
print(args)
if args.gpuid:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuid}'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parralel_n=2
os.environ["OMP_NUM_THREADS"] = f"{parralel_n}"       # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = f"{parralel_n}"  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = f"{parralel_n}"       # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{parralel_n}"  # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = f"{parralel_n}"   # NumExpr
"""
WARNING: OMP_NUM_THREADS set to 4, not 1. The computation speed will not be optimized if you use data parallel. It will fail if this PaddlePaddle binary is compiled with OpenBlas since OpenBlas does not support multi-threads.
PLEASE USE OMP_NUM_THREADS WISELY.

"""


import shutil
import pandas as pd
# print(sys.path)
print(__file__)
cur_dir = os.path.dirname(os.path.abspath(__file__))
print(cur_dir)
python_path=cur_dir

sys.path.append(python_path)
# model_path='I2M_realv2.onnx'
# model_abs_path = os.path.abspath(model_path)
# if os.path.exists(model_abs_path):
#     print(model_abs_path)

# from src.solver.det_engine import *
import cv2

import sys,copy
import torch
import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from det_engine import N_C_H_expand, C_H_expand,C_H_expand2, C_F_expand, formula_regex, RTDETRPostProcessor
from det_engine import SmilesEvaluator, molfpsim


import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import DataStructs


# 安全释放资源
def release_ocr(ocr_instance):
    # 关闭所有相关模型
    if hasattr(ocr_instance, 'detector'):
        ocr_instance.detector = None
    if hasattr(ocr_instance, 'recognizer'):
        ocr_instance.recognizer = None
    if hasattr(ocr_instance, 'cls'):
        ocr_instance.cls = None



print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
# print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# In[ ]:

# 计算bbox面积并找到最小的
def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)
    
def mol_idx( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

# 移除原子索引
def mol_idx_del(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('molAtomMapNumber'):  # 检查属性是否存在
            atom.ClearProp('molAtomMapNumber')  # 清除属性
    return mol

def is_contained_in(bbox_small, bbox_large):
    x_min_s, y_min_s, x_max_s, y_max_s = bbox_small
    x_min_l, y_min_l, x_max_l, y_max_l = bbox_large
    return (x_min_s >= x_min_l and x_max_s <= x_max_l and 
            y_min_s >= y_min_l and y_max_s <= y_max_l)


def NoRadical_Smi(smi):
    aa=Chem.MolFromSmiles(smi)
    for atom in aa.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:  # 检查是否有自由基
            # print(f"找到自由基原子: {atom.GetSymbol()}, 自由电子数: {atom.GetNumRadicalElectrons()}")
            # 添加氢原子以去除自由基
            atom.SetNumRadicalElectrons(0)  # 将自由电子数设为 0
            # 根据硫原子的化合价调整氢原子数
            atom.SetNumExplicitHs(atom.GetTotalValence() - atom.GetExplicitValence())
    san_before=Chem.MolToSmiles(aa)
    # print(san_before)
    return san_before


def select_longest_smiles(smiles):
    # 将 SMILES 以 '.' 分割为多个部分
    components = smiles.split('.')
    # 选择字符数最多的部分作为主结构
    longest_component = max(components, key=len)
    return longest_component

# 解析电荷值
def parse_charge(charge_str):
    if charge_str.endswith('+'):
        return int(charge_str[:-1]) if charge_str[:-1] else 1  # "1+" -> 1, "+" -> 1
    elif charge_str.endswith('-'):
        return -int(charge_str[:-1]) if charge_str[:-1] else -1  # "2-" -> -2, "-" -> -1
    else :
        return int(charge_str)



def set_bondDriection(rwmol_,bondWithdirct):
    #set direction 
    chiral_center_ids = Chem.FindMolChiralCenters(rwmol_, includeUnassigned=True)
    # chiral_center_ids
    chirai_ai2sterolab=dict()
    if len(chiral_center_ids)>0:
        chirai_ai2sterolab={ai:stero_lab for ai, stero_lab in chiral_center_ids }

    for bi, binfo in bondWithdirct.items():
        atom1_idx, atom2_idx, bond_type, score, w_d = binfo
        bt= rwmol_.GetBondBetweenAtoms(atom1_idx, atom2_idx)#RDKit 的键是无向的，返回的是同一个 Bond 对象
        current_begin = bt.GetBeginAtomIdx()
        current_end = bt.GetEndAtomIdx()
        if w_d=='wdge':
            bond_dir_=rdchem.BondDir.BEGINWEDGE
            reverse_dir = rdchem.BondDir.BEGINDASH 
        elif w_d=='dash':
            bond_dir_=rdchem.BondDir.BEGINDASH
            reverse_dir = rdchem.BondDir.BEGINWEDGE 
        
        if atom1_idx in chirai_ai2sterolab.keys():
            if current_begin == atom1_idx:
                bt.SetBondDir(bond_dir_)
                print(f'atom1_idx dir')
            else:
                # 如果手性原子是终点，反转方向（例如用相反的楔形键）
                bt.SetBondDir(reverse_dir)
                print(f'atom1_idx reverse_dir')
        elif atom2_idx in chirai_ai2sterolab.keys():
            if current_begin == atom2_idx:
                bt.SetBondDir(bond_dir_)
                print(f'atom2_idx dir {bond_dir_} {reverse_dir}')
            else:
                # 如果手性原子是终点，反转方向（例如用相反的楔形键）,but not work, just remove and add
                rwmol_.RemoveBond(current_begin, current_end)
                rwmol_.AddBond(current_end, current_begin, bt.GetBondType())
                bond = rwmol_.GetBondBetweenAtoms(current_end, current_begin)
                bond.SetBondDir(bond_dir_)
                print(f'atom2_idx reverse_dir {bond_dir_} {reverse_dir}')
        else:
            print('bond stro not with chiral atom???, will ignore this stero bond infors')
            print(f"{[bi, bond_dir_, current_begin,current_end]}")
        return rwmol_

atom_labels = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
bond_labels = [13,14,15,16,17,18]
charge_labels=[19,20,21,22,23]


idx_to_labels={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                    9:'I',10:'P',11:'H',12:'Si',
                    #bond
                    13:'single',14:'wdge',15:'dash',
                    16:'=',17:'#',18:':',#aromatic
                    #charge
                    19:'-4',20:'-2',
                    21:'-1',#-
                    22:'+1',#+
                    23:'+2',
                    }
lab2idx={ v:k for k,v in idx_to_labels.items()}   
bond_labels_symb={idx_to_labels[i] for i in bond_labels}                 

bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
            'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
            'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
            'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,
            }
    

import pandas as pd
from typing import Iterable, List
from PIL import Image
import json,re

#TODO now abc single bond and OCR checking 
#OCR 得到纯数字box 离原子距离应该小于最小的bond 距离，否则丢弃
from utils import calculate_iou,adjust_bbox1
from scipy.spatial import cKDTree, KDTree
import numpy as np
from rdkit import Chem
from paddleocr import PaddleOCR
from rdkit.Chem import rdchem, RWMol, CombineMols

from det_engine import ABBREVIATIONS,remove_SP
from det_engine import molExpanding,remove_bond_directions_if_no_chiral
from det_engine import (comparing_smiles,comparing_smiles2, remove_SP, expandABB, 
                ELEMENTS,
                ABBREVIATIONS)



from det_engine import expandABB

def bbox2shapes(bboxes, classes, lab2idx):
    shapes = []
    for bbox, label in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox
        if label not in lab2idx : 
            label='other'
        
        # Create shape dictionary
        shape = {
            "kie_linking": [],
            "label": label,
            "score": 1.0,
            "points": [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ],
            "group_id": None,
            "description": None,
            "difficult": False,
            "shape_type": "rectangle",
            "flags": None,
            "attributes": {}
        }
        shapes.append(shape)
    return shapes

def get_longest_part(smi_string):
    if '.' in smi_string:  # 如果包含点号
        parts = smi_string.split('.')  # 按点号分割
        longest_part = max(parts, key=len)  # 取最长的部分
        return longest_part
    else:
        return smi_string  # 如果不包含点号，返回原字符串


def split_output_by_numeric_classes(output):
    # 初始化两个结果字典
    numeric_output = {key: [] for key in output.keys()}
    non_numeric_output = {key: [] for key in output.keys()}
    
    # 遍历所有元素
    for i in range(len(output['pred_classes'])):
        class_name = output['pred_classes'][i]
        
        # 检查是否是纯数字（包括正负号）
        if re.fullmatch(r'^[+-]?\d+[+-]?$', class_name):
            target_dict = numeric_output
        else:
            target_dict = non_numeric_output
        
        # 将当前元素添加到相应的字典中
        for key in output.keys():
            target_dict[key].append(output[key][i])
    
    return numeric_output, non_numeric_output
    

def convert_shapes_to_output(json_data):
    output = {
        'bbox': [],
        'bbox_centers': [],
        'scores': [],
        'pred_classes': []
    }
    for shape in json_data['shapes']:
        # Extract bbox coordinates (assuming shape['points'] is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
        points = shape['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        # Calculate bbox as [x_min, y_min, x_max, y_max]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        # Calculate center coordinates
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Get score (use 1.0 if not available)
        score = shape.get('score', 1.0)
        
        # Get class label (assuming shape['label'] contains the class)
        pred_class = shape['label']
        
        # Append to output
        output['bbox'].append(bbox)
        output['bbox_centers'].append([center_x, center_y])
        output['scores'].append(score)
        output['pred_classes'].append(pred_class)
    
    return output


def getJsonData(src_json):
    with open(src_json, 'r') as f:
            coco_data = json.load(f)
    return coco_data

def replace_cg_notation(astr):
    def replacer(match):
        h_count = int(match.group(1))
        c_count = (h_count - 1) // 2
        return f'C{c_count}H{h_count}'

    return re.sub(r'CgH(\d+)', replacer, astr)

def viewcheck(image_path,bbox,color='red'):
    image = Image.open(image_path)
    image_array = np.array(image)
    # 创建绘图
    plt.figure(figsize=(5, 4))  # 设置图像大小
    plt.imshow(image_array)  # 显示图像
    bbox = np.array(bbox)
    x_coords = (bbox[:, 0]+bbox[:, 2])*0.5
    y_coords =( bbox[:, 1]+bbox[:, 3])*0.5
    plt.scatter(x_coords, y_coords, c=color, s=50, label='Atom Centers', edgecolors='black')
    # 添加标注（可选）
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f'a {i}', fontsize=12, color=color, ha='center', va='bottom')

bclass_simple={"single":'-', "wdge":'w', "dash":'--', 
                "=":'=', "#":"#", ":":"aro"}

def viewcheck_b(image_path,bbox,bclass,color='red',figsize=(5,4)):
    image = Image.open(image_path)
    image_array = np.array(image)
    # 创建绘图
    plt.figure(figsize=figsize)  # 设置图像大小
    plt.imshow(image_array)  # 显示图像
    # 提取 bbox
    bbox = np.array(bbox)
    x_coords = (bbox[:, 0]+bbox[:, 2])*0.5
    y_coords =( bbox[:, 1]+bbox[:, 3])*0.5
    plt.scatter(x_coords, y_coords, c=color, s=50, label='bond Centers', edgecolors='black')
    # 添加标注（可选）
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        simpl_b=bclass_simple[bclass[i]]
        plt.text(x, y, f'b{i}{simpl_b}', fontsize=12, color=color, ha='center', va='bottom')    


def anchor_draw(image_path, bond_bbox):
    # 加载图像
    image = Image.open(image_path)
    image_array = np.array(image)

    # 初始化
    _margin = 3
    all_anchor_positions = []
    all_oposite_anchor_positions = []

    # 计算所有 bond 的锚点
    for bi, bbox in enumerate(bond_bbox):
        # 计算锚点
        anchor_positions = (np.array(bbox) + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])
        
        # 存储前两个点为 anchor_positions，后两个点为 oposite_anchor_positions
        all_anchor_positions.append(anchor_positions[:2])  # [上左, 下右]
        all_oposite_anchor_positions.append(anchor_positions[2:])  # [下左, 上右]

    # 转换为 numpy 数组
    all_anchor_positions = np.array(all_anchor_positions).reshape(-1, 2)
    all_oposite_anchor_positions = np.array(all_oposite_anchor_positions).reshape(-1, 2)

    # 图 1：绘制 anchor_positions
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.scatter(all_anchor_positions[:, 0], all_anchor_positions[:, 1], c='red', s=50, label='Anchor Positions', edgecolors='black')
    for i, (x, y) in enumerate(all_anchor_positions):
        plt.text(x, y, f'B{int(i/2)}:{i%2}', fontsize=10, color='white', ha='center', va='bottom')
    plt.title('Anchor Positions (Upper Left, Lower Right)')
    plt.legend()
    plt.axis('off')
    plt.savefig('anchor_positions.png')

    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.scatter(all_oposite_anchor_positions[:, 0], all_oposite_anchor_positions[:, 1], c='blue', s=50, label='Opposite Anchor Positions', edgecolors='black')
    for i, (x, y) in enumerate(all_oposite_anchor_positions):
        plt.text(x, y, f'B{int(i/2)}:{i%2}', fontsize=10, color='white', ha='center', va='bottom')
    plt.title('Opposite Anchor Positions (Lower Left, Upper Right)')
    plt.legend()
    plt.axis('off')
    plt.savefig('Opposite_anchor_positions.png')


# 计算 4 个顶点
def get_corners(bbox):
    x_min, y_min, x_max, y_max = bbox
    return np.array([
        [x_min, y_min], [x_max, y_min],  # 上左，上右
        [x_min, y_max], [x_max, y_max]   # 下左，下右
    ])

# 计算两组顶点之间的最小距离并返回最近的 atom_idx
def find_nearest_atom(bond_corners, atom_bboxes, exclude_idx=None):
    min_dist = float('inf')
    nearest_idx = None
    for i, atom_bbox in enumerate(atom_bboxes):
        if exclude_idx is not None and i in exclude_idx:
            continue
        atom_corners = get_corners(atom_bbox)
        for bc in bond_corners:
            for ac in atom_corners:
                dist = np.sqrt((bc[0] - ac[0])**2 + (bc[1] - ac[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
    return nearest_idx, min_dist
# 计算顶点到顶点的距离
def get_min_distance_to_atom_box(vertices, atom_bboxes, exclude_idx=None):
    min_dist = float('inf')
    closest_atom_idx = -1
    for i, ab in enumerate(atom_bboxes):
        if exclude_idx is not None and i in  exclude_idx:
            continue
        ab_vertices = np.array([[ab[0], ab[1]], [ab[2], ab[3]], [ab[0], ab[3]], [ab[2], ab[1]]])
        for v in vertices:
            for av in ab_vertices:
                dist = np.linalg.norm(v - av)
                if dist < min_dist:
                    min_dist = dist
                    closest_atom_idx = i
    return min_dist, closest_atom_idx


# 检查孤立原子并添加键
def boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def min_corner_distance(box1, box2):
    corners1 = [[box1[0], box1[1]], [box1[2], box1[3]], [box1[0], box1[3]], [box1[2], box1[1]]]
    corners2 = [[box2[0], box2[1]], [box2[2], box2[3]], [box2[0], box2[3]], [box2[2], box2[1]]]
    min_dist = float('inf')
    for c1 in corners1:
        for c2 in corners2:
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            min_dist = min(min_dist, dist)
    return min_dist

def clear_directory(path):
    if os.path.exists(path):
        print(f"Clearing contents of: {path}")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除子目录
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory does not exist: {path}")


def NHR_string(text):
    # 模式 1: 匹配 NHR 后跟一个数字
    pattern1 = r'NHR\d'
    # 模式 2: 匹配 RHN 后跟至少一个数字或小写字母
    pattern2 = r'RHN[0-9a-z]+'
    # 模式 3: 匹配 R 后跟至少一个数字或小写字母，再跟 NH，替换为 NHR
    pattern3 = r'R[0-9a-z]+NH'
    # 先处理模式 3，替换为 NHR
    # text = re.sub(pattern3, 'NHR', text)
    # 检查是否匹配模式 1
    if re.search(pattern1, text):
        # print(f"Matched pattern 1: {text}")
        text='NH*'
    # 检查是否匹配模式 2
    elif re.search(pattern2, text):
        # print(f"Matched pattern 2: {text}")
        text='NH*'
    elif re.search(pattern3, text):
        text='NH*'

    return text

from det_engine import normalize_ocr_text, check_and_fix_valence, rdkit_canonicalize_smiles
from det_engine import is_valid_chem_text,select_chem_expression
# Preprocess atom boxes to handle large functional groups
def preprocess_atom_boxes(atom_centers, atom_bbox, size_threshold_factor=2.5, min_subboxes=2):
    """
    Identify large atom boxes and split them into smaller sub-boxes of approximately average size.
    Returns updated atom_centers, atom_bbox, and a mapping of sub-boxes to original box IDs.
    """
    # Calculate areas of atom boxes
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in atom_bbox]
    # Compute average area, excluding max and min to avoid outliers
    if len(areas) > 2:
        sorted_areas = sorted(areas)
        avg_area = np.mean(sorted_areas[1:-1])  # Exclude min and max
    else:
        avg_area = np.mean(areas) if areas else 1.0

    new_atom_centers = []
    new_atom_bbox = []
    original_to_subbox = {}  # Maps original atom index to list of new sub-box indices
    subbox_to_original = {}  # Maps new sub-box index to original atom index
    new_idx = 0

    for i, (bbox, center) in enumerate(zip(atom_bbox, atom_centers)):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # Identify large boxes (e.g., functional groups like CH2CH2CH2CH)
        if area > avg_area * size_threshold_factor:
            # Estimate number of sub-boxes based on area ratio
            num_subboxes = max(min_subboxes, int(round(area / avg_area)))
            # Split box along the longer dimension (x or y)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width >= height:
                # Split horizontally
                sub_width = width / num_subboxes
                subboxes = [
                    [bbox[0] + j * sub_width, bbox[1], bbox[0] + (j + 1) * sub_width, bbox[3]]
                    for j in range(num_subboxes)
                ]
            else:
                # Split vertically
                sub_height = height / num_subboxes
                subboxes = [
                    [bbox[0], bbox[1] + j * sub_height, bbox[2], bbox[1] + (j + 1) * sub_height]
                    for j in range(num_subboxes)
                ]
            # Compute centers for sub-boxes
            sub_centers = [
                [(subbox[0] + subbox[2]) / 2, (subbox[1] + subbox[3]) / 2]
                for subbox in subboxes
            ]
            # Add sub-boxes and centers
            new_atom_bbox.extend(subboxes)
            new_atom_centers.extend(sub_centers)
            original_to_subbox[i] = list(range(new_idx, new_idx + num_subboxes))
            for j in range(num_subboxes):
                subbox_to_original[new_idx + j] = i
            new_idx += num_subboxes
        else:
            # Keep original box
            new_atom_bbox.append(bbox)
            new_atom_centers.append(center)
            original_to_subbox[i] = [new_idx]
            subbox_to_original[new_idx] = i
            new_idx += 1

    return np.array(new_atom_centers), new_atom_bbox, original_to_subbox, subbox_to_original






#add OCR here for  placeholder_atoms adding
other2ppsocr=True
if other2ppsocr:
    ocr = PaddleOCR(
    use_angle_cls=True,
    lang='latin',use_space_char=True,use_debug=False,
    use_gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)

    ocr2 = ocr2 = PaddleOCR(use_angle_cls=True,use_gpu =False,use_debug=False,
                rec_algorithm='SVTR_LCNet', 
                # rec_model_dir='/nfs_home/bowen/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
                lang="en") 
    # outcsv_filename=f"{output_directory}/{prefix_f}_withOCR.csv"

# box_thresh=0.45# 292 -240=52
box_thresh=0.5# 292 -233=59
useocr=True
box_matter=0
getacc=False
getfpsim=False
visual_check=False



# da='acs'
# src_dir=f"D:\RPA\codes_share\wsl_\chem_box\\real\\real\{da}"
# src_file=f"{src_dir}.csv"
# df = pd.read_csv(src_file)
# dst_dirac = f"D:\RPA\codes_share\wsl_\chem_box\\need2check\{da}_ac"
# dst_dirb = f"D:\RPA\codes_share\wsl_\chem_box\\need2check\{da}_b"
da='acs'
#198th row, fixwd with expanded smiles
#326th, Tos  we use the SO2Ph not SiC3 version,conflict fixed

#for view
# view_check_dir=f"D:\RPA\codes_share\wsl_\chem_box\\need2check\\{da}_fixedView"
da='CLEF'#NOTE 
#462 S[O]a fixed, 
#179 NHR8 fixed as NH-R8
#fix rows@582,750,411,7612,761 [(CH2)m] [(CH2)q] [(CH2)s] RDKIT NOT readable  fixed as [CH2]
# 30,214, 795, 856, 583, 654, 618,339,138, 927, 203, 869, 261, 634, 180, 63,758, 718, 741,832,88,250, 799，303,956,810,596|bond erro,  wrong smiles 
#TODO still failed:1||SO2 mutil-rows from 992

da='UOB'#NOTE TODO fix rows@5119, 3420,990,1082,2451,3626,1634,627,5385
#all paseed @ v3
da='USPTO'#NOTE TODO fix rows@ 458, 10+ [O.], [NH2.],|| 4566,5523, ima!=smi, 3927，5234，4062|poly unitProb
#1658,4625,4944 #also SO3, SOOO erro
#1164 CHO,2703 CN NC err, 4421 CH2O erro
#4921, Rgroup error fixwed
#58, SMILES WRONG fixed (NH4NO)2
#2352 Fix wrong smiles
#4590 fix wrong smi
#3381, 4921 wrong smi fixed
#3071 image  C8H13 may not expandable 
# da='staker'#NOTE TODO fix rows@11422(del 11420.png, as it is table not chemMol)
#SO3 as SOOO, should be S(=O)(=O)O, as o-o-o strange in chemicstry, this erro 50 as below
#1971,5770,5972,5973,7541,7542,7666,7854,8258,8917,8918,11129,13281,14109,17131,17132,17189,17493,21091,21093,22314,22315,24524,24525,27294,27295,27296,27297,27586,27587,29562,29766,32835,33517,36197,36198,38199,38200,38661,38663,39174,39410,46717,48380,48381,48382,48443,48624,48625




da='JPO'
# da='staker'

if args.dataname:
    da=args.dataname

# ac_b=False
ac_b=False    
ac_b_smilesnotsame_writJson=True
if ac_b:
    view_check_dir=f"D:\RPA\codes_share\wsl_\X-AnyLabeling\\need2check\\view_check_{da}\\failed"
    view_dirac=f"{view_check_dir}/{da}_ac"
    view_dirb=f"{view_check_dir}/{da}_b"
    dst_dirac =view_dirac#when double check used 
    dst_dirb =view_dirb

# Construct paths using os.path.join
src_dir = cur_dir
src_file = os.path.join(src_dir, f"{da}.csv")
# df = pd.read_csv(src_file)
# print(f"src_file:\n{src_file}")
# Construct check and view directories
# view_check_dir2 = os.path.join(src_dir, f"{da}_fixedView", "failed")
# view_check_dir2 = os.path.join(src_dir, f"view_check_{da}", "failed")
view_check_dir2 = os.path.join(src_dir, f"view_check_{da}", "v3")#v3 has the manulay ac b json

N=1
if args.number:
    N=int(args.number)

# view_dirac2 = os.path.join(view_check_dir2, f"{da}_ac_N_{N}")
# view_dirb2 = os.path.join(view_check_dir2, f"{da}_b_N_{N}")
view_dirac2 = os.path.join(view_check_dir2, f"{da}_ac")
view_dirb2 = os.path.join(view_check_dir2, f"{da}_b")

view_dirac_tmp = os.path.join(view_check_dir2, f"{da}_actmp")
view_dirac_tmp_debug=True


# if ac_b:
#     need2mkdir=[view_check_dir,view_dirac, view_dirb, view_check_dir2,view_dirac2, view_dirb2]
# else:
#     need2mkdir=[ view_check_dir2,view_dirac2, view_dirb2,view_dirac_tmp]
# for dir_v in need2mkdir :

#     if not os.path.exists(dir_v):
#         os.makedirs(dir_v)

# ac_b=False
ac_b=False
# if ac_b:#update _ac _b
#     # 清空两个目录
#     clear_directory(view_dirac2) #NOTE we only check for the better models faileds
#     clear_directory(view_dirb2)

# #note box not equal as abbv eixsits, process single bond..TODO need check and fixing, may be need rdkit smiles Match
# df['file_name'] = df['file_path'].str.split('/').str[-1]
# # df['file_base'] =f"{da}_" + df['file_name'].str.replace('.png', '', regex=False)
# df['file_base'] = df['file_name'].str.replace('.png', '', regex=False)



# outcsv_filename=f"{src_dir}/{da}_OUTPUTwithOCR.csv"
outcsv_filename=os.path.join(src_dir, f"{da}_OUTPUTwithOCR.csv")

if getacc:
    acc_summary=f"{outcsv_filename}.I2Msummary.txt"
    flogout = open(f'{acc_summary}' , 'w')
    flogout2 = open(f'{outcsv_filename}_acBoxWrong' , 'a')
    failed=[]
    failed_fb=[]
    mydiff=[]
    simRD=0
    sim=0
    simRDlist=[]
    mysum=0

smiles_data = pd.DataFrame({'file_name': [],
                                'SMILESori':[],
                                'SMILESpre':[],
                                'SMILESexp':[],
                                })

# rows_check = df
miss_file=[]
miss_filejs=[]
# for id_, row in rows_check.iterrows():
debug=False

rt_out=False
if not ac_b:
    view_dirac=view_dirac2
    view_dirb=view_dirb2
    dst_dirac =view_dirac#when double check used 
    dst_dirb =view_dirb
    test_dir=f'./test/'#TODO  WEB_dev put test images here
    # pngs=[f for f in os.listdir(view_dirac2) if '.png' in f]
    pngs=[f for f in os.listdir(test_dir) if '.png' in f]
    # if da=='staker':
    #     pngs=[f for f in os.listdir("/nfs_home/bowen/works/pys/codes/i2m/datas/real/staker") if '.png' in f]
    
    rt_out=True
    # view_check_dir3=f"D:\RPA\codes_share\wsl_\chem_box\\need2check\\{da}_fixedView\\v3"
    # view_check_dir3= os.path.join(src_dir, f"{da}_fixedView", "failed")
    view_check_dir3= os.path.join(src_dir, f"view_check_{da}", "v4")#with model output
    view_dirac3=f"{view_check_dir3}/{da}_ac"
    view_dirb3=f"{view_check_dir3}/{da}_b"
    # for dir_v in [view_check_dir3,view_dirac3, view_dirb3]:
    #     if not os.path.exists(dir_v):
    #         os.makedirs(dir_v)
    # pngs=[f for f in os.listdir(view_dirac3) if '.png' in f]

#as abbrev expanded lead a b not equal as original
acn=False
bn=False


import torchvision.transforms.v2 as T

def image_to_tensor(image_path,debug=True):
    image = Image.open(image_path)
    w, h = image.size
    
    # 处理灰度或其他模式
    if image.mode == "L":
        if debug: print("检测到灰度图像 (1 通道)，转换为 RGB...")
        image = image.convert("RGB")
    elif image.mode != "RGB":
        if debug: print(f"检测到 {image.mode} 模式，转换为 RGB...")
        image = image.convert("RGB")
    # Define a transform to convert the image to a tensor and normalize it
    transform = T.Compose([
            T.Resize((640, 640)),  # 调整大小
            # T.ToImageTensor(),  # 转换为 PyTorch Tensor
            T.ToTensor(),
            lambda x: x.to(torch.float32),  # 手动转换数据类型# T.ConvertDtype(dtype=torch.float32),  # 转换数据类型
        ])
    
    # Apply the transform to the image
    tensor = transform(image)
    
    return tensor,w,h
def ouptnp2abc(output,idx_to_labels):
    # Define label lists
    atom_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bond_labels = [13, 14, 15, 16, 17, 18]
    charge_labels = [19, 20, 21, 22, 23]
    # Create masks for atoms, bonds, and charges
    atom_mask = np.isin(output['pred_classes'], atom_labels)
    bond_mask = np.isin(output['pred_classes'], bond_labels)
    charge_mask = np.isin(output['pred_classes'], charge_labels)
    # Initialize output dictionaries
    output_a = {'bbox': [], 'bbox_centers': [], 'scores': [], 'pred_classes': []}
    output_b = {'bbox': [], 'bbox_centers': [], 'scores': [], 'pred_classes': []}
    output_c = {'bbox': [], 'bbox_centers': [], 'scores': [], 'pred_classes': []}
    # Filter and convert for atoms (output_a)
    if np.any(atom_mask):
        output_a['bbox'] = output['bbox'][atom_mask].tolist()
        output_a['bbox_centers'] = output['bbox_centers'][atom_mask].tolist()
        output_a['scores'] = output['scores'][atom_mask].tolist()
        output_a['pred_classes'] = output['pred_classes'][atom_mask].tolist()
        output_a['pred_classes'] = [idx_to_labels[idx] for idx in output_a['pred_classes']]

    # Filter and convert for bonds (output_b)
    if np.any(bond_mask):
        output_b['bbox'] = output['bbox'][bond_mask].tolist()
        output_b['bbox_centers'] = output['bbox_centers'][bond_mask].tolist()
        output_b['scores'] = output['scores'][bond_mask].tolist()
        output_b['pred_classes'] = output['pred_classes'][bond_mask].tolist()
        output_b['pred_classes'] = [idx_to_labels[idx] for idx in output_b['pred_classes']]

    # Filter and convert for charges (output_c)
    if np.any(charge_mask):
        output_c['bbox'] = output['bbox'][charge_mask].tolist()
        output_c['bbox_centers'] = output['bbox_centers'][charge_mask].tolist()
        output_c['scores'] = output['scores'][charge_mask].tolist()
        output_c['pred_classes'] = output['pred_classes'][charge_mask].tolist()
        output_c['pred_classes'] = [idx_to_labels[idx] for idx in output_c['pred_classes']]

        
    return output_a, output_b, output_c

def bbox2center(bbox):
    x_center = (bbox[:, 0] + bbox[:, 2]) / 2
    y_center = (bbox[:, 1] + bbox[:, 3]) / 2
    # center_coords = torch.stack((x_center, y_center), dim=1)
    centers = np.stack((x_center, y_center), axis=1)
    return centers

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

postprocessor=RTDETRPostProcessor(classes_dict=idx_to_labels, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)

#load onnx model
import torch.onnx
import onnx
import onnxruntime as ort
# onnx_model_path = "/nfs_home/bowen/works/pys/codes/i2m/I2M_R4.onnx"#20250605
onnx_model_path="./I2M_R4.onnx"
def image_to_tensor2(image_path):
    # img_path="/cadd_data/samba_share/from_docker/data/work_space/ori/real/acs/op300209p-Scheme-c2-4.png"
    img_path= image_path
    if img_path is not None and os.path.exists(img_path):
        # Load Image From Path Directly
        # NOTE: Potential issue - unable to handle the flipped image.
        # Temporary workaround: cv_image = cv2.imread(img_path)
        cv_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        input_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    image_h, image_w = input_image.shape[:2]
    input_h, input_w = 640,640

    # Compute the scaling factors
    ratio_h = input_h / image_h
    ratio_w = input_w / image_w
    print(ratio_h,ratio_w)
    # Perform the pre-processing steps
    image = cv2.resize(
        input_image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2
    )
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = np.ascontiguousarray(image).astype("float32")
    image /= 255  # 0 - 255 to 0.0 - 1.0
    if len(image.shape) == 3:
        image = image[None]
    wh=image_w,image_h
    return torch.from_numpy(image), image_w, image_h

# 准备输入数据
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 加载并检查ONNX模型
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX模型检查通过")
# 使用ONNX Runtime进行推理
ort_session = ort.InferenceSession(onnx_model_path)
onnx_=True
dfm=0


# ff='US20030130506A1_p0046_x1541_y1396_c00157'
# for ff in pngs:
# for id_, ff in enumerate(pngs):

def predict_from_image(
        ff='/op300209p-Scheme-c2-4.png',
        id_=1
    ):
    image_path= os.path.join(test_dir, f"{ff}")
    SMILESori=''
    print(f"@@@@@@@@@@@@@@@@@@@@@@@ {id_}\n{image_path}\n {SMILESori}")
    # print(image_path,b_datadir,ac_datadir)    

    img_ori = Image.open(image_path).convert('RGB')
    w_ori, h_ori = img_ori.size  # 获取原始图像的尺寸
    # if [w_ori, h_ori]!=[256,256] and da=='staker':
    #     print(f"图像的尺寸不为256x256,而是{w_ori}x{h_ori},请检查图像是否正确:\n{ff}")
    # continue

    # print(f"图像的尺寸",[w_ori, h_ori ])
    scale_x = 1000 / w_ori
    scale_y = 1000 / h_ori
    img_ori_1k = img_ori.resize((1000,1000))
    # Example usage: #change thie image
    tensor,w,h = image_to_tensor(image_path)
    # tensor,w,h = image_to_tensor2(image_path)
    tensor=tensor.unsqueeze(0)
    if onnx_:
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(tensor),
            # ort_session.get_inputs()[1].name: to_numpy(dummy_grid)
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        # 转换为PyTorch格式
        onnx_pred_logits = torch.from_numpy(ort_outputs[0])
        onnx_pred_boxes = torch.from_numpy(ort_outputs[1])
        # 构建与原模型一致的输出字典
        onnx_output_dict = {
            "pred_logits": onnx_pred_logits,
            "pred_boxes": onnx_pred_boxes,
        }

    ori_size=torch.Tensor([w,h]).long().unsqueeze(0)
    # result_ = postprocessor(outputs_tensor, ori_size)
    result_ = postprocessor(onnx_output_dict, ori_size)

    score_=result_[0]['scores']
    boxe_=result_[0]['boxes']
    label_=result_[0]['labels']
    selected_indices =score_ > box_thresh
    output={
    'labels': label_[selected_indices].to("cpu").numpy(),
    'boxes': boxe_[selected_indices].to("cpu").numpy(),
    'scores': score_[selected_indices].to("cpu").numpy()
    }
    center_coords=bbox2center(output['boxes'])
    output = {'bbox':         output["boxes"],
            'bbox_centers': center_coords,
            'scores':       output["scores"],
        'pred_classes': output["labels"]}
    output_a, output_b, output_c= ouptnp2abc(output,idx_to_labels)



    if debug:print("c,a,b>>>>>",len(output_c['pred_classes']),len(output_a['pred_classes']),len(output_b['pred_classes']))
    if len(output_a['pred_classes'])==0:
        file_path = 'Check_AboxIs0.txt'
        content = f'{image_path}@@{id_}---{image_path}\n'
        # 文件存在则追加写入，不存在则创建并写入
        # with open(file_path, 'a', encoding='utf-8') as f:
        #     f.write(content)
        # # continue #may need manulay labeling

    overlap_records = []
    to_remove = set()
    bond_boxes = output_b['bbox']

    bboxes = output_a['bbox'].copy()
    a_center = output_a['bbox_centers'].copy()

    scores = output_a['scores'].copy()
    pred_classes = output_a['pred_classes'].copy()
    to_remove = set()

    # 计算所有 atom bbox 之间的 IoU, 并根据 IoU 进行处理
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            # iou, relationship, inter_area, union_area = calculate_iou(bboxes[i], bboxes[j])
            x_min1, y_min1, x_max1, y_max1 = bboxes[i]
            x_min2, y_min2, x_max2, y_max2 = bboxes[j]
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
            score_i = scores[i] if scores[i] is not None else -1
            score_j = scores[j] if scores[j] is not None else -1
            # 完全重合
            if iou == 1:
                if score_i > score_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
            elif iou>=0.8 and iou <1.0:#NOTE fix me if not right
                if score_i > score_j:
                    to_remove.add(j)
                    if debug: print([i,j,score_i,score_j],iou,f"will remove j {j}, i-j {i,j}")
                else:
                    to_remove.add(i)
                    if debug: print([i,j,score_i,score_j],iou,f"will remove i {i}, i-j {i,j} ")

            # 包含关系
            elif iou > 0 and iou < 0.89 :
                if debug: print([i,j,score_i,score_j],iou,"<<<<<<111")
                if inter_area == area1 and area1 < area2:  # bbox[j] 包含 bbox[i]
                    large_idx, small_idx = j, i
                elif inter_area == area2 and area2 < area1:  # bbox[i] 包含 bbox[j]
                    large_idx, small_idx = i, j
                else:
                    if debug: print([i,j,score_i,score_j],iou,'OVERLAP without processed this version')
                    continue
                # 检查是否包含 bond box
                contains_bond = False
                for bond_bbox in bond_boxes:
                    if is_contained_in(bond_bbox, bboxes[large_idx]):
                        contains_bond = True
                        # 调整较大 bbox
                        bboxes[large_idx] = adjust_bbox1(bboxes[large_idx], bboxes[small_idx], bond_bbox)
                        # to_remove.add(small_idx)
                        break
                if not contains_bond:
                    to_remove.add(small_idx)#NOTE use the cutoff >0.45, 
            elif iou==0:#==0
                pass
            else:
                print([i,j,score_i,score_j],iou,"<<<<<<222")
                print('what this case ???')   

    # 删除被移除的 bbox
    atom_bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in to_remove]
    atom_scores = [scores[i] for i in range(len(scores)) if i not in to_remove]
    atom_centers = [a_center[i] for i in range(len(a_center)) if i not in to_remove]
    atom_classes = [pred_classes[i] for i in range(len(pred_classes)) if i not in to_remove]
    #TODO need sort box with x first, then y dim, useful for * with multi neiborbond
    #  Sort atom_bboxes and atom_scores by x1 (bbox[0]) first, then y1 (bbox[1])
    sorted_indices = sorted(range(len(atom_bboxes)), key=lambda i: (atom_bboxes[i][0], atom_bboxes[i][1]))
    atom_bboxes = [atom_bboxes[i] for i in sorted_indices]
    atom_scores = [atom_scores[i] for i in sorted_indices]
    atom_centers = [atom_centers[i] for i in sorted_indices]
    atom_classes = [atom_classes[i] for i in sorted_indices]

    print(len(atom_classes),'xxxxxxxx')
    bond_bbox = output_b['bbox'].copy()
    bond_scores = output_b['scores'].copy()
    bond_classes = output_b['pred_classes'].copy()

    # atom_bbox=final_bboxes
    bonds = dict()
    b2aa = dict()
    singleAtomBond = dict()
    bondWithdirct = dict()
    _margin = 0
    bond_direction = dict()

    # Preprocess atom boxes
    atom_centers_, atom_bbox_, original_to_subbox, subbox_to_original = preprocess_atom_boxes(atom_centers, atom_bboxes)
    # Build KDTree with updated atom centers
    tree_atom = KDTree(atom_centers_)#have to includ the splited box
    if debug:
        print(f"KDTree built with {len(atom_centers_)} atom centers")

    for bi, (bbox, bond_type) in enumerate(zip(bond_bbox, bond_classes)):
        score = bond_scores[bi]
        if score is None:
            score = 1.0  # From manual addition
            bond_scores[bi] = score

        anchor_positions = (np.array(bbox) + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # Query KDTree for nearest atoms
        dists, neighbours = tree_atom.query(anchor_positions, k=1)
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            begin_idx, end_idx = neighbours[:2]
        else:
            begin_idx, end_idx = neighbours[2:]

        # Map sub-box indices back to original atom indices
        atom1_idx = int(subbox_to_original[int(begin_idx)])
        atom2_idx = int(subbox_to_original[int(end_idx)])

        if atom1_idx == atom2_idx:
            if debug:
                print(f"singleAtomBond detected with bond id:{bi} atomIdx1 == atomIdx2 ::{[atom1_idx, atom2_idx]}")
            singleAtomBond[bi] = [atom1_idx]

        min_ai = min([atom1_idx, atom2_idx])
        max_ai = max([atom1_idx, atom2_idx])

        if debug:
            print(f"Bond {bi}: [{min_ai}, {max_ai}]")

        # Assign bond type
        if bond_type in ['single', 'wdge', 'dash', '-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bond_ = [min_ai, max_ai, 'SINGLE', score]
            if bond_type in ['wdge', 'dash', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                bondWithdirct[bi] = [min_ai, max_ai, 'SINGLE', score, bond_type]
        elif bond_type == '=':
            bond_ = [min_ai, max_ai, 'DOUBLE', score]
        elif bond_type == '#':
            bond_ = [min_ai, max_ai, 'TRIPLE', score]
        elif bond_type == ':':
            bond_ = [min_ai, max_ai, 'AROMATIC', score]
        else:
            if debug:
                print(f"Unknown bond_type: {bond_type} for bond {bi} [{min_ai, max_ai}]")
            bond_ = [min_ai, max_ai, 'SINGLE', score]

        bonds[bi] = bond_
        b2aa[bi] = sorted([min_ai, max_ai])

    if debug:
        print(f"bonds {len(bonds)}, b2aa {len(b2aa)}, singleAtomBond {len(singleAtomBond)}, bondWithdirct {len(bondWithdirct)}")


    #try to set up a2b, baesed on bond-anchor_positions--atom center relationship
    a2b=dict()#may be updated as following singleAtomBond cases process
    isolated_a=set()
    aa2b_d2=dict()
    for k,v in b2aa.items():
        vt=(v[0],v[1])
        if vt in aa2b_d2:
            aa2b_d2[vt].append(k)
        else:
            aa2b_d2[vt]=[k]
        
        for a in set(v):
            if a not in a2b.keys():
                a2b[a]=[k]
            else:
                a2b[a].append(k)

    # 初始化 a2neib, iso_lated atom box and singleAtomBond box process need
    a2neib = {}
    # 遍历 a2b，构建邻居关系
    for atom, bns in a2b.items():
        neighbors = set()  # 使用集合去重
        for bond in bns:
            atom_pair = b2aa[bond]  # 获取 bond 连接的原子对
            # 如果当前原子在 atom_pair 中，添加另一个原子作为邻居
            nei={ai for ai in atom_pair if ai !=atom }
            neighbors.update(nei)
            # if atom in atom_pair:
            #     other_atom = atom_pair[0] if atom == atom_pair[1] else atom_pair[1]
            #     neighbors.add(other_atom)
        a2neib[atom] = sorted(list(neighbors))  # 转换为有序列表

    #check isolated atom exsit, if need add bond for isloated atom box when overlaping with other atom box
    isolated_a=set()
    for ai, a_lab in enumerate(atom_classes):
        if ai not in a2b.keys():
            isolated_a.add(ai)
    if debug:print("detected possible isolated atom:", isolated_a)


    repeate_bonds={k:v for k,v in aa2b_d2.items() if len(v)>=2 }
    if debug:print(f"repeat bond box ids {repeate_bonds}")
    #get the minimu size of bond box, check isolated_a atom box overlap with other atom box, if overlap, then add bond box (default bond label with single, score 1.0) between them
    # update a2b,b2aa, and bond box bond_classes, elif not box not overlap, the isolated_a box min(4 point of box cornners to other atom box connrer) enough small than the existed bond box size
    if len(isolated_a)>0:
        isolated_a2del=[]
        # 计算现有键的最小尺寸
        bond_sizes = []
        for bbox in bond_bbox:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = min(width, height)  # 使用较小边作为键的尺寸
            bond_sizes.append(size)
        min_bond_size = min(bond_sizes) if bond_sizes else 10.0  # 默认值若无键
        if debug:print("min_bond_size ",min_bond_size, 10)
        new_bond_idx = len(bond_bbox)
        isolated_aFound=[]
        singleAtomBond_fixed=[]
        # at2b_dist=dict()

        for iso_atom in isolated_a:
            iso_box = atom_bboxes[iso_atom]
        
            #with SingleAtomBond first then check with other atom box, may a1a2 repeat on >=two bonds
            for bi,atom_idx_list in singleAtomBond.items():
                bond_box = bond_bbox[bi]
                atom1_idx = atom_idx_list[0]
                bond_vertices = get_corners(bond_box)
                # 计算 atom1_center 到 bond box 4 个顶点的距离
                atom1_center = atom_centers[atom1_idx]
                distances = [np.linalg.norm(np.array(atom1_center) - v) for v in bond_vertices]
                closest_indices = np.argsort(distances)[:2] # 距离最小的两个顶点
                connected_vertices = bond_vertices[closest_indices]
                unconnected_vertices = bond_vertices[[i for i in range(4) if i not in closest_indices]]
                # exclude_=a2neib[atom1_idx]
                exclude_=[atom1_idx]+a2neib[atom1_idx]#add it self
                print(f'exclude this atom itself :: {exclude_},and its neiboughs {a2neib[atom1_idx]}')
                # 找到 atom2（未连接端到所有 atom box 顶点的最小距离）
                # _, atom2_idx_ = get_min_distance_to_atom_box(unconnected_vertices, atom_bboxes, exclude_idx=exclude_)
                atom2_idx_, dist2 = find_nearest_atom(unconnected_vertices, atom_bboxes, exclude_idx=exclude_)
                if iso_atom == atom2_idx_:
                    # 从 atom1 找到最近的另一个 atom (atom2_1)
                    if atom2_idx_< atom1_idx:
                        k=[atom2_idx_, atom1_idx]
                    else:
                        k=[atom1_idx, atom2_idx_]
                    
                    if atom2_idx_ not in a2neib[atom1_idx]:
                        b2aa[bi]=k
                        bonds[bi][0]=k[0]
                        bonds[bi][1]=k[1]
                        a2b.setdefault(iso_atom, []).append(bi)

                        if debug: print(f'@@isolated_a fix the SingleAtomBond {bi} as bond:{bonds[bi]} !!')
                        singleAtomBond_fixed.append(bi)
                        isolated_aFound.append(atom2_idx_)
        
            if len(repeate_bonds)>0:
                at2b_dist=dict()#NOTE the case repeate bonds with isolated atom box
                iso_box_vertices = get_corners(iso_box)
                iso_atom_center = atom_centers[iso_atom]
                bond_box_idx_, bond_box_dist = find_nearest_atom(iso_box_vertices, bond_bbox, exclude_idx=[])
                for a1a2,bis in repeate_bonds.items():#{(2, 3): [3, 4]} 
                    for bi in bis:
                        if bi ==bond_box_idx_:
                            bond_box = bond_bbox[bi]
                            bond_vertices = get_corners(bond_box)
                            a1_,a2_=a1a2
                            a1_atombox= atom_bboxes[a1_]
                            a2_atombox= atom_bboxes[a2_]
                            a1_flag= boxes_overlap(a1_atombox, bond_box)
                            a2_flag= boxes_overlap(a2_atombox, bond_box)
                            if a1_flag: 
                                atom1_idx_=a1_
                                dist1=0
                            elif a2_flag: 
                                atom1_idx_=a2_
                                dist1=0
                            else:
                                distances = [np.linalg.norm(np.array(iso_atom_center) - v) for v in bond_vertices]
                                closest_indices2 = np.argsort(distances)[:2] # 距离最小的两个顶点
                                connected_vertices2 = bond_vertices[closest_indices2]#isolated_close
                                connected_vertices1 = bond_vertices[[i for i in range(4) if i not in closest_indices2]]
                                atom1_idx_, dist1 = find_nearest_atom(connected_vertices1, atom_bboxes, exclude_idx=[iso_atom])
                            if debug:print("a1_flag,a2_flag,atom1_idx_, iso_atom",[a1_flag,a2_flag,atom1_idx_,iso_atom])
                            min_ai=min([atom1_idx_,iso_atom])
                            max_ai=max([atom1_idx_,iso_atom])
                            k=(min_ai,max_ai)
                            print(k,'repeate',bi)
                            if k not in at2b_dist:
                                at2b_dist[k]=[bi,a1a2,dist1]
                            else:
                                if dist1< at2b_dist[k][1]:
                                    at2b_dist[k]=[bi,a1a2,dist1]
                            if debug:print(f"repate bond box id: {bi} fixed with {at2b_dist}")
                            isolated_aFound.append(iso_atom)
                            # for k,v in at2b_dist.items():
                            #update bond atom box mapping
                            isolated_a2del.append(iso_atom)
                            b2aa[bi] = [min_ai,max_ai]
                            a2b.setdefault(iso_atom, []).append(bi)
                            bonds[bi][0]=k[0]
                            bonds[bi][1]=k[1]
                            if bi in bondWithdirct:
                                bondWithdirct[bi][0]=k[0]
                                bondWithdirct[bi][1]=k[1]

        isolated_a=[ ai for ai in isolated_a if ai not in isolated_aFound]#updated
        singleAtomBond={bi:aili for bi,aili in singleAtomBond.items() if bi not in singleAtomBond_fixed}#updated

        for iso_atom in isolated_a:
            iso_box = atom_bboxes[iso_atom]
            #with SingleAtomBond first then chec
            for other_idx, other_box in enumerate(atom_bboxes):
                if other_idx == iso_atom\
                    or (atom_classes[other_idx] in ['other',"*"] and atom_classes[iso_atom] in ['other',"*"]):
                    #also not inlcude other -- *
                    continue
                # 检查重叠
                min_ai=min([iso_atom,other_idx])
                max_ai=max([iso_atom,other_idx])

                if boxes_overlap(iso_box, other_box):
                    # 添加默认单键
                    new_bbox = [
                        min(iso_box[0], other_box[0]),
                        min(iso_box[1], other_box[1]),
                        max(iso_box[2], other_box[2]),
                        max(iso_box[3], other_box[3])
                    ]
                    bond_bbox.append(new_bbox)
                    bond_classes.append('single')
                    bond_scores.append(1.0)
                    b2aa[new_bond_idx] = [iso_atom, other_idx]
                    a2b.setdefault(iso_atom, []).append(new_bond_idx)
                    a2b.setdefault(other_idx, []).append(new_bond_idx)
                    isolated_a2del.append(iso_atom)
                    new_bond_idx += 1
                    bond_=[min_ai, max_ai, 'SINGLE', 1.0]
                    last_=len(bonds)
                    bonds[last_] = bond_

                    if debug:
                        print(f"添加键 {new_bond_idx-1} 连接原子 {iso_atom} 和 {other_idx},as isoated box overlap with it ")
                    # break
                else:
                    # 检查角点最小距离
                    min_dist = float('inf')
                    closest_atom = None
                    dist = min_corner_distance(iso_box, other_box)
                    if dist < min_dist:
                        min_dist = dist
                        closest_atom = other_idx
                    if min_dist < min_bond_size:
                        # 添加默认单键
                        new_bbox = [
                            min(iso_box[0], atom_bboxes[closest_atom][0]),
                            min(iso_box[1], atom_bboxes[closest_atom][1]),
                            max(iso_box[2], atom_bboxes[closest_atom][2]),
                            max(iso_box[3], atom_bboxes[closest_atom][3])
                        ]
                        bond_bbox.append(new_bbox)
                        bond_classes.append('single')
                        bond_scores.append(1.0)
                        b2aa[new_bond_idx] = [iso_atom, closest_atom]
                        a2b.setdefault(iso_atom, []).append(new_bond_idx)
                        a2b.setdefault(closest_atom, []).append(new_bond_idx)
                        isolated_a2del.append(iso_atom)
                        new_bond_idx += 1
                        if debug:
                            print(f"添加键 {new_bond_idx-1} 连接原子 {iso_atom} 和 {closest_atom} (距离 {min_dist})")
                        bond_=[min_ai, max_ai, 'SINGLE', 1.0]
                        last_=len(bonds)
                        bonds[last_] = bond_

                        # break#as isolated may be get more than 2 bonds
        if debug:
            print('isolated_a2del and isolated_a number',len(isolated_a2del),len(isolated_a))
            print('isolated_a ',isolated_a)
            print('isolated_a2del ',isolated_a2del)
        
    a2b = dict(sorted(a2b.items()))

    # 先处理 singleAtomBond， 再removed duplicated
    if len(singleAtomBond) > 0:
        # 初始化 a2neib
        a2neib = {}
        # 遍历 a2b，构建邻居关系
        for atom, bns in a2b.items():
            neighbors = set()  # 使用集合去重
            for bond in bns:
                atom_pair = b2aa[bond]  # 获取 bond 连接的原子对
                # 如果当前原子在 atom_pair 中，添加另一个原子作为邻居
                nei={ai for ai in atom_pair if ai !=atom }
                neighbors.update(nei)
                # if atom in atom_pair:
                #     other_atom = atom_pair[0] if atom == atom_pair[1] else atom_pair[1]
                #     neighbors.add(other_atom)
            a2neib[atom] = sorted(list(neighbors))  # 转换为有序列表

        # 找到所有 C 的 bbox 尺寸
        c_bboxes = [bbox for bbox, cls in zip(output_a['bbox'], output_a['pred_classes']) if cls == 'C']
        if not c_bboxes:
            # 如果没有C原子，使用所有bbox中最小的
            print("Warning: No 'C' atoms found, using smallest bbox in output_a instead.")
            all_bboxes = output_a['bbox']
            if not all_bboxes:
                raise ValueError("No bboxes found in output_a at all.")            
            smallest_bbox = min(all_bboxes, key=bbox_area)
            c_bboxes = [smallest_bbox]    # 计算最小宽度和高度
        min_width = min([bbox[2] - bbox[0] for bbox in c_bboxes])
        min_height = min([bbox[3] - bbox[1] for bbox in c_bboxes])
        
        # 处理 singleAtomBond
        for bi, atom_idx_list in singleAtomBond.items():
            bond_box = bond_bbox[bi]
            atom1_idx = atom_idx_list[0]
            bond_vertices = get_corners(bond_box)
            # 计算 atom1_center 到 bond box 4 个顶点的距离
            atom1_center = atom_centers[atom1_idx]
            distances = [np.linalg.norm(np.array(atom1_center) - v) for v in bond_vertices]
            closest_indices = np.argsort(distances)[:2] # 距离最小的两个顶点
            connected_vertices = bond_vertices[closest_indices]
            unconnected_vertices = bond_vertices[[i for i in range(4) if i not in closest_indices]]
            # exclude_=a2neib[atom1_idx]
            exclude_=[atom1_idx]#add it self
            print(f'exclude this atom itself :: {exclude_},and its neiboughs {a2neib[atom1_idx]}')
            # 找到 atom2（未连接端到所有 atom box 顶点的最小距离）
            # _, atom2_idx_ = get_min_distance_to_atom_box(unconnected_vertices, atom_bboxes, exclude_idx=exclude_)
            atom2_idx_, dist2 = find_nearest_atom(unconnected_vertices, atom_bboxes, exclude_idx=exclude_)
            # 从 atom1 找到最近的另一个 atom (atom2_1)
            atom1_corners = get_corners(atom_bboxes[atom1_idx])
            atom2_1_idx, dist2_1 = find_nearest_atom(atom1_corners, atom_bboxes, exclude_idx=exclude_)
            if debug:print("atom2_idx_ , atom2_1_idx,atom1_idx:",atom2_idx_, atom2_1_idx,atom1_idx)
            if atom2_idx_< atom1_idx:
                k=[atom2_idx_, atom1_idx]
            else:
                k=[atom1_idx, atom2_idx_]

            if atom2_idx_ == atom2_1_idx :
                if atom2_idx_ not in a2neib[atom1_idx]:
                    if debug: print('add new bond with existed atom')
                    b2aa[bi]=k
                    bonds[bi][0]=k[0]
                    bonds[bi][1]=k[1]
                else:#need insert new atom box at this bond terminal site, default with C
                    new_center=np.mean(unconnected_vertices, axis=0)
                    # 生成新 C 的 bbox
                    new_bbox = [
                        new_center[0] - min_width / 2,
                        new_center[1] - min_height / 2,
                        new_center[0] + min_width / 2,
                        new_center[1] + min_height / 2
                    ]
                    if debug: print('new atom box adding as C')
                    atom_bboxes.append(new_bbox)
                    atom_centers.append(new_center.tolist())
                    atom_scores.append(bond_scores[bi])  # 使用 bond 的 score
                    atom_classes.append('C')
                    #updating 
                    atom2_idx_= len(atom_classes)-1
                    k=[atom1_idx, atom2_idx_]
                    bonds[bi][1]=atom2_idx_
                    b2aa[bi][1]=atom2_idx_
            else:     #atom2_idx_ != atom2_1_idx, keep atom2_idx_ from bond box privlage
                if atom2_idx_ not in a2neib[atom1_idx]:
                    if debug: print(f'atom2_idx_ != atom2_1_idx| {atom2_idx_} != {atom2_1_idx} @add new bond with existed atom')
                    b2aa[bi]=k
                    bonds[bi][0]=k[0]
                    bonds[bi][1]=k[1]
                else:#need insert new atom box at this bond terminal site, default with C
                    new_center=np.mean(unconnected_vertices, axis=0)
                    # 生成新 C 的 bbox
                    new_bbox = [
                        new_center[0] - min_width / 2,
                        new_center[1] - min_height / 2,
                        new_center[0] + min_width / 2,
                        new_center[1] + min_height / 2
                    ]
                    atom_bboxes.append(new_bbox)#updateing atom box
                    atom_centers.append(new_center.tolist())
                    atom_scores.append(bond_scores[bi])  # 使用 bond 的 score
                    atom_classes.append('C')
                    #updating 
                    atom2_idx_= len(atom_classes)-1
                    k=[atom1_idx, atom2_idx_]
                    bonds[bi][1]=atom2_idx_
                    b2aa[bi][1]=atom2_idx_
                    if debug: print(f'atom2_idx_ != atom2_1_idx@new atom box {atom2_idx_}adding as C, with bond {bi} a1a2 {k}')
                
            if bi in bondWithdirct.keys():
                bondWithdirct[bi][0]=k[0]
                bondWithdirct[bi][1]=k[1]#update atom2 index

    #TODO， fix me, this case, may need ocr.ocr first, try to dicide need isolated atom added bond times
    if debug:print(f"before del bonds  {len(bond_bbox)}")
    # viewcheck_b(image_path,bond_bbox,bond_classes,color='green',figsize=(10,7))
    # viewcheck(image_path,atom_bboxes,color='red')
    #update aa2b for remove duplicated bonds
    aa2b=dict()
    for bi, aa in b2aa.items():
        min_ai=min(aa)
        max_ai=max(aa)
        if bond_scores[bi] is None:
            bond_scores[bi]=1.0
        score_=bond_scores[bi]
        bond_type=bond_classes[bi]
        # print([bond_type,score_])
        #bond_type check afte singleAtomBond
        if bond_type in ['single','wdge','dash', '-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bond_ = [min_ai, max_ai, 'SINGLE', score]
            if bond_type in ['wdge','dash','ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                bondWithdirct[bi]=[min_ai, max_ai,'SINGLE', score, bond_type]
        elif bond_type == '=':
            bond_ = [min_ai, max_ai, 'DOUBLE', score]
            # print(bond_,"@@@@")
        elif bond_type == '#':
            bond_ = [min_ai, max_ai, 'TRIPLE', score]
        elif bond_type == ':':
            bond_ = [min_ai, max_ai, 'AROMATIC', score]
        else:
            print(f"what case here !!! with bond_type: {bond_type} || {[bi,min_ai, max_ai]}")
            bond_=[min_ai, max_ai, 'SINGLE', score]

        if (min_ai, max_ai) not in aa2b.keys() or aa2b[(min_ai, max_ai)][-2]<score_:
            aa2b[(min_ai, max_ai)]=[bi,score_,bond_[-2]]
            #SINGEL Atom bond 本来是不重复的，会误认repeate and remove TODO

    #remove duplicated bonds based on score
    if len(aa2b)!=len(b2aa):
        # 1. 去重并生成新的 bi 映射
        new_bi_map = {}  # 格式: {old_bi: new_bi}
        new_bonds = {}
        new_aa2b = {}
        new_b2aa = {}
        new_bondWithdirct = {}
        new_singleAtomBond = {}
        # 按 aa2b 的顺序分配新 bi（保留分数高的键）
        for new_bi, ((min_ai, max_ai), (old_bi, score, bond_type)) in enumerate(
            sorted(aa2b.items(), key=lambda x: x[1][1], reverse=True)  # 按分数降序排序
        ):
            new_bi_map[old_bi] = new_bi
            new_bonds[new_bi] = [min_ai, max_ai, bond_type, score]
            new_aa2b[(min_ai, max_ai)] = [new_bi, score, bond_type]
            new_b2aa[new_bi] = [min_ai, max_ai]

        # 2. 更新 bondWithdirct & singleAtomBond
        for old_bi, bond_info in bondWithdirct.items():
            if old_bi in new_bi_map:
                new_bi = new_bi_map[old_bi]
                new_bondWithdirct[new_bi] = bond_info

        for old_bi, bond_info in singleAtomBond.items():
            if old_bi in new_bi_map:
                new_bi = new_bi_map[old_bi]
                new_singleAtomBond[new_bi] = bond_info
                
        # 3. 替换旧数据结构, TODO ad bond box class scores here
        bonds = new_bonds
        aa2b = new_aa2b
        b2aa = new_b2aa
        bondWithdirct = new_bondWithdirct
        singleAtomBond = new_singleAtomBond
        if debug: print(f"去重完成: bonds={len(bonds)}, aa2b={len(aa2b)}, b2aa={len(b2aa)}, bondWithdirct={len(bondWithdirct)}")
        #remove duplicated bonds based on score
        # 4. 更新 bond_bbox, bond_scores, bond_classes
        old_bns=max(new_bi_map.keys())
        to_remove_bonds=set()
        for i in range(old_bns):
            if i not in new_bi_map.keys():
                to_remove_bonds.add(i)
        print(to_remove_bonds)
        # 删除被移除的 bbox
        bond_scores = [bond_scores[i] for i in range(len(bond_scores)) if i not in to_remove_bonds]
        bond_classes = [bond_classes[i] for i in range(len(bond_classes)) if i not in to_remove_bonds]
        bond_bbox = [bond_bbox[i] for i in range(len(bond_bbox)) if i not in to_remove_bonds]
        bond_center = [[ (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2 ] for bbox in bond_bbox]
        

    a2b=dict()
    isolated_a=set()
    for k,v in b2aa.items():
        # a1,a2=v
        for a in v:
            if a not in a2b.keys():
                a2b[a]=[k]
            else:
                a2b[a].append(k)

    for ai, a_lab in enumerate(atom_classes):
        if ai not in a2b.keys():
            isolated_a.add(ai)
    a2b = dict(sorted(a2b.items()))

    # 初始化 a2neib
    a2neib = {}
    # 遍历 a2b，构建邻居关系
    for atom, bns in a2b.items():
        neighbors = set()  # 使用集合去重
        for bond in bns:
            atom_pair = b2aa[bond]  # 获取 bond 连接的原子对
            # 如果当前原子在 atom_pair 中，添加另一个原子作为邻居
            nei={ai for ai in atom_pair if ai !=atom }
            neighbors.update(nei)
            # if atom in atom_pair:
            #     other_atom = atom_pair[0] if atom == atom_pair[1] else atom_pair[1]
            #     neighbors.add(other_atom)
        a2neib[atom] = sorted(list(neighbors))  # 转换为有序列表

    debug2=False
    if debug2:
        # 输出结果
        print("\nBonds:")
        for bi, bond_info in bonds.items():
            print(f"Bond {bi}: {bond_info}")
        print("\nSingle Atom Bonds:")
        for bi, atom_idx in singleAtomBond.items():
            print(f"Bond {bi}: {atom_idx}")
        print("Atom to Bonds box idx maping:")
        for ai, bond_ids in a2b.items():
            print(f"a2b-id {ai}: {bond_ids}")
        print(f"isolated_ atom box:: {isolated_a}")
        print(f"b2aa::{b2aa}")
        # 输出结果
        print("a2neib:")
        for atom, neighbors in a2neib.items():
            print(f"Atom {atom}: {neighbors}")

    other2ppsocr = True
    ocr_ai2lab = dict()
    ocr_bbs = dict()
    scale_crop = False
    ocr_ai2lab_ori=dict()
    ocr_ai2lab_sca=dict()


    if other2ppsocr:
        elements = ['S', 'N', 'P', 'C', 'O']
        keys = [f"{e}{suffix}" for e in elements for suffix in ['R"', "R'", "R", "*"]]
        replacement_map = {key: f'{key[0]}*' for key in keys}
        if da=='staker':
            _margin=2#as staker use small image 256X256
        else:
            _margin=0
        for i, atc in enumerate(atom_classes):
            if 'other' == atc:  # 30 idx_lab version OH-->Cl with high 
                # Initialize variables to store both results
                orig_result = None
                orig_score = 0
                scaled_result = None
                scaled_score = 0
                
                # Process original image crop
                abox_orig = np.array(atom_bboxes[i]) + np.array([-_margin, -_margin,_margin, _margin])
                cropped_img_orig = img_ori.crop(abox_orig)
                image_npocr_orig = np.array(cropped_img_orig)
                result_ocr_orig = ocr.ocr(image_npocr_orig, det=False)
                
                if result_ocr_orig:
                    orig_text = result_ocr_orig[0][0][0]
                    orig_score = result_ocr_orig[0][0][1]
                    if debug: print(f'oriCrop:\t {orig_text}   {orig_score}')
                    orig_text = normalize_ocr_text(orig_text, replacement_map)
                    ocr_ai2lab_ori[i]=[orig_text,orig_score]
                # Process scaled image crop
                abox_scaled = np.array(atom_bboxes[i]) * np.array([scale_x, scale_y, scale_x, scale_y]) +  np.array([-_margin, -_margin,_margin, _margin])
                cropped_img_scaled = img_ori_1k.crop(abox_scaled)
                image_npocr_scaled = np.array(cropped_img_scaled)
                result_ocr_scaled = ocr.ocr(image_npocr_scaled, det=False)
                
                if result_ocr_scaled:
                    scaled_text = result_ocr_scaled[0][0][0]
                    scaled_score = result_ocr_scaled[0][0][1]
                    if debug:  print(f'scaled:\t {scaled_text}   {scaled_score}')
                    scaled_text = normalize_ocr_text(scaled_text, replacement_map)
                    ocr_ai2lab_sca[i]=[scaled_text,scaled_score]

                

                final_text, final_score, final_crop = select_chem_expression(
                    orig_text, orig_score, scaled_text, scaled_score, cropped_img_orig, cropped_img_scaled
                )

                if orig_text=='NO2' or scaled_text=='NO2':
                    final_text='NO2'#AS stm NO score >NO2
                elif orig_text=='SO2' or scaled_text=='SO2':
                    final_text='SO2'#AS stm NO score >SO2
                # elif orig_starts_upper == scaled_starts_upper:
                #     # If both start with uppercase or both don't, use the higher score
                #     final_text = orig_text if orig_score >= scaled_score else scaled_text
                # elif orig_starts_upper != scaled_starts_upper:
                #     # If one starts with uppercase, use that one
                #     final_text = orig_text if orig_starts_upper else scaled_text

                if final_text:
                    ocr_ai2lab[i] = [final_text, final_score]
                    ocr_bbs[i] = final_crop
                    atom_classes[i] = final_text
        if debug:
            print("ori",ocr_ai2lab_ori)
            print("sca",ocr_ai2lab_sca)
        print(ocr_ai2lab)
        #TODO make  chem-group  recongized dataBase next works !!!

    if len(ocr_bbs)>0:
        if debug:print(f'numbs of ocr {len(ocr_bbs)} crop_ images')
    #merge the isolated_a Ph3Br into closet atom box
    # 3 in isolated_a, isolated_a, isolated_aFound
    giveup_isolateds=dict()
    if len(isolated_a):#after updated isolated_a still has the isolatd item
        for iso_atom in isolated_a:
            atom1_corners = get_corners(atom_bboxes[iso_atom])
            atom2_1_idx, dist2_1 = find_nearest_atom(atom1_corners, atom_bboxes, exclude_idx=[iso_atom])
            atom1_lab=atom_classes[iso_atom]
            atom2_lab=atom_classes[atom2_1_idx]
            if atom1_lab in ['Ph3Br','Ph3Br-']:
                if iso_atom not in giveup_isolateds.keys():
                    giveup_isolateds[iso_atom]=[atom1_lab]
                else:
                    giveup_isolateds[iso_atom].append(atom1_lab)
                
                if atom2_lab in ['P','P+']:#merge as new group
                    atom2_lab='P+Ph3Br-'
                elif atom2_lab in ['N','N+']:#merge as new group
                    atom2_lab='N+Ph3Br-'

            atom_classes[atom2_1_idx]=atom2_lab #update bonded atom label with the merged
            
                #TODO add cases that need merge OCR results with bonded atom box
    if debug:
        print(f"giveup_isolateds {giveup_isolateds}")
        print(len(atom_classes),len(bond_classes),'<<<<<<<<<<<')#,len(charges_classes))
    ###########################start build mol ##########################
    rwmol_ = Chem.RWMol()
    boxi2ai = {}  # 预测索引 -> RDKit 索引
    placeholder_atoms=dict()
    # print(len(atom_classes),len(bond_classes))#,len(charges_classes))
    #assign atom
    J=0
    for i, (bbox, a) in enumerate(zip(atom_bboxes, atom_classes)):
        a2labl=False
        a=replace_cg_notation(a)
        # print(a,'atom box class label')
        if a in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:#  '*', I2M's defined atom types
            # if a=='H':continue#skip H fristly,only with heavy atom then addH 
            ad = Chem.Atom(a)#TODO consider non chemical group and label for using
        #TODO add pd rdkit known elemetns here
        elif a in ELEMENTS:
            ad = Chem.Atom(a)
            
        elif a in ABBREVIATIONS :
            ad = Chem.Atom("*")
            placeholder_atoms[i] = a # 记录非标准原但有定义的官能团   类型及其位置,
            a2labl=True
        else:
            if  N_C_H_expand(a):
                ad = Chem.Atom("*")
                placeholder_atoms[i] = a # 记录非标准原但有定义的官能团   类型及其位置,
                a2labl=True
            elif C_H_expand(a):
                ad = Chem.Atom("*")
                placeholder_atoms[i] = a # 记录非标准原但有定义的官能团   类型及其位置,
                a2labl=True
            elif C_H_expand2(a):
                        ad = Chem.Atom("*")
                        placeholder_atoms[i] = a # 记录非标准原但有定义的官能团   类型及其位置,
                        a2labl=True
                            
            elif  formula_regex(a):
                ad = Chem.Atom("*")
                placeholder_atoms[i] = a # 记录非标准原但有定义的官能团   类型及其位置,
                a2labl=True
            else:
                ad = Chem.Atom("*")
                if a not in ['*',"other"]:
                    a2labl=True
            # placeholder_atoms[idx] = a  
        # atom = Chem.Atom(symbol)
        rwmol_.AddAtom(ad)
        boxi2ai[J] = rwmol_.GetNumAtoms() - 1
        if a2labl: rwmol_.GetAtomWithIdx(J).SetProp("atomLabel", f"{a}")#mol set with label, mol_rebuild not
        J+=1


    charges_classes= output_c['pred_classes']
    charges_centers= output_c['bbox_centers']
    charges_scores= output_c['scores']
    charges_bbox=  output_c['bbox']
    a2c=dict()
    c2a=dict()

    # #assign charge
    if len(charges_classes) > 0:
        kdt = cKDTree(atom_centers)
        c2a = {}  # 电荷索引到原子索引的映射
        used_atoms = set()  # 跟踪已分配电荷的原子
        for i, charge_box in enumerate(charges_bbox):
            charge_value = parse_charge(charges_classes[i])
            overlapped_atoms = []
            # 检查重叠
            for ai, atom_box in enumerate(atom_bboxes):
                if boxes_overlap(charge_box, atom_box):
                    overlapped_atoms.append(ai)
            if overlapped_atoms:
                # 如果有重叠，选择第一个未使用的原子（假设一个电荷只分配一个原子）
                for ai in overlapped_atoms:
                    if ai not in used_atoms:
                        c2a[i] = ai
                        used_atoms.add(ai)
                        break
            else:
                # 不重叠时，使用角点距离和 KDTree 验证
                x, y = charges_centers[i]
                dist_kdt, ai_kdt = kdt.query([x, y], k=1)
                # 计算角点距离最近的原子
                min_dist = float('inf')
                ai_corner = None
                for ai, atom_box in enumerate(atom_bboxes):
                    dist = min_corner_distance(charge_box, atom_box)
                    if dist < min_dist:
                        min_dist = dist
                        ai_corner = ai
                # 比较 KDTree 和角点距离结果
                if ai_kdt == ai_corner and ai_kdt not in used_atoms:
                    c2a[i] = ai_kdt
                    used_atoms.add(ai_kdt)
                else:
                    # 检查电荷值和原子类型
                    if charge_value != 0:
                        symbol_kdt =atom_classes[ai_kdt]
                        symbol_corner =atom_classes[ai_corner]
                        # 如果电荷值不为零，分配给非C的原子，如果都是非C， 则根据kdt k=1来分配电荷
                        if symbol_kdt == 'C' and symbol_corner != 'C' and ai_corner not in used_atoms:
                            # KDTree 是碳，角点不是碳，优先分配给角点原子
                            c2a[i] = ai_corner
                            used_atoms.add(ai_corner)
                        elif symbol_corner == 'C' and symbol_kdt != 'C' and ai_kdt not in used_atoms:
                            # 角点是碳，KDTree 不是碳，优先分配给 KDTree 原子
                            c2a[i] = ai_kdt
                            used_atoms.add(ai_kdt)
                        else:
                            # 两个都是非碳，或两个都是碳，默认使用 KDTree 结果
                            if ai_kdt not in used_atoms:
                                c2a[i] = ai_kdt
                                used_atoms.add(ai_kdt)
                            elif ai_corner not in used_atoms:
                                # 如果 KDTree 结果已使用，尝试角点结果
                                c2a[i] = ai_corner
                                used_atoms.add(ai_corner)

        #assign charge
        a2c={v:k for k,v in c2a.items()}
        for k,v in a2c.items():
            fc=int(charges_classes[v])
            rwmol_.GetAtomWithIdx(k).SetFormalCharge(fc)
            # if k in placeholder_atoms:
            if atom_classes[k] in ['COO','CO2']:#TODO add more charge if need
                if fc==-1:
                    atom_classes[k]=f"{atom_classes[k]}-"
                    placeholder_atoms[k]=atom_classes[k]
                    atom = rwmol_.GetAtomWithIdx(k)
                    atom.SetProp("atomLabel",placeholder_atoms[k])
                elif fc==1:
                    atom_classes[k]=f"{atom_classes[k]}+"
                    placeholder_atoms[k]=atom_classes[k]
                    atom = rwmol_.GetAtomWithIdx(k)
                    atom.SetProp("atomLabel",placeholder_atoms[k])
                else:
                    print(f"charge adding {fc} @ {atom_classes[v]}")
        print(f'placeholder_atoms {placeholder_atoms}')
    #add bonds
    for bi, bond in bonds.items():
        atom1_idx, atom2_idx, bond_type, score = bond
        if atom1_idx ==atom2_idx:print(f"self bond should be avoid or del on previous process!!")
        # print(f"Adding bond between atoms {atom1_idx} and {atom2_idx} of type {bond_type}")
        if bond_type == 'SINGLE':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
        elif bond_type == 'DOUBLE':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.DOUBLE)
        elif bond_type == 'TRIPLE':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.TRIPLE)
        elif bond_type == 'AROMATIC':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.AROMATIC)
        else:
            print(f"Unknown bond type: {bond_type}")

    if debug:    print(f"all a2b b2a a2c c2a done, start mol built done")
    #set direction 
    if len(bondWithdirct)>0:
        print(f"set bond direction for mollecule ")
        # rwmol_=set_bondDriection(rwmol_,bondWithdirct)

    skeleton_smi = Chem.MolToSmiles(rwmol_) #TODO WEB_dev, use this rwmol_ for display without expand the R groups
    #ASSIGN COORDS
    coords = [(x,-y,0) for x,y in atom_centers]
    coords = tuple(coords)
    coords = tuple(tuple(num / 100 for num in sub_tuple) for sub_tuple in coords)

    mol2D = rwmol_.GetMol()
    mol2D.RemoveAllConformers()
    conf = Chem.Conformer(mol2D.GetNumAtoms())
    conf.Set3D(True)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))
    mol2D.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol2D)
        Chem.AssignStereochemistryFrom3D(mol2D)
        mol_rebuit2d=Chem.RWMol(mol2D) 
    except Exception as e:
        print(e)
        print('before expanding!!! try to sanizemol and assign stereo')
        mol_rebuit2d=Chem.RWMol(rwmol_) 
    if len(giveup_isolateds)>0:
        #clean with remove giveup_isolateds
        # 1. 先为每个原子设置一个“old_index”属性
        for atom in mol_rebuit2d.GetAtoms():
            atom.SetProp('old_index', str(atom.GetIdx()))

        # 2. 删除原子时建议按照降序删除，避免索引变化带来的问题
        for ai in sorted(giveup_isolateds.keys(), reverse=True):
            mol_rebuit2d.RemoveAtom(ai)
            print(f"atom {ai} label {giveup_isolateds[ai]} removed")

        # 3. 删除操作完成后，构建老索引到新索引的映射
        old_to_new = {}
        for atom in mol_rebuit2d.GetAtoms():
            old_idx = int(atom.GetProp('old_index'))
            new_idx = atom.GetIdx()
            old_to_new[old_idx] = new_idx

        if len(placeholder_atoms)>0:#update placeholder_atoms
            placeholder_atoms2=dict()
            for k,v in placeholder_atoms.items():
                placeholder_atoms2[old_to_new[k]]=v

            placeholder_atoms=placeholder_atoms2    
    try:
        SMILESpre = Chem.MolToSmiles(mol_rebuit2d)
    except Exception as e:
        print(f"Error during SMILES generation: {e}")
        SMILESpre = Chem.MolToSmiles(mol_rebuit2d, canonical=False)

        
    if len(placeholder_atoms)>0:
        mol_expan=copy.deepcopy(mol_rebuit2d)
        if debug: print(f'MOL will be expanded with {placeholder_atoms} !!')
        wdbs=[]
        bond_dirs_rev={v:k for k,v in bond_dirs.items()}

        for b in mol_expan.GetBonds():
            bd=b.GetBondDir()
            bt=b.GetBondType()
            # print(bd)
            if bd ==bond_dirs['BEGINDASH'] or  bd==bond_dirs['BEGINWEDGE']:
                a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                wdbs.append([a1,a2,bt,bond_dirs_rev[bd]])
                
        expandStero_smi1,molexp= molExpanding(mol_expan,placeholder_atoms,wdbs,bond_dirs)#TODO fix me whe n multi strings on a atom will missing this ocr infors
        molexp=remove_bond_directions_if_no_chiral(molexp)
        try:
            Chem.SanitizeMol(molexp)
            expandStero_smi=Chem.MolToSmiles(molexp)
        except Exception as e:
            print(f"Error during sanitization: {e}")
            expandStero_smi = expandStero_smi1

        expandStero_smi=remove_SP(expandStero_smi)

    else:
        molexp=mol_rebuit2d
        expandStero_smi=SMILESpre #save into csv files,

    #TODO WEB_dev, now can display mol with expanded abbev from molexp
    new_row = {'file_name':image_path, "SMILESori":SMILESori,
                    'SMILESpre':SMILESpre,
                    'SMILESexp':expandStero_smi, 
                    }

    smiles_data = smiles_data._append(new_row, ignore_index=True)#TODO WEB_dev  task done here, we can save predicted Rdkit Obj or smiles  or display on web
    print(f"final prediction:\n {expandStero_smi}")

    # 调用释放函数
    release_ocr(ocr)
    del ocr
    release_ocr(ocr2)
    del ocr2

    return SMILESpre,expandStero_smi
########################################################################################################################################################################################################################################################################



# predict_from_image(ff='./op300209p-Scheme-c2-4.png')
