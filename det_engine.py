"""
Copyright (c)  All Rights Reserved
by bowen
"""

import json
import math
import os
import sys
import pathlib
from typing import Iterable, List
import random
import itertools

import numpy as np
import pandas as pd
import tqdm
import torch
import torch.amp 
from PIL import Image
# from src.data import CocoEvaluator
# from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
# from src.solver.utils import output_to_smiles, output_to_smiles2
# from src.solver.utils import bbox_to_graph_with_charge, mol_from_graph_with_chiral
# from src.misc.draw_box_utils import draw_objs

# from sklearn.metrics import f1_score
# from src.postprocess.abbreviation_detector import get_ocr_recognition_only
# from src.postprocess.utils_dataset import CaptionRemover
from skimage.measure import label
######################################add metric postprocess
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdchem, RWMol, CombineMols
from rdkit import Chem
from rdkit.Chem import rdFMCS
import copy
from paddleocr import PaddleOCR
import re
from rdkit import DataStructs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.spatial import cKDTree, KDTree
from rdkit.Geometry import Point3D
import multiprocessing



def select_longest_smiles(smiles):
    # 将 SMILES 以 '.' 分割为多个部分
    components = smiles.split('.')
    # 选择字符数最多的部分作为主结构
    longest_component = max(components, key=len)
    return longest_component

def MCS_mol(mcs):
    #mcs_smart = mcs.smartsString
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    AllChem.Compute2DCoords(mcs_mol)
    return mcs_mol

def g_atompair_matches(pair,mcs):
    mcs_mol = MCS_mol(mcs)
    matches0 = pair[0].GetSubstructMatches(mcs_mol, useQueryQueryMatches=True,uniquify=False, maxMatches=1000, useChirality=False)
    matches1 = pair[1].GetSubstructMatches(mcs_mol, useQueryQueryMatches=True,uniquify=False, maxMatches=1000, useChirality=False)
    if len(matches0) != len(matches1):
        matches0=list(matches0)
        matches1=list(matches1)
        print( " g_atompair_matches noted: matcher not equal !!")
        if len(matches0)>len(matches1) and len(matches1) !=0:
            for i in range(0,len(matches0)):
                if i < len(matches1):
                    pass
                else:
                    ii=i % len(matches1)
                    matches1.append(matches1[ii])
        else:
            for i in range(0,len(matches1)):
                if i < len(matches0) and len(matches0):
                    pass
                else:
                    ii=i % len(matches0)
                    matches0.append(matches0[ii])
    # assert len(matches0) == len(matches1), "matcher not equal break!!"
    if len(matches0) != len(matches1):
        atommaping_pairs=[[]]
    else:atommaping_pairs=[list(zip(matches0[i],matches1[i])) for i in range(0,len(matches0))]
    return atommaping_pairs


class CustomError(Exception):
    """A custom exception for specific errors."""
    pass

bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
                'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
                'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
                'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,}

BONDTYPE = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
                'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
                'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
                'AROMATIC': Chem.rdchem.BondType.AROMATIC}
BOND_DIRS = {'NONE':    Chem.rdchem.BondDir.NONE,
        'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
        'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
        'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
        'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,}
BONDDIRECT=['ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']


BONDTYPE2ORD={ 
                    'wdge':1,
                    'dash':1,
                    Chem.rdchem.BondType.SINGLE: 1,
                 Chem.rdchem.BondType.DOUBLE: 2,
                 Chem.rdchem.BondType.TRIPLE: 3,
                 Chem.rdchem.BondType.AROMATIC: 1.5,
                 }

BONDTYPE={'SINGLE': Chem.BondType.SINGLE,
 'DOUBLE': Chem.BondType.DOUBLE,
 'TRIPLE': Chem.BondType.TRIPLE,
 'AROMATIC': Chem.BondType.AROMATIC}

VALENCES = {
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1], "*":[3,4,5,6], 
}   

ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "Ru", "Rh","Rn","Rf", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",  "Sr", "Zr",
    "Nb", "Mo", "Tc", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At",  "Fr",  "Ac", "Th",
    "Pa", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",  "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Cn", "Nh", "Fl", "Mc", "Lv", "Og"
]
    # "Rg", "Rb", "Re", "Ra"as RGROUP in the Molscribe data
    #"V",  "Y","U",   # be viewed as C for paddleOCR smt  ONELEMENTS ['A','J]
    #"Ts" #as a chemical group [S](C1=CC=C(C=C1)C)(=O)=O
RGROUP_SYMBOLS = ['R',"R'" 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd','Re','Rg', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar',
                  "V",  "Y","U",'M', 'G','L',
                  'Nr','Tt','Uu','Vv','Ww',#CLEF Nr is not in periodic table
                  'D',#CLEF as [2H] but not recongited by rdkit chemdraw
                  ]

COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}

class Substitution(object):
    '''Define common substitutions for chemical shorthand'''
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability

SUBSTITUTIONS: List[Substitution] = [
    #abbrvs, smarts, smiles
    #patch4 USPTO,try put the longer one first, as re use match by order
    Substitution(['CH2CH2NSO2CH3'], '[CH2][CH]',  '[CH2]CNS(=O)(C)=O', 0.5),
    Substitution(['NHNHCOCF3'], 'NHNHCOCF3',  '[NH]NC(=O)C(F)(F)(F)', 0.5),
    Substitution(['CO2CysPr'], 'CO2CysPr',  '[C](=O)ON[C@H](C(CCC)=O)CS', 0.5),
    Substitution(['OCH2CHOHCH2'], 'OCH2CHOHCH2',  '[O]CC(O)C', 0.5),
    Substitution(['OCH2CHOHCH2OH'], 'OCH2CHOHCH2',  '[O]CC(O)CO', 0.5),
        # elif symbol in ['SO2(CH2)3SO2NHCH2CHCH2OH']:smiles='[S](=O)(=O)CCCS(=O)(=O)NC[C]CO'
    Substitution(['SO2(CH2)3SO2NHCH2CHCH2OH'], 'OCH2CHOHCH2',  '[S](=O)(=O)CCCS(=O)(=O)NC[C]CO', 0.5),




    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    # Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),

    Substitution(['OAc','AcO'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz','BzO'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['COOBn','BnO2C'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)OCc1ccccc1", 0.7),  # Benzyl
    Substitution(['OBn','BnO'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl
    Substitution(['NHBn'], '[NH]Cc1ccccc1', "[NH]Cc1ccccc1", 0.2),  # Benzyl
    Substitution(['NBn2'], '[NH]Cc1ccccc1', "[N](Cc1ccccc1)Cc1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc','BocHN',"BOCHN"], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc','BOc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['NHCbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[NH]C(=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['FmocHN','FmOcHN', 'NHFmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[NH]C(=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs','MsO'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),

    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['PMBN'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[N]Cc1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    # Substitution(['SEM','MES'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['SEM','MES'], '[CH2;D2][O][CH2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]OCC[Si](C)(C)C", 0.2),#fix above 

    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TFAH2N'], 'C(=O)C(F)(F)F', "[NH]C(=O)C(F)(F)F", 0.3),
    Substitution(['TMS'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Ts
    Substitution(['TsO','OTs'], '[O]S(C1=CC=C(C=C1)C)(=O)=O', "[O]S(C1=CC=C(C=C1)C)(=O)=O", 0.6),  # Ts

    Substitution(['COCH3'], '[OH0;D2][CH3;D1]', "[C](=O)C", 0.3),
    # Alkyl chains
    Substitution(['OMe', 'MeO','H;CO', 'CH3O','OCH3', 'H3CO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[N]C", 0.3),#modified as [NH]not wanted
    Substitution(['NMe2', 'Me2N'], '[N;X3](C)[CH3;D1]', "[N](C)C", 0.3),#modified as [NH]not wanted

    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO','C2H5O','OC2H5'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['MeOH2C','CH2OMe'], '[CH2;D2]O[CH3]', "[CH2]OC", 0.5),
    Substitution(['Et', 'CH2CH3','CH3CH2'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    

    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),
    # Substitution(['nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu','tBuO'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['OCCl3', 'Cl3CO'], '[OH0;X2][CH0;D4](Cl)(Cl)Cl', "[O]C(Cl)(Cl)Cl", 0.5),
    Substitution(['SCF3', 'F3CS'], '[SH0;X2][CH0;D4](F)(F)F', "[S]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CO2NH4','COONH4','H4NOOC','H4NO2C'], 'C(=O)[OH]', "[C](=O)ON", 0.5),  # COOH
    Substitution([ 'COO-','CO2-'], 'C(=O)[OH]', "[C](=O)[O-]", 0.5),  # COOH
    # Substitution([ 'COO'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    # Substitution(['OCH3', 'H3CO'], '[OH0;D2][CH3]', "[O]C", 0.4),
    #TODO if need just addit here
    Substitution(['N3'], '[N]=[N+]=[N-]', "[N]=[N+]=[N-]", 0.4),#ACS image dataset has
    # [N-]=[N+]
    Substitution(['N2+Cl-','Cl-N2+'], '[N+]#[N].[Cl-]', "[N+]#[N].[Cl-]", 0.4),#ACS image dataset has
    Substitution(['N2'], '[N]=[N-]', "[N]=[N-]", 0.4),#ACS image dataset has
    Substitution(['N2H'], '[N]=[N-]', "[N]=[NH]", 0.4),#ACS image dataset has
    Substitution(['NO','N=O','O=N','ON'], '[N]=[O]', "[N]=O", 0.4),#ACS image dataset has
    Substitution(['NCH3'], '[N]C', "[NH]C", 0.4),#ACS image dataset has
    Substitution(['NOMe'], '[N]OC', "[N]OC", 0.4),#ACS image dataset has
    Substitution(['OCH2'], '[O]C', "[O]C", 0.4),#FORMULA_REGEX
    Substitution(['C=O','O=C'], '[C]=[O]', "[C]=O", 0.4),#ACS image dataset has
    Substitution(['NPh','PhN'], 'NC1=CC=CC=C1', "[N]C1=CC=CC=C1", 0.4),#ACS image dataset has
    Substitution(['NHPh','PhNH','PhHN'], 'NC1=CC=CC=C1', "[NH]C1=CC=CC=C1", 0.4),#ACS image dataset has
    Substitution(['TMSO','OSMT'], 'O[Si](C)(C)C', "[O][Si](C)(C)C", 0.5),
    Substitution(['SPh','PhS'], 'SC1=CC=CC=C1', "[S]C1=CC=CC=C1", 0.4),#ACS image dataset has
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),
    Substitution(['SO3NH2','SO3NH4','H4NO3S'], 'S(=O)(=O)[OH]', "[S](=O)(=O)ON", 0.4),
    Substitution(['SO3'], 'S(=O)(=O)[OH]', "[S](=O)(=O)[O-]", 0.4),
    Substitution(['SO2CF3'], '[S](=O)(=O)C(F)(F)F',  "[S](=O)(=O)C(F)(F)F", 0.5),
    Substitution(['SO2Cl'], '[S](=O)(=O)Cl',  "[S](=O)(=O)Cl", 0.5),
    Substitution(['SO2F'], '[S](=O)(=O)F',  "[S](=O)(=O)F", 0.5),
    Substitution(['SO2'], '[S](=O)(=O)',  "[S](=O)(=O)", 0.5),
    Substitution(['SO2NH'], '[S](=O)(=O)[N]',  "[S](=O)(=O)[N]", 0.5),#US07323045-20080129-C00062 may lead wrong connext
    Substitution(['SO2NH2'], '[S](=O)(=O)[NH2]',  "[S](=O)(=O)[NH2]", 0.5),
    Substitution(['SO2Me','SO2CH3'], '[S](=O)(=O)C',  "[S](=O)(=O)C", 0.5),
    Substitution(['NHO2S'], '[S](=O)(=O)[N]',  "[N][S](=O)(=O)", 0.5),#US07323045-20080129-C00062 may lead wrong connext
    Substitution(['OSO2Me'], '[O]S(=O)(=O)C',  "[O]S(=O)(=O)C", 0.5),
    Substitution(['NHSO2Me'], '[NH]S(=O)(=O)C',  "[NH]S(=O)(=O)C", 0.5),
    Substitution(['SOCH3','SOMe'], '[S](=O)(=O)',  "[S](=O)C", 0.5),

    Substitution(['P+Ph3Br-'], '[P+](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3',  "[P+](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3", 0.5),
    Substitution(['N+Ph3Br-'], '[N+](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3',  "[N+](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3", 0.5),
    Substitution(['PPh2'], "[P](C1=CC=CC=C1)C2=CC=CC=C2",  "[P](C1=CC=CC=C1)C2=CC=CC=C2", 0.5),
    # Substitution(['BOcHN',"BOCHN"], "[NH]C(OC(C)(C)C)=O",  "[NH]C(OC(C)(C)C)=O", 0.5),
    Substitution(['CO2Me', 'COOMe'], 'C(=O)[OH0;D2][CH3]', "[C](=O)OC", 0.5),
    Substitution(['ONa', 'NaO'], '[O][Na]', "[O][Na]", 0.5),
    Substitution(['OTBDMS', 'TBDMSO'], "[O][Si](C)(C)C(C)(C)C", "[O][Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['CONH2'], '[C](O)(N)', "[C](=O)[NH2]", 0.5),
    Substitution(['NHNH2'], '[NH2;D1]', "[NH]N", 0.1),
    Substitution(['CONH'], 'CONH',  '[C](=O)N', 0.5),
    Substitution(['CH3CONH'], '[NH]C(=O)C',  '[NH]C(=O)C', 0.5),
    Substitution(['NH3Cl'], '[NH]Cl',  '[NH]Cl', 0.5),

    Substitution(['SAc','AcS'], '[S]C(C)=O', "[S]C(C)=O", 0.5),
    Substitution(['OAll'], '[O]CC=C', '[O]CC=C', 0.5),
    # Substitution(['Tos'], '[Si](C)(C)C', '[Si](C)(C)C', 0.5),#NOTE different case ?? @@acs dataset ,we use the SO2here
    Substitution(['Tos','TOs'], '[Si](C)(C)C', '[S](=O)(=O)C(C=C1)=CC=C1C', 0.5),#NOTE different case ??
    Substitution(['OTos','OTOs','soTO'], '[Si](C)(C)C', '[O]S(=O)(=O)C(C=C1)=CC=C1C', 0.5),#NOTE different case ??
    Substitution(['TsN'], '[N]S(C1=CC=C(C=C1)C)(=O)=O', '[N]S(C1=CC=C(C=C1)C)(=O)=O', 0.5),
    Substitution(['Ts'], '[S](C1=CC=C(C=C1)C)(=O)=O', '[S](C1=CC=C(C=C1)C)(=O)=O', 0.5),
    Substitution(['COCF3'], '[C](=O)C(F)(F)(F)', '[C](=O)C(F)(F)(F)', 0.5),
    Substitution(['CF2', 'F2C'], '[C;D4](F)(F)', "[C](F)(F)", 0.5),
    Substitution(['PMB'], '[CH2]C1=CC=C(C=C1)OC', '[CH2]C1=CC=C(C=C1)OC', 0.5),
    Substitution(['NHCOtBu'], '[NH]C(C(C)(C)C)=O','[NH]C(C(C)(C)C)=O', 0.5),
    Substitution(['OCN'], '[N]=C=O', "[N]=C=O", 0.5),
    Substitution(['Me3Si'], '[Si](C)(C)(C)', "[Si](C)(C)(C)", 0.5),
    Substitution(['PhO','OPh'], '[O]C1=CC=CC=C1', "[O]C1=CC=CC=C1", 0.5),
    Substitution(['Allyl'], '[CH2]C=C', '[CH2]C=C', 0.5),
    Substitution(['C7H3'], '[C]#CC#CC#CC', '[C]#CC#CC#CC', 0.5), 
    Substitution(['C5H11'], '[CH2]CCCC', '[CH2]CCCC',  0.5), 
    Substitution(['R1R2N'], "[N]([*])[*]",  "[N]([*])[*]", 0.5),
    Substitution(['CO2R'], '[C](=O)O*', '[C](=O)O*',  0.5), 
    Substitution(['CCl3CH2O2C'], '[C](=O)OCC(Cl)(Cl)Cl', '[C](=O)OCC(Cl)(Cl)Cl',  0.5), 
    Substitution(['NHOH'], '[NH]O', '[NH]O',  0.5),
    Substitution(['CO2'], '[C](=O)[O]', '[C](=O)[O]',  0.5),
    Substitution(['O2C'], '[C](=O)[O]', '[O][C](=O)',  0.5),#NOTE the direction matters

    Substitution(['PPh3'], '[P](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3', '[P](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3', 0.5),
    Substitution(['TfO'], '[C](=O)[O]', '[O]S(=O)(C(F)(F)F)=O',  0.5),
    Substitution(['OCH2Ph'], '[O]CC1=CC=CC=C1',  '[O]CC1=CC=CC=C1', 0.5),
    Substitution(['OCH2CF3'], '[O]CC(F)(F)(F)',  '[O]CC(F)(F)(F)', 0.5),
    Substitution(['COOCH2Ph'], '[C](=O)OCC1=CC=CC=C1',  '[C](=O)OCC1=CC=CC=C1', 0.5),
    Substitution(['OCH2OC2H5'], '[C](=O)C(C)(C)C',  '[O]COCC', 0.5),
    
    Substitution(['Trt'], '[C](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3', '[C](C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3',  0.5),
    Substitution(['SF5'], '[S](F)(F)(F)(F)F',  '[S](F)(F)(F)(F)F', 0.5),

    # Substitution(['CH2CH'], '[CH2][CH]',  '[CH2][CH]', 0.5),
    # Substitution(['CH2CH2'], '[CH2][CH2]',  '[CH2][CH2]', 0.5),

    # #SIMPLE abbv
    Substitution(['S*'], '[S]*',  '[S]*', 0.5),
    Substitution(['N*, NH*'], '[NH]*',  '[NH]*', 0.5),
    Substitution(['C*','CH2*'], '[C]*',  '[CH2]*', 0.5),
    Substitution(['P*',"PH*"], '[P]*',  '[PH]*', 0.5),
    Substitution(['O*'], '[O]*',  '[O]*', 0.5),
    #（） effect
    Substitution(['N(CH3)2'], '[N](C)(C)', "[N](C)(C)", 0.5),
    Substitution(['(C2H5)2N','Et2N'], '[N](C)(C)', "[N](CC)(CC)", 0.5),
    Substitution(['B(OH)2'], '[B](O)O', "[B](O)O", 0.5),
    Substitution(['CO2C(CH3)3'], '[C](=O)C(C)(C)C',  '[C](=O)C(C)(C)C', 0.5),
    Substitution(['P(O)(OEt)2', 'P(OEt)2(O)'], "[P](OCC)(=O)CCO", "[P](OCC)(=O)OCC", 0.5),
    Substitution(['(CH2)16Me'], '[CH2]CCCCCCCCCCCCCCCC', "[CH2]CCCCCCCCCCCCCCCC", 0.3),
    Substitution(['(CH2)11Me'], '[CH2]CCCCCCCCCCC', "[CH2]CCCCCCCCCCC", 0.3),
    Substitution(['N(H)Et','Et(H)N'], '[NH]CC', '[NH]CC',  0.5),
    Substitution(['N(H)Me','Me(H)N'], '[NH]C', '[NH]C',  0.5),



]
ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}


def extract_abbreviation_key(item):
    if isinstance(item, list):
        while isinstance(item, list):
            item = item[0]
        return item
    return item


def clean_unpaired_brackets(text):
    #keep paired, del unpared 
    result = []
    stack = []
    bracket_pairs = {')': '(', ']': '['}
    opening_brackets = {'(', '['}
    
    for char in text:
        if char in opening_brackets:
            stack.append(char)
            result.append(char)
        elif char in bracket_pairs:
            if stack and stack[-1] == bracket_pairs[char]:
                stack.pop()
                result.append(char)
            else:
                # 未配对的闭合括号，跳过
                continue
        else:
            result.append(char)
    return ''.join(result)

# def del_unpairebrackets(opening_brackets):    
#     # 移除未配对的开括号
#     keep paired, del unpared 
#     result = []
#     stack = []
#     bracket_pairs = {')': '(', ']': '['}
#     opening_brackets = {'(', '['}
#     for char in result:
#         if char in opening_brackets:
#             stack.append(char)
#         elif char in bracket_pairs:
#             if stack and stack[-1] == bracket_pairs[char]:
#                 stack.pop()
#                 final_result.append(char)
#             else:
#                 continue
#         else:
#             final_result.append(char)
    
    # # 如果仍有未闭合的开括号，移除它们
    # return ''.join(c for c in final_result if not stack or c not in opening_brackets)

def replace_c1(text):
    # Use negative lookahead to ensure 'C1' isn't followed by another digit
    return re.sub(r'C1(?!\d)', 'Cl', text)
def transform_formula(formula):
    # 匹配 C 后面的数字和 Hg（允许 Hg 后跟其他元素）
    match = re.match(r'C(\d+)(.*?)Hg(.*)', formula)
    if not match:
        return formula
    
    n = int(match.group(1))
    prefix = match.group(2)  # Hg 前的部分（如空字符串或其他元素）
    suffix = match.group(3)  # Hg 后的部分（如 O2）
    g_new = n * 2 + 1
    return f"C{n}{prefix}H{g_new}{suffix}"
def Cg_transform_formula(formula):
    # 匹配 C 后面的数字和 Hg（允许 Hg 后跟其他元素）
    match = re.match(r'CgH(\d+)(.*?)', formula)
    if not match:
        return formula
    
    n = int(match.group(1))
    suffix = match.group(2)  # Hg 后的部分（如 O2）
    g_new = (n-1)// 2
    return f"C{g_new}H{n}{suffix}"

def normalize_ocr_text(text, replacement_map):
    """Normalize OCR text using the predefined mapping rules"""
    if 'C1'in text:
        text=replace_c1(text)
    if 'Hg' in text:
        text= transform_formula(text)
    if 'Cg' in text:
        text= Cg_transform_formula(text)
    if 'Q' in text:
        pattern = r'Q([A-Z])(\w+)'
        replacement = r'O\1\2'
        text = re.sub(pattern, replacement, text)
    if text in ELEMENTS:
        return text  
    #remove space
    if ' ' in text:
        text = text.replace(" ", "")
    if any(c in text for c in '0oO'):
        # Step 1: Replace 'o' or 'O' with '0' when after a digit and before a letter or end of string
        # text = re.sub(r'(?<=[1-9])[oO](?=[a-zA-GI-Z]|$)', '0', text)
        text = re.sub(r'(?<![CF,CH]\d)[oO](?=[a-zA-GI-Z]|$)', '0', text)
        if '00' in text:    text = re.sub(r'00', 'OO', text)#CH0 to CHO
        # text= re.sub(r'(?<=\d)[oO](?=[a-zA-GI-Z]|$)', '0', text)
        # Step 2: Replace '0' with 'O' when preceded by a letter or followed by optional digits/commas and a letter
        # pattern = r'(?<=[a-zA-Z])0(?=([a-zA-Z]|$))'
        if text in ['R20']: return text

        text = re.sub(r'(?<=[a-zA-Z])0(?=([a-zA-Z]|$))', 'O', text)#CH0 to CHO
        text = re.sub( r'^(0)|(?<=[a-zA-Z][?\d])0(?=[a-zA-Z0-9]*$|[a-zA-Z])', 'O', text)#CF20 to CF2O
        # result = re.sub(r'(?<=[a-zA-Z])0|0(?=[,\d]*[a-zA-Z])', 'O', text)
        # Step 3: Only apply '0' to 'O' replacement if '0' doesn't follow digits 1-9
        # if not re.search(r'[1-9]0', text):
        #     text = result
    text=clean_unpaired_brackets(text)
    pattern_n1 = r'^NHR[0-9a-z]$'

    # Your existing text normalization rules
    if text in ['OzN','O2N', 'O,N', 'NOz','NO2', 'NO,', '0;N','02N','N20']: text = 'NO2'
    #jpo
    elif text in ['CHzCH','CH,CH',]:text='CH3CH'
    elif text in ["NHCHzCOOH","NHCH2COOH",]:text='NHCH2COOH'
    elif text in ['CIOC','COCE','ClOC','COCI']:text='COCl'
    
    elif text in ['CHCOOCHs','CH2COOCH5']:text='CH2COOC2H5'
    #staker
    elif text in ['(t-Bu)','t-Bu']:text='t-Bu'

    #ACS
    elif text in ['SiMe2','Me2Si']:text='SiMe2'
    elif text in ['ArzP(O)','Ar2P(O)']:text='Ar2P(O)'
    elif text in ['P(O)(0Et)2','P(O)(OEt)2']:text='P(O)(OEt)2'
    elif text in ['PhOzS','PhO2S']:text='PhO2S'
    elif text in ['CH3O','CHzO']:text='CH3O'
    elif text in ['NH.HCI','NH,.Hcl']:text='NH2.HCl'
    
    #CLEF
    elif text in ['2','Z']:text='Z'
    elif text in ['(CH2)m','(CH2)q','(CH2)s']:text='CH2'
    elif text in ['Arl','Ari','Ar2','Ar1',]:text='Ar'
    elif text in [ '"0ls','"ols','S[0]a']:text='S[O]a'
    elif text in ['NHR%','NHR*']:text='NHR8'
    elif text in ['Vv','Vy']:text='Vv'


    elif text in ['N3','NY','Ny']:text='N3'
    elif text in ['C2H52N','N(CH,CH3)2','C;H52N','(C;H5)2N','N(C;Hs)2','N(C;H5)2','(CHzCH2)2N','N(CHCH3)2','(CH3CH2)2N','(C2H52N', '(CHzCH)2N','(C2H5)2N','Et2N']:text='(C2H5)2N'
    elif text in ['(CH3)2N','Me2NH','Me,N','Me2N']:text='Me2N'
    elif text in ['(C;H4O)H','(C2H4O)H']:text='(C2H4O)H'
    elif text in ['(C;H4O)4CH3','(C2H4O)4CH3' ]:text='(C2H4O)4CH3'
    elif text in ['(CH2)16Me' ]:text='(CH2)16Me'
    elif text in ['(CH2)11Me']:text='(CH2)11Me'
    elif text in ['CO2CH2Ph','COOCH2Ph','COOCH,Ph']:text='COOCH2Ph'
    elif text in ['CO2C(CH3)3','(CH3)3CO2C',]:text='CO2C(CH3)3'
    elif text in ['OCH2Ph','OCH,Ph','OCHAPH','OCH;Ph']:text='OCH2Ph'
    elif text in ['(CF2)8H','(CF2)gH','(CF2)sH','CF2sH', 'CF:H','CF2)sH','CF):H' ]: text = '(CF2)8H'
    elif text in ['NHSO,Bu','NHSO2Bu',]: text = 'NHSO2Bu'
    elif text in ['NHSO,CH3','NHSO2CH3','NHSO2Me']: text = 'NHSO2CH3'
    elif text in ['1231','1231','23T', 'l23I']: text = 'l23I'



    elif text in ['CF3','CFs', 'CF,', '13','CF 3','F;C', 'F:C', 'F sC', 'CF', 'CF;', 'CFa', 'FzC', 'CFz']: text = 'F3C'
    elif text in ['OCCl3','Cl3CO',]: text = 'OCCl3'
    elif text in ['CCl3','Cl3C',]: text = 'CCl3'
    elif text in ['F;CN', 'NCF;']: text = 'F3CN'
    elif text in ['NCH3','NHCH3',  'NCH;','CH3N','MeN','MeNH']: text = 'NCH3'
    elif text in ['NOMe']:text='NOMe'
    elif text in ['R,R,N']: text = 'R1R2N'
    elif text in ['HzC','HyC','CHy','CHE','H3C.','1;.C', '1;C', 'M e','Mé', 'CH 3', 'CH:', 'HsC', 'HaC', 'H3C', 'CH3', 'CHa', 'H;C', 'CH,', 'CHs', 'CH;']: text = 'Me'
    # elif text in ['CH2']: text = 'C'
    elif text in ['PhzBr']: text = 'Ph3Br'
    elif text in ['PPh3', 'PPha']: text = 'PPh3'
    elif text in ['Et', 'CH,CH3','Catls','Cafls','CH2CH3','H3CH2C','C:H5','HzCH2C','H3CH2C', 'C,H5', 'CzH5','C2H5','C2Hs']: text = 'CH2CH3'
    elif text in ['Ovle', 'HzCO','OCH', 'OCH:','H2CO', 'CH3O', 'CH,O', 'HsCO','OMe','AME', 'AMe','H3CO', 'MeO']: text = 'OMe'
    elif text in ['OCHa','HgCO', 'OCH','HaCO', 'OCH:','H2CO', 'CH3O', 'CH,O', 'OMe','AME', 'AMe', 'MeO']: text = 'OMe'
    elif text in ['SO2Cl', 'SOzCl']: text = 'SO2Cl'
    elif text in ['SO2F', 'SOzF']: text = 'SO2F'
    elif text in ['SONH', 'HNOS','SON', 'SO2NH']: text = 'SO2NH'
    elif text in ['HNO2S','NHO2S']: text = 'NHO2S'
    elif text in ['SO2Cl', 'SOzCl']: text = 'SO2Cl'
    elif text in ['SO2F', 'SOzF']: text = 'SO2F'
    elif text in ['SONH', 'HNOS','SON', 'SO2NH']: text = 'SO2NH'
    elif text in ['SO2NH2', 'SO,NH', 'SO:NH2', 'SONH2']: text = 'SO2NH2'
    elif text in ['SOzCF3', 'SO2CF3', 'CF3SO2']: text = 'SO2CF3'
    elif text in ['SOz','O2S', '$02', 'S02','SO,', '62','O:S','SO2']: text = 'SO2'
    elif text in ['H3CO2S','SO2CH3']: text='SO2CH3'
    elif text in ['SO3H','SOsH','SOaH', 'HO3S','SOzH','HOzS']: text = 'SO3H'
    elif text in ['MeO2SO','OSO2CH3','OSO2Me']:text='OSO2Me'
    elif text in ['MeO2SHN','NHSO2Me']:text='NHSO2Me'
    

    elif text in ['PIME', 'PMB']: text = 'PMB'
    elif text in ['1-BU', '-BU', '-Bu', 't-BU','t-Bu']: text = 't-Bu'
    elif text in ['NTS', 'NTs', 'TsN']: text = 'TsN'
    elif text in ['TsO', 'OTs']: text = 'OTs'
    elif text in ['Nz* Cl', "N2+Cl-"]: text = 'N2+Cl-'
    elif text in ['NH3Cl', 'NHzCl','NH;Cl']: text = 'NH3Cl'
    elif text in ['B(OH)2']: text = 'B(OH)2'
    elif text in ['NHAC', 'NHAc']: text = 'NHAc'
    elif text in ['1CO', 'NCO', 'OCN', 'OON']: text = 'OCN'
    elif text in ['COCFs','COCF3', 'COCF s']: text = 'COCF3'
    elif text in ['OCF3', 'OCF 3','OCE', 'OCE:','OCEE', 'F3CO', 'OCF', 'OCF:']: text = 'OCF3'
    elif text in ['SCF3', 'SCE', 'SCEE', 'F3CS', 'SCF', 'SCF:']: text = 'SCF3'
    elif text in ['HzCS', 'SCH3', 'SMe','MeS','H3SC' ]: text = 'SMe'
    elif text in ['CHzCHzO', 'CH3CH2O','H5C2O','OC2H5']: text = 'OEt'
    elif text in ['CO,Et','COzEt', 'CO2Et','H3CH2COOC','CO2C2H5']:text = 'CO2Et'
    elif text in ['OTBS', 'TBSO', 'OTBDMS']: text = 'OTBDMS'
    elif text in ['PhO', 'Pho']: text = 'PhO'
    elif text in ['CI', 'C1']: text = 'Cl'
    elif text in ['P h', 'Ph']: text = 'Ph'
    elif text in ['FAHN', 'TFAH,N','TFAH2N',]: text = 'TFAH2N'
    elif text in ['MeaSi', 'Me3Si']: text = 'Me3Si'

    elif text in ['PHzC','PH;C', 'PH3C']: text = 'PH3C'
    elif text in ['COOH','OOOH','1OOC', 'HOOO','HOOC', 'DOOH', 'CO:H','HO,C','CO,H','CO2H']: text = 'CO2H'
    # elif text in ['COO','COO-']: text = 'COO-'#coo-bond
    elif text in ['CO2R','RO2C', 'RO,C','CO2*', "COzR'"]: text = 'CO2R'
    elif text in ['CO2', 'COO','OOC', "COz"]: text = 'CO2'
    #direction matter
    elif text in ['O2C', '02C']: text = 'O2C'
    elif text in ['CaH;', 'CHS', 'C2H5']: text = 'C2H5'
    elif text in ['NHBoc','NHBOc', 'BocHN','BOcHN', "BOCHN"]: text = "NHBoc"
    elif text in ['C7H', 'C7H3']: text = 'C7H3'
    elif text in ['CsH11', 'C5H11']: text = 'C5H11'
    elif text in ['CC3CH2O2C', 'CCl3CH2O2C']: text = 'CCl3CH2O2C'
    elif text in ['CH2OMe','MeOH,C','CH,0Me', 'CH,OMe','MeOH2C']: text = 'CH2OMe'
    elif text in ['R', "R'"]: text = '*'
    elif text in ['U', 'U.']: text = 'U'
    elif text in ['RO']: text = 'O*'
    elif text in ['OAc', 'OAC']: text = 'OAc'
    elif text in ['Rg', 'R9']: text = 'R9'
    elif text in ['OQ', '00', '0Q','OCH3']: text = 'OMe'
    # elif text in ['NH', 'HN']: text = '[NH]'
    elif text in ['NH', 'HN', "NH2", 'H2N', 'H,N']: text = 'N'
    elif text in ['OH', 'HO', 'OH2', '0']: text = 'O'
    elif text in  ['N(H)Et','Et(H)N']: text = 'N(H)Et'
    elif text in  ['N(H)Me','Me(H)N']: text = 'N(H)Me'
    elif text in ['HNOC','CONH']: text='CONH'
    elif text in ['HNOCCH3','CH,CONH','CH3CONH']: text='CH3CONH'
    elif text in ['PPh2','Ph,P','Ph2P']: text='PPh2'
    elif text in ['SF5','F5S']: text = 'SF5'
    elif text in ['OCH2CF3','F3CH2CO']: text = 'OCH2CF3'
    elif text in ['NHCbz','CbzHN']: text = 'NHCbz'
    elif text in ['NHNH2','H2NHN']: text = 'NHNH2'
    elif text in ['CHzCH22N','N2(CH2CH3)','(CH3CH2)2N']: text = '(CH3CH2)2N'
    #NOTE this with 3 neibor bonds, whic in x order, direction matters
    elif text in ['CHCHCH2CH-3','CH2CH2CH2CH']: text = 'CH2CH2CH2CH'
    elif text in ['HCH2CH2CH2C','HCH2CH2CH2C' ]: text = 'HCH2CH2CH2C'
    
    elif text in ['(HzC)2HC','(H3C)2HC']: text = '(H3C)2HC'
    elif text in ['13CO2SHNH2CH2C','H3CO2SHNH2CH2C','CH2CH2NSO2CH3']: text = 'CH2CH2NSO2CH3'#USPTO
    elif text in ['CgH19','C9H19']: text = 'C9H19'
    elif text in ['(CF2):H','(CF2)8H']: text = '(CF2)8H'

    elif text in ['COOCH3','HzCO2C', 'CO,Me','H3CO2C','CO2CH3','MeOOC','CO2Me','COzMe','MeO2C','MeO,C']: text = 'CO2Me'
    elif text in ['(CHCHO)','CH2CH2O']: text = 'CH2CH2O'
    elif text in ['CO,CysPr','CO2CysPr']: text = 'CO2CysPr'
    elif text in ['CH2CH2C(O)OCHCH3','CH;CH2C(O)OCHCH3']:text='CH2CH2C(O)OCH2CH3'
    elif text in ['H4NOzS','H4NO3S']: text = 'H4NO3S'
    elif text in ['C1OH21','C1oH21','CloH21', 'C10H21']: text = 'C10H21'

    elif text in ['']: text = 'CF2'

    elif text in replacement_map:
        text = replacement_map[text]
    # elif 'NHR' in text or 'RHN' in text:
    #     text = NHR_string(text)
    # elif text in ['RHN']: text = 'N*'
    
    return text





def C_H_affixExpand(group):
    """
    Expands CnHm or HmCn chemical group notation into SMILES format.
    Supports formats like C6H11, NHC6H11, H11C6, H11C6HN where H = 2C - 1.
    Returns SMILES string or False if invalid.
    """
    # Regex patterns
    p_cn_hm = r'^C(\d+)H(\d+)$'  # Standalone CnHm (e.g., C6H11)
    p_hm_cn = r'^H(\d+)C(\d+)$'  # Standalone HmCn (e.g., H11C6)
    p_prefix = r'^([A-Za-z]+)(C(\d+)H(\d+))$'  # Prefix + CnHm (e.g., NHC6H11)
    p_suffix = r'^(C(\d+)H(\d+))([A-Za-z]+)$'  # CnHm + Suffix (e.g., C6H11NH)
    p_hm_cn_prefix = r'^([A-Za-z]+)(H(\d+)C(\d+))$'  # Prefix + HmCn (e.g., H11C6HN)
    p_hm_cn_suffix = r'^(H(\d+)C(\d+))([A-Za-z]+)$'  # HmCn + Suffix (e.g., H11C6NH)

    # 2. Handle CnHm or HmCn with prefix/suffix
    patterns = [
    #pattern, sub_pattern,aff_idx, group_idx, c_idx, h_idx, aff_type
    (p_prefix, p_cn_hm, 1, 2, 3, 4, 'prefix'),
    (p_suffix, p_cn_hm, 4, 1, 2, 3, 'suffix'),
    (p_hm_cn_prefix, p_hm_cn,1, 2, 4, 3, 'prefix'),
    (p_hm_cn_suffix, p_hm_cn, 4, 1, 3, 2, 'suffix')
    ]

    # Abbreviation map for common groups
    ABBREVIATIONS2 = {
        'NH': '[NH]', 'HNOC': '[C](=O)[NH]',
        'CONH': '[C](=O)[NH]', 'HN': '[NH]', 'HNO': '[NH]O', 'NO': '[N]=O',
        'COO':'[C](=O)O',
        'CO2':'[C](=O)O',

    }#TODO may need more 

    def validate_and_expand(c_count, h_count, prefix=None, suffix=None):
        """Helper to validate CnHm/HmCn and generate SMILES."""
        if h_count != 2 * c_count + 1:  # Check if H = 2C + 1 CmHn
            return False
        # Base SMILES: [CH] for single carbon, or [CH]C...C for multiple
        smiles = '[CH2]C' if c_count == 2 else '[CH2]'+'C' * int(c_count - 1)#NOTE C have to 2n
        print([c_count, h_count, prefix, suffix],'[c_count, h_count, prefix, suffix]')
        if prefix:
            prefix = ABBREVIATIONS2.get(prefix, prefix)
            smiles = prefix + smiles
        if suffix:  # Changed from elif to if to handle both prefix and suffix
            suffix = ABBREVIATIONS2.get(suffix, suffix)
            smiles = suffix + smiles #as CmHn are always n=2m+1
        return smiles

    # 1. Handle standalone CnHm or HmCn
    match_cn_hm = re.match(p_cn_hm, group)
    if match_cn_hm:
        c_count, h_count = int(match_cn_hm.group(1)), int(match_cn_hm.group(2))
        return validate_and_expand(c_count, h_count)

    match_hm_cn = re.match(p_hm_cn, group)
    if match_hm_cn:
        h_count, c_count = int(match_hm_cn.group(1)), int(match_hm_cn.group(2))
        return validate_and_expand(c_count, h_count)

    for pattern, sub_pattern,aff_idx, group_idx, c_idx, h_idx, aff_type in patterns:
        match = re.match(pattern, group)
        if match:
            cn_hm = match.group(group_idx)
            affix = match.group(aff_idx)  # Other group is prefix/suffix
            c_count = int(match.group(c_idx))
            h_count = int(match.group(h_idx))
            print(cn_hm,affix,c_count,h_count,'cn_hm,affix,c_count,h_count')
            return validate_and_expand(
                    c_count, h_count,
                    prefix=affix if aff_type == 'prefix' else None,
                    suffix=affix if aff_type == 'suffix' else None
                )

    return False

def N_C_H_expand(group):
    # 使用正则表达式匹配 NHCnHm 中的 n
    match = re.match(r'NHC(\d+)H(\d+)', group)
    match1 = re.match(r'NC(\d+)H(\d+)', group)
    if not match and not match1:
        return False
    # 获取碳原子数
    if match:
        C_count = int(match.group(1))
        H_count = int(match.group(2))
    if match1:
        C_count = int(match1.group(1))
        H_count = int(match1.group(2))
    if H_count== C_count*2 +1 :
        # 构建 SMILES：'[N]' + 'C' * 碳原子数
        smiles = '[N]' + 'C' * C_count
    return smiles

def C_F_expand(group):
    # 尝试匹配 CnFm 格式 (e.g., C2F5)
    match_cnfm = re.match(r'C(\d+)F(\d+)', group)
    match_cnfm_2 = re.match(r'F(\d+)C(\d+)', group)
    if match_cnfm:
        C_count = int(match_cnfm.group(1))
        F_count = int(match_cnfm.group(2))
        # 验证氟原子数是否符合全氟烷基的规则：F_count = 2 * C_count + 1
        if F_count != 2 * C_count + 1:
            return False
    else:
        # 尝试匹配 CF2CF3 格式 (e.g., CF2CF3, CF2CF2CF3)
        # 匹配一个或多个 CF2 后跟一个 CF3
        match_cfx = re.match(r'(CF2)*CF3$', group)
        if not match_cfx:
            return False
        # 计算碳原子和氟原子数
        cf2_count = group.count('CF2')  # 每个 CF2 贡献 1 碳和 2 氟
        C_count = cf2_count + 1  # +1 for the terminal CF3
        F_count = cf2_count * 2 + 3  # Each CF2 has 2F, CF3 has 3F
        # 验证氟原子数是否符合全氟烷基的规则
        if F_count != 2 * C_count + 1:
            return False
    # 构建 SMILES 字符串
    smiles = []
    for i in range(C_count):
        if i < C_count - 1:
            # 前面的碳原子：2个氟原子，形式为 C(F)(F)
            if len(smiles)==0:
                smiles.append('[C](F)(F)')
            else:
                smiles.append('C(F)(F)')
        else:
            # 最后一个碳原子：3个氟原子，形式为 [C](F)(F)(F)
            smiles.append('C(F)(F)(F)')
    
    # 连接所有部分
    return ''.join(smiles)
    
# def C_H_expand(group):
#     """
#     Expands CnHm or HmCn chemical group notation into SMILES format.
#     Supports formats like C18H37HNOC, CONHC3H7, C3H7, H23C11.
#     Returns SMILES string or False if invalid.
#     """
#     # Regex patterns
#     # Regex patterns
#     p_cn_hm = r'^C(\d+)H(\d+)$'  # Standalone CnHm (e.g., C6H11)
#     p_hm_cn = r'^H(\d+)C(\d+)$'  # Standalone HmCn (e.g., H11C6)
#     p_prefix = r'^([A-Za-z]+)(C\d+H\d+)$'  # Prefix + CnHm (e.g., NHC6H11)
#     p_suffix = r'^(C\d+H\d+)([A-Za-z]+)$'  # CnHm + Suffix (e.g., C6H11NH)
#     p_hm_cn_prefix = r'^([A-Za-z]+)(H\d+C\d+)$'  # Prefix + HmCn (e.g., H11C6HN)
#     p_hm_cn_suffix = r'^(H\d+C\d+)([A-Za-z]+)$'  # HmCn + Suffix (e.g., H11C6NH)

#     # Element and suffix replacement map
#     elements = ['S', 'N', 'P', 'C', 'O']
#     keys = [f"{e}{suffix}" for e in elements for suffix in ['R"', "R'", "R", "*"]]
#     replacement_map = {key: f'{key[0]}*' for key in keys}
#     def validate_and_expand(c_count, h_count, prefix=None, suffix=None):
#         """Helper to validate CnHm/HmCn and generate SMILES."""
#         if h_count != 2 * c_count + 1:  # Check if valid alkyl group
#             return False
#         smiles = '[CH2]' + 'C' * (c_count - 1)
#         if prefix:
#             prefix = normalize_ocr_text(prefix, replacement_map)
#             smiles = ABBREVIATIONS.get(prefix, prefix) + 'C' * c_count
#         elif suffix:
#             suffix = normalize_ocr_text(suffix, replacement_map)
#             smiles = ABBREVIATIONS.get(suffix, suffix) + 'C' * c_count
#         return smiles

#     # 1. Handle standalone CnHm or HmCn first
#     match_cn_hm = re.match(p_cn_hm, group)
#     if match_cn_hm:
#         c_count, h_count = int(match_cn_hm.group(1)), int(match_cn_hm.group(2))
#         return validate_and_expand(c_count, h_count)

#     match_hm_cn = re.match(p_hm_cn, group)
#     if match_hm_cn:
#         h_count, c_count = int(match_hm_cn.group(1)), int(match_hm_cn.group(2))
#         return validate_and_expand(c_count, h_count)

#     # 2. Handle CnHm or HmCn with prefix/suffix
#     patterns = [
#         (p_prefix, p_cn_hm, 1, 2, 'suffix'),
#         (p_suffix, p_cn_hm, 2, 1, 'prefix'),
#         (p_hm_cn_prefix, p_hm_cn, 1, 2, 'suffix'),
#         (p_hm_cn_suffix, p_hm_cn, 2, 1, 'prefix')
#     ]

#     for pattern, sub_pattern, c_idx, h_idx, aff_type in patterns:
#         match = re.match(pattern, group)
#         if match:
#             cn_hm = match.group(1 if aff_type == 'suffix' else 2)
#             affix = match.group(2 if aff_type == 'suffix' else 1)
#             sub_match = re.match(sub_pattern, cn_hm)
#             if sub_match:
#                 c_count = int(sub_match.group(c_idx))
#                 h_count = int(sub_match.group(h_idx))
#                 return validate_and_expand(
#                     c_count, h_count,
#                     prefix=affix if aff_type == 'prefix' else None,
#                     suffix=affix if aff_type == 'suffix' else None
#                 )

#     return False

import re

def C_H_expand(group):
    """
    Expands CnHm or HmCn chemical group notation into SMILES format.
    Supports formats like C18H37HNOC, CONHC3H7, C3H7, H23C11, and (H7C3)2N.
    Returns SMILES string or False if invalid.
    """
    # Regex patterns
    p_cn_hm = r'^C(\d+)H(\d+)$'  # Standalone CnHm (e.g., C6H11)
    p_hm_cn = r'^H(\d+)C(\d+)$'  # Standalone HmCn (e.g., H11C6)
    p_prefix = r'^([A-Za-z]+)(C\d+H\d+)$'  # Prefix + CnHm (e.g., NHC6H11)
    p_suffix = r'^(C\d+H\d+)([A-Za-z]+)$'  # CnHm + Suffix (e.g., C6H11NH)
    p_hm_cn_prefix = r'^([A-Za-z]+)(H\d+C\d+)$'  # Prefix + HmCn (e.g., H11C6HN)
    p_hm_cn_suffix = r'^(H\d+C\d+)([A-Za-z]+)$'  # HmCn + Suffix (e.g., H11C6NH)
    
    # New pattern for handling (H7C3)2N format
    p_bracketed_group  = r'^\((H(\d+)C(\d+))\)(\d+)([A-Za-z]+)$'  # Adjusted to handle (H7C3)2N, etc.
    p_reverse_bracketed_group = r'^([A-Za-z]+)\((C(\d+)H(\d+))\)(\d+)$'  # Handles N(C3H7)2, etc.

    # Element and suffix replacement map
    elements = ['S', 'N', 'P', 'C', 'O']
    keys = [f"{e}{suffix}" for e in elements for suffix in ['R"', "R'", "R", "*"]]
    replacement_map = {key: f'{key[0]}*' for key in keys}

    def validate_and_expand(c_count, h_count, prefix=None, suffix=None):
        """Helper to validate CnHm/HmCn and generate SMILES."""
        if h_count != 2 * c_count + 1:  # Check if valid alkyl group
            return False
        smiles = '[CH2]' + 'C' * (c_count - 1)
        if prefix:
            prefix = normalize_ocr_text(prefix, replacement_map)
            smiles = ABBREVIATIONS.get(prefix, prefix) + 'C' * c_count
        elif suffix:
            suffix = normalize_ocr_text(suffix, replacement_map)
            smiles = ABBREVIATIONS.get(suffix, suffix) + 'C' * c_count
        return smiles

    # 1. Handle standalone CnHm or HmCn first
    match_cn_hm = re.match(p_cn_hm, group)
    if match_cn_hm:
        c_count, h_count = int(match_cn_hm.group(1)), int(match_cn_hm.group(2))
        return validate_and_expand(c_count, h_count)

    match_hm_cn = re.match(p_hm_cn, group)
    if match_hm_cn:
        h_count, c_count = int(match_hm_cn.group(1)), int(match_hm_cn.group(2))
        return validate_and_expand(c_count, h_count)

    # 2. Handle CnHm or HmCn with prefix/suffix
    patterns = [
        (p_prefix, p_cn_hm, 1, 2, 'suffix'),
        (p_suffix, p_cn_hm, 2, 1, 'prefix'),
        (p_hm_cn_prefix, p_hm_cn, 1, 2, 'suffix'),
        (p_hm_cn_suffix, p_hm_cn, 2, 1, 'prefix')
    ]

    for pattern, sub_pattern, c_idx, h_idx, aff_type in patterns:
        match = re.match(pattern, group)
        if match:
            cn_hm = match.group(1 if aff_type == 'suffix' else 2)
            affix = match.group(2 if aff_type == 'suffix' else 1)
            sub_match = re.match(sub_pattern, cn_hm)
            if sub_match:
                c_count = int(sub_match.group(c_idx))
                h_count = int(sub_match.group(h_idx))
                return validate_and_expand(
                    c_count, h_count,
                    prefix=affix if aff_type == 'prefix' else None,
                    suffix=affix if aff_type == 'suffix' else None
                )

    base_smiles=False
    # 3. Handle the new (H7C3)2N case TODO may need N2(C3H7)adding 
    match_bracketed_group = re.match(p_bracketed_group, group)
    if match_bracketed_group:
        h_count, c_count = int(match_bracketed_group.group(2)), int(match_bracketed_group.group(3))
        prefix = match_bracketed_group.group(5)
        prefix_n = int(match_bracketed_group.group(4))
        print("h_count, c_count,prefix",[h_count, c_count,prefix])
        unit_smi='C'*c_count
        BACKET_SM=f"({unit_smi})"* prefix_n
        base_smiles=f"[{prefix}]{BACKET_SM}"

    # 4. Handle the new  N(C3H7)2 
    match_reverse_bracketed_group = re.match(p_reverse_bracketed_group, group)
    if match_reverse_bracketed_group:
        c_count, h_count = int(match_reverse_bracketed_group.group(3)), int(match_reverse_bracketed_group.group(4))
        prefix = match_reverse_bracketed_group.group(1)
        prefix_n = int(match_reverse_bracketed_group.group(5))
        print("h_count, c_count,prefix",[h_count, c_count,prefix])
        unit_smi='C'*c_count
        BACKET_SM=f"({unit_smi})"* prefix_n
        base_smiles=f"[{prefix}]{BACKET_SM}"

    if base_smiles:
        # If valid, return the SMILES with the appropriate number of repetitions for the group
        return f"{base_smiles}" 
    
    
    return False


def C_H_expand2(group):
    """
    Expands CnHm or HmCn chemical group notation into SMILES format.
    Supports formats like C6H11, NHC6H11, H11C6, H11C6HN where H = 2C - 1.
    Returns SMILES string or False if invalid.
    """
    # Regex patterns
    p_cn_hm = r'^C(\d+)H(\d+)$'  # Standalone CnHm (e.g., C6H11)
    p_hm_cn = r'^H(\d+)C(\d+)$'  # Standalone HmCn (e.g., H11C6)
    p_prefix = r'^([A-Za-z]+)(C\d+H\d+)$'  # Prefix + CnHm (e.g., NHC6H11)
    p_suffix = r'^(C\d+H\d+)([A-Za-z]+)$'  # CnHm + Suffix (e.g., C6H11NH)
    p_hm_cn_prefix = r'^([A-Za-z]+)(H\d+C\d+)$'  # Prefix + HmCn (e.g., H11C6HN)
    p_hm_cn_suffix = r'^(H\d+C\d+)([A-Za-z]+)$'  # HmCn + Suffix (e.g., H11C6NH)

    # Abbreviation map for common groups
    ABBREVIATIONS2 = {
        'NH': '[NH]', 'CONH': '[C](=O)[NH]', 'HN': '[NH]', 'HNO': '[NH]O', 'NO': '[N]=O'
    }#TODO may need more 

    def validate_and_expand(c_count, h_count, prefix=None, suffix=None):
        """Helper to validate CnHm/HmCn and generate SMILES."""
        if h_count != 2 * c_count - 1:  # Check if H = 2C - 1
            return False
        if c_count % 2 != 0:
            print(f"C#C , c_count have to be 2n!!!")
            return False
        # Base SMILES: [CH] for single carbon, or [CH]C...C for multiple
        smiles = '[C]#C unit repeat' if c_count == 2 else '[C]#C'+'C#C' * int(c_count/2 - 1)#NOTE C have to 2n
        if prefix:
            prefix = ABBREVIATIONS2.get(prefix, prefix)
            smiles = prefix + smiles
        if suffix:  # Changed from elif to if to handle both prefix and suffix
            suffix = ABBREVIATIONS2.get(suffix, suffix)
            smiles += suffix
        return smiles

    # 1. Handle standalone CnHm or HmCn
    match_cn_hm = re.match(p_cn_hm, group)
    if match_cn_hm:
        c_count, h_count = int(match_cn_hm.group(1)), int(match_cn_hm.group(2))
        return validate_and_expand(c_count, h_count)

    match_hm_cn = re.match(p_hm_cn, group)
    if match_hm_cn:
        h_count, c_count = int(match_hm_cn.group(1)), int(match_hm_cn.group(2))
        return validate_and_expand(c_count, h_count)

    # 2. Handle CnHm or HmCn with prefix/suffix
    patterns = [
        (p_prefix, p_cn_hm, 2, 1, 2, 'prefix'),
        (p_suffix, p_cn_hm, 1, 1, 2, 'suffix'),
        (p_hm_cn_prefix, p_hm_cn, 2, 2, 1, 'prefix'),
        (p_hm_cn_suffix, p_hm_cn, 1, 2, 1, 'suffix')
    ]

    for pattern, sub_pattern, group_idx, c_idx, h_idx, aff_type in patterns:
        match = re.match(pattern, group)
        if match:
            cn_hm = match.group(group_idx)
            affix = match.group(3 - group_idx)  # Other group is prefix/suffix
            sub_match = re.match(sub_pattern, cn_hm)
            if sub_match:
                c_count = int(sub_match.group(c_idx))
                h_count = int(sub_match.group(h_idx))
                return validate_and_expand(
                    c_count, h_count,
                    prefix=affix if aff_type == 'prefix' else None,
                    suffix=affix if aff_type == 'suffix' else None
                )

    return False


def H_C_expand(group):
    # 1. 处理 CnHm 在前的格式，例如 'C18H37HNOC'
    match_cn_hm_prefix = re.match(r'(H\d+C\d+)(.+)', group)
    elements = ['S', 'N', 'P', 'C', 'O']
    keys = [f"{e}{suffix}" for e in elements for suffix in ['R"', "R'", "R", "*"]]
    replacement_map = {key: f'{key[0]}*' for key in keys}

    if match_cn_hm_prefix:
        cn_hm = match_cn_hm_prefix.group(1)  # e.g., 'C18H37'
        suffix = match_cn_hm_prefix.group(2)  # e.g., 'HNOC'
        # 处理 CnHm 部分
        match_cn_hm = re.match(r'H(\d+)C(\d+)', cn_hm)
        if match_cn_hm:
            C_count = int(match_cn_hm.group(1))
            H_count = int(match_cn_hm.group(2))
            if H_count != 2 * C_count + 1:
                return False
            else:
                smiles = '[C]' + 'C' * (C_count - 1)
                if suffix:
                    suffix = normalize_ocr_text(suffix, replacement_map)
                    suffix_smi=ABBREVIATIONS[suffix].smiles if suffix in ABBREVIATIONS else suffix
                    sub_smic=sub_smic=suffix_smi +  'C' * (C_count )
                    return sub_smic
                else:
                    return smiles        
        return False
    # 2. 处理 CnHm 在后的格式，例如 'CONHC3H7'
    match_cn_hm_suffix = re.match(r'(.+)(H\d+C\d+)$', group)
    if match_cn_hm_suffix:
        prefix = match_cn_hm_suffix.group(1)  # e.g., 'CONH'
        cn_hm = match_cn_hm_suffix.group(2)  # e.g., 'C3H7'
        # 处理 CnHm 部分
        match_cn_hm = re.match(r'H(\d+)C(\d+)', cn_hm)
        if match_cn_hm:
            C_count = int(match_cn_hm.group(1))
            H_count = int(match_cn_hm.group(2))
            # 可选：验证 H_count，例如直链烷基 H_count = 2 * C_count + 1
            if H_count != 2 * C_count + 1:
                return False
            else:
                smiles = '[C]' + 'C' * (C_count - 1)
                if prefix:
                    prefix = normalize_ocr_text(prefix, replacement_map)
                    prefix_smi=ABBREVIATIONS[prefix].smiles if prefix in ABBREVIATIONS else prefix
                    sub_smic=sub_smic=prefix_smi +  'C' * (C_count )
                    return sub_smic
                else:
                    return smiles  
        return False

    # 3. 原有逻辑处理 CnFm 格式 (e.g., C2F5)
    match_cnfm = re.match(r'H(\d+)C(\d+)', group)
    if match_cnfm:
        C_count = int(match_cnfm.group(1))
        F_count = int(match_cnfm.group(2))
        # 验证氟原子数是否符合全氟烷基的规则：F_count = 2 * C_count + 1
        if F_count != 2 * C_count + 1:
            return False
        smiles = '[C]' + 'C' * (C_count - 1)
        return smiles

def C_F_expand(group):
    # 尝试匹配 CnFm 格式 (e.g., C2F5)
    match_cnfm = re.match(r'C(\d+)F(\d+)', group)
    if match_cnfm:
        C_count = int(match_cnfm.group(1))
        F_count = int(match_cnfm.group(2))
        # 验证氟原子数是否符合全氟烷基的规则：F_count = 2 * C_count + 1
        if F_count != 2 * C_count + 1:
            return False
    else:
        # 尝试匹配 CF2CF3 格式 (e.g., CF2CF3, CF2CF2CF3)
        # 匹配一个或多个 CF2 后跟一个 CF3
        match_cfx = re.match(r'(CF2)*CF3$', group)
        if not match_cfx:
            return False
        # 计算碳原子和氟原子数
        cf2_count = group.count('CF2')  # 每个 CF2 贡献 1 碳和 2 氟
        C_count = cf2_count + 1  # +1 for the terminal CF3
        F_count = cf2_count * 2 + 3  # Each CF2 has 2F, CF3 has 3F
        # 验证氟原子数是否符合全氟烷基的规则
        if F_count != 2 * C_count + 1:
            return False
    # 构建 SMILES 字符串
    smiles = []
    for i in range(C_count):
        if i < C_count - 1:
            # 前面的碳原子：2个氟原子，形式为 C(F)(F)
            if len(smiles)==0:
                smiles.append('[C](F)(F)')
            else:
                smiles.append('C(F)(F)')
        else:
            # 最后一个碳原子：3个氟原子，形式为 [C](F)(F)(F)
            smiles.append('[C](F)(F)(F)')
    
    # 连接所有部分
    return ''.join(smiles)


# '|'.join(list(ABBREVIATIONS.keys()))
original_str ='|'.join(list(ABBREVIATIONS.keys()))
escaped_str = original_str.replace('*', r'\*').replace('(', r'\(').replace(')', r'\)')

FORMULA_REGEX_str='(' + escaped_str + '|R[0-9]*|[A-Z][a-z]+|[A-Z]|[0-9]+|\(|\))' 
# print(escaped_str)
# print(FORMULA_REGEX_str)
FORMULA_REGEX = re.compile(FORMULA_REGEX_str)
# placeholder_atoms
def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    # if ''.join(tokens) != formula:
    #     tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # skip unreasonable number of atoms
        if num > 100000:
            i += 1; continue
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            if next_num > 100000:
                i += 1; continue
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    if expanded==[]:
        return False
    else:
        return expanded    

def replace_bracket(match):
    content = match.group(1)
    # 条件1：包含数字或 '+' 或 '-'，保留整个 [content]
    if re.search(r'\d|\+|-', content):
        return f'[{content}]'
    # 条件2：仅为 'H'，保留
    elif content == 'H':
        return '[H]'
    # 条件3：字符长度 >=2 且包含 'H'，则去除括号和 H
    elif len(content) >= 2 and 'H' in content:
        return ''.join([ch for ch in content if ch != 'H'])
    # 条件4：其他情况，去掉括号
    else:
        return content

    # return re.sub(r'\[([^\[\]]+)\]', replace_bracket, smi)

def formula_regex(abbrev):# molscribe way for the combine abbver style
    tokens = FORMULA_REGEX.findall(abbrev)
    # elements=_parse_tokens(tokens)
    abbrev_exp=_expand_carbon(_parse_tokens(tokens))
    if abbrev_exp==[]:
        return False
    else:
        return abbrev_exp    

def _expand_abbreviationMS(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    # if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] in RGROUP_SYMBOLS and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    elif bond_num == 1:
        return ''
    elif bond_num == 2:
        return '='
    elif bond_num == 3:
        return '#'
    else:
        print(f"check this val  {bond_num} !!!" )

    return ''
def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula_list`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `num_trials`: number of trials
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1

    def dfs(smiles, bonds_left, cur_idx, add_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """
        num_trials = 1
        # end of formula: return result
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  # atom to be added to current atom
        if not isinstance(to_add, str):
            return  smiles, bonds_left, num_trials, False
        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:
                # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # put smiles of "atom" in parentheses and append to smiles; go to next atom to add to current atom
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                # "atom" added uses up remaining bonds of current atom
                # get smiles of "atom" and bonds left on it
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success
        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, [1]):  # try all possible valences of atom added
            add_str = _expand_abbreviationMS(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False

    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)

def swap_paren_bracket(text):
    # Check if string starts with '('
    if not text.startswith('('):
        return text
    # Pattern: match (...) followed by [...]
    pattern = r'^\((.*?)\)\[(.*?)\]'
    # Find match
    match = re.match(pattern, text)
    if match:
        # Swap the groups: [group2](group1)
        return f'[{match.group(2)}]({match.group(1)})'
    
    return text

def convert_ch2_string(s):
    # 匹配 (CH2)后面跟数字或字母的模式
    pattern = r'\(CH2\)(\d+|[a-zA-Z]+)'
    match = re.fullmatch(pattern, s)
    if not match:
        return s  # 如果不是目标模式，返回原字符串
    
    suffix = match.group(1)
    
    if suffix.isdigit():
        n = int(suffix)
        if n == 1:
            return '[CH2]'
        else:
            return '[CH2]' + 'C' * (n - 1)
    else:
        # 处理变量情况，如 (CH2)m
        var = suffix
        print(var,s)
        return s


def process_string_joinused(s):
    # 检查字符串是否以[]开头
    match = re.match(r'^\[([^\]]*)\](.*)$', s)
    if not match:
        return s  # 如果不匹配，直接返回原字符串
    
    content, rest = match.groups()
    # 计算[]中字符数
    char_count = len(content)
    
    # 如果字符数大于1且包含H
    if char_count > 1 and 'H' in content:
        # 移除H及其后连续的数字
        new_content = re.sub(r'H\d*', '', content)
        return f'[{new_content}]{rest}'
    return s

def all_elements_in_dict(lst, dictionary):
    """
    递归检查列表（可能嵌套）中的所有元素是否都存在于字典的键中
    
    :param lst: 要检查的列表（可能包含嵌套列表）
    :param dictionary: 要检查的字典
    :return: 如果所有元素都在字典键中返回True，否则返回False
    """
    for element in lst:
        if isinstance(element, list):
            # 如果是嵌套列表，递归检查
            if not all_elements_in_dict(element, dictionary):
                return False
        else:
            # 如果是普通元素，检查是否在字典键中
            if element not in dictionary:
                return False
    return True

def expand_cf2_to_smiles(input_string):
    # 正则表达式匹配 (CF2)nX 的模式，X 为任意字母数字字符串
    pattern = r'\(CF2\)(\d+)([A-Za-z0-9]+)'
    match = re.match(pattern, input_string)
    if not match:
        return input_string  
    # 提取数字 n 和末尾的化学基团 X
    n = int(match.group(1))
    tail_group = f"[{match.group(2)}]"
    # 构建 SMILES 字符串
    # 每个 CF2 单元是 [C](F)(F)，重复 n 次，最后接 tail_group
    cf2_unit = 'C(F)(F)'
    smiles = '[C](F)(F)' + cf2_unit * (n-1) + tail_group if n > 0 else tail_group
    return smiles

def find_repeating_unit_and_smiles(s):
    match = re.fullmatch(r'(.+?)(?:\1)+', s)
    if match:
        unit = match.group(1)
        repeat_count = len(s) // len(unit)
        # 根据重复单元生成SMILES（适当处理CH2 -> C, CF2 -> CF2）
        if unit == "CH2":
            smiles_unit = "C"  # CH2 -> C
            smi_init="[CH2]"
        elif unit == "CF2":
            smiles_unit = "C(F)(F)"  # CF2保持原样
            smi_init="[C](F)(F)"
        elif unit == "SO2":
            smiles_unit = "S(=O)(=O)"  # SO2保持原样
            smi_init="[S](=O)(=O)"
        else:
            smiles_unit,smi_init='',''
            print(f'please add the repateat patter here !!! for: {s}')
            # smiles_unit = unit  # 其他单元直接使用
        # 生成最终的SMILES
        smiles = smi_init +  smiles_unit * (repeat_count - 1 )
        
        return smiles, repeat_count, unit
    else:
        return None, 0, None  # 如果没有匹配到，则返回None
    
def get_smiles_from_symbol(symbol, mol, bonds):
    """
    Convert symbol (abbrev. or condensed formula) to smiles
    If condensed formula, determine parsing direction and num. bonds on each side using coordinates
    """
    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles
    if symbol in RGROUP_SYMBOLS or (symbol[0] in RGROUP_SYMBOLS and symbol[1:].isdigit()):
        if symbol[1:].isdigit():
            return f'[{symbol[1:]}*]'
        return '*'
    
    if len(symbol) > 20:
        return None
    smiles=convert_ch2_string(symbol)
    if smiles !=symbol:
        return smiles
    if '(CF2)' in symbol:
        smiles=expand_cf2_to_smiles(symbol)
        return smiles
    smiles, repeat_count, unit = find_repeating_unit_and_smiles(symbol)    
    if repeat_count>0:
        return smiles
    
    #TODO@@@ add as speical case or add function, 
    # this is hard encode NOTE fix this next version
    if symbol in ['CH2CH','CHCH2','CH2CH2', 'CH2CH2CH','CH2CH2CH','H2CH2CHC','CHCH2CH2','(CH2)10', 'H2C','CH2',#'CH2CH2NSO2CH3',
              'OCH2CHOHCH2NH','OCH2CHOHCH2','CF2O','OF2C','EtO2CHN','EtO2C',
              'CH2CH2C(O)0CH2CH3','CH2CH2C(O)OCH2CH3','l23I',
              'OCH2CH2OH','OCH2CHCH2CCH3','CH2O',
            '(H4NO)2','SO2NHCH2CH','OCH2CH','OCF2H','COCOOCH2CH3','CH2CH2CH2CH','HCH2CH2CH2C','CF3CF2CF2CF2SO3',
            # 'SO2(CH2)3SO2NHCH2CHCH2OH',
            '(CF2)8H','PH3C','CO','OC',
            'CF2CF2H','NHSO2CH3','CH2CH2C','CH;CH2C(O)0CHCH3','CH2CH2C(O)OCHCH3',
              'NH2','H2N', 'CHO', 'OHC',   'N(SO2CH3)2','CH2CH2O','CH2CH2C(O)OCH2CH3',
              #ACS
              'Ar2P(O)','PhO2S','NHP(O)Ph2','P*Ph3','P+Ph3','NH2.HCl',
              #CLEF
              'S[O]a',
            #USPTO
             '(C3H6O)7CH3','HC','(HC','(CH2CH2CH2CH-)','3(CHCHCHCH272',
            #UOB
            'NHzBrH','NH2BrH',
            #staker
            '(co)','(CO)',
            #JPO
            'CH3CH','CH3CCH3','CH3CO','CH3OCH2','CO2C','CH2CO2CH3',"COCl",
         ]:#NOTE this are not passed by _condensed_formula_list_to_smiles function
        #TODO fix me in next version, may be need LLM to track this
            # Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
            # Substitution(['NH2','H2N'], '[NH2;D1]', "[NH2]", 0.1),
        #TODO symbol2SMILES() need dig ChemDraw 
        if symbol in ['CH2CH','CHCH2']:smiles='[CH2][CH]'
        elif symbol in ['PH3C']:smiles='[CH2]P'
        elif symbol in ['l23I']:smiles='[I]'
        elif symbol in ['HC','(HC']:smiles='[CH]'
        elif symbol in ['NHzBrH','NH2BrH']:smiles='[NH2].Br'
        elif symbol in ['(C3H6O)7CH3']:smiles="[O]CCC"+"OCCC"*6+'C'#TODO maybe as function
        elif symbol in ['NH2.HCl']:smiles="[NH2].Cl"
        elif symbol in ['CH2CH2CH2CH','(CH2CH2CH2CH-)']:smiles='[CH2]CC[CH]'
        elif symbol in ['3(CHCHCHCH272', 'CHCHCHCH2']:smiles='[CH]CC[CH2]'
        # elif symbol in ['D']:smiles='[2H]'
        elif symbol in [ 'CH3CH']:smiles='[CH]C'
        elif symbol in [ 'CH2CO2CH3']:smiles='[CH2]C(=O)OC'
        elif symbol in [ 'CO2C']:smiles='[C](=O)O[C]'
        elif symbol in [ 'CH3CCH3']:smiles='[C](C)(C)'
        elif symbol in [ 'CH3CO']:smiles='[C](=O)C'
        elif symbol in [ 'CH3OCH2']:smiles='[CH2]OC'

        elif symbol in [ '(co)','(CO)']:smiles='[C](=O)'
        elif symbol in ['Ar2P(O)']:smiles='[P](*)(*)(=O)'
        elif symbol in ['PhO2S']:smiles='[S](=O)(=O)c1ccccc1'
        elif symbol in ['CO','OC']:smiles='[C](=O)'
        elif symbol in ['CH2O']:smiles='[CH2][O]'
        elif symbol in ['P*Ph3','P+Ph3',]:smiles='[P+](c1ccccc1)(c1ccccc1)(c1ccccc1)'
        elif symbol in ['NHP(O)Ph2']:smiles='[NH]P(=O)(c1ccccc1)c1ccccc1'
        elif symbol in ['CH;CH2C(O)0CHCH3','CH2CH2C(O)OCHCH3']:smiles='[CH2]CC(=O)OCC'
        elif symbol in ['CH2CH2CH','H2CH2CHC','CHCH2CH2']:smiles='[CH2][CH2][CH]'
        elif symbol in ['CH2CH2CH2CH']:smiles='[CH2]CC[CH]'
        elif symbol in ['HCH2CH2CH2C']:smiles='[CH]CC[CH2]'
        elif symbol in ['H2C','CH2']:smiles='[CH2]'
        elif symbol in ['H2CH2C','CH2CH2']:smiles='[CH2][CH2]'
        elif symbol in ['CHO', 'OHC']:smiles="[CH](=O)"
        elif symbol in ['NH2','H2N']:smiles="[NH2]"
        elif symbol in ['(CF2)8H',]:smiles="[C](F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)"
        elif symbol in ['CH2CH2C(O)OCH2CH3','CH2CH2C(O)0CH2CH3']:smiles='[CH2]CC(=O)OCC'
        elif symbol in ['CF3CF2CF2CF2SO3']:smiles='[S](=O)(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)(F)'
        elif symbol in ['S[O]a']:smiles='[S](=O)'
        elif symbol in ['COCl']:smiles='[C](=O)Cl'



        elif symbol in ['OCF2H']:smiles="[O]C(F)(F)"
        elif symbol in ['CF2O']:smiles="[C](F)(F)[O]"
        elif symbol in ['OF2C']:smiles="[O][C](F)(F)"
        elif symbol in ['CF2CF2H']:smiles="[C](F)(F)C(F)(F)"
        # elif symbol in ['CH2CH2NSO2CH3']:smiles='[CH2]CNS(=O)(C)=O'
        elif symbol in ['CH2CH2O']:smiles='[CH2]CO'
        elif symbol in ['OCH2CH2OH']:smiles='[O]CCO'#NOTE Chemdraw may give some idea
        elif symbol in ['EtO2CHN']:smiles='[N]C(=O)OCC'
        elif symbol in ['OCH2CHOHCH2NH']:smiles='[O]CC(O)CN'
        elif symbol in ['OCH2CHCH2CCH3']:smiles='[O]C[CH]C[C]C'
        elif symbol in ['(H4NO)2']:smiles='[O]NON'
        elif symbol in ['SO2NHCH2CH']:smiles='[S](=O)(=O)NC[CH]'
        elif symbol in ['N(SO2CH3)2']:smiles='[N](S(=O)(=O)C)(S(=O)(=O)C)'
        elif symbol in ['CH2CH2C(O)OCH2CH3']:smiles='[CH2]CC(=O)OCC'
        elif symbol in ['OCH2CH']:smiles='[O]C[CH]'
        elif symbol in ['EtO2C']:smiles='C(=O)OCC'
        elif symbol in ['CH2CH2C']:smiles='[CH2]C[C]'
        elif symbol in ['NHSO2CH3']:smiles='[NH]S(=O)(=O)C'
        elif symbol in ['COCOOCH2CH3']:smiles='C(=O)C(=O)OCC'
        # elif symbol in ['SO2(CH2)3SO2NHCH2CHCH2OH']:smiles='[S](=O)(=O)CCCS(=O)(=O)NC[C]CO'
        # elif symbol in ['H4NO3S']:smiles='[S]NCC'
        # elif symbol in ['(CH2)10','[CH]CCCCCCCCC']:smiles='[CH]CCCCCCCCC'#as in  convert_ch2_string()
        else:smiles=None
        return smiles

    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))#TODO aromtaic bond effect ??
    formula_list = _expand_carbon(_parse_formula(symbol))
    # all_in_dict = all(fl in ABBREVIATIONS for fl in formula_list)
    all_in_dict=all_elements_in_dict(formula_list,ABBREVIATIONS)
    #total_bonds, bonds_left 机制是有问题的， 所以需要以上的修补，机制不完善
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    # if debug:
    print(f'{[formula_list, total_bonds]} use _condensed_formula_list_to_smiles {success} <<-------\n {smiles}')
    if success:
        smiles=swap_paren_bracket(smiles)
        return smiles
    elif all_in_dict :#NOTE resolve abbv combine 
        # smiles=ABBREVIATIONS[formula_list[0]].smiles
        key = extract_abbreviation_key(formula_list[0])
        if key in ABBREVIATIONS:
            smiles = ABBREVIATIONS[key].smiles
        else:
            # raise ValueError(f"Abbreviation {key} not found in ABBREVIATIONS.")
            print(f"Abbreviation {key} not found in ABBREVIATIONS.")
            smiles=''
        for fl_i in range(1,len(formula_list)):
            cur_smi=process_string_joinused(ABBREVIATIONS[formula_list[fl_i]].smiles)
            smiles += cur_smi
        return smiles

    return None

def abbrev2smile(abbrev,abbrev_exp,mol,idx):
    
    atom_gost = mol.GetAtomWithIdx(idx)
    bonds_gost = atom_gost.GetBonds()
    sub_smi = get_smiles_from_symbol(abbrev, mol, bonds_gost)

    if sub_smi:
        # print(f"succes expanding {abbrev},{abbrev_exp}\n{sub_smi}\t{idx}")
        return sub_smi
    else:
        print(f"failed expanding {abbrev},{abbrev_exp}\n{sub_smi}\t{idx}")
        return '[*]'

    # if abbrev_exp[0] in ABBREVIATIONS: 
    #     init_smi=ABBREVIATIONS[abbrev_exp[0]].smiles
    # else:
    #     if len(abbrev_exp[0])==1:
    #         init_smi=f'[{abbrev_exp[0]}]'
    #     else:
    #         print(f"{abbrev_exp[0]} @@@formula_regex")
    #         init_smi=f'[{abbrev_exp[0]}]'
    # # init_smi=ABBREVIATIONS[abbrev_exp[0]].smiles if abbrev_exp[0] in ABBREVIATIONS else 
    # if len(abbrev_exp)==1:
    #     sub_smi=init_smi
    #     return sub_smi
    # elif len(abbrev_exp)>1:
    #     sub_smi=init_smi
    #     for i_ in range(1,len(abbrev_exp)):

    #         smi_=ABBREVIATIONS[abbrev_exp[i_]].smiles if abbrev_exp[i_] in ABBREVIATIONS else  f'[{abbs[i_]}]'
    #         smi_2=re.sub(r'\[([^\[\]]+)\]', replace_bracket, smi_)        
    #         sub_smi +=smi_2#default combine them with single bond TODO fixme ifneed
    #     return sub_smi
    # else:
    #     return False
def replace_cg_notation(astr):
    def replacer(match):
        h_count = int(match.group(1))
        c_count = (h_count - 1) // 2
        return f'C{c_count}H{h_count}'

    return re.sub(r'CgH(\d+)', replacer, astr)


def _expand_abbreviation(abbrev, mol,idx):# ABBREVIATIONS, RGROUP_SYMBOLS, ELEMENTS):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*].
    """

    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    # elif sub_smi_HC:return sub_smi_HC
    elif N_C_H_expand(abbrev):return N_C_H_expand(abbrev)
    elif C_F_expand(abbrev):return C_F_expand(abbrev)
    elif C_H_expand2(abbrev):return C_H_expand2(abbrev)
    elif C_H_expand(abbrev):return C_H_expand(abbrev)
    elif C_H_affixExpand(abbrev):return C_H_affixExpand(abbrev)
    # elif abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
    elif abbrev in RGROUP_SYMBOLS or (abbrev[0] in RGROUP_SYMBOLS and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
    elif abbrev in ELEMENTS:
        return f'[{abbrev}]'

    elif formula_regex(abbrev):
        abbrev_exp= formula_regex(abbrev)
        return abbrev2smile(abbrev,abbrev_exp,mol,idx)#last use Molscribe way
    
    match = re.match(r'^(\d+)?(.*)', abbrev)
    if match:
        numeric_part, remaining_part = match.groups()
        if remaining_part in ELEMENTS:
            return f'[{abbrev}]'
        elif numeric_part:
            return f'[{numeric_part}*]'

    else:
        print(f"fixme !!!@@@@: {abbrev}")

    return '[*]'

def count_current_bonds(mol, atom_idx):
    """Count current bonds (including bond order) for an atom."""
    atom = mol.GetAtomWithIdx(atom_idx)
    return sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())

debug_not=True

def expandABB(mol, ABBREVIATIONS, placeholder_atoms):#, RGROUP_SYMBOLS, ELEMENTS):
    mols = [mol]
    # 逆序遍历 placeholder_atoms，确保删除后不会影响后续索引
    for idx in sorted(placeholder_atoms.keys(), reverse=True) :
        group = placeholder_atoms[idx]
        group_smiles = _expand_abbreviation(group,mol,idx)
        submol = Chem.MolFromSmiles(group_smiles)  # 获取官能团的子分子
        try:
            submol_rw = Chem.RWMol(submol)  # 转换为可编辑的 RWMol
        except Exception as e:
            print(f"abbver: {group}")
            print(f'try to convert {group_smiles} to sub_mol')
            print(e)
            if debug_not:
                print(f"Failed to convert {group_smiles} to sub_mol, using placeholder [*] instead.")
                submol = Chem.MolFromSmiles('[*]') 
                submol_rw = Chem.RWMol(submol)
            else:
                raise e#NOTE use it when debugging with adding abber and fixing rules in det_engine.py

        # 1. 识别 submol 的 anchor atoms（连接点）
        anchor_atoms = [0]#always use the fisrt atom as anchor atom
        for atom in submol_rw.GetAtoms():
            # 具有自由基的原子或标记为连接点的原子（例如 [*]）
            if atom.GetNumRadicalElectrons() > 0 and atom.GetIdx() not in anchor_atoms:# or atom.GetSymbol() == '*':
                anchor_atoms.append(atom.GetIdx())
        # 2. 复制主分子
        new_mol = Chem.RWMol(mol)
        placeholder_idx = idx
        # 3. 记录 placeholder (*) 原子的邻居及其键类型
        bonds_info = []
        for bond in new_mol.GetBonds():
            if bond.GetBeginAtomIdx() == placeholder_idx:
                bonds_info.append({
                    "neighbor": bond.GetEndAtomIdx(),
                    "bond_type": bond.GetBondType()
                })
            elif bond.GetEndAtomIdx() == placeholder_idx:
                bonds_info.append({
                    "neighbor": bond.GetBeginAtomIdx(),
                    "bond_type": bond.GetBondType()
                })

        # 4. 断开 placeholder 的所有键
        for bond_info in bonds_info:
            new_mol.RemoveBond(placeholder_idx, bond_info["neighbor"])

        # 5. 删除 placeholder 原子
        new_mol.RemoveAtom(placeholder_idx)

        # 6. 重新计算邻居索引（删除后索引变化）
        adjusted_bonds_info = []
        for bond_info in bonds_info:
            neighbor = bond_info["neighbor"]
            if neighbor < placeholder_idx:
                adjusted_neighbor = neighbor
            else:
                adjusted_neighbor = neighbor - 1  # 索引因删除原子而减 1
            adjusted_bonds_info.append({
                "neighbor": adjusted_neighbor,
                "bond_type": bond_info["bond_type"]
            })

        # 7. 合并 submol
        new_mol = Chem.RWMol(Chem.CombineMols(new_mol, submol_rw))

        # 8. 计算 submol 的 anchor atoms 在合并后的索引
        submol_atom_offset = new_mol.GetNumAtoms() - submol_rw.GetNumAtoms()
        new_anchor_indices = [submol_atom_offset + anchor_idx for anchor_idx in anchor_atoms]

        # 9. 重新连接官能团，使用原始键类型
        if len(new_anchor_indices) == 1:
            # 单连接点情况：所有邻居连接到唯一的 anchor atom
            anchor_idx = new_anchor_indices[0]
            for bond_info in adjusted_bonds_info:
                neighbor = bond_info["neighbor"]
                bond_type = bond_info["bond_type"]
                new_mol.AddBond(neighbor, anchor_idx, bond_type)
                # 重置自由基电子数
                a1 = new_mol.GetAtomWithIdx(neighbor)
                a2 = new_mol.GetAtomWithIdx(anchor_idx)
                a1.SetNumRadicalElectrons(0)
                a2.SetNumRadicalElectrons(0)
        else:
            #   # 多连接点情况：先尝试按顺序连接, 如果* 连*  会存在多种合理价态的不同分子情况
            # 多连接点情况：根据邻居数量和 anchor atoms 分配连接           
            if len(adjusted_bonds_info) > len(new_anchor_indices):
                print(adjusted_bonds_info,'  <---adjusted_bonds_info')
                print(new_anchor_indices,'<---new_anchor_indices')
                # raise ValueError(f"Too many neighbors ({len(adjusted_bonds_info)}) for submol with {len(new_anchor_indices)} anchor atoms.")
            # for i, bond_info in enumerate(adjusted_bonds_info):
            #     # 按顺序将邻居连接到 anchor atoms
            #     anchor_idx = new_anchor_indices[i % len(new_anchor_indices)]
            #     neighbor = bond_info["neighbor"]
            #     bond_type = bond_info["bond_type"]
            #     new_mol.AddBond(neighbor, anchor_idx, bond_type)
            #     # 重置自由基电子数
            #     a1 = new_mol.GetAtomWithIdx(neighbor)
            #     a2 = new_mol.GetAtomWithIdx(anchor_idx)
            #     a1.SetNumRadicalElectrons(0)
            #     a2.SetNumRadicalElectrons(0)
            # 跟踪每个 anchor 的当前成键数
            anchor_bond_counts = {idx: new_mol.GetAtomWithIdx(idx).GetTotalValence() for idx in new_anchor_indices}
            print(anchor_bond_counts,'<---anchor_bond_counts')
            # max_valence = {6: 4, 7: 3, 8: 2}  # 示例：C=4, N=3, O=2，需根据实际原子类型扩展
            adjusted_bonds_info = sorted(adjusted_bonds_info, key=lambda x: x['neighbor'])
            if mol.GetNumConformers() > 0:#as some mol may not have the conf dispite pass the 2d assign process
                pos_0 = mol.GetConformer().GetAtomPosition(adjusted_bonds_info[0]['neighbor'])
                pos_1 = mol.GetConformer().GetAtomPosition(adjusted_bonds_info[-1]['neighbor'])
                print(pos_0.x,pos_1.x,"xxx",adjusted_bonds_info)
                # if group =='SO2NH':
                #     if pos_0.x <pos_1.x:
                #         adjusted_bonds_info=[adjusted_bonds_info[-1],adjusted_bonds_info[0]]
                # elif group =='NHO2S':
                #     if pos_0.x <pos_1.x:
                #         adjusted_bonds_info=[adjusted_bonds_info[-1],adjusted_bonds_info[0]]


            for bond_info in adjusted_bonds_info:
                neighbor = bond_info["neighbor"]
                bond_type = bond_info["bond_type"]
                bond_valence = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(bond_type, 1)
                # 寻找未饱和的 anchor
                selected_anchor_idx = None
                for anchor_idx in new_anchor_indices:
                    atom = new_mol.GetAtomWithIdx(anchor_idx)
                    atomic_num = atom.GetAtomicNum()
                    current_valence = anchor_bond_counts[anchor_idx]
                    max_allowed = max(VALENCES.get( atom.GetSymbol(), [1]))  # 默认最大价态为1（可根据需求调整）
                    if current_valence + bond_valence <= max_allowed:
                        selected_anchor_idx = anchor_idx
                        break
                if selected_anchor_idx is None:
                    continue  # 跳过，当前没有可用的未饱和 anchor
                # 添加键
                new_mol.AddBond(neighbor, selected_anchor_idx, bond_type)
                # 更新成键数
                anchor_bond_counts[selected_anchor_idx] += bond_valence
                # 重置自由基电子数
                a1 = new_mol.GetAtomWithIdx(neighbor)
                a2 = new_mol.GetAtomWithIdx(selected_anchor_idx)
                a1.SetNumRadicalElectrons(0)
                a2.SetNumRadicalElectrons(0)



            # 多连接点情况：先尝试按顺序连接
            # success = False
            # temp_mol = Chem.RWMol(new_mol)  # 备份分子
            # try:
            #     for i, bond_info in enumerate(adjusted_bonds_info):
            #         anchor_idx = new_anchor_indices[i % len(new_anchor_indices)]
            #         neighbor = bond_info["neighbor"]
            #         bond_type = bond_info["bond_type"]
            #         temp_mol.AddBond(neighbor, anchor_idx, bond_type)
            #         # 重置自由基电子数
            #         a1 = temp_mol.GetAtomWithIdx(neighbor)
            #         a2 = temp_mol.GetAtomWithIdx(anchor_idx)
            #         a1.SetNumRadicalElectrons(0)
            #         a2.SetNumRadicalElectrons(0)
            #     # 验证价态
            #     Chem.SanitizeMol(temp_mol)
            #     new_mol = temp_mol
            #     success = True
            # except Chem.rdchem.MolSanitizeException:
            #     # 价态不合理，尝试反序 anchor atoms
            #     temp_mol = Chem.RWMol(new_mol)  # 恢复备份
            #     reversed_anchors = new_anchor_indices[::-1]  # 反序 anchor atoms
            #     try:
            #         for i, bond_info in enumerate(adjusted_bonds_info):
            #             anchor_idx = reversed_anchors[i % len(reversed_anchors)]
            #             neighbor = bond_info["neighbor"]
            #             bond_type = bond_info["bond_type"]
            #             temp_mol.AddBond(neighbor, anchor_idx, bond_type)
            #             # 重置自由基电子数
            #             a1 = temp_mol.GetAtomWithIdx(neighbor)
            #             a2 = temp_mol.GetAtomWithIdx(anchor_idx)
            #             a1.SetNumRadicalElectrons(0)
            #             a2.SetNumRadicalElectrons(0)
            #         # 验证价态
            #         Chem.SanitizeMol(temp_mol)
            #         new_mol = temp_mol
            #         success = True
            #     except Chem.rdchem.MolSanitizeException:
            #         print(f"Failed to connect submol with {len(new_anchor_indices)} anchor atoms to {len(adjusted_bonds_info)} neighbors.")
            #         raise ValueError("Unable to create valid molecule with either anchor order.")
            # if not success:
            #     raise ValueError("Unable to create valid molecule.")
       
        # 10. 更新主分子
        mol = new_mol
        mols.append(mol)

    # 输出修改后的分子 SMILES
    modified_smiles = Chem.MolToSmiles(mols[-1])
    return mols[-1], modified_smiles


def is_valid_chem_text(text):
    """检查化学表达式是否只包含大小写字母、数字和成对括号，且括号成对"""
    if not text:
        return False
    if text.isdigit():
        return False
    # 检查是否只包含大小写字母、数字、括号
    if not re.match(r'^[A-Za-z0-9()]+$', text):
        return False
    # 检查括号是否成对
    stack = []
    for char in text:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack or stack[-1] != '(':
                return False
            stack.pop()
    return len(stack) == 0

def select_chem_expression(orig_text, orig_score, scaled_text, scaled_score, cropped_img_orig, cropped_img_scaled):
    """选择更合理的化学表达式"""
    # 计算分数的绝对值差
    score_diff = abs(orig_score - scaled_score)
    if scaled_text in orig_text and orig_text in ABBREVIATIONS:
        print(f'use orig_text as include the sacled and in ABBREVIATIONS {orig_text}')
        return orig_text, orig_score, cropped_img_orig
    elif orig_text in scaled_text and scaled_text in ABBREVIATIONS:
        print(f'use scaled_text as include the orig_text and in ABBREVIATIONS {scaled_text}')
        return scaled_text, scaled_score, cropped_img_scaled
    
    # 检查两个表达式的有效性
    orig_valid = is_valid_chem_text(orig_text)
    scaled_valid = is_valid_chem_text(scaled_text)
    
    #other condition here
    # 如果分差大于0.1，选择分数高的
    if score_diff > 0.1:
        if orig_valid and scaled_valid:
            if orig_score >= scaled_score and orig_text:
                return orig_text, orig_score, cropped_img_orig
            elif scaled_text:
                return scaled_text, scaled_score, cropped_img_scaled
        elif orig_valid and not scaled_valid:
            return orig_text, orig_score, cropped_img_orig
        elif scaled_valid and not orig_valid:
            return scaled_text, scaled_score, cropped_img_scaled
        else:
            print(f"Both texts are invalid: orig_text='{orig_text}', scaled_text='{scaled_text}'")
            if orig_score >= scaled_score:
                return orig_text, orig_score, cropped_img_orig
            else:
                return scaled_text, scaled_score, cropped_img_scaled
    # 如果分差小于0.1，选择更合理的化学表达式
    else:
        # 如果只有一个有效，选择有效的
        if orig_valid and not scaled_valid:
            return orig_text, orig_score, cropped_img_orig
        elif scaled_valid and not orig_valid:
            return scaled_text, scaled_score, cropped_img_scaled
        # 如果都有效，比较长度
        elif orig_valid and scaled_valid:
            if orig_text in ABBREVIATIONS and scaled_text not in ABBREVIATIONS:
                if  N_C_H_expand(scaled_text) or C_F_expand(scaled_text) or C_H_expand2(scaled_text) or C_H_expand(scaled_text):
                    if len(scaled_text)> len(orig_text):
                        return scaled_text, scaled_score, cropped_img_scaled
                return orig_text, orig_score, cropped_img_orig
            elif orig_text not in ABBREVIATIONS and scaled_text  in ABBREVIATIONS:
                if  N_C_H_expand(orig_text) or C_F_expand(orig_text) or C_H_expand2(orig_text) or C_H_expand(orig_text):
                    if len(orig_text)> len(scaled_text):
                        return  orig_text, orig_score, cropped_img_orig
                return scaled_text, scaled_score, cropped_img_scaled
            elif orig_text not in ABBREVIATIONS and scaled_text  not in ABBREVIATIONS:
                if len(orig_text) > len(scaled_text):
                    return orig_text, orig_score, cropped_img_orig
                else:
                    if len(orig_text) == len(scaled_text):
                        if orig_score >= scaled_score :
                            return orig_text, orig_score, cropped_img_orig
                        else:
                            return scaled_text, scaled_score, cropped_img_scaled
                    return scaled_text, scaled_score, cropped_img_scaled

            elif orig_text in ABBREVIATIONS and scaled_text  in ABBREVIATIONS:
                if len(orig_text) >= len(scaled_text):
                    return orig_text, orig_score, cropped_img_orig
                else:
                    return scaled_text, scaled_score, cropped_img_scaled
        # 如果都不有效，优先选择 orig（若存在）
        elif orig_text:
            return orig_text, orig_score, cropped_img_orig
        elif scaled_text:
            return scaled_text, scaled_score, cropped_img_scaled
    
    # 默认返回 scaled（若存在）
    return scaled_text, scaled_score, cropped_img_scaled if scaled_text else (None, None, None)

# def expandABB(mol,ABBREVIATIONS,  placeholder_atoms):# RGROUP_SYMBOLS, ELEMENTS):
#     mols = [mol]
   
#     # Process placeholders in reverse order to avoid index issues
#     for idx in sorted(placeholder_atoms.keys(), reverse=True):
#         group = placeholder_atoms[idx]
#         group_smiles = _expand_abbreviation(group)# ABBREVIATIONS, RGROUP_SYMBOLS, ELEMENTS)
        
#         try:
#             submol = Chem.MolFromSmiles(group_smiles)
#             if not submol:
#                 raise ValueError(f"Invalid SMILES for group {group}: {group_smiles}")
#             submol_rw = RWMol(submol)
#         except Exception as e:
#             print(f"Error processing SMILES for group {group}: {e}")
#             continue
        
#         # Create a new editable molecule
#         new_mol = RWMol(mol)
#         placeholder_idx = idx
        
#         # Get neighbors of the placeholder atom
#         neighbors = [nb.GetIdx() for nb in new_mol.GetAtomWithIdx(placeholder_idx).GetNeighbors()]
        
#         # Identify anchor atoms in submol (atoms marked as [*] or with isotope labels)
#         anchor_atoms = []
#         for atom in submol.GetAtoms():
#             if atom.GetNumRadicalElectrons() > 0:
#                 #atom.GetSymbol() == '*' or atom.GetIsotope() > 0:
#                 anchor_atoms.append(atom.GetIdx())
        
#         # Validate number of anchor atoms vs. neighbors
#         if len(anchor_atoms) != len(neighbors):
#             print(f"Warning: Mismatch between anchor atoms ({len(anchor_atoms)}) and neighbors ({len(neighbors)}) for group {group}")
#             print(len(anchor_atoms), len(neighbors))
#             if len(anchor_atoms)==0:
#                anchor_atoms.append(0)# use the first atom of submol as default such as PPh3
        
        
#         # Remove bonds involving the placeholder atom
#         bonds_to_remove = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
#                           for bond in new_mol.GetBonds()
#                           if bond.GetBeginAtomIdx() == placeholder_idx or bond.GetEndAtomIdx() == placeholder_idx]
#         for bond in bonds_to_remove:
#             new_mol.RemoveBond(bond[0], bond[1])
        
#         # Remove the placeholder atom
#         new_mol.RemoveAtom(placeholder_idx)
        
#         # Adjust neighbor indices after atom removal
#         new_neighbors = [n - 1 if n > placeholder_idx else n for n in neighbors]
        
#         # Combine molecules
#         new_mol = RWMol(CombineMols(new_mol, submol_rw))
        
#         # Connect anchor atoms to neighbors
#         submol_offset = new_mol.GetNumAtoms() - submol.GetNumAtoms()
#         for anchor_idx, neighbor_idx in zip(anchor_atoms, new_neighbors):
#             new_anchor_idx = submol_offset + anchor_idx
#             new_mol.AddBond(neighbor_idx, new_anchor_idx, Chem.BondType.SINGLE)
            
#             # Reset radical electrons
#             new_mol.GetAtomWithIdx(neighbor_idx).SetNumRadicalElectrons(0)
#             new_mol.GetAtomWithIdx(new_anchor_idx).SetNumRadicalElectrons(0)
        
#         mol = new_mol
#         mols.append(mol)
    
#     # Generate final SMILES
#     try:
#         modified_smiles = Chem.MolToSmiles(mols[-1])
#     except Exception as e:
#         print(f"Error generating SMILES: {e}")
#         return mols[-1], None
    
#     return mols[-1], modified_smiles


# def _expand_abbreviation(abbrev):
#     """
#     Expand abbreviation into its SMILES; also converts [Rn] to [n*]
#     Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
#     """
#     if abbrev in ABBREVIATIONS: 
#         return ABBREVIATIONS[abbrev].smiles
#     elif abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):

#         if abbrev[1:].isdigit():
#             return f'[{abbrev[1:]}*]'
#     elif abbrev in ELEMENTS:#ocr tool need this
#         return f'[{abbrev}]'
#     # try  abbrev    

#     match = re.match(r'^(\d+)?(.*)', abbrev)
#     if match:
#         numeric_part, remaining_part = match.groups()
#         if remaining_part in ELEMENTS:
#             return f'[{abbrev}]'
#         else:
#             if numeric_part:
#                 abbrev=f'[{numeric_part}*]'
#     return '[*]'



# def expandABB(mol,ABBREVIATIONS, placeholder_atoms):
#     mols = [mol]
#     # **第三步: 替换 * 并合并官能团**
#     # 逆序遍历 placeholder_atoms，确保删除后不会影响后续索引
#     for idx in sorted(placeholder_atoms.keys(), reverse=True):
#         group = placeholder_atoms[idx]  # 获取官能团名称
#         # print(idx, group)
#         group=_expand_abbreviation(group)
#         submol = Chem.MolFromSmiles(group)  # 获取官能团的子分子
#         submol_rw = RWMol(submol)  # 让 submol 变成可编辑的 RWMol
#         anchor_atom_idx = 0  # 选择 `submol` 的第一个原子作为连接点 as defined in ABBREVIATIONS
#         # **1. 复制主分子**
#         new_mol = RWMol(mol)
#         # **2. 计算 `*` 在 `new_mol` 中的索引**
#         placeholder_idx = idx
#         # **3. 记录 `*` 原子的邻居**
#         neighbors = [nb.GetIdx() for nb in new_mol.GetAtomWithIdx(placeholder_idx).GetNeighbors()]
#         # **4. 断开 `*` 的所有键**
#         bonds_to_remove = []  # 记录要断开的键
#         for bond in new_mol.GetBonds():
#             if bond.GetBeginAtomIdx() == placeholder_idx or bond.GetEndAtomIdx() == placeholder_idx:
#                 bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
#         for bond in bonds_to_remove:
#             new_mol.RemoveBond(bond[0], bond[1])
#         # **5. 删除 `*` 原子**
#         new_mol.RemoveAtom(placeholder_idx)
#         # **6. 重新计算 `neighbors`（删除后索引变化）**
#         new_neighbors = []
#         for neighbor in neighbors:
#             if neighbor < placeholder_idx:
#                 new_neighbors.append(neighbor)
#             else:
#                 new_neighbors.append(neighbor - 1)  # 因为删除了一个原子，所有索引 -1
#         # **7. 合并 `submol`**
#         new_mol = RWMol(CombineMols(new_mol, submol_rw))

#         # **8. 计算 `submol` 的第一个原子在合并后的位置**
#         new_anchor_idx = new_mol.GetNumAtoms() - len(submol_rw.GetAtoms()) + anchor_atom_idx

#         # **9. 重新连接官能团**
#         for neighbor in new_neighbors:
#             # print(neighbor, new_anchor_idx, "!!")
#             new_mol.AddBond(neighbor, new_anchor_idx, Chem.BondType.SINGLE)
#             a1=new_mol.GetAtomWithIdx(neighbor)
#             a2=new_mol.GetAtomWithIdx(new_anchor_idx)
#             a1.SetNumRadicalElectrons(0)
#             a2.SetNumRadicalElectrons(0)## 将自由基电子数设为 0,as has added new bond
#         # **10. 更新主分子**
#         mol = new_mol
#         mols.append(mol)
#     # 输出修改后的分子 SMILES
#     modified_smiles = Chem.MolToSmiles(mols[-1])
#     # print(f"修改后的分子 SMILES: {modified_smiles}")            
#     return mols[-1], modified_smiles





# Helper function to check if two boxes overlap
def boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    bx1, by1, bx2, by2 = box2
    return not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2)

def boxes_overlap2(atombonx, bondbox):
    """
    检查两个矩形框是否重叠，并返回 bondbox 中不重叠一端到中心 10% 位置的坐标。
    
    参数:
        atombonx: tuple (x1, y1, x2, y2) 表示原子框的坐标
        bondbox: tuple (bx1, by1, bx2, by2) 表示键框的坐标
        
    返回:
        tuple (x, y) 表示 bondbox 不重叠一端到中心 80% 位置的坐标，如果完全包含返回 (None, None)
    """
    x1, y1, x2, y2 = atombonx
    bx1, by1, bx2, by2 = bondbox
    
    # 计算 bond_box 的中心坐标
    bond_center_x = (bx1 + bx2) / 2
    bond_center_y = (by1 + by2) / 2
    
    # 辅助函数：计算点到 atom_box 中心的距离
    def distance_to_center(x, y):
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
    
    # 辅助函数：计算从中心到端点 80% 位置的坐标
    def get_80_percent_point(far_x, far_y):
        # 从中心到端点的向量，按 80% 比例缩放
        dx = far_x - bond_center_x
        dy = far_y - bond_center_y
        new_x = bond_center_x + 0.7 * dx#let added H close to the heavy neighbor
        new_y = bond_center_y + 0.7 * dy
        return new_x, new_y
    
    # 检查是否完全不相交
    if (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
        # 完全不相交，返回较远一端到中心 80% 位置
        dist1 = distance_to_center(bx1, by1)
        dist2 = distance_to_center(bx2, by2)
        far_x, far_y = (bx2, by2) if dist2 > dist1 else (bx1, by1)
        return get_80_percent_point(far_x, far_y)
    
    # 检查是否完全包含在 atom_box 内
    if (bx1 >= x1 and bx2 <= x2 and by1 >= y1 and by2 <= y2):
        # bondbox 完全在 atom_box 内，无法确定不重叠部分，返回 bond_center_x, bond_center_y
        # return None, None
        return bond_center_x, bond_center_y

    # 检查一端是否在 atom_box 内
    if (bx1 >= x1 and bx1 <= x2 and by1 >= y1 and by1 <= y2):
        # bx1, by1 在 atom_box 内，返回 bx2, by2 到中心 80% 位置
        return get_80_percent_point(bx2, by2)
    elif (bx2 >= x1 and bx2 <= x2 and by2 >= y1 and by2 <= y2):
        # bx2, by2 在 atom_box 内，返回 bx1, by1 到中心 80% 位置
        return get_80_percent_point(bx1, by1)
    
    # 处理部分相交但两端都不在 atom_box 内的情况
    # 返回较远一端到中心 80% 位置
    dist1 = distance_to_center(bx1, by1)
    dist2 = distance_to_center(bx2, by2)
    far_x, far_y = (bx2, by2) if dist2 > dist1 else (bx1, by1)
    return get_80_percent_point(far_x, far_y)


charge_labels = [19,20,21,22,23]
def outputbox_update(output,charge_labels,bond_labels,lab2idx):
    bonds_mask = np.array([True if ins  in bond_labels else False for ins in output['pred_classes']])
    bond_bbox=output['bbox'][bonds_mask]
    atoms_mask = np.array([True if ins not in bond_labels and ins not in charge_labels else False for ins in output['pred_classes']])
    atom_bbox=output['bbox'][atoms_mask]
    new_atoms=[]
    b_len=3
    single_odd_b2a=dict()
    for bi,bb in enumerate(bond_bbox):
        overlapped_atoms = []
        overlapped_abox=[]
        for ai,aa in enumerate(atom_bbox):
            overlap_flag=boxes_overlap(bb, aa)#TODO use tghe atom bond box overlap get bond atom mapping,then built mol
            if overlap_flag:
                # print(bb, aa,overlap_flag)
                overlapped_atoms.append(ai)
                overlapped_abox.append(aa)
        if len(overlapped_atoms) == 1:
            single_odd_b2a[bi]=overlapped_atoms
            # Compute the non-overlapping part of the bond box to place hydrogen
            non_overlapping_x,non_overlapping_y=boxes_overlap2(overlapped_abox[0], bb)
            new_atom_out={'bbox':    np.array([non_overlapping_x - b_len, 
                                    non_overlapping_y - b_len,
                                    non_overlapping_x + b_len, 
                                    non_overlapping_y + b_len]).reshape(-1,4),
                'bbox_centers': np.array([non_overlapping_x,non_overlapping_y]).reshape(-1,2),
                'scores':       np.array([1.0]),
                'pred_classes': np.array([lab2idx['H']])}
            new_atoms.append(new_atom_out)

    output2_=copy.deepcopy(output)
    for boxout in new_atoms:
        for k,arr in boxout.items():
            value_or_row=output2_[k]
            if arr.ndim == 1:
                output2_[k]=np.append(value_or_row, arr)
            elif arr.ndim >= 2:
                output2_[k] = np.concatenate([value_or_row, arr], axis=0)
            else:
                print('errprs, unkown conditions !!!@')
    return output2_, single_odd_b2a


def remove_unconnected_hydrogens(mol):
    """
    移除分子中不与重原子相连的氢原子（包括孤立 H 和只连到其他 H 的 H）。
    
    参数:
        mol: RDKit Mol 对象
        
    返回:
        移除氢原子后的 RWMol 对象
    """
    # 转换为可编辑的 RWMol 对象
    molexp = Chem.RWMol(mol)
    to_remove = []

    # 遍历所有原子
    for atom in molexp.GetAtoms():
        if atom.GetSymbol() == 'H':  # 只处理氢原子
            neighbors = atom.GetNeighbors()
            # 检查邻居中是否有重原子
            has_heavy_atom = False
            for neighbor in neighbors:
                if neighbor.GetSymbol() != 'H':  # 如果邻居不是 H，则是重原子
                    has_heavy_atom = True
                    break
            # 如果没有重原子邻居，标记为移除
            if not has_heavy_atom:
                to_remove.append(atom.GetIdx())
    # 按索引从大到小排序，避免移除时索引混乱
    to_remove.sort(reverse=True)
    
    # 移除标记的原子
    for ai in to_remove:
        molexp.RemoveAtom(ai)
    return molexp

from rdkit import Chem
from rdkit.Chem import AllChem

def remove_unconnected_hydrogens2(mol):
    """
    移除分子中不与重原子相连的氢原子（包括孤立 H 和只连到其他 H 的 H），并返回移除的氢原子坐标。

    参数:
        mol: RDKit Mol 对象

    返回:
        rw_mol: 移除氢原子后的 RWMol 对象
        removed_h_coords: 移除的氢原子的坐标列表 [(x1, y1, z1), (x2, y2, z2), ...]
    """
    # 转换为可编辑的 RWMol 对象
    rw_mol = Chem.RWMol(mol)
    to_remove = []

    # 获取分子的构象（假设只有一个构象）
    conformer = rw_mol.GetConformer()

    # 存储移除的氢原子坐标
    removed_h_coords = []

    # 遍历所有原子
    for atom in rw_mol.GetAtoms():
        if atom.GetSymbol() == 'H':  # 只处理氢原子
            neighbors = atom.GetNeighbors()
            # 检查邻居中是否有重原子
            has_heavy_atom = False
            for neighbor in neighbors:
                if neighbor.GetSymbol() != 'H':  # 如果邻居不是 H，则是重原子
                    has_heavy_atom = True
                    break
            # 如果没有重原子邻居，标记为移除，并记录坐标
            if not has_heavy_atom:
                to_remove.append(atom.GetIdx())
                pos = conformer.GetAtomPosition(atom.GetIdx())
                removed_h_coords.append((pos.x, pos.y, pos.z))
    # 按索引从大到小排序，避免移除时索引混乱
    to_remove.sort(reverse=True)
    # 移除标记的原子
    for ai in to_remove:
        rw_mol.RemoveAtom(ai)

    return rw_mol, removed_h_coords

def detect_unconnected_hydrogens(mol):
    rw_mol = Chem.RWMol(mol)
    to_remove = []
    # 获取分子的构象（假设只有一个构象）
    conformer = rw_mol.GetConformer()
    # 存储移除的氢原子坐标
    removed_h_coords = []
    # 遍历所有原子
    for atom in rw_mol.GetAtoms():
        if atom.GetSymbol() == 'H':  # 只处理氢原子
            neighbors = atom.GetNeighbors()
            # 检查邻居中是否有重原子
            has_heavy_atom = False
            for neighbor in neighbors:
                if neighbor.GetSymbol() != 'H':  # 如果邻居不是 H，则是重原子
                    has_heavy_atom = True
                    break
            # 如果没有重原子邻居，标记为移除，并记录坐标
            if not has_heavy_atom:
                to_remove.append(atom.GetIdx())
                pos = conformer.GetAtomPosition(atom.GetIdx())
                removed_h_coords.append((pos.x, pos.y, pos.z))
    # 按索引从大到小排序，避免移除时索引混乱
    to_remove.sort(reverse=True)
    return to_remove

def view_box_center2(bond_bbox, bond_centers, bond_scores, bond_classes,overlap_dist_thresh=5.0, 
                     max_centers_per_box=5,
                     plot_view=False,
                     ):
    """
    筛选和可视化 bond_bbox 和 bond_centers，处理重叠圆和过多中心的框。
    
    参数:
        bond_bbox: numpy array, [x1, y1, x2, y2] 格式的框坐标
        bond_centers: numpy array, [x, y] 格式的中心坐标
        bond_scores: numpy array, 得分
        overlap_dist_thresh: float，判断圆重叠的距离阈值（默认为 5 个单位）
        max_centers_per_box: int，一个框内允许的最大中心数（超过则移除）
    
    返回:
        tuple: (筛选后的 bond_bbox, bond_centers, bond_scores)
    """
    # 确保输入形状匹配
    assert len(bond_bbox) == len(bond_centers) == len(bond_scores), "Input arrays must have equal length"
    n = len(bond_bbox)
    # Step 1: 处理重叠的 bond_centers（保留得分最高的）
    keep_centers = np.ones(n, dtype=bool)  # 标记要保留的中心
    for i in range(n):
        if not keep_centers[i]:
            continue
        for j in range(i + 1, n):
            if not keep_centers[j]:
                continue
            # 计算两个中心之间的欧几里得距离
            dist = np.sqrt(np.sum((bond_centers[i] - bond_centers[j]) ** 2))
            if dist < overlap_dist_thresh:
                # 如果重叠，保留得分较高的
                if bond_scores[i] > bond_scores[j]:
                    keep_centers[j] = False
                else:
                    keep_centers[i] = False
    # 应用初步筛选
    bond_bbox = bond_bbox[keep_centers]
    bond_centers = bond_centers[keep_centers]
    bond_scores = bond_scores[keep_centers]
    bond_classes= bond_classes[keep_centers]
    n = len(bond_bbox)  # 更新数量
    # Step 2: 检查每个框内的中心数量
    keep_boxes = np.ones(n, dtype=bool)  # 标记要保留的框
    for i in range(n):
        # 计算框内的中心数量
        x1, y1, x2, y2 = bond_bbox[i]
        centers_in_box = np.sum((bond_centers[:, 0] >= x1) & (bond_centers[:, 0] <= x2) &
                                (bond_centers[:, 1] >= y1) & (bond_centers[:, 1] <= y2))
        if centers_in_box > max_centers_per_box:
            keep_boxes[i] = False
    # 应用最终筛选
    final_bond_bbox = bond_bbox[keep_boxes]
    final_bond_centers = bond_centers[keep_boxes]
    final_bond_scores = bond_scores[keep_boxes]
    final_bond_classes= bond_classes[keep_boxes]
    if plot_view:
        # 可视化（可选）
        fig, ax = plt.subplots(figsize=(10, 10))
        for box in final_bond_bbox:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        for center in final_bond_centers:
            circle = Circle(center, radius=5, edgecolor='red', facecolor='none', linewidth=1)
            ax.add_patch(circle)
        
        # 设置坐标轴范围
        x_min = min(final_bond_bbox[:, 0].min(), final_bond_centers[:, 0].min()) - 10
        x_max = max(final_bond_bbox[:, 2].max(), final_bond_centers[:, 0].max()) + 10
        y_min = min(final_bond_bbox[:, 1].min(), final_bond_centers[:, 1].min()) - 10
        y_max = max(final_bond_bbox[:, 3].max(), final_bond_centers[:, 1].max()) + 10
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        ax.set_title("Filtered Boxes and Centers")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.show()
    else:
        fig=None
    return final_bond_bbox, final_bond_centers, final_bond_scores,final_bond_classes,fig

def calculate_iou(box1, box2):
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




import numpy as np
def get_overlap_region(box1, box2):
    """
    Get the overlapping region of two boxes.
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max]
    
    Returns:
        tuple: (x_min, y_min, x_max, y_max) of overlap region, or None if no overlap
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return None  # No overlap
    return (x1, y1, x2, y2)

def are_bond_connected(box1, box2, bond_bboxes, bond_iou_threshold=0.1):
    """
    Check if two atom boxes are connected by a bond box, with bond center in overlap region.
    
    Args:
        box1, box2: atom boxes to check
        bond_bboxes: array of bond boxes
        bond_iou_threshold: IoU threshold for initial bond overlap
    
    Returns:
        bool: True if connected by a bond with center in overlap region
    """
    # Get the overlap region of the two atom boxes
    overlap_region = get_overlap_region(box1, box2)
    if overlap_region is None:
        return False  # No overlap between atom boxes

    ox_min, oy_min, ox_max, oy_max = overlap_region

    for bond_box in bond_bboxes:
        # Preliminary IoU check
        iou1 = calculate_iou(box1, bond_box)
        iou2 = calculate_iou(box2, bond_box)
        if iou1 > bond_iou_threshold and iou2 > bond_iou_threshold:
            # Calculate bond box center
            bond_center_x = (bond_box[0] + bond_box[2]) / 2
            bond_center_y = (bond_box[1] + bond_box[3]) / 2
            
            # Check if bond center is within the overlap region
            if (ox_min <= bond_center_x <= ox_max and 
                oy_min <= bond_center_y <= oy_max):
                return True
    return False

def calculate_iou(box1, box2):
    """
    计算两个边界框的 IoU
    box1, box2: [x_min, y_min, x_max, y_max]
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

def nms(atom_bboxes, atom_scores, atom_classes, iou_threshold=0.5):
    """
    应用非极大值抑制 (NMS)
    atom_bboxes: 列表，包含所有边界框 [x_min, y_min, x_max, y_max]
    atom_scores: 列表，包含每个边界框的置信度
    atom_classes: 列表，包含每个边界框的类别
    iou_threshold: IoU 阈值，用于判断是否抑制
    返回: 保留的边界框、类别和置信度的索引
    """
    # 按置信度排序，获取索引
    indices = np.argsort(atom_scores)[::-1]  # 从高到低排序

    keep_indices = []
    while len(indices) > 0:  # 使用 len(indices) 替代 indices.size
        # 保留当前最高置信度的框
        current_idx = indices[0]
        keep_indices.append(current_idx)

        # 计算当前框与其他框的 IoU
        ious = np.array([calculate_iou(atom_bboxes[current_idx], atom_bboxes[idx]) for idx in indices[1:]])
        # 找出 IoU > threshold 的索引（相对于 indices[1:] 的偏移）
        suppress_indices = indices[1:][ious > iou_threshold]
        # 更新 indices，去除当前框和被抑制的框
        indices = np.setdiff1d(indices, np.concatenate(([current_idx], suppress_indices)))
        # 调试信息
        # print(f"Current idx: {current_idx}, rmoved: {suppress_indices}, Remaining: {indices}")
        # print(f"Current idx: {current_idx}, rmoved: {suppress_indices}, IOU: {ious}")

    # 返回保留的框、类别和置信度
    kept_bboxes = np.array([atom_bboxes[i] for i in keep_indices])
    kept_classes = np.array([atom_classes[i] for i in keep_indices])
    kept_scores = np.array([atom_scores[i] for i in keep_indices])

    return kept_bboxes, kept_classes, kept_scores

def count_bond_overlaps(box, bond_bboxes, bond_iou_threshold=0.1):
    """
    Count how many bond boxes overlap with an atom box.
    
    Args:
        box: atom box [x_min, y_min, x_max, y_max]
        bond_bboxes: array of bond boxes
        bond_iou_threshold: IoU threshold for overlap
    
    Returns:
        int: number of overlapping bond boxes
    """
    return sum(1 for bond_box in bond_bboxes if calculate_iou(box, bond_box) > bond_iou_threshold)


def count_bond_overlaps(box, bond_bboxes, bond_iou_threshold=0.01):
    """Count how many bond boxes overlap with an atom box."""
    return sum(1 for bond_box in bond_bboxes if calculate_iou(box, bond_box) > bond_iou_threshold)

def count_atom_overlaps(box, all_bboxes, exclude_idx, min_iou=0.01):
    """Count how many other atom boxes overlap with this box."""
    return sum(1 for i, other_box in enumerate(all_bboxes) 
               if i != exclude_idx and calculate_iou(box, other_box) > min_iou)

def merge_low_iou_boxes(kept_bboxes, kept_classes, kept_scores, bond_bboxes, 
                       merge_threshold=0.5, score_threshold=0.7, bond_iou_threshold=0.01, 
                       high_iou_threshold=0.8, large_score_threshold=0.5):
    """
    Merge or filter boxes with IoU conditions, removing large low-score boxes first.
    
    Args:
        kept_bboxes: array, atom bounding boxes [x_min, y_min, x_max, y_max]
        kept_classes: array, class labels (e.g., 0 for 'C')
        kept_scores: array, confidence scores
        bond_bboxes: array, bond bounding boxes
        merge_threshold: float, upper IoU threshold for merging
        score_threshold: float, score threshold to preserve boxes
        bond_iou_threshold: float, IoU threshold for bond connectivity
        high_iou_threshold: float, IoU threshold for high-IoU merging
        large_score_threshold: float, score threshold for large box removal (default 0.5)
    
    Returns:
        tuple: (merged_bboxes, merged_classes, merged_scores)
    """
    if len(kept_bboxes) <= 1:
        return kept_bboxes, kept_classes, kept_scores

    kept_bboxes = np.array(kept_bboxes)
    kept_classes = np.array(kept_classes)
    kept_scores = np.array(kept_scores)
    bond_bboxes = np.array(bond_bboxes)

    # Step 0: Remove large boxes with low scores, high atom overlaps, and high bond overlaps
    areas = (kept_bboxes[:, 2] - kept_bboxes[:, 0]) * (kept_bboxes[:, 3] - kept_bboxes[:, 1])
    median_area = np.median(areas)
    keep_mask = np.ones(len(kept_bboxes), dtype=bool)

    for i in range(len(kept_bboxes)):
        if kept_scores[i] < large_score_threshold:
            atom_overlaps = count_atom_overlaps(kept_bboxes[i], kept_bboxes, i)
            bond_overlaps = count_bond_overlaps(kept_bboxes[i], bond_bboxes, bond_iou_threshold)
            is_large = areas[i] > median_area  # Define "large" as above median
            if is_large and atom_overlaps >= 2 and bond_overlaps >= 3:
                keep_mask[i] = False
                print(f"Removed large low-score box idx {i}: score {kept_scores[i]}, "
                      f"area {areas[i]}, atom overlaps {atom_overlaps}, bond overlaps {bond_overlaps}")

    # Filter boxes
    kept_bboxes = kept_bboxes[keep_mask]
    print(f"afterRemoved large low-score atom box::{len(kept_bboxes)} ")
    kept_classes = kept_classes[keep_mask]
    kept_scores = kept_scores[keep_mask]
    if len(kept_bboxes) == 0:
        return np.array([]), np.array([]), np.array([])

    merged_bboxes = []
    merged_classes = []
    merged_scores = []
    used_indices = set()

    # Step 1: Merge boxes with IoU > high_iou_threshold
    i = 0
    while i < len(kept_bboxes):
        if i in used_indices:
            i += 1
            continue

        high_iou_group = [i]
        for j in range(len(kept_bboxes)):
            if j in used_indices or j == i:
                continue
            iou = calculate_iou(kept_bboxes[i], kept_bboxes[j])
            if iou > high_iou_threshold:
                high_iou_group.append(j)

        if len(high_iou_group) > 1:#atom box ovrlaped
            group_scores = kept_scores[high_iou_group]
            max_score_idx = high_iou_group[np.argmax(group_scores)]
            merged_bboxes.append(kept_bboxes[max_score_idx])
            merged_classes.append(kept_classes[max_score_idx])
            merged_scores.append(kept_scores[max_score_idx])
            used_indices.update(high_iou_group)
            print(f"Merged high-IoU (> {high_iou_threshold}) boxes: {high_iou_group}, "
                  f"kept index: {max_score_idx}")
        i += 1

    # Step 2: Process remaining boxes
    i = 0
    while i < len(kept_bboxes):
        if i in used_indices:
            i += 1
            continue

        current_indices = [i]
        for j in range(len(kept_bboxes)):
            if j in used_indices or j == i:
                continue
            iou = calculate_iou(kept_bboxes[i], kept_bboxes[j])#IOU between atoms box
            if 0.05 <= iou < merge_threshold:#better detect model with score matters
                #any small IOU between atoms will processed here
                if kept_scores[j]<0.7:
                    current_indices.append(j)

        group_indices = current_indices
        group_scores = kept_scores[group_indices]
        group_classes = kept_classes[group_indices]
        group_bboxes = kept_bboxes[group_indices]

        max_score = np.max(group_scores)
        max_score_idx = group_indices[np.argmax(group_scores)]

        if max_score >= score_threshold:
            bond_connected = False
            if len(group_indices) > 1:
                for idx1, idx2 in zip(group_indices[:-1], group_indices[1:]):
                    if are_bond_connected(kept_bboxes[idx1], kept_bboxes[idx2], 
                                        bond_bboxes, bond_iou_threshold):
                        bond_connected = True
                        break
            if bond_connected:
                for idx in group_indices:
                    merged_bboxes.append(kept_bboxes[idx])
                    merged_classes.append(kept_classes[idx])
                    merged_scores.append(kept_scores[idx])
                print(f"Kept all bond-connected boxes: {group_indices}")
            else:
                bond_overlap_counts = [count_bond_overlaps(kept_bboxes[idx], bond_bboxes, 
                                      bond_iou_threshold) for idx in group_indices]
                max_overlaps = max(bond_overlap_counts)
                candidates = [idx for idx, count in zip(group_indices, bond_overlap_counts) 
                            if count == max_overlaps]
                best_idx = max(candidates, key=lambda idx: kept_scores[idx])
                merged_bboxes.append(kept_bboxes[best_idx])
                merged_classes.append(kept_classes[best_idx])
                merged_scores.append(kept_scores[best_idx])
                # print(f"No bond box overlap, kept box with most bond overlaps: {best_idx}, "
                #       f"overlap count: {max_overlaps}")
        else:
            if len(group_indices) == 1:
                merged_bboxes.append(kept_bboxes[i])
                merged_classes.append(kept_classes[i])
                merged_scores.append(kept_scores[i])
                print(f"Merged lower IOU @@ ONLY ONE box {i}")
            else:
                new_bbox = [
                    np.min(group_bboxes[:, 0]),  # x_min
                    np.min(group_bboxes[:, 1]),  # y_min
                    np.max(group_bboxes[:, 2]),  # x_max
                    np.max(group_bboxes[:, 3])   # y_max
                ]
                merged_bboxes.append(new_bbox)
                merged_classes.append(group_classes[np.argmax(group_scores)])
                merged_scores.append(max_score)
                print(f"Merged low-score boxes: {group_indices}")
        used_indices.update(group_indices)
        i += 1
    
    print(f"after processs low IOU atom box::{len(merged_bboxes)} ")
    return (np.array(merged_bboxes), np.array(merged_classes), np.array(merged_scores))


def refine_boxes(atom_bboxes, atom_scores, atom_classes, bond_bboxes, 
                 nms_iou_threshold=0.5, merge_threshold=0.5, score_threshold=0.5, 
                 bond_iou_threshold=0.01, high_iou_threshold=0.8):
    """
    Iteratively apply NMS and merge until the number of boxes stabilizes.
    
    Args:
        atom_bboxes, atom_scores, atom_classes: Initial atom box data
        bond_bboxes: Bond box data
        nms_iou_threshold, merge_threshold, score_threshold, bond_iou_threshold, high_iou_threshold: Parameters
    
    Returns:
        tuple: (final_bboxes, final_classes, final_scores)
    """
    current_bboxes = np.array(atom_bboxes)
    current_classes = np.array(atom_classes)
    current_scores = np.array(atom_scores)
    prev_count = len(current_bboxes) + 1  # Ensure at least one iteration

    iteration = 0
    while len(current_bboxes) < prev_count:
        print(f"\nIteration {iteration}: Starting with {len(current_bboxes)} boxes")
        prev_count = len(current_bboxes)

        # Apply NMS
        kept_bboxes, kept_classes, kept_scores = nms(
            current_bboxes, current_scores, current_classes, iou_threshold=nms_iou_threshold
        )
        print(f"After NMS: {len(kept_bboxes)} boxes")

        # Apply merge_low_iou_boxes
        merged_bboxes, merged_classes, merged_scores = merge_low_iou_boxes(
            kept_bboxes, kept_classes, kept_scores, bond_bboxes,
            merge_threshold=merge_threshold, score_threshold=score_threshold,
            bond_iou_threshold=bond_iou_threshold, high_iou_threshold=high_iou_threshold
        )
        print(f"After merge: {len(merged_bboxes)} boxes")

        # Update for next iteration
        current_bboxes = merged_bboxes
        current_classes = merged_classes
        current_scores = merged_scores
        iteration += 1

    print(f"Converged after {iteration} iterations with {len(current_bboxes)} boxes")
    return current_bboxes, current_scores, current_classes

def merge_low_iou_boxes_old(kept_bboxes, kept_classes, kept_scores, merge_threshold=0.3):
    """
    合并 IoU < merge_threshold 的边界框，使用较高 score 的 class
    """
    if len(kept_bboxes) <= 1:
        return kept_bboxes, kept_classes, kept_scores

    merged_bboxes = []
    merged_classes = []
    merged_scores = []
    used_indices = set()

    for i in range(len(kept_bboxes)):
        if i in used_indices:
            continue

        # 找到 IoU < merge_threshold 的框组
        current_indices = [i]
        for j in range(i + 1, len(kept_bboxes)):
            if j in used_indices:
                continue
            iou = calculate_iou(kept_bboxes[i], kept_bboxes[j])
            if iou < merge_threshold and iou >0.01:
                current_indices.append(j)

        # 获取相关框的 score, class, 和 bbox
        scores = kept_scores[current_indices]
        classes = kept_classes[current_indices]
        bboxes = kept_bboxes[current_indices]

        max_score = np.max(scores)
        max_score_idx = current_indices[np.argmax(scores)]

        if max_score > 0.5:
            # 保留 score 最大的框
            merged_bboxes.append(kept_bboxes[max_score_idx])
            merged_classes.append(kept_classes[max_score_idx])
            merged_scores.append(kept_scores[max_score_idx])
        else:
            # 合并框，取最小和最大坐标
            new_bbox = [
                np.min(bboxes[:, 0]),  # x_min
                np.min(bboxes[:, 1]),  # y_min
                np.max(bboxes[:, 2]),  # x_max
                np.max(bboxes[:, 3])   # y_max
            ]
            merged_bboxes.append(new_bbox)
            merged_classes.append(0)#repalce with *
            merged_scores.append(max_score)

        # 标记已使用的索引
        used_indices.update(current_indices)

    # 转换为 NumPy 数组
    merged_bboxes = np.array(merged_bboxes)
    merged_classes = np.array(merged_classes)
    merged_scores = np.array(merged_scores)

    return merged_bboxes, merged_classes, merged_scores

############################################################################################################################################################
#molscrbe evaluate
from SmilesPE.pretokenizer import atomwise_tokenizer

def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success

def convert_smiles_to_canonsmiles(
    smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)

def tanimoto_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto
    except:
        return 0


def compute_tanimoto_similarities(gold_smiles, pred_smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        similarities = p.starmap(tanimoto_similarity, [(gs, ps) for gs, ps in zip(gold_smiles, pred_smiles)])
    return similarities

class SmilesEvaluator(object):
    def __init__(self, gold_smiles, num_workers=16, tanimoto=False):
        self.gold_smiles = gold_smiles
        self.num_workers = num_workers
        self.tanimoto = tanimoto
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_cistrans=True,
                                                                     num_workers=num_workers)
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                   ignore_chiral=True, ignore_cistrans=True,
                                                                   num_workers=num_workers)
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)

    def _replace_empty(self, smiles_list):
        """Replace empty SMILES in the gold, otherwise it will be considered correct if both pred and gold is empty."""
        return [smiles if smiles is not None and type(smiles) is str and smiles != "" else "<empty>"
                for smiles in smiles_list]

    def evaluate(self, pred_smiles, include_details=False):
        results = {}
        if self.tanimoto:
            results['tanimoto'] = np.mean(compute_tanimoto_similarities(self.gold_smiles, pred_smiles))
        # Ignore double bond cis/trans
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                                ignore_cistrans=True,
                                                                num_workers=self.num_workers)
        results['canon_smiles'] = np.mean(np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        if include_details:
            results['canon_smiles_details'] = (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        # Ignore chirality (Graph exact match)
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                              ignore_chiral=True, ignore_cistrans=True,
                                                              num_workers=self.num_workers)
        results['graph'] = np.mean(np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral))
        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if '@' in g])
        results['chiral'] = np.mean(chiral[:, 0] == chiral[:, 1]) if len(chiral) > 0 else -1
        return results



############################################################################################################################################################
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# @torch.no_grad()
# def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir,
#     annot_file=f'/home/jovyan/rt-detr/data/real_processed/CLEF_with_charge/annotations/val.json',
#     outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/output_charge_CLEF.csv',
#     ):
#     model.eval()
#     criterion.eval()

#     metric_logger = MetricLogger(delimiter="  ")
#     header = 'Test:'

#     iou_types = postprocessors.iou_types
#     coco_evaluator = CocoEvaluator(base_ds, iou_types)

#     panoptic_evaluator = None
    
#     # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     # home='/home/jovyan/rt-detr'
#     # dataset = 'CLEF'
#     # annot_file=f'/home/jovyan/rt-detr/data/real_processed/{dataset}_with_charge/annotations/test.json'
#     # outcsv_filename/home/jovyan/rt-detr/rt-detr/output/output_charge_{dataset}.csv'


#     # annot_file=f'/home/jovyan/rt-detr/data/real_processed/{dataset}_with_charge/annotations/test.json'
#     # outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/output_charge_{dataset}.csv'
#     with open(annot_file, 'r') as file: 
#         data = json.load(file)




#     image_id_to_name = {}

#     for image_data in data['images']:
#         image_id = image_data['id']
#         image_path = image_data['file_name']
#         image_name = os.path.basename(image_path)
#         image_id_to_name[image_id] = image_name

#     res_smiles = []
#     bond_labels = [13,14,15,16,17,18]
#     idx_to_labels={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
#                         9:'I',10:'P',11:'H',12:'Si',
#                         #bond
#                         13:'single',14:'wdge',15:'dash',
#                         16:'=',17:'#',18:':',#aromatic
#                         #charge
#                         19:'-4',20:'-2',
#                         21:'-1',#-
#                         22:'+1',#+
#                         23:'2',
#                         }
#     lab2idx={v:k for k,v in idx_to_labels.items()}
#     #indigo bond type stero maping
#     indi_bond={
#             "1":'single', "2":'=',"3":'#',"4":':',"5":'wdge',"6":'dash',
#     }


#     smiles_data = pd.DataFrame({'file_name': [],
#                                 'SMILES':[]})
    
#     output_dict = {}
#     target_dict = {}
#     filtered_output_dict = {}
#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         outputs = model(samples)

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
#         results = postprocessors(outputs, orig_target_sizes)#RTDETRPostProcessor@@src/zoo/rtertr
#         #results: a list of dict  label box score
#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}

#         for target, output in zip(targets, results):
#             output_dict[target['image_id'].item()] = output
    
#     stats = {}
#     # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     if coco_evaluator is not None:
#         if 'bbox' in iou_types:
#             # stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats
#         if 'segm' in iou_types:
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()



#     # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
#     # ocr_recognition_only = get_ocr_recognition_only(force_cpu=False)   
#     # caption_remover = CaptionRemover(force_cpu=True)
#     for key, value in output_dict.items():#TODO improve here
#         selected_indices = value['scores'] > 0.5#may be >=0.5 cut off, as used the sigmoid?
#         if value['labels'][selected_indices].size(0) != 0:#no good prediction
#             filtered_output_dict[key] = {
#                 'labels': value['labels'][selected_indices],# may be selected_indices ==0 as all small than0.5
#                 'boxes': value['boxes'][selected_indices],
#                 'scores': value['scores'][selected_indices]
#             }
#         else:
#             ima_name=image_id_to_name[key]
#             print(key,"all prediction scores small 0.5!!",len(output_dict),f"{ima_name}")##

#     for i,(key,value) in enumerate(filtered_output_dict.items()):
#         result = []#TODO need a box2mol or graph
#         smi_mol=output_to_smiles(value,idx_to_labels,bond_labels,result)#TODO use the idx_to_labels numer to if --else
#         if smi_mol:
#             res_smiles.append(smi_mol[0])  #TODO check this erro other0
#         else:
#             res_smiles.append('')
            
#         new_row = {'file_name':image_id_to_name[key], 'SMILES':res_smiles[i]}
#         smiles_data = smiles_data._append(new_row, ignore_index=True)
    
#     print(f"will save {len(smiles_data)} dataframe into csv") 
#     smiles_data.to_csv(outcsv_filename, index=False)

#     return stats, coco_evaluator

def remove_bond_directions_if_no_chiral(mol):
    # 检查分子是否有效
    if mol is None:
        return None
    # 计算手性中心
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    # 如果没有手性中心，移除单键的立体化学标记
    if not chiral_centers:
        for bond in mol.GetBonds():
            # 只处理单键
            if bond.GetBondType() == Chem.BondType.SINGLE:
                # 移除楔形和虚线标记
                bond.SetBondDir(Chem.BondDir.NONE)
    return mol
#######################################################################################
def molExpanding(mol_rebuit,placeholder_atoms,wdbs,bond_dirs,alignmol=False):
    cm=copy.deepcopy(mol_rebuit)
    # print(placeholder_atoms)
    expand_mol, expand_smiles= expandABB(cm,ABBREVIATIONS, placeholder_atoms)
    rdm=copy.deepcopy(expand_mol)
    AllChem.Compute2DCoords(rdm)
    target_mol, ref_mol=rdm, cm

    if alignmol:
        mcs=rdFMCS.FindMCS([target_mol, ref_mol], # larger,small order
                        atomCompare=rdFMCS.AtomCompare.CompareAny,
                        # bondCompare=rdFMCS.BondCompare.CompareAny,
                        ringCompare=rdFMCS.RingCompare.IgnoreRingFusion,
                        matchChiralTag=False,
        )
        atommaping_pairs=g_atompair_matches([target_mol, ref_mol],mcs)
        atomMap=atommaping_pairs[0]
        try:
            rmsd2=rdkit.Chem.rdMolAlign.AlignMol(prbMol=target_mol, refMol=ref_mol, atomMap=atomMap,maxIters=2000000)
        except Exception as e:
            print(atomMap,"@@@@")
            print(e)
        #after get atomMap
        c2p={cur:pre for cur, pre in atomMap}
        p2c={pre:cur for cur, pre in atomMap}
        for b in wdbs:#add bond direction
            p0,p1=int(b[0]), int(b[1])#may be not in the atomMap as the mcs_sub
            if p0 in p2c.keys() and p1 in p2c.keys():
                c0,c1=p2c[p0],p2c[p1]
                # print("[pre0,pre1]vs[c0,c1]current atom id",[p0,p1],[c0,c1])
                b_=target_mol.GetBondBetweenAtoms(c0,c1)
                if b_:
                    b_.SetBondDir(bond_dirs[b[3]])
        expandStero_smi=Chem.MolToSmiles(target_mol)#directly will not add the stero info into smiles, must have the assing steps
    else:
        expandStero_smi =expand_smiles 
        
    m=target_mol.GetMol()
    # Chem.SanitizeMol(m)
    Chem.DetectBondStereochemistry(m)
    Chem.AssignChiralTypesFromBondDirs(m)
    Chem.AssignStereochemistry(m)#expandStero_smi ,  m 

    return expandStero_smi, m  


def remove_backslash_and_slash(input_string):
    if "\\" in input_string:
        input_string = input_string.replace("\\", "")
    if "/" in input_string:
        input_string = input_string.replace("/", "")

    return input_string


def remove_number_before_star(input_string):
    result = list(input_string) 

    i = 0
    while i < len(result):
        if result[i] == '*' and i!= len(result) -1:  
            #*c1c(*)c(*)c(C(*)(*)C(C)C)c(*)c1* --> *c1c(*)c(*)c(C(*)(*)C(C)C)c(*)c1*
            j = i - 1
            if result[j-1].isalpha(): 
                continue
            while j >= 0 and result[j].isdigit():
                result[j] = ''  
                j -= 1
        i += 1

    return ''.join(result)

def remove_SP(input_string):
    pattern = r'\[([^@]*)@?[A-Z0-9]*\]'
    # if "S@SP1" in input_string:
    #     input_string = input_string.replace("S@SP1", "S")
    # elif "S@SP2" in input_string:
    #     input_string = input_string.replace("S@SP2", "S")
    # elif "S@SP3" in input_string:
    #     input_string = input_string.replace("S@SP3", "S")
    input_string = re.sub(r'@SP[1-3]', '', input_string)
    if '@TB' in input_string:
        result = re.sub(pattern, r'[\1]', input_string)
        input_string=result
    return input_string

def rdkit_canonicalize_smiles(smiles):
    Aad_string = r'([A-Z][a-z]*)([0-9]+)'
    tokens = atomwise_tokenizer(smiles)
    for j, token in enumerate(tokens):
        if token[0] == '[' and token[-1] == ']':
            symbol = token[1:-1]
            # matches = re.findall(Aad_string, symbol)#findall may give not wanted, such as [BH2], shuld not change
            matches = re.match(Aad_string, symbol)
            if matches:
                letters, numbers = matches.groups()
                print(f"{letters} {numbers}")
                # tokens[j] = f'[{symbol[1:]}*]'
                tokens[j] = '*'
            elif symbol in RGROUP_SYMBOLS:# or (symbol[0] in RGROUP_SYMBOLS and abbrev[1:].isdigit()):
                tokens[j] = '*'
            elif Chem.AtomFromSmiles(token) is None:
                tokens[j] = '*'

    smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=False)
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success

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

import logging

def check_and_fix_valence(smiles_or_list):
    """
    Check atom valences in a SMILES string or a list [smiles, suffix/prefix].
    Fix unusual valences (e.g., N(2)) by adding/removing hydrogens to maintain neutrality.
    Returns: (corrected_smiles_or_list, warnings)
    """
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    warnings = []

    # Standard valence dictionary for common atoms
    standard_valences = {
        'C': [4],
        'N': [3],  # Prioritize valence 3 for neutral nitrogen (e.g., amines, amides)
        'O': [2],
        'H': [1],
        'F': [1]
    }

    # Handle input: SMILES string or list from C_H_expand
    if isinstance(smiles_or_list, list):
        smiles, other_part = smiles_or_list
    else:
        smiles, other_part = smiles_or_list, None

    # Process main SMILES
    mol = Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
    if mol is None:
        warnings.append(f"Invalid SMILES: {smiles}")
        return smiles_or_list, warnings

    # Process other_part if it exists and is a valid SMILES
    other_part_mol = None
    if other_part:
        try:
            other_part_mol = Chem.MolFromSmiles(other_part, sanitize=False)
        except:
            pass  # other_part may not be valid SMILES (e.g., a suffix/prefix)

    # Helper function to check and fix valence for a molecule
    def process_molecule(mol, is_other_part=False):
        nonlocal warnings
        corrected = False
        prefix = "other_part" if is_other_part else "SMILES"

        # Compute valence explicitly to avoid precondition violation
        mol.UpdatePropertyCache(strict=False)

        # Check valences
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            expected_valences = standard_valences.get(symbol, [valence])
            if valence not in expected_valences:
                warnings.append(f"Unusual valence in {prefix} for {symbol}: {valence} (expected {expected_valences})")

        # Fix nitrogen valence issues by adjusting hydrogens
        if any('N' in w for w in warnings if prefix in w):
            rw_mol = Chem.RWMol(mol)  # Editable molecule
            for atom in rw_mol.GetAtoms():
                if atom.GetSymbol() != 'N':
                    continue
                valence = atom.GetTotalValence()
                if valence < 3:
                    # Add hydrogens to reach valence 3
                    hydrogens_needed = 3 - valence
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + hydrogens_needed)
                    corrected = True
                elif valence > 3:
                    # Remove hydrogens if possible
                    hydrogens_to_remove = valence - 3
                    current_hydrogens = atom.GetNumExplicitHs()
                    if current_hydrogens >= hydrogens_to_remove:
                        atom.SetNumExplicitHs(current_hydrogens - hydrogens_to_remove)
                        corrected = True
                    else:
                        warnings.append(f"Cannot reduce N valence in {prefix} to 3 without removing non-H bonds")
            if corrected:
                mol = rw_mol.GetMol()

        # Sanitize molecule after corrections
        if corrected:
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
                return mol, True
            except Exception as e:
                warnings.append(f"Failed to sanitize {prefix} after correction: {str(e)}")
                return mol, False
        return mol, False

    # Process main molecule
    mol, mol_corrected = process_molecule(mol)

    # Convert main molecule back to SMILES
    corrected_smiles = Chem.MolToSmiles(mol) if mol_corrected else smiles

    # Process other_part if it's a valid molecule
    corrected_other_part = other_part
    if other_part_mol:
        other_part_mol, other_corrected = process_molecule(other_part_mol, is_other_part=True)
        corrected_other_part = Chem.MolToSmiles(other_part_mol) if other_corrected else other_part

    # Return based on input type
    if other_part:
        return [corrected_smiles, corrected_other_part], warnings
    return corrected_smiles, warnings

def molfpsim(original_smiles,test_smiles):#I2M use the coordinates, so 2D coformation should be always
    #only use longest for desalts, one molecule comparing
    test_smiles= select_longest_smiles(test_smiles)
    original_smiles= select_longest_smiles(original_smiles)
    test_smiles, warnings=check_and_fix_valence(test_smiles)

    original_smiles = remove_backslash_and_slash(original_smiles)#c/s 
    test_smiles = remove_backslash_and_slash(test_smiles)
    original_smiles = re.sub(r'\[(\d+)\*', '[*',original_smiles)#[1*]-->[*]
    test_smiles = re.sub(r'\[(\d+)\*', '[*',test_smiles)
    original_smiles = remove_SP(original_smiles)#additional complex space stero from coordinates, most not used
    test_smiles = remove_SP(test_smiles)
    
    rd_smi_ori, success1=rdkit_canonicalize_smiles(original_smiles)#R-->*
    if "S" in rd_smi_ori and success1:#NOTE H replace radical electron
        rd_smi_ori=NoRadical_Smi(rd_smi_ori)
    rd_smi, success2=rdkit_canonicalize_smiles(test_smiles)
    original_smiles,test_smiles=rd_smi_ori,rd_smi

    mol1 = Chem.MolFromSmiles(original_smiles)#TODO considering smiles with rdkit not recongized in real data
    mol2 = Chem.MolFromSmiles(test_smiles)#TODO considering smiles with rdkit not recongized in real data

    morganfps1 = AllChem.GetMorganFingerprint(mol1, useChirality=False)
    morganfps2 = AllChem.GetMorganFingerprint(mol2, useChirality=False)
    morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
    return morgan_tani, tanimoto




def comparing_smiles2(original_smiles,test_smiles):#I2M use the coordinates, so 2D coformation should be always
    original_smiles = remove_backslash_and_slash(original_smiles)#c/s 
    test_smiles = remove_backslash_and_slash(test_smiles)
    original_smiles = re.sub(r'\[(\d+)\*', '[*',original_smiles)#[1*]-->[*]
    test_smiles = re.sub(r'\[(\d+)\*', '[*',test_smiles)
    original_smiles = remove_SP(original_smiles)#additional complex space stero from coordinates, most not used
    test_smiles = remove_SP(test_smiles)
    
    rd_smi_ori, success1=rdkit_canonicalize_smiles(original_smiles)#R-->*
    if "S" in rd_smi_ori and success1:#NOTE H replace radical electron
        rd_smi_ori=NoRadical_Smi(rd_smi_ori)

    rd_smi, success2=rdkit_canonicalize_smiles(test_smiles)
    original_smiles,test_smiles=rd_smi_ori,rd_smi

    try:
        original_mol = Chem.MolFromSmiles(original_smiles)#considering whe nmmet abbrev
        test_mol = Chem.MolFromSmiles(test_smiles,sanitize=False)#as build mol may not sanitized for rdkit
        if success2 and success1:
            # if original_smiles!=test_smiles:
            #     print(f'smiles ori,pred after Chem.CanonSmiles:\n{original_smiles}\n{test_smiles}')
            RDarom_smi=Chem.MolToSmiles(original_mol)
            RDarom_smi_test=Chem.MolToSmiles(test_mol)
            if RDarom_smi==RDarom_smi_test:
                return True
            else:
                print(f'smiles ori,pred after Chem.CanonSmiles:\n{RDarom_smi}\n{RDarom_smi_test}\n')
  
        if original_mol:
            Chem.SanitizeMol(original_mol)
            keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
        else:
            keku_smi_ori=original_smiles
        
        if test_mol:
            Chem.SanitizeMol(test_mol)
            keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
        else:
            keku_smi=test_smiles
            
        if '*' not in keku_smi:
            keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
            keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
        else:
            keku_inch_ori=  1
            keku_inch_test=  2

        rd_smi=Chem.MolToSmiles(test_mol)#need improve the acc
        rd_smi_ori=Chem.MolToSmiles(original_mol)
    except Exception as e:#TODO fixme here
        print(f"comparing_smiles@@@ kekulize or SanitizeMol problems")# original_smiles,test_smiles\n{original_smiles}\n{test_smiles}")
        print(e,"!!!!!!!\n")
        keku_inch_ori=  1
        keku_inch_test=  2
        keku_smi=1
        keku_smi_ori=2
        #add molscribe rules here
        if not success1:#ori smiles still invaild even after * replaced
            rd_smi_ori = rd_smi
        # else:
        #     if canon_smiles1 == canon_smiles2:
        #         rd_smi_ori = rd_smi
            # else:
    if rd_smi_ori == rd_smi or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test :#as orinial smiles may use kekuleSmiles style
        return True
    else:return False

def smiles12_comparing(original_smiles,test_smiles):
    original_smiles = remove_backslash_and_slash(original_smiles)#c/s 
    test_smiles = remove_backslash_and_slash(test_smiles)
    original_smiles = re.sub(r'\[(\d+)\*', '[*',original_smiles)#[1*]-->[*]
    test_smiles = re.sub(r'\[(\d+)\*', '[*',test_smiles)
    original_smiles = remove_SP(original_smiles)#additional complex space stero from coordinates, most not used
    test_smiles = remove_SP(test_smiles)
    
    rd_smi_ori, success1=rdkit_canonicalize_smiles(original_smiles)
    rd_smi, success2=rdkit_canonicalize_smiles(test_smiles)
    original_smiles,test_smiles=rd_smi_ori,rd_smi
    try:
        original_mol = Chem.MolFromSmiles(original_smiles)#considering whe nmmet abbrev
        test_mol = Chem.MolFromSmiles(test_smiles,sanitize=False)#as build mol may not sanitized for rdkit
        if original_mol:
            Chem.SanitizeMol(original_mol)
            keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
        else:
            keku_smi_ori=original_smiles
        
        if test_mol:
            Chem.SanitizeMol(test_mol)
            keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
        else:
            keku_smi=test_smiles
            
        if '*' not in keku_smi:
            keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
            keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
        else:
            keku_inch_ori=  1
            keku_inch_test=  2

        rd_smi=Chem.MolToSmiles(test_mol)#need improve the acc
        rd_smi_ori=Chem.MolToSmiles(original_mol)
    except Exception as e:#TODO fixme here
        print(f"comparing_smiles@@@ kekulize or SanitizeMol problems")# original_smiles,test_smiles\n{original_smiles}\n{test_smiles}")
        print(e,"!!!!!!!\n")
        keku_inch_ori=  1
        keku_inch_test=  2
        keku_smi=1
        keku_smi_ori=2
        #add molscribe rules here
        if not success1:#ori smiles still invaild even after * replaced
            rd_smi_ori = rd_smi
        # else:
        #     if canon_smiles1 == canon_smiles2:
        #         rd_smi_ori = rd_smi
            # else:
    if rd_smi_ori == rd_smi or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test :#as orinial smiles may use kekuleSmiles style
        return True
    else:return False


def comparing_smiles(new_row,test_smiles):#I2M use the coordinates, so 2D coformation should be always
    original_smiles=new_row['SMILESori']
    original_smiles = remove_backslash_and_slash(original_smiles)#c/s 
    test_smiles = remove_backslash_and_slash(test_smiles)
    original_smiles = re.sub(r'\[(\d+)\*', '[*',original_smiles)#[1*]-->[*]
    test_smiles = re.sub(r'\[(\d+)\*', '[*',test_smiles)
    original_smiles = remove_SP(original_smiles)#additional complex space stero from coordinates, most not used
    test_smiles = remove_SP(test_smiles)
    
    rd_smi_ori, success1=rdkit_canonicalize_smiles(original_smiles)
    rd_smi, success2=rdkit_canonicalize_smiles(test_smiles)
    original_smiles,test_smiles=rd_smi_ori,rd_smi
    try:
        original_mol = Chem.MolFromSmiles(original_smiles)#considering whe nmmet abbrev
        test_mol = Chem.MolFromSmiles(test_smiles,sanitize=False)#as build mol may not sanitized for rdkit
        if original_mol:
            Chem.SanitizeMol(original_mol)
            keku_smi_ori=Chem.MolToSmiles(original_mol,kekuleSmiles=True)
        else:
            keku_smi_ori=original_smiles
        
        if test_mol:
            Chem.SanitizeMol(test_mol)
            keku_smi=Chem.MolToSmiles(test_mol,kekuleSmiles=True)
        else:
            keku_smi=test_smiles
            
        if '*' not in keku_smi:
            keku_inch_ori=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi_ori))
            keku_inch_test=  Chem.MolToInchi(Chem.MolFromSmiles(keku_smi))
        else:
            keku_inch_ori=  1
            keku_inch_test=  2

        rd_smi=Chem.MolToSmiles(test_mol)#need improve the acc
        rd_smi_ori=Chem.MolToSmiles(original_mol)
    except Exception as e:#TODO fixme here
        print(f"comparing_smiles@@@ kekulize or SanitizeMol problems")# original_smiles,test_smiles\n{original_smiles}\n{test_smiles}")
        print(new_row)
        print(e,"!!!!!!!\n")
        keku_inch_ori=  1
        keku_inch_test=  2
        keku_smi=1
        keku_smi_ori=2
        #add molscribe rules here
        if not success1:#ori smiles still invaild even after * replaced
            rd_smi_ori = rd_smi
        # else:
        #     if canon_smiles1 == canon_smiles2:
        #         rd_smi_ori = rd_smi
            # else:
    if rd_smi_ori == rd_smi or keku_smi_ori == keku_smi or keku_inch_ori==keku_inch_test :#as orinial smiles may use kekuleSmiles style
        return True
    else:return False







def bbox2center(bbox):
    x_center = (bbox[:, 0] + bbox[:, 2]) / 2
    y_center = (bbox[:, 1] + bbox[:, 3]) / 2
    # center_coords = torch.stack((x_center, y_center), dim=1)
    centers = np.stack((x_center, y_center), axis=1)
    return centers

import cv2
BONDDIRECT=['ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']


def reorder_bond_bbox(bond_bbox, single_atom_bond):
    # 分离普通索引和需要后置的索引
    normal_indices = []
    special_indices = []
    # 获取需要后置的 key
    keys_to_move = set(single_atom_bond.keys())
    # 分类所有索引
    for i in range(len(bond_bbox)):
        if i in keys_to_move:
            special_indices.append(i)
        else:
            normal_indices.append(i)
    # 新顺序：普通索引在前，特殊索引在后
    new_order = normal_indices + special_indices
    # 重排 bond_bbox
    reordered_bbox = [bond_bbox[i] for i in new_order]
    return reordered_bbox

def boxes_overlap(box1, box2):
    """
    检查两个边界框是否重叠
    box1, box2: [x1, y1, x2, y2]
    """
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])
def calculate_center(box):
    """
    计算边界框的中心点
    """
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
def merge_boxes(box1, box2):
    """
    合并两个边界框，返回新边界框 [x1, y1, x2, y2]
    """
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    ]


def get_merged_box(boxes):
    """Calculate the smallest box encompassing all given boxes."""
    x_mins = [box[0] for box in boxes]
    y_mins = [box[1] for box in boxes]
    x_maxs = [box[2] for box in boxes]
    y_maxs = [box[3] for box in boxes]
    return [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]

def box_area(box):
    """Calculate the area of a box."""
    return (box[2] - box[0]) * (box[3] - box[1])

def Newbox_(atom_bbox,bond_bbox, lab2idx):
    #add H atom box when on direction bond
    new_atoms=[]
    b_len=3
    single_odd_b2a=dict()
    for bi,bb in enumerate(bond_bbox):
        overlapped_atoms = []
        overlapped_abox=[]
        for ai,aa in enumerate(atom_bbox):
            overlap_flag=boxes_overlap(bb, aa)#TODO use tghe atom bond box overlap get bond atom mapping,then built mol
            if overlap_flag:
                # print(bb, aa,overlap_flag)
                overlapped_atoms.append(ai)
                overlapped_abox.append(aa)
        if len(overlapped_atoms) == 1:
            single_odd_b2a[bi]=overlapped_atoms
            # Compute the non-overlapping part of the bond box to place hydrogen
            non_overlapping_x,non_overlapping_y=boxes_overlap2(overlapped_abox[0], bb)
            new_atom_out={'bbox':    np.array([non_overlapping_x - b_len, 
                                    non_overlapping_y - b_len,
                                    non_overlapping_x + b_len, 
                                    non_overlapping_y + b_len]).reshape(-1,4),
                'bbox_centers': np.array([non_overlapping_x,non_overlapping_y]).reshape(-1,2),
                'scores':       np.array([1.0]),
                'pred_classes': np.array([lab2idx['H']])}
            new_atoms.append(new_atom_out)
    return new_atoms, single_odd_b2a


def has_boxes(data):
    #TO CHECK OCR detct used or not
    return isinstance(data, list) and len(data) > 0 and all(
        isinstance(item, list) and len(item) == 2 and 
        isinstance(item[0], list) and len(item[0]) == 4
        for item in data
    )

def AtomBox2bondBox(atom_box,bond_bbox):
    b_nei=[]
    overlap=True
    for bi,bb in enumerate(bond_bbox):
        overlap_flag=boxes_overlap(bb, atom_box)#TODO use tghe atom bond box overlap get bond atom mapping,then built mol
        if overlap_flag:
            b_nei.append(bi)
    if len(b_nei)==0:
        # delt_hei.append(hei)
        overlap=False
    return overlap, b_nei


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



# from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor

@torch.no_grad()
def evaluate_x(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, 
        data_loader, device,
        outcsv_filename=f'/home/jovyan/rt-detr/rt-detr/output/output_charge_CLEF.csv',
        visual_check=False,
        other2ppsocr=True,
        getacc=False,
        ):
    
    postprocessor2=RTDETRPostProcessor(num_classes=30, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)
    output_directory = os.path.dirname(outcsv_filename)
    prefix_f = os.path.basename(outcsv_filename).split('.')[0]
    if other2ppsocr:
        ocr = PaddleOCR(
        use_angle_cls=True,
        lang='latin',use_space_char=True,use_debug=False,
        use_gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)

        ocr2 = ocr2 = PaddleOCR(use_angle_cls=True,use_gpu =False,use_debug=False,
                    rec_algorithm='SVTR_LCNet',
                    #   rec_model_dir='/nfs_home/bowen/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
                    lang="en") 
        outcsv_filename=f"{output_directory}/{prefix_f}_withOCR.csv"


    if visual_check:
        output_directory = os.path.dirname(outcsv_filename)
        prefix_f = os.path.basename(outcsv_filename).split('.')[0]
        ima_checkdir=f"{output_directory}/{prefix_f}_Boxed"
        os.makedirs(ima_checkdir, exist_ok=True)

    if getacc:
        acc_summary=f"{outcsv_filename}.I2Msummary.txt"
        flogout = open(f'{acc_summary}' , 'w')
        failed=[]
        mydiff=[]
        simRD=0
        sim=0
        mysum=0

    model.eval()
    criterion.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Infering:'
    res_smiles = []
    idx_to_labels23={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                        9:'I',10:'P',11:'*',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
                        16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2',} 
    idx_to_labels30 = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                        9:'I',10:'P',11:'H',12:'Si',13:'NONE',14:'BEGINWEDGE',15:'BEGINDASH',
                        16:'=',17:'#',18:'-4',19:'-2',20:'-1',21:'1',22:'2',
                        23:'CF3',#NOTE rdkit get element not supporting group
                        24:'CN',
                        25:'Me',
                        26:'CO2Et',
                        27:'R',
                        28:'Ph',
                        29:'*',
                        }
    bond_labels = [13,14,15,16,17]

    if postprocessors.num_classes==23:
        # print(data["categories"])
        print(f'usage idx_to_labels23',idx_to_labels23)
        idx_to_labels = idx_to_labels23
    elif postprocessors.num_classes==30:
        # print(data["categories"])#NOTE 11 is H not * now
        print(f'usage idx_to_labels30',idx_to_labels30)
        idx_to_labels = idx_to_labels30
    else:
        print(f"error unkown ways@@@@@@@@@@@!!!!!!!!!!idx_to_labels::{len(idx_to_labels)}\n{idx_to_labels}")
    abrevie={"[23*]":'CF3',
                                "[24*]":'CN',
                                "[25*]":'Me',
                                "[26*]":'CO2Et',
                                "[27*]":'R',
                                "[28*]":'Ph',
                                "[29*]":'3~7UP',
        }
    # idx_to_labels = idx_to_labels23
    lab2idx={ v:k  for k,v in idx_to_labels.items() }

    smiles_data = pd.DataFrame({'file_name': [],
                                'SMILESori':[],
                                'SMILESpre':[],
                                'SMILESexp':[],
                                }
                                )
    output_dict = {}
    output_ori={}
    filtered_output_dict = {}
    box_thresh=0.1
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    #     samples = samples.to(device)
    #     # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #     outputs = model(samples)
    #     # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)#.to(device)    
    #     orig_target_sizes = targets["orig_size"].to(device)  
    #     results = postprocessors(outputs, orig_target_sizes)#RTDETRPostProcessor@@src/zoo/rtertr
    #     for i_, z in enumerate(zip(targets['image_id'], results)):
    #         ti, output=z
    #         output_dict[ti.item()] = [     
    #                                     output,
    #                                     targets['img_path'][i_], 
    #                                     targets['SMILES'][i_],
    #                                 ]

    #         output_ori[ti.item()] =[     
    #                     targets['img_path'][i_], 
    #                     targets['SMILES'][i_],
    #                                 ]
    # print(len(output_ori),len(output_dict))     
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # orig_target_sizes = targets["orig_size"].to(device)  
        for i_, ti in enumerate(targets['image_id']):
            output_dict[ti.item()] = [     
                                    targets['img_path'][i_], 
                                    targets['SMILES'][i_],
                                ]


    for key, value in output_dict.items():
        
        image_path = value[0]
        SMILESori = value[1]

        # selected_indices = value['scores'] > 0.5#may be >=0.5 cut off, as used the sigmoid?
        # selected_indices = value[0]['scores']  > box_thresh
        # true_count = selected_indices.sum().item()
        #testing here
        image_path='/cadd_data/samba_share/from_docker/data/work_space/ori/real/acs/ol020229e-Scheme-c3-10.png'

        tensor,w,h = image_to_tensor(image_path)
        tensor=tensor.unsqueeze(0).to(device)
        print(tensor.size())  # Output tensor shape (C x H x W)
        ori_size=torch.Tensor([w,h]).long().unsqueeze(0).to(device)
        outputs = model(tensor)
        result_ = postprocessor2(outputs, ori_size)
        # result_ = postprocessors(outputs, ori_size)
        score_=result_[0]['scores']
        boxe_=result_[0]['boxes']
        label_=result_[0]['labels']
        #---------------------------------################################
        selected_indices =score_ > box_thresh
        true_count = selected_indices.sum().item()
        output={
            'labels': label_[selected_indices].to("cpu").numpy(),
            'boxes': boxe_[selected_indices].to("cpu").numpy(),
            'scores': score_[selected_indices].to("cpu").numpy()
        }

        img_ori = Image.open(image_path).convert('RGB')
        w_ori, h_ori = img_ori.size  # 获取原始图像的尺寸
        print(w_ori, h_ori, "orignianl vs 1000,1000")

        print(f"selected_indices 中 True 的数量: {true_count}")
        print(f"before nms_per_class, :: box 的数量:{len(output['labels'])}")
        output = nms_per_class(output['labels'], output['boxes'], output['scores'], iou_thresh=0.5)
        print(f"after nms_per_class, :: box 的数量:{len(output['labels'])}")

        
        # filtered_output_dict={image_path: output}
        x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
        y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2
        # center_coords = torch.stack((x_center, y_center), dim=1)
        center_coords = np.stack((x_center, y_center), axis=1)
        # center_coords=np.stack((x_center, y_center)).reshape(-1,2)#NOTE not do this, mix element order shits
        #TODO split atom_charge \ bond drawing
        output = {'bbox':         output["boxes"],#.to("cpu").numpy(),
                    'bbox_centers': center_coords,#.to("cpu").numpy(),
                    'scores':       output["scores"],#.to("cpu").numpy(),
                    'pred_classes': output["labels"],#.to("cpu").numpy()
                    }
        ############################################################################################################################
        img_ori = Image.open(image_path).convert('RGB')
        w_ori, h_ori = img_ori.size  # 获取原始图像的尺寸
        print(w_ori, h_ori, "orignianl vs 1000,1000")
        # 计算缩放比例
        scale_x = 1000 / w_ori
        scale_y = 1000 / h_ori
        img_ori_1k = img_ori.resize((1000,1000))
        img = Image.open(image_path).convert('RGB')
        img = img.resize((1000,1000))
        # atom_bondBox_check=True

        print(f"from oupt socore > {box_thresh},get box {len(output['bbox'])} after nms_per_class ")
        # split into atom bond charge nms， then mergd , then box2 mol NOTE charege and bond confidence at least >10%
        charge_mask = np.array([True if ins  in charge_labels and  output['scores'][i]>0.1  else False  for i, ins in enumerate(output['pred_classes'])])
        charges_bbox=output['bbox'][charge_mask]
        charges_centers= output['bbox_centers'][charge_mask]
        charges_classes= output['pred_classes'][charge_mask]
        charges_scores= output['scores'][charge_mask]
        charges_bbox, charges_centers, charges_scores,charges_classes,figc =view_box_center2(charges_bbox, charges_centers, charges_scores,charges_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)
        #view_box_center2 help remove large box if boxscore small than 0.5
        # bonds_mask2 = np.array([True if ins  in bond_labels else False for ins in output['pred_classes']])
        # bonds_mask= output['scores'][bonds_mask2]>=0.1# TODO fix me, as training bond box overlap with bondbox,aussme bond socre make sense
        bonds_mask = np.array([True if ins  in bond_labels and output['scores'][i]>0.2 else False for i, ins in enumerate(output['pred_classes'])])
        bond_bbox=output['bbox'][bonds_mask]
        bond_centers= output['bbox_centers'][bonds_mask]
        bond_classes= output['pred_classes'][bonds_mask]
        bond_scores= output['scores'][bonds_mask]
        # bond_bbox2, bond_centers2, bond_scores2,bond_classes2,fig=view_box_center2(bond_bbox, bond_centers, bond_scores,bond_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)
        bond_bbox, bond_centers, bond_scores,bond_classes,fig =view_box_center2(bond_bbox, bond_centers, bond_scores,bond_classes, overlap_dist_thresh=5.0, max_centers_per_box=3)
        bond_bbox, bond_classes, bond_scores = nms(bond_bbox, bond_scores,bond_classes, iou_threshold=0.5)

        heavy_mask= np.array([True if ins not in bond_labels and ins not in charge_labels and ins != lab2idx['H'] else False for ins in output['pred_classes']])
        h_mask= np.array([True if ins not in bond_labels and ins not in charge_labels and ins == lab2idx['H'] else False for ins in output['pred_classes']])

        #TODO fix me if heavy or H all need this view_box_center2 filtering
        heavy_bbox = output['bbox'][heavy_mask]
        heavy_classes = output['pred_classes'][heavy_mask]
        heavy_centers= output['bbox_centers'][heavy_mask]
        heavy_scores= output['scores'][heavy_mask]
        heavy_bbox, heavy_centers, heavy_scores,heavy_classes,fighv =view_box_center2(heavy_bbox, heavy_centers, heavy_scores,heavy_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)

        #TODO del isolated C without bond box overlap
        delt_hei=[]
        for hei,hebox in enumerate(heavy_bbox):
            he_class=idx_to_labels[heavy_classes[hei]]
            b_nei=[]
            if he_class in ['C']:#TODO add other cases
                for bi,bb in enumerate(bond_bbox):
                    overlap_flag=boxes_overlap(bb, hebox)#TODO use tghe atom bond box overlap get bond atom mapping,then built mol
                    if overlap_flag:
                        b_nei.append(bi)
                if len(b_nei)==0:
                    delt_hei.append(hei)
        n = len(heavy_scores)  # 更新数量
        keep_boxes = np.ones(n, dtype=bool)  
        keep_boxes[delt_hei]=False
        heavy_bbox, heavy_centers, heavy_scores,heavy_classes=heavy_bbox[keep_boxes], heavy_centers[keep_boxes], heavy_scores[keep_boxes],heavy_classes[keep_boxes]

        h_bbox = output['bbox'][h_mask]
        h_centers= output['bbox_centers'][h_mask]
        h_classes= output['pred_classes'][h_mask]
        h_scores= output['scores'][h_mask]
        h_bbox, h_centers, h_scores,h_classes,figh =view_box_center2(h_bbox, h_centers, h_scores,h_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)

        #NOTE need keep the order heavy atom first then following with Hs
        # atoms_mask = np.array([True if ins not in bond_labels and ins not in charge_labels else False for ins in output['pred_classes']])
        # atom_bbox=output['bbox'][atoms_mask]
        # atom_classes=output['pred_classes'][atoms_mask]
        # 合并 bbox，保持重原子在前，氢原子在后
        atom_bbox = np.concatenate([heavy_bbox, h_bbox], axis=0)
        atom_classes = np.concatenate([heavy_classes, h_classes], axis=0)
        # atom_centers = np.concatenate([heavy_centers, h_centers], axis=0)
        atom_scores = np.concatenate([heavy_scores, h_scores], axis=0)
        #TODO nms checking
        # kept_bboxes, kept_classes, kept_scores=nms(atom_bbox, atom_scores, atom_classes, iou_threshold=0.5)
        # # kept_bboxes, kept_classes, kept_scores=nms_atomBox(atom_bbox, atom_scores, atom_classes, iou_threshold=0.5)
        # merged_bboxes, merged_classes, merged_scores = merge_low_iou_boxes(kept_bboxes, kept_classes, kept_scores, merge_threshold=0.5, score_threshold=0.7)
        # print(f'ater nms kept_box {len(kept_bboxes)}, followd merge_low_iou_boxes  kept_box:: {len(merged_bboxes)}')
        # atom_bbox, atom_classes, atom_scores=merged_bboxes, merged_classes, merged_scores
        atom_bbox, atom_scores, atom_classes = refine_boxes(atom_bbox, atom_scores, atom_classes,  bond_bbox)


        x_center = (atom_bbox[:, 0] + atom_bbox[:, 2]) / 2
        y_center = (atom_bbox[:, 1] + atom_bbox[:, 3]) / 2
        # center_coords = torch.stack((x_center, y_center), dim=1)
        center_coords = np.stack((x_center, y_center), axis=1)
        atom_centers=center_coords

        print(f"before NMS :: heavy box {len(heavy_bbox)} ---- H box {len(h_bbox)}---bond box{len(bond_bbox)}")
        print(f"after  NMS+view_box_center2 :: atom box {len(atom_bbox)} bond box {len(bond_bbox)}  charge box {len(charges_bbox)} ")
        # print(f"bond box with only single atom box overlap:: {single_odd_bi}")
        print(f"atom box afte NMS and merge_low_iou_boxes")
        print(f"get box {len(output['bbox'])} with NMS")
        print(f"atom score >0.1 bond score >0.2, then folllowed with NMS")
        print(f"bond_bbox nums::",bond_bbox.shape,len(bond_bbox))
        print(f" OCR will start involved ")#
        #check if ODD single-bonds with only one atom exisits, try add the atoms box for this bond
        new_atoms, single_odd_b2a= Newbox_(atom_bbox,bond_bbox, lab2idx )
        print(f"new_atoms number {len(new_atoms)}\n{new_atoms}")
        if len(new_atoms)>0:
            for boxout in new_atoms:
                for k,arr in boxout.items():
                    value_or_row=output[k]
                    if arr.ndim == 1:
                        output[k]=np.append(value_or_row, arr)
                    elif arr.ndim >= 2:
                        output[k] = np.concatenate([value_or_row, arr], axis=0)
                    else:
                        print('errprs, unkown conditions !!!@')
        #NOTE try to use OCR to help postprocess box adding and del
        # 加载图像 OCR
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 预处理图像突出下标
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        # print(_, thresh)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        # cv2.imwrite("preprocessed.jpg", dilated)#NOTE comment if need checking
        # result = ocr.ocr("preprocessed.jpg", cls=True)
        #  ocr.ocr(image_npocr, cls=True, det=False)
        result = ocr.ocr(dilated, cls=True)  # 直接传递 NumPy 数组
        # 解析结果
        text_boxes = []
        text_contents = []
        confidences = []
        for line in result:
            print(line)
            if line:
                for box_info in line:
                    box = box_info[0]
                    x_coords = [point[0] for point in box]
                    y_coords = [point[1] for point in box]
                    text_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    text = box_info[1][0]
                    text_boxes.append(text_box)
                    text_contents.append(text)
                    confidences.append(box_info[1][1])
        print("Detected text boxes:", text_boxes)
        print("Detected text contents:", text_contents)
        print("Confidences:", confidences)        
        #after whole img OCRed
        # Initialize dictionaries and lists
        ai2text = {}
        ai2relplace = {}
        ai2rdkitlab_unknown = {}
        non_overlapping_texts = []
        # Build initial KDTree
        tree = cKDTree(atom_centers)
        # Collect indices to delete after the loop to keep tree valid during processing
        indices_to_delete = set()
        # Process each OCR text box
        for ti, text_box in enumerate(text_boxes):
            text_center = calculate_center(text_box)
            ocr_text = text_contents[ti]

            # Normalize OCR text
            if ocr_text in ['OH', 'HO']:
                ocr_text = 'O'
            elif ocr_text in ['SH', 'HS']:
                ocr_text = 'S'
            elif ocr_text in ['NH', 'HN']:
                ocr_text = 'N'
            elif ocr_text in ['CH', 'HC']:
                ocr_text = 'C'
            elif ocr_text == '0':
                ocr_text = 'O'
            elif ocr_text == 'L':
                ocr_text = 'Li'
            elif ocr_text[-1]=='-':
                if ocr_text[:-1] in  ABBREVIATIONS:
                    ocr_text=ocr_text[:-1]
            
            # Find all overlapping atom boxes
            overlapping_indices = []
            for idx in range(len(atom_bbox)):
                if idx not in indices_to_delete and boxes_overlap(atom_bbox[idx], text_box):
                    overlapping_indices.append(idx)

            if overlapping_indices:
                # If there are overlapping atom boxes, merge them
                if len(overlapping_indices) > 1:
                    # Get the smallest box encompassing all overlapping atom boxes
                    overlapping_boxes = [atom_bbox[idx] for idx in overlapping_indices]
                    merged_box = get_merged_box(overlapping_boxes)
                    overlapping_indices_atomboxclass=[idx_to_labels[atom_classes[i]] for i in overlapping_indices]
                    print(f"Merging {len(overlapping_indices)} atom boxes overlapping with OCR text: {ocr_text}")
                    print(f" {overlapping_indices} boxes type{overlapping_indices_atomboxclass}  merged as OCR text: {ocr_text}")
                    merged_area = box_area(merged_box)
                    text_area = box_area(text_box)
                    final_box = merged_box if merged_area >= text_area else text_box
                else:
                    # If only one overlap, use the text box directly
                    final_box = text_box
                # Use the OCR text box as the merged box
                primary_idx = overlapping_indices[0]
                # atom_bbox[primary_idx] = text_box
                
                # Update the primary atom box
                atom_bbox[primary_idx] = final_box
                # Update class and dictionaries based on OCR text
                if ocr_text in ABBREVIATIONS:
                    ai2relplace[primary_idx] = ocr_text
                    atom_classes[primary_idx] = 0
                    if ocr_text in lab2idx:
                        atom_classes[primary_idx] = lab2idx[ocr_text]
                elif ocr_text in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:
                    atom_classes[primary_idx] = lab2idx[ocr_text]
                elif ocr_text in RGROUP_SYMBOLS or (ocr_text[0] == 'R' and ocr_text[1:].isdigit()):
                    atom_classes[primary_idx] = 0
                else:
                    ai2rdkitlab_unknown[primary_idx] = ocr_text
                    atom_classes[primary_idx] = 0
                
                ai2text[primary_idx] = ocr_text

                # Mark redundant indices for deletion
                indices_to_delete.update(overlapping_indices[1:])

            else:
                # No overlap: record the text box and nearest atom index
                distance, nearest_idx = tree.query(text_center)
                if nearest_idx not in indices_to_delete:  # Only record if nearest_idx is still valid
                    print(f"No overlap for OCR text '{ocr_text}', nearest atom box index: {nearest_idx}")
                    non_overlapping_texts.append({
                        'text': ocr_text,
                        'text_box': text_box,
                        'nearest_atom_idx': nearest_idx,
                        'distance': distance
                    })

        #set up atom_ocr match atom_class
        atom_ocr=[]
        for i,ai in enumerate(atom_classes):
            if i in ai2text:
                atom_ocr.append(ai2text[i])
            # elif i in ai2rdkitlab_unknown:
            #     atom_ocr.append(ai2rdkitlab_unknown[i])
            else:
                atom_ocr.append(idx_to_labels[ai])
        print(f"atom class + ocr presented as symbols::\n{atom_ocr}")
        atom_ocr=np.array(atom_ocr)
        # Perform deletions after the loop
        if indices_to_delete:
            indices_to_keep = np.setdiff1d(np.arange(len(atom_bbox)), list(indices_to_delete))
            atom_bbox = atom_bbox[indices_to_keep]
            atom_classes = atom_classes[indices_to_keep]
            atom_centers = atom_centers[indices_to_keep]
            atom_scores = atom_scores[indices_to_keep]
            atom_ocr= atom_ocr[indices_to_keep]

            # Adjust dictionary indices
            for d in [ai2text, ai2relplace, ai2rdkitlab_unknown]:
                d_new = {}
                for old_idx, value in d.items():
                    new_idx = np.where(indices_to_keep == old_idx)[0][0] if old_idx in indices_to_keep else None
                    if new_idx is not None:
                        d_new[new_idx] = value
                d.clear()
                d.update(d_new)

            # Adjust nearest_atom_idx in non_overlapping_texts
            for entry in non_overlapping_texts:
                old_idx = entry['nearest_atom_idx']
                if old_idx in indices_to_keep:
                    entry['nearest_atom_idx'] = np.where(indices_to_keep == old_idx)[0][0]
                else:
                    entry['nearest_atom_idx'] = -1  # Mark as invalid if the nearest atom was deleted

        # Rebuild KDTree if needed for further use
        tree = cKDTree(atom_centers)

        # Final output
        print("Whole img with OCR :: ai2relplace, ai2rdkitlab_unknown:", [ai2relplace, ai2rdkitlab_unknown])
        print(f"Adjusted ai ocr_text: {ai2text}")
        print(f"Atom box num: {len(atom_bbox)}:: {[idx_to_labels[i] for i in atom_classes]}")
        print("Non-overlapping OCR text boxes:", non_overlapping_texts)

        #for all  heavy atom labels, consider N3 pred as N, or other cases, I2M not good as paddle on ABC 
        atomcorp_img = Image.open(image_path).convert('RGB')
        atomcorp_img1k=atomcorp_img.resize([1000,1000])
        text_contents_star=[]
        text_confidences_star=[]
        text_boxes_star=[]
        boxid2del=dict()
        ocr_discrepancies = {}  # New dictionary to record OCR vs. AI mismatches
        print(f"has atom_bbox number {len(atom_bbox)}")
        for i,box in enumerate(atom_bbox):#split ocr image
            # if i in ai2text: continue #may be need comment this, if splited OCR acc better!!
            abox =box* [scale_x, scale_y, scale_x, scale_y]
            cropped_img=atomcorp_img1k.crop(abox)#if use the small ori image will not get infos
            image_npocr = np.array(cropped_img)
            result_ocr= ocr2.ocr(image_npocr, det=False)#,cls=True,use_debug=False, det=False)#det fale not box but get rcongized more 
            # result_ocr= ocr.ocr(image_npocr, cls=True, det=False)#,cls=True, det=False)#det fale not box but get rcongized more 
            if result_ocr:
                for line in result_ocr:
                    # print(f"Atom box--- {i}, OCR result---: {line}")
                    if line:
                        box_flag=has_boxes(line)
                        for box_info in line:
                            # print(len(box_info))
                            if not box_flag:
                                text=box_info[0]
                                #[^a-zA-Z0-9\*\-\+] 表示匹配除了字母、数字、*、- 和 + 之外的所有字符。
                                text=re.sub(r'[^a-zA-Z0-9,\*\-\+]', '', text)#remove special chars
                                score_=box_info[1]
                                text_contents_star.append(text)
                                text_confidences_star.append(score_)
                            else:#when paddleOCRuse detection model get text box info
                                box = box_info[0]
                                x_coords = [point[0] for point in box]
                                y_coords = [point[1] for point in box]
                                text_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                text = box_info[1][0]
                                text=re.sub(r'[^a-zA-Z0-9,\*\-\+]', '', text)#remove special chars
                                text_boxes_star.append(text_box)
                                text_contents_star.append(text)
                                score_=box_info[1][1]
                                text_confidences_star.append(score_)
                            if i in ai2text:#ocr 全img vs  split img 
                                # print(f'from whole img ocr atom box {i}----from whole img::{ai2text[i]}')
                                if  ai2text[i] != text:
                                    text=ai2text[i] if len(ai2text[i])>=len(text) else text
                            print(f"Atom box {i}@@ OCR text: {text}, score: {score_}, AI class: {idx_to_labels[atom_classes[i]]}, AI score: {atom_scores[i]}")
                            # Normalize OCR text
                            if text in ['OH', 'HO']:
                                text = 'O'
                            elif text in ['SH', 'HS']:
                                text = 'S'
                            elif text in ['NH', 'HN']:
                                text = 'N'
                            elif text in ['CH', 'HC']:
                                text = 'C'
                            elif text == '0':
                                text = 'O'
                            elif text == 'L':
                                text = 'Li'
                            elif '-' in text:
                                if text[:-1] in  ABBREVIATIONS:
                                    text=text[:-1]

                            # Check if OCR text is a single character and not a valid element
                            is_single_char = len(text) == 1
                            ai_pred = idx_to_labels[atom_classes[i]]
                            #TOD add more simpfiled 
                            if text=='0':
                                atom_classes[i]=lab2idx['O']
                            elif text in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:
                                atom_classes[i]=lab2idx[text]#need update to keep H following Heavy
                            # elif  # ocr recongnized on lable C as other things chars 
                            elif is_single_char and text not in ELEMENTS and ai_pred == 'C':
                                # Do not replace AI prediction, just record discrepancy
                                ocr_discrepancies[i] = {
                                    'ocr_text': text,
                                    'ocr_score': score_,
                                    'ai_class': ai_pred,
                                    'ai_score': atom_scores[i]
                                }
                            else:
                                overlap, b_nei=AtomBox2bondBox(atom_bbox[i],bond_bbox)
                                if not overlap:
                                    if text not in ELEMENTS and text not in ABBREVIATIONS:
                                        # print(f"new cases::{text} for atombox {i}  {atom_bbox[i]}check how to fix it  !!!")
                                        # print(f'OCR text:: {text} score ::{box_info}||atom clss::{idx_to_labels[atom_classes[i]]} {atom_scores[i]}')
                                        if text != idx_to_labels[atom_classes[i]]:
                                            boxid2del[i]= [text,idx_to_labels[atom_classes[i]]]#will delt this atom box infos
                                else:
                                    if text != idx_to_labels[atom_classes[i]]:
                                        if atom_scores[i]<=score_:
                                            if text in RGROUP_SYMBOLS or text in ABBREVIATIONS:
                                                ai2relplace[i]=text
                                                atom_classes[i]=0
                                                if text in lab2idx and  lab2idx[text] in list(range(23,29)):atom_classes[i]=lab2idx[text]
                                            elif text in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:
                                                atom_classes[i]=lab2idx[text]
                                            else:
                                                ai2relplace[i]=text
                                                atom_classes[i]=0
                                                
        # 按照 value 的第一个元素（假设是字符串）的长度进行排序，长度大的排前
        boxid2del = dict(sorted(boxid2del.items(), key=lambda item: item[0], reverse=True))
        print(f"considering del box",boxid2del)                                    
        print("after split img  OCR:: ai2relplace,ai2rdkitlab_unknown",[ai2relplace,ai2rdkitlab_unknown])
        print(f"considering delet atomb box :{boxid2del}")
        syms=[]
        for i in range(len(atom_classes)):
            if  i in ai2relplace: syms.append(ai2relplace[i])
            elif i in ai2rdkitlab_unknown:syms.append(ai2rdkitlab_unknown[i])
            else:
                syms.append(idx_to_labels[atom_classes[i]])
        print(f"atombox {atom_classes}:: number {len(atom_classes)}\n",[idx_to_labels[i] for i in atom_classes])
        print(f" {syms}")
        #chedck isolated box, if need add bond box between the isolated box or not
        isolated_ais = []
        # 第一步：构建 bond 到 atom 的映射，并计算 distance_threshold
        bond_distances = []
        singleAtomBond=dict()
        for bi, bb in enumerate(bond_bbox):
            overlapped_atoms = []
            overlapped_abox = []
            for ai, aa in enumerate(atom_bbox):
                overlap_flag = boxes_overlap(bb, aa)
                if overlap_flag:
                    overlapped_atoms.append(ai)
                    overlapped_abox.append(aa)
                    # if bi not in b2a.keys():
                    #     b2a[bi] = [ai]
                    # else:
                    #     b2a[bi].append(ai)
            if len(overlapped_atoms) == 2:
                center1 = calculate_center(atom_bbox[overlapped_atoms[0]])
                center2 = calculate_center(atom_bbox[overlapped_atoms[1]])
                distance = np.linalg.norm(center1 - center2)
                bond_distances.append(distance)
                # print(f"Bond {bi} connects atoms {overlapped_atoms}, distance: {distance:.2f}")
            elif len(overlapped_atoms) == 1:
                print(f"single bond - atom still exists for bond {bi}, need porcess this !!")
                if bi not in singleAtomBond:
                    singleAtomBond[bi]=overlapped_atoms#considering use the add H box for solve TODO 

        # 动态计算 distance_threshold
        distance_threshold = max(bond_distances) if bond_distances else 100.0  # 默认值 10 如果无 bond
        distance_threshold_min = min(bond_distances) if bond_distances else 100.0  # 默认值 10 如果无 bond
        print(f"Calculated distance_threshold center based: {distance_threshold:.2f}")

        # 第二步：构建 atom 到 bond 的映射，并检测孤立原子
        a2b=dict()
        for ai, aa in enumerate(atom_bbox):
            b_nei = []
            for bi, bb in enumerate(bond_bbox):
                overlap_flag = boxes_overlap(bb, aa)
                if overlap_flag:
                    b_nei.append(bi)
            a2b[ai] = b_nei
            if a2b[ai] ==[]:
                if ai not in isolated_ais:
                    isolated_ais.append(ai)

        isolated_ais=sorted(isolated_ais,reverse=True)#avoid delte atom with index errors
        print(f"isolated_ais atom box {isolated_ais}\n ", [idx_to_labels[i] for i in atom_classes[isolated_ais]])

        # 第三步：处理孤立原子，尝试合并或删除
        updated_atom_bbox = atom_bbox.copy()
        updated_atom_classes = atom_classes.copy()
        updated_atom_scores = atom_scores.copy()
        print(f"atom bbox num {len(atom_bbox)}")#ttt
        new_bond_bbox=[]
        deleted_ais=[]
        del4boxid2del=set()
        for isolated_ai in isolated_ais:
            isolated_box = atom_bbox[isolated_ai]
            isolated_center = calculate_center(isolated_box)
            nearest_distance = float('inf')
            nearest_ai = -1
            # 找到最近的非孤立原子
            for ai, aa in enumerate(atom_bbox):
                if ai not in isolated_ais and ai != isolated_ai:
                    center = calculate_center(aa)
                    distance = np.linalg.norm(isolated_center - center)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_ai = ai
            # 合并或删除逻辑
            if nearest_ai != -1:
                if nearest_distance<=distance_threshold_min or (nearest_distance <=distance_threshold and nearest_distance>=distance_threshold_min):#this the centers dist not bond length
                    nearest_box = atom_bbox[nearest_ai]
                    nearest_class = atom_classes[nearest_ai]
                    nearest_center = calculate_center(nearest_box)
                    if isolated_ai in boxid2del:
                        textocr2del=boxid2del[isolated_ai][0]
                    else:
                        textocr2del=None
                    #NOTE based ont the class and ovelap bond box to adjust
                    overlap1,bondnei=AtomBox2bondBox(nearest_box,bond_bbox)
                    if len(bondnei)==1:#could be add two other bond, add bond box
                        # if textocr2del in [',', '+', '-'] or not any(c.isupper() for c in textocr2del):
                        if textocr2del is not None and  not any(c.isupper() for c in textocr2del):
                            # del4boxid2del.add(isolated_ai)
                            deleted_ais.append(isolated_ai)
                            pass
                        else:
                            new_bc = (isolated_center + nearest_center)*0.5
                            new_bondbox=np.array([new_bc[0] - nearest_distance*0.5,
                                                new_bc[1] - nearest_distance*0.5,
                                                new_bc[0] + nearest_distance*0.5,
                                                new_bc[1] + nearest_distance*0.5]
                                        )
                            new_bond_bbox.append(new_bondbox.reshape(-1,4))
                            print(f'add a new bond box new_bc for two atom boxes {isolated_ai} ---- {nearest_ai}::\n {idx_to_labels[atom_classes[isolated_ai]]}   --- {idx_to_labels[atom_classes[nearest_ai]]}')
                    else:#TODO fix me when get the case with >=2 bonds need add bond also
                        try:
                            new_box = merge_boxes(isolated_box, nearest_box)
                            updated_atom_bbox[nearest_ai] = new_box
                            chosed_score_ = max(atom_scores[isolated_ai], atom_scores[nearest_ai])
                            updated_atom_scores[nearest_ai] = chosed_score_
                        except Exception as e:
                            print(f"file_name@: {image_path}\n SMILES in csv:\n{SMILESori}")
                            print(e)
                            print('nearest_ai  ', nearest_ai)
                            check2=True
                            if check2:
                                padding=5
                                # box_thresh=0.3
                                atombox_img=draw_objs(copy.deepcopy(img),
                                                    atom_bbox* [scale_x, scale_y, scale_x, scale_y],
                                                    atom_classes, atom_scores ,
                                                    category_index=idx_to_labels,
                                                    box_thresh=box_thresh,
                                                    line_thickness=3,
                                                    font='arial.ttf',
                                                    font_size=10)
                                bonbox_img=draw_objs(copy.deepcopy(img),
                                                    bond_bbox* [scale_x, scale_y, scale_x, scale_y],
                                                    bond_classes, bond_scores ,
                                                    category_index=idx_to_labels,
                                                    box_thresh=0.01,
                                                    line_thickness=3,
                                                    font='arial.ttf',
                                                    font_size=10)
                                # Get sizes of the individual images
                                atom_width, atom_height = atombox_img.size
                                bon_width, bon_height = bonbox_img.size
                                combined_width = atom_width + bon_width + padding * 3
                                combined_height = max(atom_height, bon_height) + padding * 2
                                combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))  # White background
                                # Paste the images onto the new canvas
                                combined_img.paste(atombox_img, (padding, padding))  # Top-left
                                combined_img.paste(bonbox_img, (atom_width + padding * 2, padding))
                                print(f"atom box afte NMS and merge_low_iou_boxes")
                            combined_img.save(f"tttttttttttttttttttttttBoxed.png"
                                              )
                            raise Exception("@debug this!!\n")
                        
                        if chosed_score_>=0.5:
                            if chosed_score_==atom_scores[isolated_ai]:
                                updated_atom_classes[nearest_ai] = 0 # mrege replaced with *
                            # else:
                            #     updated_atom_classes[nearest_ai] = atom_classes[nearest_ai]  # 保留较高 score 的类别
                        updated_atom_bbox = np.delete(updated_atom_bbox, isolated_ai, axis=0)#after mreged need del it
                        # updated_atom_bbox = np.delete(updated_atom_bbox, isolated_ai, axis=0)
                        updated_atom_classes = np.delete(updated_atom_classes, isolated_ai)
                        updated_atom_scores = np.delete(updated_atom_scores, isolated_ai)
                        print(f"Merged atom {isolated_ai} into {nearest_ai}, new box: {new_box}")
                        isolated_ais.remove(isolated_ai)
                        deleted_ais.append(isolated_ai)
                # elif nearest_distance<=distance_threshold_min:#very close,mrege with nearest one
                elif atom_scores[isolated_ai] < 0.5:
                    # 删除低分孤立原子
                    updated_atom_bbox = np.delete(updated_atom_bbox, isolated_ai, axis=0)
                    updated_atom_classes = np.delete(updated_atom_classes, isolated_ai)
                    updated_atom_scores = np.delete(updated_atom_scores, isolated_ai)
                    print(f"DELET isolated atom {isolated_ai} with score {atom_scores[isolated_ai]}")
                    deleted_ais.append(isolated_ai)
                    # 更新索引，因为数组维度变化
                    isolated_ais = [i if i < isolated_ai else i - 1 for i in isolated_ais if i != isolated_ai]
                else:
                    print(f"KEEP isolated atom {isolated_ai} with score {atom_scores[isolated_ai]} >= 0.5")
            

            else:
                if atom_scores[isolated_ai] < 0.5:
                    updated_atom_bbox = np.delete(updated_atom_bbox, isolated_ai, axis=0)
                    updated_atom_classes = np.delete(updated_atom_classes, isolated_ai)
                    updated_atom_scores = np.delete(updated_atom_scores, isolated_ai)
                    print(f"DELET isolated atom {isolated_ai} with score {atom_scores[isolated_ai]}")
                    deleted_ais.append(isolated_ai)
                    isolated_ais = [i if i < isolated_ai else i - 1 for i in isolated_ais if i != isolated_ai]
                else:
                    print(f"KEEP isolated atom {isolated_ai} with score {atom_scores[isolated_ai]} >= 0.5")

        if len(new_bond_bbox)>0:
            for i,bond_box in enumerate(new_bond_bbox):
                bond_bbox= np.concatenate([bond_bbox,bond_box],axis=0)
                bond_scores= np.concatenate((bond_scores,np.array([0.9])),axis=0)
                bond_classes= np.concatenate([bond_classes,np.array([13])],axis=0)
            #reset bond center
            x_center = (bond_bbox[:, 0] + bond_bbox[:, 2]) / 2
            y_center = (bond_bbox[:, 1] + bond_bbox[:, 3]) / 2
            # center_coords = torch.stack((x_center, y_center), dim=1)
            center_coords = np.stack((x_center, y_center), axis=1)
            bond_centers=center_coords         

        #del the additional atom box that not connected by bond box also mismatch other rules
        if len(deleted_ais) > 0:  # 如果有需要删除的索引
            print(f"will delete atom box with idx :: {deleted_ais}")
            # 使用 np.delete 一次性删除所有指定的行
            atom_classes = np.delete(atom_classes, deleted_ais, axis=0)
            atom_scores = np.delete(atom_scores, deleted_ais, axis=0)
            atom_bbox = np.delete(atom_bbox, deleted_ais, axis=0)
            atom_ocr = np.delete(atom_ocr, deleted_ais, axis=0)

        # eles=[idx_to_labels[i] for i in atom_classes]
        # print(eles,len(eles))        
        cur_atomSymbols=[idx_to_labels[i] for i in atom_classes]
        ocr_wholeImg=[]
        for i in atom_classes:
            if i in ai2relplace:
                ocr_wholeImg.append(ai2relplace[i])
            elif  i in ai2rdkitlab_unknown:
                ocr_wholeImg.append(ai2rdkitlab_unknown[i])
            else:
                ocr_wholeImg.append(idx_to_labels[i])
        print("ai2relplace,ai2rdkitlab_unknown",ai2relplace,ai2rdkitlab_unknown)
        print("cur_atomSymbols:",cur_atomSymbols)
        print(" atomSymbolsOCR:",ocr_wholeImg)
        
        # 找到 'H' 的索引, H after Heavy
        h_indices = np.where(atom_classes == lab2idx['H'])[0]
        non_h_indices = np.where(atom_classes != lab2idx['H'])[0]
        # print(h_indices,non_h_indices)
        # 重新排序
        new_order = np.concatenate((non_h_indices, h_indices)).astype(np.int64)
        # newid2old_Hafter={ i:j for i,j in enumerate(new_order)}
        # old2newid_Hafter={ j:i for i,j in enumerate(new_order)}
        atom_classes = atom_classes[new_order]
        atom_bbox = atom_bbox[new_order]
        atom_scores = atom_scores[new_order]
        x_center = (atom_bbox[:, 0] + atom_bbox[:, 2]) / 2
        y_center = (atom_bbox[:, 1] + atom_bbox[:, 3]) / 2
        # center_coords = torch.stack((x_center, y_center), dim=1)
        center_coords = np.stack((x_center, y_center), axis=1)
        atom_centers=center_coords#TODO 记得把 abbve idx label same reoder or mapping then bond
        #bond box reoder like atom box, let the singleAtomBond later
        bond_bbox = reorder_bond_bbox(bond_bbox, singleAtomBond)
        bond_classes = reorder_bond_bbox(bond_classes, singleAtomBond)
        bond_scores = reorder_bond_bbox(bond_scores, singleAtomBond)
        bond_centers = reorder_bond_bbox(bond_centers, singleAtomBond)

        # 第二步：构建 atom 到 bond 的映射，并检测孤立原子
        a2b=dict()
        for ai, aa in enumerate(atom_bbox):
            b_nei = []
            for bi, bb in enumerate(bond_bbox):
                overlap_flag = boxes_overlap(bb, aa)
                if overlap_flag:
                    b_nei.append(bi)
            a2b[ai] = b_nei
            if a2b[ai] ==[]:
                if ai not in isolated_ais:
                    isolated_ais.append(ai)

        b2a=dict()
        for bi,bb in enumerate(bond_bbox):
            overlapped_atoms = []
            overlapped_abox=[]
            for ai,aa in enumerate(atom_bbox):
                overlap_flag=boxes_overlap(bb, aa)#TODO use tghe atom bond box overlap get bond atom mapping,then built mol
                if overlap_flag:
                    # print(bb, aa,overlap_flag)
                    overlapped_atoms.append(ai)
                    overlapped_abox.append(aa)
                    if bi not in b2a.keys():
                        b2a[bi]=[ai]
                    else:
                        # vais=b2a[bi]
                        b2a[bi].append(ai)
            if len(overlapped_atoms) == 1:
                print(f"single bond -atom still exists  {overlapped_atoms}")

        #c2a a2c
        #charge atom idx maping
        if len(charges_classes) > 0:
            # print(charges_bbox,charges_classes,len(charges_classes))
            kdt = cKDTree(atom_centers)
            atid_list=list(range(len(atom_centers)))
            used_charge_indices=set()
            c2a=dict()
            for i, (x,y) in enumerate(charges_centers):
                overlapped_abox=[]
                cc=charges_bbox[i]
                for ai, aa in  enumerate(atom_bbox):
                    overlap_flag=boxes_overlap(cc, aa)
                    ac_iou=calculate_iou(cc, aa)
                    charge_=charges_classes[i]
                    charge_score=charges_scores[i]
                    if overlap_flag:
                        if i in c2a:
                            c2a[i].append(ai) 
                        else:
                            c2a[i]=[ai] 
                        if ai not in atid_list:
                            print(f"Warning: ai {ai} is out of range for atom_list.")
                            continue  # 跳过当前循环迭代
            # idx_to_labels[charges_classes[0]]
            a2c=dict()
            for ci,v in c2a.items():
                charge_=idx_to_labels[charges_classes[ci]]
                if len(v)==1:
                    a2c[v[0]]=ci
                else:
                    for ai in v:
                        ats=idx_to_labels[atom_classes[ai]]
                        if ats=='other':
                            ats='*'
                        if ats in ['F','Cl','I','Br','O'] and int(charge_)<0:
                            a2c[ai]=ci
                        elif ats in ['N','H','P'] and int(charge_)>0:
                            a2c[ai]=ci
                        else:
                            print(f'unusuaal case charge {charge_} with atom {ats}!!')

        print(f"all a2b b2a a2c c2a done, start mol built")
        #finsh the update of box back to the output for retraining used 
        output={
        'bbox':   np.concatenate([atom_bbox, bond_bbox,charges_bbox], axis=0),
        'bbox_centers': np.concatenate([atom_centers, bond_centers,charges_centers],axis=0),
        'scores':       np.concatenate([atom_scores, bond_scores, charges_scores],axis=0),
        'pred_classes': np.concatenate([atom_classes, bond_classes, charges_classes],axis=0),
        'image_path': image_path
        }
        # boxinfo
        boxinfor={
        'bbox':   output['bbox'],
        'scores': output['scores'],#TODO use same vocabl ?
        'pred_classes': output['pred_classes'],#[ lab2idx[x] for x in output['pred_classes']],#changet it back to character
        'image_path': image_path
        }
        #split agin for buit mol
        charge_mask = np.array([True if ins  in charge_labels else False for ins in output['pred_classes']])
        charges_bbox=output['bbox'][charge_mask]
        charges_centers=bbox2center(charges_bbox)
        # charges_centers= output['bbox_centers'][charge_mask]
        charges_classes= output['pred_classes'][charge_mask]
        charges_scores= output['scores'][charge_mask]
        charges_bbox, charges_centers, charges_scores,charges_classes,figc =view_box_center2(charges_bbox, charges_centers, charges_scores,charges_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)
        #view_box_center2 help remove large box if boxscore small than 0.5
        # bonds_mask2 = np.array([True if ins  in bond_labels else False for ins in output['pred_classes']])
        # bonds_mask= output['scores'][bonds_mask2]>=0.1# TODO fix me, as training bond box overlap with bondbox,aussme bond socre make sense
        bonds_mask = np.array([True if ins  in bond_labels and output['scores'][i]>0.2 else False for i, ins in enumerate(output['pred_classes'])])
        bond_bbox=output['bbox'][bonds_mask]
        bond_centers=bbox2center(bond_bbox)
        # bond_centers= output['bbox_centers'][bonds_mask]
        bond_classes= output['pred_classes'][bonds_mask]
        bond_scores= output['scores'][bonds_mask]
        print(f"before view_box_center2 bond nums {len(bond_scores)}")
        # bond_bbox2, bond_centers2, bond_scores2,bond_classes2,fig=view_box_center2(bond_bbox, bond_centers, bond_scores,bond_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)
        bond_bbox, bond_centers, bond_scores,bond_classes,fig =view_box_center2(bond_bbox, bond_centers, bond_scores,bond_classes, overlap_dist_thresh=5.0, max_centers_per_box=3)
        print(f"after view_box_center2 bond nums {len(bond_scores)}")

        heavy_mask= np.array([True if ins not in bond_labels and ins not in charge_labels and ins != lab2idx['H'] else False for ins in output['pred_classes']])
        h_mask= np.array([True if ins not in bond_labels and ins not in charge_labels and ins == lab2idx['H'] else False for ins in output['pred_classes']])

        #TODO fix me if heavy or H all need this view_box_center2 filtering
        heavy_bbox = output['bbox'][heavy_mask]
        # heavy_classes = output['pred_classes'][heavy_mask]
        heavy_centers=bbox2center(heavy_bbox)
        # heavy_centers= output['bbox_centers'][heavy_mask]
        heavy_scores= output['scores'][heavy_mask]
        heavy_classes = output['pred_classes'][heavy_mask]
        heavy_bbox, heavy_centers, heavy_scores,heavy_classes,fighv =view_box_center2(heavy_bbox, heavy_centers, heavy_scores,heavy_classes, overlap_dist_thresh=5.0, max_centers_per_box=5)                            
        ###########################start build mol ##########################
        rwmol_ = Chem.RWMol()
        boxi2ai = {}  # 预测索引 -> RDKit 索引
        placeholder_atoms=dict()
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

        # 使用 KDTree 构建重原子间的键（如果提供了 bond_bbox）
        if len(charges_classes) > 0:
            for k,v in a2c.items():
                fc=int(idx_to_labels[charges_classes[v]])
                rwmol_.GetAtomWithIdx(k).SetFormalCharge(fc)
        # print(f"mol with heavy atoms number {i+1}, max heavy atom id {i}")
        print(f"mol with  atoms number {i+1}, max  atom id {i}")
        print(f"mol with bond box number {len(bond_classes)}")
        print(f"placeholder_atoms@@ {placeholder_atoms}")

        #重原子 skeleton mol
        bonds=dict()
        existing_bonds = set()
        b2aa=dict()
        singleAtomBond=[]
        bondWithdirct=[]

        # tree_heavy = KDTree(heavy_centers)#TODO before add bond consdiering reodering bond ??
        tree_atom = KDTree(atom_centers)#TODO as atom bond are all reodered to kee H last
        if len(idx_to_labels)==30:
            _margin=0#ad this version bond dynamicaly changed
        for bi, (bbox, idx_) in enumerate(zip(bond_bbox, bond_classes)):#not work for cross-bond, longer bond, as the center of bond may be close to as atoms not it two atoms
            bond_type = idx_to_labels[idx_]
            if len(idx_to_labels)==23:
                if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                    _margin = 5
                else:
                    _margin = 8
            anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
            oposite_anchor_positions = anchor_positions.copy()
            oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
            # Upper left, lower right, lower left, upper right
            # x1y1, x2y2, x1y2, x2y1 : dinuogl lines
            anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])
            # print(f"anchor_positions {anchor_positions.shape}\n{anchor_positions}")
            dists, neighbours = tree_atom.query(anchor_positions, k=1)
            if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
                # visualize setup
                begin_idx, end_idx = neighbours[:2]
            else:
                # visualize setup
                begin_idx, end_idx = neighbours[2:]
            atom1_idx = boxi2ai[begin_idx]
            atom2_idx = boxi2ai[end_idx]
            if atom1_idx == atom2_idx:#NOTE when bond with only one terminal atom, other side H not used
                print(f"attempt to add self-bond:{bi}  atomIdx1 == atomIdx2 ::{[atom1_idx, atom2_idx]}")
                print(f"for bond bi {bi} H atom may involbed   dists:",dists)
                print(neighbours)
                print("anchor_positions",anchor_positions)
            else:
                if bond_type in  ['-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                    if bond_type in BONDDIRECT:
                        bonds[bi] = (atom1_idx, atom2_idx, 'SINGLE', bond_type)
                        bondWithdirct.append(bi)
                    else:
                        bonds[bi] = (atom1_idx, atom2_idx, 'SINGLE', None)
                    bond_type=BONDTYPE['SINGLE']
                elif bond_type == '=':
                    bonds[bi] = (atom1_idx, atom2_idx, 'DOUBLE', None)
                    bond_type=BONDTYPE['DOUBLE']
                elif bond_type == '#':
                    bonds[bi] = (atom1_idx, atom2_idx, 'TRIPLE', None)
                    bond_type=BONDTYPE['TRIPLE']
                else:
                    print(f'unkown bond type relaced with single@@ {bond_type}')
                    bonds[bi] = (atom1_idx, atom2_idx, 'SINGLE', None)
                    bond_type=BONDTYPE['SINGLE']
                # 检查价态
                atom1 = rwmol_.GetAtomWithIdx(atom1_idx)
                atom2 = rwmol_.GetAtomWithIdx(atom2_idx)
                val1 = sum(b.GetBondTypeAsDouble() for b in atom1.GetBonds())
                val2 = sum(b.GetBondTypeAsDouble() for b in atom2.GetBonds())
                max_val1 = max(VALENCES[atom1.GetSymbol()])
                max_val2 = max(VALENCES[atom2.GetSymbol()])
                # bond_order = bond_type.AsDouble()
                bond_order=BONDTYPE2ORD[bond_type]
                if val1 + bond_order <= max_val1 and val2 + bond_order <= max_val2:
                    bond1 = rwmol_.GetBondBetweenAtoms(atom1_idx, atom2_idx)
                    bond2 = rwmol_.GetBondBetweenAtoms(atom2_idx, atom1_idx)
                    if bond1 or bond2:
                        # print(f'bond exists for {[atom1_idx, atom2_idx]}')
                        pass
                    # if (atom1_idx, atom2_idx) not in existing_bonds and (atom2_idx, atom1_idx) not in existing_bonds:
                    else:    
                        # print(atom1_idx, atom2_idx, bond_type,[ bi, idx_to_labels[idx_] ])
                        rwmol_.AddBond(atom1_idx, atom2_idx, bond_type)
                else:
                    print(f"Skipping bond {bi}: Exceeds valence.")
            existing_bonds.add((atom1_idx, atom2_idx))
            b2aa[bi]=sorted([atom1_idx, atom2_idx])

        if len(bond_bbox)==1 and len(atom_bbox)==2:
            ca1='[*:0][C:2]#[C:3][*:1]'#acs phC#CpH
            rwmol_ = Chem.RWMol()
            ats= ['*','*','C','C']
            for ia in ats:
                a=Chem.Atom(ia)
                id_=rwmol_.AddAtom(a)
                # print(ia,id_)
            rwmol_.AddBond(2, 3, Chem.BondType.TRIPLE)
            rwmol_.AddBond(0, 2, Chem.BondType.SINGLE)
            rwmol_.AddBond(1, 3, Chem.BondType.SINGLE)
            
            # Chem.MolFromSmiles(ca1)
            for i in range(len(atom_classes)):
                atom_classes[i]=lab2idx['*']
            AllChem.Compute2DCoords(rwmol_)
        else:
            rwmol_=copy.deepcopy(rwmol_)
        print(f"placeholder_atoms {placeholder_atoms}")
        
        #assign 2D coords
        mol = rwmol_.GetMol()
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        # conf.Set3D(True)
        # for i, (x, y) in enumerate(heavy_centers):
        for i, (x, y) in enumerate(atom_centers):
            x, y=float(x),float(y)
            conf.SetAtomPosition(i, (x, y, 0))#TODO why some time need -y, just display same as ori?
        mol.AddConformer(conf)
        # Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        rwmol_=Chem.RWMol(mol) 
        #as afte H a\lso didthis
        skeleton_mol=copy.deepcopy(rwmol_)
        print(skeleton_mol.GetNumBonds())
        chiral_centers_aids = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        
        # H realted post-process
        heavyNumber=len(heavy_centers)
        print(f'mol with heavy number atoms {heavyNumber}, max id {heavyNumber-1}')    
        onlyHeayMol=copy.deepcopy(rwmol_)
        chiral_centers = Chem.FindMolChiralCenters(
                        rwmol_, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers] 
        Hais=[]
        Hais_bt=[]
        Hbd=[]
        # H_existing_bonds = set()
        for bi, ais in b2a.items():#from box overlap
            bt=bond_classes[bi]# in [14,15]#directon bond
            for ai in ais:
                if ai>heavyNumber-1:
                    if bt in  [14,15]:#directon bond
                        Hais.append(ais)#NOTE ais ai increasing order as two for loop increasing
                        print(f"within H  bond box id {bi} bond direction {idx_to_labels[bt]} atoms box id {ais} ")
                        Hais_bt.append(idx_to_labels[bt])
                        Hbd.append(bi)
                        # print(bonds[bi] )
        # add Hbonds with direction
        H_existing_bonds = set()
        ha2boxa=dict()
        for ais, bt in zip(Hais,Hais_bt):
            idx_2=ais[-1]
            idx_1=ais[0]
            hbond=rwmol_.GetBondBetweenAtoms(idx_1,idx_2)
            if hbond is not None:
                if idx_1 in chiral_center_ids:#if not in the chiral atom, will not set bond directions
                    hbond.SetBondDir(BOND_DIRS[bt])
            else:
                had = Chem.Atom("H")
                addHatom_idx = rwmol_.AddAtom(had)
                ha2boxa[addHatom_idx]=idx_2
                # print(idx_2,addHatom_idx)#Note if detected H box will lead idx_2 != addHatom_idx
                atom= rwmol_.GetAtomWithIdx(idx_1)
                max_val=max(VALENCES[atom.GetSymbol()])
                val = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
                if (idx_1, addHatom_idx) not in H_existing_bonds and (addHatom_idx, idx_1) not in H_existing_bonds:
                    if val<=max_val-1:
                        # print(f"atom id {idx_1} val {val} max_val {max_val}")
                        print(idx_1, addHatom_idx)#let check bond exist or not!!
                        rwmol_.AddBond(idx_1,addHatom_idx, Chem.BondType.SINGLE)#BOND_DIRS[bt]
                        b=rwmol_.GetBondBetweenAtoms(idx_1,addHatom_idx)
                        if idx_1 in chiral_center_ids:#if not in the chiral atom, will not set bond directions
                            b.SetBondDir(BOND_DIRS[bt])#############Note can be done in the following tree
                H_existing_bonds.add((idx_1,addHatom_idx))
        i
        if len(ha2boxa)>0:#consider Hnow
            #use box coords assign 2D, remove extra Hs also update box
            rwmol_.RemoveAllConformers()#
            conf = Chem.Conformer(rwmol_.GetNumAtoms())
            conf.Set3D(True)
            coords2d=[]
            for i, (x, y) in enumerate(heavy_centers):
                position = Point3D(float(x), float(y), 0.)  # Create a Point3D object with x, y, and z=0
                conf.SetAtomPosition(i, position)
                coords2d.append([x,y])
            for k,v in ha2boxa.items():
                x,y=atom_centers[v]
                position = Point3D(float(x), float(y), 0.)  # Create a Point3D object with x, y, and z=0
                conf.SetAtomPosition(k, position)
                coords2d.append([x,y])
            rwmol_.AddConformer(conf)
            
        additonalH=detect_unconnected_hydrogens(rwmol_)
        if len(additonalH)>0:
            rwmol_,rmovedAtomcoords=remove_unconnected_hydrogens2(rwmol_) #NOTE 留给将来WEB开发用will dercease h atom,but the box have not updated TODO fix me this in feature activate learning
            #update atom box infors
            if len(rmovedAtomcoords)>0:#update box infors
                delbb=[]
                kdt = cKDTree(atom_centers)
                for i, (x,y,z) in enumerate(rmovedAtomcoords):#z=0
                    dist, idx_=kdt.query([x,y], k=1)
                    delbb.append(idx_)
                mask = np.ones(len(atom_classes), dtype=bool)  # 初始化为 True
                mask[delbb] = False 
                atom_bbox = atom_bbox[mask]
                atom_classes = atom_classes[mask]
                atom_centers = atom_centers[mask]
        # mol# mol_rebuit=copy.deepcopy(mol)

        mol=copy.deepcopy(rwmol_)
        conf=mol.GetConformers()[0]
        mola2xy=dict()
        mola2d=[]
        for i,a in enumerate(mol.GetAtoms()):
            x,y,z=conf.GetAtomPosition(i)
            mola2xy[i]=[x,y]
            mola2d.append([x,y])
            # print( x,y,z)
        kdt = cKDTree(mola2d)
        chiral_centers = Chem.FindMolChiralCenters(
                        mol, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers] 

        for bi,bcent in enumerate(bond_centers):
            if bi in bondWithdirct :#and bi not in Hbd:#Note as set Hbd previously
                dists, a1a2 = kdt.query(bcent, k=2)
                a1,a2=sorted(a1a2)
                a1,a2=int(a1),int(a2)
                bt= mol.GetBondBetweenAtoms(a1, a2)#RDKit 的键是无向的，返回的是同一个 Bond 对象
                if bt:
                    # 获取键的当前起点和终点
                    current_begin = bt.GetBeginAtomIdx()
                    current_end = bt.GetEndAtomIdx()
                    bond_dir=bond_dirs[idx_to_labels[bond_classes[bi]]]
                    if bond_dir == rdchem.BondDir.BEGINWEDGE: 
                        reverse_dir = rdchem.BondDir.BEGINDASH 
                    elif bond_dir == rdchem.BondDir.BEGINDASH: 
                        reverse_dir = rdchem.BondDir.BEGINWEDGE
                    # else:
                    #      reverse_dir= rdchem.BondDir.BEGINWEDGE
                    if a1 in chiral_center_ids:
                        if current_begin == a1:
                            bt.SetBondDir(bond_dir)
                            print(f'a1 dir')
                        else:
                            # 如果手性原子是终点，反转方向（例如用相反的楔形键）
                            bt.SetBondDir(reverse_dir)
                            print(f'a1 reverse_dir')
                        # print(f'set bond direction a1a2 {[bi, a1,a2]}')
                        # bt.SetBondDir(bond_dirs[idx_to_labels[bond_classes[bi]]])
                    elif a2 in chiral_center_ids:
                        if current_begin == a2:
                            bt.SetBondDir(bond_dir)
                            print(f'a2 dir {bond_dir} {reverse_dir}')
                        else:
                            # 如果手性原子是终点，反转方向（例如用相反的楔形键）,but not work, just remove and add
                            mol.RemoveBond(current_begin, current_end)
                            mol.AddBond(current_end, current_begin, bt.GetBondType())
                            bond = mol.GetBondBetweenAtoms(current_end, current_begin)
                            bond.SetBondDir(bond_dir)
                            print(f'a2 reverse_dir {bond_dir} {reverse_dir}')
                        # bt= mol.GetBondBetweenAtoms(a2, a1)
                        # print(f'set bond direction a2a1  {[bi, a2,a1]}')            
                        # bt.SetBondDir(bond_dirs[idx_to_labels[bond_classes[bi]]])
                    else:
                        print('bond stro not with chiral atom???, will ignore this stero bond infors')
                        print(f"{[bi, bond_dir, current_begin,current_end]}")
                        # beginatom=mol.GetAtomWithIdx(current_begin)
                        # Endatom=mol.GetAtomWithIdx(current_end)
                        # beginatom_neis=len(beginatom.GetBonds())
                        # Endatom_neis=len(Endatom.GetBonds())
        try:
            mol_rebuit=mol.GetMol()
            conf = mol_rebuit.GetConformer()
            Chem.WedgeMolBonds(mol_rebuit,conf)#
            Chem.DetectBondStereochemistry(mol_rebuit)
            Chem.AssignChiralTypesFromBondDirs(mol_rebuit)
            Chem.AssignStereochemistry(mol_rebuit)
            #
            smiH=Chem.MolToSmiles(mol_rebuit)
            print(F"smiH\n",smiH)
            # canon_smilesH = Chem.CanonSmiles(smiH)
            # print(F"canon_smilesH\n",canon_smilesH)
            # rdkit_coni_smiH=Chem.MolToSmiles(Chem.MolFromSmiles(smiH))
            # print(f"Chem.MolToSmiles(Chem.MolFromSmiles(smiH))\n {rdkit_coni_smiH}")
            #
            mol = rdkit.Chem.RWMol(mol_rebuit)
            other2ppsocr=True
            if other2ppsocr:
                print()
                need_cut=[]
                ppstr=[]
                ppstr_score=[]
                crops=[]
                index_token=dict()
                expan=0#NOTE this control how much the part of bond in crop_Img
                for i_,(heav_c,heav_box) in enumerate(zip(atom_classes,atom_bbox)):
                    if lab2idx['*']==heav_c or lab2idx['other']==heav_c or lab2idx['Cl']==heav_c:
                        need_cut.append(i_)
                        a=heav_box+np.array([-expan,-expan,expan,expan])
                        # print(heav_box.shape,a.shape)
                        box=a * [scale_x, scale_y, scale_x, scale_y]#TODO need the fix as w h may not equal!!
                        # print(a,box,[scale_x, scale_y, scale_x, scale_y])
                        cropped_img = img_ori_1k.crop(box)
                        crops.append(cropped_img)
                        image_npocr = np.array(cropped_img)
                        result_ocr= ocr2.ocr(image_npocr, det=False)
                        s_, score_ =result_ocr[0][0]
                        s_previos=atom_ocr[i_]
                        if s_previos != "other" :
                            s_=s_previos if len(s_previos)>=len(s_) else s_
                        print(f'ocr::idx:{i_}',s_, score_ )
                        if score_<=0.1:# process cropped_img and try again
                            # print(s_, "xxx",score_)
                            s_='*'
                        if s_=='+' or s_=='-':
                            s_="*"
                        if len(s_)>1:
                            s_=re.sub(r'[^a-zA-Z0-9,\*\-\+]', '', s_)#remove special chars
                            if re.match(r'^\d+$', s_):
                                s_=f'{s_}*'#number+ *
                                # print(f'why only numbers ?  {s_}')
                        if s_=='L':s_='Li'
                        elif s_=='0':s_='O'
                        elif s_  in ['N,+ CI','N,+ Cl' ,'N,+Cl','N,+CI','N+CI']:s_='N2+Cl-'
                        elif s_  in ['NO,','O,N' ]:s_='NO2'
                        

                        match = re.match(r'^(\d+)?(.*)', s_)
                        # print(s_,'xxxx')
                        if match:
                            numeric_part, remaining_part = match.groups()
                            fc_=mol.GetAtomWithIdx(i_).GetFormalCharge()
                            if remaining_part in ELEMENTS:
                                new_atom = Chem.Atom(remaining_part)
                                mol.ReplaceAtom(i_, new_atom)
                                print(i_, remaining_part,"@@@")
                            elif remaining_part in ABBREVIATIONS:# can be expanded with placeholder_atoms
                                placeholder_atoms[i_]=s_# such 2Na will be get for rdkit
                            elif remaining_part=='OH':
                                new_atom = Chem.Atom("O")
                                mol.ReplaceAtom(i_, new_atom)
                            elif remaining_part=='SH':
                                new_atom = Chem.Atom("S")
                                mol.ReplaceAtom(i_, new_atom)
                            elif remaining_part=='NH':
                                new_atom = Chem.Atom("N")
                                mol.ReplaceAtom(i_, new_atom)
                            mol.GetAtomWithIdx(i_).SetFormalCharge(fc_)
                        index_token[i_]=f'{s_}:{i_}'
                        print(f"idx:{i_}, atm: <{idx_to_labels[heav_c]}> --- [{s_}:{i_}] with score:{score_} ||previousOCR:: {atom_ocr[i_]}")
                        if s_ in ELEMENTS :
                            new_atom = Chem.Atom(s_)
                            mol.ReplaceAtom(i_, new_atom)
                        mol.GetAtomWithIdx(i_).SetProp("atomLabel", f"{s_}")#mol set with label, mol_rebuit not
                        ppstr.append(s_)
                        ppstr_score.append(score_)
                        if  s_ in ABBREVIATIONS.keys():
                            placeholder_atoms[i_]=s_
            #            
            bond_dirs_rev={v:k for k,v in bond_dirs.items()}
            wdbs=[]
            for b in mol.GetBonds():
                bd=b.GetBondDir()
                bt=b.GetBondType()
                # print(bd)
                if bd ==bond_dirs['BEGINDASH'] or  bd==bond_dirs['BEGINWEDGE']:
                    a1,a2=b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                    wdbs.append([a1,a2,bt,bond_dirs_rev[bd]])

            #expand mol if exists
            # if len(placeholder_atoms)>0:###
            cm=copy.deepcopy(mol)
            # print(placeholder_atoms)
            expand_mol, expand_smiles= expandABB(cm,ABBREVIATIONS, placeholder_atoms)
            SMILESpre=expand_smiles
            rdm=copy.deepcopy(expand_mol)
            target_mol, ref_mol=rdm, cm
            AllChem.Compute2DCoords(target_mol)
            pair=[target_mol, ref_mol]
            mcs=rdFMCS.FindMCS([target_mol, ref_mol], # larger,small order
                                # atomCompare=rdFMCS.AtomCompare.CompareAny,
                                bondCompare=rdFMCS.BondCompare.CompareAny,
                                ringCompare=rdFMCS.RingCompare.IgnoreRingFusion,
                                matchChiralTag=False,
                )
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            AllChem.Compute2DCoords(mcs_mol)

            matches0 = pair[0].GetSubstructMatches(mcs_mol, useQueryQueryMatches=True,uniquify=False, maxMatches=1000, useChirality=False)
            matches1 = pair[1].GetSubstructMatches(mcs_mol, useQueryQueryMatches=True,uniquify=False, maxMatches=1000, useChirality=False)
            if len(matches0) != len(matches1):
                matches0=list(matches0)
                matches1=list(matches1)
                # print( "noted: matcher not equal !!")
                if len(matches0)>len(matches1):
                    for i in range(0,len(matches0)):
                        if i < len(matches1):
                            pass
                        else:
                            ii=i % len(matches1)
                            matches1.append(matches1[ii])
                else:
                    for i in range(0,len(matches1)):
                        if i < len(matches0):
                            pass
                        else:
                            ii=i % len(matches0)
                            matches0.append(matches0[ii])
            assert len(matches0) == len(matches1), "matcher not equal break!!"
            atommaping_pairs=[list(zip(matches0[i],matches1[i])) for i in range(0,len(matches0))]
            atomMap=atommaping_pairs[0]
            rmsd2=rdkit.Chem.rdMolAlign.AlignMol(prbMol=target_mol, refMol=ref_mol, atomMap=atomMap,maxIters=2000000)
            print(f"rmsd {rmsd2}")
            #ocr_mol
            ocr_mol = copy.deepcopy(target_mol)
            AllChem.Compute2DCoords(ocr_mol)
            ocr_smi = Chem.MolToSmiles(ocr_mol)
            molexp=ocr_mol
            expandStero_smi, success= rdkit_canonicalize_smiles(ocr_smi)
            # expandStero_smi =  Chem.CanonSmiles(ocr_smi)#, useChiral=(not ignore_chiral))

            # TODO #[3H] 2H prpared box for training are too smalled, need adjust
            if visual_check:
                boxed_img = draw_objs(img,
                                    atom_bbox,
                                    atom_classes,
                                    atom_scores,
                                    category_index=idx_to_labels,
                                    box_thresh=0.5,
                                    line_thickness=3,
                                    font='arial.ttf',
                                    font_size=10)
                opts = Draw.MolDrawOptions()
                opts.addAtomIndices = False
                opts.addStereoAnnotation = False
                img_ori = Image.open(image_path).convert('RGB')
                img_ori_1k = img_ori.resize((1000,1000))
                if other2ppsocr:
                    img_rebuit = Draw.MolToImage(ocr_mol, options=opts,size=(1000, 1000))
                else:
                    img_rebuit = Draw.MolToImage(ocr_mol, options=opts,size=(1000, 1000))
                combined_img = Image.new('RGB', (img_ori_1k.width + boxed_img.width + img_rebuit.width, img_ori_1k.height))
                combined_img.paste(img_ori_1k, (0, 0))
                combined_img.paste(boxed_img, (img_ori_1k.width, 0))
                combined_img.paste(img_rebuit, (img_ori_1k.width + boxed_img.width, 0))
                imprefix=os.path.basename(image_path).split('.')[0]
                combined_img.save(f"{ima_checkdir}/{imprefix}Boxed.png")
            
            new_row = {'file_name':image_path, "SMILESori":SMILESori,
                    'SMILESpre':SMILESpre,
                    'SMILESexp':expandStero_smi, 
                    }
            smiles_data = smiles_data._append(new_row, ignore_index=True)
            
            #accu  similarity calculation 
            if getacc:
                sameWithOutStero=comparing_smiles(new_row,SMILESpre)#try to ingnore cis chiral, as 2d coords including all the infos
                sameWithOutStero_exp=comparing_smiles(new_row,expandStero_smi)#this ignore chairity and *number be * NOTE

                if (type(SMILESori)!=type('a')) or (type(SMILESpre)!=type('a')):
                    if sameWithOutStero or sameWithOutStero_exp:
                        mysum += 1
                    else:
                        print(f"smiles problems\n{SMILESori}\n{SMILESpre}\n{image_path}")
                        failed.append([SMILESori,SMILESpre,image_path])
                        mydiff.append([SMILESori,SMILESpre,image_path])
                        continue
                mol1 = Chem.MolFromSmiles(SMILESori)#TODO considering smiles with rdkit not recongized in real data
                if mol1 is None:
                    rd_smi_ori, success1_=rdkit_canonicalize_smiles(SMILESori)
                    mol1=Chem.MolFromSmiles(rd_smi_ori)
                if (mol_rebuit is None) or (mol1 is None):
                    if sameWithOutStero or sameWithOutStero_exp:
                        mysum += 1
                    else:
                        print(f'get rdkit mol None\n{SMILESori}\n{SMILESpre}\n{image_path}')
                        failed.append([SMILESori,SMILESpre,image_path])
                        mydiff.append([SMILESori,SMILESpre,image_path])
                        continue
                if mol1:
                    rdk_smi1=Chem.MolToSmiles(mol1)
                else:
                    rdk_smi1=SMILESori
                if mol_rebuit:
                    rdk_smi2=Chem.MolToSmiles(mol_rebuit)
                else:
                    rdk_smi2=''
                # if rdk_smi1==rdk_smi2 or rdk_smi1==expandStero_smi or sameWithOutStero:#also considering the abbre in Ori
                if rdk_smi1==rdk_smi2 or rdk_smi1==expandStero_smi:
                    mysum += 1
                else:
                    if sameWithOutStero or sameWithOutStero_exp:
                        mysum += 1
                    else:
                        mydiff.append([SMILESori,SMILESpre,image_path])
                        if visual_check:
                            combined_img.save(f"{ima_checkdir}/{imprefix}Boxed_diff{len(mydiff)}.png")
                try:
                    morganfps1 = AllChem.GetMorganFingerprint(mol1, 3,useChirality=True)
                    morganfps2 = AllChem.GetMorganFingerprint(mol_rebuit, 3,useChirality=True)
                    morgan_tani = DataStructs.DiceSimilarity(morganfps1, morganfps2)
                    fp1 = Chem.RDKFingerprint(mol1)
                    fp2 = Chem.RDKFingerprint(mol_rebuit)
                    tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
                    if expandStero_smi!= '':
                        fp3 = Chem.RDKFingerprint(molexp)
                        morganfps3 = AllChem.GetMorganFingerprint(molexp, 3,useChirality=True)
                        morgan_tani3 = DataStructs.DiceSimilarity(morganfps1, morganfps3)
                        tanimoto3 = DataStructs.FingerprintSimilarity(fp1, fp3)
                    if morgan_tani3> morgan_tani or tanimoto3> tanimoto :
                        sim+=morgan_tani3
                        simRD+=tanimoto3
                    else:
                        simRD+=tanimoto
                        sim+=morgan_tani
                except Exception as e:
                    print(f"mol to fingerprint erros")
                    simRD+=0
                    sim+=0
                    continue
        except Exception as e:
            print(f"file_name@: {image_path}\n SMILES in csv:\n{SMILESori}")
            raise Exception("@debug this!!\n")

    if getacc:
        sim_100 = 100*sim/len(smiles_data)
        simrd100 = 100*simRD/len(smiles_data)
        flogout.write(f"rdkit concanlized==smiles:{100*mysum/len(smiles_data)}%\n")
        flogout.write(f"failed:{len(failed)}\n totoal saved in csv : {len(smiles_data)}\n")
        flogout.write(f"avarage similarity morgan tanimoto: RDKFp tanimoto:: {sim_100}%,  {simrd100}%  \n")#morgan_tani considering chiraty
        flogout.write(f'I2M@@:: match--{mysum},unmatch--{len(mydiff)},failed--{len(failed)},correct %{100*mysum/len(smiles_data)} \n')
        #molscribe evalutate
        from src.solver.evaluate import SmilesEvaluator
        evaluator = SmilesEvaluator(smiles_data['SMILESori'], tanimoto=False)
        res_pre=evaluator.evaluate(smiles_data['SMILESpre'])
        res_exp=evaluator.evaluate(smiles_data['SMILESexp'])
        flogout.write(f'MolScribe style evaluation@SMILESpre:: {str(res_pre)} \n')
        flogout.write(f'MolScribe style evaluation@SMILESexp:: {str(res_exp)} \n')
        flogout.close()
    print(f"will save {len(smiles_data)} dataframe into csv") 
    smiles_data.to_csv(outcsv_filename, index=False)


import torch.nn as nn 
import torch.nn.functional as F 
import torchvision


class RTDETRPostProcessor(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, classes_dict=None, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        if classes_dict is None:
            classes_dict = {0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
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
        num_classes=len(classes_dict)
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

        mscoco_category2label = {k: i for i, k in enumerate(classes_dict.keys())}
        mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
        self.mscoco_label2category=mscoco_label2category

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes):

        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            boxes = bbox_pred
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            # from ...data.coco import mscoco_label2category
            labels = torch.tensor([self.mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)
        
        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )