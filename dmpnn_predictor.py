#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于DMPNN的分子属性预测系统

这个模块实现了基于DMPNN (Directed Message Passing Neural Network) 的分子属性预测系统。
DMPNN是专门为分子设计的图神经网络，能够有效处理分子的原子和键信息。

主要功能:
- 支持回归和分类任务
- 五折交叉验证
- 结果可视化和保存
- 兼容SMILES格式的分子数据

作者: AI Assistant
日期: 2024年
版本: 1.0
"""

import os
import json
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
import re
from datetime import datetime

# 设置中文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

class MolecularDataProcessor:
    """分子数据处理器"""
    
    def __init__(self):
        # 动态计算特征维度
        self.atom_features_dim = self._calculate_atom_features_dim()
        self.bond_features_dim = self._calculate_bond_features_dim()
    
    def _calculate_atom_features_dim(self):
        """计算原子特征维度"""
        # 原子类型 (44个)
        atom_types_dim = 44
        # 度数 (6个)
        degree_dim = 6
        # 形式电荷 (5个: -2, -1, 0, 1, 2)
        formal_charge_dim = 5
        # 杂化类型 (5个)
        hybridization_dim = 5
        # 芳香性 (1个)
        aromatic_dim = 1
        # 氢原子数 (5个: 0, 1, 2, 3, 4+)
        num_hs_dim = 5
        # 价电子数 (7个: 0, 1, 2, 3, 4, 5, 6+)
        valence_dim = 7
        # 在环中 (1个)
        in_ring_dim = 1
        # 原子质量 (1个)
        mass_dim = 1
        
        total_dim = (atom_types_dim + degree_dim + formal_charge_dim + 
                    hybridization_dim + aromatic_dim + num_hs_dim + 
                    valence_dim + in_ring_dim + mass_dim)
        
        return total_dim
    
    def _calculate_bond_features_dim(self):
        """计算键特征维度"""
        # 键类型 (4个)
        bond_types_dim = 4
        # 共轭 (1个)
        conjugated_dim = 1
        # 在环中 (1个)
        in_ring_dim = 1
        # 立体化学 (4个)
        stereo_dim = 4
        # 键长和键角 (2个占位符)
        geometry_dim = 2
        
        total_dim = bond_types_dim + conjugated_dim + in_ring_dim + stereo_dim + geometry_dim
        
        return total_dim
    
    def get_atom_features(self, atom):
        """提取原子特征"""
        features = []
        
        # 原子类型 (one-hot编码)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        atom_type = atom.GetSymbol()
        features.extend([1 if atom_type == t else 0 for t in atom_types])
        
        # 度数
        degree = atom.GetDegree()
        features.extend([1 if degree == i else 0 for i in range(6)])
        
        # 形式电荷
        formal_charge = atom.GetFormalCharge()
        features.extend([1 if formal_charge == i else 0 for i in range(-2, 3)])
        
        # 杂化类型
        hybridization_types = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, 
                              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                              Chem.rdchem.HybridizationType.SP3D2]
        hybridization = atom.GetHybridization()
        features.extend([1 if hybridization == h else 0 for h in hybridization_types])
        
        # 芳香性
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # 氢原子数
        num_hs = atom.GetTotalNumHs()
        features.extend([1 if num_hs == i else 0 for i in range(5)])
        
        # 价电子数
        valence = atom.GetTotalValence()
        features.extend([1 if valence == i else 0 for i in range(7)])
        
        # 在环中
        features.append(1 if atom.IsInRing() else 0)
        
        # 原子质量 (标准化)
        mass = atom.GetMass()
        features.append(mass / 100.0)
        
        return features
    
    def get_bond_features(self, bond):
        """提取键特征"""
        features = []
        
        # 键类型
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                     Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type = bond.GetBondType()
        features.extend([1 if bond_type == bt else 0 for bt in bond_types])
        
        # 共轭
        features.append(1 if bond.GetIsConjugated() else 0)
        
        # 在环中
        features.append(1 if bond.IsInRing() else 0)
        
        # 立体化学
        stereo_types = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY, 
                       Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
        stereo = bond.GetStereo()
        features.extend([1 if stereo == st else 0 for st in stereo_types])
        
        # 键长 (估算)
        features.append(1.0)  # 占位符
        
        # 键角 (估算)
        features.append(1.0)  # 占位符
        
        return features
    
    def mol_to_graph(self, smiles):
        """将SMILES转换为图数据"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        # 边和边特征
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # DMPNN使用有向边
            edge_indices.extend([[i, j], [j, i]])
            
            bond_feat = self.get_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])
        
        # 转换为张量
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_molecular_descriptors(self, smiles):
        """计算分子描述符"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * 10
        
        try:
            # 尝试使用FractionCsp3，如果不存在则使用替代方案
            try:
                fraction_csp3 = Descriptors.FractionCsp3(mol)
            except AttributeError:
                # 计算Csp3原子的比例作为替代
                total_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
                sp3_carbons = sum(1 for atom in mol.GetAtoms() 
                                if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
                fraction_csp3 = sp3_carbons / total_carbons if total_carbons > 0 else 0.0
            
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumSaturatedRings(mol),
                fraction_csp3,
                rdMolDescriptors.BertzCT(mol)
            ]
        except Exception as e:
            # 如果计算描述符失败，返回默认值
            descriptors = [0.0] * 10
        
        return descriptors

class MolecularDataset(Dataset):
    """分子数据集"""
    
    def __init__(self, smiles_list, targets, processor, task_type='regression'):
        self.processor = processor
        self.task_type = task_type
        self.data = []
        
        for i, smiles in enumerate(smiles_list):
            graph_data = processor.mol_to_graph(smiles)
            if graph_data is not None:
                descriptors = processor.get_molecular_descriptors(smiles)
                graph_data.descriptors = torch.tensor(descriptors, dtype=torch.float)
                
                if task_type == 'regression':
                    graph_data.y = torch.tensor(targets[i], dtype=torch.float)
                else:
                    graph_data.y = torch.tensor(targets[i], dtype=torch.long)
                
                self.data.append(graph_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DMPNNLayer(nn.Module):
    """DMPNN消息传递层"""
    
    def __init__(self, atom_features_dim, bond_features_dim, hidden_dim):
        super(DMPNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 消息网络
        self.message_net = nn.Sequential(
            nn.Linear(atom_features_dim + bond_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(atom_features_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        # 计算消息
        messages = self.message_net(torch.cat([x[row], edge_attr], dim=1))
        
        # 聚合消息
        aggregated = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        aggregated.index_add_(0, col, messages)
        
        # 更新节点特征
        updated_x = self.update_net(torch.cat([x, aggregated], dim=1))
        
        return updated_x

class MolecularDMPNN(nn.Module):
    """DMPNN分子属性预测模型"""
    
    def __init__(self, atom_features_dim, bond_features_dim, descriptor_dim, 
                 hidden_dim=300, num_layers=3, dropout=0.0, task_type='regression', num_classes=1):
        super(MolecularDMPNN, self).__init__()
        
        self.task_type = task_type
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Linear(atom_features_dim, hidden_dim)
        
        # DMPNN层
        self.dmpnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.dmpnn_layers.append(DMPNNLayer(hidden_dim, bond_features_dim, hidden_dim))
        
        # 读出层
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 预测层
        if task_type == 'regression':
            self.predictor = nn.Linear(hidden_dim // 2, 1)
        else:
            self.predictor = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        
        # 输入投影
        x = self.input_projection(x)
        
        # DMPNN消息传递
        for layer in self.dmpnn_layers:
            x = layer(x, edge_index, edge_attr)
        
        # 全局池化
        batch_size = batch.batch.max().item() + 1
        graph_repr = torch.zeros(batch_size, x.size(1), device=x.device)
        graph_repr.index_add_(0, batch.batch, x)
        
        # 添加分子描述符
        if hasattr(batch, 'descriptors') and batch.descriptors is not None:
            # 确保描述符的维度正确
            descriptors = batch.descriptors
            if descriptors.dim() == 1:
                descriptors = descriptors.unsqueeze(0)
            if descriptors.size(0) != batch_size:
                # 如果描述符数量不匹配，重复或截取
                if descriptors.size(0) == 1:
                    descriptors = descriptors.repeat(batch_size, 1)
                else:
                    descriptors = descriptors[:batch_size]
            
            graph_repr = torch.cat([graph_repr, descriptors], dim=1)
        
        # 预测
        out = self.readout(graph_repr)
        out = self.predictor(out)
        
        if self.task_type == 'regression':
            return out.view(-1)  # 确保输出是1D张量
        else:
            return out

class MolecularPropertyPredictor:
    """分子属性预测器主类 - 改进版本，支持批量处理和完整结果管理"""
    
    def __init__(self, task_type='regression', device=None, algorithm_name='DMPNN', base_results_dir='DMPNN_results'):
        self.task_type = task_type
        
        # 改进的GPU检测和设置
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"🚀 检测到GPU: {torch.cuda.get_device_name(0)}")
                print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                self.device = torch.device('cpu')
                print("💻 使用CPU进行计算")
        else:
            self.device = device
            
        self.processor = MolecularDataProcessor()
        self.model = None
        self.algorithm_name = algorithm_name
        self.csv_filename = None
        self.base_results_dir = base_results_dir
        
        # 创建分类结果目录
        self.results_dirs = {
            'scatter_plots': os.path.join(base_results_dir, 'scatter_plots'),
            'detailed_results': os.path.join(base_results_dir, 'detailed_results'),
            'summary_results': os.path.join(base_results_dir, 'summary_results'),
            'visualizations': os.path.join(base_results_dir, 'visualizations'),
            'best_models': os.path.join(base_results_dir, 'best_models')
        }
        
        # 创建所有结果目录
        for dir_path in self.results_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # 全局摘要文件路径
        self.global_summary_file = os.path.join(self.results_dirs['summary_results'], 
                                               f'global_summary_{self.algorithm_name}.csv')
        
        print(f"🔧 使用设备: {self.device}")
        print(f"🏷️  算法名称: {self.algorithm_name}")
        print(f"📁 结果保存目录: {base_results_dir}")
        print(f"📊 全局摘要文件: {self.global_summary_file}")
        
        # 设置CUDA优化（如果使用GPU）
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def load_data(self, csv_file, smiles_col=None, target_col=None, test_size=0.2, random_state=42):
        """加载CSV数据，自动识别SMILES/CSMILES列和目标列"""
        print(f"正在加载数据: {csv_file}")
        
        # 保存CSV文件名（不含路径和扩展名）
        self.csv_filename = os.path.splitext(os.path.basename(csv_file))[0]
        
        df = pd.read_csv(csv_file)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 自动检测SMILES列
        if smiles_col is None:
            # 可能的SMILES列名（按优先级排序）
            possible_smiles_cols = ['SMILES', 'CSMILES', 'smiles', 'csmiles', 'Smiles', 'CSmiles', 
                                  'canonical_smiles', 'canonical_SMILES', 'CANONICAL_SMILES']
            
            smiles_col = None
            for col_name in possible_smiles_cols:
                if col_name in df.columns:
                    smiles_col = col_name
                    print(f"🔍 自动检测到SMILES列: {smiles_col}")
                    break
            
            if smiles_col is None:
                # 如果没有找到常见的SMILES列名，提示用户
                available_cols = list(df.columns)
                raise ValueError(f"未能自动检测到SMILES列。请手动指定smiles_col参数。\n"
                               f"可用列: {available_cols}\n"
                               f"支持的SMILES列名: {possible_smiles_cols}")
        else:
            print(f"📋 使用指定的SMILES列: {smiles_col}")
        
        # 保存SMILES列名用于记录
        self.smiles_column_name = smiles_col
        
        # 如果未指定目标列，使用最后一列
        if target_col is None:
            target_col = df.columns[-1]
            print(f"🎯 自动识别目标列: {target_col}")
        
        # 保存目标列名用于可视化
        self.target_column_name = target_col
        
        # 检查最后一列是否为property_log，如果是，则使用倒数第二列的列名作为散点图标题
        if target_col.lower() == 'property_log' and len(df.columns) >= 2:
            self.scatter_plot_target_name = df.columns[-2]
            print(f"检测到property_log列，散点图将使用列名: {self.scatter_plot_target_name}")
        else:
            self.scatter_plot_target_name = target_col
        
        # 检查必要的列
        if smiles_col not in df.columns:
            raise ValueError(f"未找到SMILES列: {smiles_col}。可用列: {list(df.columns)}")
        if target_col not in df.columns:
            raise ValueError(f"未找到目标列: {target_col}")
        
        # 移除缺失值
        df = df.dropna(subset=[smiles_col, target_col])
        print(f"移除缺失值后数据形状: {df.shape}")
        
        smiles_list = df[smiles_col].tolist()
        targets = df[target_col].tolist()
        
        # 处理分类任务的标签编码
        if self.task_type == 'classification':
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            targets = label_encoder.fit_transform(targets)
            self.num_classes = len(np.unique(targets))
            print(f"分类任务，类别数: {self.num_classes}")
        else:
            self.num_classes = 1
            print(f"回归任务，目标值范围: {min(targets):.3f} - {max(targets):.3f}")
        
        # 保存完整数据用于交叉验证
        self.full_smiles = smiles_list
        self.full_targets = targets
        
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        train_smiles, test_smiles, train_targets, test_targets = train_test_split(
            smiles_list, targets, test_size=test_size, random_state=random_state, 
            stratify=targets if self.task_type == 'classification' else None
        )
        
        # 创建数据集
        self.train_dataset = MolecularDataset(train_smiles, train_targets, self.processor, self.task_type)
        self.test_dataset = MolecularDataset(test_smiles, test_targets, self.processor, self.task_type)
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"测试集大小: {len(self.test_dataset)}")
        
        return self.train_dataset, self.test_dataset
    
    def _setup_results_directories(self):
        """设置结果目录"""
        base_dir = f"{self.csv_filename}_results"
        
        self.results_dirs = {
            'base': base_dir,
            'best_models': os.path.join(base_dir, 'best_models'),
            'detailed_results': os.path.join(base_dir, 'detailed_results'),
            'summary_results': os.path.join(base_dir, 'summary_results'),
            'scatter_plots': os.path.join(base_dir, 'scatter_plots'),
            'visualizations': os.path.join(base_dir, 'visualizations')
        }
        
        for dir_path in self.results_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def cross_validate(self, n_splits=5, epochs=100, batch_size=32, learning_rate=0.001, 
                      weight_decay=1e-4, patience=10, hidden_dim=300, num_layers=3, dropout=0.0):
        """五折交叉验证"""
        print(f"\n🔄 开始 {n_splits} 折交叉验证...")
        print(f"🖥️  使用设备: {self.device}")
        print(f"📊 参数: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        if not hasattr(self, 'full_smiles'):
            raise ValueError("请先调用 load_data() 加载数据")
        
        start_time = time.time()
        
        # 交叉验证分割
        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = kfold.split(self.full_smiles, self.full_targets)
        else:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = kfold.split(self.full_smiles)
        
        # 结果存储
        cv_results = {
            'fold_scores': [],
            'fold_train_metrics': [],
            'fold_test_metrics': [],
            'best_fold_idx': 0,
            'best_score': -float('inf') if self.task_type == 'regression' else 0,
            'training_time': 0,
            'final_fold_train_pred': [],
            'final_fold_train_true': [],
            'final_fold_test_pred': [],
            'final_fold_test_true': []
        }
        
        temp_model_files = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n📊 第 {fold + 1}/{n_splits} 折...")
            
            # 准备数据
            train_smiles = [self.full_smiles[i] for i in train_idx]
            val_smiles = [self.full_smiles[i] for i in val_idx]
            train_targets = [self.full_targets[i] for i in train_idx]
            val_targets = [self.full_targets[i] for i in val_idx]
            
            train_dataset = MolecularDataset(train_smiles, train_targets, self.processor, self.task_type)
            val_dataset = MolecularDataset(val_smiles, val_targets, self.processor, self.task_type)
            
            print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
            
            # 创建模型
            if len(train_dataset) > 0:
                descriptor_dim = len(train_dataset[0].descriptors)
            else:
                descriptor_dim = 10
            
            model = MolecularDMPNN(
                atom_features_dim=self.processor.atom_features_dim,
                bond_features_dim=self.processor.bond_features_dim,
                descriptor_dim=descriptor_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task_type,
                num_classes=self.num_classes
            ).to(self.device)
            
            # 训练模型
            temp_model_path = os.path.join(self.results_dirs['best_models'], 
                                         f'temp_{self.csv_filename}_{self.algorithm_name}_fold_{fold+1}.pth')
            self._train_single_fold(model, train_dataset, val_dataset, epochs, batch_size, 
                                  learning_rate, weight_decay, patience, temp_model_path)
            
            temp_model_files.append(temp_model_path)
            
            # 评估模型
            train_metrics = self._evaluate_with_metrics(model, train_dataset)
            test_metrics = self._evaluate_with_metrics(model, val_dataset)
            
            cv_results['fold_train_metrics'].append(train_metrics)
            cv_results['fold_test_metrics'].append(test_metrics)
            
            # 计算折叠得分
            if self.task_type == 'regression':
                fold_score = test_metrics['r2']
            else:
                fold_score = test_metrics['accuracy']
            
            cv_results['fold_scores'].append(fold_score)
            
            if fold_score > cv_results['best_score']:
                cv_results['best_score'] = fold_score
                cv_results['best_fold_idx'] = fold
            
            # 保存最后一折的预测结果用于可视化
            if fold == n_splits - 1:
                cv_results['final_fold_train_pred'] = train_metrics['predictions']
                cv_results['final_fold_train_true'] = train_metrics['true_values']
                cv_results['final_fold_test_pred'] = test_metrics['predictions']
                cv_results['final_fold_test_true'] = test_metrics['true_values']
            
            if self.task_type == 'regression':
                print(f"   训练集 - R²: {train_metrics['r2']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}")
                print(f"   测试集 - R²: {test_metrics['r2']:.4f}, MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        # 计算统计结果
        total_training_time = time.time() - start_time
        mean_score = np.mean(cv_results['fold_scores'])
        std_score = np.std(cv_results['fold_scores'])
        
        # 计算平均指标
        avg_train_metrics = self._calculate_average_metrics(cv_results['fold_train_metrics'])
        avg_test_metrics = self._calculate_average_metrics(cv_results['fold_test_metrics'])
        
        print(f"\n🎯 交叉验证结果:")
        print(f"   平均得分: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   训练时间: {total_training_time:.2f} 秒")
        print(f"   最佳折叠: 第 {cv_results['best_fold_idx'] + 1} 折 (得分: {cv_results['best_score']:.4f})")
        
        # 保存结果
        self._save_cv_results(cv_results, mean_score, std_score, total_training_time, avg_train_metrics, avg_test_metrics)
        self._save_best_model_only(cv_results, temp_model_files)
        
        # 更新全局摘要文件
        self._update_global_summary(avg_train_metrics, avg_test_metrics, total_training_time, mean_score, std_score)
        
        # 生成可视化图表
        self.plot_results(cv_results)
        
        return cv_results, mean_score, std_score, total_training_time
    
    def _train_single_fold(self, model, train_dataset, val_dataset, epochs, batch_size, 
                          learning_rate, weight_decay, patience, save_path):
        """训练单个折叠"""
        
        def custom_collate_fn(data_list):
            """自定义批处理函数，正确处理描述符"""
            batch = Batch.from_data_list(data_list)
            # 重新处理描述符 - 堆叠而不是连接
            if hasattr(data_list[0], 'descriptors'):
                descriptors_list = [data.descriptors for data in data_list]
                batch.descriptors = torch.stack(descriptors_list, dim=0)
            return batch
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=custom_collate_fn)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        if self.task_type == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            val_loss = self._evaluate_loss(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(save_path))
    
    def _evaluate_loss(self, model, data_loader, criterion):
        """评估损失"""
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = model(batch)
                loss = criterion(output, batch.y)
                total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def _evaluate_with_metrics(self, model, dataset):
        """评估模型并返回指标"""
        
        def custom_collate_fn(data_list):
            """自定义批处理函数，正确处理描述符"""
            batch = Batch.from_data_list(data_list)
            # 重新处理描述符 - 堆叠而不是连接
            if hasattr(data_list[0], 'descriptors'):
                descriptors_list = [data.descriptors for data in data_list]
                batch.descriptors = torch.stack(descriptors_list, dim=0)
            return batch
        
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                               collate_fn=custom_collate_fn)
        
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = model(batch)
                
                if self.task_type == 'regression':
                    predictions.extend(output.cpu().numpy())
                    true_values.extend(batch.y.cpu().numpy())
                else:
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                    true_values.extend(batch.y.cpu().numpy())
        
        if self.task_type == 'regression':
            r2 = r2_score(true_values, predictions)
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            
            return {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'predictions': predictions,
                'true_values': true_values
            }
        else:
            accuracy = accuracy_score(true_values, predictions)
            
            return {
                'accuracy': accuracy,
                'predictions': predictions,
                'true_values': true_values
            }
    
    def _calculate_average_metrics(self, metrics_list):
        """计算平均指标"""
        if not metrics_list:
            return {}
        
        if self.task_type == 'regression':
            return {
                'r2': np.mean([m['r2'] for m in metrics_list]),
                'mse': np.mean([m['mse'] for m in metrics_list]),
                'mae': np.mean([m['mae'] for m in metrics_list])
            }
        else:
            return {
                'accuracy': np.mean([m['accuracy'] for m in metrics_list])
            }
    
    def _save_cv_results(self, cv_results, mean_score, std_score, total_training_time, avg_train_metrics, avg_test_metrics):
        """保存交叉验证结果 - 改进版本，与GCN项目格式保持一致"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换numpy类型为Python原生类型的辅助函数
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # 转换所有指标为可序列化格式
        avg_train_metrics = convert_to_serializable(avg_train_metrics)
        avg_test_metrics = convert_to_serializable(avg_test_metrics)
        
        # 转换fold_train_metrics和fold_test_metrics
        fold_train_metrics = []
        fold_test_metrics = []
        
        for metrics in cv_results['fold_train_metrics']:
            converted_metrics = {}
            for key, value in metrics.items():
                if key in ['predictions', 'true_values']:
                    converted_metrics[key] = [float(x) for x in value]
                else:
                    converted_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            fold_train_metrics.append(converted_metrics)
        
        for metrics in cv_results['fold_test_metrics']:
            converted_metrics = {}
            for key, value in metrics.items():
                if key in ['predictions', 'true_values']:
                    converted_metrics[key] = [float(x) for x in value]
                else:
                    converted_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            fold_test_metrics.append(converted_metrics)
        
        # 保存详细结果到detailed_results文件夹（与GCN项目格式一致）
        results_data = {
            'dataset': self.csv_filename,
            'algorithm': self.algorithm_name,
            'task_type': self.task_type,
            'timestamp': timestamp,
            'training_time_seconds': float(total_training_time),
            'cross_validation': {
                'n_folds': len(cv_results['fold_scores']),
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'fold_scores': [float(score) for score in cv_results['fold_scores']],
                'best_fold': int(cv_results['best_fold_idx'] + 1),
                'best_score': float(cv_results['best_score']),
                'metric': 'R²' if self.task_type == 'regression' else 'Accuracy'
            },
            'average_metrics': {
                'train': avg_train_metrics,
                'test': avg_test_metrics
            },
            'fold_details': {
                'train_metrics': fold_train_metrics,
                'test_metrics': fold_test_metrics
            },
            # 最后一折的训练集和测试集数据（与GCN项目格式一致）
            'final_fold_data': {
                'train_predictions': [float(p) for p in cv_results['final_fold_train_pred']],
                'train_true_values': [float(t) for t in cv_results['final_fold_train_true']],
                'test_predictions': [float(p) for p in cv_results['final_fold_test_pred']],
                'test_true_values': [float(t) for t in cv_results['final_fold_test_true']]
            }
        }
        
        # 保存JSON详细结果
        json_filename = f'{self.algorithm_name}_{self.csv_filename}_results.json'
        json_path = os.path.join(self.results_dirs['detailed_results'], json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存:")
        print(f"   详细结果: {json_path}")
        print(f"   📊 摘要已更新到全局文件: {self.global_summary_file}")
    
    def _save_best_model_only(self, cv_results, temp_model_files):
        """保存最佳模型"""
        best_fold_idx = cv_results['best_fold_idx']
        best_temp_model = temp_model_files[best_fold_idx]
        
        final_model_path = os.path.join(self.results_dirs['best_models'], 
                                      f'{self.algorithm_name}_{self.csv_filename}_best_model.pth')
        
        shutil.copy2(best_temp_model, final_model_path)
        
        # 清理临时文件
        for temp_file in temp_model_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print(f"   🏆 最佳模型已保存: {final_model_path}")
    
    def plot_results(self, cv_results):
        """绘制结果图表"""
        if not cv_results:
            print("❌ 没有可用的交叉验证结果进行绘图")
            return
        
        print(f"\n📊 生成DMPNN结果图表...")
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算平均指标用于可视化
        avg_train_metrics = self._calculate_average_metrics(cv_results['fold_train_metrics'])
        avg_test_metrics = self._calculate_average_metrics(cv_results['fold_test_metrics'])
        
        # 计算统计信息
        mean_score = np.mean(cv_results['fold_scores'])
        std_score = np.std(cv_results['fold_scores'])
        training_time = cv_results.get('training_time', 0)
        
        # 生成综合可视化图表
        self._plot_cv_results(cv_results, mean_score, std_score, timestamp, training_time, avg_train_metrics, avg_test_metrics)
        
        # 生成改进版散点图（如果是回归任务）
        if self.task_type == 'regression':
            self._plot_improved_scatter(
                cv_results['final_fold_test_true'],
                cv_results['final_fold_test_pred'],
                cv_results['final_fold_train_true'],
                cv_results['final_fold_train_pred'],
                timestamp
            )
        
        print(f"   📈 图表已保存到: {self.results_dirs['scatter_plots']}")
        print(f"   🎨 可视化图表已保存到: {self.results_dirs['visualizations']}")

    def _plot_cv_results(self, cv_results, mean_score, std_score, timestamp, training_time, avg_train_metrics, avg_test_metrics):
        """绘制交叉验证结果 - 综合可视化（参考GCN项目）"""
        plt.figure(figsize=(16, 12))
        
        # 1. 折叠得分条形图
        plt.subplot(2, 3, 1)
        folds = [f'Fold {i+1}' for i in range(len(cv_results['fold_scores']))]
        plt.bar(folds, cv_results['fold_scores'], alpha=0.7, color='skyblue')
        plt.axhline(y=mean_score, color='red', linestyle='--', label=f'Mean: {mean_score:.4f}')
        plt.title(f'Cross-Validation Fold Scores\n{self.algorithm_name} on {self.csv_filename}', fontsize=12)
        plt.ylabel('Score', fontsize=10)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. 预测vs真实值散点图（区分训练集和测试集）
        plt.subplot(2, 3, 2)
        
        # 检查是否有最后一折的数据
        if cv_results.get('final_fold_train_pred') and cv_results.get('final_fold_test_pred'):
            # 训练集散点图（蓝色）
            train_pred = cv_results['final_fold_train_pred']
            train_true = cv_results['final_fold_train_true']
            plt.scatter(train_true, train_pred, alpha=0.6, color='blue', s=20, label='Training Set')
            
            # 测试集散点图（红色）
            test_pred = cv_results['final_fold_test_pred']
            test_true = cv_results['final_fold_test_true']
            plt.scatter(test_true, test_pred, alpha=0.8, color='red', s=30, label='Test Set')
            
            # 计算整体范围
            all_true = train_true + test_true
            all_pred = train_pred + test_pred
        else:
            # 如果没有分离的数据，使用默认显示
            all_pred = []
            all_true = []
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=plt.gca().transAxes)
        
        if self.task_type == 'regression' and all_true and all_pred:
            min_val = min(min(all_true), min(all_pred))
            max_val = max(max(all_true), max(all_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
            plt.xlabel('True Values', fontsize=10)
            plt.ylabel('Predicted Values', fontsize=10)
            plt.title(f'Predictions vs True Values\nR² = {mean_score:.4f}', fontsize=12)
        else:
            plt.xlabel('True Class', fontsize=10)
            plt.ylabel('Predicted Class', fontsize=10)
            plt.title(f'Predictions vs True Class\nAccuracy = {mean_score:.4f}', fontsize=12)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 得分分布箱线图
        plt.subplot(2, 3, 3)
        plt.boxplot(cv_results['fold_scores'], labels=['Cross-Validation'])
        plt.title(f'Score Distribution\nMean ± Std: {mean_score:.4f} ± {std_score:.4f}', fontsize=12)
        plt.ylabel('Score', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 4. 残差图（仅回归任务）
        if self.task_type == 'regression':
            plt.subplot(2, 3, 4)
            if cv_results.get('final_fold_test_pred'):
                # 使用测试集数据绘制残差图
                test_pred = np.array(cv_results['final_fold_test_pred'])
                test_true = np.array(cv_results['final_fold_test_true'])
                residuals = test_pred - test_true
                plt.scatter(test_pred, residuals, alpha=0.6, color='red', s=20)
            else:
                residuals = np.array([0])  # 默认值
            
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.8)
            plt.xlabel('Predicted Values', fontsize=10)
            plt.ylabel('Residuals', fontsize=10)
            plt.title('Residual Plot', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 5. 残差分布直方图
            plt.subplot(2, 3, 5)
            if len(residuals) > 1:
                plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
            plt.xlabel('Residuals', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('Residual Distribution', fontsize=12)
            plt.grid(True, alpha=0.3)
        else:
            # 分类任务：混淆矩阵风格的可视化
            from sklearn.metrics import confusion_matrix
            
            if cv_results.get('final_fold_test_pred'):
                cm = confusion_matrix(cv_results['final_fold_test_true'], cv_results['final_fold_test_pred'])
            else:
                cm = np.array([[1, 0], [0, 1]])  # 默认2x2矩阵
            
            plt.subplot(2, 3, 4)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix', fontsize=12)
            plt.colorbar()
            
            # 添加数值标签
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            plt.xlabel('Predicted Class', fontsize=10)
            plt.ylabel('True Class', fontsize=10)
        
        # 6. 算法信息
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # GPU信息
        gpu_info = "GPU" if torch.cuda.is_available() and self.device.type == 'cuda' else "CPU"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and self.device.type == 'cuda' else "N/A"
        
        if self.task_type == 'regression':
            info_text = f"""Algorithm Information:
• Dataset: {self.csv_filename}
• Algorithm: {self.algorithm_name}
• Task Type: {self.task_type}
• Cross-Validation: {len(cv_results['fold_scores'])} folds
• Training Time: {training_time:.2f}s
• Test R²: {avg_test_metrics['r2']:.4f}
• Test MSE: {avg_test_metrics['mse']:.4f}
• Test MAE: {avg_test_metrics['mae']:.4f}
• Device: {gpu_info}
• GPU: {gpu_name if gpu_name != "N/A" else "Not Used"}
• Timestamp: {timestamp}
            """
        else:
            info_text = f"""Algorithm Information:
• Dataset: {self.csv_filename}
• Algorithm: {self.algorithm_name}
• Task Type: {self.task_type}
• Cross-Validation: {len(cv_results['fold_scores'])} folds
• Training Time: {training_time:.2f}s
• Test Accuracy: {avg_test_metrics['accuracy']:.4f}
• Device: {gpu_info}
• GPU: {gpu_name if gpu_name != "N/A" else "Not Used"}
• Timestamp: {timestamp}
            """
        
        plt.text(0.05, 0.95, info_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        # 保存图片到visualizations文件夹
        plot_filename = f'{self.csv_filename}_{self.algorithm_name}_cv_plot_{timestamp}.png'
        plot_path = os.path.join(self.results_dirs['visualizations'], plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，不显示
        
        print(f"   📊 综合可视化结果: {plot_path}")
    
    def _plot_improved_scatter(self, test_true, test_pred, train_true, train_pred, timestamp=None):
        """绘制改进版散点图 - 参考GCN项目样式"""
        if not test_pred or not train_pred:
            print("   ⚠️ 没有可用的预测数据，跳过改进版散点图")
            return
        
        train_pred = np.array(train_pred)
        train_true = np.array(train_true)
        test_pred = np.array(test_pred)
        test_true = np.array(test_true)
        
        if self.task_type != 'regression':
            print("   ⚠️ 改进版散点图仅支持回归任务")
            return
        
        # 创建图形，设置Arial字体
        plt.figure(figsize=(8, 8))
        plt.rcParams['font.family'] = 'Arial'
        
        # 绘制散点图
        plt.scatter(train_true, train_pred, s=50, c='#005BAD', alpha=0.7, label='Train')
        plt.scatter(test_true, test_pred, s=50, c='#F56476', alpha=0.7, label='Test')
        
        # 计算数据范围
        all_true = np.concatenate([train_true, test_true])
        all_pred = np.concatenate([train_pred, test_pred])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        buffer = (max_val - min_val) * 0.02
        
        # 设置轴范围
        plt.xlim(min_val - buffer, max_val + buffer)
        plt.ylim(min_val - buffer, max_val + buffer)
        
        # 添加理想预测线
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='grey', label='ideal')
        
        # 处理标题中的变量名（获取目标列名）
        target_name = "Property"  # 默认名称
        if hasattr(self, 'scatter_plot_target_name'):
            # 优先使用专门为散点图设置的目标名称
            target_name = self.scatter_plot_target_name
        elif hasattr(self, 'target_column_name'):
            target_name = self.target_column_name
        elif hasattr(self, 'csv_filename'):
            # 从CSV文件名推断
            import re
            match = re.match(r'^(.*?)[\[\(].*?[\]\)]$', self.csv_filename)
            target_name = match.group(1).strip() if match else self.csv_filename
        
        # 添加标题（使用Arial字体）
        plt.text(0.5, 0.95, f"{self.algorithm_name}_{target_name}", 
                 transform=plt.gca().transAxes, fontsize=18, fontfamily='Arial',
                 verticalalignment='top', horizontalalignment='center')
        
        # 设置刻度和标签（使用Arial字体）
        plt.tick_params(which='major', direction='in', length=5, labelsize=16)
        plt.xlabel(f"Actual {target_name}", fontsize=16, fontfamily='Arial')
        plt.ylabel(f"Predicted {target_name}", fontsize=16, fontfamily='Arial')
        plt.legend(loc='upper left', fontsize=16, prop={'family': 'Arial'})
        
        # 计算评估指标
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        train_r2 = r2_score(train_true, train_pred)
        train_mse = mean_squared_error(train_true, train_pred)
        train_mae = mean_absolute_error(train_true, train_pred)
        test_r2 = r2_score(test_true, test_pred)
        test_mse = mean_squared_error(test_true, test_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        
        # 添加评估指标文本（使用Arial字体）
        metrics_text = (
            f"R²_train: {train_r2:.3f}\n"
            f"MSE_train: {train_mse:.3f}\n"
            f"MAE_train: {train_mae:.3f}\n"
            f"R²_test: {test_r2:.3f}\n"
            f"MSE_test: {test_mse:.3f}\n"
            f"MAE_test: {test_mae:.3f}"
        )
        
        plt.text(0.65, 0.05, metrics_text,
                 transform=plt.gca().transAxes, fontsize=16, fontfamily='Arial',
                 verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保持比例一致
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 保存图像
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"{self.algorithm_name}_{self.csv_filename}_improved_scatter_{timestamp}.png"
        image_path = os.path.join(self.results_dirs['scatter_plots'], image_filename)
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✨ 改进版散点图: {image_path}")

    def _update_global_summary(self, avg_train_metrics, avg_test_metrics, training_time, mean_score, std_score):
        """更新全局摘要文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 准备新的摘要数据
        if self.task_type == 'regression':
            new_summary = {
                'Dataset': self.csv_filename,
                'Algorithm': self.algorithm_name,
                'Test_R2': float(avg_test_metrics['r2']),
                'Test_MSE': float(avg_test_metrics['mse']),
                'Test_MAE': float(avg_test_metrics['mae']),
                'Train_R2': float(avg_train_metrics['r2']),
                'Train_MSE': float(avg_train_metrics['mse']),
                'Train_MAE': float(avg_train_metrics['mae']),
                'CV_Mean_Score': float(mean_score),
                'CV_Std_Score': float(std_score),
                'Training_Time': float(training_time),
                'Timestamp': timestamp
            }
        else:
            new_summary = {
                'Dataset': self.csv_filename,
                'Algorithm': self.algorithm_name,
                'Test_Accuracy': float(avg_test_metrics['accuracy']),
                'Train_Accuracy': float(avg_train_metrics['accuracy']),
                'CV_Mean_Score': float(mean_score),
                'CV_Std_Score': float(std_score),
                'Training_Time': float(training_time),
                'Timestamp': timestamp
            }
        
        # 读取现有的全局摘要文件（如果存在）
        if os.path.exists(self.global_summary_file):
            try:
                existing_df = pd.read_csv(self.global_summary_file)
                
                # 检查是否已存在相同数据集的记录
                mask = (existing_df['Dataset'] == self.csv_filename) & (existing_df['Algorithm'] == self.algorithm_name)
                
                if mask.any():
                    # 更新现有记录
                    for key, value in new_summary.items():
                        existing_df.loc[mask, key] = value
                    print(f"   📝 更新现有记录: {self.csv_filename}")
                else:
                    # 添加新记录
                    new_row_df = pd.DataFrame([new_summary])
                    existing_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                    print(f"   📝 添加新记录: {self.csv_filename}")
                
            except Exception as e:
                print(f"⚠️ 读取现有摘要文件失败: {str(e)}")
                # 创建新的DataFrame
                existing_df = pd.DataFrame([new_summary])
        else:
            # 创建新的DataFrame
            existing_df = pd.DataFrame([new_summary])
            print(f"   📝 创建新的全局摘要文件")
        
        # 按数据集名称排序
        existing_df = existing_df.sort_values('Dataset')
        
        # 保存更新后的摘要文件
        existing_df.to_csv(self.global_summary_file, index=False)
        
        print(f"   💾 全局摘要已更新: {self.global_summary_file}")
        print(f"   📊 当前包含 {len(existing_df)} 个数据集的结果")

    def load_data_from_folder(self, folder_path, smiles_col=None, target_col=None):
        """从文件夹中加载所有CSV文件"""
        import glob
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"在文件夹 {folder_path} 中未找到CSV文件")
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件:")
        for file in csv_files:
            print(f"   - {os.path.basename(file)}")
        
        return csv_files
    
    def process_multiple_datasets(self, folder_path, smiles_col=None, target_col=None, 
                                 n_splits=5, epochs=50, batch_size=32, learning_rate=0.001,
                                 hidden_dim=300, num_layers=3, dropout=0.0):
        """处理多个数据集"""
        csv_files = self.load_data_from_folder(folder_path, smiles_col, target_col)
        
        all_results = []
        
        for csv_file in csv_files:
            print(f"\n{'='*60}")
            print(f"🔄 处理数据集: {os.path.basename(csv_file)}")
            print(f"{'='*60}")
            
            try:
                # 加载单个数据集
                self.load_data(csv_file, smiles_col, target_col)
                
                # 执行交叉验证
                cv_results, mean_score, std_score, training_time = self.cross_validate(
                    n_splits=n_splits, epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate, hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=dropout
                )
                
                # 保存结果
                result_summary = {
                    'dataset': self.csv_filename,
                    'algorithm': self.algorithm_name,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'training_time': training_time,
                    'cv_results': cv_results
                }
                
                all_results.append(result_summary)
                
                print(f"✅ {self.csv_filename} 处理完成")
                
            except Exception as e:
                print(f"❌ 处理 {os.path.basename(csv_file)} 时出错: {str(e)}")
                continue
        
        # 保存总体摘要
        self._save_overall_summary(all_results)
        
        return all_results

    def batch_process_folder(self, folder_path, n_splits=5, epochs=100, batch_size=32, 
                           learning_rate=0.001, hidden_dim=300, num_layers=3, dropout=0.0):
        """
        批处理文件夹中的所有CSV文件
        
        Args:
            folder_path: 包含CSV文件的文件夹路径
            其他参数: 交叉验证参数
            
        Returns:
            list: 所有数据集的结果列表
        """
        print(f"\n🔄 开始批处理文件夹: {folder_path}")
        
        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ 文件夹中没有找到CSV文件")
            return []
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件")
        
        all_results = []
        successful_count = 0
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*50}")
            print(f"📊 处理第 {i}/{len(csv_files)} 个文件: {csv_file}")
            print(f"{'='*50}")
            
            csv_path = os.path.join(folder_path, csv_file)
            
            try:
                # 加载数据
                train_dataset, test_dataset = self.load_data(csv_path, smiles_col=None, target_col=None)
                
                # 执行交叉验证
                cv_results, mean_score, std_score, training_time = self.cross_validate(
                    n_splits=n_splits,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout
                )
                
                # 保存结果
                result = {
                    'dataset': os.path.splitext(csv_file)[0],
                    'algorithm': self.algorithm_name,
                    'cv_results': cv_results,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'training_time': training_time
                }
                
                all_results.append(result)
                successful_count += 1
                
                print(f"✅ {csv_file} 处理成功")
                
            except Exception as e:
                print(f"❌ {csv_file} 处理失败: {str(e)}")
                continue
        
        # 显示总体摘要（全局摘要文件已经在每次处理时更新）
        if all_results:
            self._save_overall_summary(all_results)
        
        print(f"\n📊 批处理完成:")
        print(f"   成功处理: {successful_count}/{len(csv_files)} 个文件")
        
        return all_results

    def _save_overall_summary(self, all_results):
        """保存所有数据集的总体摘要 - 已被全局摘要文件替代"""
        # 这个方法已经被全局摘要文件机制替代
        print(f"\n📊 所有结果已保存在全局摘要文件: {self.global_summary_file}")
        
        # 读取并显示当前摘要统计
        if os.path.exists(self.global_summary_file):
            try:
                summary_df = pd.read_csv(self.global_summary_file)
                print(f"   📋 总计处理了 {len(summary_df)} 个数据集")
                
                if self.task_type == 'regression':
                    avg_test_r2 = summary_df['Test_R2'].mean()
                    print(f"   📈 平均测试集R²: {avg_test_r2:.4f}")
                else:
                    avg_test_acc = summary_df['Test_Accuracy'].mean()
                    print(f"   📈 平均测试集准确率: {avg_test_acc:.4f}")
                    
            except Exception as e:
                print(f"⚠️ 读取摘要统计失败: {str(e)}")
        
        return self.global_summary_file


# 使用示例
if __name__ == "__main__":
    def main():
        """主函数 - 改进版本"""
        print("🧬 DMPNN分子性质预测系统 - 改进版本")
        print("=" * 50)
        
        # 一次性选择运行模式
        print("\n🔄 选择运行模式:")
        print("1. 处理单个CSV文件")
        print("2. 批量处理文件夹中的所有CSV文件")
        
        mode = input("请选择模式 (1 或 2，默认为 2): ").strip()
        
        # 选择任务类型
        print("\n📊 选择任务类型:")
        print("1. 回归任务 (regression)")
        print("2. 分类任务 (classification)")
        
        task_choice = input("请选择任务类型 (1 或 2，默认为 1): ").strip()
        task_type = 'classification' if task_choice == '2' else 'regression'
        
        # 创建预测器
        predictor = MolecularPropertyPredictor(task_type=task_type, algorithm_name='DMPNN')
        
        if mode == '1':
            # 单个文件模式
            csv_file = input("请输入CSV文件路径: ").strip()
            if not csv_file:
                csv_file = r'G:\jupyter\work_test_one\dataset\SMILES\Glass transition temperature.csv'
                print(f"使用默认文件: {csv_file}")
            
            try:
                # 加载数据（自动识别最后一列为目标列）
                train_dataset, test_dataset = predictor.load_data(
                    csv_file=csv_file,
                    smiles_col=None  # 自动检测SMILES/CSMILES列
                )
                
                print("\n🔄 开始五折交叉验证...")
                cv_results, mean_score, std_score, training_time = predictor.cross_validate(
                    n_splits=5,
                    epochs=50,
                    batch_size=32,
                    learning_rate=0.001,
                    hidden_dim=300,
                    num_layers=3,
                    dropout=0.0
                )
                
                print(f"\n✅ 处理完成！")
                print(f"   平均得分: {mean_score:.4f} ± {std_score:.4f}")
                print(f"   训练时间: {training_time:.2f} 秒")
                
            except FileNotFoundError:
                print("❌ 未找到数据文件，请确保CSV文件存在")
                print("📋 CSV文件应包含以下列:")
                print("   - SMILES: 分子的SMILES字符串")
                print("   - 最后一列: 目标性质值（自动识别）")
                
        else:
            # 批量处理模式（默认）
            folder_path = input("请输入包含CSV文件的文件夹路径: ").strip()
            if not folder_path:
                folder_path = r'G:\jupyter\work_test_one\dataset\SMILES'
                print(f"使用默认文件夹: {folder_path}")
            
            try:
                print(f"\n🔄 开始批量处理文件夹: {folder_path}")
                all_results = predictor.batch_process_folder(
                    folder_path=folder_path,
                    n_splits=5,
                    epochs=100,
                    batch_size=32,
                    learning_rate=0.001,
                    hidden_dim=300,
                    num_layers=3,
                    dropout=0.0
                )
                
                print(f"\n🎉 批量处理完成！")
                print(f"   成功处理 {len(all_results)} 个数据集")
                print(f"   结果保存在: {predictor.base_results_dir}")
                
                # 显示简要统计
                if all_results:
                    if task_type == 'regression':
                        avg_r2 = np.mean([r['mean_score'] for r in all_results])
                        print(f"   平均R²得分: {avg_r2:.4f}")
                    else:
                        avg_acc = np.mean([r['mean_score'] for r in all_results])
                        print(f"   平均准确率: {avg_acc:.4f}")
                
            except Exception as e:
                print(f"❌ 批量处理时出错: {str(e)}")
        
        print(f"\n📁 所有结果已保存到以下文件夹:")
        print(f"   📊 散点图: {predictor.results_dirs['scatter_plots']}")
        print(f"   📋 详细结果: {predictor.results_dirs['detailed_results']}")
        print(f"   📈 结果摘要: {predictor.results_dirs['summary_results']}")
        print(f"   🎨 可视化图表: {predictor.results_dirs['visualizations']}")
        print(f"   🏆 最佳模型: {predictor.results_dirs['best_models']}")

    main() 