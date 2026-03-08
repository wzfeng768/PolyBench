#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图卷积神经网络分子性质预测模型
使用SMILES/CSMILES字符串预测化学分子的性质
支持回归和分类任务，支持五折交叉验证，自动检测SMILES/CSMILES列
改进版本：支持批量处理多个CSV文件，完整的评估指标，分文件夹保存结果，修复输出形状问题
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
import os
import json
import glob
import time
import re
from datetime import datetime

# 设置中文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

class MolecularDataProcessor:
    """分子数据处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.atom_features_dim = 9  # 原子特征维度
        
    def smiles_to_graph(self, smiles):
        """将SMILES字符串转换为图数据"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 获取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
                atom.GetTotalValence(),
                int(atom.IsInRing()),
                atom.GetNumRadicalElectrons()
            ]
            atom_features.append(features)
        
        # 获取边信息
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 添加双向边
            edge_indices.extend([[i, j], [j, i]])
            
            # 边特征：键类型
            bond_type = bond.GetBondType()
            bond_features = [
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_attrs.extend([bond_features, bond_features])
        
        # 转换为张量
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.empty((0, 6))
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def calculate_molecular_descriptors(self, smiles):
        """计算分子描述符"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 10  # 返回默认值
        
        descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol)
        ]
        return descriptors

class MolecularDataset(Dataset):
    """分子数据集类"""
    
    def __init__(self, smiles_list, targets, processor, task_type='regression'):
        self.smiles_list = smiles_list
        self.targets = targets
        self.processor = processor
        self.task_type = task_type
        
        # 预处理所有分子
        self.graphs = []
        self.descriptors = []
        self.valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            graph = processor.smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.descriptors.append(processor.calculate_molecular_descriptors(smiles))
                self.valid_indices.append(i)
        
        # 过滤有效的目标值
        self.targets = [targets[i] for i in self.valid_indices]
        
        print(f"成功处理 {len(self.graphs)} / {len(smiles_list)} 个分子")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        descriptors = torch.tensor(self.descriptors[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float if self.task_type == 'regression' else torch.long)
        
        # 将描述符添加到图数据中
        graph.descriptors = descriptors
        graph.y = target
        
        return graph

class MolecularGCN(nn.Module):
    """分子图卷积神经网络"""
    
    def __init__(self, atom_features_dim=9, descriptor_dim=10, hidden_dim=128, num_layers=3, 
                 dropout=0.2, task_type='regression', num_classes=1):
        super(MolecularGCN, self).__init__()
        
        self.task_type = task_type
        self.num_layers = num_layers
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(atom_features_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 注意力层（可选）
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        
        # 批归一化
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # 分子描述符处理
        self.descriptor_fc = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 最终预测层
        final_dim = hidden_dim + hidden_dim // 4  # 图特征 + 描述符特征
        
        if task_type == 'regression':
            self.predictor = nn.Sequential(
                nn.Linear(final_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
        else:  # classification
            self.predictor = nn.Sequential(
                nn.Linear(final_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, num_classes)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        descriptors = batch.descriptors
        
        # 重新整形描述符以匹配批次大小
        batch_size = batch_idx.max().item() + 1
        descriptor_dim = descriptors.shape[0] // batch_size
        descriptors = descriptors.view(batch_size, descriptor_dim)
        
        # 图卷积
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 注意力机制
        x = self.attention(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 图级别池化
        graph_features = global_mean_pool(x, batch_idx)
        
        # 处理分子描述符
        descriptor_features = self.descriptor_fc(descriptors)
        
        # 特征融合
        combined_features = torch.cat([graph_features, descriptor_features], dim=1)
        
        # 最终预测
        output = self.predictor(combined_features)
        
        if self.task_type == 'regression':
            return output.squeeze()
        else:
            return output

class MolecularPropertyPredictor:
    """分子性质预测器主类"""
    
    def __init__(self, task_type='regression', device=None, algorithm_name='GCN', base_results_dir='GCN_results'):
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
        self.train_losses = []
        self.val_losses = []
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
            targets = self.processor.label_encoder.fit_transform(targets)
            self.num_classes = len(np.unique(targets))
            print(f"分类任务，类别数: {self.num_classes}")
        else:
            self.num_classes = 1
            print(f"回归任务，目标值范围: {min(targets):.3f} - {max(targets):.3f}")
        
        # 保存完整数据用于交叉验证
        self.full_smiles = smiles_list
        self.full_targets = targets
        
        # 划分训练集和测试集
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
    
    def create_model(self, hidden_dim=128, num_layers=3, dropout=0.2):
        """创建模型"""
        # 获取描述符维度
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            sample_descriptors = self.train_dataset[0].descriptors
            descriptor_dim = len(sample_descriptors)
        else:
            descriptor_dim = 10  # 默认值
        
        self.model = MolecularGCN(
            atom_features_dim=self.processor.atom_features_dim,
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            task_type=self.task_type,
            num_classes=self.num_classes
        ).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        return self.model
    
    def cross_validate(self, n_splits=5, epochs=100, batch_size=32, learning_rate=0.001, 
                      weight_decay=1e-4, patience=10, hidden_dim=128, num_layers=3, dropout=0.2):
        """五折交叉验证 - 改进版本，只保存最佳模型"""
        print(f"\n🔄 开始 {n_splits} 折交叉验证...")
        print(f"🖥️  使用设备: {self.device}")
        
        if not hasattr(self, 'full_smiles'):
            raise ValueError("请先调用 load_data() 加载数据")
        
        # 记录开始时间
        start_time = time.time()
        
        # 选择交叉验证策略
        if self.task_type == 'classification':
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kfold.split(self.full_smiles, self.full_targets))
        else:
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kfold.split(self.full_smiles))
        
        cv_results = {
            'fold_scores': [],
            'fold_train_metrics': [],
            'fold_test_metrics': [],
            'fold_models': [],
            'best_fold_idx': 0,
            'best_score': -float('inf') if self.task_type == 'regression' else 0,
            # 保存最后一折的训练集和测试集数据
            'final_fold_train_pred': [],
            'final_fold_train_true': [],
            'final_fold_test_pred': [],
            'final_fold_test_true': []
        }
        
        # 临时模型文件列表，用于最后清理
        temp_model_files = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n📊 第 {fold + 1}/{n_splits} 折...")
            
            # 准备折叠数据
            train_smiles = [self.full_smiles[i] for i in train_idx]
            val_smiles = [self.full_smiles[i] for i in val_idx]
            train_targets = [self.full_targets[i] for i in train_idx]
            val_targets = [self.full_targets[i] for i in val_idx]
            
            # 创建数据集
            train_dataset = MolecularDataset(train_smiles, train_targets, self.processor, self.task_type)
            val_dataset = MolecularDataset(val_smiles, val_targets, self.processor, self.task_type)
            
            print(f"   训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
            
            # 创建模型
            if len(train_dataset) > 0:
                descriptor_dim = len(train_dataset[0].descriptors)
            else:
                descriptor_dim = 10
                
            model = MolecularGCN(
                atom_features_dim=self.processor.atom_features_dim,
                descriptor_dim=descriptor_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task_type,
                num_classes=self.num_classes
            ).to(self.device)
            
            # 训练模型 - 使用临时文件名
            temp_model_path = os.path.join(self.results_dirs['best_models'], 
                                         f'temp_{self.csv_filename}_{self.algorithm_name}_fold_{fold+1}.pth')
            train_losses, val_losses = self._train_single_fold(
                model, train_dataset, val_dataset, epochs, batch_size, 
                learning_rate, weight_decay, patience, temp_model_path
            )
            
            temp_model_files.append(temp_model_path)
            
            # 评估模型 - 获取完整指标
            train_metrics = self._evaluate_with_metrics(model, train_dataset)
            test_metrics = self._evaluate_with_metrics(model, val_dataset)
            
            cv_results['fold_train_metrics'].append(train_metrics)
            cv_results['fold_test_metrics'].append(test_metrics)
            cv_results['fold_models'].append(temp_model_path)
            
            # 使用测试集R²作为主要评估指标
            if self.task_type == 'regression':
                fold_score = test_metrics['r2']
            else:
                fold_score = test_metrics['accuracy']
            
            cv_results['fold_scores'].append(fold_score)
            
            # 记录最佳折叠
            if fold_score > cv_results['best_score']:
                cv_results['best_score'] = fold_score
                cv_results['best_fold_idx'] = fold
            
            # 如果是最后一折，保存训练集和测试集的预测结果
            if fold == n_splits - 1:
                print(f"   💾 保存最后一折的训练集和测试集预测结果...")
                cv_results['final_fold_train_pred'] = train_metrics['predictions']
                cv_results['final_fold_train_true'] = train_metrics['true_values']
                cv_results['final_fold_test_pred'] = test_metrics['predictions']
                cv_results['final_fold_test_true'] = test_metrics['true_values']
            
            print(f"   第 {fold + 1} 折得分: {fold_score:.4f}")
            if self.task_type == 'regression':
                print(f"   训练集 - R²: {train_metrics['r2']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}")
                print(f"   测试集 - R²: {test_metrics['r2']:.4f}, MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        # 计算总训练时间
        total_training_time = time.time() - start_time
        
        # 计算交叉验证统计
        mean_score = np.mean(cv_results['fold_scores'])
        std_score = np.std(cv_results['fold_scores'])
        
        # 计算平均指标
        if self.task_type == 'regression':
            avg_train_metrics = {
                'r2': np.mean([m['r2'] for m in cv_results['fold_train_metrics']]),
                'mse': np.mean([m['mse'] for m in cv_results['fold_train_metrics']]),
                'mae': np.mean([m['mae'] for m in cv_results['fold_train_metrics']])
            }
            avg_test_metrics = {
                'r2': np.mean([m['r2'] for m in cv_results['fold_test_metrics']]),
                'mse': np.mean([m['mse'] for m in cv_results['fold_test_metrics']]),
                'mae': np.mean([m['mae'] for m in cv_results['fold_test_metrics']])
            }
        else:
            avg_train_metrics = {
                'accuracy': np.mean([m['accuracy'] for m in cv_results['fold_train_metrics']])
            }
            avg_test_metrics = {
                'accuracy': np.mean([m['accuracy'] for m in cv_results['fold_test_metrics']])
            }
        
        print(f"\n📈 交叉验证结果:")
        print(f"   平均得分: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   训练时间: {total_training_time:.2f} 秒")
        print(f"   最佳折叠: 第 {cv_results['best_fold_idx'] + 1} 折 (得分: {cv_results['best_score']:.4f})")
        
        if self.task_type == 'regression':
            print(f"   平均训练集指标 - R²: {avg_train_metrics['r2']:.4f}, MSE: {avg_train_metrics['mse']:.4f}, MAE: {avg_train_metrics['mae']:.4f}")
            print(f"   平均测试集指标 - R²: {avg_test_metrics['r2']:.4f}, MSE: {avg_test_metrics['mse']:.4f}, MAE: {avg_test_metrics['mae']:.4f}")
        
        # 保存交叉验证结果
        self._save_cv_results(cv_results, mean_score, std_score, total_training_time, avg_train_metrics, avg_test_metrics)
        
        # 只保存最佳模型，删除其他临时模型
        self._save_best_model_only(cv_results, temp_model_files)
        
        # 更新全局摘要文件
        self._update_global_summary(avg_train_metrics, avg_test_metrics, total_training_time, mean_score, std_score)
        
        return cv_results, mean_score, std_score, total_training_time
    
    def _train_single_fold(self, model, train_dataset, val_dataset, epochs, batch_size, 
                          learning_rate, weight_decay, patience, save_path):
        """训练单个折叠的模型"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=lambda x: Batch.from_data_list(x))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x: Batch.from_data_list(x))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        if self.task_type == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
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
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            val_loss = self._evaluate_loss(model, val_loader, criterion)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # 早停检查
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
        return train_losses, val_losses
    
    def _evaluate_single_fold(self, model, val_dataset):
        """评估单个折叠的模型"""
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                              collate_fn=lambda x: Batch.from_data_list(x))
        
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                output = model(batch)
                
                if self.task_type == 'regression':
                    # Handle both single and batch outputs properly
                    output_np = output.cpu().numpy()
                    if output_np.ndim == 0:
                        predictions.append(float(output_np))
                    else:
                        predictions.extend(output_np.flatten())
                    
                    true_np = batch.y.cpu().numpy()
                    if true_np.ndim == 0:
                        true_values.append(float(true_np))
                    else:
                        true_values.extend(true_np.flatten())
                else:
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                    true_values.extend(batch.y.cpu().numpy())
        
        # 计算得分
        if self.task_type == 'regression':
            score = r2_score(true_values, predictions)
        else:
            score = accuracy_score(true_values, predictions)
        
        return predictions, true_values, score
    
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
    
    def _save_cv_results(self, cv_results, mean_score, std_score, training_time, avg_train_metrics, avg_test_metrics):
        """保存交叉验证结果 - 改进版本，不再保存单独的CSV摘要"""
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
        
        # 保存详细结果到detailed_results文件夹
        results_data = {
            'dataset': self.csv_filename,
            'algorithm': self.algorithm_name,
            'task_type': self.task_type,
            'timestamp': timestamp,
            'training_time_seconds': float(training_time),
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
            # 最后一折的训练集和测试集数据
            'final_fold_data': {
                'train_predictions': [float(p) for p in cv_results['final_fold_train_pred']],
                'train_true_values': [float(t) for t in cv_results['final_fold_train_true']],
                'test_predictions': [float(p) for p in cv_results['final_fold_test_pred']],
                'test_true_values': [float(t) for t in cv_results['final_fold_test_true']]
            }
        }
        
        # 保存JSON详细结果
        json_filename = f'{self.csv_filename}_{self.algorithm_name}_detailed_results_{timestamp}.json'
        json_path = os.path.join(self.results_dirs['detailed_results'], json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存:")
        print(f"   详细结果: {json_path}")
        print(f"   📊 摘要已更新到全局文件: {self.global_summary_file}")
        
        # 生成可视化（不显示，直接保存）
        self._plot_cv_results(cv_results, mean_score, std_score, timestamp, training_time, avg_train_metrics, avg_test_metrics)
    
    def _plot_cv_results(self, cv_results, mean_score, std_score, timestamp, training_time, avg_train_metrics, avg_test_metrics):
        """绘制交叉验证结果 - 改进版本（不显示，直接保存）"""
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
        if cv_results['final_fold_train_pred'] and cv_results['final_fold_test_pred']:
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
            # 如果没有分离的数据，使用所有预测结果
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
            if cv_results['final_fold_test_pred']:
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
            
            if cv_results['final_fold_test_pred']:
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
        
        print(f"   可视化结果: {plot_path}")
        
        # 额外保存一个专门的训练集vs测试集对比图
        if cv_results['final_fold_train_pred'] and cv_results['final_fold_test_pred']:
            self._plot_improved_scatter(cv_results, timestamp, avg_train_metrics, avg_test_metrics)
    
    def _plot_improved_scatter(self, cv_results, timestamp, avg_train_metrics, avg_test_metrics):
        """绘制改进版散点图 - 参考用户提供的样式"""
        if not cv_results['final_fold_train_pred'] or not cv_results['final_fold_test_pred']:
            print("   ⚠️ 没有可用的预测数据，跳过改进版散点图")
            return
        
        train_pred = np.array(cv_results['final_fold_train_pred'])
        train_true = np.array(cv_results['final_fold_train_true'])
        test_pred = np.array(cv_results['final_fold_test_pred'])
        test_true = np.array(cv_results['final_fold_test_true'])
        
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
        image_filename = f"{self.algorithm_name}_{self.csv_filename}_improved_scatter_{timestamp}.png"
        image_path = os.path.join(self.results_dirs['scatter_plots'], image_filename)
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   改进版散点图: {image_path}")

    def train(self, epochs=100, batch_size=32, learning_rate=0.001, weight_decay=1e-4, 
              patience=10, save_path=None):
        """训练模型（单次训练，非交叉验证）"""
        if self.model is None:
            self.create_model()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dirs['best_models'], f'{self.csv_filename}_{self.algorithm_name}_model_{timestamp}.pth')
        
        # 数据加载器
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=lambda x: Batch.from_data_list(x))
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        if self.task_type == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        print("开始训练...")
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                output = self.model(batch)
                loss = criterion(output, batch.y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段（使用部分训练数据）
            val_loss = self.evaluate(train_loader, criterion, max_batches=5)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f"早停于第 {epoch} 轮")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(save_path))
        print(f"训练完成，最佳模型已保存至: {save_path}")
    
    def evaluate(self, data_loader, criterion, max_batches=None):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if max_batches and i >= max_batches:
                    break
                    
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = criterion(output, batch.y)
                total_loss += loss.item()
        
        return total_loss / min(len(data_loader), max_batches or len(data_loader))
    
    def predict(self, smiles_list):
        """预测新的SMILES"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 创建临时数据集
        dummy_targets = [0] * len(smiles_list)  # 虚拟目标值
        temp_dataset = MolecularDataset(smiles_list, dummy_targets, self.processor, self.task_type)
        
        if len(temp_dataset) == 0:
            return []
        
        data_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False,
                               collate_fn=lambda x: Batch.from_data_list(x))
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                
                if self.task_type == 'regression':
                    # Handle both single and batch outputs properly
                    output_np = output.cpu().numpy()
                    if output_np.ndim == 0:
                        predictions.append(float(output_np))
                    else:
                        predictions.extend(output_np.flatten())
                else:
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
        
        return predictions
    
    def test_model(self):
        """测试模型性能"""
        if self.test_dataset is None:
            raise ValueError("测试数据集不存在")
        
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False,
                               collate_fn=lambda x: Batch.from_data_list(x))
        
        self.model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                
                if self.task_type == 'regression':
                    # Handle both single and batch outputs properly
                    output_np = output.cpu().numpy()
                    if output_np.ndim == 0:
                        predictions.append(float(output_np))
                    else:
                        predictions.extend(output_np.flatten())
                    
                    true_np = batch.y.cpu().numpy()
                    if true_np.ndim == 0:
                        true_values.append(float(true_np))
                    else:
                        true_values.extend(true_np.flatten())
                else:
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                    true_values.extend(batch.y.cpu().numpy())
        
        # 计算指标
        if self.task_type == 'regression':
            mse = mean_squared_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            print(f"测试集 MSE: {mse:.4f}")
            print(f"测试集 R²: {r2:.4f}")
            
            # 绘制预测vs真实值图
            self.plot_predictions(true_values, predictions)
            
        else:
            accuracy = accuracy_score(true_values, predictions)
            print(f"测试集准确率: {accuracy:.4f}")
            print("\n分类报告:")
            print(classification_report(true_values, predictions))
        
        return predictions, true_values
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练历史')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-50:], label='训练损失 (最后50轮)')
        plt.plot(self.val_losses[-50:], label='验证损失 (最后50轮)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练历史 (最后50轮)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, true_values, predictions):
        """绘制预测vs真实值图"""
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(true_values, predictions, alpha=0.6)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测 vs 真实值')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        residuals = np.array(predictions) - np.array(true_values)
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('残差')
        plt.ylabel('频次')
        plt.title('残差分布')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q图')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def load_data_from_folder(self, folder_path, smiles_col=None, target_col=None):
        """从文件夹中加载所有CSV文件"""
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"在文件夹 {folder_path} 中未找到CSV文件")
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件:")
        for file in csv_files:
            print(f"   - {os.path.basename(file)}")
        
        return csv_files
    
    def process_multiple_datasets(self, folder_path, smiles_col=None, target_col=None, 
                                 n_splits=5, epochs=50, batch_size=32, learning_rate=0.001,
                                 hidden_dim=128, num_layers=3, dropout=0.2):
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

    def _evaluate_with_metrics(self, model, dataset):
        """评估模型并返回完整指标"""
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False,
                               collate_fn=lambda x: Batch.from_data_list(x))
        
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = model(batch)
                
                if self.task_type == 'regression':
                    # Handle both single and batch outputs properly
                    output_np = output.cpu().numpy()
                    if output_np.ndim == 0:
                        predictions.append(float(output_np))
                    else:
                        predictions.extend(output_np.flatten())
                    
                    true_np = batch.y.cpu().numpy()
                    if true_np.ndim == 0:
                        true_values.append(float(true_np))
                    else:
                        true_values.extend(true_np.flatten())
                else:
                    predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
                    true_values.extend(batch.y.cpu().numpy())
        
        # 计算指标
        if self.task_type == 'regression':
            r2 = r2_score(true_values, predictions)
            mse = mean_squared_error(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
            
            metrics = {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'predictions': predictions,
                'true_values': true_values
            }
        else:
            accuracy = accuracy_score(true_values, predictions)
            
            metrics = {
                'accuracy': accuracy,
                'predictions': predictions,
                'true_values': true_values
            }
        
        return metrics
    
    def _save_best_model_only(self, cv_results, temp_model_files):
        """只保存最佳模型，删除其他临时模型"""
        best_fold_idx = cv_results['best_fold_idx']
        best_temp_model = temp_model_files[best_fold_idx]
        
        # 创建最佳模型的最终路径
        best_model_filename = f'{self.csv_filename}_{self.algorithm_name}_best_model.pth'
        best_model_final_path = os.path.join(self.results_dirs['best_models'], best_model_filename)
        
        # 复制最佳模型到最终位置
        import shutil
        shutil.copy2(best_temp_model, best_model_final_path)
        
        # 删除所有临时模型文件
        for temp_file in temp_model_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ 删除临时文件失败 {temp_file}: {str(e)}")
        
        print(f"   💎 最佳模型已保存: {best_model_final_path}")
        print(f"   🗑️ 已清理 {len(temp_model_files)} 个临时模型文件")
        
        return best_model_final_path
    
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

    def batch_process_folder(self, folder_path, n_splits=5, epochs=100, batch_size=32, 
                           learning_rate=0.001, hidden_dim=64, num_layers=3, dropout=0.2):
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

    def _copy_best_model(self, cv_results):
        """复制最佳模型到专门文件夹 - 已被_save_best_model_only替代"""
        # 这个方法已经被_save_best_model_only替代，保留以防兼容性问题
        print("⚠️ _copy_best_model方法已被_save_best_model_only替代")
        pass


def main():
    """主函数 - 改进版本"""
    print("🧬 GCN分子性质预测系统 - 改进版本")
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
    predictor = MolecularPropertyPredictor(task_type=task_type, algorithm_name='GCN')
    
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
                hidden_dim=128,
                num_layers=3,
                dropout=0.2
            )
            
            print(f"\n✅ 处理完成！")
            print(f"   平均得分: {mean_score:.4f} ± {std_score:.4f}")
            print(f"   训练时间: {training_time:.2f} 秒")
            
        except FileNotFoundError:
            print("❌ 未找到数据文件，请确保CSV文件存在")
            print("📋 CSV文件应包含以下列:")
            print("   - SMILES/CSMILES: 分子的SMILES字符串（自动检测）")
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
                hidden_dim=128,
                num_layers=3,
                dropout=0.2
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


if __name__ == "__main__":
    main() 