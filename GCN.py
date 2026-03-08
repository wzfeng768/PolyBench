#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图卷积神经网络分子性质预测模型 - 改进版
使用SMILES字符串预测化学分子的性质
参考RNN代码结构，支持从训练集和测试集文件夹读取数据
支持回归和分类任务，支持五折交叉验证
基于GCN/GAT网络处理分子图结构特征
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
import joblib
import psutil
import GPUtil
from datetime import datetime
from collections import Counter
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


# 路径管理类 - 参考RNN
class PathManager:
    """
    统一管理所有输出路径的类
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.getcwd()
        
        self.paths = {
            'results': os.path.join(self.base_dir, 'Results'),
            'models': os.path.join(self.base_dir, 'Models'),
            'images': os.path.join(self.base_dir, 'Images'),
            'metrics': os.path.join(self.base_dir, 'Metrics'),
            'summaries': os.path.join(self.base_dir, 'Summaries'),
            'predictions': os.path.join(self.base_dir, 'Predictions')
        }
    
    def get_path(self, path_type, method=None, model=None, create=True):
        if path_type not in self.paths:
            raise ValueError(f"Unknown path type: {path_type}")
        
        path = self.paths[path_type]
        
        if model:
            path = os.path.join(path, model)
        if method:
            path = os.path.join(path, method)
            
        if create and not os.path.exists(path):
            os.makedirs(path)
            
        return path
    
    def get_file_path(self, path_type, filename, method=None, model=None, create=True):
        directory = self.get_path(path_type, method, model, create)
        return os.path.join(directory, filename)


def get_base_filename(filename):
    """从文件名中移除扩展名和_train/_test后缀"""
    base_name = os.path.splitext(filename)[0]
    # 移除_train和_test后缀
    if base_name.endswith('_train'):
        base_name = base_name[:-6]  # 移除'_train'
    elif base_name.endswith('_test'):
        base_name = base_name[:-5]  # 移除'_test'
    return base_name


class ProgressManager:
    """进度管理类 - 参考RNN"""
    def __init__(self):
        self.progress_bars = {}
    
    def create_progress_bar(self, name, total, desc="Processing"):
        self.progress_bars[name] = tqdm(
            total=total, 
            desc=desc, 
            position=len(self.progress_bars),
            leave=True,
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'
        )
        return self.progress_bars[name]
    
    def update_progress_bar(self, name, advance=1, postfix=None):
        if name in self.progress_bars:
            self.progress_bars[name].update(advance)
            if postfix:
                self.progress_bars[name].set_postfix(postfix)
    
    def close_progress_bar(self, name):
        if name in self.progress_bars:
            self.progress_bars[name].close()
            del self.progress_bars[name]
    
    def close_all(self):
        for pbar in self.progress_bars.values():
            pbar.close()
        self.progress_bars.clear()


def detect_gpu_availability():
    """检测GPU可用性 - 参考RNN"""
    gpu_info = {
        'has_gpu': False,
        'gpu_count': 0,
        'gpu_memory_total': 0,
        'gpu_memory_free': 0,
        'recommended_task_type': 'CPU',
        'gpu_details': [],
        'selected_gpu_id': 0
    }
    
    try:
        if torch.cuda.is_available():
            gpu_info['has_gpu'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['recommended_task_type'] = 'GPU'
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_detail = {
                    'id': i,
                    'name': props.name,
                    'memory_total': props.total_memory / (1024**2),  # MB
                    'memory_free': props.total_memory / (1024**2),   # 简化处理
                    'memory_used': 0,
                    'utilization': 0
                }
                gpu_info['gpu_details'].append(gpu_detail)
                gpu_info['gpu_memory_total'] += gpu_detail['memory_total']
                gpu_info['gpu_memory_free'] += gpu_detail['memory_free']
                
        print("=== GPU检测结果 ===")
        print(f"GPU可用: {gpu_info['has_gpu']}")
        print(f"GPU数量: {gpu_info['gpu_count']}")
        if gpu_info['has_gpu']:
            print(f"总GPU内存: {gpu_info['gpu_memory_total']:.0f} MB")
            print(f"推荐task_type: {gpu_info['recommended_task_type']}")
            
            for gpu_detail in gpu_info['gpu_details']:
                print(f"GPU {gpu_detail['id']}: {gpu_detail['name']} "
                      f"({gpu_detail['memory_total']:.0f} MB)")
        print("=" * 20)
            
    except Exception as e:
        print(f"GPU检测时出错，将使用CPU: {str(e)}")
        gpu_info['recommended_task_type'] = 'CPU'
    
    return gpu_info


def monitor_system_resources():
    """监控系统资源使用情况 - 参考RNN"""
    resources = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    try:
        if torch.cuda.is_available():
            resources['gpu_info'] = []
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_allocated = torch.cuda.memory_allocated(i)
                resources['gpu_info'].append({
                    'id': i,
                    'utilization': 0,  # 简化处理
                    'memory_used_percent': (gpu_allocated / gpu_memory) * 100,
                    'memory_free_mb': (gpu_memory - gpu_allocated) / (1024**2),
                    'temperature': 0  # 简化处理
                })
    except:
        resources['gpu_info'] = []
    
    return resources


def read_files_in_folder(folder_path, progress_manager=None, progress_name="reading_files"):   
    """读取指定文件夹中的所有CSV文件 - 参考RNN"""
    dataframes = []
    filenames = []
    last_column_names = []

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if progress_manager and csv_files:
        file_pbar = progress_manager.create_progress_bar(
            progress_name, 
            len(csv_files),
            desc="读取文件"
        )

    for i, filename in enumerate(csv_files):
        file_path = os.path.join(folder_path, filename)
        
        if progress_manager:
            progress_manager.update_progress_bar(
                progress_name, 
                1 if i > 0 else 0,
                {'当前': filename[:20] + '...' if len(filename) > 20 else filename}
            )
        
        print(f"读取文件: {filename}")
        
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
            filenames.append(filename)
            
            last_column_name = df.columns[-1]
            if last_column_name == 'property_log':
                last_column_name = df.columns[-2]
            last_column_names.append(last_column_name)
            print(f"文件 {filename} 的最后一列列名为: {last_column_name}")
            
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {str(e)}")
            continue
    
    if progress_manager and csv_files:
        progress_manager.update_progress_bar(progress_name, 1, {'完成': f'{len(dataframes)}个文件'})
        progress_manager.close_progress_bar(progress_name)

    return dataframes, filenames, last_column_names


def prepare_data_from_df(data, smiles_col='SMILES'):    
    """从DataFrame分离SMILES和目标变量 - 参考RNN的prepare_data"""
    # 自动检测SMILES列
    possible_smiles_cols = ['SMILES', 'CSMILES', 'smiles', 'csmiles', 'Smiles', 'CSmiles']
    
    actual_smiles_col = None
    for col in possible_smiles_cols:
        if col in data.columns:
            actual_smiles_col = col
            break
    
    if actual_smiles_col is None:
        raise ValueError(f"未找到SMILES列。可用列: {list(data.columns)}")
    
    # 获取目标列
    if data.columns[-1] == 'property_log':
        target_col = data.columns[-2]
        y = data[data.columns[-1]]  # 使用property_log作为目标
    else:
        target_col = data.columns[-1]
        y = data[data.columns[-1]]

    smiles_data = data[actual_smiles_col]
    
    return smiles_data, y, actual_smiles_col, target_col


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
            return [0] * 12  # 返回默认值，调整数量与RNN一致
        
        try:
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
                rdMolDescriptors.CalcNumSpiroAtoms(mol),
                Descriptors.NumHeteroatoms(mol),
                Descriptors.RingCount(mol)
            ]
        except Exception as e:
            print(f"计算描述符失败: {str(e)}")
            return [0] * 12
        
        return descriptors


class MolecularDataset(Dataset):
    """分子数据集类（仅支持回归任务）"""
    
    def __init__(self, smiles_list, targets, processor, task_type='regression'):
        self.smiles_list = smiles_list
        self.targets = targets
        self.processor = processor
        self.task_type = 'regression'  # 固定为回归任务
        
        # 预处理所有分子
        self.graphs = []
        self.descriptors = []
        self.valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                graph = processor.smiles_to_graph(smiles)
                if graph is not None:
                    self.graphs.append(graph)
                    self.descriptors.append(processor.calculate_molecular_descriptors(smiles))
                    self.valid_indices.append(i)
            except Exception as e:
                print(f"处理SMILES失败: {smiles}, 错误: {str(e)}")
                continue
        
        # 过滤有效的目标值
        self.targets = [targets[i] for i in self.valid_indices]
        
        print(f"成功处理 {len(self.graphs)} / {len(smiles_list)} 个分子")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        descriptors = torch.tensor(self.descriptors[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float)  # 回归任务使用float
        
        # 将描述符添加到图数据中
        graph.descriptors = descriptors
        graph.y = target
        
        return graph


class MolecularGCN(nn.Module):
    """分子图卷积神经网络"""
    
    def __init__(self, atom_features_dim=9, descriptor_dim=12, hidden_dim=128, num_layers=3, 
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
        
        # 回归任务的预测层
        self.predictor = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
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
        
        # 对于回归任务，确保输出保持正确的形状
        output = output.squeeze(-1)  # 只压缩最后一个维度，保持batch维度
        return output


class MolecularPropertyPredictor:
    """分子性质预测器主类 - 改进版，参考RNN结构（仅支持回归任务）"""
    
    def __init__(self, device=None, algorithm_name='GCN', base_dir=None):
        self.task_type = 'regression'  # 固定为回归任务
        self.algorithm_name = algorithm_name
        
        # GPU检测
        self.gpu_info = detect_gpu_availability()
        
        # 设备设置
        if device is None:
            if self.gpu_info['has_gpu']:
                self.device = torch.device('cuda')
                print(f"🚀 使用GPU: {self.gpu_info['gpu_details'][0]['name']}")
            else:
                self.device = torch.device('cpu')
                print("💻 使用CPU进行计算")
        else:
            self.device = device
            
        self.processor = MolecularDataProcessor()
        self.model = None
        
        # 创建路径管理器 - 参考RNN
        self.path_manager = PathManager(base_dir)
        
        print(f"🔧 使用设备: {self.device}")
        print(f"🏷️ 算法名称: {self.algorithm_name}")
        print(f"📁 结果保存目录: {self.path_manager.base_dir}")
        
        # 设置CUDA优化（如果使用GPU）
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def load_data(self, csv_file, smiles_col=None, target_col=None, test_size=0.2, random_state=42):
        """加载CSV数据，自动识别SMILES列和目标列"""
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
        
        # 如果未指定目标列，使用最后一列
        if target_col is None:
            target_col = df.columns[-1]
            print(f"🎯 自动识别目标列: {target_col}")
        else:
            print(f"📋 使用指定的目标列: {target_col}")
        
        # 保存目标列名用于可视化
        self.target_column_name = target_col
        
        # 检查最后一列是否为property_log，如果是，则使用倒数第二列的列名作为散点图标题
        if target_col.lower() == 'property_log' and len(df.columns) >= 2:
            self.scatter_plot_target_name = df.columns[-2]
            print(f"🔍 检测到property_log列，散点图将使用列名: {self.scatter_plot_target_name}")
        else:
            self.scatter_plot_target_name = target_col
        
        # 检查必要的列
        if smiles_col not in df.columns:
            raise ValueError(f"未找到SMILES列: {smiles_col}。可用列: {list(df.columns)}")
        if target_col not in df.columns:
            raise ValueError(f"未找到目标列: {target_col}。可用列: {list(df.columns)}")
        
        # 保存SMILES列名用于后续使用
        self.smiles_column_name = smiles_col
        
        # 移除缺失值
        df = df.dropna(subset=[smiles_col, target_col])
        print(f"移除缺失值后数据形状: {df.shape}")
        
        smiles_list = df[smiles_col].tolist()
        targets = df[target_col].tolist()
        
        # 回归任务处理
        self.num_classes = 1
        print(f"回归任务，目标值范围: {min(targets):.3f} - {max(targets):.3f}")
        
        # 保存完整数据用于交叉验证
        self.full_smiles = smiles_list
        self.full_targets = targets
        
        # 划分训练集和测试集
        train_smiles, test_smiles, train_targets, test_targets = train_test_split(
            smiles_list, targets, test_size=test_size, random_state=random_state
        )
        
        # 创建数据集
        self.train_dataset = MolecularDataset(train_smiles, train_targets, self.processor, self.task_type)
        self.test_dataset = MolecularDataset(test_smiles, test_targets, self.processor, self.task_type)
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"测试集大小: {len(self.test_dataset)}")
        
        return self.train_dataset, self.test_dataset 

    def load_data_from_folders(self, train_folder, test_folder, progress_manager=None):
        """从训练集和测试集文件夹加载数据 - 参考RNN"""
        if not os.path.exists(train_folder):
            raise ValueError(f"训练数据文件夹不存在: {train_folder}")
        
        if not os.path.exists(test_folder):
            raise ValueError(f"测试数据文件夹不存在: {test_folder}")
        
        # 读取训练和测试数据
        print("正在读取训练数据...")
        train_data_list, train_file_names, train_last_column_names = read_files_in_folder(
            train_folder, progress_manager, "train_read"
        )
        
        print("正在读取测试数据...")
        test_data_list, test_file_names, test_last_column_names = read_files_in_folder(
            test_folder, progress_manager, "test_read"
        )
        
        if not train_data_list:
            raise ValueError(f"训练文件夹中没有找到CSV文件: {train_folder}")
        
        if not test_data_list:
            raise ValueError(f"测试文件夹中没有找到CSV文件: {test_folder}")
        
        # 创建文件名到数据的映射
        train_data_dict = {get_base_filename(name): (data, col_name) 
                           for data, name, col_name in zip(train_data_list, train_file_names, train_last_column_names)}
        test_data_dict = {get_base_filename(name): (data, col_name) 
                          for data, name, col_name in zip(test_data_list, test_file_names, test_last_column_names)}
        
        # 找到训练和测试数据的共同文件
        common_files = set(train_data_dict.keys()) & set(test_data_dict.keys())
        
        if not common_files:
            raise ValueError("训练和测试数据没有共同的文件")
        
        print(f"找到 {len(common_files)} 个共同文件进行处理: {list(common_files)}")
        
        return train_data_dict, test_data_dict, common_files
    
    def create_model(self, hidden_dim=128, num_layers=3, dropout=0.2):
        """创建模型"""
        # 获取描述符维度
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            sample_descriptors = self.train_dataset[0].descriptors
            descriptor_dim = len(sample_descriptors)
        else:
            descriptor_dim = 12  # 默认值
        
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
                      weight_decay=1e-4, patience=10, hidden_dim=128, num_layers=3, dropout=0.2, naming_style='simple'):
        """五折交叉验证 - 简化版本，只保留必要的训练和评估"""
        print(f"\n🔄 开始 {n_splits} 折交叉验证...")
        print(f"🖥️  使用设备: {self.device}")
        
        if not hasattr(self, 'full_smiles'):
            raise ValueError("请先调用 load_data() 加载数据")
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用K折交叉验证（回归任务）
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kfold.split(self.full_smiles))
        
        cv_results = {
            'fold_scores': [],
            'fold_models': [],
            'best_fold_idx': 0,
            'best_score': -float('inf') if self.task_type == 'regression' else 0
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
                descriptor_dim = 12
                
            model = MolecularGCN(
                atom_features_dim=self.processor.atom_features_dim,
                descriptor_dim=descriptor_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task_type,
                num_classes=self.num_classes
            ).to(self.device)
            
            # 训练模型
            temp_model_path = self.path_manager.get_file_path('models', 
                                         f'temp_{self.algorithm_name}_fold_{fold+1}.pth', create=True)
            train_losses, val_losses = self._train_single_fold(
                model, train_dataset, val_dataset, epochs, batch_size, 
                learning_rate, weight_decay, patience, temp_model_path
            )
            
            temp_model_files.append(temp_model_path)
            
            # 简单评估
            val_metrics = self._evaluate_with_metrics(model, val_dataset)
            
            cv_results['fold_models'].append(temp_model_path)
            
            # 使用验证集R²得分
            fold_score = val_metrics['r2']
            
            cv_results['fold_scores'].append(fold_score)
            
            # 记录最佳折叠
            if fold_score > cv_results['best_score']:
                cv_results['best_score'] = fold_score
                cv_results['best_fold_idx'] = fold
            
            print(f"   第 {fold + 1} 折验证得分: {fold_score:.4f}")
        
        # 计算总训练时间
        total_training_time = time.time() - start_time
        
        # 计算交叉验证统计
        mean_score = np.mean(cv_results['fold_scores'])
        std_score = np.std(cv_results['fold_scores'])
        
        print(f"\n📈 交叉验证结果:")
        print(f"   平均得分: {mean_score:.4f} ± {std_score:.4f}")
        print(f"   训练时间: {total_training_time:.2f} 秒")
        print(f"   最佳折叠: 第 {cv_results['best_fold_idx'] + 1} 折 (得分: {cv_results['best_score']:.4f})")
        
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
        # 加载最佳模型
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=self.device))
        return train_losses, val_losses
    
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

    def evaluate_model_on_datasets(self, model, train_dataset, test_dataset):
        """评估模型在训练集和测试集上的性能 - 参考RNN的evaluate_model"""
        train_metrics = self._evaluate_with_metrics(model, train_dataset)
        test_metrics = self._evaluate_with_metrics(model, test_dataset)
        
        # 组织返回格式，匹配RNN（回归任务）
        metrics = {
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_mse': train_metrics['mse'],
            'test_mse': test_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae']
        }
        
        predictions = {
            'train': {'true': train_metrics['true_values'], 'pred': train_metrics['predictions']},
            'test': {'true': test_metrics['true_values'], 'pred': test_metrics['predictions']}
        }
        
        return metrics, predictions

    def save_predictions(self, predictions, method, model_name, file_name):
        """保存训练集和测试集的真实值和预测值 - 参考RNN"""
        for data_type in ['train', 'test']:
            pred_df = pd.DataFrame({
                'true_values': predictions[data_type]['true'],
                'predicted_values': predictions[data_type]['pred']
            })
            
            pred_file = f"{method}_{model_name}_{file_name}_{data_type}_predictions.csv"
            pred_path = self.path_manager.get_file_path('predictions', pred_file, method, model_name)
            
            pred_df.to_csv(pred_path, index=False)
            print(f"{data_type.capitalize()} 预测结果保存至: {pred_path}")

    def save_metrics(self, metrics, cv_scores, best_params, training_time, training_resources, method, model_name, file_name):
        """保存评估指标 - 参考RNN"""
        metrics_dict = {
            'file_name': [file_name],
            'method': [method],
            'model': [model_name],
            'train_r2': [metrics['train_r2']],
            'train_mse': [metrics['train_mse']],
            'train_mae': [metrics['train_mae']],
            'test_r2': [metrics['test_r2']],
            'test_mse': [metrics['test_mse']],
            'test_mae': [metrics['test_mae']],
            'cv_mean_score': [cv_scores.get('mean_test_score', 'N/A')],
            'cv_std_score': [cv_scores.get('std_test_score', 'N/A')],
            'best_epochs': [best_params.get('epochs', 'N/A')],
            'best_learning_rate': [best_params.get('learning_rate', 'N/A')],
            'best_hidden_dim': [best_params.get('hidden_dim', 'N/A')],
            'training_time': [training_time],
            'max_memory_used_percent': [training_resources.get('max_memory_used', 'N/A')],
            'device_type': [str(self.device)]
        }
        
        # 添加GPU相关信息
        if training_resources.get('start_resources', {}).get('gpu_info'):
            gpu_start = training_resources['start_resources']['gpu_info'][0]
            gpu_end = training_resources['end_resources']['gpu_info'][0]
            metrics_dict.update({
                'gpu_utilization_start': [gpu_start['utilization']],
                'gpu_utilization_end': [gpu_end['utilization']],
                'gpu_memory_used_start': [gpu_start['memory_used_percent']],
                'gpu_memory_used_end': [gpu_end['memory_used_percent']],
                'gpu_temperature_max': [max(gpu_start['temperature'], gpu_end['temperature'])]
            })
        else:
            metrics_dict.update({
                'gpu_utilization_start': ['N/A'],
                'gpu_utilization_end': ['N/A'],
                'gpu_memory_used_start': ['N/A'],
                'gpu_memory_used_end': ['N/A'],
                'gpu_temperature_max': ['N/A']
            })
        
        metrics_df = pd.DataFrame(metrics_dict)
        
        metrics_file = f"{method}_{model_name}_{file_name}_metrics.csv"
        metrics_path = self.path_manager.get_file_path('metrics', metrics_file, method, model_name)
        
        metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
        print(f"评估指标保存至: {metrics_path}")
        
        # 显示资源使用摘要
        print(f"资源使用摘要:")
        print(f"  - 最大内存使用: {training_resources.get('max_memory_used', 'N/A')}%")
        if training_resources.get('start_resources', {}).get('gpu_info'):
            gpu_info = training_resources['end_resources']['gpu_info'][0]
            print(f"  - GPU最终使用率: {gpu_info['utilization']:.1f}%")
            print(f"  - GPU最终显存使用: {gpu_info['memory_used_percent']:.1f}%")
            print(f"  - GPU最终温度: {gpu_info['temperature']}°C")
        
        return metrics_df

    def plot_and_save_scatter(self, predictions, metrics, method, model_name, file_name, target_column_name):
        """绘制并保存预测值与真实值的散点图 - 参考RNN"""
        y_train = np.array(predictions['train']['true'])
        y_train_pred = np.array(predictions['train']['pred'])
        y_test = np.array(predictions['test']['true'])
        y_test_pred = np.array(predictions['test']['pred'])

        plt.figure(figsize=(8, 8))

        plt.scatter(y_train, y_train_pred, s=50, c='#005BAD', alpha=0.7, label='Train')
        plt.scatter(y_test, y_test_pred, s=50, c='#F56476', alpha=0.7, label='Test')

        all_true = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        min_val = min(all_true.min(), all_pred.min())
        max_val = max(all_true.max(), all_pred.max())
        buffer = (max_val - min_val) * 0.02

        plt.xlim(min_val - buffer, max_val + buffer)
        plt.ylim(min_val - buffer, max_val + buffer)

        plt.plot([min_val, max_val], [min_val, max_val], '--', color='grey', label='ideal')

        # 处理标题
        match = re.match(r'^(.*?)[\[\(].*?[\]\)]$', target_column_name)
        title_name = match.group(1).strip() if match else target_column_name

        plt.text(0.5, 0.95, f"{model_name}_{method}_{title_name}", 
                 transform=plt.gca().transAxes, fontsize=18,
                 verticalalignment='top', horizontalalignment='center')

        plt.tick_params(which='major', direction='in', length=5, labelsize=16)
        plt.xlabel(f"Actual {target_column_name}", fontsize=16)
        plt.ylabel(f"Predicted {target_column_name}", fontsize=16)
        plt.legend(loc='upper left', fontsize=16)

        metrics_text = (
            r"$R^2_{\mathrm{train}}$: " + f"{metrics['train_r2']:.3f}" + "\n" +
            r"$\mathrm{MSE}_{\mathrm{train}}$: " + f"{metrics['train_mse']:.3f}" + "\n" +
            r"$\mathrm{MAE}_{\mathrm{train}}$: " + f"{metrics['train_mae']:.3f}" + "\n" +
            r"$R^2_{\mathrm{test}}$: " + f"{metrics['test_r2']:.3f}" + "\n" +
            r"$\mathrm{MSE}_{\mathrm{test}}$: " + f"{metrics['test_mse']:.3f}" + "\n" +
            r"$\mathrm{MAE}_{\mathrm{test}}$: " + f"{metrics['test_mae']:.3f}"
        )

        plt.text(0.65, 0.05, metrics_text,
                 transform=plt.gca().transAxes, fontsize=16,
                 verticalalignment='bottom', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_aspect('equal', adjustable='box')

        image_file = f"{method}_{model_name}_{file_name}_scatter.png"
        image_path = self.path_manager.get_file_path('images', image_file, method, model_name)
        
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"散点图保存至: {image_path}")

    def save_summary_metrics(self, all_metrics, timestamp=None):
        """保存所有文件的汇总评估指标 - 参考RNN"""
        if not all_metrics:
            print("警告: 没有指标可保存")
            return None
        
        summary_df = pd.concat(all_metrics, ignore_index=True)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_file = f'{self.algorithm_name}_Train_Test_Results_{timestamp}.csv'
        summary_path = self.path_manager.get_file_path('summaries', summary_file, create=True)
        
        summary_df.to_csv(summary_path, index=False, float_format='%.6f')
        print(f"汇总评估指标保存至: {summary_path}")
        
        return summary_df

    def process_single_file(self, train_data, test_data, file_name, target_column_name, 
                           progress_manager=None, method="GCN", model_name=None, 
                           epochs=100, batch_size=32, learning_rate=0.001, 
                           hidden_dim=128, num_layers=3, dropout=0.2):
        """处理单个文件的训练和测试 - 参考RNN的process_method_data中的单文件处理逻辑"""
        
        if model_name is None:
            model_name = self.algorithm_name
        
        print(f"\n处理文件: {file_name}")
        print("-" * 60)
        
        try:
            # 准备训练数据
            train_smiles, train_targets, smiles_col, _ = prepare_data_from_df(train_data)
            
            # 准备测试数据
            test_smiles, test_targets, _, _ = prepare_data_from_df(test_data)
            
            print(f"数据维度 - 训练集: {len(train_smiles)}, 测试集: {len(test_smiles)}")
            
            # 回归任务处理
            self.num_classes = 1
            print(f"回归任务，目标值范围: {min(train_targets):.3f} - {max(train_targets):.3f}")
            
            # 创建数据集
            train_dataset = MolecularDataset(train_smiles.tolist(), train_targets.tolist(), 
                                           self.processor, self.task_type)
            test_dataset = MolecularDataset(test_smiles.tolist(), test_targets.tolist(), 
                                          self.processor, self.task_type)
            
            start_time = time.time()
            
            # 记录训练开始前的资源状态
            start_resources = monitor_system_resources()
            print(f"训练开始 - CPU: {start_resources['cpu_percent']:.1f}%, "
                  f"内存: {start_resources['memory_percent']:.1f}%")
            
            # 使用5折交叉验证训练模型（基于训练集）
            print("开始5折交叉验证训练...")
            
            # 保存完整数据用于交叉验证
            self.full_smiles = train_smiles.tolist()
            self.full_targets = train_targets.tolist()
            
            # 执行交叉验证（只在训练集上）
            cv_results, mean_score, std_score, cv_training_time = self.cross_validate(
                n_splits=5,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                naming_style='simple'
            )
            
            # 获取最佳模型（从交叉验证结果中）
            best_fold_idx = cv_results['best_fold_idx']
            best_model_path = cv_results['fold_models'][best_fold_idx]
            
            # 重新创建模型并加载最佳权重
            descriptor_dim = len(train_dataset[0].descriptors)
            best_model = MolecularGCN(
                atom_features_dim=self.processor.atom_features_dim,
                descriptor_dim=descriptor_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task_type,
                num_classes=self.num_classes
            ).to(self.device)
            
            best_model.load_state_dict(torch.load(best_model_path))
            
            training_time = time.time() - start_time
            print(f"训练完成，总耗时: {training_time:.2f} 秒")
            
            # 评估模型（在训练集和测试集上）
            print("正在评估模型性能...")
            metrics, predictions = self.evaluate_model_on_datasets(best_model, train_dataset, test_dataset)
            
            # 记录训练结束后的资源状态
            end_resources = monitor_system_resources()
            training_resources = {
                'start_resources': start_resources,
                'end_resources': end_resources,
                'max_memory_used': max(start_resources['memory_percent'], end_resources['memory_percent'])
            }
            
            # 保存最佳模型
            model_file = f"{method}_{model_name}_{file_name}_best_model.pth"
            model_path = self.path_manager.get_file_path('models', model_file, method, model_name)
            torch.save(best_model.state_dict(), model_path)
            print(f"最佳模型保存至: {model_path}")
            
            # 保存预测结果
            self.save_predictions(predictions, method, model_name, file_name)
            
            # 保存评估指标
            cv_scores = {
                'mean_test_score': mean_score,
                'std_test_score': std_score
            }
            best_params = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'hidden_dim': hidden_dim
            }
            
            metrics_df = self.save_metrics(
                metrics, cv_scores, best_params, training_time, training_resources,
                method, model_name, file_name
            )
            
            # 绘制和保存散点图
            print("正在生成可视化图表...")
            self.plot_and_save_scatter(
                predictions, metrics, method, 
                model_name, file_name, target_column_name
            )
            
            print(f"文件 {file_name} 处理完成")
            print("=" * 60)
            
            return metrics_df
            
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def process_folder_data(self, train_folder, test_folder, progress_manager=None, 
                           method="GCN", model_name=None, epochs=100, batch_size=32, 
                           learning_rate=0.001, hidden_dim=128, num_layers=3, dropout=0.2):
        """处理文件夹中的所有数据 - 参考RNN的process_method_data"""
        
        if model_name is None:
            model_name = self.algorithm_name
            
        method_metrics = []
        
        try:
            # 加载数据
            train_data_dict, test_data_dict, common_files = self.load_data_from_folders(
                train_folder, test_folder, progress_manager
            )
            
            # 创建文件处理进度条
            if progress_manager:
                file_progress = progress_manager.create_progress_bar(
                    f"{method}_files",
                    len(common_files),
                    desc=f"{method} 文件处理"
                )
            
            for i, file_name in enumerate(sorted(common_files)):
                # 更新文件处理进度
                if progress_manager:
                    progress_manager.update_progress_bar(
                        f"{method}_files",
                        1 if i > 0 else 0,
                        {'当前文件': file_name[:15] + '...' if len(file_name) > 15 else file_name}
                    )
                
                # 获取训练和测试数据
                train_data, train_col_name = train_data_dict[file_name]
                test_data, test_col_name = test_data_dict[file_name]
                
                # 处理单个文件
                metrics_df = self.process_single_file(
                    train_data, test_data, file_name, train_col_name,
                    progress_manager, method, model_name, epochs, batch_size,
                    learning_rate, hidden_dim, num_layers, dropout
                )
                
                if metrics_df is not None:
                    method_metrics.append(metrics_df)
            
            # 完成文件处理进度条
            if progress_manager:
                progress_manager.update_progress_bar(f"{method}_files", 1, {'状态': '全部完成'})
                progress_manager.close_progress_bar(f"{method}_files")
            
            return method_metrics
            
        except Exception as e:
            print(f"处理文件夹数据时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return method_metrics
    
    def _save_cv_results(self, cv_results, mean_score, std_score, training_time, avg_train_metrics=None, avg_test_metrics=None):
        """保存交叉验证结果 - 简化版本"""
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
        json_path = os.path.join(self.path_manager.get_path('results'), json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存:")
        print(f"   详细结果: {json_path}")
        print(f"   �� 摘要已更新到全局文件: {self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv')}")
        
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
        plot_path = os.path.join(self.path_manager.get_path('images'), plot_filename)
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
        image_path = os.path.join(self.path_manager.get_path('images'), image_filename)
        
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
            save_path = os.path.join(self.path_manager.get_path('models'), f'{self.csv_filename}_{self.algorithm_name}_model_{timestamp}.pth')
        
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
        best_model_final_path = os.path.join(self.path_manager.get_path('models'), best_model_filename)
        
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
        if os.path.exists(self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv')):
            try:
                existing_df = pd.read_csv(self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv'))
                
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
        existing_df.to_csv(self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv'), index=False)
        
        print(f"   💾 全局摘要已更新: {self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv')}")
        print(f"   📊 当前包含 {len(existing_df)} 个数据集的结果")

    def _save_overall_summary(self, all_results):
        """保存所有数据集的总体摘要 - 已被全局摘要文件替代"""
        # 这个方法已经被全局摘要文件机制替代
        print(f"\n📊 所有结果已保存在全局摘要文件: {self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv')}")
        
        # 读取并显示当前摘要统计
        if os.path.exists(self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv')):
            try:
                summary_df = pd.read_csv(self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv'))
                print(f"   📋 总计处理了 {len(summary_df)} 个数据集")
                
                if self.task_type == 'regression':
                    avg_test_r2 = summary_df['Test_R2'].mean()
                    print(f"   📈 平均测试集R²: {avg_test_r2:.4f}")
                else:
                    avg_test_acc = summary_df['Test_Accuracy'].mean()
                    print(f"   📈 平均测试集准确率: {avg_test_acc:.4f}")
                    
            except Exception as e:
                print(f"⚠️ 读取摘要统计失败: {str(e)}")
        
        return self.path_manager.get_file_path('summaries', f'global_summary_{self.algorithm_name}.csv')

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
    """主函数 - 改进版，参考RNN的处理流程"""
    print("=" * 80)
    print("🧬 GCN分子性质预测系统 - 改进版 (参考RNN结构)")
    print("=" * 80)
    
    # 创建进度管理器
    progress_manager = ProgressManager()
    
    try:
        # 检测GPU可用性
        gpu_info = detect_gpu_availability()
        
        # 固定为回归任务
        task_type = 'regression'
        algorithm_name = 'GCN'
        print(f"\n📊 任务类型: 回归任务 (regression)")
        
        # 定义基础路径
        base_dir = os.getcwd()
        
        # 创建预测器
        predictor = MolecularPropertyPredictor(algorithm_name=algorithm_name, base_dir=base_dir)
        
        # 定义数据文件夹路径 - 参考RNN
        train_base = os.path.join(base_dir, 'train')
        test_base = os.path.join(base_dir, 'test')
        
        if not os.path.exists(train_base) or not os.path.exists(test_base):
            print(f"❌ 未找到训练或测试数据文件夹:")
            print(f"   训练文件夹: {train_base}")
            print(f"   测试文件夹: {test_base}")
            print("请确保数据文件夹存在并包含CSV文件")
            return
        
        # 用于存储所有指标的列表
        all_metrics = []
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 显示系统信息
        print(f"\n📊 系统信息:")
        print(f"CPU核心数: {psutil.cpu_count()}")
        print(f"总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"使用设备: {predictor.device}")
        
        print(f"\n{'='*80}")
        print(f"🔬 开始处理数据...")
        print(f"   训练文件夹: {train_base}")
        print(f"   测试文件夹: {test_base}")
        print(f"   算法: {algorithm_name}")
        print(f"   任务类型: {task_type}")
        print(f"{'='*80}")
        
        try:
            # 处理数据 - 参考RNN的处理方式
            method_metrics = predictor.process_folder_data(
                train_base, test_base, 
                progress_manager, method=algorithm_name, model_name=algorithm_name,
                epochs=100, batch_size=32, learning_rate=0.001,
                hidden_dim=128, num_layers=3, dropout=0.2
            )
            
            # 将指标添加到总列表中
            all_metrics.extend(method_metrics)
            
            print(f"✅ 完成数据处理, 共处理 {len(method_metrics)} 个文件")
            
            # 显示当前系统资源状态
            current_resources = monitor_system_resources()
            print(f"📈 当前系统状态 - CPU: {current_resources['cpu_percent']:.1f}%, "
                  f"内存: {current_resources['memory_percent']:.1f}%")
            
        except Exception as e:
            print(f"❌ 处理数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 保存汇总结果
        if all_metrics:
            summary_df = predictor.save_summary_metrics(all_metrics, timestamp)
            print(f"\n✅ 汇总结果已保存")
            
            # 统计信息
            print(f"\n📊 处理统计:")
            print(f"总文件数: {len(all_metrics)}")
            
            avg_test_r2 = np.mean([df['test_r2'].iloc[0] for df in all_metrics if not df.empty])
            print(f"平均测试集R²: {avg_test_r2:.4f}")
        else:
            print("⚠️ 警告: 没有成功处理任何数据，未保存汇总评估指标")
        
        print(f"\n🎉 所有处理完成！")
        print(f"📁 结果保存在: {predictor.path_manager.base_dir}")
        print("=" * 80)
        
        return {
            'all_metrics': all_metrics,
            'timestamp': timestamp,
            'gpu_info': gpu_info,
            'algorithm': algorithm_name
        }
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断训练过程")
        return None
        
    except Exception as e:
        print(f"\n\n❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # 确保所有进度条都被关闭
        progress_manager.close_all()
        print("🔧 进度条资源已清理")


if __name__ == "__main__":
    main() 