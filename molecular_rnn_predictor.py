#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环神经网络分子性质预测模型 - 改进版
使用SMILES字符串预测化学分子的性质
参考CatBoost代码结构，支持从训练集和测试集文件夹读取数据
支持回归和分类任务，支持五折交叉验证
基于RNN/LSTM/GRU网络处理SMILES序列特征
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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


# 路径管理类 - 参考CatBoost
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
    """进度管理类 - 参考CatBoost"""
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
    """检测GPU可用性 - 参考CatBoost"""
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
    """监控系统资源使用情况 - 参考CatBoost"""
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
    """读取指定文件夹中的所有CSV文件 - 参考CatBoost"""
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
    """从DataFrame分离SMILES和目标变量 - 参考CatBoost的prepare_data"""
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


class SMILESProcessor:
    """SMILES字符串处理器"""
    
    def __init__(self, max_length=200):
        self.max_length = max_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.scaler = StandardScaler()
        
    def build_vocabulary(self, smiles_list):
        """构建SMILES字符词汇表"""
        all_chars = set()
        for smiles in smiles_list:
            all_chars.update(list(smiles))
        
        # 添加特殊字符
        special_chars = ['<PAD>', '<UNK>', '<START>', '<END>']
        all_chars = special_chars + sorted(list(all_chars))
        
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(all_chars)
        
        print(f"构建词汇表完成，词汇量: {self.vocab_size}")
        print(f"词汇表: {list(self.char_to_idx.keys())}")
        
    def smiles_to_sequence(self, smiles):
        """将SMILES字符串转换为数字序列"""
        if not hasattr(self, 'char_to_idx') or not self.char_to_idx:
            raise ValueError("词汇表未构建，请先调用build_vocabulary")
        
        # 添加开始和结束标记
        smiles = '<START>' + smiles + '<END>'
        
        sequence = []
        i = 0
        while i < len(smiles):
            if i < len(smiles) - 1:
                # 尝试匹配两字符的原子符号（如Cl, Br等）
                two_char = smiles[i:i+2]
                if two_char in self.char_to_idx:
                    sequence.append(self.char_to_idx[two_char])
                    i += 2
                    continue
            
            # 单字符匹配
            char = smiles[i]
            if char in self.char_to_idx:
                sequence.append(self.char_to_idx[char])
            else:
                sequence.append(self.char_to_idx['<UNK>'])
            i += 1
        
        # 填充或截断到固定长度
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence.extend([self.char_to_idx['<PAD>']] * (self.max_length - len(sequence)))
        
        return sequence
    
    def calculate_molecular_descriptors(self, smiles):
        """计算分子描述符"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 12  # 返回默认值，调整数量
        
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
        self.sequences = []
        self.descriptors = []
        self.valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                sequence = processor.smiles_to_sequence(smiles)
                descriptors = processor.calculate_molecular_descriptors(smiles)
                
                self.sequences.append(sequence)
                self.descriptors.append(descriptors)
                self.valid_indices.append(i)
            except Exception as e:
                print(f"处理SMILES失败: {smiles}, 错误: {str(e)}")
                continue
        
        # 过滤有效的目标值
        self.targets = [targets[i] for i in self.valid_indices]
        
        print(f"成功处理 {len(self.sequences)} / {len(smiles_list)} 个分子")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        descriptors = torch.tensor(self.descriptors[idx], dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float)  # 回归任务使用float
        
        return {
            'sequence': sequence,
            'descriptors': descriptors,
            'target': target
        }

class MolecularRNN(nn.Module):
    """分子RNN神经网络"""
    
    def __init__(self, vocab_size, descriptor_dim=12, embedding_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.2, task_type='regression', num_classes=1, 
                 rnn_type='LSTM'):
        super(MolecularRNN, self).__init__()
        
        self.task_type = task_type
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN层
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                              bidirectional=True)
            rnn_output_dim = hidden_dim * 2  # 双向
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                             bidirectional=True)
            rnn_output_dim = hidden_dim * 2  # 双向
        else:  # RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                             bidirectional=True)
            rnn_output_dim = hidden_dim * 2  # 双向
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(rnn_output_dim, num_heads=8, dropout=dropout)
        
        # 分子描述符处理
        self.descriptor_fc = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # 最终预测层
        final_dim = rnn_output_dim + hidden_dim // 4  # RNN特征 + 描述符特征
        
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
        sequences = batch['sequence']  # [batch_size, seq_len]
        descriptors = batch['descriptors']  # [batch_size, descriptor_dim]
        
        # 嵌入
        embedded = self.embedding(sequences)  # [batch_size, seq_len, embedding_dim]
        
        # RNN处理
        rnn_output, _ = self.rnn(embedded)  # [batch_size, seq_len, hidden_dim*2]
        
        # 注意力机制 - 转换维度用于MultiheadAttention
        rnn_output_transposed = rnn_output.transpose(0, 1)  # [seq_len, batch_size, hidden_dim*2]
        attended_output, _ = self.attention(rnn_output_transposed, rnn_output_transposed, rnn_output_transposed)
        attended_output = attended_output.transpose(0, 1)  # [batch_size, seq_len, hidden_dim*2]
        
        # 全局池化 - 使用最后一个时间步的输出和平均池化的组合
        last_output = attended_output[:, -1, :]  # [batch_size, hidden_dim*2]
        avg_output = torch.mean(attended_output, dim=1)  # [batch_size, hidden_dim*2]
        sequence_features = (last_output + avg_output) / 2  # 组合特征
        
        # 处理分子描述符
        descriptor_features = self.descriptor_fc(descriptors)
        
        # 特征融合
        combined_features = torch.cat([sequence_features, descriptor_features], dim=1)
        
        # 最终预测
        output = self.predictor(combined_features)
        
        # 对于回归任务，确保输出保持正确的形状
        output = output.squeeze(-1)  # 只压缩最后一个维度，保持batch维度
        return output

class MolecularPropertyPredictor:
    """分子性质预测器主类 - 改进版，参考CatBoost结构（仅支持回归任务）"""
    
    def __init__(self, device=None, algorithm_name='RNN', base_dir=None):
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
            
        self.processor = SMILESProcessor()
        self.model = None
        
        # 创建路径管理器 - 参考CatBoost
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
        
        # 构建词汇表
        self.processor.build_vocabulary(smiles_list)
        
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
        """从训练集和测试集文件夹加载数据 - 参考CatBoost"""
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

    def create_model(self, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2, rnn_type='LSTM'):
        """创建模型"""
        # 获取描述符维度
        if hasattr(self, 'train_dataset') and len(self.train_dataset) > 0:
            sample_descriptors = self.train_dataset[0]['descriptors']
            descriptor_dim = len(sample_descriptors)
        else:
            descriptor_dim = 12  # 调整为12
        
        self.model = MolecularRNN(
            vocab_size=self.processor.vocab_size,
            descriptor_dim=descriptor_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            task_type=self.task_type,
            num_classes=self.num_classes,
            rnn_type=rnn_type
        ).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"RNN类型: {rnn_type}")
        return self.model
    
    def cross_validate(self, n_splits=5, epochs=100, batch_size=32, learning_rate=0.001, 
                      weight_decay=1e-4, patience=10, embedding_dim=128, hidden_dim=256, 
                      num_layers=2, dropout=0.2, rnn_type='LSTM', naming_style='simple'):
        """五折交叉验证 - 简化版本，只保留必要的训练和评估"""
        print(f"\n🔄 开始 {n_splits} 折交叉验证...")
        print(f"🖥️  使用设备: {self.device}")
        print(f"🧠 RNN类型: {rnn_type}")
        
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
                descriptor_dim = len(train_dataset[0]['descriptors'])
            else:
                descriptor_dim = 12
                
            model = MolecularRNN(
                vocab_size=self.processor.vocab_size,
                descriptor_dim=descriptor_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task_type,
                num_classes=self.num_classes,
                rnn_type=rnn_type
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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                # 将数据移到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                output = model(batch)
                loss = criterion(output, batch['target'])
                
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
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path, map_location=self.device))
        return train_losses, val_losses
    
    def _evaluate_loss(self, model, data_loader, criterion):
        """评估损失"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = model(batch)
                loss = criterion(output, batch['target'])
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _evaluate_with_metrics(self, model, dataset):
        """评估模型并返回完整指标"""
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = model(batch)
                
                # 调试信息（仅在第一个批次打印）
                if len(predictions) == 0:
                    print(f"   🔍 调试信息 - 输出张量形状: {output.shape}, 目标张量形状: {batch['target'].shape}")
                
                # 处理回归任务输出
                # 确保输出是1维张量，避免0维标量问题
                output_np = output.cpu().numpy()
                if output_np.ndim == 0:
                    predictions.append(float(output_np))
                else:
                    predictions.extend(output_np.flatten())
                
                target_np = batch['target'].cpu().numpy()
                if target_np.ndim == 0:
                    true_values.append(float(target_np))
                else:
                    true_values.extend(target_np.flatten())
        
        # 计算回归指标
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
        
        return metrics

    def evaluate_model_on_datasets(self, model, train_dataset, test_dataset):
        """评估模型在训练集和测试集上的性能 - 参考CatBoost的evaluate_model"""
        train_metrics = self._evaluate_with_metrics(model, train_dataset)
        test_metrics = self._evaluate_with_metrics(model, test_dataset)
        
        # 组织返回格式，匹配CatBoost（回归任务）
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
        """保存训练集和测试集的真实值和预测值 - 参考CatBoost"""
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
        """保存评估指标 - 参考CatBoost"""
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
        """绘制并保存预测值与真实值的散点图 - 参考CatBoost"""
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
        """保存所有文件的汇总评估指标 - 参考CatBoost"""
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
                           progress_manager=None, method="RNN", model_name=None, 
                           epochs=100, batch_size=32, learning_rate=0.001, 
                           embedding_dim=128, hidden_dim=256, num_layers=2, 
                           dropout=0.2, rnn_type='LSTM'):
        """处理单个文件的训练和测试 - 参考CatBoost的process_method_data中的单文件处理逻辑"""
        
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
            
            # 构建词汇表
            all_smiles = train_smiles.tolist() + test_smiles.tolist()
            self.processor.build_vocabulary(all_smiles)
            
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
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                rnn_type=rnn_type,
                naming_style='simple'
            )
            
            # 获取最佳模型（从交叉验证结果中）
            best_fold_idx = cv_results['best_fold_idx']
            best_model_path = cv_results['fold_models'][best_fold_idx]
            
            # 重新创建模型并加载最佳权重
            descriptor_dim = len(train_dataset[0]['descriptors'])
            best_model = MolecularRNN(
                vocab_size=self.processor.vocab_size,
                descriptor_dim=descriptor_dim,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task_type=self.task_type,
                num_classes=self.num_classes,
                rnn_type=rnn_type
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
                           method="RNN", model_name=None, epochs=100, batch_size=32, 
                           learning_rate=0.001, embedding_dim=128, hidden_dim=256, 
                           num_layers=2, dropout=0.2, rnn_type='LSTM'):
        """处理文件夹中的所有数据 - 参考CatBoost的process_method_data"""
        
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
                    learning_rate, embedding_dim, hidden_dim, num_layers,
                    dropout, rnn_type
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
def main():
    """主函数 - 改进版，参考CatBoost的处理流程"""
    print("=" * 80)
    print("🧬 RNN分子性质预测系统 - 改进版 (参考CatBoost结构)")
    print("=" * 80)
    
    # 创建进度管理器
    progress_manager = ProgressManager()
    
    try:
        # 检测GPU可用性
        gpu_info = detect_gpu_availability()
        
        # 选择RNN类型
        print("\n🧠 选择RNN类型:")
        print("1. LSTM (默认)")
        print("2. GRU") 
        print("3. RNN")
        
        rnn_choice = input("请选择RNN类型 (1、2 或 3，默认为 1): ").strip()
        if rnn_choice == '2':
            rnn_type = 'GRU'
            algorithm_name = 'GRU'
        elif rnn_choice == '3':
            rnn_type = 'RNN'
            algorithm_name = 'RNN'
        else:
            rnn_type = 'LSTM'
            algorithm_name = 'LSTM'
        
        # 固定为回归任务
        task_type = 'regression'
        print(f"\n📊 任务类型: 回归任务 (regression)")
        
        # 定义基础路径
        base_dir = os.getcwd()
        
        # 创建预测器
        predictor = MolecularPropertyPredictor(algorithm_name=algorithm_name, base_dir=base_dir)
        
        # 定义数据文件夹路径 - 参考CatBoost
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
            # 处理数据 - 参考CatBoost的处理方式
            method_metrics = predictor.process_folder_data(
                train_base, test_base, 
                progress_manager, method=algorithm_name, model_name=algorithm_name,
                epochs=100, batch_size=32, learning_rate=0.001,
                embedding_dim=128, hidden_dim=256, num_layers=2,
                dropout=0.2, rnn_type=rnn_type
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