#!/usr/bin/env python3
"""
ExtraTrees回归训练系统 - 基于原CatBoost系统改进
支持GPU加速的数据预处理和多进程并行训练
"""

import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib
import psutil
import GPUtil
from datetime import datetime
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import sys
import warnings
import argparse

# 抑制所有警告
warnings.filterwarnings('ignore')


# 路径管理类
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
    """从文件名中移除扩展名"""
    return os.path.splitext(filename)[0]


class ProgressManager:
    """进度管理类"""
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
    """检测GPU可用性"""
    gpu_info = {
        'has_gpu': False,
        'gpu_count': 0,
        'gpu_memory_total': 0,
        'gpu_memory_free': 0,
        'recommended_task_type': 'CPU',
        'recommended_gpu_ram_part': 0.95,
        'gpu_details': [],
        'selected_gpu_id': 0  # 默认选择GPU 0
    }
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_info['has_gpu'] = True
            gpu_info['gpu_count'] = len(gpus)
            gpu_info['recommended_task_type'] = 'GPU'
            
            for i, gpu in enumerate(gpus):
                gpu_detail = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'utilization': gpu.load * 100
                }
                gpu_info['gpu_details'].append(gpu_detail)
                gpu_info['gpu_memory_total'] += gpu.memoryTotal
                gpu_info['gpu_memory_free'] += gpu.memoryFree
            
            if gpu_info['gpu_memory_total'] > 8000:
                gpu_info['recommended_gpu_ram_part'] = 0.95
            elif gpu_info['gpu_memory_total'] > 4000:
                gpu_info['recommended_gpu_ram_part'] = 0.90
            else:
                gpu_info['recommended_gpu_ram_part'] = 0.80
                
        print("=== GPU检测结果 ===")
        print(f"GPU可用: {gpu_info['has_gpu']}")
        print(f"GPU数量: {gpu_info['gpu_count']}")
        if gpu_info['has_gpu']:
            print(f"总GPU内存: {gpu_info['gpu_memory_total']:.0f} MB")
            print(f"可用GPU内存: {gpu_info['gpu_memory_free']:.0f} MB")
            print(f"推荐task_type: {gpu_info['recommended_task_type']} (仅用于数据预处理)")
            print(f"推荐gpu_ram_part: {gpu_info['recommended_gpu_ram_part']}")
            
            for gpu_detail in gpu_info['gpu_details']:
                print(f"GPU {gpu_detail['id']}: {gpu_detail['name']} "
                      f"({gpu_detail['memory_free']:.0f}/{gpu_detail['memory_total']:.0f} MB可用, "
                      f"使用率: {gpu_detail['utilization']:.1f}%)")
            
            # GPU选择逻辑
            # 从全局变量获取命令行参数（如果有的话）
            args_gpu_id = getattr(select_gpu, '_args_gpu_id', None)
            auto_select = getattr(select_gpu, '_auto_select', False)
            
            if auto_select:
                # 自动选择显存最多的GPU
                best_gpu = max(range(len(gpu_info['gpu_details'])), 
                              key=lambda i: gpu_info['gpu_details'][i]['memory_free'])
                gpu_info['selected_gpu_id'] = best_gpu
                print(f"🤖 自动选择GPU {best_gpu}: {gpu_info['gpu_details'][best_gpu]['name']} (显存最多)")
            else:
                gpu_info['selected_gpu_id'] = select_gpu(gpu_info['gpu_details'], args_gpu_id)
            
            print(f"选择的GPU: {gpu_info['selected_gpu_id']} (用于数据预处理)")
        print("=" * 20)
            
    except Exception as e:
        print(f"GPU检测时出错，将使用CPU: {str(e)}")
        gpu_info['recommended_task_type'] = 'CPU'
    
    return gpu_info


def select_gpu(gpu_details, args_gpu_id=None):
    """选择GPU的函数"""
    if not gpu_details:
        return 0
    
    if len(gpu_details) == 1:
        print(f"🔧 只有一个GPU，自动选择GPU 0")
        return 0
    
    # 优先级：命令行参数 > 环境变量 > 交互式选择
    
    # 1. 检查命令行参数
    if args_gpu_id is not None:
        if 0 <= args_gpu_id < len(gpu_details):
            print(f"🎯 使用命令行参数指定的GPU {args_gpu_id}: {gpu_details[args_gpu_id]['name']}")
            return args_gpu_id
        else:
            print(f"⚠️ 命令行参数--gpu={args_gpu_id}超出范围，忽略")
    
    # 2. 检查环境变量
    env_gpu = os.environ.get('CATBOOST_GPU_ID', '').strip()
    if env_gpu:
        try:
            env_gpu_id = int(env_gpu)
            if 0 <= env_gpu_id < len(gpu_details):
                print(f"🌍 使用环境变量指定的GPU {env_gpu_id}: {gpu_details[env_gpu_id]['name']}")
                return env_gpu_id
            else:
                print(f"⚠️ 环境变量CATBOOST_GPU_ID={env_gpu}超出范围，忽略")
        except ValueError:
            print(f"⚠️ 环境变量CATBOOST_GPU_ID={env_gpu}不是有效数字，忽略")
    
    # 3. 交互式选择
    print(f"\n🎯 检测到 {len(gpu_details)} 个GPU，请选择要使用的GPU:")
    for i, gpu in enumerate(gpu_details):
        status = "推荐" if gpu['memory_free'] == max(g['memory_free'] for g in gpu_details) else ""
        print(f"  [{i}] GPU {i}: {gpu['name']} "
              f"(可用显存: {gpu['memory_free']:.0f}MB, "
              f"使用率: {gpu['utilization']:.1f}%) {status}")
    
    # 自动选择显存最多的GPU
    best_gpu = max(range(len(gpu_details)), key=lambda i: gpu_details[i]['memory_free'])
    
    try:
        print(f"\n💡 自动推荐: GPU {best_gpu} (显存最多)")
        print(f"💡 提示: 可使用 --gpu=N 参数或设置环境变量 CATBOOST_GPU_ID=N 来自动选择GPU")
        choice = input(f"请输入GPU编号 [0-{len(gpu_details)-1}] (直接回车使用推荐): ").strip()
        
        if choice == "":
            selected_gpu = best_gpu
        else:
            selected_gpu = int(choice)
            if selected_gpu < 0 or selected_gpu >= len(gpu_details):
                print(f"⚠️ 无效选择，使用推荐的GPU {best_gpu}")
                selected_gpu = best_gpu
                
        print(f"✅ 选择GPU {selected_gpu}: {gpu_details[selected_gpu]['name']}")
        return selected_gpu
        
    except (ValueError, KeyboardInterrupt):
        print(f"⚠️ 输入无效或用户中断，使用推荐的GPU {best_gpu}")
        return best_gpu


def get_optimal_extratrees_params(gpu_info, data_size=None, cpu_jobs=None):
    """根据GPU信息和数据大小获取最优的ExtraTrees参数"""
    
    # 基础参数配置
    base_params = {
        'random_state': 42,
        'bootstrap': True,
        'oob_score': True,
        'criterion': 'squared_error'
    }
    
    # 根据数据大小智能调整参数
    if data_size:
        print(f"📊 根据数据大小 ({data_size}) 优化参数...")
        
        if data_size <= 500:
            base_params.update({
                'n_estimators': 200,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'max_samples': 0.8
            })
            print("🔹 使用极小数据集配置: 少量估计器、浅树、强正则化")
            
        elif data_size <= 1000:
            base_params.update({
                'n_estimators': 300,
                'max_depth': 8,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'max_samples': 0.85
            })
            print("🔹 使用小数据集配置: 适中估计器数、中等树深")
            
        elif data_size <= 3000:
            base_params.update({
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'max_samples': 0.9
            })
            print("🔹 使用中小数据集配置: 标准配置")
            
        elif data_size <= 8000:
            base_params.update({
                'n_estimators': 800,
                'max_depth': 12,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'log2',
                'max_samples': 0.9
            })
            print("🔹 使用中等数据集配置: 更多估计器、深树")
            
        else:
            base_params.update({
                'n_estimators': 1000,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'log2',
                'max_samples': 0.95
            })
            print("🔹 使用大数据集配置: 大量估计器、深树")
    else:
        # 默认中等配置
        base_params.update({
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'max_samples': 0.9
        })
    
    # CPU并行配置 - ExtraTrees不支持GPU，但可以利用多核CPU
    if cpu_jobs is not None:
        # 用户指定了CPU线程数
        if cpu_jobs == -1:
            # 使用所有可用核心
            n_jobs = psutil.cpu_count()
            print(f"💻 使用所有可用CPU核心: {n_jobs}")
        elif cpu_jobs > 0:
            # 使用指定的线程数，但不超过系统核心数
            n_jobs = min(cpu_jobs, psutil.cpu_count())
            if cpu_jobs > psutil.cpu_count():
                print(f"⚠️ 指定的CPU线程数({cpu_jobs})超过系统核心数({psutil.cpu_count()})，使用{n_jobs}个线程")
            else:
                print(f"💻 使用用户指定的CPU线程数: {n_jobs}")
        else:
            print(f"⚠️ 无效的CPU线程数({cpu_jobs})，使用默认配置")
            n_jobs = min(psutil.cpu_count(), 8)
    else:
        # 自动选择CPU线程数
        if gpu_info['has_gpu'] and gpu_info['recommended_task_type'] == 'GPU':
            # 即使有GPU，ExtraTrees也只能使用CPU，但可以使用更多线程来补偿无GPU加速
            n_jobs = min(psutil.cpu_count(), 16)
            print(f"💻 ExtraTrees使用CPU训练，自动选择线程数: {n_jobs} (GPU用于数据预处理)")
        else:
            n_jobs = min(psutil.cpu_count(), 8)
            print(f"💻 使用CPU训练，自动选择线程数: {n_jobs}")
    
    cpu_params = {'n_jobs': n_jobs}
    base_params.update(cpu_params)
    
    return base_params


def monitor_system_resources():
    """监控系统资源使用情况"""
    resources = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            resources['gpu_info'] = []
            for gpu in gpus:
                resources['gpu_info'].append({
                    'id': gpu.id,
                    'utilization': gpu.load * 100,
                    'memory_used_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'memory_free_mb': gpu.memoryFree,
                    'temperature': gpu.temperature
                })
    except:
        resources['gpu_info'] = []
    
    return resources


def read_files_in_folder(folder_path, progress_manager=None, progress_name="reading_files"):   
    """读取指定文件夹中的所有CSV文件"""
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


def prepare_data(data):    
    """分离输入特征和目标变量"""
    if data.columns[-1] == 'property_log':
        X = data.drop(columns=[data.columns[-1], data.columns[-2]])
        y = data[data.columns[-1]]
    else:
        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

    feature_names = X.columns.tolist()
    return X, y, feature_names


def train_model_with_cv(X_train, y_train, gpu_info, progress_manager=None, cpu_jobs=None):
    """使用5折交叉验证训练ExtraTrees回归模型"""
    
    base_params = get_optimal_extratrees_params(gpu_info, len(X_train), cpu_jobs)
    
    # 智能参数网格搜索
    data_size = len(X_train)
    print(f"🔍 为数据大小 {data_size} 设计参数网格...")
    
    if data_size <= 500:
        param_grid = {
            'extratreesregressor__n_estimators': [200],
            'extratreesregressor__max_depth': [6],
            'extratreesregressor__min_samples_split': [5]
        }
        print("🔹 极小数据集参数网格: 1组合 (极速模式)")
        
    elif data_size <= 1000:
        param_grid = {
            'extratreesregressor__n_estimators': [300],
            'extratreesregressor__max_depth': [8],
            'extratreesregressor__min_samples_split': [4]
        }
        print("🔹 小数据集参数网格: 1组合 (极速模式)")
        
    elif data_size <= 3000:
        param_grid = {
            'extratreesregressor__n_estimators': [500],
            'extratreesregressor__max_depth': [10],
            'extratreesregressor__min_samples_split': [3]
        }
        print("🔹 中小数据集参数网格: 1组合 (极速模式)")
        
    else:
        if gpu_info['has_gpu']:
            param_grid = {
                'extratreesregressor__n_estimators': [800],
                'extratreesregressor__max_depth': [12],
                'extratreesregressor__min_samples_split': [2]
            }
        else:
            param_grid = {
                'extratreesregressor__n_estimators': [600],
                'extratreesregressor__max_depth': [10],
                'extratreesregressor__min_samples_split': [3]
            }
        print("🔹 大数据集参数网格: 1组合 (极速模式)")
    
    total_combinations = (len(param_grid['extratreesregressor__n_estimators']) * 
                         len(param_grid['extratreesregressor__max_depth']) * 
                         len(param_grid['extratreesregressor__min_samples_split']))
    
    if progress_manager:
        grid_pbar = progress_manager.create_progress_bar(
            'grid_search', 
            total_combinations,
            desc=f"网格搜索 (CPU多线程模式)"
        )
    
    # 记录训练开始前的资源状态
    start_resources = monitor_system_resources()
    print(f"训练开始 - CPU: {start_resources['cpu_percent']:.1f}%, "
          f"内存: {start_resources['memory_percent']:.1f}%, "
          f"可用内存: {start_resources['memory_available_gb']:.1f}GB")
    
    if start_resources['gpu_info']:
        for i, gpu in enumerate(start_resources['gpu_info']):
            print(f"GPU {i} - 使用率: {gpu['utilization']:.1f}%, "
                  f"显存使用: {gpu['memory_used_percent']:.1f}%, "
                  f"温度: {gpu['temperature']}°C")
    
    # 创建ExtraTrees模型
    et_model = ExtraTreesRegressor(**base_params)
    pipeline = make_pipeline(et_model)
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 使用标准GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=kf, 
        scoring='r2', 
        n_jobs=1,
        verbose=0  # 关闭详细输出
    )
    
    print(f"开始网格搜索 - 参数组合数: {total_combinations}")
    print(f"使用CPU多线程模式训练")
    
    try:
        if progress_manager:
            progress_manager.update_progress_bar('grid_search', 0, {'状态': '开始训练'})
        
        grid_search.fit(X_train, y_train)
        
        if progress_manager:
            progress_manager.update_progress_bar('grid_search', total_combinations, {'状态': '完成'})
            progress_manager.close_progress_bar('grid_search')
        
        end_resources = monitor_system_resources()
        
        training_resources = {
            'start_resources': start_resources,
            'end_resources': end_resources,
            'max_memory_used': max(start_resources['memory_percent'], end_resources['memory_percent'])
        }
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_scores = grid_search.cv_results_
        
        print(f"最佳交叉验证R²得分: {grid_search.best_score_:.4f}")
        print(f"最佳参数: {best_params}")
        
        return best_model, cv_scores, best_params, training_resources
        
    except Exception as e:
        if progress_manager:
            progress_manager.close_progress_bar('grid_search')
            
        print(f"ExtraTrees训练失败: {str(e)}")
        # ExtraTrees不需要GPU回退，因为它本身就是CPU算法
        # 尝试使用更保守的参数重新训练
        print("尝试使用更保守的参数重新训练...")
        
        # 减少参数复杂度
        base_params['n_estimators'] = min(base_params.get('n_estimators', 500), 300)
        base_params['max_depth'] = min(base_params.get('max_depth', 10), 8)
        base_params['n_jobs'] = min(psutil.cpu_count(), 4)  # 减少并行度
        
        et_model = ExtraTreesRegressor(**base_params)
        pipeline = make_pipeline(et_model)
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=kf, 
            scoring='r2', 
            n_jobs=1,  # 减少并行度避免内存问题
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        end_resources = monitor_system_resources()
        training_resources = {
            'start_resources': start_resources,
            'end_resources': end_resources,
            'max_memory_used': max(start_resources['memory_percent'], end_resources['memory_percent']),
            'fallback_to_conservative': True
        }
        
        return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_params_, training_resources


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """评估模型性能"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }
    
    predictions = {
        'train': {'true': y_train.values, 'pred': y_train_pred},
        'test': {'true': y_test.values, 'pred': y_test_pred}
    }
    
    return metrics, predictions


def save_predictions(predictions, path_manager, method, model_name, file_name):
    """保存训练集和测试集的真实值和预测值"""
    for data_type in ['train', 'test']:
        pred_df = pd.DataFrame({
            'true_values': predictions[data_type]['true'],
            'predicted_values': predictions[data_type]['pred']
        })
        
        pred_file = f"{method}_{model_name}_{file_name}_{data_type}_predictions.csv"
        pred_path = path_manager.get_file_path('predictions', pred_file, method, model_name)
        
        pred_df.to_csv(pred_path, index=False)
        print(f"{data_type.capitalize()} 预测结果保存至: {pred_path}")


def save_metrics(metrics, cv_scores, best_params, training_time, training_resources, path_manager, method, model_name, file_name):
    """保存评估指标"""
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
        'cv_mean_score': [cv_scores['mean_test_score'].max()],
        'cv_std_score': [cv_scores['std_test_score'][cv_scores['mean_test_score'].argmax()]],
        'best_n_estimators': [best_params.get('extratreesregressor__n_estimators', 'N/A')],
        'best_max_depth': [best_params.get('extratreesregressor__max_depth', 'N/A')],
        'best_min_samples_split': [best_params.get('extratreesregressor__min_samples_split', 'N/A')],
        'training_time': [training_time],
        'max_memory_used_percent': [training_resources.get('max_memory_used', 'N/A')],
        'fallback_to_conservative': [training_resources.get('fallback_to_conservative', False)]
    }
    
    # 添加GPU相关信息
    if training_resources['start_resources']['gpu_info']:
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
    metrics_path = path_manager.get_file_path('metrics', metrics_file, method, model_name)
    
    metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
    print(f"评估指标保存至: {metrics_path}")
    
    # 显示资源使用摘要
    print(f"资源使用摘要:")
    print(f"  - 最大内存使用: {training_resources.get('max_memory_used', 'N/A')}%")
    if training_resources['start_resources']['gpu_info']:
        gpu_info = training_resources['end_resources']['gpu_info'][0]
        print(f"  - GPU最终使用率: {gpu_info['utilization']:.1f}%")
        print(f"  - GPU最终显存使用: {gpu_info['memory_used_percent']:.1f}%")
        print(f"  - GPU最终温度: {gpu_info['temperature']}°C")
    
    return metrics_df


def plot_and_save_scatter(predictions, metrics, path_manager, method, model_name, file_name, last_column_name):
    """绘制并保存预测值与真实值的散点图"""
    y_train = predictions['train']['true']
    y_train_pred = predictions['train']['pred']
    y_test = predictions['test']['true']
    y_test_pred = predictions['test']['pred']

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

    match = re.match(r'^(.*?)[\[\(].*?[\]\)]$', last_column_name)
    title_name = match.group(1).strip() if match else last_column_name

    plt.text(0.5, 0.95, f"{model_name}_{method}_{title_name}", 
             transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', horizontalalignment='center')

    plt.tick_params(which='major', direction='in', length=5, labelsize=16)
    plt.xlabel(f"Actual {last_column_name}", fontsize=16)
    plt.ylabel(f"Predicted {last_column_name}", fontsize=16)
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
    image_path = path_manager.get_file_path('images', image_file, method, model_name)
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"散点图保存至: {image_path}")


def save_summary_metrics(all_metrics, path_manager, timestamp=None):
    """保存所有文件的汇总评估指标"""
    if not all_metrics:
        print("警告: 没有指标可保存")
        return None
    
    summary_df = pd.concat(all_metrics, ignore_index=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_file = f'ExtraTrees_Train_Test_Results_{timestamp}.csv'
    summary_path = path_manager.get_file_path('summaries', summary_file, create=True)
    
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"汇总评估指标保存至: {summary_path}")
    
    return summary_df


def append_summary_metrics(method_metrics, path_manager, timestamp, method_name):
    """追加单个方法的评估指标到汇总文件"""
    if not method_metrics:
        print(f"⚠️ {method_name} 方法没有指标可保存")
        return None
    
    summary_file = f'ExtraTrees_Train_Test_Results_{timestamp}.csv'
    summary_path = path_manager.get_file_path('summaries', summary_file, create=True)
    
    # 将method_metrics合并为DataFrame
    method_df = pd.concat(method_metrics, ignore_index=True)
    
    # 检查文件是否存在
    if os.path.exists(summary_path):
        # 文件存在，追加数据
        method_df.to_csv(summary_path, mode='a', header=False, index=False, float_format='%.6f')
        print(f"📝 {method_name} 方法的指标已追加至汇总文件: {summary_path}")
    else:
        # 文件不存在，创建新文件
        method_df.to_csv(summary_path, mode='w', header=True, index=False, float_format='%.6f')
        print(f"📝 创建汇总文件并保存 {method_name} 方法指标: {summary_path}")
    
    return method_df


def process_method_data(method, train_folder, test_folder, path_manager, gpu_info, progress_manager=None, model_name="ExtraTrees", cpu_jobs=None):
    """处理单个方法的训练和测试数据"""
    method_metrics = []
    
    if not os.path.exists(train_folder):
        print(f"警告: 训练数据文件夹不存在: {train_folder}")
        return method_metrics
    
    if not os.path.exists(test_folder):
        print(f"警告: 测试数据文件夹不存在: {test_folder}")
        return method_metrics
    
    # 读取训练和测试数据（带进度条）
    print("正在读取训练数据...")
    train_data_list, train_file_names, train_last_column_names = read_files_in_folder(
        train_folder, progress_manager, f"{method}_train_read"
    )
    
    print("正在读取测试数据...")
    test_data_list, test_file_names, test_last_column_names = read_files_in_folder(
        test_folder, progress_manager, f"{method}_test_read"
    )
    
    if not train_data_list:
        print(f"警告: {train_folder} 文件夹中没有找到CSV文件")
        return method_metrics
    
    if not test_data_list:
        print(f"警告: {test_folder} 文件夹中没有找到CSV文件")
        return method_metrics
    
    # 创建文件名到数据的映射
    train_data_dict = {get_base_filename(name): (data, col_name) 
                       for data, name, col_name in zip(train_data_list, train_file_names, train_last_column_names)}
    test_data_dict = {get_base_filename(name): (data, col_name) 
                      for data, name, col_name in zip(test_data_list, test_file_names, test_last_column_names)}
    
    # 找到训练和测试数据的共同文件
    common_files = set(train_data_dict.keys()) & set(test_data_dict.keys())
    
    if not common_files:
        print(f"警告: {method} 方法的训练和测试数据没有共同的文件")
        return method_metrics
    
    print(f"找到 {len(common_files)} 个共同文件进行处理: {list(common_files)}")
    
    # 创建文件处理进度条
    if progress_manager:
        file_progress = progress_manager.create_progress_bar(
            f"{method}_files",
            len(common_files),
            desc=f"{method} 文件处理"
        )
    
    for i, file_name in enumerate(sorted(common_files)):
        try:
            # 更新文件处理进度
            if progress_manager:
                progress_manager.update_progress_bar(
                    f"{method}_files",
                    1 if i > 0 else 0,
                    {'当前文件': file_name[:15] + '...' if len(file_name) > 15 else file_name}
                )
            
            print(f"\n处理文件 ({i+1}/{len(common_files)}): {file_name}")
            print("-" * 60)
            
            # 获取训练和测试数据
            train_data, train_col_name = train_data_dict[file_name]
            test_data, test_col_name = test_data_dict[file_name]
            
            # 准备训练数据
            X_train, y_train, feature_names = prepare_data(train_data)
            
            # 准备测试数据
            X_test, y_test, _ = prepare_data(test_data)
            
            print(f"数据维度 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            # 检查特征维度是否匹配
            if X_train.shape[1] != X_test.shape[1]:
                print(f"警告: 文件 {file_name} 的训练和测试数据特征维度不匹配")
                continue
            
            start_time = time.time()
            
            # 使用5折交叉验证训练模型（带进度条）
            print("开始5折交叉验证训练...")
            best_model, cv_scores, best_params, training_resources = train_model_with_cv(
                X_train, y_train, gpu_info, progress_manager, cpu_jobs
            )
            
            training_time = time.time() - start_time
            print(f"训练完成，总耗时: {training_time:.2f} 秒")
            
            # 评估模型
            print("正在评估模型性能...")
            metrics, predictions = evaluate_model(best_model, X_train, y_train, X_test, y_test)
            
            # 保存最佳模型
            model_file = f"{method}_{model_name}_{file_name}_best_model.joblib"
            model_path = path_manager.get_file_path('models', model_file, method, model_name)
            joblib.dump(best_model, model_path)
            print(f"最佳模型保存至: {model_path}")
            
            # 保存预测结果
            save_predictions(predictions, path_manager, method, model_name, file_name)
            
            # 保存评估指标（包含资源使用信息）
            metrics_df = save_metrics(
                metrics, cv_scores, best_params, training_time, training_resources,
                path_manager, method, model_name, file_name
            )
            method_metrics.append(metrics_df)
            
            # 绘制和保存散点图
            print("正在生成可视化图表...")
            plot_and_save_scatter(
                predictions, metrics, path_manager, method, 
                model_name, file_name, train_col_name
            )
            
            print(f"文件 {file_name} 处理完成")
            print("=" * 60)
            
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 完成文件处理进度条
    if progress_manager:
        progress_manager.update_progress_bar(f"{method}_files", 1, {'状态': '全部完成'})
        progress_manager.close_progress_bar(f"{method}_files")
    
    return method_metrics


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='ExtraTrees回归训练系统 - 支持多线程CPU并行训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python ExtraTrees_Fixed_Clean.py                        # 交互式选择GPU(用于数据预处理)
  python ExtraTrees_Fixed_Clean.py --gpu=1               # 使用GPU 1(用于数据预处理)
  python ExtraTrees_Fixed_Clean.py --cpu                 # 强制使用CPU
  python ExtraTrees_Fixed_Clean.py --cpu-jobs=8          # 指定使用8个CPU线程
  python ExtraTrees_Fixed_Clean.py --gpu=0 --cpu-jobs=16 # 使用GPU 0预处理 + 16个CPU线程训练
  CATBOOST_GPU_ID=1 python ExtraTrees_Fixed_Clean.py     # 通过环境变量指定GPU

CPU线程数说明:
  - 不指定时: 有GPU环境默认使用min(CPU核心数, 16)个线程，无GPU环境使用min(CPU核心数, 8)个线程
  - --cpu-jobs=N: 强制使用N个线程 (建议不超过CPU核心数)
  - --cpu-jobs=-1: 使用所有可用CPU核心
  - 系统CPU核心数: {}

注意: ExtraTrees本身只使用CPU训练，GPU仅用于数据预处理加速
优先级: 命令行参数 > 环境变量 > 交互式选择
        '''.format(psutil.cpu_count())
    )
    
    # GPU相关参数组
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        '--gpu', 
        type=int, 
        metavar='N',
        help='指定使用的GPU编号 (例如: --gpu=0, 仅用于数据预处理)'
    )
    gpu_group.add_argument(
        '--cpu', 
        action='store_true',
        help='强制使用CPU模式 (不使用GPU进行数据预处理)'
    )
    gpu_group.add_argument(
        '--auto', 
        action='store_true',
        help='自动选择最佳GPU（显存最多的，仅用于数据预处理）'
    )
    
    # CPU线程数参数
    parser.add_argument(
        '--cpu-jobs', 
        type=int, 
        metavar='N',
        help=f'指定CPU线程数 (1-{psutil.cpu_count()} 或 -1表示使用全部核心，默认: 自动选择)'
    )
    
    return parser.parse_args()


def main():
    """主函数，处理训练和测试数据进行建模与评估"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 80)
    print("🚀 ExtraTrees回归训练系统启动")
    if args.gpu is not None:
        print(f"🎯 命令行指定GPU: {args.gpu} (用于数据预处理)")
    elif args.cpu:
        print("💻 命令行指定CPU模式")
    elif args.auto:
        print("🤖 自动选择最佳GPU (用于数据预处理)")
    
    # 显示CPU线程数配置
    if args.cpu_jobs is not None:
        if args.cpu_jobs == -1:
            print(f"🔧 CPU线程数: 全部核心 ({psutil.cpu_count()})")
        else:
            print(f"🔧 CPU线程数: {args.cpu_jobs}")
    else:
        print("🔧 CPU线程数: 自动选择")
    
    print("=" * 80)
    
    # 验证CPU线程数参数
    if args.cpu_jobs is not None:
        if args.cpu_jobs < -1 or args.cpu_jobs == 0:
            print(f"❌ 错误: 无效的CPU线程数 {args.cpu_jobs}")
            print("💡 提示: 使用 --cpu-jobs=N (N>0) 指定线程数，或 --cpu-jobs=-1 使用所有核心")
            return None
        elif args.cpu_jobs > psutil.cpu_count():
            print(f"⚠️ 警告: 指定的CPU线程数({args.cpu_jobs})超过系统核心数({psutil.cpu_count()})")
            print(f"💡 系统将自动调整为 {psutil.cpu_count()} 个线程")
    
    # 创建进度管理器
    progress_manager = ProgressManager()
    
    try:
        # 将命令行参数传递给select_gpu函数
        if hasattr(args, 'gpu'):
            select_gpu._args_gpu_id = args.gpu
            select_gpu._force_cpu = args.cpu
            select_gpu._auto_select = args.auto
        
        # 检测GPU可用性
        gpu_info = detect_gpu_availability()
        
        # 如果命令行指定了CPU模式，强制使用CPU
        if args.cpu:
            gpu_info['has_gpu'] = False
            gpu_info['recommended_task_type'] = 'CPU'
            print("💻 强制使用CPU模式")
        
        # 定义基础路径
        base_dir = os.getcwd()
        
        # 创建路径管理器
        path_manager = PathManager(base_dir)
        
        # 定义数据文件夹
        train_base = os.path.join(base_dir, 'train_data')
        test_base = os.path.join(base_dir, 'test_data')
        
        # 定义方法映射
        methods = {
            # Fingers 指纹方法
            'AtomPair_512': ('Fingers_/Atompair_512', 'Fingers_/Atompair_512'),
            'AtomPair_1024': ('Fingers_/Atompair_1024', 'Fingers_/Atompair_1024'),
            'AtomPair_2048': ('Fingers_/Atompair_2048', 'Fingers_/Atompair_2048'),
            'Maccs': ('Fingers_/Maccs', 'Fingers_/Maccs'),
            'Morgan_512': ('Fingers_/Morgan_512', 'Fingers_/Morgan_512'),
            'Morgan_1024': ('Fingers_/Morgan_1024', 'Fingers_/Morgan_1024'),
            'Morgan_2048': ('Fingers_/Morgan_2048', 'Fingers_/Morgan_2048'),
            'RDKit_512': ('Fingers_/RDKit_512', 'Fingers_/RDKit_512'),
            'RDKit_1024': ('Fingers_/RDKit_1024', 'Fingers_/RDKit_1024'),
            'RDKit_2048': ('Fingers_/RDKit_2048', 'Fingers_/RDKit_2048'),
            'Torsion_512': ('Fingers_/Torsion_512', 'Fingers_/Torsion_512'),
            'Torsion_1024': ('Fingers_/Torsion_1024', 'Fingers_/Torsion_1024'),
            'Torsion_2048': ('Fingers_/Torsion_2048', 'Fingers_/Torsion_2048'),
            # Mordred 描述符
            'Mordred': ('Mordred', 'Mordred')
        }
        
        # 用于存储所有指标的列表
        all_metrics = []
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 显示系统信息
        print(f"\n📊 系统信息:")
        print(f"CPU核心数: {psutil.cpu_count()}")
        print(f"总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        
        # 创建总体进度条
        overall_progress = progress_manager.create_progress_bar(
            'overall',
            len(methods),
            desc="总体进度"
        )
        
        # 处理每种方法的数据
        for method_idx, (method, (train_subfolder, test_subfolder)) in enumerate(methods.items()):
            print(f"\n{'='*80}")
            print(f"🔬 开始处理 {method} 数据... ({method_idx+1}/{len(methods)})")
            print(f"{'='*80}")
            
            try:
                train_folder = os.path.join(train_base, train_subfolder)
                test_folder = os.path.join(test_base, test_subfolder)
                
                # 处理该方法的数据（带进度条）
                method_metrics = process_method_data(
                    method, train_folder, test_folder, 
                    path_manager, gpu_info, progress_manager, model_name="ExtraTrees", cpu_jobs=args.cpu_jobs
                )
                
                # 立即保存该方法的汇总指标
                if method_metrics:
                    append_summary_metrics(method_metrics, path_manager, timestamp, method)
                
                # 将该方法的指标添加到总列表中（保留用于最终统计）
                all_metrics.extend(method_metrics)
                
                print(f"✅ 完成 {method} 数据处理, 共处理 {len(method_metrics)} 个文件")
                
                # 显示当前系统资源状态
                current_resources = monitor_system_resources()
                print(f"📈 当前系统状态 - CPU: {current_resources['cpu_percent']:.1f}%, "
                      f"内存: {current_resources['memory_percent']:.1f}%")
                
                # 更新总体进度
                progress_manager.update_progress_bar(
                    'overall', 
                    1,
                    {
                        '当前方法': method[:12] + '...' if len(method) > 12 else method,
                        '文件数': len(method_metrics)
                    }
                )
                
            except Exception as e:
                print(f"❌ 处理 {method} 数据时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 即使出错也要更新进度条
                progress_manager.update_progress_bar('overall', 1, {'状态': '出错'})
        
        # 关闭总体进度条
        progress_manager.close_progress_bar('overall')
        
        # 显示最终统计信息（汇总结果已实时保存）
        if all_metrics:
            summary_file = f'ExtraTrees_Train_Test_Results_{timestamp}.csv'
            summary_path = path_manager.get_file_path('summaries', summary_file, create=False)
            print(f"\n✅ 汇总结果已实时保存至: {summary_path}")
            
            # 统计训练情况
            conservative_count = sum(1 for df in all_metrics if 'fallback_to_conservative' in df.columns and df['fallback_to_conservative'].iloc[0])
            total_count = len(all_metrics)
            
            print(f"\n📊 训练统计:")
            print(f"总文件数: {total_count}")
            print(f"正常训练: {total_count - conservative_count}")
            print(f"保守参数训练: {conservative_count}")
            print(f"成功率: {(total_count - conservative_count)/total_count*100:.1f}%" if total_count > 0 else "成功率: 0%")
            
        else:
            print("⚠️ 警告: 没有成功处理任何数据，未保存汇总评估指标")
        
        print(f"\n🎉 所有处理完成！")
        print("=" * 80)
        
        return {
            'all_metrics': all_metrics,
            'methods_processed': list(methods.keys()),
            'timestamp': timestamp,
            'gpu_info': gpu_info
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
    results = main()
    print("所有处理完成！") 