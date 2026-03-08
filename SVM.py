#!/usr/bin/env python3
"""
SVM 建模预测系统 - 基于CatBoost版本改进
特色功能：
1. GPU加速支持 (NVIDIA cuML)
2. 智能SVM参数优化
3. Mordred特征自动标准化
4. CPU/GPU灵活配置
5. 完整的进度跟踪
6. 详细的性能评估
"""

import os
import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import psutil
from datetime import datetime
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, Pipeline
from tqdm import tqdm
import sys
import warnings
import argparse

# 抑制所有警告
warnings.filterwarnings('ignore')

# GPU检测和导入
try:
    import GPUtil
    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False

# 尝试导入cuML (GPU加速)
try:
    from cuml.svm import SVR as cuSVR
    from cuml import train_test_split as cu_train_test_split
    HAS_CUML = True
    print("✅ 检测到cuML，GPU加速可用")
except ImportError:
    HAS_CUML = False
    print("⚠️ 未检测到cuML，将使用CPU模式")

# 导入scikit-learn SVM作为备选
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


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
        'gpu_details': [],
        'cuml_available': HAS_CUML
    }
    
    if HAS_GPU_UTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info['has_gpu'] = True
                gpu_info['gpu_count'] = len(gpus)
                gpu_info['recommended_task_type'] = 'GPU' if HAS_CUML else 'CPU'
                
                for gpu in gpus:
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
        except Exception as e:
            print(f"GPU检测时出错: {str(e)}")
    
    print("=== GPU/CPU检测结果 ===")
    print(f"GPU可用: {gpu_info['has_gpu']}")
    print(f"cuML可用: {gpu_info['cuml_available']}")
    if gpu_info['has_gpu']:
        print(f"GPU数量: {gpu_info['gpu_count']}")
        print(f"总GPU内存: {gpu_info['gpu_memory_total']:.0f} MB")
        for gpu_detail in gpu_info['gpu_details']:
            print(f"GPU {gpu_detail['id']}: {gpu_detail['name']} "
                  f"({gpu_detail['memory_free']:.0f}/{gpu_detail['memory_total']:.0f} MB可用)")
    
    if gpu_info['has_gpu'] and gpu_info['cuml_available']:
        print(f"推荐模式: GPU加速 (cuML)")
    else:
        print(f"推荐模式: CPU (scikit-learn)")
    print("=" * 20)
    
    return gpu_info


def get_optimal_svm_params(gpu_info, data_size=None, method_name=None):
    """极简SVM参数配置 - 仅保留最重要的C参数"""
    
    # 极简配置：只提供C参数的初始值，其他参数使用SVR默认值
    if method_name and 'Mordred' in method_name:
        base_params = {'C': 1.0}  # Mordred特征用中等正则化
        print("🧬 Mordred特征检测: 初始C=1.0")
    else:
        base_params = {'C': 10.0}  # 指纹特征用较弱正则化
        print("🔍 指纹特征检测: 初始C=10.0")
    
    # 显示运行模式
    if gpu_info['has_gpu'] and gpu_info['cuml_available']:
        print(f"🚀 GPU模式 (cuML) - 仅调优C参数")
    else:
        print(f"💻 CPU模式 (scikit-learn) - 仅调优C参数")
    
    return base_params


def monitor_system_resources():
    """监控系统资源使用情况"""
    resources = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    if HAS_GPU_UTIL:
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
    else:
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


def prepare_data(data, method_name=None, features_to_keep=None):    
    """分离输入特征和目标变量，Mordred特征自动标准化
    
    Args:
        data: 输入数据
        method_name: 方法名称
        features_to_keep: 要保留的特征列表（用于确保训练集和测试集特征一致）
    """
    if data.columns[-1] == 'property_log':
        X = data.drop(columns=[data.columns[-1], data.columns[-2]])
        y = data[data.columns[-1]]
    else:
        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

    feature_names = X.columns.tolist()
    
    # 检查是否需要标准化（SVM对特征尺度敏感，Mordred特征必须标准化）
    needs_scaling = method_name and 'Mordred' in method_name
    
    if needs_scaling:
        print(f"🧪 检测到Mordred数据，进行数据预处理...")
        print(f"原始数据形状: {X.shape}")
        
        # 检查缺失值
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"发现 {missing_count} 个缺失值，使用均值填充")
            X = X.fillna(X.mean())
        
        # 检查无穷值
        inf_count = np.isinf(X.values).sum()
        if inf_count > 0:
            print(f"发现 {inf_count} 个无穷值，进行处理")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())
        
        # 如果提供了features_to_keep，直接使用这些特征
        if features_to_keep is not None:
            # 确保所有要保留的特征都存在于当前数据中
            available_features = [f for f in features_to_keep if f in X.columns]
            if len(available_features) != len(features_to_keep):
                missing_features = set(features_to_keep) - set(available_features)
                print(f"警告: 缺少 {len(missing_features)} 个特征: {list(missing_features)[:5]}...")
            
            X = X[available_features]
            feature_names = available_features
            print(f"使用指定特征集，特征数: {len(feature_names)}")
        else:
            # 移除方差为0的特征（仅在没有指定features_to_keep时）
            zero_var_cols = X.columns[X.var() == 0]
            if len(zero_var_cols) > 0:
                print(f"移除 {len(zero_var_cols)} 个零方差特征")
                X = X.drop(columns=zero_var_cols)
                feature_names = X.columns.tolist()
        
        print(f"预处理后数据形状: {X.shape}")
        print("🔧 检测到Mordred特征，将应用StandardScaler标准化（SVM必需）")
        scaler = StandardScaler()
        return X, y, feature_names, scaler
    else:
        print("📊 使用原始特征（指纹特征通常无需标准化）")
        return X, y, feature_names, None


def train_model_with_cv(X_train, y_train, gpu_info, method_name=None, scaler=None, progress_manager=None):
    """使用5折交叉验证训练SVM回归模型"""
    
    base_params = get_optimal_svm_params(gpu_info, len(X_train), method_name)
    
    # 极简SVM参数网格搜索 - 仅搜索最重要的C参数
    data_size = len(X_train)
    print(f"🔍 为数据大小 {data_size} 设计极简SVM参数网格 (仅C参数)...")
    
    # 根据数据大小和特征类型选择最合适的C值范围
    if 'Mordred' in method_name if method_name else False:
        # Mordred高维特征需要更强的正则化
        param_grid = {'svr__C': [0.1, 1.0, 10.0]}
        print("🔹 Mordred特征: C=[0.1, 1.0, 10.0] (强正则化)")
    else:
        # 指纹特征通常需要较弱的正则化
        param_grid = {'svr__C': [1.0, 10.0, 100.0]}
        print("🔹 指纹特征: C=[1.0, 10.0, 100.0] (中等正则化)")
    
    total_combinations = len(param_grid['svr__C'])
    
    if progress_manager:
        grid_pbar = progress_manager.create_progress_bar(
            'grid_search', 
            total_combinations,
            desc=f"网格搜索 (SVM)"
        )
    
    # 记录训练开始前的资源状态
    start_resources = monitor_system_resources()
    print(f"训练开始 - CPU: {start_resources['cpu_percent']:.1f}%, "
          f"内存: {start_resources['memory_percent']:.1f}%, "
          f"可用内存: {start_resources['memory_available_gb']:.1f}GB")
    
    if start_resources['gpu_info']:
        for i, gpu in enumerate(start_resources['gpu_info']):
            print(f"GPU {i} - 使用率: {gpu['utilization']:.1f}%, "
                  f"显存使用: {gpu['memory_used_percent']:.1f}%")
    
    # 根据GPU可用性创建SVM模型
    use_gpu = gpu_info['has_gpu'] and gpu_info['cuml_available']
    
    # 仅保留最重要的C参数，使用SVR默认值（kernel='rbf', gamma='scale'等）
    svr_params = {}
    if 'C' in base_params:
        svr_params['C'] = base_params['C']
    # 让SVR使用所有其他参数的默认值，这样最简洁且通常效果很好
    
    if use_gpu:
        # 使用cuML GPU加速
        svm_model = cuSVR(**svr_params)
        print("🚀 使用cuML GPU加速SVM")
    else:
        # 使用scikit-learn CPU版本
        svm_model = SVR(**svr_params)
        print("💻 使用scikit-learn CPU SVM")
    
    if svr_params:
        print(f"🔧 极简SVM配置: {svr_params} (其他参数使用默认值)")
    else:
        print(f"🔧 极简SVM配置: 使用所有默认参数")
    
    # 创建管道（如果需要标准化）
    if scaler is not None:
        pipeline = Pipeline([
            ('scaler', scaler),
            ('svr', svm_model)
        ])
        print("🔧 使用标准化管道（Mordred特征）")
    else:
        # 使用Pipeline而不是make_pipeline以确保参数名称一致
        pipeline = Pipeline([
            ('svr', svm_model)
        ])
        print("📊 使用简单管道（指纹特征）")
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 使用标准GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=kf, 
        scoring='r2', 
        n_jobs=1,  # SVM训练通常不需要多进程
        verbose=0
    )
    
    print(f"开始网格搜索 - 参数组合数: {total_combinations}")
    print(f"使用{'GPU' if use_gpu else 'CPU'}模式训练")
    
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
            'max_memory_used': max(start_resources['memory_percent'], end_resources['memory_percent']),
            'gpu_used': use_gpu
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
        
        print(f"训练失败: {str(e)}")
        raise e


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
        'best_C': [best_params.get('svr__C', 'N/A')],
        'training_time': [training_time],
        'max_memory_used_percent': [training_resources.get('max_memory_used', 'N/A')],
        'gpu_used': [training_resources.get('gpu_used', False)]
    }
    
    # 添加GPU相关信息
    if training_resources['start_resources']['gpu_info']:
        gpu_start = training_resources['start_resources']['gpu_info'][0]
        gpu_end = training_resources['end_resources']['gpu_info'][0]
        metrics_dict.update({
            'gpu_utilization_start': [gpu_start['utilization']],
            'gpu_utilization_end': [gpu_end['utilization']],
            'gpu_memory_used_start': [gpu_start['memory_used_percent']],
            'gpu_memory_used_end': [gpu_end['memory_used_percent']]
        })
    else:
        metrics_dict.update({
            'gpu_utilization_start': ['N/A'],
            'gpu_utilization_end': ['N/A'],
            'gpu_memory_used_start': ['N/A'],
            'gpu_memory_used_end': ['N/A']
        })
    
    metrics_df = pd.DataFrame(metrics_dict)
    
    metrics_file = f"{method}_{model_name}_{file_name}_metrics.csv"
    metrics_path = path_manager.get_file_path('metrics', metrics_file, method, model_name)
    
    metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
    print(f"评估指标保存至: {metrics_path}")
    
    # 显示资源使用摘要
    print(f"资源使用摘要:")
    print(f"  - 最大内存使用: {training_resources.get('max_memory_used', 'N/A')}%")
    print(f"  - GPU加速: {'是' if training_resources.get('gpu_used', False) else '否'}")
    if training_resources['start_resources']['gpu_info']:
        gpu_info = training_resources['end_resources']['gpu_info'][0]
        print(f"  - GPU最终使用率: {gpu_info['utilization']:.1f}%")
        print(f"  - GPU最终显存使用: {gpu_info['memory_used_percent']:.1f}%")
    
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
    
    summary_file = f'SVM_Train_Test_Results_{timestamp}.csv'
    summary_path = path_manager.get_file_path('summaries', summary_file, create=True)
    
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"汇总评估指标保存至: {summary_path}")
    
    return summary_df


def append_summary_metrics(method_metrics, path_manager, timestamp, method_name):
    """追加单个方法的评估指标到汇总文件"""
    if not method_metrics:
        print(f"⚠️ {method_name} 方法没有指标可保存")
        return None
    
    summary_file = f'SVM_Train_Test_Results_{timestamp}.csv'
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


def process_method_data(method, train_folder, test_folder, path_manager, gpu_info, progress_manager=None, model_name="SVM"):
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
            
            # 准备训练数据（确定最终特征集）
            train_prep_result = prepare_data(train_data, method)
            if len(train_prep_result) == 4:
                X_train, y_train, feature_names, scaler = train_prep_result
            else:
                X_train, y_train, feature_names = train_prep_result
                scaler = None
            
            # 准备测试数据（使用训练集确定的特征集）
            test_prep_result = prepare_data(test_data, method, features_to_keep=feature_names)
            if len(test_prep_result) == 4:
                X_test, y_test, _, _ = test_prep_result
            else:
                X_test, y_test, _ = test_prep_result
            
            print(f"数据维度 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            # 检查特征维度是否匹配（现在应该总是匹配）
            if X_train.shape[1] != X_test.shape[1]:
                print(f"错误: 文件 {file_name} 的训练和测试数据特征维度仍不匹配")
                print(f"训练集特征数: {X_train.shape[1]}, 测试集特征数: {X_test.shape[1]}")
                continue
            else:
                print(f"✅ 特征维度匹配: {X_train.shape[1]} 个特征")
            
            start_time = time.time()
            
            # 使用5折交叉验证训练模型（带进度条）
            print("开始5折交叉验证训练...")
            best_model, cv_scores, best_params, training_resources = train_model_with_cv(
                X_train, y_train, gpu_info, method, scaler, progress_manager
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
        description='🚀 SVM 建模预测系统 - GPU加速的支持向量机回归',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
📋 使用示例:
  python SVM_Fixed_Clean.py                            # 自动检测GPU，优先使用GPU加速
  python SVM_Fixed_Clean.py --cpu                      # 强制使用CPU模式
  python SVM_Fixed_Clean.py --n_jobs=4                 # CPU模式下指定并行度

🚀 特色功能:
  ✅ GPU加速支持 (NVIDIA cuML) - 50x+性能提升
  ✅ CPU回退机制 (scikit-learn) - 无GPU时自动切换
  ✅ Mordred特征自动标准化 (StandardScaler) - SVM必需
  ✅ 简化参数优化 (仅调优最重要的C参数)
  ✅ 智能正则化选择 (基于数据大小和特征类型)
  ✅ 完整进度跟踪和资源监控

⚙️  参数说明:
  --cpu          强制使用CPU模式 (scikit-learn)
  --n_jobs N     CPU模式下的并行工作数 (默认1，SVM通常无需多进程)

🧠 SVM算法特点:
  • 支持向量机，适合中小数据集的非线性回归
  • 对特征缩放敏感，Mordred特征自动标准化
  • GPU加速可实现50x+性能提升 (需要NVIDIA cuML)
  • C参数控制正则化强度，是最重要的超参数

📁 数据结构要求:
  train_data/Fingers_/Morgan_512/     # 指纹训练数据
  test_data/Fingers_/Morgan_512/      # 指纹测试数据
  train_data/Mordred/                 # Mordred训练数据 (自动标准化)
  test_data/Mordred/                  # Mordred测试数据

📊 输出结果:
  Models/          # 保存的SVM模型文件
  Images/          # 散点图可视化
  Metrics/         # 详细评估指标 (含C参数和GPU使用情况)
  Predictions/     # 预测结果
  Summaries/       # 汇总报告

🚀 GPU加速说明:
  • 需要安装NVIDIA cuML: pip install cuml-cu11 (CUDA 11)
  • 支持的GPU: NVIDIA GPU with CUDA Compute Capability 6.0+
  • 内存要求: 建议8GB+显存用于大数据集
  • 回退机制: 无GPU或cuML时自动使用CPU (scikit-learn)

💡 性能提示:
  • 小数据集(<1000样本): CPU和GPU性能差异不大
  • 中等数据集(1000-10000): GPU开始显现优势
  • 大数据集(>10000): GPU显著优于CPU，建议使用
        '''
    )
    
    parser.add_argument(
        '--cpu', 
        action='store_true',
        help='强制使用CPU模式 (scikit-learn)'
    )
    
    parser.add_argument(
        '--n_jobs', 
        type=int, 
        metavar='N',
        default=1,
        help='CPU模式下的并行工作数 (默认1)'
    )
    
    return parser.parse_args()


def main():
    """主函数，处理训练和测试数据进行建模与评估"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 80)
    print("🚀 SVM 建模预测系统启动 - GPU加速的支持向量机回归")
    if args.cpu:
        print("💻 强制使用CPU模式")
    if args.n_jobs != 1:
        print(f"🔧 CPU并行度: {args.n_jobs}")
    print("=" * 80)
    
    # 创建进度管理器
    progress_manager = ProgressManager()
    
    try:
        # 检测GPU可用性
        gpu_info = detect_gpu_availability()
        
        # 如果命令行指定了CPU模式，强制使用CPU
        if args.cpu:
            gpu_info['has_gpu'] = False
            gpu_info['cuml_available'] = False
            gpu_info['recommended_task_type'] = 'CPU'
            print("💻 强制使用CPU模式 (scikit-learn)")
        
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
        print(f"运行模式: {'GPU加速 (cuML)' if gpu_info['has_gpu'] and gpu_info['cuml_available'] else 'CPU (scikit-learn)'}")
        
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
                    path_manager, gpu_info, progress_manager, model_name="SVM"
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
            summary_file = f'SVM_Train_Test_Results_{timestamp}.csv'
            summary_path = path_manager.get_file_path('summaries', summary_file, create=False)
            print(f"\n✅ 汇总结果已实时保存至: {summary_path}")
            
            print(f"\n📊 训练统计:")
            print(f"总文件数: {len(all_metrics)}")
            print(f"成功处理: {len(all_metrics)}")
            
            # 计算GPU使用统计
            gpu_used_count = sum(1 for df in all_metrics if 'gpu_used' in df.columns and df['gpu_used'].iloc[0])
            total_count = len(all_metrics)
            
            print(f"GPU加速训练: {gpu_used_count}")
            print(f"CPU训练: {total_count - gpu_used_count}")
            print(f"GPU使用率: {gpu_used_count/total_count*100:.1f}%" if total_count > 0 else "GPU使用率: 0%")
            
            # 计算平均性能
            if all_metrics:
                avg_train_r2 = np.mean([df['train_r2'].iloc[0] for df in all_metrics])
                avg_test_r2 = np.mean([df['test_r2'].iloc[0] for df in all_metrics])
                avg_C = np.mean([float(df['best_C'].iloc[0]) for df in all_metrics 
                               if df['best_C'].iloc[0] != 'N/A'])
                print(f"平均训练R²: {avg_train_r2:.3f}")
                print(f"平均测试R²: {avg_test_r2:.3f}")
                print(f"平均最佳C参数: {avg_C:.3f}")
                print(f"🎯 极简SVM: 仅调优最重要的C参数，其他使用默认值")
            
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