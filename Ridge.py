#!/usr/bin/env python3
"""
Ridge回归建模系统 - 基于CatBoost系统改进
特别优化了Mordred数据的StandardScaler处理
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
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


def get_optimal_ridge_params(data_size=None, method_name=None):
    """根据数据大小和方法类型获取最优的Ridge参数"""
    
    # 基础参数配置
    base_params = {
        'random_state': 42,
        'max_iter': 10000,  # 增加最大迭代次数确保收敛
        'solver': 'auto'
    }
    
    # 根据数据大小智能调整参数网格
    print(f"📊 根据数据大小 ({data_size}) 和方法 ({method_name}) 优化参数...")
    
    # 检查是否是Mordred数据 - 需要更大的正则化范围
    is_mordred = method_name and 'mordred' in method_name.lower()
    
    if data_size:
        if data_size <= 500:
            if is_mordred:
                alpha_range = [0.1, 1.0, 10.0, 100.0, 1000.0]
                print("🔹 极小Mordred数据集: 使用较大的正则化范围")
            else:
                alpha_range = [0.01, 0.1, 1.0, 10.0, 100.0]
                print("🔹 极小指纹数据集: 使用标准正则化范围")
                
        elif data_size <= 1000:
            if is_mordred:
                alpha_range = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                print("🔹 小Mordred数据集: 使用扩展正则化范围")
            else:
                alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                print("🔹 小指纹数据集: 使用标准正则化范围")
                
        elif data_size <= 3000:
            if is_mordred:
                alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                print("🔹 中小Mordred数据集: 使用中等正则化范围")
            else:
                alpha_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
                print("🔹 中小指纹数据集: 使用细粒度正则化范围")
                
        elif data_size <= 8000:
            if is_mordred:
                alpha_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
                print("🔹 中等Mordred数据集: 使用标准正则化范围")
            else:
                alpha_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
                print("🔹 中等指纹数据集: 使用精细正则化范围")
                
        else:
            if is_mordred:
                alpha_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
                print("🔹 大Mordred数据集: 使用精细正则化范围")
            else:
                alpha_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
                print("🔹 大指纹数据集: 使用超精细正则化范围")
    else:
        # 默认中等配置
        if is_mordred:
            alpha_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            print("🔹 默认Mordred配置: 使用标准正则化范围")
        else:
            alpha_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            print("🔹 默认指纹配置: 使用标准正则化范围")
    
    # 创建参数网格
    param_grid = {
        'ridge__alpha': alpha_range
    }
    
    print(f"Alpha值范围: {alpha_range}")
    print(f"参数组合总数: {len(alpha_range)}")
    
    return base_params, param_grid


def monitor_system_resources():
    """监控系统资源使用情况"""
    resources = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }
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
    """分离输入特征和目标变量，并根据方法类型决定是否需要标准化
    
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
    
    # 检查是否需要处理Mordred数据的缺失值和异常值
    is_mordred = method_name and 'mordred' in method_name.lower()
    if is_mordred:
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
    
    return X, y, feature_names


def create_pipeline(base_params, method_name=None):
    """根据方法类型创建合适的Pipeline"""
    is_mordred = method_name and 'mordred' in method_name.lower()
    
    if is_mordred:
        # Mordred数据需要标准化
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(**base_params))
        ])
        print("🔧 创建包含StandardScaler的Pipeline (Mordred数据)")
    else:
        # 指纹数据通常不需要标准化
        pipeline = Pipeline([
            ('ridge', Ridge(**base_params))
        ])
        print("🔧 创建简单的Ridge Pipeline (指纹数据)")
    
    return pipeline


def train_model_with_cv(X_train, y_train, method_name=None, progress_manager=None):
    """使用5折交叉验证训练Ridge回归模型"""
    
    base_params, param_grid = get_optimal_ridge_params(len(X_train), method_name)
    
    total_combinations = len(param_grid['ridge__alpha'])
    
    if progress_manager:
        grid_pbar = progress_manager.create_progress_bar(
            'grid_search', 
            total_combinations,
            desc=f"Ridge网格搜索"
        )
    
    # 记录训练开始前的资源状态
    start_resources = monitor_system_resources()
    print(f"训练开始 - CPU: {start_resources['cpu_percent']:.1f}%, "
          f"内存: {start_resources['memory_percent']:.1f}%, "
          f"可用内存: {start_resources['memory_available_gb']:.1f}GB")
    
    # 创建Pipeline
    pipeline = create_pipeline(base_params, method_name)
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 使用标准GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=kf, 
        scoring='r2', 
        n_jobs=-1,  # 使用所有CPU核心
        verbose=0
    )
    
    print(f"开始网格搜索 - 参数组合数: {total_combinations}")
    print(f"使用Ridge回归模型")
    
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
        print(f"Ridge训练失败: {str(e)}")
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
        'best_alpha': [best_params.get('ridge__alpha', 'N/A')],
        'training_time': [training_time],
        'max_memory_used_percent': [training_resources.get('max_memory_used', 'N/A')],
        'uses_scaler': [method and 'mordred' in method.lower()]
    }
    
    metrics_df = pd.DataFrame(metrics_dict)
    
    metrics_file = f"{method}_{model_name}_{file_name}_metrics.csv"
    metrics_path = path_manager.get_file_path('metrics', metrics_file, method, model_name)
    
    metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
    print(f"评估指标保存至: {metrics_path}")
    
    # 显示资源使用摘要
    print(f"资源使用摘要:")
    print(f"  - 最大内存使用: {training_resources.get('max_memory_used', 'N/A')}%")
    print(f"  - 最佳Alpha值: {best_params.get('ridge__alpha', 'N/A')}")
    
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
    
    summary_file = f'Ridge_Train_Test_Results_{timestamp}.csv'
    summary_path = path_manager.get_file_path('summaries', summary_file, create=True)
    
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"汇总评估指标保存至: {summary_path}")
    
    return summary_df


def append_summary_metrics(method_metrics, path_manager, timestamp, method_name):
    """追加单个方法的评估指标到汇总文件"""
    if not method_metrics:
        print(f"⚠️ {method_name} 方法没有指标可保存")
        return None
    
    summary_file = f'Ridge_Train_Test_Results_{timestamp}.csv'
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


def process_method_data(method, train_folder, test_folder, path_manager, progress_manager=None, model_name="Ridge"):
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
            X_train, y_train, feature_names = prepare_data(train_data, method)
            
            # 准备测试数据（使用训练集确定的特征集）
            X_test, y_test, _ = prepare_data(test_data, method, features_to_keep=feature_names)
            
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
                X_train, y_train, method, progress_manager
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
        description='Ridge回归建模系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python Ridge_Fixed_Clean.py                    # 默认运行
  python Ridge_Fixed_Clean.py --n_jobs=4         # 指定CPU核心数

特性:
  - Mordred数据自动使用StandardScaler
  - 指纹数据使用原始特征
  - 智能参数网格搜索
        '''
    )
    
    parser.add_argument(
        '--n_jobs', 
        type=int, 
        default=-1,
        help='并行处理的CPU核心数 (默认: -1, 使用所有核心)'
    )
    
    return parser.parse_args()


def main():
    """主函数，处理训练和测试数据进行建模与评估"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 80)
    print("🚀 Ridge回归建模系统启动")
    print("📊 特色功能: Mordred数据自动标准化处理")
    print("=" * 80)
    
    # 创建进度管理器
    progress_manager = ProgressManager()
    
    try:
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
            # 'AtomPair_512': ('Fingers_/Atompair_512', 'Fingers_/Atompair_512'),
            # 'AtomPair_1024': ('Fingers_/Atompair_1024', 'Fingers_/Atompair_1024'),
            # 'AtomPair_2048': ('Fingers_/Atompair_2048', 'Fingers_/Atompair_2048'),
            # 'Maccs': ('Fingers_/Maccs', 'Fingers_/Maccs'),
            # 'Morgan_512': ('Fingers_/Morgan_512', 'Fingers_/Morgan_512'),
            # 'Morgan_1024': ('Fingers_/Morgan_1024', 'Fingers_/Morgan_1024'),
            # 'Morgan_2048': ('Fingers_/Morgan_2048', 'Fingers_/Morgan_2048'),
            # 'RDKit_512': ('Fingers_/RDKit_512', 'Fingers_/RDKit_512'),
            # 'RDKit_1024': ('Fingers_/RDKit_1024', 'Fingers_/RDKit_1024'),
            # 'RDKit_2048': ('Fingers_/RDKit_2048', 'Fingers_/RDKit_2048'),
            # 'Torsion_512': ('Fingers_/Torsion_512', 'Fingers_/Torsion_512'),
            # 'Torsion_1024': ('Fingers_/Torsion_1024', 'Fingers_/Torsion_1024'),
            # 'Torsion_2048': ('Fingers_/Torsion_2048', 'Fingers_/Torsion_2048'),
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
        print(f"并行核心数: {args.n_jobs if args.n_jobs > 0 else psutil.cpu_count()}")
        
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
            if 'mordred' in method.lower():
                print("🧪 检测到Mordred数据，将自动应用StandardScaler")
            else:
                print("🔍 检测到指纹数据，使用原始特征")
            print(f"{'='*80}")
            
            try:
                train_folder = os.path.join(train_base, train_subfolder)
                test_folder = os.path.join(test_base, test_subfolder)
                
                # 处理该方法的数据（带进度条）
                method_metrics = process_method_data(
                    method, train_folder, test_folder, 
                    path_manager, progress_manager, model_name="Ridge"
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
            summary_file = f'Ridge_Train_Test_Results_{timestamp}.csv'
            summary_path = path_manager.get_file_path('summaries', summary_file, create=False)
            print(f"\n✅ 汇总结果已实时保存至: {summary_path}")
            
            # 统计Scaler使用情况
            scaler_used_count = sum(1 for df in all_metrics if 'uses_scaler' in df.columns and df['uses_scaler'].iloc[0])
            total_count = len(all_metrics)
            
            print(f"\n📊 训练统计:")
            print(f"总文件数: {total_count}")
            print(f"使用StandardScaler: {scaler_used_count} (Mordred数据)")
            print(f"使用原始特征: {total_count - scaler_used_count} (指纹数据)")
            
        else:
            print("⚠️ 警告: 没有成功处理任何数据，未保存汇总评估指标")
        
        print(f"\n🎉 所有处理完成！")
        print("=" * 80)
        
        return {
            'all_metrics': all_metrics,
            'methods_processed': list(methods.keys()),
            'timestamp': timestamp
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