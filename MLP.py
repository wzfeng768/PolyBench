#!/usr/bin/env python3
"""
MLPRegressor 神经网络训练系统 - 精简版本
基于GBDT版本改进，使用sklearn的MLPRegressor进行回归建模
专注于最重要的参数调优：hidden_layer_sizes, alpha, learning_rate_init
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline, Pipeline
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


def get_optimal_mlp_params(data_size=None, n_features=None):
    """根据数据大小和特征数获取最优的MLPRegressor参数 - 专注3个核心参数"""
    
    # 基础参数配置 - 固定不变的参数
    base_params = {
        'random_state': 42,
        'max_iter': 1000,  # 最大迭代次数
        'solver': 'adam',  # 优化器
        'activation': 'relu',  # 激活函数
    }
    
    # 根据数据大小和特征数智能调整最重要的3个参数
    if data_size and n_features:
        print(f"📊 根据数据大小 ({data_size}) 和特征数 ({n_features}) 优化MLP参数...")
        
        # 根据特征数调整隐藏层大小
        base_hidden_size = min(max(n_features // 2, 50), 200)
        
        if data_size <= 500:
            base_params.update({
                'hidden_layer_sizes': (base_hidden_size,),
                'alpha': 0.001,  # 较小的正则化
                'learning_rate_init': 0.01
            })
            print("🔹 使用极小数据集配置: 单隐藏层、小正则化、高学习率")
            
        elif data_size <= 1000:
            base_params.update({
                'hidden_layer_sizes': (base_hidden_size, base_hidden_size//2),
                'alpha': 0.0001,
                'learning_rate_init': 0.005
            })
            print("🔹 使用小数据集配置: 双隐藏层、适中参数")
            
        elif data_size <= 3000:
            base_params.update({
                'hidden_layer_sizes': (base_hidden_size, base_hidden_size//2),
                'alpha': 0.0001,
                'learning_rate_init': 0.001
            })
            print("🔹 使用中小数据集配置: 标准双层网络")
            
        elif data_size <= 8000:
            base_params.update({
                'hidden_layer_sizes': (base_hidden_size, base_hidden_size//2, base_hidden_size//4),
                'alpha': 0.00001,
                'learning_rate_init': 0.001
            })
            print("🔹 使用中等数据集配置: 三隐藏层、低正则化")
            
        else:
            base_params.update({
                'hidden_layer_sizes': (base_hidden_size*2, base_hidden_size, base_hidden_size//2),
                'alpha': 0.00001,
                'learning_rate_init': 0.0005
            })
            print("🔹 使用大数据集配置: 深度网络、精细调优")
    else:
        # 默认中等配置
        base_params.update({
            'hidden_layer_sizes': (100, 50),
            'alpha': 0.0001,
            'learning_rate_init': 0.001
        })
        print("🔹 使用默认配置")
    
    print(f"🧠 MLP参数: hidden_layer_sizes={base_params['hidden_layer_sizes']}, "
          f"alpha={base_params['alpha']}, learning_rate_init={base_params['learning_rate_init']}")
    
    return base_params


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


def create_pipeline(base_params, method_name):
    """创建预处理和模型的pipeline"""
    steps = []
    
    # 仅对Mordred数据进行标准化，指纹数据不标准化
    if 'mordred' in method_name.lower():
        print("🔧 检测到Mordred数据，添加特征标准化步骤")
        steps.append(('scaler', StandardScaler()))
    else:
        print("🔧 指纹数据无需标准化，直接使用原始特征")
    
    # 添加MLPRegressor模型
    steps.append(('mlp', MLPRegressor(**base_params)))
    
    return Pipeline(steps)


def train_model_with_cv(X_train, y_train, progress_manager=None, method_name=""):
    """使用5折交叉验证训练MLP回归模型 - 专注3个核心参数"""
    
    base_params = get_optimal_mlp_params(len(X_train), X_train.shape[1])
    
    # 精简的参数网格搜索 - 只搜索3个最重要参数
    data_size = len(X_train)
    n_features = X_train.shape[1]
    print(f"🔍 为数据大小 {data_size} 和特征数 {n_features} 设计精简参数网格...")
    
    # 基础隐藏层大小
    base_hidden = min(max(n_features // 2, 50), 200)
    
    if data_size <= 500:
        param_grid = {
            'mlp__hidden_layer_sizes': [(base_hidden,)],
            'mlp__alpha': [0.001],
            'mlp__learning_rate_init': [0.01]
        }
        print("🔹 极小数据集参数网格: 1组合 (极速模式)")
        
    elif data_size <= 1000:
        param_grid = {
            'mlp__hidden_layer_sizes': [(base_hidden,), (base_hidden, base_hidden//2)],
            'mlp__alpha': [0.001, 0.0001],
            'mlp__learning_rate_init': [0.01, 0.005]
        }
        print("🔹 小数据集参数网格: 8组合")
        
    elif data_size <= 3000:
        param_grid = {
            'mlp__hidden_layer_sizes': [(base_hidden, base_hidden//2), (base_hidden,)],
            'mlp__alpha': [0.0001, 0.00001],
            'mlp__learning_rate_init': [0.005, 0.001]
        }
        print("🔹 中小数据集参数网格: 8组合")
        
    else:
        param_grid = {
            'mlp__hidden_layer_sizes': [(base_hidden, base_hidden//2), (base_hidden*2, base_hidden, base_hidden//2)],
            'mlp__alpha': [0.0001, 0.00001],
            'mlp__learning_rate_init': [0.001, 0.0005]
        }
        print("🔹 大数据集参数网格: 8组合")
    
    total_combinations = (len(param_grid['mlp__hidden_layer_sizes']) * 
                         len(param_grid['mlp__alpha']) * 
                         len(param_grid['mlp__learning_rate_init']))
    
    if progress_manager:
        grid_pbar = progress_manager.create_progress_bar(
            'grid_search', 
            total_combinations,
            desc="MLP网格搜索"
        )
    
    # 记录训练开始前的资源状态
    start_resources = monitor_system_resources()
    print(f"训练开始 - CPU: {start_resources['cpu_percent']:.1f}%, "
          f"内存: {start_resources['memory_percent']:.1f}%, "
          f"可用内存: {start_resources['memory_available_gb']:.1f}GB")
    
    # 创建pipeline
    pipeline = create_pipeline(base_params, method_name)
    
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
    
    print(f"开始MLP网格搜索 - 参数组合数: {total_combinations}")
    print("⚠️ 神经网络训练可能需要较长时间，请耐心等待...")
    print("🔧 已关闭早停机制，训练将运行完整的迭代次数")
    
    try:
        if progress_manager:
            progress_manager.update_progress_bar('grid_search', 0, {'状态': '开始训练'})
        
        # 添加详细的错误信息输出
        print(f"开始训练 - 数据形状: {X_train.shape}, 目标变量范围: [{y_train.min():.3f}, {y_train.max():.3f}]")
        
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
        
        print(f"✅ MLP训练成功！最佳交叉验证R²得分: {grid_search.best_score_:.4f}")
        print(f"最佳参数: {best_params}")
        
        # 输出训练过程信息
        final_mlp = best_model.named_steps['mlp']
        print(f"📊 训练统计: 迭代次数={final_mlp.n_iter_}, 损失={final_mlp.loss_:.6f}")
        
        return best_model, cv_scores, best_params, training_resources
        
    except Exception as e:
        if progress_manager:
            progress_manager.close_progress_bar('grid_search')
            
        print(f"❌ MLP训练失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        
        # 如果出现错误，尝试使用更保守的参数重新训练
        print("🔄 尝试使用更保守的MLP参数重新训练...")
        
        # 使用更保守的参数
        conservative_params = base_params.copy()
        conservative_params.update({
            'hidden_layer_sizes': (50,),  # 简单的单层网络
            'alpha': 0.01,  # 更强的正则化
            'learning_rate_init': 0.01,  # 较高的学习率
            'max_iter': 500  # 减少最大迭代次数
        })
        
        print(f"保守参数: {conservative_params}")
        
        pipeline = create_pipeline(conservative_params, method_name)
        
        # 简化的参数网格
        simple_param_grid = {
            'mlp__hidden_layer_sizes': [conservative_params['hidden_layer_sizes']],
            'mlp__alpha': [conservative_params['alpha']],
            'mlp__learning_rate_init': [conservative_params['learning_rate_init']]
        }
        
        grid_search = GridSearchCV(
            pipeline, 
            simple_param_grid, 
            cv=kf, 
            scoring='r2', 
            n_jobs=1,
            verbose=0
        )
        
        try:
            print("🔄 开始保守参数训练...")
            grid_search.fit(X_train, y_train)
            
            end_resources = monitor_system_resources()
            training_resources = {
                'start_resources': start_resources,
                'end_resources': end_resources,
                'max_memory_used': max(start_resources['memory_percent'], end_resources['memory_percent']),
                'fallback_to_conservative': True
            }
            
            print(f"✅ 保守参数训练成功！R²得分: {grid_search.best_score_:.4f}")
            return grid_search.best_estimator_, grid_search.cv_results_, grid_search.best_params_, training_resources
            
        except Exception as conservative_e:
            print(f"❌ 保守参数训练也失败: {str(conservative_e)}")
            raise conservative_e


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
        'best_hidden_layer_sizes': [str(best_params.get('mlp__hidden_layer_sizes', 'N/A'))],
        'best_alpha': [best_params.get('mlp__alpha', 'N/A')],
        'best_learning_rate_init': [best_params.get('mlp__learning_rate_init', 'N/A')],
        'training_time': [training_time],
        'max_memory_used_percent': [training_resources.get('max_memory_used', 'N/A')],
        'fallback_to_conservative': [training_resources.get('fallback_to_conservative', False)]
    }
    
    metrics_df = pd.DataFrame(metrics_dict)
    
    metrics_file = f"{method}_{model_name}_{file_name}_metrics.csv"
    metrics_path = path_manager.get_file_path('metrics', metrics_file, method, model_name)
    
    metrics_df.to_csv(metrics_path, index=False, float_format='%.6f')
    print(f"评估指标保存至: {metrics_path}")
    
    # 显示资源使用摘要
    print(f"资源使用摘要:")
    print(f"  - 最大内存使用: {training_resources.get('max_memory_used', 'N/A')}%")
    
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
    
    summary_file = f'MLP_Train_Test_Results_{timestamp}.csv'
    summary_path = path_manager.get_file_path('summaries', summary_file, create=True)
    
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"汇总评估指标保存至: {summary_path}")
    
    return summary_df


def append_summary_metrics(method_metrics, path_manager, timestamp, method_name):
    """追加单个方法的评估指标到汇总文件"""
    if not method_metrics:
        print(f"⚠️ {method_name} 方法没有指标可保存")
        return None
    
    summary_file = f'MLP_Train_Test_Results_{timestamp}.csv'
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


def process_method_data(method, train_folder, test_folder, path_manager, progress_manager=None, model_name="MLP"):
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
            print("开始5折交叉验证MLP训练...")
            best_model, cv_scores, best_params, training_resources = train_model_with_cv(
                X_train, y_train, progress_manager, method
            )
            
            training_time = time.time() - start_time
            print(f"MLP训练完成，总耗时: {training_time:.2f} 秒")
            
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
    """解析命令行参数 - 简化版"""
    parser = argparse.ArgumentParser(
        description='MLPRegressor 神经网络训练系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python MLP_Fixed_Clean.py                    # 标准运行
        '''
    )
    
    return parser.parse_args()


def main():
    """主函数，处理训练和测试数据进行建模与评估"""
    # 解析命令行参数
    args = parse_arguments()
    
    print("=" * 80)
    print("🧠 MLPRegressor 神经网络训练系统启动 (精简版)")
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
            'Torsion_2048': ('Fingers_/Torsion_2048', 'Fingers_/Torsion_2048')
            # Mordred 描述符
            # 'Mordred': ('Mordred', 'Mordred')
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
        print("⚠️ 注意: MLPRegressor使用CPU训练，训练时间可能较长")
        
        # 创建总体进度条
        overall_progress = progress_manager.create_progress_bar(
            'overall',
            len(methods),
            desc="总体进度"
        )
        
        # 处理每种方法的数据
        for method_idx, (method, (train_subfolder, test_subfolder)) in enumerate(methods.items()):
            print(f"\n{'='*80}")
            print(f"🧠 开始处理 {method} 数据 (MLP)... ({method_idx+1}/{len(methods)})")
            print(f"{'='*80}")
            
            try:
                train_folder = os.path.join(train_base, train_subfolder)
                test_folder = os.path.join(test_base, test_subfolder)
                
                # 处理该方法的数据（带进度条）
                method_metrics = process_method_data(
                    method, train_folder, test_folder, 
                    path_manager, progress_manager, model_name="MLP"
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
            summary_file = f'MLP_Train_Test_Results_{timestamp}.csv'
            summary_path = path_manager.get_file_path('summaries', summary_file, create=False)
            print(f"\n✅ MLP汇总结果已实时保存至: {summary_path}")
            
            # 统计训练情况
            conservative_count = sum(1 for df in all_metrics if 'fallback_to_conservative' in df.columns and df['fallback_to_conservative'].iloc[0])
            total_count = len(all_metrics)
            
            print(f"\n📊 MLP训练统计:")
            print(f"总文件数: {total_count}")
            print(f"标准参数训练: {total_count - conservative_count}")
            print(f"保守参数训练: {conservative_count}")
            print(f"成功率: {(total_count - conservative_count)/total_count*100:.1f}%" if total_count > 0 else "成功率: 0%")
            
        else:
            print("⚠️ 警告: 没有成功处理任何数据，未保存汇总评估指标")
        
        print(f"\n🎉 所有MLP处理完成！")
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
    print("所有MLP处理完成！") 