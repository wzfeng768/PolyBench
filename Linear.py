import os
import re
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
from datetime import datetime
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 路径管理类
class PathManager:
    """
    统一管理所有输出路径的类
    """
    def __init__(self, base_dir=None):
        """
        初始化路径管理器
        
        Parameters:
        ----------
        base_dir : str, optional
            基础目录，如果为None则使用当前目录
        """
        self.base_dir = base_dir or os.getcwd()
        
        # 定义默认的路径结构
        self.paths = {
            'results': os.path.join(self.base_dir, 'Results'),
            'models': os.path.join(self.base_dir, 'Models'),
            'images': os.path.join(self.base_dir, 'Images'),
            'shap': os.path.join(self.base_dir, 'SHAP_Results'),
            'metrics': os.path.join(self.base_dir, 'Metrics'),
            'summaries': os.path.join(self.base_dir, 'Summaries'),
            # [改进 3.1] 增加SHAP值保存路径
            'shap_values': os.path.join(self.base_dir, 'SHAP_Values')
        }
    
    def get_path(self, path_type, method=None, model=None, create=True):
        """
        获取特定类型的路径
        
        Parameters:
        ----------
        path_type : str
            路径类型 ('results', 'models', 'images', 'shap', 'metrics', 'summaries')
        method : str, optional
            方法名称 (如 'AtomPair_512', 'Morgan_1024')
        model : str, optional
            模型名称 (如 'Linear')
        create : bool, optional
            是否创建文件夹，默认为True
            
        Returns:
        -------
        str
            构建好的路径
        """
        if path_type not in self.paths:
            raise ValueError(f"Unknown path type: {path_type}")
        
        path = self.paths[path_type]
        
        # 添加模型名称子目录
        if model:
            path = os.path.join(path, model)
            
        # 添加方法名称子目录
        if method:
            path = os.path.join(path, method)
            
        # 创建目录
        if create and not os.path.exists(path):
            os.makedirs(path)
            
        return path
    
    def get_file_path(self, path_type, filename, method=None, model=None, create=True):
        """
        获取文件完整路径
        
        Parameters:
        ----------
        path_type : str
            路径类型
        filename : str
            文件名
        method : str, optional
            方法名称
        model : str, optional
            模型名称
        create : bool, optional
            是否创建目录，默认为True
            
        Returns:
        -------
        str
            文件的完整路径
        """
        directory = self.get_path(path_type, method, model, create)
        return os.path.join(directory, filename)


# [改进 2.1] 提取不包含扩展名的文件名
def get_base_filename(filename):
    """
    从文件名中移除扩展名
    
    Parameters:
    ----------
    filename : str
        原始文件名
        
    Returns:
    -------
    str
        没有扩展名的文件名
    """
    return os.path.splitext(filename)[0]


# 辅助函数 - 读取文件
def read_files_in_folder(folder_path):   
    """
    读取指定文件夹中的所有CSV文件，并返回DataFrame列表、文件名列表和最后一列列名列表。
    
    Parameters:
    ----------
    folder_path : str
        包含CSV文件的文件夹路径
        
    Returns:
    -------
    tuple
        (dataframes, filenames, last_column_names)
    """
    dataframes = []
    filenames = []
    last_column_names = []  # 用于存储每个CSV文件的最后一列列名

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            print(f"读取文件: {filename}")
            # 读取文件
            df = pd.read_csv(file_path)

            dataframes.append(df)
            filenames.append(filename)
            
            # 获取最后一列的列名
            last_column_name = df.columns[-1]
            if last_column_name == 'property_log':
                # 如果最后一列列名为 'property_log'，则返回倒数第二列列名
                last_column_name = df.columns[-2]
            last_column_names.append(last_column_name)
            print(f"文件 {filename} 的最后一列列名为: {last_column_name}")

    return dataframes, filenames, last_column_names


# 辅助函数 - 数据准备
def prepare_data(data):    
    """
    分离输入特征和目标变量
    
    Parameters:
    ----------
    data : pandas.DataFrame
        输入数据
        
    Returns:
    -------
    tuple
        (X, y, feature_names)
    """
    if data.columns[-1] == 'property_log':
        X = data.drop(columns=[data.columns[-1], data.columns[-2]])
        y = data[data.columns[-1]]
    else:
        X = data.drop(columns=[data.columns[-1]])
        y = data[data.columns[-1]]

    feature_names = X.columns.tolist()  # 保存特征名称

    return X, y, feature_names


# 模型训练与评估相关函数
def train_model(X_train, y_train):
    """
    训练Linear回归模型
    
    Parameters:
    ----------
    X_train : pandas.DataFrame
        训练特征数据
    y_train : pandas.Series
        训练目标变量
        
    Returns:
    -------
    tuple
        (best_model, best_alpha)
    """
    # 创建LinearRegression模型
    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # LinearRegression没有alpha参数，返回None作为最佳参数
    return pipeline, None


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    评估模型性能
    
    Parameters:
    ----------
    model : sklearn estimator
        训练好的模型
    X_train : pandas.DataFrame
        训练特征数据
    y_train : pandas.Series
        训练目标变量
    X_test : pandas.DataFrame
        测试特征数据
    y_test : pandas.Series
        测试目标变量
        
    Returns:
    -------
    dict
        评估指标字典
    """
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算评估指标
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }
    
    # 保存预测结果
    prediction_data = {
        'train': {'true': y_train, 'pred': y_train_pred},
        'test': {'true': y_test, 'pred': y_test_pred}
    }
    
    return metrics, prediction_data


# [改进 3.2] 修改SHAP分析函数，单独保存SHAP值
def perform_shap_analysis(model, X_train, X_test, feature_names, path_manager, method, model_name, file_name):
    """
    执行SHAP值分析并保存结果，仅计算前20个重要特征的SHAP值
    
    Parameters:
    ----------
    model : sklearn estimator
        训练好的模型
    X_train : pandas.DataFrame
        训练特征数据
    X_test : pandas.DataFrame
        测试特征数据
    feature_names : list
        特征名称列表
    path_manager : PathManager
        路径管理器实例
    method : str
        使用的方法名称
    model_name : str
        模型名称
    file_name : str
        输入文件名（不含扩展名）
        
    Returns:
    -------
    numpy.ndarray
        SHAP值
    """
    # 首先，通过线性模型的系数直接识别最重要的特征
    # 这样我们可以先筛选特征，然后只计算重要特征的SHAP值
    linear_model = model.named_steps['linearregression']
    coef_importance = np.abs(linear_model.coef_)
    
    # 获取前20个最重要特征的索引
    top_indices = np.argsort(coef_importance)[::-1][:20]
    
    # 筛选出这些特征
    X_train_top = X_train.iloc[:, top_indices]
    X_test_top = X_test.iloc[:, top_indices]
    feature_names_top = [feature_names[i] for i in top_indices]
    
    # 使用最新的SHAP API创建解释器
    # 注意：我们不使用已弃用的feature_perturbation参数
    try:
        # 首先尝试使用新的API
        import shap
        masker = shap.maskers.Independent(X_train_top)
        explainer = shap.LinearExplainer(
            (linear_model.coef_[top_indices], linear_model.intercept_), 
            masker
        )
    except:
        # 如果新API不可用，则回退到旧版本但不使用已弃用的参数
        explainer = shap.LinearExplainer(
            (linear_model.coef_[top_indices], linear_model.intercept_), 
            X_train_top
        )
    
    # 只计算筛选后的特征的SHAP值
    shap_values_top = explainer.shap_values(X_test_top)
    
    # 绘制并保存SHAP图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_top, X_test_top, feature_names=feature_names_top, show=False)
    
    # 构建SHAP图保存路径
    shap_file = f"{method}_{model_name}_{file_name}_shap_summary.png"
    shap_path = path_manager.get_file_path('shap', shap_file, method, model_name)
    
    plt.savefig(shap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存SHAP值数据
    shap_data = {
        'shap_values_top': shap_values_top,
        'feature_names_top': feature_names_top,
        'X_test_top': X_test_top.values,
        'top_indices': top_indices,
        'coef_importance': coef_importance
    }
    
    shap_data_file = f"{method}_{model_name}_{file_name}_shap_values.pkl"
    shap_data_path = path_manager.get_file_path('shap_values', shap_data_file, method, model_name)
    
    joblib.dump(shap_data, shap_data_path)
    
    print(f"SHAP分析结果保存至: {shap_path} (仅显示前20个最重要特征)")
    print(f"SHAP值数据保存至: {shap_data_path}")
    
    # 构造一个与原始输出格式兼容的结果
    # 注意：这里我们只返回筛选后的结果，不计算全部特征的SHAP值
    full_shap_values = np.zeros((X_test.shape[0], X_test.shape[1]))
    for i, idx in enumerate(top_indices):
        full_shap_values[:, idx] = shap_values_top[:, i]
    
    return full_shap_values




# [改进 3.3] 添加根据保存的SHAP值绘图的函数
def plot_shap_from_saved_data(shap_data_path, output_path=None, max_features=20):
    """
    从保存的SHAP值数据绘制SHAP摘要图
    
    Parameters:
    ----------
    shap_data_path : str
        包含SHAP值数据的文件路径
    output_path : str, optional
        输出图片路径，默认与输入路径相同但扩展名为.png
    max_features : int, optional
        要显示的最大特征数量，默认为20
        
    Returns:
    -------
    None
    """
    # 加载SHAP值数据
    shap_data = joblib.load(shap_data_path)
    
    # 如果数据中已包含前20个特征的信息，直接使用
    if 'shap_values_top' in shap_data and 'feature_names_top' in shap_data:
        shap_values = shap_data['shap_values_top']
        feature_names = shap_data['feature_names_top']
        X_test = shap_data['X_test'][:, shap_data['top_indices']]
    else:
        # 否则，处理完整数据并筛选前N个特征
        shap_values = shap_data['shap_values']
        feature_names = shap_data['feature_names']
        X_test = shap_data['X_test']
        
        # 计算特征重要性
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # 获取重要性排序的索引
        importance_indices = np.argsort(feature_importance)[::-1]  # 降序排列
        
        # 只保留前N个最重要特征的索引
        top_indices = importance_indices[:max_features]
        
        # 筛选前N个最重要特征的数据
        X_test = X_test[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        shap_values = shap_values[:, top_indices]
    
    # 设置默认输出路径
    if output_path is None:
        output_path = os.path.splitext(shap_data_path)[0] + "_replot.png"
    
    # 绘制SHAP摘要图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names, 
        show=False
    )
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"重新绘制的SHAP摘要图保存至: {output_path} (显示前{len(feature_names)}个最重要特征)")



def save_fold_results(fold_results, path_manager, method, model_name, file_name):
    """
    保存每折的详细结果
    
    Parameters:
    ----------
    fold_results : list
        包含每折详细结果的列表
    path_manager : PathManager
        路径管理器实例
    method : str
        使用的方法名称
    model_name : str
        模型名称
    file_name : str
        输入文件名（不含扩展名）
        
    Returns:
    -------
    pandas.DataFrame
        每折详细结果DataFrame
    """
    # 创建每折详细结果的DataFrame
    fold_df = pd.DataFrame(fold_results)
    fold_df['method'] = method
    fold_df['model'] = model_name
    fold_df['file_name'] = file_name
    
    # 重排列列顺序
    fold_columns = ['method', 'model', 'file_name', 'fold', 
                   'train_r2', 'train_mse', 'train_mae',
                   'test_r2', 'test_mse', 'test_mae',
                   'best_alpha']
    fold_df = fold_df[fold_columns]
    
    # 保存为CSV文件
    fold_file = f"{method}_{model_name}_{file_name}_fold_metrics.csv"
    fold_path = path_manager.get_file_path('metrics', fold_file, method, model_name)
    
    fold_df.to_csv(fold_path, index=False, float_format='%.6f')
    print(f"每折详细指标保存至: {fold_path}")
    
    return fold_df


def save_average_metrics(avg_metrics, best_alpha, training_time, path_manager, method, model_name, file_name):
    """
    保存平均评估指标
    
    Parameters:
    ----------
    avg_metrics : dict
        平均评估指标    
    best_alpha : float
        最佳alpha值(用于Lasso)，LinearRegression中为None
    training_time : float
        训练时间
    path_manager : PathManager
        路径管理器实例
    method : str
        使用的方法名称
    model_name : str
        模型名称
    file_name : str
        输入文件名（不含扩展名）
        
    Returns:
    -------
    pandas.DataFrame
        平均指标DataFrame
    """
    # 创建平均指标的DataFrame
    avg_dict = {
        'file_name': [file_name],
        'method': [method],
        'model': [model_name],
        'avg_train_r2': [avg_metrics['train_r2']],
        'avg_train_mse': [avg_metrics['train_mse']],
        'avg_train_mae': [avg_metrics['train_mae']],
        'avg_test_r2': [avg_metrics['test_r2']],
        'avg_test_mse': [avg_metrics['test_mse']],
        'avg_test_mae': [avg_metrics['test_mae']],
        'avg_best_alpha': [best_alpha if best_alpha is not None else "N/A"],  # LinearRegression无alpha参数
        'training_time': [training_time]
    }
    avg_df = pd.DataFrame(avg_dict)
    
    # 保存为CSV文件
    avg_file = f"{method}_{model_name}_{file_name}_average_metrics.csv"
    avg_path = path_manager.get_file_path('metrics', avg_file, method, model_name)
    
    avg_df.to_csv(avg_path, index=False, float_format='%.6f')
    print(f"平均指标保存至: {avg_path}")
    
    return avg_df


def create_summary_metrics(fold_df, avg_metrics, best_alpha, training_time, path_manager, method, model_name, file_name):
    """
    创建并保存汇总指标
    
    Parameters:
    ----------
    fold_df : pandas.DataFrame
        每折详细指标的DataFrame
    avg_metrics : dict
        平均评估指标    
    best_alpha : float
        最佳alpha值(用于Lasso)，LinearRegression中为None
    training_time : float
        训练时间
    path_manager : PathManager
        路径管理器实例
    method : str
        使用的方法名称
    model_name : str
        模型名称
    file_name : str
        输入文件名（不含扩展名）
        
    Returns:
    -------
    pandas.DataFrame
        汇总指标DataFrame
    """
    # 创建包含所有信息的汇总表
    summary_dict = {
        'Metric': [
            'Average Train R²',
            'Average Train MSE',
            'Average Train MAE',
            'Average Test R²',
            'Average Test MSE',
            'Average Test MAE',
            'Average Best Alpha',  # LinearRegression无此项，但保留为N/A
            'Total Training Time (s)',
            'Best Fold R² (Test)',
            'Worst Fold R² (Test)',
            'Standard Deviation R² (Test)'
        ],
        'Value': [
            avg_metrics['train_r2'],
            avg_metrics['train_mse'],
            avg_metrics['train_mae'],
            avg_metrics['test_r2'],
            avg_metrics['test_mse'],
            avg_metrics['test_mae'],
            "N/A" if best_alpha is None else best_alpha,  # LinearRegression无alpha参数
            training_time,
            max(fold_df['test_r2']),
            min(fold_df['test_r2']),
            np.std(fold_df['test_r2'])
        ]
    }
    
    summary_df = pd.DataFrame(summary_dict)
    summary_file = f"{method}_{model_name}_{file_name}_summary_metrics.csv"
    summary_path = path_manager.get_file_path('summaries', summary_file, method, model_name)
    
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"汇总指标保存至: {summary_path}")
    
    return summary_df


def save_last_fold_data(last_fold_data, path_manager, method, model_name, file_name):
    """
    保存最后一折的预测结果
    
    Parameters:
    ----------
    last_fold_data : dict
        最后一折的预测结果
    path_manager : PathManager
        路径管理器实例
    method : str
        使用的方法名称
    model_name : str
        模型名称
    file_name : str
        输入文件名（不含扩展名）
    """
    # 保存最后一折数据
    for data_type in ['train', 'test']:
        data_df = pd.DataFrame({
            'true_values': last_fold_data[data_type]['true'],
            'predicted_values': last_fold_data[data_type]['pred']
        })
        
        fold_file = f"{method}_{model_name}_{file_name}_last_fold_{data_type}.csv"
        fold_path = path_manager.get_file_path('results', fold_file, method, model_name)
        
        data_df.to_csv(fold_path, index=False)
    
    print(f"最后一折预测结果保存完成")


def cross_validate_and_evaluate(X, y, feature_names, path_manager, method="GNN", model_name="Linear", file_name=""):
    """
    执行交叉验证训练和模型评估
    
    Parameters:
    ----------
    X : pandas.DataFrame
        特征数据
    y : pandas.Series
        目标变量
    feature_names : list
        特征名称列表
    path_manager : PathManager
        路径管理器实例
    method : str, optional
        使用的方法名称，默认为"GNN"
    model_name : str, optional
        模型名称，默认为"Linear"
    file_name : str, optional
        输入文件名（不含扩展名），默认为空字符串
        
    Returns:
    -------
    dict
        包含训练结果的字典
    """
    random_seed = 42
    np.random.seed(random_seed)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    
    # 初始化存储列表
    fold_results = []
    models = []
    best_r2 = -float('inf')
    best_model = None
    last_fold_data = None

    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 训练模型
        model, _ = train_model(X_train, y_train)
        models.append(model)

        # 评估模型
        metrics, prediction_data = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics['fold'] = fold + 1
        metrics['best_alpha'] = None  # LinearRegression没有alpha参数
        fold_results.append(metrics)

        # 保存最佳模型
        if metrics['test_r2'] > best_r2:
            best_r2 = metrics['test_r2']
            best_model = model

        # 如果是最后一折，保存预测结果
        if fold == 4:
            last_fold_data = prediction_data

    # 计算训练时间
    training_time = time.time() - start_time

    # 计算平均指标
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_results]) 
                  for metric in ['train_r2', 'test_r2', 'train_mse', 'test_mse', 'train_mae', 'test_mae']}

    # SHAP分析
    X_train, X_test = X.iloc[kf.split(X).__next__()[0]], X.iloc[kf.split(X).__next__()[1]]
    shap_values = perform_shap_analysis(
        best_model, X_train, X_test, feature_names, 
        path_manager, method, model_name, file_name
    )

    # 保存每折详细结果
    fold_df = save_fold_results(
        fold_results, path_manager, method, model_name, file_name
    )

    # 保存平均指标
    avg_df = save_average_metrics(
        avg_metrics, None, training_time, 
        path_manager, method, model_name, file_name
    )

    # 创建并保存汇总指标
    summary_df = create_summary_metrics(
        fold_df, avg_metrics, None, training_time, 
        path_manager, method, model_name, file_name
    )

    # 保存最后一折数据
    save_last_fold_data(last_fold_data, path_manager, method, model_name, file_name)

    # 保存最佳模型
    model_file = f"{method}_{model_name}_{file_name}_best_model.joblib"
    model_path = path_manager.get_file_path('models', model_file, method, model_name)
    joblib.dump(best_model, model_path)
    print(f"最佳模型保存至: {model_path}")

    # 整合所有结果
    results = {
        'best_model': best_model,
        'metrics': {
            'fold_results': fold_results,
            'average_metrics': avg_metrics,
            'best_alpha': None,
            'training_time': training_time
        },
        'last_fold_data': last_fold_data,
        'shap_values': shap_values,
        'fold_metrics_df': fold_df,
        'average_metrics_df': avg_df,
        'summary_metrics_df': summary_df
    }

    return results


def plot_and_save_scatter(results_dict, path_manager, method, model_name, file_name, last_column_name):
    """
    绘制并保存预测值与真实值的散点图
    
    Parameters:
    ----------
    results_dict: 包含模型训练结果的字典
    path_manager: PathManager 路径管理器实例
    method: 使用的方法名称
    model_name: 模型名称
    file_name: 输入文件名（不含扩展名）
    last_column_name: 目标变量名称
    """
    # 提取数据
    last_fold_data = results_dict['last_fold_data']
    metrics = results_dict['metrics']['average_metrics']
    
    y_train = last_fold_data['train']['true']
    y_train_pred = last_fold_data['train']['pred']
    y_test = last_fold_data['test']['true']
    y_test_pred = last_fold_data['test']['pred']

    # 创建图形
    plt.figure(figsize=(8, 8))

    # 绘制散点图
    plt.scatter(y_train, y_train_pred, s=50, c='#005BAD', alpha=0.7, label='Train')
    plt.scatter(y_test, y_test_pred, s=50, c='#F56476', alpha=0.7, label='Test')

    # 计算数据范围
    min_val = min(min(y_train), min(y_test))
    max_val = max(max(y_train), max(y_test))
    buffer = (max_val - min_val) * 0.02

    # 设置轴范围
    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)

    # 添加理想预测线
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='grey', label='ideal')

    # 处理标题中的变量名
    match = re.match(r'^(.*?)[\[\(].*?[\]\)]$', last_column_name)
    title_name = match.group(1).strip() if match else last_column_name

    # 添加标题和标签
    plt.text(0.5, 0.95, f"{model_name}_{method}_{title_name}", 
             transform=plt.gca().transAxes, fontsize=18,
             verticalalignment='top', horizontalalignment='center')

    plt.tick_params(which='major', direction='in', length=5, labelsize=16)
    plt.xlabel(f"Actual {last_column_name}", fontsize=16)
    plt.ylabel(f"Predicted {last_column_name}", fontsize=16)
    plt.legend(loc='upper left', fontsize=16)

    # 添加评估指标文本
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

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保持比例一致
    plt.gca().set_aspect('equal', adjustable='box')

    # 保存图像
    image_file = f"{method}_{model_name}_{file_name}_scatter.png"
    image_path = path_manager.get_file_path('images', image_file, method, model_name)
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"散点图保存至: {image_path}")


# [改进 1.1] 修改汇总表格保存函数，支持增量更新
def save_all_metrics_to_csv(metrics_list, path_manager, timestamp=None, update_existing=True):
    """
    将所有文件的评估指标汇总保存到CSV文件
    
    Parameters:
    ----------
    metrics_list : list
        包含评估指标的列表
    path_manager : PathManager
        路径管理器实例
    timestamp : str, optional
        时间戳，如果为None则生成新的
    update_existing : bool, optional
        是否更新现有文件，默认为True
        
    Returns:
    -------
    pandas.DataFrame
        包含所有指标的DataFrame
    """
    if not metrics_list:
        print("警告: 没有指标可保存")
        return None
    
    columns = ["Filename", "Method", "Model", 
               "Test_R2", "Test_MSE", "Test_MAE", 
               "Train_R2", "Train_MSE", "Train_MAE",
               "Training_Time"]
    
    # 创建需要保存的DataFrame
    new_df = pd.DataFrame(metrics_list, columns=columns)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建文件路径
    latest_file = 'Linear_evaluation_results.csv'
    latest_path = path_manager.get_file_path('summaries', latest_file, create=True)
    
    # 如果选择更新现有文件并且文件存在，则合并数据
    if update_existing and os.path.exists(latest_path):
        try:
            # 读取现有文件
            existing_df = pd.read_csv(latest_path)
            
            # 合并数据，保留最新的结果（基于Filename, Method, Model）
            combined_df = pd.concat([existing_df, new_df])
            # 删除重复项，保留最后出现的（即新添加的）
            combined_df = combined_df.drop_duplicates(
                subset=['Filename', 'Method', 'Model'], 
                keep='last'
            )
            
            # 保存合并后的数据
            combined_df.to_csv(latest_path, index=False)
            print(f"更新汇总评估指标到: {latest_path}")
            
            # 保存带时间戳的版本
            timestamp_file = f'Linear_evaluation_results_{timestamp}.csv'
            timestamp_path = path_manager.get_file_path('summaries', timestamp_file, create=True)
            combined_df.to_csv(timestamp_path, index=False)
            
            return combined_df
            
        except Exception as e:
            print(f"更新现有汇总文件时出错: {str(e)}")
            print("将创建新的汇总文件...")
    
    # 如果不更新或更新失败，直接保存新数据
    new_df.to_csv(latest_path, index=False)
    print(f"汇总评估指标保存到: {latest_path}")
    
    # 保存带时间戳的版本
    timestamp_file = f'Linear_evaluation_results_{timestamp}.csv'
    timestamp_path = path_manager.get_file_path('summaries', timestamp_file, create=True)
    new_df.to_csv(timestamp_path, index=False)
    
    return new_df


def process_folder_data(data_list, file_names, last_column_names, path_manager, method, model_name="Linear", timestamp=None):
    """
    处理文件夹中的所有数据文件
    
    Parameters:
    ----------
    data_list : list
        包含所有DataFrame的列表
    file_names : list
        文件名列表
    last_column_names : list
        目标变量名称列表
    path_manager : PathManager
        路径管理器实例
    method : str
        使用的方法名称
    model_name : str, optional
        模型名称，默认为"Linear"
    timestamp : str, optional
        时间戳，如果为None则生成新的
        
    Returns:
    -------
    list
        所有文件的评估指标
    """
    metrics_list = []
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (data, file_name_with_ext, last_column_name) in enumerate(zip(data_list, file_names, last_column_names)):
        if isinstance(data, pd.DataFrame) and not data.empty:
            try:
                # [改进 2.2] 提取不含扩展名的文件名
                file_name = get_base_filename(file_name_with_ext)
                
                print(f"\n处理文件 ({i+1}/{len(data_list)}): {file_name}")
                print("-" * 60)
                
                # 准备数据
                X, y, feature_names = prepare_data(data)
                
                # 训练和评估模型
                results = cross_validate_and_evaluate(
                    X, y, feature_names, path_manager,
                    method=method, model_name=model_name, 
                    file_name=file_name
                )
                
                # 绘制和保存散点图
                plot_and_save_scatter(
                    results, path_manager, method, 
                    model_name, file_name, last_column_name
                )
                
                # 收集评估指标
                avg_metrics = results['metrics']['average_metrics']
                metrics_list.append([
                    file_name,
                    method,
                    model_name,
                    avg_metrics['test_r2'],
                    avg_metrics['test_mse'],
                    avg_metrics['test_mae'],
                    avg_metrics['train_r2'],
                    avg_metrics['train_mse'],
                    avg_metrics['train_mae'],
                    results['metrics']['training_time']
                ])
                
                # [改进 1.2] 每处理完一个文件就更新汇总表格
                current_metrics = [[
                    file_name,
                    method,
                    model_name,
                    avg_metrics['test_r2'],
                    avg_metrics['test_mse'],
                    avg_metrics['test_mae'],
                    avg_metrics['train_r2'],
                    avg_metrics['train_mse'],
                    avg_metrics['train_mae'],
                    results['metrics']['training_time']
                ]]
                
                save_all_metrics_to_csv(current_metrics, path_manager, timestamp)
                
                print(f"文件 {file_name} 处理完成")
                print("=" * 60)
                
            except Exception as e:
                print(f"处理文件 {file_name_with_ext} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
    
    return metrics_list


def main():
    """
    主函数，处理不同类型的分子指纹数据进行建模与评估
    """
    # 定义基础路径 - 根据实际环境调整
    base_dir = os.getcwd()  # 使用当前工作目录  # 使用当前工作目录
    
    # 创建路径管理器
    path_manager = PathManager(base_dir)
    
    # 定义数据文件夹与方法映射
    data_folders = {
        # AtomPair 指纹
        'AtomPair_512': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Atompair_512'),
        'AtomPair_1024': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'AtomPair_1024'),
        'AtomPair_2048': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'AtomPair_2048'),
        
        # MACCS 指纹
        'Maccs': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Maccs'),
        
        # Morgan 指纹
        'Morgan_512': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Morgan_512'),
        'Morgan_1024': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Morgan_1024'),
        'Morgan_2048': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Morgan_2048'),
        
        # RDKit 指纹
        'RDKit_512': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'RDKit_512'),
        'RDKit_1024': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'RDKit_1024'),
        'RDKit_2048': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'RDKit_2048'),
        
        # Torsion 指纹
        'Torsion_512': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Torsion_512'),
        'Torsion_1024': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Torsion_1024'),
        'Torsion_2048': os.path.join(base_dir, 'Polymer_Describe', 'Fingers_', 'Torsion_2048'),
        
        # Mordred 描述符
        'Mordred': os.path.join(base_dir, 'Polymer_Describe', 'Mordred')
    }
    
    # 打印基础目录，帮助调试
    print(f"基础目录: {base_dir}")
    
    
    # 用于存储所有指标的列表
    all_metrics = []
    
    # 创建时间戳，所有输出文件共用
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 处理每种方法的数据
    for method, data_folder in data_folders.items():
        print(f"\n{'='*80}")
        print(f"开始处理 {method} 数据...")
        print(f"{'='*80}")
        
        try:
            if not os.path.exists(data_folder):
                print(f"警告: 数据文件夹不存在: {data_folder}")
                continue
                
            # 读取数据
            data_list, file_names, last_column_names = read_files_in_folder(data_folder)
            
            if not data_list:
                print(f"警告: {data_folder} 文件夹中没有找到CSV文件")
                continue
                
            print(f"找到 {len(data_list)} 个文件进行处理")
            
            # 处理该方法的所有文件
            method_metrics = process_folder_data(
                data_list, 
                file_names, 
                last_column_names,
                path_manager,
                method, 
                model_name="Linear",
                timestamp=timestamp
            )
            
            # 将该方法的指标添加到总列表中
            all_metrics.extend(method_metrics)
            
            print(f"完成 {method} 数据处理, 共处理 {len(method_metrics)} 个文件")
            
        except Exception as e:
            print(f"处理 {method} 数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存所有文件的评估指标
    if all_metrics:
        save_all_metrics_to_csv(all_metrics, path_manager, timestamp)
    else:
        print("警告: 没有成功处理任何数据，未保存汇总评估指标")
    
    print("\n所有处理完成！")
    
    # 返回聚合的结果，以便可能的进一步分析
    return {
        'all_metrics': all_metrics,
        'folders_processed': list(data_folders.keys()),
        'timestamp': timestamp
    }

if __name__ == "__main__":
    main()
    print("所有处理完成！")
