import time
import warnings
from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
import os
from pathlib import Path
import json
from datetime import datetime
import glob

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning, 
                       message='.*edgecolor/edgecolors.*unfilled marker.*')

# 尝试使用科学风格绘图
try:
    plt.style.use(['science', 'nature'])
except:
    logger.info("提示: 安装scienceplots可获得更好的图表风格: pip install scienceplots")


class ImprovedClusteringFramework:
    """改进的聚类算法比较框架，支持结果保存和可视化分离"""
    
    def __init__(self, random_state=42, verbose=True, output_dir="clustering_results"):
        self.random_state = random_state
        self.algorithms = []  # 存储算法列表
        self.X = None  # 原始数据
        self.X_2d = None  # 降维后数据
        self.y_true = None  # 真实标签（如果有）
        self.params = None  # 自适应参数
        self.results = {}  # 存储结果
        self.optimal_clusters = {}  # 存储最优聚类数量
        self.verbose = verbose  # 是否显示进度条
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 数据集信息
        self.dataset_name = None
        self.dataset_info = {}
        self.folder_name = None  # 文件夹名称
        self.current_file_path = None  # 当前处理的文件路径
        
        # 批量处理相关
        self.csv_files = []  # CSV文件列表
        self.batch_results = {}  # 批量处理结果
        
        # 嵌套文件夹处理相关
        self.nested_folders = {}  # 嵌套文件夹结构
        self.current_parent_folder = None  # 当前处理的父文件夹
        self.current_sub_folder = None  # 当前处理的子文件夹
        self.nested_batch_results = {}  # 嵌套批量处理结果
        
        # 降维相关配置
        self.reduction_method = 'tsne'  # 默认降维方法
        self.n_components = 2  # 默认降维维度
        self.reduction_params = {}  # 降维参数
        
        # 定义填充和未填充的标记
        self.filled_markers = ['o', 's', '^', 'd', '*', 'p', 'v', '<', '>']
        self.unfilled_markers = ['x', '+', '|', '_', '1', '2', '3', '4']
        
        # 保存路径记录
        self.saved_paths = {
            'dataset_info': None,
            'original_data': None,
            'target_column': None,
            'reduced_data': None,
            'clustering_labels': None,
            'clustering_summary': None,
            'visualization': None
        }
        
    def _log(self, message, level='info'):
        """统一的日志输出"""
        if self.verbose:
            if level == 'info':
                logger.info(message)
            elif level == 'warning':
                logger.warning(message)
            elif level == 'error':
                logger.error(message)
                
    def set_reduction_config(self, method='tsne', n_components=2, **kwargs):
        """
        设置降维配置
        
        Parameters:
        -----------
        method : str, default='tsne'
            降维方法，支持 'pca', 'tsne'
        n_components : int, default=2
            降维后的维度数量
        **kwargs : dict
            降维方法的额外参数
            - PCA: 无额外参数
            - t-SNE: perplexity, learning_rate, n_iter等
        """
        supported_methods = ['pca', 'tsne']
        if method.lower() not in supported_methods:
            self._log(f"不支持的降维方法: {method}，支持的方法: {supported_methods}", 'error')
            return self
        
        self.reduction_method = method.lower()
        self.n_components = max(1, int(n_components))
        self.reduction_params = kwargs
        
        self._log(f"降维配置已设置: 方法={self.reduction_method}, 维度={self.n_components}")
        if kwargs:
            self._log(f"额外参数: {kwargs}")
        
        # 如果已有降维数据，清除以便重新计算
        self.X_2d = None
        
        return self
    
    def get_available_reduction_methods(self):
        """获取可用的降维方法列表"""
        methods = {
            'pca': {
                'name': 'Principal Component Analysis (PCA)',
                'description': '主成分分析，线性降维方法，速度快',
                'parameters': ['n_components'],
                'suitable_for': '高维数据的快速降维'
            },
            'tsne': {
                'name': 't-Distributed Stochastic Neighbor Embedding (t-SNE)',
                'description': '非线性降维方法，保持局部结构，适合可视化',
                'parameters': ['n_components', 'perplexity', 'learning_rate', 'n_iter'],
                'suitable_for': '数据可视化，发现聚类结构'
            }
        }
        
        print("可用的降维方法:")
        print("=" * 60)
        for method, info in methods.items():
            print(f"方法: {method.upper()}")
            print(f"名称: {info['name']}")
            print(f"描述: {info['description']}")
            print(f"参数: {', '.join(info['parameters'])}")
            print(f"适用场景: {info['suitable_for']}")
            print("-" * 60)
        
        return methods
                
    def load_folder(self, folder_path, file_pattern="*.csv"):
        """加载文件夹中的所有CSV文件"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            self._log(f"错误: 文件夹不存在 {folder_path}", 'error')
            return self
        
        if not folder_path.is_dir():
            self._log(f"错误: 路径不是文件夹 {folder_path}", 'error')
            return self
        
        # 获取文件夹名称
        self.folder_name = folder_path.name
        
        # 查找CSV文件
        csv_pattern = folder_path / file_pattern
        self.csv_files = list(glob.glob(str(csv_pattern)))
        
        if not self.csv_files:
            self._log(f"错误: 在文件夹 {folder_path} 中没有找到CSV文件", 'error')
            return self
        
        self._log(f"找到 {len(self.csv_files)} 个CSV文件在文件夹 '{self.folder_name}' 中:")
        for i, file_path in enumerate(self.csv_files, 1):
            file_name = Path(file_path).name
            self._log(f"  {i}. {file_name}")
        
        return self
    
    def _generate_dataset_name(self, file_path):
        """生成数据集名称：文件夹名_文件名"""
        file_name = Path(file_path).stem  # 不含扩展名的文件名
        if self.folder_name:
            return f"{self.folder_name}_{file_name}"
        else:
            return file_name
    
    def _load_single_csv(self, file_path, label_column=None):
        """加载单个CSV文件"""
        try:
            # 生成数据集名称
            self.dataset_name = self._generate_dataset_name(file_path)
            self.current_file_path = file_path
            
            df = pd.read_csv(file_path)
            
            # 检查最后一列的名称，决定是否排除
            last_column_name = df.columns[-1]
            if last_column_name == 'property_log':
                # 如果最后一列名为property_log，排除最后一列
                df_features = df.iloc[:, :-1]
                self.y_true = None  # 没有目标列
                excluded_columns = 1
                self._log(f"检测到最后一列为 'property_log'，排除最后一列")
            else:
                # 否则保留最后一列作为目标列，用于可视化着色
                df_features = df.iloc[:, :-1]
                self.y_true = df.iloc[:, -1].values  # 最后一列作为目标列
                excluded_columns = 0  # 实际上没有排除，只是分离了特征和目标
                self._log(f"最后一列为 '{last_column_name}'，保留作为目标列用于可视化着色")
            
            if label_column is not None and label_column >= 0:
                # 如果指定了标签列，从特征数据中排除该列
                if label_column >= df_features.shape[1]:
                    self._log(f"警告: 标签列索引 {label_column} 超出范围，忽略标签列", 'warning')
                    self.X = df_features.values
                else:
                    self.X = df_features.iloc[:, [i for i in range(df_features.shape[1]) if i != label_column]].values
                    # 如果指定了标签列，则使用指定的标签列而不是最后一列
                    if last_column_name != 'property_log':
                        # 如果最后一列不是property_log，需要从原始df中获取标签列
                        self.y_true = df.iloc[:, label_column].values
            else:
                self.X = df_features.values
                # y_true已经在上面设置了
                
            # 保存数据集信息
            self.dataset_info = {
                'file_path': str(file_path),
                'folder_name': self.folder_name,
                'dataset_name': self.dataset_name,
                'original_file_name': Path(file_path).name,
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'has_labels': self.y_true is not None,
                'has_target_column': self.y_true is not None and last_column_name != 'property_log',
                'target_column_name': last_column_name if last_column_name != 'property_log' else None,
                'load_time': datetime.now().isoformat(),
                'reduction_method': self.reduction_method,
                'reduction_components': self.n_components,
                'reduction_params': self.reduction_params,
                'last_column_name': last_column_name,
                'excluded_columns': excluded_columns
            }
            
            self._log(f"成功加载数据集 '{self.dataset_name}': {self.X.shape[0]}行, {self.X.shape[1]}列")
            if self.y_true is not None and last_column_name != 'property_log':
                self._log(f"目标列 '{last_column_name}' 将用于可视化着色")
            return True
        except Exception as e:
            self._log(f"加载CSV文件 {file_path} 时出错: {e}", 'error')
            return False
    
    def process_folder(self, folder_path, label_column=None, file_pattern="*.csv", 
                      auto_select_clusters=True, cluster_method='bic',
                      reduction_method='tsne', n_components=2, **reduction_kwargs):
        """
        批量处理文件夹中的所有CSV文件
        
        Parameters:
        -----------
        folder_path : str
            文件夹路径
        label_column : int, optional
            标签列索引
        file_pattern : str, default="*.csv"
            文件匹配模式
        auto_select_clusters : bool, default=True
            是否自动选择最优聚类数
        cluster_method : str, default='bic'
            聚类数选择方法
        reduction_method : str, default='tsne'
            降维方法
        n_components : int, default=2
            降维维度
        **reduction_kwargs : dict
            降维方法的额外参数
        """
        # 设置降维配置
        self.set_reduction_config(reduction_method, n_components, **reduction_kwargs)
        
        # 加载文件夹
        self.load_folder(folder_path, file_pattern)
        
        if not self.csv_files:
            return self
        
        self._log(f"开始批量处理 {len(self.csv_files)} 个CSV文件...")
        self._log(f"降维配置: {self.reduction_method.upper()}, 维度: {self.n_components}")
        
        # 处理每个CSV文件
        for i, csv_file in enumerate(self.csv_files, 1):
            self._log(f"\n{'='*60}")
            self._log(f"处理文件 {i}/{len(self.csv_files)}: {Path(csv_file).name}")
            self._log(f"{'='*60}")
            
            try:
                # 重置状态
                self._reset_for_new_file()
                
                # 加载单个文件
                if not self._load_single_csv(csv_file, label_column):
                    continue
                
                # 预处理数据
                self.preprocess()
                
                # 添加算法
                if auto_select_clusters:
                    self.add_all_sklearn_algorithms(auto_select=True, method=cluster_method)
                else:
                    self.add_all_sklearn_algorithms(auto_select=False, n_clusters=3)
                
                # 运行算法
                self.run()
                
                # 保存结果
                self.save_results()
                
                # 生成可视化
                self.visualize_from_saved_data()
                
                # 比较性能
                self.compare_performance()
                
                # 打印保存路径信息
                self.print_saved_paths()
                
                # 保存批量处理结果
                self.batch_results[self.dataset_name] = {
                    'file_path': csv_file,
                    'success': True,
                    'n_algorithms': len([r for r in self.results.values() if r.get('success', False)]),
                    'processing_time': datetime.now().isoformat(),
                    'saved_paths': self.saved_paths.copy()
                }
                
                self._log(f"文件 '{Path(csv_file).name}' 处理完成")
                
            except Exception as e:
                self._log(f"处理文件 {csv_file} 时出错: {e}", 'error')
                self.batch_results[self._generate_dataset_name(csv_file)] = {
                    'file_path': csv_file,
                    'success': False,
                    'error': str(e),
                    'processing_time': datetime.now().isoformat()
                }
        
        # 保存批量处理摘要
        self._save_batch_summary()
        
        self._log(f"\n{'='*60}")
        self._log("批量处理完成!")
        self._log(f"成功处理: {sum(1 for r in self.batch_results.values() if r['success'])}/{len(self.csv_files)} 个文件")
        self._log(f"{'='*60}")
        
        return self
    
    def _reset_for_new_file(self):
        """为处理新文件重置状态"""
        self.algorithms = []
        self.X = None
        self.X_2d = None
        self.y_true = None
        self.params = None
        self.results = {}
        self.optimal_clusters = {}
        self.dataset_name = None
        self.dataset_info = {}
        self.saved_paths = {
            'dataset_info': None,
            'original_data': None,
            'target_column': None,
            'reduced_data': None,
            'clustering_labels': None,
            'clustering_summary': None,
            'visualization': None
        }
    
    def _save_batch_summary(self):
        """保存批量处理摘要"""
        if not self.batch_results:
            return
        
        # 创建批量处理摘要目录
        batch_summary_dir = self.output_dir / f"batch_summary_{self.folder_name}"
        batch_summary_dir.mkdir(exist_ok=True)
        
        # 保存批量处理结果
        summary_data = []
        for dataset_name, result in self.batch_results.items():
            summary_data.append({
                '数据集名称': dataset_name,
                '文件路径': result['file_path'],
                '处理状态': '成功' if result['success'] else '失败',
                '算法数量': result.get('n_algorithms', 0) if result['success'] else 0,
                '错误信息': result.get('error', '') if not result['success'] else '',
                '处理时间': result['processing_time']
            })
        
        # 保存为CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = batch_summary_dir / f"batch_processing_summary_{self.folder_name}.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存为JSON
        json_file = batch_summary_dir / f"batch_processing_summary_{self.folder_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.batch_results, f, ensure_ascii=False, indent=2)
        
        self._log(f"批量处理摘要已保存到: {batch_summary_dir}")
    
    def load_data(self, X, y=None):
        """加载数据"""
        self.X = X
        self.y_true = y
        return self
        
    def load_from_csv(self, file_path, label_column=None):
        """从CSV文件加载数据并设置数据集名称"""
        if self._load_single_csv(file_path, label_column):
            return self
        else:
            return None
    
    def preprocess(self, scale=True):
        """预处理数据"""
        if scale and self.X is not None:
            self.X = StandardScaler().fit_transform(self.X)
        return self
    
    def reduce_dimensions(self, method=None, n_components=None, force_recompute=False, **kwargs):
        """
        降维用于可视化
        
        Parameters:
        -----------
        method : str, optional
            降维方法，如果不指定则使用配置的方法
        n_components : int, optional
            降维维度，如果不指定则使用配置的维度
        force_recompute : bool, default=False
            是否强制重新计算
        **kwargs : dict
            降维方法的额外参数
        """
        if self.X is None:
            logger.error("错误: 请先加载数据")
            return self
        
        # 使用传入的参数或配置的参数
        method = method or self.reduction_method
        n_components = n_components or self.n_components
        params = {**self.reduction_params, **kwargs}
        
        # 如果已有降维数据且不强制重新计算，直接返回
        if self.X_2d is not None and not force_recompute:
            self._log("使用已有的降维数据")
            return self
        
        if self.X.shape[1] <= n_components:
            self.X_2d = self.X
            self._log("数据维度已经足够低，无需降维")
            return self
            
        self._log(f"开始降维: {method.upper()}, 目标维度: {n_components}")
        if params:
            self._log(f"降维参数: {params}")
        
        try:
            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=self.random_state, **params)
            elif method == 'tsne':
                # 设置t-SNE的默认参数
                tsne_params = {
                    'perplexity': min(30, max(5, self.X.shape[0] // 20)),
                    'learning_rate': 200,
                    'n_iter': 1000,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                tsne_params.update(params)
                reducer = TSNE(n_components=n_components, **tsne_params)
            else:
                logger.error(f"未知的降维方法: {method}")
                return self
            
            self.X_2d = reducer.fit_transform(self.X)
            self._log(f"降维完成: {method.upper()}, 形状: {self.X_2d.shape}")
            
            # 更新数据集信息
            if hasattr(self, 'dataset_info') and self.dataset_info:
                self.dataset_info.update({
                    'reduction_method': method,
                    'reduction_components': n_components,
                    'reduction_params': params,
                    'reduced_shape': self.X_2d.shape
                })
            
        except Exception as e:
            logger.error(f"降维时出错: {e}")
            self.X_2d = self.X[:, :2] if self.X.shape[1] > 1 else np.column_stack((self.X, np.zeros_like(self.X)))
            self._log("使用前两个特征作为降维数据")
        
        return self
    
    def compute_adaptive_params(self):
        """计算自适应参数"""
        if self.X is None:
            logger.error("错误: 请先加载数据")
            return self
            
        n_samples, n_features = self.X.shape
        params = {}
        
        # 根据样本数量自适应计算参数
        params["quantile"] = max(0.1, min(0.3, 30 / n_samples))
        params["n_neighbors"] = min(30, max(5, int(np.sqrt(n_samples))))
        params["min_samples"] = max(5, int(np.log(n_samples)))
        params["hdbscan_min_samples"] = max(3, int(np.log(n_samples) / 2))
        params["hdbscan_min_cluster_size"] = max(5, int(np.log(n_samples) * 2))
        params["min_cluster_size"] = max(0.05, min(0.1, 10 / n_samples))
        
        # 计算最佳eps参数
        try:
            nn = NearestNeighbors(n_neighbors=params["n_neighbors"])
            nn.fit(self.X)
            distances, _ = nn.kneighbors(self.X)
            distances = np.sort(distances[:, -1])
            
            try:
                # 尝试使用KneeLocator找到最佳拐点
                knee_finder = KneeLocator(
                    range(len(distances)), distances, 
                    curve='convex', direction='increasing'
                )
                if knee_finder.knee is not None:
                    params["eps"] = distances[knee_finder.knee]
                else:
                    # 如果找不到明显拐点，使用百分位数
                    params["eps"] = np.percentile(distances, 90)
            except:
                # 如果KneeLocator失败，使用默认方法
                params["eps"] = np.percentile(distances, 90)
        except:
            params["eps"] = 0.5  # 默认值
            
        # 基于特征数量调整偏好参数
        params["preference"] = -100 * n_features
        params["damping"] = min(0.99, max(0.5, 0.8 + n_features / 100))
        
        # 固定参数
        params["xi"] = 0.05
        params["allow_single_cluster"] = True
        params["random_state"] = self.random_state
        
        self.params = params
        return self
    
    def add_algorithm(self, name, algorithm):
        """添加聚类算法"""
        self.algorithms.append((name, algorithm))
        return self
    
    def add_sklearn_kmeans(self, n_clusters=3, name="KMeans"):
        """添加KMeans算法"""
        from sklearn.cluster import KMeans
        algorithm = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_minibatch_kmeans(self, n_clusters=3, name="MiniBatch\nKMeans"):
        """添加MiniBatchKMeans算法"""
        from sklearn.cluster import MiniBatchKMeans
        
        if self.X is not None:
            batch_size = max(100, min(1000, self.X.shape[0] // 10))
        else:
            batch_size = 100
            
        algorithm = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            batch_size=batch_size
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_affinity_propagation(self, name="Affinity\nPropagation"):
        """添加AffinityPropagation算法"""
        from sklearn.cluster import AffinityPropagation
        
        if self.params is None:
            self.compute_adaptive_params()
            
        algorithm = AffinityPropagation(
            damping=self.params["damping"],
            preference=self.params["preference"],
            random_state=self.random_state,
            max_iter=300
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_meanshift(self, name="MeanShift"):
        """添加MeanShift算法"""
        from sklearn.cluster import MeanShift, estimate_bandwidth
        
        if self.params is None:
            self.compute_adaptive_params()
            
        if self.X is not None:
            bandwidth = estimate_bandwidth(self.X, quantile=self.params["quantile"])
        else:
            bandwidth = 0.5
            
        algorithm = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=True,
            cluster_all=True
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_spectral_clustering(self, n_clusters=3, name="Spectral\nClustering"):
        """添加SpectralClustering算法"""
        from sklearn.cluster import SpectralClustering
        
        algorithm = SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver="arpack",
            affinity="nearest_neighbors",
            random_state=self.random_state
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_agglomerative(self, n_clusters=3, linkage="ward", name=None):
        """添加AgglomerativeClustering算法"""
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph
        
        if self.params is None:
            self.compute_adaptive_params()
        
        if name is None:
            name = f"Agglomerative\n{linkage.capitalize()}"
            
        if self.X is not None:
            connectivity = kneighbors_graph(
                self.X, 
                n_neighbors=self.params["n_neighbors"], 
                include_self=False
            )
            connectivity = 0.5 * (connectivity + connectivity.T)
        else:
            connectivity = None
            
        if linkage == "ward":
            algorithm = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage="ward",
                connectivity=connectivity
            )
        else:
            algorithm = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
                metric="cityblock" if linkage == "average" else "euclidean",
                connectivity=connectivity
            )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_dbscan(self, name="DBSCAN"):
        """添加DBSCAN算法"""
        from sklearn.cluster import DBSCAN
        
        if self.params is None:
            self.compute_adaptive_params()
            
        algorithm = DBSCAN(
            eps=self.params["eps"],
            min_samples=self.params["min_samples"]
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_hdbscan(self, name="HDBSCAN"):
        """添加HDBSCAN算法"""
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            import hdbscan
            HDBSCAN = hdbscan.HDBSCAN
        
        if self.params is None:
            self.compute_adaptive_params()
            
        algorithm = HDBSCAN(
            min_samples=self.params["hdbscan_min_samples"],
            min_cluster_size=self.params["hdbscan_min_cluster_size"],
            allow_single_cluster=self.params["allow_single_cluster"]
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_optics(self, name="OPTICS"):
        """添加OPTICS算法"""
        from sklearn.cluster import OPTICS
        
        if self.params is None:
            self.compute_adaptive_params()
            
        algorithm = OPTICS(
            min_samples=self.params["min_samples"],
            xi=self.params["xi"],
            min_cluster_size=self.params["min_cluster_size"]
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_birch(self, n_clusters=3, name="BIRCH"):
        """添加BIRCH算法"""
        from sklearn.cluster import Birch
        
        if self.X is not None:
            branching_factor = min(50, max(10, self.X.shape[0] // 100))
        else:
            branching_factor = 50
            
        algorithm = Birch(
            n_clusters=n_clusters,
            threshold=0.01,
            branching_factor=branching_factor
        )
        return self.add_algorithm(name, algorithm)
    
    def add_sklearn_gaussian_mixture(self, n_components=3, name="Gaussian\nMixture"):
        """添加GaussianMixture算法"""
        from sklearn.mixture import GaussianMixture
        
        algorithm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=200
        )
        return self.add_algorithm(name, algorithm)
    
    def add_all_sklearn_algorithms(self, n_clusters=None, auto_select=True, method='elbow'):
        """添加所有scikit-learn聚类算法"""
        if self.X is None:
            raise ValueError("请先加载数据")
            
        # 为不同算法选择最优聚类数量的方法
        def select_optimal_clusters(algorithm_name):
            try:
                if algorithm_name in ['KMeans', 'MiniBatchKMeans']:
                    return self.estimate_optimal_clusters(algorithm_name, method='elbow')
                elif algorithm_name in ['SpectralClustering', 'AgglomerativeClustering']:
                    return self.estimate_optimal_clusters(algorithm_name, method='silhouette')
                elif algorithm_name in ['GaussianMixture']:
                    return self.estimate_optimal_clusters(algorithm_name, method='bic')
                elif algorithm_name in ['BIRCH']:
                    return self.estimate_optimal_clusters(algorithm_name, method='silhouette')
                else:
                    return self.estimate_optimal_clusters(algorithm_name, method=method)
            except Exception as e:
                self._log(f"{algorithm_name}自动选择聚类数量失败: {e}", 'error')
                return 3
        
        # 如果需要自动选择聚类数量
        if auto_select and n_clusters is None:
            optimal_clusters = {}
            for algo_name in ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 
                            'AgglomerativeClustering', 'GaussianMixture', 'BIRCH']:
                optimal_clusters[algo_name] = select_optimal_clusters(algo_name)
                self._log(f"{algo_name}自动选择的最优聚类数量: {optimal_clusters[algo_name]}")
        else:
            n_clusters = self.validate_n_clusters(n_clusters)
            optimal_clusters = {algo_name: n_clusters for algo_name in 
                              ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 
                               'AgglomerativeClustering', 'GaussianMixture', 'BIRCH']}
        
        # 添加需要指定聚类数量的算法
        self.add_sklearn_kmeans(optimal_clusters['KMeans'])
        self.add_sklearn_minibatch_kmeans(optimal_clusters['MiniBatchKMeans'])
        self.add_sklearn_spectral_clustering(optimal_clusters['SpectralClustering'])
        self.add_sklearn_agglomerative(optimal_clusters['AgglomerativeClustering'], 
                                     linkage="ward", name="Ward")
        self.add_sklearn_agglomerative(optimal_clusters['AgglomerativeClustering'], 
                                     linkage="average")
        self.add_sklearn_birch(optimal_clusters['BIRCH'])
        self.add_sklearn_gaussian_mixture(optimal_clusters['GaussianMixture'])
        
        # 添加不需要指定聚类数量的算法
        self.add_sklearn_affinity_propagation()
        self.add_sklearn_meanshift()
        self.add_sklearn_dbscan()
        
        # 尝试添加HDBSCAN（可能需要额外安装）
        try:
            self.add_sklearn_hdbscan()
        except:
            self._log("HDBSCAN未安装，跳过")
            
        self.add_sklearn_optics()
        
        return self
    
    def run(self):
        """运行所有算法并收集结果"""
        if self.X is None:
            self._log("错误: 请先加载数据", 'error')
            return self
            
        if len(self.algorithms) == 0:
            self._log("警告: 没有添加任何算法", 'warning')
            return self
            
        # 运行所有算法
        self.results = {}
        self._log("开始运行聚类算法...")
        
        # 使用tqdm创建进度条
        for name, algorithm in tqdm(self.algorithms, desc="运行算法", disable=not self.verbose):
            t0 = time.time()
            
            # 尝试拟合算法
            try:
                algorithm.fit(self.X)
                
                # 获取聚类标签
                if hasattr(algorithm, "labels_"):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(self.X)
                
                t1 = time.time()
                
                # 计算聚类数量
                unique_labels = np.unique(y_pred)
                n_clusters = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1
                
                # 计算评估指标
                silhouette_avg = None
                calinski_harabasz = None
                
                if n_clusters > 1 and len(unique_labels) > 1:
                    try:
                        silhouette_avg = silhouette_score(self.X, y_pred)
                    except:
                        pass
                    try:
                        calinski_harabasz = calinski_harabasz_score(self.X, y_pred)
                    except:
                        pass
                
                self.results[name] = {
                    "labels": y_pred,
                    "time": t1 - t0,
                    "success": True,
                    "n_clusters": n_clusters,
                    "silhouette_score": silhouette_avg,
                    "calinski_harabasz_score": calinski_harabasz
                }
                
                self._log(f"算法 {name} 完成: 找到 {n_clusters} 个聚类，用时 {t1-t0:.2f} 秒")
                
            except Exception as e:
                self._log(f"算法 {name} 出错: {e}", 'error')
                self.results[name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return self
    
    def save_results(self):
        """保存聚类结果到文件，支持嵌套文件夹结构"""
        if not self.results or self.dataset_name is None:
            self._log("错误: 没有结果可保存或未设置数据集名称", 'error')
            return self
        
        # 根据是否为嵌套结构决定保存路径
        if hasattr(self, 'current_sub_folder') and self.current_sub_folder:
            # 嵌套结构：按子文件夹组织
            sub_folder_dir = self.output_dir / self.current_sub_folder
            sub_folder_dir.mkdir(exist_ok=True)
            dataset_dir = sub_folder_dir / self.dataset_name
        else:
            # 普通结构：直接以数据集名称创建目录
            dataset_dir = self.output_dir / self.dataset_name
        
        dataset_dir.mkdir(exist_ok=True)
        
        # 保存数据集信息
        info_file = dataset_dir / f"{self.dataset_name}_dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(self.dataset_info, f, ensure_ascii=False, indent=2)
        self.saved_paths['dataset_info'] = str(info_file)
        
        # 保存原始数据
        data_file = dataset_dir / f"{self.dataset_name}_original_data.csv"
        pd.DataFrame(self.X).to_csv(data_file, index=False)
        self.saved_paths['original_data'] = str(data_file)
        
        # 保存目标列数据（如果存在）
        if self.y_true is not None and self.dataset_info.get('has_target_column', False):
            target_file = dataset_dir / f"{self.dataset_name}_target_column.csv"
            target_column_name = self.dataset_info.get('target_column_name', 'target')
            target_df = pd.DataFrame({target_column_name: self.y_true})
            target_df.to_csv(target_file, index=False)
            self.saved_paths['target_column'] = str(target_file)
            self._log(f"目标列数据已保存: {target_file}")
        
        # 保存降维数据（如果存在）
        if self.X_2d is not None:
            reduced_file = dataset_dir / f"{self.dataset_name}_reduced_data_{self.reduction_method}_{self.n_components}d.csv"
            # 创建降维数据的DataFrame，包含列名
            reduced_columns = [f'{self.reduction_method.upper()}_dim_{i+1}' for i in range(self.X_2d.shape[1])]
            reduced_df = pd.DataFrame(self.X_2d, columns=reduced_columns)
            reduced_df.to_csv(reduced_file, index=False)
            self.saved_paths['reduced_data'] = str(reduced_file)
            
            self._log(f"降维数据已保存: {reduced_file}")
            self._log(f"降维方法: {self.reduction_method.upper()}, 维度: {self.X_2d.shape[1]}")
        
        # 保存聚类结果
        results_data = []
        labels_data = {}
        
        for name, result in self.results.items():
            if result.get("success", False):
                # 收集结果摘要
                results_data.append({
                    '算法名称': name,
                    '聚类数量': result["n_clusters"],
                    '运行时间(秒)': round(result["time"], 4),
                    '轮廓系数': round(result["silhouette_score"], 4) if result["silhouette_score"] is not None else None,
                    'CH指数': round(result["calinski_harabasz_score"], 4) if result["calinski_harabasz_score"] is not None else None,
                    '状态': '成功'
                })
                
                # 收集标签数据
                labels_data[f"{name}_labels"] = result["labels"]
            else:
                results_data.append({
                    '算法名称': name,
                    '聚类数量': None,
                    '运行时间(秒)': None,
                    '轮廓系数': None,
                    'CH指数': None,
                    '状态': f'失败: {result.get("error", "未知错误")}'
                })
        
        # 保存结果摘要
        results_df = pd.DataFrame(results_data)
        summary_file = dataset_dir / f"{self.dataset_name}_clustering_summary.csv"
        results_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        self.saved_paths['clustering_summary'] = str(summary_file)
        
        # 保存所有标签
        if labels_data:
            labels_df = pd.DataFrame(labels_data)
            labels_file = dataset_dir / f"{self.dataset_name}_clustering_labels.csv"
            labels_df.to_csv(labels_file, index=False)
            self.saved_paths['clustering_labels'] = str(labels_file)
        
        self._log(f"聚类结果已保存到目录: {dataset_dir}")
        return self
    
    def print_saved_paths(self):
        """打印所有保存的文件路径和数据信息"""
        if not any(self.saved_paths.values()):
            self._log("没有保存的文件", 'warning')
            return
        
        print(f"\n{'='*80}")
        print(f"数据集: {self.dataset_name}")
        print(f"{'='*80}")
        
        # 打印数据信息
        if self.dataset_info:
            print(f"📊 数据信息:")
            print(f"   原始数据形状: {self.dataset_info.get('n_samples', 'N/A')} 行 × {self.dataset_info.get('n_features', 'N/A')} 列")
            if self.X_2d is not None:
                print(f"   降维数据形状: {self.X_2d.shape[0]} 行 × {self.X_2d.shape[1]} 列")
                print(f"   降维方法: {self.reduction_method.upper()}")
            print(f"   聚类算法数量: {len([r for r in self.results.values() if r.get('success', False)])}")
            print()
        
        # 打印保存路径
        print(f"💾 保存的文件路径:")
        
        if self.saved_paths['dataset_info']:
            print(f"   📋 数据集信息: {self.saved_paths['dataset_info']}")
        
        if self.saved_paths['original_data']:
            print(f"   📊 原始数据: {self.saved_paths['original_data']}")
        
        if self.saved_paths['target_column']:
            print(f"   🎯 目标列数据: {self.saved_paths['target_column']}")
        
        if self.saved_paths['reduced_data']:
            print(f"   🔄 降维数据: {self.saved_paths['reduced_data']}")
        
        if self.saved_paths['clustering_labels']:
            print(f"   🏷️  聚类标签: {self.saved_paths['clustering_labels']}")
        
        if self.saved_paths['clustering_summary']:
            print(f"   📈 聚类摘要: {self.saved_paths['clustering_summary']}")
        
        if self.saved_paths['visualization']:
            print(f"   🖼️  可视化图片: {self.saved_paths['visualization']}")
        
        print(f"{'='*80}")
    
    def get_saved_paths(self):
        """获取所有保存的文件路径"""
        return self.saved_paths.copy()
    
    def load_saved_results(self, dataset_name, sub_folder=None):
        """从保存的文件中加载结果，支持嵌套文件夹结构"""
        if sub_folder:
            # 嵌套结构路径
            dataset_dir = self.output_dir / sub_folder / dataset_name
        else:
            # 尝试在根目录查找
            dataset_dir = self.output_dir / dataset_name
            
            # 如果根目录没找到，尝试在子文件夹中查找
            if not dataset_dir.exists():
                for item in self.output_dir.iterdir():
                    if item.is_dir():
                        potential_path = item / dataset_name
                        if potential_path.exists():
                            dataset_dir = potential_path
                            sub_folder = item.name
                            break
        
        if not dataset_dir.exists():
            self._log(f"错误: 找不到数据集目录 {dataset_dir}", 'error')
            return self
        
        # 加载数据集信息
        info_file = dataset_dir / f"{dataset_name}_dataset_info.json"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
            self.dataset_name = dataset_name
            # 设置子文件夹信息
            if sub_folder:
                self.current_sub_folder = sub_folder
        
        # 加载原始数据
        data_file = dataset_dir / f"{dataset_name}_original_data.csv"
        if data_file.exists():
            self.X = pd.read_csv(data_file).values
        
        # 加载目标列数据（如果存在）
        target_file = dataset_dir / f"{dataset_name}_target_column.csv"
        if target_file.exists():
            target_df = pd.read_csv(target_file)
            self.y_true = target_df.iloc[:, 0].values  # 假设目标列是第一列
            self._log(f"已加载目标列数据: {target_df.columns[0]}")
        else:
            # 检查数据集信息中是否有目标列信息
            if self.dataset_info.get('has_target_column', False):
                self._log("警告: 数据集信息显示有目标列，但未找到目标列文件", 'warning')
        
        # 加载降维数据（如果存在）
        if self.dataset_info:
            reduction_method = self.dataset_info.get('reduction_method', 'tsne')
            n_components = self.dataset_info.get('reduction_components', 2)
            reduced_file = dataset_dir / f"{dataset_name}_reduced_data_{reduction_method}_{n_components}d.csv"
            if reduced_file.exists():
                self.X_2d = pd.read_csv(reduced_file).values
                self._log(f"已加载降维数据: {reduced_file}")
                # 更新降维配置
                self.reduction_method = reduction_method
                self.n_components = n_components
                self.reduction_params = self.dataset_info.get('reduction_params', {})
        
        # 加载聚类标签
        labels_file = dataset_dir / f"{dataset_name}_clustering_labels.csv"
        if labels_file.exists():
            labels_df = pd.read_csv(labels_file)
            
            # 重构results字典
            self.results = {}
            for col in labels_df.columns:
                if col.endswith('_labels'):
                    algo_name = col.replace('_labels', '')
                    labels = labels_df[col].values
                    
                    # 计算基本统计信息
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1
                    
                    self.results[algo_name] = {
                        "labels": labels,
                        "success": True,
                        "n_clusters": n_clusters
                    }
        
        # 加载结果摘要
        summary_file = dataset_dir / f"{dataset_name}_clustering_summary.csv"
        if summary_file.exists():
            self.summary_df = pd.read_csv(summary_file)
        
        self._log(f"已加载数据集 '{dataset_name}' 的保存结果")
        if sub_folder:
            self._log(f"来自子文件夹: {sub_folder}")
        return self
    
    def visualize_from_saved_data(self, dataset_name=None, figsize=(12, 12), dpi=300, save_path=None):
        """从保存的数据生成可视化，确保XY轴比例一致为正方形，动态调整高度避免标题重合"""
        if dataset_name:
            self.load_saved_results(dataset_name)
        
        if not self.results or self.X is None:
            self._log("错误: 没有可用的结果数据", 'error')
            return self
        
        # 确保有降维数据用于可视化
        if self.X_2d is None:
            self._log("正在降维...")
            self.reduce_dimensions()
        
        self._log("正在生成可视化结果...")
        
        # 计算行列数以适应算法数量
        n_algorithms = sum(1 for r in self.results.values() if r.get("success", False))
        if n_algorithms <= 3:
            n_rows, n_cols = 1, n_algorithms
        else:
            n_cols = min(3, n_algorithms)
            n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        # 动态调整图形高度，确保标题不重合
        # 基础高度：每行子图需要的高度
        base_height_per_row = 4.0  # 每行子图的基础高度
        # 额外空间：为时间标注和标题预留空间
        extra_space_per_row = 1.0  # 每行额外需要的空间
        # 顶部空间：为整体标题预留空间
        top_space = 1.5
        
        # 计算总高度
        total_height = n_rows * (base_height_per_row + extra_space_per_row) + top_space
        
        # 宽度保持原有逻辑，但确保合理比例
        width = figsize[0] if isinstance(figsize, tuple) else figsize
        
        # 设置动态调整的图形大小
        dynamic_figsize = (width, max(total_height, figsize[1] if isinstance(figsize, tuple) else figsize))
        
        plt.figure(figsize=dynamic_figsize)
        
        # 动态调整布局参数
        # 计算顶部边距，为整体标题预留足够空间
        top_margin = max(0.92, 1 - (top_space / dynamic_figsize[1]))
        # 计算底部边距
        bottom_margin = 0.05
        # 计算子图间距，确保时间标注有足够空间
        hspace = max(0.4, 0.3 + (extra_space_per_row / base_height_per_row))
        
        plt.subplots_adjust(left=0.05, right=0.95, bottom=bottom_margin, top=top_margin, 
                        wspace=0.2, hspace=hspace)
        
        # 用于循环的颜色和标记
        colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        markers = cycle(self.filled_markers + self.unfilled_markers)
        
        # 绘制每个算法的结果
        plot_num = 1
        for name, result in tqdm(self.results.items(), desc="生成图表", disable=not self.verbose):
            if not result.get("success", False):
                continue
                
            # 创建子图
            ax = plt.subplot(n_rows, n_cols, plot_num)
            
            y_pred = result["labels"]
            n_clusters = result.get("n_clusters", 0)
            
            # 检查是否有目标列用于着色
            has_target_column = (self.y_true is not None and 
                               self.dataset_info.get('has_target_column', False))
            
            if has_target_column:
                # 使用目标列数值进行着色
                target_column_name = self.dataset_info.get('target_column_name', 'Target')
                
                # 创建颜色映射
                scatter = plt.scatter(
                    self.X_2d[:, 0], self.X_2d[:, 1],
                    c=self.y_true, 
                    s=8, 
                    alpha=0.6,
                    cmap='viridis',  # 使用viridis颜色映射
                    edgecolors='k', 
                    linewidths=0.3
                )
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label(target_column_name, rotation=270, labelpad=15, fontsize=9)
                cbar.ax.tick_params(labelsize=8)
                
            else:
                # 使用聚类标签进行着色（原有逻辑）
                # 获取唯一的聚类标签
                unique_labels = np.unique(y_pred)
                
                # 为每个聚类绘制点
                color_dict = {}
                
                # 先处理噪声点（如果有）
                if -1 in unique_labels:
                    mask = (y_pred == -1)
                    plt.scatter(
                        self.X_2d[mask, 0], self.X_2d[mask, 1],
                        s=6, color='k', marker='x',
                        alpha=0.6, linewidths=0.5,
                        label='Noise'
                    )
                
                # 处理非噪声点
                for label in unique_labels:
                    if label == -1:
                        continue
                        
                    if label not in color_dict:
                        color_dict[label] = (next(colors), next(markers))
                        
                    color, marker = color_dict[label]
                    mask = (y_pred == label)
                    
                    if marker in self.filled_markers:
                        marker_params = {'edgecolors': 'k', 'linewidths': 0.5}
                    else:
                        marker_params = {'linewidths': 0.5}
                    
                    plt.scatter(
                        self.X_2d[mask, 0], self.X_2d[mask, 1],
                        s=8, color=color, marker=marker,
                        alpha=0.6, **marker_params,
                        label=f'Cluster {label}'
                    )
                
                # 优化图例显示（仅在使用聚类标签着色时显示）
                if len(unique_labels) > 20:
                    plt.legend([],[], frameon=False)
                    plt.text(0.5, 0.02, f"Too many clusters ({len(unique_labels)}) to show legend",
                            ha='center', va='bottom', transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                elif len(unique_labels) > 10:
                    plt.legend(loc='upper right', fontsize='x-small', ncol=2, 
                            markerscale=0.8, handletextpad=0.5)
                else:
                    plt.legend(loc='upper right', fontsize='small')
            
            # 设置XY轴比例一致（正方形），这是子图的核心要求
            ax.set_aspect('equal', adjustable='box')
            
            # 添加运行时间标注到图例之上，在XY轴框外但仍在子图内
            time_taken = result.get("time", 0)
            if time_taken > 0:
                time_text = f"Time: {time_taken:.3f}s"
                # 将时间标注放在图例上方，使用更高的y坐标位置
                plt.text(0.98, 1.05, time_text,
                        ha='right', va='bottom', transform=plt.gca().transAxes,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                        fontsize=9, fontweight='bold', clip_on=False)
            
            # 子图标题，调整位置避免与时间标注重合
            title_suffix = f" (colored by {self.dataset_info.get('target_column_name', 'target')})" if has_target_column else ""
            plt.title(f"{name}\nClusters: {n_clusters}{title_suffix}", fontsize=11, pad=20)
            plt.xticks([])
            plt.yticks([])
            
            plot_num += 1
        
        # 添加整体标题，包含降维信息
        title = f"Clustering Results - {self.dataset_name}" if self.dataset_name else "Clustering Results"
        if self.X_2d is not None:
            title += f" ({self.reduction_method.upper()}-{self.n_components}D)"
        
        # 动态调整整体标题位置，确保不与子图标题重合
        suptitle_y = top_margin + (1 - top_margin) * 0.5
        plt.suptitle(title, fontsize=16, y=suptitle_y)
        
        # 使用更精细的布局调整
        plt.tight_layout(rect=[0, 0, 1, top_margin])
        
        # 保存图表
        if save_path is None and self.dataset_name:
            # 根据新的保存结构确定保存路径
            if hasattr(self, 'current_sub_folder') and self.current_sub_folder:
                save_dir = self.output_dir / self.current_sub_folder / self.dataset_name
            else:
                save_dir = self.output_dir / self.dataset_name
            save_path = save_dir / f"{self.dataset_name}_clustering_visualization_{self.reduction_method}_{self.n_components}d.png"
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            self.saved_paths['visualization'] = str(save_path)
            self._log(f"图表已保存至 {save_path}")
        
        plt.show()
        return self
    
    def compare_performance(self):
        """比较算法性能"""
        if not self.results:
            logger.error("错误: 请先运行算法")
            return self
            
        # 提取成功运行的算法及其运行时间
        performance = {
            name: result["time"] 
            for name, result in self.results.items() 
            if result.get("success", False)
        }
        
        if not performance:
            logger.info("没有成功运行的算法")
            return self
            
        # 按照运行时间排序
        algorithms_ordered = sorted(performance.items(), key=lambda x: x[1])
        
        logger.info("\n算法性能比较（从快到慢）:")
        for name, time_taken in algorithms_ordered:
            n_clusters = self.results[name].get("n_clusters", 0)
            silhouette = self.results[name].get("silhouette_score")
            silhouette_str = f", 轮廓系数: {silhouette:.3f}" if silhouette is not None else ""
            logger.info(f"{name:<20}: {time_taken:.3f}秒 - 找到{n_clusters}个聚类{silhouette_str}")
            
        return self
    
    def validate_n_clusters(self, n_clusters, algorithm_name=None):
        """验证聚类数量的有效性"""
        if self.X is None:
            raise ValueError("请先加载数据")
            
        if n_clusters < 1:
            raise ValueError(f"聚类数量必须大于0，当前值: {n_clusters}")
            
        if n_clusters > self.X.shape[0]:
            raise ValueError(f"聚类数量不能超过样本数量，当前值: {n_clusters}")
            
        return n_clusters
    
    def estimate_optimal_clusters(self, algorithm_name, max_clusters=10, method='silhouette'):
        """自动选择最优聚类数量"""
        if self.X is None:
            self._log("错误: 请先加载数据", 'error')
            return None
            
        max_clusters = min(max_clusters, self.X.shape[0] - 1)
        scores = []
        k_range = range(2, max_clusters + 1)
        
        self._log(f"正在为{algorithm_name}计算最优聚类数量...")
        
        try:
            if method == 'silhouette':
                for k in tqdm(k_range, desc="计算轮廓系数", disable=not self.verbose):
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                    labels = kmeans.fit_predict(self.X)
                    score = silhouette_score(self.X, labels)
                    scores.append(score)
                optimal_k = k_range[np.argmax(scores)]
                
            elif method == 'elbow':
                for k in tqdm(k_range, desc="计算惯性", disable=not self.verbose):
                    kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                    kmeans.fit(self.X)
                    scores.append(kmeans.inertia_)
                
                # 使用KneeLocator找到拐点
                knee_finder = KneeLocator(
                    list(k_range), scores,
                    curve='convex', direction='decreasing'
                )
                optimal_k = knee_finder.knee if knee_finder.knee is not None else 3
                
            elif method == 'bic':
                from sklearn.mixture import GaussianMixture
                for k in tqdm(k_range, desc="计算BIC", disable=not self.verbose):
                    gmm = GaussianMixture(n_components=k, random_state=self.random_state)
                    gmm.fit(self.X)
                    scores.append(gmm.bic(self.X))
                optimal_k = k_range[np.argmin(scores)]  # BIC越小越好
                
            else:
                self._log(f"不支持的评估方法: {method}，使用默认方法'silhouette'", 'warning')
                return self.estimate_optimal_clusters(algorithm_name, max_clusters, 'silhouette')
                
            self.optimal_clusters[algorithm_name] = optimal_k
            return optimal_k
            
        except Exception as e:
            self._log(f"计算最优聚类数量时出错: {e}", 'error')
            self._log("使用默认聚类数量: 3", 'warning')
            return 3

    def load_nested_folders(self, root_path, file_pattern="*.csv"):
        """
        加载嵌套文件夹结构：主文件夹包含多个子文件夹，每个子文件夹包含多个CSV文件
        
        Parameters:
        -----------
        root_path : str
            根文件夹路径
        file_pattern : str, default="*.csv"
            文件匹配模式
        """
        root_path = Path(root_path)
        
        if not root_path.exists():
            self._log(f"错误: 根文件夹不存在 {root_path}", 'error')
            return self
        
        if not root_path.is_dir():
            self._log(f"错误: 路径不是文件夹 {root_path}", 'error')
            return self
        
        self.nested_folders = {}
        total_csv_count = 0
        
        # 遍历根文件夹下的所有子文件夹
        for sub_folder_path in root_path.iterdir():
            if sub_folder_path.is_dir():
                sub_folder_name = sub_folder_path.name
                self.nested_folders[sub_folder_name] = []
                
                # 在子文件夹中查找CSV文件
                csv_pattern = sub_folder_path / file_pattern
                csv_files = list(glob.glob(str(csv_pattern)))
                
                if csv_files:
                    self.nested_folders[sub_folder_name] = csv_files
                    total_csv_count += len(csv_files)
                    
                    self._log(f"在子文件夹 '{sub_folder_name}' 中找到 {len(csv_files)} 个CSV文件:")
                    for i, file_path in enumerate(csv_files, 1):
                        file_name = Path(file_path).name
                        self._log(f"    {i}. {file_name}")
                else:
                    self._log(f"子文件夹 '{sub_folder_name}' 中没有找到CSV文件")
                    # 移除空的子文件夹
                    del self.nested_folders[sub_folder_name]
        
        if not self.nested_folders:
            self._log(f"错误: 在根文件夹 {root_path} 的子文件夹中没有找到任何CSV文件", 'error')
            return self
        
        self._log(f"总共在 {len(self.nested_folders)} 个子文件夹中找到 {total_csv_count} 个CSV文件")
        return self
    
    def _generate_nested_dataset_name(self, file_path, parent_folder, sub_folder):
        """生成嵌套结构的数据集名称：子文件夹_文件名（不包含主文件夹名称）"""
        file_name = Path(file_path).stem  # 不含扩展名的文件名
        return f"{sub_folder}_{file_name}"
    
    def _load_single_csv_nested(self, file_path, parent_folder, sub_folder, label_column=None):
        """加载嵌套结构中的单个CSV文件"""
        try:
            # 生成数据集名称（不包含主文件夹名称）
            self.dataset_name = self._generate_nested_dataset_name(file_path, parent_folder, sub_folder)
            self.current_file_path = file_path
            self.current_parent_folder = parent_folder
            self.current_sub_folder = sub_folder
            
            df = pd.read_csv(file_path)
            
            # 检查最后一列的名称，决定是否排除
            last_column_name = df.columns[-1]
            if last_column_name == 'property_log':
                # 如果最后一列名为property_log，排除最后一列
                df_features = df.iloc[:, :-1]
                self.y_true = None  # 没有目标列
                excluded_columns = 1
                self._log(f"检测到最后一列为 'property_log'，排除最后一列")
            else:
                # 否则保留最后一列作为目标列，用于可视化着色
                df_features = df.iloc[:, :-1]
                self.y_true = df.iloc[:, -1].values  # 最后一列作为目标列
                excluded_columns = 0  # 实际上没有排除，只是分离了特征和目标
                self._log(f"最后一列为 '{last_column_name}'，保留作为目标列用于可视化着色")
            
            if label_column is not None and label_column >= 0:
                # 如果指定了标签列，从特征数据中排除该列
                if label_column >= df_features.shape[1]:
                    self._log(f"警告: 标签列索引 {label_column} 超出范围，忽略标签列", 'warning')
                    self.X = df_features.values
                else:
                    self.X = df_features.iloc[:, [i for i in range(df_features.shape[1]) if i != label_column]].values
                    # 如果指定了标签列，则使用指定的标签列而不是最后一列
                    if last_column_name != 'property_log':
                        # 如果最后一列不是property_log，需要从原始df中获取标签列
                        self.y_true = df.iloc[:, label_column].values
            else:
                self.X = df_features.values
                # y_true已经在上面设置了
                
            # 保存数据集信息
            self.dataset_info = {
                'file_path': str(file_path),
                'parent_folder_name': parent_folder,
                'sub_folder_name': sub_folder,
                'dataset_name': self.dataset_name,
                'original_file_name': Path(file_path).name,
                'n_samples': self.X.shape[0],
                'n_features': self.X.shape[1],
                'has_labels': self.y_true is not None,
                'has_target_column': self.y_true is not None and last_column_name != 'property_log',
                'target_column_name': last_column_name if last_column_name != 'property_log' else None,
                'load_time': datetime.now().isoformat(),
                'reduction_method': self.reduction_method,
                'reduction_components': self.n_components,
                'reduction_params': self.reduction_params,
                'last_column_name': last_column_name,
                'excluded_columns': excluded_columns
            }
            
            self._log(f"成功加载数据集 '{self.dataset_name}': {self.X.shape[0]}行, {self.X.shape[1]}列")
            if self.y_true is not None and last_column_name != 'property_log':
                self._log(f"目标列 '{last_column_name}' 将用于可视化着色")
            return True
        except Exception as e:
            self._log(f"加载CSV文件 {file_path} 时出错: {e}", 'error')
            return False

    def process_nested_folders(self, root_path, label_column=None, file_pattern="*.csv", 
                              auto_select_clusters=True, cluster_method='bic',
                              reduction_method='tsne', n_components=2, **reduction_kwargs):
        """
        批量处理嵌套文件夹结构中的所有CSV文件
        
        Parameters:
        -----------
        root_path : str
            根文件夹路径
        label_column : int, optional
            标签列索引
        file_pattern : str, default="*.csv"
            文件匹配模式
        auto_select_clusters : bool, default=True
            是否自动选择最优聚类数
        cluster_method : str, default='bic'
            聚类数选择方法
        reduction_method : str, default='tsne'
            降维方法
        n_components : int, default=2
            降维维度
        **reduction_kwargs : dict
            降维方法的额外参数
        """
        # 设置降维配置
        self.set_reduction_config(reduction_method, n_components, **reduction_kwargs)
        
        # 加载嵌套文件夹结构
        self.load_nested_folders(root_path, file_pattern)
        
        if not self.nested_folders:
            return self
        
        # 计算总文件数
        total_files = sum(len(files) for files in self.nested_folders.values())
        
        self._log(f"开始批量处理嵌套文件夹结构中的 {total_files} 个CSV文件...")
        self._log(f"降维配置: {self.reduction_method.upper()}, 维度: {self.n_components}")
        
        # 获取根文件夹名称作为父文件夹名称
        parent_folder_name = Path(root_path).name
        
        # 初始化嵌套批量处理结果
        self.nested_batch_results = {}
        
        file_counter = 0
        
        # 处理每个子文件夹
        for sub_folder_name, csv_files in self.nested_folders.items():
            self._log(f"\n{'='*80}")
            self._log(f"处理子文件夹: {sub_folder_name} ({len(csv_files)} 个文件)")
            self._log(f"{'='*80}")
            
            # 初始化子文件夹的结果记录
            if parent_folder_name not in self.nested_batch_results:
                self.nested_batch_results[parent_folder_name] = {}
            self.nested_batch_results[parent_folder_name][sub_folder_name] = {}
            
            # 处理子文件夹中的每个CSV文件
            for i, csv_file in enumerate(csv_files, 1):
                file_counter += 1
                self._log(f"\n{'-'*60}")
                self._log(f"处理文件 {file_counter}/{total_files} (子文件夹 {i}/{len(csv_files)}): {Path(csv_file).name}")
                self._log(f"路径: {csv_file}")
                self._log(f"{'-'*60}")
                
                try:
                    # 重置状态
                    self._reset_for_new_file()
                    
                    # 加载单个文件
                    if not self._load_single_csv_nested(csv_file, parent_folder_name, sub_folder_name, label_column):
                        continue
                    
                    # 预处理数据
                    self.preprocess()
                    
                    # 添加算法
                    if auto_select_clusters:
                        self.add_all_sklearn_algorithms(auto_select=True, method=cluster_method)
                    else:
                        self.add_all_sklearn_algorithms(auto_select=False, n_clusters=3)
                    
                    # 运行算法
                    self.run()
                    
                    # 保存结果
                    self.save_results()
                    
                    # 生成可视化
                    self.visualize_from_saved_data()
                    
                    # 比较性能
                    self.compare_performance()
                    
                    # 打印保存路径信息
                    self.print_saved_paths()
                    
                    # 保存嵌套批量处理结果
                    self.nested_batch_results[parent_folder_name][sub_folder_name][self.dataset_name] = {
                        'file_path': csv_file,
                        'success': True,
                        'n_algorithms': len([r for r in self.results.values() if r.get('success', False)]),
                        'processing_time': datetime.now().isoformat(),
                        'saved_paths': self.saved_paths.copy()
                    }
                    
                    self._log(f"文件 '{Path(csv_file).name}' 处理完成")
                    
                except Exception as e:
                    self._log(f"处理文件 {csv_file} 时出错: {e}", 'error')
                    dataset_name = self._generate_nested_dataset_name(csv_file, parent_folder_name, sub_folder_name)
                    self.nested_batch_results[parent_folder_name][sub_folder_name][dataset_name] = {
                        'file_path': csv_file,
                        'success': False,
                        'error': str(e),
                        'processing_time': datetime.now().isoformat()
                    }
        
        # 保存嵌套批量处理摘要
        self._save_nested_batch_summary()
        
        # 统计处理结果
        total_success = 0
        total_files_processed = 0
        for parent_results in self.nested_batch_results.values():
            for sub_results in parent_results.values():
                for result in sub_results.values():
                    total_files_processed += 1
                    if result.get('success', False):
                        total_success += 1
        
        self._log(f"\n{'='*80}")
        self._log("嵌套文件夹批量处理完成!")
        self._log(f"成功处理: {total_success}/{total_files_processed} 个文件")
        self._log(f"处理了 {len(self.nested_folders)} 个子文件夹")
        self._log(f"{'='*80}")
        
        return self
    
    def _save_nested_batch_summary(self):
        """保存嵌套批量处理摘要"""
        if not self.nested_batch_results:
            return
        
        # 创建嵌套批量处理摘要目录
        nested_summary_dir = self.output_dir / "nested_batch_summary"
        nested_summary_dir.mkdir(exist_ok=True)
        
        # 保存详细的嵌套处理结果
        summary_data = []
        for parent_folder, sub_folders in self.nested_batch_results.items():
            for sub_folder, datasets in sub_folders.items():
                for dataset_name, result in datasets.items():
                    summary_data.append({
                        '父文件夹': parent_folder,
                        '子文件夹': sub_folder,
                        '数据集名称': dataset_name,
                        '文件路径': result['file_path'],
                        '处理状态': '成功' if result['success'] else '失败',
                        '算法数量': result.get('n_algorithms', 0) if result['success'] else 0,
                        '错误信息': result.get('error', '') if not result['success'] else '',
                        '处理时间': result['processing_time']
                    })
        
        # 保存为CSV
        summary_df = pd.DataFrame(summary_data)
        summary_file = nested_summary_dir / f"nested_batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        # 保存为JSON
        json_file = nested_summary_dir / f"nested_batch_processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.nested_batch_results, f, ensure_ascii=False, indent=2)
        
        # 生成统计摘要
        stats_data = []
        for parent_folder, sub_folders in self.nested_batch_results.items():
            for sub_folder, datasets in sub_folders.items():
                total_files = len(datasets)
                successful_files = sum(1 for r in datasets.values() if r.get('success', False))
                failed_files = total_files - successful_files
                
                stats_data.append({
                    '父文件夹': parent_folder,
                    '子文件夹': sub_folder,
                    '总文件数': total_files,
                    '成功文件数': successful_files,
                    '失败文件数': failed_files,
                    '成功率': f"{successful_files/total_files*100:.1f}%" if total_files > 0 else "0%"
                })
        
        # 保存统计摘要
        stats_df = pd.DataFrame(stats_data)
        stats_file = nested_summary_dir / f"nested_processing_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        self._log(f"嵌套批量处理摘要已保存到: {nested_summary_dir}")


def show_menu():
    """显示模式选择菜单"""
    print("\n" + "="*60)
    print("🔬 改进聚类框架 - 模式选择")
    print("="*60)
    print("请选择运行模式:")
    print("1. 单个文件处理 (demo_single_file)")
    print("2. 批量处理文件夹 (main)")
    print("3. 嵌套文件夹处理 (demo_nested_folders)")
    print("4. 混合处理模式 (demo_mixed_processing)")
    print("5. 退出")
    print("="*60)


def interactive_mode():
    """交互式模式选择"""
    while True:
        show_menu()
        try:
            choice = input("请输入选择 (1-5): ").strip()
            
            if choice == '1':
                print("\n🚀 启动单个文件处理模式...")
                demo_single_file()
                break
            elif choice == '2':
                print("\n🚀 启动批量处理文件夹模式...")
                main()
                break
            elif choice == '3':
                print("\n🚀 启动嵌套文件夹处理模式...")
                demo_nested_folders()
                break
            elif choice == '4':
                print("\n🚀 启动混合处理模式...")
                demo_mixed_processing()
                break
            elif choice == '5':
                print("\n👋 退出程序")
                break
            else:
                print("❌ 无效选择，请输入 1-5 之间的数字")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出程序")
            break
        except Exception as e:
            print(f"❌ 输入错误: {e}")


def main():
    """主函数示例 - 演示批量处理功能"""
    # 创建框架实例
    cf = ImprovedClusteringFramework(output_dir="clustering_results")
    
    # 方法1: 批量处理文件夹中的所有CSV文件
    folder_path = r"G:\jupyter\work_test_two\Polymer_Describe\Mordred"
    print("=== 批量处理文件夹中的CSV文件 ===")
    cf.process_folder(
        folder_path=folder_path,
        label_column=None,  # 如果有标签列，指定列索引
        file_pattern="*.csv",  # 文件匹配模式
        auto_select_clusters=True,  # 自动选择最优聚类数
        cluster_method='bic',  # 聚类数选择方法
        reduction_method='tsne',
        n_components=2
    )
    
    print("\n=== 批量处理完成 ===")


def demo_nested_folders():
    """演示嵌套文件夹处理功能"""
    print("=== 嵌套文件夹处理演示 ===")
    
    # 创建框架实例
    cf = ImprovedClusteringFramework(output_dir="Cluster_Silhouette")
    
    # 处理嵌套文件夹结构
    # 假设文件夹结构如下：
    # root_folder/
    #   ├── subfolder1/
    #   │   ├── data1.csv
    #   │   ├── data2.csv
    #   │   └── data3.csv
    #   ├── subfolder2/
    #   │   ├── data4.csv
    #   │   └── data5.csv
    #   └── subfolder3/
    #       └── data6.csv
    
    root_path = "/home/wzfeng/PolyBench/Polymer_Describe_"  # 替换为你的根文件夹路径
    
    cf.process_nested_folders(
        root_path=root_path,
        label_column=None,  # 如果有标签列，指定列索引
        file_pattern="*.csv",  # 文件匹配模式
        auto_select_clusters=True,  # 自动选择最优聚类数
        cluster_method='silhouette',  # 聚类数选择方法
        reduction_method='tsne',
        n_components=2
    )
    
    print("\n=== 嵌套文件夹处理完成 ===")


def demo_mixed_processing():
    """演示混合处理模式：既有单层文件夹，也有嵌套文件夹"""
    print("=== 混合处理模式演示 ===")
    
    # 创建框架实例
    cf = ImprovedClusteringFramework(output_dir="clustering_results")
    
    # 首先处理单层文件夹
    print("\n--- 处理单层文件夹 ---")
    single_folder_path = r"G:\jupyter\work_test_two\single_data"
    cf.process_folder(
        folder_path=single_folder_path,
        label_column=None,
        auto_select_clusters=True,
        cluster_method='silhouette',
        reduction_method='pca',
        n_components=2
    )
    
    # 然后处理嵌套文件夹
    print("\n--- 处理嵌套文件夹 ---")
    nested_folder_path = r"G:\jupyter\work_test_two\nested_data"
    cf.process_nested_folders(
        root_path=nested_folder_path,
        label_column=None,
        auto_select_clusters=True,
        cluster_method='bic',
        reduction_method='tsne',
        n_components=2
    )
    
    print("\n=== 混合处理完成 ===")


def demo_single_file():
    """演示单个文件处理功能"""
    print("=== 单个文件处理演示 ===")
    
    # 创建框架实例
    cf = ImprovedClusteringFramework(output_dir="clustering_results")
    
    # 处理单个CSV文件
    file_path = r"G:\jupyter\work_test_two\Polymer_Describe_\Mordred\Glass transition temperature.csv"  # 替换为实际文件路径
    
    try:
        # 加载单个文件
        cf.load_from_csv(file_path, label_column=None)
        
        # 设置降维配置
        cf.set_reduction_config('tsne', 2)
        
        # 预处理数据
        cf.preprocess()
        
        # 添加算法
        cf.add_all_sklearn_algorithms(auto_select=True, method='silhouette')
        
        # 运行算法
        cf.run()
        
        # 保存结果
        cf.save_results()
        
        # 生成可视化
        cf.visualize_from_saved_data()
        
        # 比较性能
        cf.compare_performance()
        
        # 打印保存路径
        cf.print_saved_paths()
        
        print("\n=== 单个文件处理完成 ===")
        
    except Exception as e:
        print(f"处理单个文件时出错: {e}")
        print("请确保文件路径正确且文件存在")


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1:
        # 命令行参数模式
        mode = sys.argv[1]
        if mode == "single":
            demo_single_file()
        elif mode == "batch":
            main()
        elif mode == "nested":
            demo_nested_folders()
        elif mode == "mixed":
            demo_mixed_processing()
        else:
            print(f"❌ 未知模式: {mode}")
            print("可用模式: single, batch, nested, mixed")
    else:
        # 交互式模式
        interactive_mode() 