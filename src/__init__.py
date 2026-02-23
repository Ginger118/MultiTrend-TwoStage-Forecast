"""
MBGA两阶段预测模型核心包

Multi-Attention Bidirectional GRU 两阶段COVID-19预测系统
用于呼吸道疾病住院数据的时间序列预测

主要组件:
- 数据预处理: MinMaxScaler, DataPreprocessor
- 模型: MABGModel, EnhancedMABG, SimpleLSTM, SimpleGRU
- 训练: ModelTrainer, train_and_evaluate_model
- 流程: TwoStagePipeline
"""

__version__ = '1.0.0'
__author__ = 'MBGA Team'

from .data import (
    MinMaxScaler,
    DataPreprocessor,
    create_data_loader,
    load_flusurveillance_csv,
    split_data_by_covid_period
)

from .models import (
    MABGModel,
    EnhancedMABG,
    SimpleLSTM,
    SimpleGRU
)

from .training import (
    ModelTrainer,
    train_and_evaluate_model
)

from .pipeline import TwoStagePipeline

from .utils import (
    get_default_config,
    load_default_config
)

__all__ = [
    # 数据处理
    'MinMaxScaler',
    'DataPreprocessor',
    'create_data_loader',
    'load_flusurveillance_csv',
    'split_data_by_covid_period',
    
    # 模型
    'MABGModel',
    'EnhancedMABG',
    'SimpleLSTM',
    'SimpleGRU',
    
    # 训练
    'ModelTrainer',
    'train_and_evaluate_model',
    
    # 流程
    'TwoStagePipeline',
    
    # 工具
    'get_default_config',
    'load_default_config',
]
