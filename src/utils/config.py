"""
配置管理模块

提供默认配置和配置文件加载功能
"""

import json
import os
from typing import Dict, Optional


def get_default_config() -> Dict:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'data': {
            'population': 1000000,
            'sequence_length': 4,  # 减少到4以适应短数据集 (64周数据)
            'processed_dir': 'data/processed',
            'csv_path': 'Rates_of_Laboratory-Confirmed_RSV,_COVID-19,_and_Flu_Hospitalizations_from_the_RESP-NET_Surveillance_Systems_20251031.csv'
        },
        'features': {
            'use_seasonal': True,
            'use_trends': True,
            'trends_lags': [1, 2],
            'target_col': 'value',
            'baseline_features': ['flu_Trends'],
            'covid_features': ['flu_Trends', 'COVID_19_Trends', 'fever_Trends', 'loss_of_smell_Trends']
        },
        'model': {
            'type': 'simple_gru',  # 模型类型: 'simple_gru', 'simple_lstm', 'mabg'
            'input_size': 1,  # 将在pipeline中动态设置
            'hidden_size': 32,  # 小数据集用32（SimpleGRU约2K参数）
            'num_layers': 1,    # 单层避免过拟合
            'dropout': 0.5,     # 强正则化
            'attention_heads': 8,  # MABG模型用
            'reduction': 16     # MABG模型用
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.01,  # 强L2正则防止过拟合
            'batch_size': 16,      # 增大batch size提高泛化
            'patience': 20,        # 早停耐心值
            'lr_scheduler': True,  # 使用学习率衰减
            'lr_patience': 10,     # LR衰减耐心值
            'lr_factor': 0.5       # LR衰减因子
        },
        'paths': {
            'save_path': 'results/',
            'model_path': 'models/',
            'plot_path': 'plots/'
        }
    }


def load_config_from_file(config_path: str) -> Dict:
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        user_config = json.load(f)
    
    return user_config


def merge_configs(default_config: Dict, user_config: Dict) -> Dict:
    """
    递归合并配置（用户配置覆盖默认配置）
    
    Args:
        default_config: 默认配置
        user_config: 用户配置
        
    Returns:
        合并后的配置
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_default_config(config_path: Optional[str] = None) -> Dict:
    """
    加载配置（如果提供路径则合并用户配置）
    
    Args:
        config_path: 配置文件路径（可选）
        
    Returns:
        最终配置字典
    """
    default_config = get_default_config()
    
    if config_path and os.path.exists(config_path):
        user_config = load_config_from_file(config_path)
        return merge_configs(default_config, user_config)
    
    return default_config


def save_config(config: Dict, save_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {save_path}")
