"""
数据预处理模块 - 精简版
只保留MABG模型训练所需的核心功能
"""

import numpy as np
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import torch


class MinMaxScaler:
    """Min-Max标准化器，将数据缩放到[min_val, max_val]范围"""
    
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.data_min_ = None
        self.data_max_ = None
        self.scale_ = None
        self.is_fitted = False
    
    def fit(self, X):
        """拟合缩放器，计算数据的最小值和最大值"""
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        
        # 避免除以零
        range_val = self.data_max_ - self.data_min_
        range_val[range_val == 0] = 1.0
        
        self.scale_ = (self.max_val - self.min_val) / range_val
        self.is_fitted = True
        
        return self
    
    def transform(self, X):
        """应用缩放变换"""
        if not self.is_fitted:
            raise ValueError("必须先调用fit()方法拟合缩放器")
        
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_scaled = (X - self.data_min_) * self.scale_ + self.min_val
        
        # 移除限制，允许外推 (MinMaxScaler通常不应截断)
        # X_scaled = np.clip(X_scaled, self.min_val, self.max_val)
        
        return X_scaled
    
    def fit_transform(self, X):
        """拟合并转换"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """反向转换，将缩放后的数据还原"""
        if not self.is_fitted:
            raise ValueError("必须先调用fit()方法拟合缩放器")
        
        X_scaled = np.asarray(X_scaled)
        if len(X_scaled.shape) == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        
        X = (X_scaled - self.min_val) / self.scale_ + self.data_min_
        
        return X
    
    def to_dict(self):
        """转换为字典（用于序列化）"""
        return {
            'min_val': self.min_val,
            'max_val': self.max_val,
            'data_min_': self.data_min_,
            'data_max_': self.data_max_,
            'scale_': self.scale_,
            'is_fitted': self.is_fitted
        }
    
    def from_dict(self, d):
        """从字典恢复（用于反序列化）"""
        self.min_val = d['min_val']
        self.max_val = d['max_val']
        self.data_min_ = d['data_min_']
        self.data_max_ = d['data_max_']
        self.scale_ = d['scale_']
        self.is_fitted = d['is_fitted']


def encode_seasonal_features(week_numbers: np.ndarray) -> np.ndarray:
    """
    编码季节特征
    
    Args:
        week_numbers: 周数数组 (1-52/53)
    
    Returns:
        seasonal_features: 形状为 (n_samples, 4) 的数组
            - season_idx_norm: 归一化的季节索引 (0-3) -> (0-1)
            - sin_annual: 年度周期的sin编码
            - cos_annual: 年度周期的cos编码
            - intensity: 季节强度 (冬季1.0, 其他0.2-0.6)
    """
    week_numbers = np.asarray(week_numbers)
    
    # 季节索引 (0=春, 1=夏, 2=秋, 3=冬)
    season_idx = ((week_numbers - 1) // 13) % 4
    season_idx_norm = season_idx / 3.0  # 归一化到 [0, 1]
    
    # 年度周期编码
    annual_phase = 2 * np.pi * (week_numbers - 1) / 52.0
    sin_annual = np.sin(annual_phase)
    cos_annual = np.cos(annual_phase)
    
    # 季节强度（冬季流感高发）
    intensity = np.where(season_idx == 3, 1.0,  # 冬季
                np.where(season_idx == 0, 0.6,  # 春季
                np.where(season_idx == 2, 0.4,  # 秋季
                         0.2)))                  # 夏季
    
    return np.column_stack([
        season_idx_norm,
        sin_annual,
        cos_annual,
        intensity
    ])


def week_of_year_from_date(dates: pd.Series) -> np.ndarray:
    """
    从日期提取周数
    
    Args:
        dates: 日期Series
    
    Returns:
        week_numbers: 周数数组
    """
    dates = pd.to_datetime(dates)
    week_numbers = dates.dt.isocalendar().week.values
    return week_numbers


class DataPreprocessor:
    """数据预处理器 - 精简版"""
    
    def __init__(self, include_seasonal_features=True):
        """
        初始化
        
        Args:
            include_seasonal_features: 是否包含季节特征
        """
        self.target_scaler = MinMaxScaler(min_val=0.0, max_val=1.0)
        self.trend_scaler = MinMaxScaler(min_val=0.0, max_val=1.0)
        self.is_fitted = False
        self.include_seasonal_features = include_seasonal_features
    
    def prepare_features(self, df: pd.DataFrame, value_col: str = 'value', feature_cols: list = None) -> np.ndarray:
        """
        准备训练特征
        
        Args:
            df: 包含数据的DataFrame
            value_col: 目标数值列名
            feature_cols: 额外的特征列名列表 (如Trends)
            
        Returns:
            features: 形状为 (n_samples, n_features) 的特征数组（未归一化）
        """
        if value_col not in df.columns:
            raise ValueError(f"DataFrame中未找到列: {value_col}")
        
        # 基础特征列表: [Target, Trend1, Trend2, ...]
        cols_to_extract = [value_col]
        if feature_cols:
            for col in feature_cols:
                if col in df.columns:
                    cols_to_extract.append(col)
                else:
                    print(f"Warning: Feature column '{col}' not found in DataFrame. Skipping.")
            
        # 提取基础特征
        base_features = df[cols_to_extract].values.astype(np.float32)
        
        if not self.include_seasonal_features:
            return base_features
        
        # 季节特征
        if 'week_number' not in df.columns:
            if 'date' in df.columns:
                week_numbers = week_of_year_from_date(df['date'])
            else:
                raise ValueError("DataFrame中必须包含 'week_number' 或 'date' 列以生成季节特征")
        else:
            week_numbers = df['week_number'].values
        
        seasonal_features = encode_seasonal_features(week_numbers).astype(np.float32)
        
        # 合并特征
        combined_features = np.hstack([base_features, seasonal_features])
        
        return combined_features
    
    def normalize_data(self, data):
        """
        使用Min-Max标准化将数据缩放到[0,1]范围
        
        Args:
            data: 输入数据，形状为 (n_samples, n_features)
            
        Returns:
            normalized_data: 标准化后的数据
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values.astype(np.float32)
        else:
            data_array = np.asarray(data, dtype=np.float32)
        
        # 确定哪些列是数值/Trends，哪些是季节特征
        # 季节特征总是最后4列 (如果启用)
        if self.include_seasonal_features:
            n_seasonal = 4
            if data_array.shape[1] <= n_seasonal:
                # 异常情况，可能只有季节特征？或者数据维度不对
                # 假设全部归一化
                cols_to_norm = data_array
                seasonal_cols = None
            else:
                cols_to_norm = data_array[:, :-n_seasonal]
                seasonal_cols = data_array[:, -n_seasonal:]
        else:
            cols_to_norm = data_array
            seasonal_cols = None
            
        # 分离 Target (第0列) 和 Trends (第1列到最后)
        target_col = cols_to_norm[:, 0:1]
        trend_cols = cols_to_norm[:, 1:] if cols_to_norm.shape[1] > 1 else None
        
        if not self.is_fitted:
            # 拟合 Target Scaler
            norm_target = self.target_scaler.fit_transform(target_col)
            
            # 拟合 Trend Scaler (如果有)
            if trend_cols is not None and trend_cols.shape[1] > 0:
                norm_trends = self.trend_scaler.fit_transform(trend_cols)
            else:
                norm_trends = None
                
            self.is_fitted = True
        else:
            norm_target = self.target_scaler.transform(target_col)
            if trend_cols is not None and trend_cols.shape[1] > 0:
                # 注意：这里假设Trends特征的数量和顺序与fit时一致
                # 如果不一致可能会报错，但在这个pipeline中通常是一致的
                norm_trends = self.trend_scaler.transform(trend_cols)
            else:
                norm_trends = None
        
        # 重新拼接: [NormTarget, NormTrends, Seasonal]
        parts = [norm_target]
        if norm_trends is not None:
            parts.append(norm_trends)
        if seasonal_cols is not None:
            parts.append(seasonal_cols)
            
        return np.hstack(parts)
    
    def inverse_transform(self, normalized_data):
        """
        将标准化数据转换回原始尺度 (只针对Target)
        
        Args:
            normalized_data: 标准化后的数据 (可以是只包含Target的列，也可以是完整特征)
            
        Returns:
            original_data: 原始尺度的Target数据
        """
        if not self.is_fitted:
            raise ValueError("Scaler尚未拟合！")
            
        normalized_data = np.asarray(normalized_data)
        
        # 如果输入是1D或只有1列，假设它是Target列
        if len(normalized_data.shape) == 1 or normalized_data.shape[1] == 1:
            return self.target_scaler.inverse_transform(normalized_data.reshape(-1, 1))
        
        # 如果输入包含多列，取第一列作为Target
        target_col = normalized_data[:, 0:1]
        return self.target_scaler.inverse_transform(target_col)
    
    def save(self, path: str):
        """将内部scaler保存到磁盘"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'target_scaler': self.target_scaler.to_dict(),
            'trend_scaler': self.trend_scaler.to_dict(),
            'is_fitted': self.is_fitted,
            'include_seasonal_features': self.include_seasonal_features
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str):
        """从磁盘加载DataPreprocessor"""
        if not os.path.exists(path):
            raise FileNotFoundError(f'预处理器文件不存在: {path}')
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        inst = cls(include_seasonal_features=state.get('include_seasonal_features', True))
        inst.target_scaler.from_dict(state['target_scaler'])
        inst.trend_scaler.from_dict(state['trend_scaler'])
        inst.is_fitted = state['is_fitted']
        return inst


class TimeSeriesDataset(Dataset):
    """时间序列数据集（用于PyTorch DataLoader）"""
    
    def __init__(self, data, sequence_length=20, target_col_idx=0):
        """
        初始化
        
        Args:
            data: 输入数据，形状为 (n_samples, n_features)
            sequence_length: 序列长度
            target_col_idx: 目标列索引（默认0，即第一列）
        """
        self.data = data
        self.sequence_length = sequence_length
        self.target_col_idx = target_col_idx
        
        # 确保数据是2D
        if len(data.shape) == 1:
            self.data = data.reshape(-1, 1)
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            sequence: 输入序列，形状为 (sequence_length, n_features)
            target: 目标值（标量）
        """
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length, self.target_col_idx]
        
        return torch.FloatTensor(sequence), torch.FloatTensor([target])


def create_data_loader(data, sequence_length=20, batch_size=8, shuffle=True):
    """
    创建PyTorch数据加载器
    
    Args:
        data: 输入数据
        sequence_length: 序列长度
        batch_size: 批次大小
        shuffle: 是否打乱
    
    Returns:
        DataLoader实例
    """
    dataset = TimeSeriesDataset(data, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_google_trends(csv_path: str, col_name: str) -> pd.DataFrame:
    """
    加载Google Trends数据 (支持每月或每周)
    
    Args:
        csv_path: CSV文件路径
        col_name: 特征列名 (如 'flu_Trends')
        
    Returns:
        DataFrame: 
            - 如果是月度数据: ['month', col_name]
            - 如果是周度数据: ['year_week', col_name]
    """
    try:
        # 尝试读取前几行判断格式
        with open(csv_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            
        # 如果是清洗后的格式 (date, value)
        if 'date' in first_line and 'value' in first_line:
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            df[col_name] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
            
            # 生成 year_week 用于合并
            # 注意：Trends通常是周日，RESP通常是周六。使用ISO周数对齐。
            # 格式: YYYY-WW
            df['year_week'] = df['date'].dt.strftime('%G-%V')
            
            # 如果同一周有多行(不太可能)，取平均
            df_agg = df.groupby('year_week')[col_name].mean().reset_index()
            return df_agg
            
        # 否则假设是旧的月度格式 (跳过前3行)
        df = pd.read_csv(csv_path, skiprows=3, names=['month', col_name])
        # 确保数值型
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        # 确保month是字符串格式 YYYY-MM
        df['month'] = df['month'].astype(str)
        return df
    except Exception as e:
        print(f"Error loading Google Trends data ({csv_path}): {e}")
        return pd.DataFrame(columns=['month', col_name])

def load_and_merge_data(resp_path: str, trends_dict: dict = None, lags: list = [1, 2]) -> pd.DataFrame:
    """
    加载流行病学数据并合并多个Google Trends数据，并生成滞后特征
    
    Args:
        resp_path: Cleaned_RESP.csv 路径
        trends_dict: 字典 {feature_name: csv_path}
        lags: 需要生成的滞后周数列表，例如 [1, 2] 表示生成 t-1 和 t-2
        
    Returns:
        DataFrame包含 ['date', 'week_number', 'value', ...trends_features..., ...lag_features...]
    """
    # 1. 加载 RESP 数据
    df_resp = pd.read_csv(resp_path)
    df_resp['date'] = pd.to_datetime(df_resp['Date'])
    df_resp['value'] = pd.to_numeric(df_resp['WeeklyRate'], errors='coerce')
    df_resp['week_number'] = df_resp['date'].dt.isocalendar().week
    
    # 创建关联键
    # 'month' (YYYY-MM) 用于月度数据合并
    df_resp['month'] = df_resp['date'].dt.strftime('%Y-%m')
    # 'year_week' (YYYY-WW) 用于周度数据合并 (ISO Year-Week)
    df_resp['year_week'] = df_resp['date'].dt.strftime('%G-%V')
    
    merged_df = df_resp
    
    # 2. 加载并合并所有 Google Trends 数据
    if trends_dict:
        for feature_name, csv_path in trends_dict.items():
            if not csv_path or not os.path.exists(csv_path):
                print(f"Warning: Trends file not found: {csv_path}")
                continue
                
            df_trends = load_google_trends(csv_path, feature_name)
            
            if df_trends.empty:
                merged_df[feature_name] = 0
                continue

            # 判断是周度还是月度
            if 'year_week' in df_trends.columns:
                # 周度合并
                merged_df = pd.merge(merged_df, df_trends, on='year_week', how='left')
            elif 'month' in df_trends.columns:
                # 月度合并
                merged_df = pd.merge(merged_df, df_trends, on='month', how='left')
            
            # 处理缺失值 (插值)
            # 对于月度数据合并后，同一月的每周会有相同的值，这是预期的
            # 对于周度数据，如果某些周缺失，interpolate会填补
            merged_df[feature_name] = merged_df[feature_name].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            # 生成滞后特征
            for lag in lags:
                lag_col_name = f"{feature_name}_lag{lag}"
                merged_df[lag_col_name] = merged_df[feature_name].shift(lag).fillna(0)

    # 选择需要的列
    cols = ['date', 'week_number', 'value']
    if trends_dict:
        for feature_name in trends_dict.keys():
            cols.append(feature_name)
            for lag in lags:
                cols.append(f"{feature_name}_lag{lag}")
        
    final_df = merged_df[cols].sort_values('date').reset_index(drop=True)
    
    return final_df

def load_flusurveillance_csv(csv_path: str, 
                              age_category: str = "Overall",
                              sex_category: str = "Overall", 
                              race_category: str = "Overall") -> pd.DataFrame:
    """
    加载并解析FluSurveillance数据
    支持多种格式（CDC下载数据、预处理数据）
    """
    try:
        # 尝试方法1：直接读取 (针对标准的 WeeklyRate/Date 格式)
        df = pd.read_csv(csv_path)
        if 'Date' in df.columns and 'WeeklyRate' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df['value'] = pd.to_numeric(df['WeeklyRate'], errors='coerce')
            df['week_number'] = df['date'].dt.isocalendar().week
            return df[['date', 'week_number', 'value']]
    except:
        pass

    try:
        # 尝试方法2：跳过前2行 (CDC下载格式)
        df = pd.read_csv(csv_path, skiprows=2)
        
        # 检查是否包含必要的列
        required_cols = ['YEAR', 'WEEK', 'WEEKLY RATE']
        # 注意: 如果有重复列名YEAR, pandas可能会重命名为 YEAR.1
        
        # 找到 WEEKLY RATE 列
        value_col = None
        for col in df.columns:
            if 'WEEKLY RATE' in str(col).upper() and 'ADJUSTED' not in str(col).upper():
                value_col = col
                break
        
        if value_col:
            # 找到 YEAR 和 WEEK 列
            # 通常 YEAR 第一个列是 '2009-10' 格式，第二个 '2009' 是数值
            # 我们寻找纯数字的 YEAR 列
            year_col = 'YEAR'
            if 'YEAR.1' in df.columns: # 处理重复列名
                year_col = 'YEAR.1'
            
            df = df[df[year_col] != ''] # 过滤空行
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            df['WEEK'] = pd.to_numeric(df['WEEK'], errors='coerce')
            
            # 由于 YEAR+WEEK 转 Date 需要处理跨年周的问题，比较复杂
            # 这里简化：使用 MMWR week 转换
            # 或者使用简单的 apply
            def week_to_date(row):
                if pd.isna(row[year_col]) or pd.isna(row['WEEK']):
                    return pd.NaT
                y = int(row[year_col])
                w = int(row['WEEK'])
                # MMWR week 1 usually contains Jan 4th
                # 使用 ISO 转换作为近似
                try:
                    return pd.to_datetime(f"{y}-W{w}-1", format="%G-W%V-%u") # Monday
                except:
                    # Fallback
                    return pd.to_datetime(f"{y}-01-01") + pd.to_timedelta(w*7, unit='D')

            df['date'] = df.apply(week_to_date, axis=1)
            df['value'] = pd.to_numeric(df[value_col], errors='coerce')
            
            # 过滤特定的 Category
            if 'AGE CATEGORY' in df.columns:
                df = df[df['AGE CATEGORY'] == age_category]
            if 'SEX CATEGORY' in df.columns:
                df = df[df['SEX CATEGORY'] == sex_category]
            if 'RACE CATEGORY' in df.columns:
                df = df[df['RACE CATEGORY'] == race_category]
                
            df['week_number'] = df['WEEK']
            
            # 清理无效值
            df = df.dropna(subset=['value', 'date'])
            
            df = df.sort_values('date').reset_index(drop=True)
            return df[['date', 'week_number', 'value']]
            
    except Exception as e:
        print(f"CDC format parse failed: {e}")

    # 尝试方法3：旧格式/其他 (直接返回空)
    print(f"Warning: Could not parse {csv_path} with known formats.")
    return pd.DataFrame(columns=['date', 'week_number', 'value'])


def split_data_by_covid_period(df: pd.DataFrame, 
                                 covid_start_date: str = "2020-01-01") -> tuple:
    """
    按COVID-19时期分割数据
    
    Args:
        df: 数据DataFrame
        covid_start_date: COVID开始日期
    
    Returns:
        (pre_covid_df, covid_df)
    """
    covid_start = pd.to_datetime(covid_start_date)
    
    pre_covid = df[df['date'] < covid_start].copy()
    covid = df[df['date'] >= covid_start].copy()
    
    return pre_covid, covid
