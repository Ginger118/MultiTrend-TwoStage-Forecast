"""Two-stage COVID-19 hospitalization forecasting pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import random

import numpy as np
import pandas as pd
import torch

try:
    from .data.preprocessor import (
        DataPreprocessor,
        create_data_loader,
        load_flusurveillance_csv,
        load_and_merge_data,
        split_data_by_covid_period,
    )
    from .models.mabg import EnhancedMABG
    from .models.simple_models import SimpleLSTM, SimpleGRU
    from .models.multi_trend_attention import MultiTrendCrossAttention, SimplifiedMultiTrendAttention
    from .training.trainer import train_and_evaluate_model
    from .utils.config import load_default_config
except ImportError:  # pragma: no cover
    from src.data.preprocessor import (  # type: ignore
        DataPreprocessor,
        create_data_loader,
        load_flusurveillance_csv,
        load_and_merge_data,
        split_data_by_covid_period,
    )
    from src.models.mabg import EnhancedMABG  # type: ignore
    from src.models.simple_models import SimpleLSTM, SimpleGRU  # type: ignore
    from src.models.multi_trend_attention import MultiTrendCrossAttention, SimplifiedMultiTrendAttention  # type: ignore
    from src.training.trainer import train_and_evaluate_model  # type: ignore
    from src.utils.config import load_default_config  # type: ignore


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # deterministic=True ensures reproducibility but may slow down training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class TwoStagePipeline:
    """Manage baseline (pre-COVID) and COVID-specific forecasting stages."""

    def __init__(self, config_path: Optional[str] = None, include_seasonal_features: bool = True) -> None:
        self.config = load_default_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.include_seasonal_features = include_seasonal_features

        self.preprocessor = DataPreprocessor(include_seasonal_features=include_seasonal_features)
        
        # 新增 residual_scaler 用于标准化第二阶段的残差目标
        self.residual_scaler = DataPreprocessor(include_seasonal_features=False).target_scaler
        
        # 计算input_size：如果包含季节特征则为5（value + 4个季节特征），否则为1
        if include_seasonal_features:
            self.config["model"]["input_size"] = 5
        else:
            self.config["model"]["input_size"] = 1

        self.full_data: Optional[pd.DataFrame] = None
        self.pre_covid_data: Optional[pd.DataFrame] = None
        self.covid_data: Optional[pd.DataFrame] = None

        self.baseline_model: Optional[torch.nn.Module] = None
        self.covid_model: Optional[torch.nn.Module] = None

        self._print_device_banner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_and_prepare_data(self, resp_path: str, flu_path: str, trends_dict: Optional[Dict[str, str]] = None, covid_start_date: str = "2020-01-01") -> None:
        print(f"\nLoading data...")
        print(f"  RESP: {resp_path}")
        print(f"  Flu: {flu_path}")
        
        if trends_dict:
            print(f"  Trends: {list(trends_dict.keys())}")
            self.flu_data = load_and_merge_data(flu_path, trends_dict)
            self.resp_data = load_and_merge_data(resp_path, trends_dict)
        else:
            self.flu_data = load_flusurveillance_csv(flu_path)
            self.resp_data = load_flusurveillance_csv(resp_path)
            
        # Stage 1: Baseline Model 训练数据 -> 使用完整的 Flu/RSV 数据
        self.pre_covid_data = self.flu_data
        
        # Stage 2: COVID Model 训练数据 -> 使用 COVID 期间的 RESP 数据
        _, self.covid_data = split_data_by_covid_period(self.resp_data, covid_start_date)
        
        # 同时保存COVID期间对应的Flu数据，用于计算真实差值 (RESP - Flu = COVID患者数)
        _, self.covid_period_flu_data = split_data_by_covid_period(self.flu_data, covid_start_date)
        
        print(f"Data loaded.")
        print(f"  Baseline Training Data (Flu/RSV): {len(self.pre_covid_data)} samples")
        print(f"  COVID Training Data (RESP, post-{covid_start_date}): {len(self.covid_data)} samples")
        print(f"  COVID-period Flu Data (for ground truth): {len(self.covid_period_flu_data)} samples")

    def run_two_stage_training(self, resp_path: str, flu_path: str, trends_dict: Optional[Dict[str, str]] = None, covid_start_date: str = "2020-01-01", seed: int = 42) -> Dict[str, Dict]:
        print("\nStarting two-stage training pipeline")
        
        # Set seed for reproducibility
        set_seed(seed)
        
        print(f"  COVID start date: {covid_start_date}")

        self.load_and_prepare_data(resp_path, flu_path, trends_dict, covid_start_date)

        baseline_metrics = self.train_baseline_model()
        covid_metrics, residuals = self.compute_residuals_and_train_covid_model()

        os.makedirs(self.config["paths"]["model_path"], exist_ok=True)
        self.preprocessor.save(os.path.join(self.config["paths"]["model_path"], "preprocessor.pkl"))

        summary = {
            "baseline_results": baseline_metrics,
            "covid_results": covid_metrics,
            "residuals_stats": self._summarize_residuals(residuals),
            "data_info": self._summarize_data_info(),
        }

        print("Training completed.")
        print(f"  Baseline test loss: {baseline_metrics.get('test_loss', float('nan')):.4f}")
        print(f"  COVID test loss: {covid_metrics.get('test_loss', float('nan')):.4f}")
        print(f"  Residual mean: {summary['residuals_stats']['mean']:.4f}")

        return summary

    # ------------------------------------------------------------------
    # Stage 1: Baseline model
    # ------------------------------------------------------------------
    def train_baseline_model(self) -> Dict[str, float]:
        if self.pre_covid_data is None or self.pre_covid_data.empty:
            raise ValueError("Pre-COVID data not loaded. Call load_and_prepare_data() first.")

        # Baseline 只使用 flu_Trends 及其滞后特征
        baseline_features = self.config["features"].get("baseline_features", ["flu_Trends"])
        baseline_features_list = self._expand_with_lags(baseline_features)
        print(f"Preparing Baseline features: {baseline_features_list}")
        
        features = self.preprocessor.prepare_features(self.pre_covid_data, feature_cols=baseline_features_list)
        normalized = self.preprocessor.normalize_data(features)
        
        # 更新 input_size
        self.config["model"]["input_size"] = normalized.shape[1]
        print(f"Baseline Model Input Size: {self.config['model']['input_size']}")
        
        seq_len = self.config["data"]["sequence_length"]

        self._validate_sequence_length(len(normalized), seq_len, "pre-COVID")

        train_data, val_data, test_data = self._split_dataset(normalized)
        self._validate_split_lengths(
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
            sequence_length=seq_len,
        )

        batch_size = self.config["training"]["batch_size"]
        train_loader = create_data_loader(train_data, seq_len, batch_size, shuffle=True)
        val_loader = create_data_loader(val_data, seq_len, batch_size, shuffle=False)
        test_loader = create_data_loader(test_data, seq_len, batch_size, shuffle=False)

        self.baseline_model = self._build_model()
        print(f"  Baseline model parameters: {sum(p.numel() for p in self.baseline_model.parameters()):,}")

        metrics = train_and_evaluate_model(
            self.baseline_model,
            train_loader,
            val_loader,
            test_loader,
            model_name="Baseline",
            epochs=self.config["training"]["epochs"],
            save_path=os.path.join(self.config["paths"]["model_path"], "baseline_model.pth"),
        )

        print("Baseline model training finished.")
        return metrics

    # ------------------------------------------------------------------
    # Stage 2: COVID residual modelling
    # ------------------------------------------------------------------
    def compute_residuals_and_train_covid_model(self) -> Tuple[Dict[str, float], np.ndarray]:
        if self.baseline_model is None:
            raise ValueError("Baseline model must be trained before computing residuals.")
        if self.covid_data is None or self.covid_data.empty:
            raise ValueError("COVID-period data not loaded. Call load_and_prepare_data() first.")

        seq_len = self.config["data"]["sequence_length"]
        
        # 1. 准备COVID期间的RESP和Flu数据
        baseline_features = self.config["features"].get("baseline_features", ["flu_Trends"])
        baseline_features_list = self._expand_with_lags(baseline_features)
        
        # 准备RESP数据（总呼吸道病人）
        covid_resp_data = self.preprocessor.prepare_features(self.covid_data, feature_cols=baseline_features_list)
        covid_resp_normalized = self.preprocessor.normalize_data(covid_resp_data)
        
        # 准备同期Flu数据（正常流感病人）
        covid_flu_data = self.preprocessor.prepare_features(self.covid_period_flu_data, feature_cols=baseline_features_list)
        covid_flu_normalized = self.preprocessor.normalize_data(covid_flu_data)
        
        self._validate_sequence_length(len(covid_resp_normalized), seq_len, "COVID RESP")
        self._validate_sequence_length(len(covid_flu_normalized), seq_len, "COVID Flu")
        
        # 确保两个数据集长度一致
        min_len = min(len(covid_resp_normalized), len(covid_flu_normalized))
        covid_resp_normalized = covid_resp_normalized[:min_len]
        covid_flu_normalized = covid_flu_normalized[:min_len]
        
        # 2. 计算真实COVID患者数 = RESP实际值 - Flu实际值（而非Baseline预测值）
        resp_actual_values = covid_resp_normalized[seq_len:, 0]
        flu_actual_values = covid_flu_normalized[seq_len:, 0]
        covid_patients_raw = resp_actual_values - flu_actual_values
        
        # [Optimization] 标准化COVID患者数
        # 因为COVID患者数的范围可能与输入特征（0-1）差异很大，标准化有助于模型收敛
        self.residual_scaler.fit(covid_patients_raw.reshape(-1, 1))
        covid_patients_scaled = self.residual_scaler.transform(covid_patients_raw.reshape(-1, 1)).flatten()
        
        print(f"COVID Patients stats (Scaled): mean={covid_patients_scaled.mean():.4f}, std={covid_patients_scaled.std():.4f}, "
              f"min={covid_patients_scaled.min():.4f}, max={covid_patients_scaled.max():.4f}")

        # 3. 准备 COVID 模型所需的特征 (包含所有 Trends 及其滞后)
        covid_features = self.config["features"].get("covid_features", 
            ["flu_Trends", "COVID_19_Trends", "fever_Trends", "loss_of_smell_Trends"])
        covid_features_list = self._expand_with_lags(covid_features)
        print(f"Preparing COVID features: {covid_features_list}")
        
        # 使用新的 Preprocessor 处理 COVID 阶段数据 (因为特征数量不同)
        self.covid_preprocessor = DataPreprocessor(include_seasonal_features=self.include_seasonal_features)
        
        covid_features_full = self.covid_preprocessor.prepare_features(self.covid_data, feature_cols=covid_features_list)
        covid_normalized_full = self.covid_preprocessor.normalize_data(covid_features_full)
        
        # 确保特征数据与COVID患者数长度一致（对齐到min_len）
        covid_normalized_full = covid_normalized_full[:min_len]
        
        # 4. 构建 COVID 模型输入
        # 使用真实COVID患者数（RESP-Flu）作为训练目标
        covid_inputs = self._build_covid_inputs(covid_normalized_full, covid_patients_scaled, seq_len)
        self._validate_sequence_length(len(covid_inputs), seq_len, "COVID residual")
        
        # 更新 input_size 为 COVID 模型的特征数
        self.config["model"]["input_size"] = covid_inputs.shape[1]
        print(f"COVID Model Input Size: {self.config['model']['input_size']}")
        
        # 重建 COVID 模型 (使用独立配置)
        self.covid_model = self._build_model(stage="covid")

        train_data, val_data, test_data = self._split_dataset(covid_inputs)
        self._validate_split_lengths(
            ("covid_train", train_data),
            ("covid_val", val_data),
            ("covid_test", test_data),
            sequence_length=seq_len,
        )

        # 使用COVID模型的独立超参数
        covid_config = self.config["model"].get("covid_model", {})
        batch_size = covid_config.get("batch_size", self.config["training"]["batch_size"])
        covid_train_loader = create_data_loader(train_data, seq_len, batch_size, shuffle=True)
        covid_val_loader = create_data_loader(val_data, seq_len, batch_size, shuffle=False)
        covid_test_loader = create_data_loader(test_data, seq_len, batch_size, shuffle=False)

        print(f"  COVID model parameters: {sum(p.numel() for p in self.covid_model.parameters()):,}")

        # 使用COVID特定的训练参数
        epochs = covid_config.get("epochs", self.config["training"]["epochs"])
        learning_rate = covid_config.get("learning_rate", self.config["training"]["learning_rate"])
        patience = covid_config.get("patience", self.config["training"]["patience"])
        weight_decay = covid_config.get("weight_decay", self.config["training"].get("weight_decay", 0.01))
        
        metrics = train_and_evaluate_model(
            self.covid_model,
            covid_train_loader,
            covid_val_loader,
            covid_test_loader,
            model_name="COVID",
            epochs=epochs,
            learning_rate=learning_rate,
            patience=patience,
            weight_decay=weight_decay,
            save_path=os.path.join(self.config["paths"]["model_path"], "covid_model.pth"),
        )
        
        # 保存 residual scaler
        with open(os.path.join(self.config["paths"]["model_path"], "residual_scaler.pkl"), "wb") as f:
            import pickle
            pickle.dump(self.residual_scaler, f)

        print("COVID residual model training finished.")
        return metrics, covid_patients_raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _expand_with_lags(self, features: list, lags: list = [1, 2]) -> list:
        """扩展特征列表，包含滞后特征"""
        expanded = []
        for f in features:
            expanded.append(f)
            for lag in lags:
                expanded.append(f"{f}_lag{lag}")
        return expanded

    def _build_model(self, stage: str = "baseline") -> torch.nn.Module:
        """根据配置构建模型
        
        Args:
            stage: "baseline" 或 "covid"，用于区分不同阶段的模型配置
        """
        # 如果是COVID阶段且有独立配置，使用COVID配置
        if stage == "covid" and "covid_model" in self.config["model"]:
            covid_config = self.config["model"]["covid_model"]
            model_type = covid_config.get("type", self.config["model"].get("type", "mabg")).lower()
            hidden_size = covid_config.get("hidden_size", self.config["model"]["hidden_size"])
            num_layers = covid_config.get("num_layers", self.config["model"]["num_layers"])
            dropout = covid_config.get("dropout", self.config["model"]["dropout"])
            
            # 多趋势注意力模型的额外参数
            num_heads = covid_config.get("num_heads", 4)
            encoder_hidden = covid_config.get("encoder_hidden", 16)
            
            print(f"  使用COVID专属配置: type={model_type}, hidden={hidden_size}, layers={num_layers}, dropout={dropout}")
        else:
            model_type = self.config["model"].get("type", "mabg").lower()
            hidden_size = self.config["model"]["hidden_size"]
            num_layers = self.config["model"]["num_layers"]
            dropout = self.config["model"]["dropout"]
            num_heads = self.config["model"].get("num_heads", 4)
            encoder_hidden = self.config["model"].get("encoder_hidden", 16)
        
        input_size = self.config["model"]["input_size"]
        
        if model_type == "simple_gru":
            print(f"  使用SimpleGRU模型 (hidden={hidden_size}, dropout={dropout})")
            return SimpleGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout
            ).to(self.device)
        elif model_type == "simple_lstm":
            print(f"  使用SimpleLSTM模型 (hidden={hidden_size}, dropout={dropout})")
            return SimpleLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout
            ).to(self.device)
        elif model_type == "multi_trend_attention":
            print(f"  使用MultiTrendCrossAttention模型 (encoder_hidden={encoder_hidden}, fusion_hidden={hidden_size}, num_heads={num_heads}, dropout={dropout})")
            # 获取trends数量和每个trend的维度
            num_trends = covid_config.get("num_trends", 4)  # 默认4个trends
            trend_dim = covid_config.get("trend_dim", 4)    # 默认每个trend 4维（1值+3lags）
            return MultiTrendCrossAttention(
                num_trends=num_trends,
                trend_dim=trend_dim,
                encoder_hidden=encoder_hidden,
                num_heads=num_heads,
                fusion_hidden=hidden_size,
                dropout=dropout
            ).to(self.device)
        elif model_type == "simplified_attention":
            print(f"  使用SimplifiedMultiTrendAttention模型 (hidden={hidden_size}, num_heads={num_heads}, dropout={dropout})")
            num_trends = covid_config.get("num_trends", 4)
            trend_dim = covid_config.get("trend_dim", 4)
            return SimplifiedMultiTrendAttention(
                num_trends=num_trends,
                trend_dim=trend_dim,
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ).to(self.device)
        else:  # 'mabg' or default
            print(f"  使用EnhancedMABG模型 (hidden={hidden_size}, layers={num_layers})")
            return EnhancedMABG(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                use_multi_scale=False,
            ).to(self.device)

    @staticmethod
    def _split_dataset(
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_end = int(len(data) * train_ratio)
        val_end = int(len(data) * (train_ratio + val_ratio))
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        return train_data, val_data, test_data

    @staticmethod
    def _validate_split_lengths(
        *splits: Tuple[str, np.ndarray],
        sequence_length: int,
    ) -> None:
        for name, split in splits:
            if len(split) <= sequence_length:
                raise ValueError(
                    f"Split '{name}' has insufficient samples for sequence length {sequence_length}."
                )

    @staticmethod
    def _validate_sequence_length(length: int, sequence_length: int, label: str) -> None:
        if length <= sequence_length + 2:
            raise ValueError(
                f"Not enough samples in {label} dataset to build sequences of length {sequence_length}."
            )

    def _build_covid_inputs(
        self,
        features: np.ndarray,
        residuals: np.ndarray,
        sequence_length: int,
    ) -> np.ndarray:
        # features: [Value, Trend1, Trend2, ..., Seasonal...]
        # residuals: [r_seq, r_seq+1, ...]
        
        # 我们需要构建一个新的数据集，其中 Target (Col 0) 是 residuals
        # 其他特征保持不变 (Trends, Seasonal)
        
        # 1. 对齐 features 和 residuals
        # residuals 对应于 features[sequence_length:] 的时间点
        features_aligned = features[sequence_length:].copy()
        
        # 2. 将 features 的第一列 (Value) 替换为 residuals
        # 这样模型就会学习用过去的 residuals 和 trends 来预测当前的 residual
        features_aligned[:, 0] = residuals.reshape(-1)
        
        return features_aligned

    def _predict_sequence(self, model: torch.nn.Module, data: np.ndarray) -> np.ndarray:
        sequence_length = self.config["data"]["sequence_length"]
        if len(data) <= sequence_length:
            raise ValueError("Sequence too short for prediction.")

        model.eval()
        preds: list[float] = []
        with torch.no_grad():
            for idx in range(sequence_length, len(data)):
                window = torch.from_numpy(data[idx - sequence_length : idx]).float().unsqueeze(0).to(self.device)
                pred = model(window)
                preds.append(float(pred.detach().cpu().numpy().reshape(-1)[0]))
        return np.asarray(preds, dtype=np.float32)

    def _summarize_residuals(self, residuals: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
        }

    def _summarize_data_info(self) -> Dict[str, int | float | bool]:
        return {
            "pre_covid_samples": int(len(self.pre_covid_data) if self.pre_covid_data is not None else 0),
            "covid_samples": int(len(self.covid_data) if self.covid_data is not None else 0),
            "total_samples": int(len(self.full_data) if self.full_data is not None else 0),
            "feature_dim": self.config["model"]["input_size"],
            "include_seasonal": self.include_seasonal_features,
        }

    def _print_device_banner(self) -> None:
        separator = "=" * 60
        print(separator)
        print("Two-stage COVID-19 forecasting pipeline")
        print(separator)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                print(f"GPU {idx}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
        else:
            print("Running on CPU")
        print(f"Seasonal features enabled: {self.include_seasonal_features}")
        print(f"Model input dimension: {self.config['model']['input_size']}")
        print(separator)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "FluSurveillance_Custom_Download_Data.csv"

    pipeline = TwoStagePipeline(include_seasonal_features=True)
    results = pipeline.run_two_stage_training(
        csv_path=str(default_csv),
        covid_start_date="2020-01-01",
    )

    print("\nSummary of results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
