"""
模型训练和评估模块
包含训练循环、评估指标、可视化等功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model, device=None):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 计算设备 (cuda/cpu)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"模型训练器初始化完成，使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU设备: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc="训练", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 处理返回tuple的模型（如注意力模型）
            if isinstance(output, tuple):
                output = output[0]  # 取预测值，忽略注意力权重
            
            loss = criterion(output.squeeze(), target.squeeze())
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/total_samples:.4f}'
            })
        
        return total_loss / total_samples
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="验证", leave=False)
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                # 处理返回tuple的模型（如注意力模型）
                if isinstance(output, tuple):
                    output = output[0]  # 取预测值，忽略注意力权重
                
                loss = criterion(output.squeeze(), target.squeeze())
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # 处理不同维度的输出
                output_np = output.squeeze().cpu().numpy()
                target_np = target.squeeze().cpu().numpy()
                
                if output_np.ndim == 0:  # 单个标量值
                    predictions.append(output_np.item())
                    targets.append(target_np.item())
                else:  # 数组
                    predictions.extend(output_np.tolist())
                    targets.extend(target_np.tolist())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples
        metrics = self.calculate_metrics(np.array(predictions), np.array(targets))
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R²计算
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE计算
        mape = np.mean(np.abs((targets - predictions) / np.maximum(np.abs(targets), 1e-8))) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              weight_decay=1e-5, patience=10, save_path=None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            save_path: 模型保存路径
        """
        print(f"开始训练模型，共 {epochs} 个epoch...")
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 主训练循环
        epoch_pbar = tqdm(range(epochs), desc="Epochs", position=0)
        
        for epoch in epoch_pbar:
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 保存历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # 更新进度条
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val R2': f'{val_metrics["R2"]:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }, save_path)
                    print(f"\n最佳模型已保存到: {save_path}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停触发，在第 {epoch+1} 个epoch停止训练")
                break
            
            # 每10个epoch打印详细信息
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"训练损失: {train_loss:.4f}")
                print(f"验证损失: {val_loss:.4f}")
                print(f"验证R2: {val_metrics['R2']:.4f}")
                print(f"验证RMSE: {val_metrics['RMSE']:.4f}")
                print(f"验证MAPE: {val_metrics['MAPE']:.2f}%")
        
        # 返回最终结果
        final_metrics = self.val_metrics[-1] if self.val_metrics else {}
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_loss': best_val_loss,
            'final_metrics': final_metrics,
            'total_epochs': len(self.train_losses)
        }

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="评估"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # 处理返回tuple的模型（如注意力模型）
                if isinstance(output, tuple):
                    output = output[0]  # 取预测值，忽略注意力权重
                
                # 处理不同维度的情况
                output_np = output.squeeze().cpu().numpy()
                target_np = target.squeeze().cpu().numpy()
                
                # 确保是数组格式
                if output_np.ndim == 0:  # 单个值
                    predictions.append(output_np.item())
                    targets.append(target_np.item())
                else:  # 多个值
                    predictions.extend(output_np.tolist())
                    targets.extend(target_np.tolist())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # 计算指标
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        mape = np.mean(np.abs((targets - predictions) / np.maximum(np.abs(targets), 1e-8))) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'predictions': predictions,
            'targets': targets
        }

class ModelVisualizer:
    """模型可视化器"""
    
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_history(self, train_losses, val_losses, val_metrics, save_path=None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失', alpha=0.8)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失', alpha=0.8)
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # R²曲线
        r2_scores = [m['R2'] for m in val_metrics]
        axes[0, 1].plot(epochs, r2_scores, 'g-', label='验证R2', linewidth=2)
        axes[0, 1].set_title('验证R2分数')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R2')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE曲线
        rmse_scores = [m['RMSE'] for m in val_metrics]
        axes[1, 0].plot(epochs, rmse_scores, 'orange', label='验证RMSE', linewidth=2)
        axes[1, 0].set_title('验证RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAPE曲线
        mape_scores = [m['MAPE'] for m in val_metrics]
        axes[1, 1].plot(epochs, mape_scores, 'purple', label='验证MAPE', linewidth=2)
        axes[1, 1].set_title('验证MAPE (%)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        else:
            plt.show()
        
        return fig
    
    def plot_predictions(self, predictions, targets, save_path=None):
        """绘制预测结果对比"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 时间序列对比
        axes[0].plot(targets, 'b-', label='真实值', alpha=0.7, linewidth=2)
        axes[0].plot(predictions, 'r--', label='预测值', alpha=0.7, linewidth=2)
        axes[0].set_title('预测结果对比')
        axes[0].set_xlabel('样本')
        axes[0].set_ylabel('值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 散点图
        axes[1].scatter(targets, predictions, alpha=0.6, s=20)
        axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', alpha=0.8)
        axes[1].set_title('预测值 vs 真实值')
        axes[1].set_xlabel('真实值')
        axes[1].set_ylabel('预测值')
        axes[1].grid(True, alpha=0.3)
        
        # 计算并显示R²
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        axes[1].text(0.05, 0.95, f'R2 = {r2:.4f}', transform=axes[1].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图已保存到: {save_path}")
        else:
            plt.show()
        
        return fig

def train_and_evaluate_model(model, train_loader, val_loader, test_loader, 
                           model_name='Model', epochs=100, learning_rate=0.001, 
                           patience=15, weight_decay=0.01, save_path=None):
    """
    训练和评估模型的主函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        model_name: 模型名称
        epochs: 训练轮数
        learning_rate: 学习率
        patience: 早停耐心值
        weight_decay: L2正则化系数
        save_path: 模型保存路径
        
    Returns:
        results: 训练和评估结果
    """
    print(f"开始训练 {model_name}...")
    
    # 创建训练器
    trainer = ModelTrainer(model)
    
    # 训练模型
    training_results = trainer.train(
        train_loader, val_loader, 
        epochs=epochs, 
        lr=learning_rate,
        patience=patience,
        weight_decay=weight_decay,
        save_path=save_path
    )
    
    # 加载最佳模型
    if save_path and os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=trainer.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型 (验证损失: {checkpoint['val_loss']:.4f})")
    
    # 评估模型
    evaluator = ModelEvaluator(model, trainer.device)
    test_results = evaluator.evaluate(test_loader)
    
    # 可视化结果
    visualizer = ModelVisualizer()
    
    # 绘制训练历史
    history_fig = visualizer.plot_training_history(
        training_results['train_losses'],
        training_results['val_losses'],
        training_results['val_metrics'],
        save_path=os.path.join(os.path.dirname(save_path) if save_path else '.', f'{model_name}_training_history.png')
    )
    
    # 绘制预测结果
    pred_fig = visualizer.plot_predictions(
        test_results['predictions'],
        test_results['targets'],
        save_path=os.path.join(os.path.dirname(save_path) if save_path else '.', f'{model_name}_predictions.png')
    )
    
    # 合并结果
    results = {
        **training_results,
        'test_metrics': {k: v for k, v in test_results.items() if k not in ['predictions', 'targets']},
        'test_predictions': test_results['predictions'],
        'test_targets': test_results['targets']
    }
    
    print(f"\n{model_name} 训练完成！")
    print(f"最终测试指标:")
    print(f"  R2: {test_results['R2']:.4f}")
    print(f"  RMSE: {test_results['RMSE']:.4f}")
    print(f"  MAE: {test_results['MAE']:.4f}")
    print(f"  MAPE: {test_results['MAPE']:.2f}%")
    
    return results

if __name__ == "__main__":
    # 测试模块
    print("模型训练模块测试完成！")

