"""
结果可视化和疫情曲线预测模块
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

from seir_model import SEIRPredictor, visualize_seir_results
from mabg_model import MABGModel

class COVIDVisualizer:
    """COVID-19疫情可视化器"""
    
    def __init__(self, save_path='plots/'):
        """
        初始化可视化器
        
        Args:
            save_path: 图片保存路径
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8')
        
        # 设置颜色主题
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'danger': '#C73E1D',
            'warning': '#FFB627',
            'info': '#7209B7'
        }
    
    def plot_baidu_index_trends(self, baidu_data, keywords, save_name='baidu_trends.png'):
        """
        绘制百度指数趋势
        
        Args:
            baidu_data: 百度指数数据
            keywords: 关键词列表
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, keyword in enumerate(keywords):
            if i < len(axes):
                col_name = f'{keyword}_指数'
                if col_name in baidu_data.columns:
                    axes[i].plot(baidu_data.index, baidu_data[col_name], 
                               color=self.colors['primary'], linewidth=2)
                    axes[i].set_title(f'{keyword} 搜索指数趋势', fontsize=14, fontweight='bold')
                    axes[i].set_xlabel('日期')
                    axes[i].set_ylabel('搜索指数')
                    axes[i].grid(True, alpha=0.3)
                    
                    # 添加趋势线
                    z = np.polyfit(range(len(baidu_data)), baidu_data[col_name], 1)
                    p = np.poly1d(z)
                    axes[i].plot(baidu_data.index, p(range(len(baidu_data))), 
                               '--', color=self.colors['danger'], alpha=0.7)
        
        # 移除多余的子图
        for i in range(len(keywords), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('百度指数关键词搜索趋势分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ili_trends(self, ili_data, save_name='ili_trends.png'):
        """
        绘制ILI趋势
        
        Args:
            ili_data: ILI数据
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ILI病例数趋势
        axes[0].plot(ili_data['日期'], ili_data['ILI病例数'], 
                    color=self.colors['danger'], linewidth=2, marker='o', markersize=3)
        axes[0].set_title('ILI病例数变化趋势', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('日期')
        axes[0].set_ylabel('ILI病例数')
        axes[0].grid(True, alpha=0.3)
        
        # 添加移动平均线
        window_size = min(30, len(ili_data) // 4)
        if window_size > 1:
            moving_avg = ili_data['ILI病例数'].rolling(window=window_size).mean()
            axes[0].plot(ili_data['日期'], moving_avg, 
                        color=self.colors['warning'], linewidth=2, alpha=0.8, 
                        label=f'{window_size}天移动平均')
            axes[0].legend()
        
        # ILI病例数分布
        axes[1].hist(ili_data['ILI病例数'], bins=30, alpha=0.7, 
                    color=self.colors['primary'], edgecolor='black')
        axes[1].set_title('ILI病例数分布', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('ILI病例数')
        axes[1].set_ylabel('频次')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, combined_data, save_name='correlation_heatmap.png'):
        """
        绘制相关性热力图
        
        Args:
            combined_data: 合并后的数据
            save_name: 保存文件名
        """
        # 计算相关性矩阵
        correlation_matrix = combined_data.corr()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": .8}, fmt='.3f')
        
        plt.title('百度指数与ILI数据相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_epidemic_prediction(self, prediction_results, save_name='epidemic_prediction.png'):
        """
        绘制疫情预测结果
        
        Args:
            prediction_results: 预测结果
            save_name: 保存文件名
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('SEIR状态变化', 'ILI预测趋势', '总病例数', '感染率变化', 
                          '每日新增病例', '疫情发展阶段'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        time = prediction_results['time']
        
        # SEIR状态变化
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['S'], name='易感人群', 
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['E'], name='暴露人群', 
                      line=dict(color='orange')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['I'], name='感染人群', 
                      line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['R'], name='康复人群', 
                      line=dict(color='green')),
            row=1, col=1
        )
        
        # ILI预测趋势
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['ili_predictions'], 
                      name='ILI预测', line=dict(color='purple')),
            row=1, col=2
        )
        
        # 总病例数
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['total_cases'], 
                      name='总病例数', line=dict(color='darkred')),
            row=2, col=1
        )
        
        # 感染率变化
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['infection_rate'] * 100, 
                      name='感染率(%)', line=dict(color='darkgreen')),
            row=2, col=2
        )
        
        # 每日新增病例
        daily_new = np.diff(prediction_results['total_cases'])
        daily_new = np.concatenate([[0], daily_new])  # 第一天新增为0
        fig.add_trace(
            go.Scatter(x=time, y=daily_new, name='每日新增', 
                      line=dict(color='red'), fill='tonexty'),
            row=3, col=1
        )
        
        # 疫情发展阶段（基于感染率）
        infection_rate = prediction_results['infection_rate']
        stages = []
        for rate in infection_rate:
            if rate < 0.05:
                stages.append('初始阶段')
            elif rate < 0.2:
                stages.append('上升阶段')
            elif rate < 0.5:
                stages.append('高峰期')
            elif rate < 0.8:
                stages.append('下降阶段')
            else:
                stages.append('尾声阶段')
        
        fig.add_trace(
            go.Scatter(x=time, y=infection_rate * 100, mode='markers',
                      marker=dict(color=stages, colorscale='RdYlBu_r'),
                      name='疫情阶段'),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='COVID-19疫情预测综合分析',
            height=1200,
            showlegend=True
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="时间 (天)", row=3, col=1)
        fig.update_xaxes(title_text="时间 (天)", row=3, col=2)
        fig.update_yaxes(title_text="人数", row=1, col=1)
        fig.update_yaxes(title_text="ILI病例数", row=1, col=2)
        fig.update_yaxes(title_text="总病例数", row=2, col=1)
        fig.update_yaxes(title_text="感染率 (%)", row=2, col=2)
        fig.update_yaxes(title_text="新增病例", row=3, col=1)
        fig.update_yaxes(title_text="感染率 (%)", row=3, col=2)
        
        # 保存图片
        fig.write_html(os.path.join(self.save_path, save_name.replace('.png', '.html')))
        fig.write_image(os.path.join(self.save_path, save_name))
        fig.show()
    
    def create_interactive_dashboard(self, baidu_data, ili_data, prediction_results):
        """
        创建交互式仪表板
        
        Args:
            baidu_data: 百度指数数据
            ili_data: ILI数据
            prediction_results: 预测结果
        """
        # 创建子图
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=('百度指数趋势', 'ILI病例数', 'SEIR状态', '疫情预测',
                          '相关性分析', '感染率变化', '新增病例', '风险评估'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 百度指数趋势（选择前3个关键词）
        keywords = ['发热', '咳嗽', '嗓子痛']
        for i, keyword in enumerate(keywords[:3]):
            col_name = f'{keyword}_指数'
            if col_name in baidu_data.columns:
                fig.add_trace(
                    go.Scatter(x=baidu_data.index, y=baidu_data[col_name], 
                              name=f'{keyword}指数', mode='lines'),
                    row=1, col=1
                )
        
        # ILI病例数
        fig.add_trace(
            go.Scatter(x=ili_data['日期'], y=ili_data['ILI病例数'], 
                      name='ILI病例数', line=dict(color='red')),
            row=1, col=2
        )
        
        # SEIR状态
        time = prediction_results['time']
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['I'], name='感染人群', 
                      line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['R'], name='康复人群', 
                      line=dict(color='green')),
            row=2, col=1
        )
        
        # 疫情预测
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['infection_rate'] * 100, 
                      name='预测感染率', line=dict(color='blue')),
            row=2, col=2
        )
        
        # 相关性分析（简化版）
        # 这里可以添加更复杂的相关性分析
        
        # 感染率变化
        fig.add_trace(
            go.Scatter(x=time, y=prediction_results['infection_rate'] * 100, 
                      name='感染率', line=dict(color='darkgreen')),
            row=3, col=2
        )
        
        # 新增病例
        daily_new = np.diff(prediction_results['total_cases'])
        daily_new = np.concatenate([[0], daily_new])
        fig.add_trace(
            go.Scatter(x=time, y=daily_new, name='每日新增', 
                      line=dict(color='red')),
            row=4, col=1
        )
        
        # 风险评估
        risk_levels = []
        for rate in prediction_results['infection_rate']:
            if rate < 0.05:
                risk_levels.append(1)  # 低风险
            elif rate < 0.2:
                risk_levels.append(2)  # 中风险
            else:
                risk_levels.append(3)  # 高风险
        
        fig.add_trace(
            go.Scatter(x=time, y=risk_levels, name='风险等级', 
                      mode='markers', marker=dict(color=risk_levels, 
                                                 colorscale='RdYlBu_r')),
            row=4, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='COVID-19疫情监测与预测仪表板',
            height=1600,
            showlegend=True
        )
        
        # 保存交互式仪表板
        fig.write_html(os.path.join(self.save_path, 'covid_dashboard.html'))
        print(f"交互式仪表板已保存到: {os.path.join(self.save_path, 'covid_dashboard.html')}")
        
        return fig
    
    def plot_model_performance_comparison(self, model_results, save_name='model_comparison.png'):
        """
        绘制模型性能比较
        
        Args:
            model_results: 模型结果字典
            save_name: 保存文件名
        """
        models = list(model_results.keys())
        metrics = ['R2', 'MAE', 'MSE', 'RMSE', 'EVS']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in models]
            bars = axes[i].bar(models, values, alpha=0.7, 
                             color=[self.colors['primary'], self.colors['secondary'], 
                                   self.colors['success'], self.colors['warning']])
            axes[i].set_title(f'{metric} 比较', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
            
            axes[i].grid(True, alpha=0.3)
        
        # 移除多余的子图
        axes[5].axis('off')
        
        plt.suptitle('模型性能比较', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, save_name), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, model_results, prediction_results, save_name='covid_report.html'):
        """
        生成HTML报告
        
        Args:
            model_results: 模型结果
            prediction_results: 预测结果
            save_name: 报告文件名
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>COVID-19疫情预测报告</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2E86AB; }}
                h2 {{ color: #A23B72; }}
                .metric {{ background-color: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
                .prediction {{ background-color: #e8f4f8; padding: 15px; margin: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>COVID-19疫情预测分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>模型性能指标</h2>
        """
        
        for model_name, results in model_results.items():
            html_content += f"""
            <div class="metric">
                <h3>{model_name} 模型</h3>
                <p>R²: {results.get('R2', 0):.4f}</p>
                <p>MAE: {results.get('MAE', 0):.4f}</p>
                <p>MSE: {results.get('MSE', 0):.4f}</p>
                <p>RMSE: {results.get('RMSE', 0):.4f}</p>
                <p>EVS: {results.get('EVS', 0):.4f}</p>
            </div>
            """
        
        # 添加预测结果摘要
        max_infection_rate = np.max(prediction_results['infection_rate']) * 100
        peak_day = np.argmax(prediction_results['infection_rate'])
        
        html_content += f"""
            <h2>疫情预测摘要</h2>
            <div class="prediction">
                <h3>关键预测指标</h3>
                <p>最大感染率: {max_infection_rate:.2f}%</p>
                <p>感染率峰值预计在第 {peak_day} 天</p>
                <p>最终总病例数: {prediction_results['total_cases'][-1]:.0f}</p>
                <p>预测时间跨度: {len(prediction_results['time'])} 天</p>
            </div>
            
            <h2>风险等级评估</h2>
            <div class="prediction">
                <p>基于预测的感染率，当前风险评估如下：</p>
        """
        
        final_infection_rate = prediction_results['infection_rate'][-1] * 100
        if final_infection_rate < 5:
            risk_level = "低风险"
        elif final_infection_rate < 20:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
        
        html_content += f"""
                <p><strong>最终风险等级: {risk_level}</strong></p>
                <p>最终感染率: {final_infection_rate:.2f}%</p>
            </div>
            
            <p><em>注意：本报告基于历史数据和模型预测生成，仅供参考。实际疫情发展可能受到多种因素影响。</em></p>
        </body>
        </html>
        """
        
        with open(os.path.join(self.save_path, save_name), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已保存到: {os.path.join(self.save_path, save_name)}")

def test_visualization():
    """测试可视化功能"""
    print("测试可视化模块...")
    
    # 创建可视化器
    visualizer = COVIDVisualizer()
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=365, freq='D')
    
    # 模拟百度指数数据
    baidu_data = pd.DataFrame({
        '发热_指数': np.random.normal(100, 20, 365),
        '咳嗽_指数': np.random.normal(80, 15, 365),
        '嗓子痛_指数': np.random.normal(60, 10, 365),
        '头痛_指数': np.random.normal(70, 12, 365),
        '乏力_指数': np.random.normal(50, 8, 365),
        '呼吸困难_指数': np.random.normal(40, 6, 365)
    }, index=dates)
    
    # 模拟ILI数据
    ili_data = pd.DataFrame({
        '日期': dates,
        'ILI病例数': np.random.poisson(1000, 365)
    })
    
    # 模拟预测结果
    prediction_results = {
        'time': np.arange(365),
        'S': np.linspace(999000, 800000, 365),
        'E': np.random.normal(1000, 200, 365),
        'I': np.random.normal(2000, 500, 365),
        'R': np.linspace(0, 190000, 365),
        'ili_predictions': np.random.normal(1000, 100, 365),
        'total_cases': np.linspace(3000, 200000, 365),
        'infection_rate': np.linspace(0.003, 0.2, 365)
    }
    
    # 测试各种可视化功能
    print("测试百度指数趋势图...")
    visualizer.plot_baidu_index_trends(baidu_data, ['发热', '咳嗽', '嗓子痛', '头痛', '乏力', '呼吸困难'])
    
    print("测试ILI趋势图...")
    visualizer.plot_ili_trends(ili_data)
    
    print("测试疫情预测图...")
    visualizer.plot_epidemic_prediction(prediction_results)
    
    print("测试交互式仪表板...")
    visualizer.create_interactive_dashboard(baidu_data, ili_data, prediction_results)
    
    print("可视化模块测试完成！")

if __name__ == "__main__":
    test_visualization()



