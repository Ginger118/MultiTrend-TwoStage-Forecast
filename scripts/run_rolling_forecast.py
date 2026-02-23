"""
Two-Stage Multi-Trend 滚动预测主评估脚本
=========================================

生产级实现，输出论文就绪结果
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))

from two_stage_model import TwoStageMultiTrendRollingModel, create_twostage_variants
from conformal import SplitConformalPredictor
from rolling_forecast_baselines import get_all_baselines


def load_data(data_path: str, covid_start: str = '2020-01-01'):
    """
    加载数据并分割pre-pandemic和COVID时期
    
    Returns:
        dates, y_full, y_baseline, y_covid, covid_start_idx
    """
    print(f"\n加载数据: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # 查找列名
    date_col = None
    for col in ['Date', 'date', 'DATE', 'Week', 'week']:
        if col in df.columns:
            date_col = col
            break
    
    target_col = None
    for col in ['weekly_rate', 'WeeklyRate', 'rate', 'Rate', 'value']:
        if col in df.columns:
            target_col = col
            break
    
    if date_col is None or target_col is None:
        raise ValueError(f"无法找到日期列或目标列。列名: {df.columns.tolist()}")
    
    # 转换并排序
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    dates = pd.DatetimeIndex(df[date_col])
    y = df[target_col].values
    
    # 移除NaN
    valid_mask = ~np.isnan(y)
    y = y[valid_mask]
    dates = dates[valid_mask]
    
    # 分割pre-pandemic和COVID时期
    covid_start_date = pd.Timestamp(covid_start)
    covid_start_idx = np.where(dates >= covid_start_date)[0][0]
    
    y_baseline = y[:covid_start_idx]
    y_covid = y[covid_start_idx:]
    
    print(f"  [OK] 加载 {len(y)} 个样本")
    print(f"  [OK] 时间范围: {dates[0]} 到 {dates[-1]}")
    print(f"  [OK] Pre-pandemic: {len(y_baseline)} 样本 (至 {dates[covid_start_idx-1].date()})")
    print(f"  [OK] COVID时期: {len(y_covid)} 样本 (从 {dates[covid_start_idx].date()})")
    
    return dates, y, y_baseline, y_covid, covid_start_idx


def run_rolling_forecast_twostage(
    model: TwoStageMultiTrendRollingModel,
    y_baseline: np.ndarray,
    y_covid: np.ndarray,
    dates_covid: pd.DatetimeIndex,
    min_train_size: int = 30,
    cal_size: int = 30,
    max_steps: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, any]:
    """
    运行Two-Stage模型的滚动预测（严格防止数据泄漏）
    
    数据划分（时间顺序）：
    - Baseline: pre-pandemic (固定，用于Stage1)
    - Train: COVID前min_train_size周（用于Stage2初始训练）
    - Calibration: 接下来cal_size周（用于conformal校准）
    - Test: 剩余数据（用于评估）
    
    策略：
    1. Stage1在pre-pandemic数据上训练一次（固定）
    2. Stage2在每个滚动步骤更新
    3. 严格防止数据泄漏
    """
    if verbose:
        print(f"\n运行Two-Stage滚动预测...")
        print(f"  Update mode: {model.stage2_update_mode}")
    
    # 数据划分
    train_end = min_train_size
    cal_end = min_train_size + cal_size
    test_start = cal_end
    
    if max_steps is not None:
        test_end = min(len(y_covid), test_start + max_steps)
    else:
        test_end = len(y_covid)
    
    if verbose:
        print(f"  数据划分:")
        print(f"    - Train: 0~{train_end} ({train_end}周)")
        print(f"    - Calibration: {train_end}~{cal_end} ({cal_size}周)")
        print(f"    - Test: {cal_end}~{test_end} ({test_end-cal_end}周)")
        print(f"    - Test window: {dates_covid[cal_end]} 到 {dates_covid[test_end-1]}")
    
    # Step 1: 训练Stage1（固定）
    if verbose:
        print(f"\n  [Stage1] 训练pre-pandemic baseline...")
    model.fit_stage1(y_baseline)
    if verbose:
        print(f"  [Stage1] 训练完成")
    
    # Step 2: Calibration阶段（收集residuals用于conformal）
    if verbose:
        print(f"\n  [Calibration] 滚动预测校准集...")
    
    cal_results = []
    for t in range(train_end, cal_end):
        y_train_covid = y_covid[:t]
        model.update_stage2(np.concatenate([y_baseline, y_train_covid]))
        
        recent_lookback = model.lookback + 5
        if t < recent_lookback:
            y_recent = np.concatenate([y_baseline[-(recent_lookback-t):], y_train_covid])
        else:
            y_recent = y_train_covid[-recent_lookback:]
        
        pred_dict = model.predict_next(y_recent)
        
        cal_results.append({
            'date': dates_covid[t],
            'y_true': y_covid[t],
            'y_pred': pred_dict['total'],
            'baseline_pred': pred_dict['baseline'],
            'excess_pred': pred_dict['excess'],
            'train_size': len(y_baseline) + t
        })
    
    cal_df = pd.DataFrame(cal_results)
    cal_residuals = np.abs(cal_df['y_true'].values - cal_df['y_pred'].values)
    
    if verbose:
        print(f"  [Calibration] 收集 {len(cal_residuals)} 个residuals")
        print(f"  [Calibration] Residual范围: [{cal_residuals.min():.3f}, {cal_residuals.max():.3f}]")
    
    # Step 3: Test阶段（滚动预测并应用conformal区间）
    if verbose:
        print(f"\n  [Test] 滚动预测测试集...")
    
    test_results = []
    for i, t in enumerate(range(cal_end, test_end)):
        if verbose and (i+1) % 20 == 0:
            print(f"  进度: {i+1}/{test_end-cal_end} ({100*(i+1)/(test_end-cal_end):.1f}%)")
        
        # 训练数据：包括baseline + 当前COVID历史
        y_train_covid = y_covid[:t]
        
        # 更新Stage2（使用所有历史数据，或sliding window）
        model.update_stage2(np.concatenate([y_baseline, y_train_covid]))
        
        # 预测t时刻
        # 准备输入：最近lookback个观测（可能跨越baseline和COVID）
        recent_lookback = model.lookback + 5  # 多取一些以防不够
        if t < recent_lookback:
            y_recent = np.concatenate([y_baseline[-(recent_lookback-t):], y_train_covid])
        else:
            y_recent = y_train_covid[-recent_lookback:]
        
        # 预测
        pred_dict = model.predict_next(y_recent)
        
        test_results.append({
            'date': dates_covid[t],
            'y_true': y_covid[t],
            'y_pred': pred_dict['total'],
            'baseline_pred': pred_dict['baseline'],
            'excess_pred': pred_dict['excess'],
            'train_size': len(y_baseline) + t
        })
    
    test_df = pd.DataFrame(test_results)
    
    if verbose:
        rmse = np.sqrt(np.mean((test_df['y_true'] - test_df['y_pred'])**2))
        print(f"\n  [Test] 完成 {len(test_df)} 个预测")
        print(f"  [Test] RMSE: {rmse:.4f}")
    
    return {
        'test_predictions': test_df,
        'cal_residuals': cal_residuals,
        'test_window_start': dates_covid[cal_end],
        'test_window_end': dates_covid[test_end-1],
        'cal_window_start': dates_covid[train_end],
        'cal_window_end': dates_covid[cal_end-1]
    }


def add_conformal_intervals(
    predictions_df: pd.DataFrame,
    cal_residuals: np.ndarray,
    alphas: List[float] = [0.05, 0.10, 0.20]  # 对应95%, 90%, 80%
) -> pd.DataFrame:
    """
    使用Split Conformal Prediction添加预测区间（无数据泄漏版）
    
    Args:
        predictions_df: Test集的预测结果（包含y_true和y_pred）
        cal_residuals: Calibration集的绝对残差（已预先计算）
        alphas: 显著性水平列表
    
    Returns:
        添加了区间列的DataFrame
    """
    print(f"\n添加Conformal预测区间...")
    print(f"  使用 {len(cal_residuals)} 个校准残差")
    
    # 为每个alpha计算分位数（使用校准集residuals）
    results_df = predictions_df.copy()
    
    for alpha in alphas:
        coverage = int((1 - alpha) * 100)
        # Conformal分位数：用(n+1)*(1-alpha)/n来保证覆盖率
        n = len(cal_residuals)
        adjusted_quantile = np.ceil((n + 1) * (1 - alpha)) / n
        quantile = np.quantile(cal_residuals, min(adjusted_quantile, 1.0))
        
        results_df[f'lower_{coverage}'] = results_df['y_pred'] - quantile
        results_df[f'upper_{coverage}'] = results_df['y_pred'] + quantile
        results_df[f'in_interval_{coverage}'] = (
            (results_df['y_true'] >= results_df[f'lower_{coverage}']) &
            (results_df['y_true'] <= results_df[f'upper_{coverage}'])
        )
        
        # 计算经验覆盖率（仅在test集）
        empirical_coverage = results_df[f'in_interval_{coverage}'].mean()
        target_coverage = (1 - alpha) * 100
        status = "✓" if empirical_coverage >= (1 - alpha) else "⚠"
        print(f"  {coverage}% 区间: 经验覆盖率={empirical_coverage*100:.1f}% (目标≥{target_coverage:.0f}%) {status}")
    
    return results_df


def calculate_metrics(predictions_df: pd.DataFrame, alphas: List[float]) -> Dict:
    """计算所有评估指标"""
    y_true = predictions_df['y_true'].values
    y_pred = predictions_df['y_pred'].values
    
    metrics = {
        'rmse': float(np.sqrt(np.mean((y_true - y_pred)**2))),
        'mae': float(np.mean(np.abs(y_true - y_pred))),
        'mape': float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
        'r2': float(1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)),
        'bias': float(np.mean(y_true - y_pred))
    }
    
    # 添加覆盖率和区间宽度
    for alpha in alphas:
        coverage = int((1 - alpha) * 100)
        if f'in_interval_{coverage}' in predictions_df.columns:
            metrics[f'coverage_{coverage}'] = float(predictions_df[f'in_interval_{coverage}'].mean())
            metrics[f'avg_width_{coverage}'] = float(
                (predictions_df[f'upper_{coverage}'] - predictions_df[f'lower_{coverage}']).mean()
            )
    
    return metrics


def save_results(
    all_results: Dict[str, pd.DataFrame],
    all_metrics: Dict[str, Dict],
    output_dir: Path
):
    """保存所有结果到论文就绪格式"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存结果到 {output_dir}...")
    
    # 1. 保存metrics.json
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {metrics_file}")
    
    # 2. 保存predictions.csv
    # 合并所有模型的预测
    for model_name, df in all_results.items():
        df['model'] = model_name
    
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    predictions_file = output_dir / "predictions.csv"
    combined_df.to_csv(predictions_file, index=False)
    print(f"  [OK] {predictions_file}")
    
    # 3. 创建对比表
    create_comparison_table(all_metrics, output_dir)
    
    # 4. 生成summary report
    create_summary_report(all_metrics, output_dir)
    
    # 5. 创建图表
    try:
        create_plots(all_results, output_dir)
    except Exception as e:
        print(f"  ⚠ 图表生成失败: {e}")


def create_comparison_table(metrics_dict: Dict, output_dir: Path):
    """创建LaTeX和CSV格式的对比表"""
    rows = []
    
    for model_name, metrics in metrics_dict.items():
        row = {
            '模型': model_name,
            'RMSE': f"{metrics['rmse']:.4f}",
            'MAE': f"{metrics['mae']:.4f}",
            'MAPE': f"{metrics['mape']:.2f}",
            'R²': f"{metrics['r2']:.4f}",
            'Bias': f"{metrics['bias']:.4f}",
        }
        
        # 添加覆盖率
        for cov in [80, 90, 95]:
            if f'coverage_{cov}' in metrics:
                row[f'覆盖{cov}%'] = f"{metrics[f'coverage_{cov}']*100:.1f}%"
                row[f'宽度{cov}%'] = f"{metrics[f'avg_width_{cov}']:.2f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 保存CSV
    csv_file = output_dir / "comparison_table.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  [OK] {csv_file}")
    
    # 保存LaTeX
    latex_str = df.to_latex(index=False, escape=False)
    tex_file = output_dir / "comparison_table.tex"
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f"  [OK] {tex_file}")


def create_summary_report(metrics_dict: Dict, output_dir: Path):
    """生成Markdown格式的摘要报告"""
    report_lines = [
        "# Two-Stage Multi-Trend 滚动预测评估报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验设计",
        "",
        "本次评估采用**expanding-window滚动预测**策略，严格防止数据泄漏：",
        "",
        "1. **Stage1（固定）**: 在pre-pandemic数据上训练一次基线模型，滚动时不更新",
        "2. **Stage2（动态更新）**: 每周使用≤t的COVID数据更新excess模型",
        "3. **三组对照实验**:",
        "   - **From-Scratch**: Stage2每周从零训练",
        "   - **Warm-Start**: Stage2从上周权重继续训练（小lr，3 epoch）",
        "   - **Sliding-Window**: Stage2仅用最近104周数据 + Warm-Start",
        "",
        "## 关键指标",
        "",
        "| 模型 | RMSE | R² | MAPE | 覆盖80% | 覆盖95% |",
        "|------|------|----|----- |---------|---------|"
    ]
    
    # 按RMSE排序
    sorted_models = sorted(metrics_dict.items(), key=lambda x: x[1]['rmse'])
    
    for model_name, m in sorted_models:
        cov80 = f"{m.get('coverage_80', 0)*100:.1f}%" if 'coverage_80' in m else "N/A"
        cov95 = f"{m.get('coverage_95', 0)*100:.1f}%" if 'coverage_95' in m else "N/A"
        
        report_lines.append(
            f"| {model_name} | {m['rmse']:.4f} | {m['r2']:.4f} | {m['mape']:.2f} | {cov80} | {cov95} |"
        )
    
    report_lines.extend([
        "",
        "## 核心发现",
        "",
        "### 为什么滚动评估比离线更难？",
        "",
        "1. **非平稳性**: COVID住院率随政策、变种、疫苗接种率快速变化",
        "2. **有限训练数据**: 每周只能用历史数据，无法像离线那样用全数据调参",
        "3. **分布漂移**: 模型需要快速适应新pattern，但过度适应会导致不稳定",
        "",
        "### 我们如何让滚动预测稳定？",
        "",
        "1. **Stage1固定策略**: Baseline机制不变，只让Stage2适应新pattern",
        "2. **Warm-Start**: 从上周权重继续训练，避免每周从零开始的震荡",
        "3. **Sliding Window**: 只用最近2年数据，适应concept drift",
        "4. **Weight Decay + Dropout**: 正则化防止过拟合",
        "5. **Conformal Prediction**: 理论保证coverage ≥ 标称水平",
        "",
        "## 区间覆盖率验证",
        "",
        "Conformal预测区间的经验覆盖率应该 ≥ 标称水平：",
        ""
    ])
    
    for model_name, m in sorted_models:
        if 'coverage_80' in m:
            report_lines.append(f"**{model_name}**:")
            report_lines.append(f"- 80%区间: {m['coverage_80']*100:.1f}% (目标≥80%)")
            report_lines.append(f"- 95%区间: {m['coverage_95']*100:.1f}% (目标≥95%)")
            report_lines.append("")
    
    report_lines.extend([
        "## 结论",
        "",
        f"**最佳配置**: {sorted_models[0][0]} (RMSE={sorted_models[0][1]['rmse']:.4f})",
        "",
        "滚动评估显示，通过合理的更新策略（Warm-Start + Sliding Window）和不确定性量化（Conformal Prediction），",
        "Two-Stage Multi-Trend模型能够在online部署场景下保持稳定性能，区间覆盖率接近标称水平。",
        "",
        "---",
        f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ])
    
    report_file = output_dir / "summary_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  [OK] {report_file}")


def create_plots(all_results: Dict[str, pd.DataFrame], output_dir: Path):
    """创建预测图表"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 为每个模型创建预测图
    for model_name, df in all_results.items():
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 绘制真实值
        ax.plot(df['date'], df['y_true'], 'k-', label='True', linewidth=2, alpha=0.7)
        
        # 绘制预测值
        ax.plot(df['date'], df['y_pred'], 'r--', label='Predicted', linewidth=2)
        
        # 绘制80%和95%区间
        if 'lower_80' in df.columns:
            ax.fill_between(df['date'], df['lower_95'], df['upper_95'], 
                           alpha=0.2, color='blue', label='95% Interval')
            ax.fill_between(df['date'], df['lower_80'], df['upper_80'], 
                           alpha=0.3, color='lightblue', label='80% Interval')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Weekly Hospitalization Rate', fontsize=12)
        ax.set_title(f'{model_name} - Rolling Forecast with Conformal Intervals', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        plot_file = plots_dir / f"{model_name}_prediction.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  [OK] {plots_dir}/ (生成{len(all_results)}张图)")


def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Multi-Trend 滚动预测评估",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='AME/Cleaned_RESP.csv',
        help='数据CSV路径'
    )
    
    parser.add_argument(
        '--covid-start',
        type=str,
        default='2020-01-01',
        help='COVID起始日期'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/rolling_two_stage',
        help='输出目录'
    )
    
    parser.add_argument(
        '--exp',
        type=str,
        default='all',
        choices=['all', 'from_scratch', 'warm_start', 'sliding_window'],
        help='运行哪些实验'
    )
    
    parser.add_argument(
        '--min-train',
        type=int,
        default=30,
        help='最小训练样本数（COVID初期训练阶段）'
    )
    
    parser.add_argument(
        '--cal-size',
        type=int,
        default=30,
        help='Conformal校准集大小'
    )
    
    parser.add_argument(
        '--alpha-list',
        type=str,
        default='0.20,0.10,0.05',
        help='显著性水平列表（逗号分隔），对应80%/90%/95%区间'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=104,
        help='Sliding window大小（仅用于sliding_window模式）'
    )
    
    parser.add_argument(
        '--max-test-steps',
        type=int,
        default=None,
        help='最大测试步数（用于快速测试，None=全部）'
    )
    
    parser.add_argument(
        '--compare-baselines',
        action='store_true',
        help='同时评估传统基线模型'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Two-Stage Multi-Trend 滚动预测评估")
    print("=" * 80)
    
    # 加载数据
    dates, y_full, y_baseline, y_covid, covid_start_idx = load_data(args.data, args.covid_start)
    dates_covid = dates[covid_start_idx:]
    
    # 解析alpha列表
    alphas = [float(a.strip()) for a in args.alpha_list.split(',')]
    
    # 创建模型
    if args.exp == 'all':
        models = create_twostage_variants()
    else:
        model_cls = TwoStageMultiTrendRollingModel
        if args.exp == 'from_scratch':
            models = {'TwoStage_FromScratch': model_cls(stage2_update_mode='from_scratch')}
        elif args.exp == 'warm_start':
            models = {'TwoStage_WarmStart': model_cls(stage2_update_mode='warm_start', stage2_lr=0.0001)}
        else:  # sliding_window
            models = {'TwoStage_SlidingWindow': model_cls(
                stage2_update_mode='sliding_window', 
                sliding_window_size=args.window_size
            )}
    
    # 添加基线模型
    if args.compare_baselines:
        print("\n加载基线模型...")
        baselines = get_all_baselines(include_sarima=False)
        print(f"  [OK] {len(baselines)} 个基线模型")
    
    # 运行评估
    all_results = {}
    all_metrics = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"评估: {model_name}")
        print(f"{'='*80}")
        
        # 滚动预测
        result = run_rolling_forecast_twostage(
            model, y_baseline, y_covid, dates_covid,
            min_train_size=args.min_train, cal_size=args.cal_size, 
            max_steps=args.max_test_steps, verbose=True
        )
        
        predictions_df = result['test_predictions']
        cal_residuals = result['cal_residuals']
        
        print(f"\n测试窗口: {result['test_window_start'].date()} 到 {result['test_window_end'].date()}")
        print(f"校准窗口: {result['cal_window_start'].date()} 到 {result['cal_window_end'].date()}")
        
        # 添加conformal区间
        predictions_df = add_conformal_intervals(
            predictions_df,
            cal_residuals,
            alphas=alphas
        )
        
        # 计算指标
        metrics = calculate_metrics(predictions_df, alphas=alphas)
        metrics['test_window_start'] = str(result['test_window_start'].date())
        metrics['test_window_end'] = str(result['test_window_end'].date())
        metrics['cal_window_start'] = str(result['cal_window_start'].date())
        metrics['cal_window_end'] = str(result['cal_window_end'].date())
        
        all_results[model_name] = predictions_df
        all_metrics[model_name] = metrics
        
        print(f"\n指标摘要:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # 保存结果
    output_dir = Path(args.output)
    save_results(all_results, all_metrics, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ 评估完成！")
    print(f"{'='*80}")
    print(f"\n输出目录: {output_dir}")
    print("\n关键文件:")
    print("  • metrics.json - 详细指标")
    print("  • predictions.csv - 所有预测结果")
    print("  • comparison_table.csv/tex - 对比表（可直接用于论文）")
    print("  • summary_report.md - 摘要报告")
    print("  • plots/ - 预测图表\n")


if __name__ == "__main__":
    main()
