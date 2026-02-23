
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional
import json

def calculate_wis(y_true, y_pred, lower, upper, alpha):
    """
    Weighted Interval Score (WIS) component for a single interval
    WIS = interval_width + (2/alpha) * (lower - y_true) * I(y_true < lower)
                         + (2/alpha) * (y_true - upper) * I(y_true > upper)
    """
    width = upper - lower
    under_coverage = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    over_coverage = (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return width + under_coverage + over_coverage

def load_and_preprocess_data(baselines_path: str, twostage_path: str):
    """Load and align datasets"""
    print("Loading data...")
    # Load baselines
    df_base = pd.read_csv(baselines_path)
    df_base['date'] = pd.to_datetime(df_base['date'])
    
    # Load twostage
    df_two = pd.read_csv(twostage_path)
    df_two['date'] = pd.to_datetime(df_two['date'])
    
    # Find common window
    min_date = max(df_base['date'].min(), df_two['date'].min())
    max_date = min(df_base['date'].max(), df_two['date'].max())
    
    print(f"Aligning to common window: {min_date.date()} to {max_date.date()}")
    
    # Filter
    df_base = df_base[(df_base['date'] >= min_date) & (df_base['date'] <= max_date)].copy()
    df_two = df_two[(df_two['date'] >= min_date) & (df_two['date'] <= max_date)].copy()
    
    print(f"Baselines samples: {len(df_base)} ({df_base['model'].nunique()} models)")
    print(f"TwoStage samples: {len(df_two)} ({df_two['model'].nunique()} models)")
    
    return df_base, df_two, min_date, max_date

def calculate_metrics_for_model(df, model_name, alphas=[0.2, 0.1, 0.05]):
    """Calculate all metrics for a single model"""
    df_m = df[df['model'] == model_name].copy()
    
    if len(df_m) == 0:
        return None
        
    y_true = df_m['y_true'].values
    y_pred = df_m['y_pred'].values
    dates = df_m['date'].values
    
    # Basic metrics
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    bias = np.mean(y_pred - y_true)
    
    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'n_forecasts': len(df_m)
    }
    
    # Coverage metrics
    total_wis = 0
    valid_intervals = 0
    
    columns = df_m.columns
    
    for alpha in alphas:
        coverage_target = int((1 - alpha) * 100)
        
        # Determine column names based on format
        # Twostage: lower_80, upper_80
        # Baselines: lower, upper (baseline usually has only one interval, mostly 80 or 95)
        # But wait, full_comparison_results.csv from baseline only has 'lower', 'upper'
        # We need to assume what alpha baseline run was. 
        # Usually baseline run has one alpha. If match, use it. If not, might need NaN
        
        lower_col = f'lower_{coverage_target}'
        upper_col = f'upper_{coverage_target}'
        
        # Check if columns exist (Twostage format)
        if lower_col in columns and upper_col in columns:
            lower = df_m[lower_col].values
            upper = df_m[upper_col].values
            
            # Check if majority of values are NaN (missing interval)
            if np.isnan(lower).all() or np.isnan(upper).all():
                metrics[f'coverage_{coverage_target}'] = np.nan
                metrics[f'avg_width_{coverage_target}'] = np.nan
                metrics[f'wis_{coverage_target}'] = np.nan
                continue

            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            width = np.mean(upper - lower)
            wis = np.mean(calculate_wis(y_true, y_pred, lower, upper, alpha))
            
            metrics[f'coverage_{coverage_target}'] = coverage
            metrics[f'avg_width_{coverage_target}'] = width
            metrics[f'wis_{coverage_target}'] = wis
            
            total_wis += wis
            valid_intervals += 1
            
        elif 'lower' in columns and 'upper' in columns:
             # Check if baseline interval width matches what we expect or just use what is there
             # For fair comparison, we only compare if we know the alpha.
             # Assuming baseline results are for 80% if not specified?
             # Let's look at full_comparison_metrics.json to check baseline alpha?
             # Or just skip if columns not found.
             pass

    # Normalize WIS
    if valid_intervals > 0:
        metrics['mean_wis'] = total_wis / valid_intervals
    else:
        metrics['mean_wis'] = np.nan
        
    return metrics, df_m

def standardize_baseline_format(df_base):
    """
    Baseline CSV 'full_comparison_results.csv' usually has columns:
    date, train_size, y_true, y_pred, error, ..., lower, upper, in_interval, interval_width, model
    
    This usually corresponds to ONE alpha (often 0.2 -> 80%).
    We need to map 'lower'/'upper' to 'lower_80'/'upper_80' if that's the case.
    
    NOTE: Currently the baseline script generates only one interval.
    To support multi-alpha for baselines, the baseline script needs update OR we duplicate rows.
    
    For now, let's assume baseline is 80% (common default).
    If we want fair comparison on 90/95, baseline will be missing those unless re-run.
    
    However, the request asks to "evaluate baselines with rolling" to output multi-intervals.
    Since we are only processing existing CSVs here, we can only work with what we have.
    
    Wait, if the user ran the previous step `rolling_forecast_full_comparison.py`, it likely generated one interval.
    If the user WANTS 80/90/95 for baselines, they need to run a script that generates them.
    
    But this script is `make_fair_window_comparison.py` which consumes EXISTING CSVs.
    If existing baseline CSV only has 80%, we can only compare 80%.
    
    Let's check if we can make do.
    """
    
    # Rename columns to match 80% if they exist and generic
    if 'lower' in df_base.columns and 'lower_80' not in df_base.columns:
        # Check width/coverage to guess? Or just assume 80%? 
        # Stdout from previous run said: "Calibration quantile (1-α=80%): ..."
        # So it is 80%.
        df_base = df_base.rename(columns={
            'lower': 'lower_80',
            'upper': 'upper_80',
            'in_interval': 'in_interval_80',
            'interval_width': 'avg_width_80' # column is interval_width
        })
    
    return df_base

def generate_plots(df_fair, output_dir):
    """Generate comparison plots"""
    
    # 1. Coverage vs Width Scatter (for 80%, since that's likely common)
    plt.figure(figsize=(10, 6))
    
    # Calculate coverage/width per model
    stats = []
    models = df_fair['model'].unique()
    
    for m in models:
        df_m = df_fair[df_fair['model'] == m]
        if 'lower_80' in df_m.columns:
            cov = np.mean((df_m['y_true'] >= df_m['lower_80']) & (df_m['y_true'] <= df_m['upper_80']))
            wid = np.mean(df_m['upper_80'] - df_m['lower_80'])
            stats.append({'model': m, 'coverage': cov, 'width': wid})
    
    if stats:
        stats_df = pd.DataFrame(stats)
        sns.scatterplot(data=stats_df, x='width', y='coverage', hue='model', s=100)
        plt.axhline(0.8, color='red', linestyle='--', label='Target 80%')
        plt.title('Coverage vs Interval Width (80% Interval)')
        plt.xlabel('Average Interval Width (lower is better)')
        plt.ylabel('Empirical Coverage (target=0.8)')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'coverage_vs_width.png')
        plt.close()

    # 2. Time Series Predictions (Best TwoStage vs Best Ridge)
    # Find best models based on RMSE on fair window
    rmse_scores = {}
    for m in models:
        df_m = df_fair[df_fair['model'] == m]
        rmse = np.sqrt(np.mean((df_m['y_true'] - df_m['y_pred'])**2))
        rmse_scores[m] = rmse
        
    best_twostage = min([m for m in models if 'TwoStage' in m], key=lambda x: rmse_scores.get(x, 999), default=None)
    best_baseline = min([m for m in models if 'TwoStage' not in m and 'Ridge' in m], key=lambda x: rmse_scores.get(x, 999), default=None)
    
    if best_twostage and best_baseline:
        plt.figure(figsize=(15, 7))
        
        # Plot last 100 points
        df_ts = df_fair[df_fair['model'] == best_twostage].tail(100)
        df_bl = df_fair[df_fair['model'] == best_baseline].tail(100)
        
        plt.plot(df_ts['date'], df_ts['y_true'], 'k-', label='Actual', linewidth=2)
        plt.plot(df_ts['date'], df_ts['y_pred'], label=f'{best_twostage} (RMSE={rmse_scores[best_twostage]:.2f})')
        plt.plot(df_bl['date'], df_bl['y_pred'], '--', label=f'{best_baseline} (RMSE={rmse_scores[best_baseline]:.2f})')
        
        # Add 95% interval for TwoStage if available
        if 'lower_95' in df_ts.columns:
            plt.fill_between(df_ts['date'], df_ts['lower_95'], df_ts['upper_95'], alpha=0.2, label=f'{best_twostage} 95% CI')
            
        plt.title('Forecast Comparison: Two-Stage vs Best Baseline (Fair Window)')
        plt.xlabel('Date')
        plt.ylabel('Weekly Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_comparison_fair.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines-results', type=str, required=True, help='Baseline results CSV')
    parser.add_argument('--twostage-results', type=str, required=True, help='TwoStage results CSV')
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    df_base, df_two, min_date, max_date = load_and_preprocess_data(
        args.baselines_results, args.twostage_results
    )
    
    print(f"\nFair Comparison Window:")
    print(f"Start: {min_date}")
    print(f"End:   {max_date}")
    
    # Standardize baseline columns
    df_base = standardize_baseline_format(df_base)
    
    # Combine
    # Select common columns
    common_cols = ['date', 'model', 'y_true', 'y_pred']
    # Add interval columns that might be present
    for alpha in [80, 90, 95]:
        common_cols.extend([f'lower_{alpha}', f'upper_{alpha}'])
    
    # Keep only columns that exist
    df_fair_list = []
    
    for df in [df_base, df_two]:
        avail_cols = [c for c in common_cols if c in df.columns]
        df_fair_list.append(df[avail_cols])
        
    df_fair = pd.concat(df_fair_list, ignore_index=True)
    df_fair = df_fair.sort_values(['model', 'date'])
    
    # Calculate metrics
    metrics_list = []
    unique_models = df_fair['model'].unique()
    
    print("\nCalculating metrics on fair window...")
    for m in unique_models:
        met, _ = calculate_metrics_for_model(df_fair, m)
        if met:
            metrics_list.append(met)
            
    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df[['model', 'rmse', 'r2']].to_string())
    
    # Save outputs
    # 1. CSV Table
    metrics_df.to_csv(out_dir / 'comparison_table_fair.csv', index=False)
    
    # 2. LaTeX Table
    latex_cols = ['model', 'rmse', 'mae', 'r2']
    # Add coverage columns if they exist
    headers = ['Model', 'RMSE', 'MAE', 'R2']
    
    for alpha in [80, 95]: # Typical for paper
        if f'coverage_{alpha}' in metrics_df.columns:
            latex_cols.extend([f'coverage_{alpha}', f'avg_width_{alpha}'])
            headers.extend([f'Cov {alpha}%', f'Width {alpha}%'])

    latex_df = metrics_df[latex_cols].copy()
    
    # Format percision
    for col in latex_df.columns:
        if 'coverage' in col:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x*100:.1f}\%" if pd.notnull(x) else "-")
        elif 'rmse' in col or 'mae' in col or 'width' in col:
             latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
        elif 'r2' in col:
             latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
            
    latex_df.columns = headers
    latex_df.to_latex(out_dir / 'comparison_table_fair.tex', index=False, escape=False)
    
    # 3. Full JSON
    metrics_dict = metrics_df.set_index('model').to_dict('index')
    # Add meta
    for m in metrics_dict:
        metrics_dict[m]['test_window_start'] = str(min_date)
        metrics_dict[m]['test_window_end'] = str(max_date)
        
    with open(out_dir / 'full_comparison_metrics_fair.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
        
    # 4. Plots
    generate_plots(df_fair, out_dir)
    
    # 5. Full Prediction CSV (Fair)
    df_fair.to_csv(out_dir / 'full_comparison_results_fair.csv', index=False)
    
    print(f"\n✅ Fair comparison complete. Results in {out_dir}")

if __name__ == "__main__":
    main()
