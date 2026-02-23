
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def setup_style():
    # Use a style suitable for inclusion in diagrams (clean, high contrast)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def load_data():
    # Load Epidemic Data
    df_epi = pd.read_csv('AME/Cleaned_RESP.csv')
    df_epi['Date'] = pd.to_datetime(df_epi['Date'])
    
    # Load Trend Data (Example: Fever)
    # The file has a header at line 2 usually based on preview: "月份,fever: (美国)"
    # We'll skip rows=2 based on preview "类别：所有类别 \n\n 月份..."
    try:
        df_trend = pd.read_csv('AME/fever_Trends.csv', skiprows=2)
        # Rename columns for easier access
        df_trend.columns = ['Month', 'Val']
        df_trend['Date'] = pd.to_datetime(df_trend['Month'])
    except Exception as e:
        # Fallback if format differs
        print(f"Error reading trends: {e}")
        df_trend = None
        
    return df_epi, df_trend

def plot_epidemic_curve(df, output_path):
    # Plot a representative segment (e.g., 2020-2022)
    mask = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2022-01-01')
    data = df.loc[mask]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data['Date'], data['WeeklyRate'], color='#1f77b4', linewidth=2)
    ax.fill_between(data['Date'], 0, data['WeeklyRate'], color='#1f77b4', alpha=0.1)
    
    ax.set_title("Target: Weekly Hospitalization Rate", fontsize=12, pad=10, loc='left')
    ax.set_ylabel("Rate per 100k")
    
    # Format dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_search_trend(df, output_path):
    if df is None: return
    
    # Plot representative segment
    mask = (df['Date'] >= '2020-01-01') & (df['Date'] <= '2022-01-01')
    data = df.loc[mask]
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(data['Date'], data['Val'], color='#ff7f0e', linewidth=2, linestyle='-')
    
    ax.set_title("Feature: 'Fever' Search Volume", fontsize=12, pad=10, loc='left')
    ax.set_ylabel("Normalized Volume")
    
    # Format dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def plot_combined_illustration(df_epi, df_trend, output_path):
    # Overlay simply for illustration
    if df_trend is None: return

    mask_e = (df_epi['Date'] >= '2020-01-01') & (df_epi['Date'] <= '2022-01-01')
    data_e = df_epi.loc[mask_e]
    
    mask_t = (df_trend['Date'] >= '2020-01-01') & (df_trend['Date'] <= '2022-01-01')
    data_t = df_trend.loc[mask_t]

    fig, ax1 = plt.subplots(figsize=(6, 3))
    
    # Epi
    ax1.plot(data_e['Date'], data_e['WeeklyRate'], color='#1f77b4', linewidth=2, label='Hospitalization')
    ax1.set_ylabel('Hospitalization Rate', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Trend
    ax2 = ax1.twinx()
    ax2.plot(data_t['Date'], data_t['Val'], color='#ff7f0e', linewidth=2, linestyle='--', label='Search Trend')
    ax2.set_ylabel('Search Interest', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')
    ax2.spines['right'].set_visible(True) # Keep right spine for dual axis
    
    ax1.set_title("Combined Signals (Lag Correlation)", fontsize=12, pad=10, loc='left')
    
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def main():
    setup_style()
    df_epi, df_trend = load_data()
    
    out_dir = Path('outputs/paper_figures/framework_inputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_epidemic_curve(df_epi, out_dir / 'input_epidemic_curve.png')
    plot_search_trend(df_trend, out_dir / 'input_search_trend.png')
    plot_combined_illustration(df_epi, df_trend, out_dir / 'input_combined_signals.png')

if __name__ == "__main__":
    main()
