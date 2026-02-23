
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    # Remove top and right spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def load_data(file_path):
    try:
        # Check if trend file (skip rows=2) or rate file (header=0)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        
        if '类别' in first_line:
            df = pd.read_csv(file_path, skiprows=2)
            df.columns = ['Date', 'Value'] if len(df.columns) == 2 else df.columns
        else:
            df = pd.read_csv(file_path)
            # Standardize rate file columns if needed
            if 'WeeklyRate' in df.columns:
                df = df[['Date', 'WeeklyRate']].rename(columns={'WeeklyRate': 'Value'})
            
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_stacked_plot(data_map, title, output_path, color_map=None):
    """
    data_map: dict of {Label: DataFrame}
    """
    n = len(data_map)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2*n), sharex=True)
    
    if n == 1: axes = [axes]
    
    # Common X limit (intersection of all data or specific range?)
    # Let's use a fixed range for better visual, e.g. 2020-2024
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2024-01-01')

    names = list(data_map.keys())
    
    for i, (name, df) in enumerate(data_map.items()):
        ax = axes[i]
        if df is None or df.empty:
            continue
            
        color = color_map.get(name, '#333333') if color_map else '#1f77b4'
        
        # Plot
        ax.plot(df['Date'], df['Value'], color=color, linewidth=1.5)
        ax.fill_between(df['Date'], 0, df['Value'], color=color, alpha=0.1)
        
        # Label inside plot
        ax.text(0.02, 0.85, name, transform=ax.transAxes, 
                fontsize=11, fontweight='bold', color=color)
        
        # Y-axis tweaks
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)
        
        # Set Grid
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Approximate common range
        ax.set_xlim(start_date, end_date)

    # X-axis format on bottom only
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

def main():
    setup_style()
    base_dir = Path('AME')
    out_dir = Path('outputs/paper_figures/framework_inputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Hospitalization Rates
    hosp_data = {
        'Total Respiratory': load_data(base_dir / 'Cleaned_RESP.csv'),
        'COVID-19': load_data(base_dir / 'Cleaned_COVID.csv'),
        'Flu & RSV': load_data(base_dir / 'Cleaned_Flu_RSV_Combined.csv')
    }
    
    hosp_colors = {
        'Total Respiratory': '#1f77b4', # Blue
        'COVID-19': '#d62728', # Red
        'Flu & RSV': '#2ca02c' # Green
    }
    
    create_stacked_plot(hosp_data, 'Weekly Hospitalization Rates (Stack)', 
                       out_dir / 'stacked_hospitalization.png', hosp_colors)
    
    # 2. Search Trends
    trend_data = {
        'Search: "Fever"': load_data(base_dir / 'fever_Trends.csv'),
        'Search: "Loss of Smell"': load_data(base_dir / 'loss_of_smell_Trends.csv'),
        'Search: "Flu"': load_data(base_dir / 'flu_Trends.csv')
    }
    
    trend_colors = {
        'Search: "Fever"': '#ff7f0e', # Orange
        'Search: "Loss of Smell"': '#9467bd', # Purple
        'Search: "Flu"': '#8c564b' # Brown
    }
    
    create_stacked_plot(trend_data, 'Search Trends Activity (Stack)', 
                       out_dir / 'stacked_trends.png', trend_colors)

if __name__ == "__main__":
    main()
