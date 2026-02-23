# MultiTrend-TwoStage-Forecast

A novel Two-Stage framework for robust epidemic forecasting, combining baseline seasonal modeling with a multi-trend attention mechanism for capturing irregular pandemic surges.

![Model Architecture](outputs/framework_diagram.png) *(To be added)*

## Key Features

*   **Dual-Stage Architecture**: 
    1.  **Stage 1**: Estimates the expected seasonal baseline using pre-pandemic historical data (GRU-based).
    2.  **Stage 2**: Corrects the baseline by modeling the "excess" component using real-time search trends (e.g., "Fever", "Loss of Smell") via a Multi-Trend Attention mechanism.
*   **Trends Encoder**: A specialized deep learning module (Bi-GRU + Multi-Head Self-Attention) designed to extract long-range dependencies from exogenous digital signals.
*   **Robust Uncertainty Quantification**: Implements Conformal Prediction to generate reliable 80%, 90%, and 95% confidence intervals.
*   **Interpretability**: Decomposes the forecast into `Baseline + Excess`, providing clear insights into the drivers of each wave.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/MultiTrend-TwoStage-Forecast.git
    cd MultiTrend-TwoStage-Forecast
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    conda create -n trend_forecast python=3.9
    conda activate trend_forecast
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Data Structure

The `AME/` directory contains the cleaned datasets used in this study:
*   `Cleaned_RESP.csv`: Historical weekly hospitalization rates for respiratory illnesses.
*   `fever_Trends.csv`, `flu_Trends.csv`: Google Trends search volume indices for related keywords.

## Usage

### 1. Run Rolling Forecast
To execute the full rolling forecast pipeline using the Two-Stage model (Warm-Start mode recommended):

```bash
python scripts/run_rolling_forecast.py --mode warm_start --epochs 20
```

This will generate predictions in the `outputs/` directory.

### 2. Evaluate Results
To calculate metrics (RMSE, MAE, Coverage, WIS) and compare against baselines (if available):

```bash
python scripts/evaluate_results.py --twostage-results outputs/predictions.csv --output-dir outputs/metrics
```

### 3. Visualizations
Generate decomposition plots to visualize the Baseline vs. Excess components:

```bash
python scripts/visualize_decomposition.py
```

## Citation

If you use this code in your research, please cite our paper:
> [Author Name] et al. "A Two-Stage Multi-Trend Framework for Robust Epidemic Forecasting." [Journal Name], 2026.

## License

MIT License
