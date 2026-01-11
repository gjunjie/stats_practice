# Statistics Practice

A collection of statistical analysis notebooks exploring key concepts in quantitative finance and data science, including fat tails, loss functions, multiple testing, and dimensionality reduction.

## Overview

This repository contains Jupyter notebooks demonstrating important statistical concepts with practical applications, particularly in financial data analysis. Each notebook is self-contained and explores a specific topic with code, visualizations, and interpretations.

## Contents

### 1. Fat Tail Analysis (`fat_tail.ipynb`)

**Topic**: Understanding fat tails in financial returns

**Key Concepts**:
- Value at Risk (VaR) and Expected Shortfall (ES) calculations
- Distribution fitting (Normal, Student-t, Generalized Normal)
- Goodness of fit metrics (AIC, BIC, Kolmogorov-Smirnov test)
- Q-Q plots and visual diagnostics

**Findings**:
- SPX returns exhibit fat tails that are poorly modeled by the Normal distribution
- Student-t distribution (df = 2.65) provides the best fit
- Normal distribution significantly underestimates tail risk (VaR gap: -0.62%, ES gap: -1.70%)

**Data**: S&P 500 (^GSPC) daily returns from 2000-01-01

**Visualizations**:
- **Distribution Fits**: Three histograms with overlaid PDFs comparing Normal, Student-t, and Generalized Normal distributions against empirical SPX returns. The Student-t distribution clearly captures the fat tails better than the Normal distribution.
- **Q-Q Plots**: Three quantile-quantile plots showing how well each distribution fits the data. The Student-t Q-Q plot shows the best alignment with the diagonal reference line, confirming its superior fit.

---

### 2. Loss Function Comparison (`loss_function.ipynb`)

**Topic**: Comparing different loss functions in the presence of outliers

**Key Concepts**:
- Ordinary Least Squares (MSE loss)
- Huber loss (robust regression)
- Mean Absolute Error (MAE) loss
- Trade-offs between overall performance and robustness

**Findings**:
- OLS achieves lower overall MSE but is sensitive to outliers
- Huber and MAE losses provide better performance on typical data points by being robust to outliers
- Demonstrates the importance of choosing appropriate loss functions based on data characteristics

**Data**: Synthetic linear model with asymmetric jumps (90% small noise, 10% large jumps)

**Visualizations**:
- **Scatter Plot**: Shows the linear relationship between x and y, with small noise points (90%) in blue and large asymmetric jumps (10%) in red. The true regression line (y = 1.5x) is overlaid, demonstrating how outliers can distort OLS estimates.
- **Noise Distribution Histogram**: Displays the bimodal distribution of errors, with a narrow peak around zero (small noise) and a broader distribution centered at -1 (asymmetric jumps), illustrating the data generation process.

---

### 3. Multiple Testing Problem (`multiple_test.ipynb`)

**Topic**: The multiple testing problem and spurious results

**Key Concepts**:
- Multiple hypothesis testing
- Family-wise error rate
- Sharpe ratio calculations
- Probability of observing extreme statistics by chance

**Findings**:
- Testing 1 strategy: P(Sharpe ≥ 1.8) = 0.0
- Testing 50 strategies: P(Sharpe ≥ 1.8) = 0.0019
- Demonstrates how multiple testing increases the probability of false discoveries

**Data**: Simulated returns under the null hypothesis (pure noise)

**Visualizations**:
- **Sharpe Ratio Distribution Comparison**: A histogram comparing the distribution of maximum Sharpe ratios when testing 1 strategy versus 50 strategies simultaneously. The plot includes a vertical red dashed line at Sharpe = 1.8, showing how multiple testing shifts the distribution rightward and dramatically increases the probability of observing spurious high Sharpe ratios by chance.

---

### 4. Principal Component Analysis (`pca.ipynb`)

**Topic**: PCA on implied volatility surfaces

**Key Concepts**:
- Principal Component Analysis (PCA)
- Explained variance and scree plots
- Principal component loadings
- Applications to trading strategies

**Findings**:
- PC1 explains 57.5% of variance (level)
- PC2 explains 30.2% of variance (skew)
- PC3 explains 6.3% of variance (curvature)
- First 3 PCs explain 94.0% of total variance

**Applications**:
- Relative-value trading (cross-sectional comparisons)
- Time-series strategies (mean reversion, regime shifts)

**Data**: Synthetic implied volatility surface matrix (100 assets × 50 strikes)

**Visualizations**:
- **IV Surface Examples**: A plot showing five sample implied volatility surfaces across different assets, displaying the typical volatility smile/skew pattern as a function of log-moneyness (k = log(K/F)).
- **Scree Plot**: A bar chart with cumulative variance line showing the explained variance ratio for each principal component. The first three components are clearly dominant, explaining 94% of total variance.
- **Principal Component Loadings**: A line plot displaying the first three principal component loadings as functions of log-moneyness. PC1 (level) is relatively flat, PC2 (skew) shows a linear pattern, and PC3 (curvature) exhibits a quadratic shape, corresponding to the classic volatility surface factors.

---

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone or download this repository:
   ```bash
   cd stats_practice
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebooks

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open any notebook (`.ipynb` file) in your browser

3. Run cells sequentially or execute the entire notebook

---

## Dependencies

- **yfinance** (≥0.2.0): Download financial data
- **pandas** (≥2.0.0): Data manipulation and analysis
- **numpy** (≥1.24.0): Numerical computations
- **matplotlib** (≥3.7.0): Data visualization
- **scipy** (≥1.10.0): Statistical functions and distributions
- **scikit-learn** (≥1.3.0): Machine learning models

---

## Project Structure

```
stats_practice/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── fat_tail.ipynb           # Fat tail analysis
├── loss_function.ipynb      # Loss function comparison
├── multiple_test.ipynb      # Multiple testing demonstration
├── pca.ipynb                # Principal Component Analysis
└── venv/                    # Virtual environment (not tracked in git)
```

---

## Notes

- The notebooks are designed to be educational and self-contained
- Some notebooks download data from the internet (yfinance) - ensure you have an active internet connection
- Results may vary slightly due to random number generation, but the overall conclusions remain consistent
- The virtual environment (`venv/`) should not be committed to version control

---

## License

This project is for educational purposes. Feel free to use and modify the code as needed.

---

## Author

Statistics practice repository for exploring key concepts in quantitative finance and statistical analysis.
