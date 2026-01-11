"""
Script to generate and save all plots from the statistical analysis notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, t, gennorm, probplot, kstest
import yfinance as yf
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Create plots directory
os.makedirs('plots', exist_ok=True)

print("Generating plots...")

# ============================================================================
# 1. FAT TAIL ANALYSIS PLOTS
# ============================================================================
print("1. Generating fat tail analysis plots...")

# Download and prepare data
spx = yf.download("^GSPC", start="2000-01-01")
returns_pd = spx["Close"].pct_change().dropna()
returns = np.asarray(returns_pd)
returns = np.squeeze(returns)
returns = returns[~np.isnan(returns)]

# Fit distributions
fits = {}
mu_norm, sigma_norm = norm.fit(returns)
fits['Normal'] = {'params': (mu_norm, sigma_norm), 'dist': norm}

df_t, loc_t, scale_t = t.fit(returns)
fits['Student-t'] = {'params': (df_t, loc_t, scale_t), 'dist': t}

beta_gnorm, loc_gnorm, scale_gnorm = gennorm.fit(returns)
fits['Generalized Normal'] = {'params': (beta_gnorm, loc_gnorm, scale_gnorm), 'dist': gennorm}

# Plot 1: Histogram with fitted PDFs
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x = np.linspace(returns.min(), returns.max(), 1000)
colors = {'Normal': 'blue', 'Student-t': 'red', 'Generalized Normal': 'orange'}

for idx, (name, fit_info) in enumerate(fits.items()):
    ax = axes[idx]
    ax.hist(returns, bins=100, density=True, alpha=0.6, color='gray', 
            label='Empirical Data', edgecolor='black', linewidth=0.5)
    params = fit_info['params']
    dist = fit_info['dist']
    pdf = dist.pdf(x, *params)
    ax.plot(x, pdf, color=colors[name], linewidth=2, label=f'{name} Fit')
    ax.set_title(f'{name} Distribution Fit', fontsize=12, fontweight='bold')
    ax.set_xlabel('Returns', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(returns.min(), returns.max())

plt.tight_layout()
plt.savefig('plots/fat_tail_histograms.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Q-Q plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (name, fit_info) in enumerate(fits.items()):
    ax = axes[idx]
    params = fit_info['params']
    dist = fit_info['dist']
    probplot(returns, dist=dist, sparams=params, plot=ax)
    ax.set_title(f'Q-Q Plot: {name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.plot(ax.get_xlim(), ax.get_ylim(), 'r--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('plots/fat_tail_qqplots.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. LOSS FUNCTION PLOTS
# ============================================================================
print("2. Generating loss function plots...")

np.random.seed(42)
n_samples = 5000
beta = 1.5
sigma_small = 0.2
prob_jump = 0.1

x = np.random.normal(0, 1, n_samples)
jump = np.where(np.random.rand(n_samples) < prob_jump,
                np.random.normal(-1, 1, n_samples), 0)
noise = np.random.normal(0, sigma_small, n_samples)
epsilon = noise + jump
y = beta * x + epsilon

is_large = (jump != 0).astype(int)
df = pd.DataFrame({'x': x, 'y': y, 'epsilon': epsilon, 'jump': jump, 'is_large_jump': is_large})

# Plot 1: Scatter and histogram
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Data Visualization: Linear Model with Asymmetric Jumps', fontsize=16, fontweight='bold')

small_mask = df['is_large_jump'] == 0
large_mask = df['is_large_jump'] == 1

axes[0].scatter(df.loc[small_mask, 'x'], df.loc[small_mask, 'y'], 
                alpha=0.6, s=20, c='blue', label='Small noise (90%)', edgecolors='none')
axes[0].scatter(df.loc[large_mask, 'x'], df.loc[large_mask, 'y'], 
                alpha=0.8, s=50, c='red', label='Large jumps (10%)', edgecolors='black', linewidth=0.5)
x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
y_true_line = beta * x_line
axes[0].plot(x_line, y_true_line, 'k--', linewidth=2, 
             label=f'True line: y = {beta}x', alpha=0.7)
axes[0].set_xlabel('x', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('Scatter Plot: x vs y (Linear Model)', fontsize=13, fontweight='bold')
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].hist(df.loc[small_mask, 'epsilon'], bins=50, alpha=0.7, color='blue', 
             label=f'Small noise (σ={sigma_small})', density=True, edgecolor='black', linewidth=0.5)
axes[1].hist(df.loc[large_mask, 'epsilon'], bins=30, alpha=0.7, color='red', 
             label=f'Asymmetric jumps (μ=-1, σ=1)', density=True, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('ε (noise)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Distribution of Noise ε (Asymmetric Tails)', fontsize=13, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/loss_function_data.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. MULTIPLE TESTING PLOTS
# ============================================================================
print("3. Generating multiple testing plots...")

T = 1260
sigma = 0.01

def sharpe_tstat(r):
    mu = np.mean(r)
    sd = np.std(r, ddof=1)
    sharpe = mu / sd * np.sqrt(252)
    tstat = mu / (sd / np.sqrt(len(r)))
    return sharpe, tstat

def max_stat_under_null(N):
    sharpes = []
    tstats = []
    for _ in range(N):
        r = np.random.normal(0, sigma, T)
        s, t = sharpe_tstat(r)
        sharpes.append(s)
        tstats.append(t)
    return np.max(sharpes), np.max(tstats)

M = 10000
np.random.seed(42)
max_sharpe_1 = []
max_sharpe_50 = []

for _ in range(M):
    s1, _ = max_stat_under_null(1)
    s50, _ = max_stat_under_null(50)
    max_sharpe_1.append(s1)
    max_sharpe_50.append(s50)

plt.figure(figsize=(10, 6))
plt.hist(max_sharpe_1, bins=50, alpha=0.6, label="1 test")
plt.hist(max_sharpe_50, bins=50, alpha=0.6, label="50 tests")
plt.axvline(1.8, color="red", linestyle="--", label="Sharpe = 1.8")
plt.legend()
plt.xlabel("Sharpe Ratio")
plt.ylabel("Frequency")
plt.title("Multiple Testing: Distribution of Maximum Sharpe Ratios", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('plots/multiple_testing.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. PCA PLOTS
# ============================================================================
print("4. Generating PCA plots...")

def create_iv_surface_matrix(
    n_stocks: int = 100,
    n_strikes: int = 50,
    noise_std: float = 0.01, 
    k_range: tuple = (-0.3, 0.3),
    random_seed: int = 42, 
    level_mean: float = 0.2,
    level_mean_std: float = 0.02,
    level_std: float = 0.01,
    skew_mean: float = -0.15,
    skew_mean_std: float = 0.1,
    skew_std: float = 0.1,
    curvature_mean: float = 0.8,
    curvature_mean_std: float = 0.5,
    curvature_std: float = 0.25,
    fourth_mean: float = 0.3,
    fourth_mean_std: float = 0.2,
    fourth_std: float = 0.1,
) -> np.ndarray:
    np.random.seed(random_seed)
    k = np.linspace(k_range[0], k_range[1], n_strikes)
    level = np.ones(n_strikes)
    skew = k.copy()
    curvature = k**2
    fourth = k**4
    B = np.column_stack([level, skew, curvature, fourth])
    stock_level_means = np.random.normal(level_mean, level_mean_std, n_stocks)
    stock_skew_means = np.random.normal(skew_mean, skew_mean_std, n_stocks)
    stock_curvature_means = np.abs(np.random.normal(curvature_mean, curvature_mean_std, n_stocks))
    stock_fourth_means = np.abs(np.random.normal(fourth_mean, fourth_mean_std, n_stocks))
    a = np.random.normal(stock_level_means, level_std)
    b = np.random.normal(stock_skew_means, skew_std)
    c = np.abs(np.random.normal(stock_curvature_means, curvature_std))
    d = np.abs(np.random.normal(stock_fourth_means, fourth_std))
    F = np.column_stack([a, b, c, d])
    X = F @ B.T
    X = np.maximum(X, 0.05)
    X += np.random.normal(0, noise_std, X.shape)
    X = np.maximum(X, 0.05)
    return X

n_stocks = 100
n_strikes = 50
X = create_iv_surface_matrix(n_stocks=n_stocks, n_strikes=n_strikes)
k = np.linspace(-0.3, 0.3, n_strikes)

# Plot 1: Sample IV surfaces
plt.figure(figsize=(10, 6))
plt.plot(k, X[0], 'b-', linewidth=2, marker='o', markersize=4, label='Stock 0')
plt.plot(k, X[20], 'r-', linewidth=2, marker='s', markersize=4, label='Stock 20')
plt.plot(k, X[40], 'g-', linewidth=2, marker='^', markersize=4, label='Stock 40')
plt.plot(k, X[60], 'm-', linewidth=2, marker='d', markersize=4, label='Stock 60')
plt.plot(k, X[80], 'c-', linewidth=2, marker='v', markersize=4, label='Stock 80')
plt.xlabel('Log-Moneyness (k = log(K/F))', fontsize=12)
plt.ylabel('Implied Volatility', fontsize=12)
plt.title('IV Surfaces: Five Sample Assets', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/pca_iv_surfaces.png', dpi=150, bbox_inches='tight')
plt.close()

# Perform PCA
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
PC_loadings = Vt.T
explained_var = s**2 / (n_stocks - 1)
explained_var_ratio = explained_var / explained_var.sum()
cumulative_var = np.cumsum(explained_var_ratio)

# Plot 2: Scree plot
n_components_plot = min(5, len(s))
components = np.arange(1, n_components_plot + 1)

plt.figure(figsize=(10, 6))
plt.bar(components, explained_var_ratio[:n_components_plot], alpha=0.7, label='Individual')
plt.plot(components, cumulative_var[:n_components_plot], 'ro-', label='Cumulative', linewidth=2)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('Scree Plot: Explained Variance by Principal Component', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(components)
plt.tight_layout()
plt.savefig('plots/pca_scree.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: PC loadings
n_components_to_plot = min(3, PC_loadings.shape[1])
plt.figure(figsize=(12, 6))
colors = ['b', 'r', 'g']
markers = ['o', 's', '^']

for i in range(n_components_to_plot):
    plt.plot(k, PC_loadings[:, i], linewidth=2, marker=markers[i], markersize=4, 
             color=colors[i], label=f'PC {i+1} (Explained Variance: {explained_var_ratio[i]:.2%})')

plt.xlabel('Log-Moneyness (k = log(K/F))', fontsize=12)
plt.ylabel('PC Loading', fontsize=12)
plt.title('Principal Component Loadings', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('plots/pca_loadings.png', dpi=150, bbox_inches='tight')
plt.close()

print("All plots generated successfully!")
print(f"Plots saved in 'plots/' directory")
