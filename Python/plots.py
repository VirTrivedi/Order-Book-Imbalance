import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Price scale constant
PRICE_SCALE = 1e9


def calculate_future_returns(df: pd.DataFrame, delta_ms_list: list = [10, 100, 1000]) -> pd.DataFrame:
    """Calculate future mid-price returns at various time horizons."""    
    df['mid_price'] = 0.5 * (df['bid1_nanos'] + df['ask1_nanos']) / PRICE_SCALE
    df = df.sort_values('ts').reset_index(drop=True)
    
    ts_array = df['ts'].values
    mid_array = df['mid_price'].values
    
    for delta_ms in delta_ms_list:
        delta_ns = delta_ms * 1_000_000
        col_name = f'future_return_{delta_ms}ms'
                
        target_ts = ts_array + delta_ns
        future_indices = np.searchsorted(ts_array, target_ts, side='left')
        valid_mask = future_indices < len(ts_array)
        returns = np.full(len(mid_array), np.nan)

        returns[valid_mask] = (
            1e4
            * (mid_array[future_indices[valid_mask]] - mid_array[valid_mask])
            / mid_array[valid_mask]
        )
        returns[future_indices >= len(ts_array) - 1] = np.nan
        df[col_name] = returns    
    return df


def plot_predictive_decay(pooled_df: pd.DataFrame, symbols: list, output_dir: str,
                          delta_ms_list: list = [10, 50, 100, 250, 500, 1000, 2000, 5000]):
    """Plot predictive decay curve showing how OBI predictive power decays over time."""
    print(f"\nGenerating predictive decay curves...")
    output_path = Path(output_dir)
    
    return_cols = [f'future_return_{delta_ms}ms' for delta_ms in delta_ms_list]
    missing_cols = [col for col in return_cols if col not in pooled_df.columns]
    
    if missing_cols:
        pooled_df = (
            pooled_df
            .groupby('symbol', group_keys=False)
            .apply(lambda df: calculate_future_returns(df, delta_ms_list), include_groups=False)
        )
    
    df_valid = pooled_df.dropna(subset=return_cols).copy()
    
    horizons = []
    ic_obi1 = []
    ic_obi3 = []
    beta_obi1 = []
    beta_obi3 = []
    r2_obi1 = []
    r2_obi3 = []
    
    for delta_ms in delta_ms_list:
        col_name = f'future_return_{delta_ms}ms'
        
        # Information Coefficient
        ic1, _ = stats.spearmanr(df_valid['obi1'], df_valid[col_name], nan_policy='omit')
        ic3, _ = stats.spearmanr(df_valid['obi3'], df_valid[col_name], nan_policy='omit')
        
        # Linear regression beta coefficient
        slope1, intercept1, r_value1, _, _ = stats.linregress(df_valid['obi1'], df_valid[col_name])
        slope3, intercept3, r_value3, _, _ = stats.linregress(df_valid['obi3'], df_valid[col_name])
        
        horizons.append(delta_ms)
        ic_obi1.append(ic1)
        ic_obi3.append(ic3)
        beta_obi1.append(slope1)
        beta_obi3.append(slope3)
        r2_obi1.append(r_value1**2)
        r2_obi3.append(r_value3**2)
        
        print(f"  Δ={delta_ms}ms: IC1={ic1:.4f}, IC3={ic3:.4f}, β1={slope1:.4f}, β3={slope3:.4f}")
    
    # Information Coefficient decay plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(horizons, ic_obi1, marker='o', linewidth=2, markersize=8, 
            color='steelblue', label='OBI1')
    ax.plot(horizons, ic_obi3, marker='s', linewidth=2, markersize=8, 
            color='darkgreen', label='OBI3')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time Horizon (ms)', fontsize=12)
    ax.set_ylabel('Information Coefficient (Spearman ρ)', fontsize=12)
    ax.set_title(f'OBI Predictive Decay - IC - Pooled ({", ".join(symbols)})', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_path / 'predictive_decay_ic.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.close()
    
    # Beta coefficient decay plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(horizons, beta_obi1, marker='o', linewidth=2, markersize=8, 
            color='steelblue', label='OBI1')
    ax.plot(horizons, beta_obi3, marker='s', linewidth=2, markersize=8, 
            color='darkgreen', label='OBI3')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time Horizon (ms)', fontsize=12)
    ax.set_ylabel('Regression Coefficient β (bps per unit OBI)', fontsize=12)
    ax.set_title(f'OBI Predictive Decay - Beta - Pooled ({", ".join(symbols)})', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_path / 'predictive_decay_beta.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # R² decay plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(horizons, r2_obi1, marker='o', linewidth=2, markersize=8, 
            color='steelblue', label='OBI1')
    ax.plot(horizons, r2_obi3, marker='s', linewidth=2, markersize=8, 
            color='darkgreen', label='OBI3')
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time Horizon (ms)', fontsize=12)
    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=12)
    ax.set_title(f'OBI Predictive Decay - R² - Pooled ({", ".join(symbols)})', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    output_file = output_path / 'predictive_decay_r2.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # IC and Beta plot
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    color1 = 'steelblue'
    ax1.set_xlabel('Time Horizon (ms)', fontsize=12)
    ax1.set_ylabel('Information Coefficient (IC)', fontsize=12, color=color1)
    ax1.plot(horizons, ic_obi1, marker='o', linewidth=2, markersize=8, 
            color=color1, label='IC - OBI1', linestyle='-')
    ax1.plot(horizons, ic_obi3, marker='o', linewidth=2, markersize=8, 
            color=color1, label='IC - OBI3', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2 = ax1.twinx()
    color2 = 'darkgreen'
    ax2.set_ylabel('Regression Beta (β)', fontsize=12, color=color2)
    ax2.plot(horizons, beta_obi1, marker='s', linewidth=2, markersize=8, 
            color=color2, label='β - OBI1', linestyle='-')
    ax2.plot(horizons, beta_obi3, marker='s', linewidth=2, markersize=8, 
            color=color2, label='β - OBI3', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    
    ax1.set_title(f'OBI Predictive Decay - Combined View - Pooled ({", ".join(symbols)})', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_path / 'predictive_decay_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_future_return_vs_obi(pooled_df: pd.DataFrame, symbols: list, output_dir: str, 
                               delta_ms_list: list = [10, 100, 1000], n_bins: int = 10):
    """Plot mean future returns vs OBI quantiles."""
    print(f"\nGenerating future return vs OBI plots ({n_bins} bins)...\n")
    output_path = Path(output_dir)
    
    return_cols = [f'future_return_{delta_ms}ms' for delta_ms in delta_ms_list]
    if not all(col in pooled_df.columns for col in return_cols):
        pooled_df = (
            pooled_df
            .groupby('symbol', group_keys=False)
            .apply(lambda df: calculate_future_returns(df, delta_ms_list), include_groups=False)
        )
    
    df_valid = pooled_df.dropna(subset=return_cols).copy()
    
    # Mean future returns vs OBI1 plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_valid['obi1_bin'] = pd.qcut(df_valid['obi1'], q=n_bins, labels=False, duplicates='drop')
    
    bin_centers = []
    for bin_idx in sorted(df_valid['obi1_bin'].unique()):
        bin_data = df_valid[df_valid['obi1_bin'] == bin_idx]
        bin_centers.append(bin_data['obi1'].mean())
    
    colors = ['steelblue', 'darkgreen', 'darkred']
    for delta_ms, color in zip(delta_ms_list, colors):
        col_name = f'future_return_{delta_ms}ms'
        mean_returns = []
        std_returns = []
        
        for bin_idx in sorted(df_valid['obi1_bin'].unique()):
            bin_data = df_valid[df_valid['obi1_bin'] == bin_idx]
            mean_ret = bin_data[col_name].mean()
            std_ret = bin_data[col_name].std() / np.sqrt(len(bin_data))
            mean_returns.append(mean_ret)
            std_returns.append(std_ret)
        
        ax.plot(bin_centers, mean_returns, marker='o', linewidth=2, 
                label=f'Δ = {delta_ms}ms', color=color, markersize=8)
        ax.fill_between(bin_centers, 
                        np.array(mean_returns) - np.array(std_returns),
                        np.array(mean_returns) + np.array(std_returns),
                        alpha=0.2, color=color)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('OBI1 (Bin Mean)', fontsize=12)
    ax.set_ylabel('Mean Future Return (bps)', fontsize=12)
    ax.set_title(f'Mean Future Return vs OBI1 - Pooled ({", ".join(symbols)})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'future_return_vs_obi1.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Mean future returns vs OBI3 plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    df_valid['obi3_bin'] = pd.qcut(df_valid['obi3'], q=n_bins, labels=False, duplicates='drop')
    
    bin_centers = []
    for bin_idx in sorted(df_valid['obi3_bin'].unique()):
        bin_data = df_valid[df_valid['obi3_bin'] == bin_idx]
        bin_centers.append(bin_data['obi3'].mean())
    
    for delta_ms, color in zip(delta_ms_list, colors):
        col_name = f'future_return_{delta_ms}ms'
        mean_returns = []
        std_returns = []
        
        for bin_idx in sorted(df_valid['obi3_bin'].unique()):
            bin_data = df_valid[df_valid['obi3_bin'] == bin_idx]
            mean_ret = bin_data[col_name].mean()
            std_ret = bin_data[col_name].std() / np.sqrt(len(bin_data))
            mean_returns.append(mean_ret)
            std_returns.append(std_ret)
        
        ax.plot(bin_centers, mean_returns, marker='o', linewidth=2, 
                label=f'Δ = {delta_ms}ms', color=color, markersize=8)
        ax.fill_between(bin_centers, 
                        np.array(mean_returns) - np.array(std_returns),
                        np.array(mean_returns) + np.array(std_returns),
                        alpha=0.2, color=color)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('OBI3 (Bin Mean)', fontsize=12)
    ax.set_ylabel('Mean Future Return (bps)', fontsize=12)
    ax.set_title(f'Mean Future Return vs OBI3 - Pooled ({", ".join(symbols)})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'future_return_vs_obi3.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    print("\n=== Future Return Analysis ===")
    for delta_ms in delta_ms_list:
        col_name = f'future_return_{delta_ms}ms'
        print(f"\nΔ = {delta_ms}ms:")
        print(f"  Mean return: {df_valid[col_name].mean():.4f} bps")
        print(f"  Std return:  {df_valid[col_name].std():.4f} bps")
        
        corr_obi1 = df_valid['obi1'].corr(df_valid[col_name])
        corr_obi3 = df_valid['obi3'].corr(df_valid[col_name])
        print(f"  Correlation with OBI1: {corr_obi1:.4f}")
        print(f"  Correlation with OBI3: {corr_obi3:.4f}")


def plot_pooled_obi_distribution(csv_files: list, symbols: list, output_dir: str):
    """Plot pooled OBI distributions across multiple symbols."""
    print("Reading and pooling data...")
    dfs = []
    for csv_file, symbol in zip(csv_files, symbols):
        df = pd.read_csv(csv_file)
        df['symbol'] = symbol
        dfs.append(df)
        print(f"  {symbol}: {len(df)} records")
    
    pooled_df = pd.concat(dfs, ignore_index=True)
    print(f"\nPooled total: {len(pooled_df)} records")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    obi1_p80 = np.percentile(np.abs(pooled_df['obi1']), 80)
    obi1_p90 = np.percentile(np.abs(pooled_df['obi1']), 90)
    obi3_p80 = np.percentile(np.abs(pooled_df['obi3']), 80)
    obi3_p90 = np.percentile(np.abs(pooled_df['obi3']), 90)
    
    print(f"\nPooled OBI1 - 80th: {obi1_p80:.4f}, 90th: {obi1_p90:.4f}")
    print(f"Pooled OBI3 - 80th: {obi3_p80:.4f}, 90th: {obi3_p90:.4f}\n")
    
    # Pooled distributions plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # OBI1 pooled
    axes[0].hist(pooled_df['obi1'], bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(obi1_p80, color='orange', linestyle='--', linewidth=2, label=f'80th: {obi1_p80:.4f}')
    axes[0].axvline(-obi1_p80, color='orange', linestyle='--', linewidth=2)
    axes[0].axvline(obi1_p90, color='red', linestyle='--', linewidth=2, label=f'90th: {obi1_p90:.4f}')
    axes[0].axvline(-obi1_p90, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('OBI1', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Pooled ({", ".join(symbols)}) - OBI1 Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # OBI3 pooled
    axes[1].hist(pooled_df['obi3'], bins=100, alpha=0.7, color='darkgreen', edgecolor='black')
    axes[1].axvline(obi3_p80, color='orange', linestyle='--', linewidth=2, label=f'80th: {obi3_p80:.4f}')
    axes[1].axvline(-obi3_p80, color='orange', linestyle='--', linewidth=2)
    axes[1].axvline(obi3_p90, color='red', linestyle='--', linewidth=2, label=f'90th: {obi3_p90:.4f}')
    axes[1].axvline(-obi3_p90, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('OBI3', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Pooled ({", ".join(symbols)}) - OBI3 Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'pooled_obi_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Overlaid density comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    pooled_df['obi1'].plot(kind='density', ax=ax, color='steelblue', linewidth=2, label='OBI1')
    pooled_df['obi3'].plot(kind='density', ax=ax, color='darkgreen', linewidth=2, label='OBI3')
    
    ax.axvline(obi1_p80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='OBI1 80th %ile')
    ax.axvline(-obi1_p80, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(obi3_p80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='OBI3 80th %ile')
    ax.axvline(-obi3_p80, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('OBI Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Pooled ({", ".join(symbols)}) - OBI Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'pooled_obi_density_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Absolute OBI distributions plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute OBI1
    abs_obi1 = np.abs(pooled_df['obi1'])
    axes[0].hist(abs_obi1, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(obi1_p80, color='orange', linestyle='--', linewidth=2, label=f'80th: {obi1_p80:.4f}')
    axes[0].axvline(obi1_p90, color='red', linestyle='--', linewidth=2, label=f'90th: {obi1_p90:.4f}')
    axes[0].fill_betweenx([0, axes[0].get_ylim()[1]], obi1_p80, axes[0].get_xlim()[1], 
                          color='orange', alpha=0.2, label='Signal region (80%)')
    axes[0].set_xlabel('|OBI1|', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Pooled ({", ".join(symbols)}) - Absolute OBI1', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Absolute OBI3
    abs_obi3 = np.abs(pooled_df['obi3'])
    axes[1].hist(abs_obi3, bins=100, alpha=0.7, color='darkgreen', edgecolor='black')
    axes[1].axvline(obi3_p80, color='orange', linestyle='--', linewidth=2, label=f'80th: {obi3_p80:.4f}')
    axes[1].axvline(obi3_p90, color='red', linestyle='--', linewidth=2, label=f'90th: {obi3_p90:.4f}')
    axes[1].fill_betweenx([0, axes[1].get_ylim()[1]], obi3_p80, axes[1].get_xlim()[1], 
                          color='orange', alpha=0.2, label='Signal region (80%)')
    axes[1].set_xlabel('|OBI3|', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Pooled ({", ".join(symbols)}) - Absolute OBI3', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'pooled_obi_absolute_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # By-symbol comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for symbol in symbols:
        symbol_df = pooled_df[pooled_df['symbol'] == symbol]
        symbol_df['obi1'].plot(kind='density', ax=axes[0], linewidth=2, label=symbol, alpha=0.7)
        symbol_df['obi3'].plot(kind='density', ax=axes[1], linewidth=2, label=symbol, alpha=0.7)
    
    axes[0].set_xlabel('OBI1', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('OBI1 Distribution by Symbol', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('OBI3', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('OBI3 Distribution by Symbol', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_path / 'obi_distribution_by_symbol.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    print("\n=== Pooled Summary Statistics ===")
    print(f"\nOBI1:")
    print(f"  Mean: {pooled_df['obi1'].mean():.4f}")
    print(f"  Std:  {pooled_df['obi1'].std():.4f}")
    print(f"  Min:  {pooled_df['obi1'].min():.4f}")
    print(f"  Max:  {pooled_df['obi1'].max():.4f}")
    print(f"  Median: {pooled_df['obi1'].median():.4f}")
    
    print(f"\nOBI3:")
    print(f"  Mean: {pooled_df['obi3'].mean():.4f}")
    print(f"  Std:  {pooled_df['obi3'].std():.4f}")
    print(f"  Min:  {pooled_df['obi3'].min():.4f}")
    print(f"  Max:  {pooled_df['obi3'].max():.4f}")
    print(f"  Median: {pooled_df['obi3'].median():.4f}")
    
    pct_above_80_obi1 = (abs_obi1 >= obi1_p80).sum() / len(abs_obi1) * 100
    pct_above_90_obi1 = (abs_obi1 >= obi1_p90).sum() / len(abs_obi1) * 100
    pct_above_80_obi3 = (abs_obi3 >= obi3_p80).sum() / len(abs_obi3) * 100
    pct_above_90_obi3 = (abs_obi3 >= obi3_p90).sum() / len(abs_obi3) * 100
    
    print(f"\n=== Signal Coverage ===")
    print(f"OBI1: {pct_above_80_obi1:.2f}% of events above 80th percentile")
    print(f"OBI1: {pct_above_90_obi1:.2f}% of events above 90th percentile")
    print(f"OBI3: {pct_above_80_obi3:.2f}% of events above 80th percentile")
    print(f"OBI3: {pct_above_90_obi3:.2f}% of events above 90th percentile")
    
    print("\n=== Per-Symbol Statistics ===")
    for symbol in symbols:
        symbol_df = pooled_df[pooled_df['symbol'] == symbol]
        print(f"\n{symbol}:")
        print(f"  OBI1 - Mean: {symbol_df['obi1'].mean():.4f}, Std: {symbol_df['obi1'].std():.4f}")
        print(f"  OBI3 - Mean: {symbol_df['obi3'].mean():.4f}, Std: {symbol_df['obi3'].std():.4f}")
        
    return pooled_df


def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python plots.py <output_dir> <csv1> <symbol1> [<csv2> <symbol2> ...]")
        print("\nExample:")
        print("  python plots.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL")
        print("  python plots.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL ../Data/Filtered/SPY_tops_filtered_output.csv SPY ../Data/Filtered/QQQ_tops_filtered_output.csv QQQ")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    csv_files = []
    symbols = []
    
    for i in range(2, len(sys.argv), 2):
        csv_files.append(sys.argv[i])
        symbols.append(sys.argv[i + 1])
    
    print(f"Generating pooled plots for: {', '.join(symbols)}")
    print(f"Output directory: {output_dir}\n")
    
    pooled_df = plot_pooled_obi_distribution(csv_files, symbols, output_dir)
    plot_future_return_vs_obi(pooled_df, symbols, output_dir)
    plot_predictive_decay(pooled_df, symbols, output_dir)

    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()