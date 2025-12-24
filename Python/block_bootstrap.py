import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count

sns.set_style("whitegrid")

# Price scale constant
PRICE_SCALE = 1e9


def calculate_future_returns(df: pd.DataFrame, delta_ms_list: list = [10, 100, 1000, 5000]) -> pd.DataFrame:
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


def _bootstrap_iteration(args):
    """Single bootstrap iteration."""
    boot_iter, bootstrap_indices_list, y_dict, X_dict = args
    bootstrap_indices = bootstrap_indices_list[boot_iter]
    
    results = {}
    
    for col_name in y_dict.keys():
        y_orig = y_dict[col_name]
        X_orig = X_dict[col_name]
        
        y_boot = y_orig[bootstrap_indices]
        X_boot = X_orig[bootstrap_indices]
        
        try:
            X_mean = np.mean(X_boot)
            y_mean = np.mean(y_boot)
            
            cov_xy = np.mean((X_boot - X_mean) * (y_boot - y_mean))
            var_x = np.mean((X_boot - X_mean) ** 2)
            
            if var_x > 1e-10:
                beta = cov_xy / var_x
            else:
                beta = np.nan
                
            results[col_name] = beta
        except:
            results[col_name] = np.nan
    
    return results


def block_bootstrap_analysis(pooled_df: pd.DataFrame, symbols: list, output_dir: str,
                             delta_ms_list: list = [10, 100, 1000, 5000],
                             block_length_ms: int = 1000,
                             n_bootstrap: int = 300,
                             random_seed: int = 42,
                             n_jobs: int = -1,
                             use_moving_blocks: bool = True,
                             bootstrap_within_symbol: bool = False):
    """Perform block bootstrap to assess stability of OBI coefficients."""
    print("\n" + "="*80)
    print("BLOCK BOOTSTRAP ANALYSIS")
    print("="*80)
    print(f"Block length: {block_length_ms}ms")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"Bootstrap type: {'Moving blocks' if use_moving_blocks else 'Non-overlapping blocks'}")
    print(f"Bootstrap scope: {'Within-symbol' if bootstrap_within_symbol else 'Pooled (time-based)'}")
    print(f"Horizons: {delta_ms_list}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return_cols = [f'future_return_{delta_ms}ms' for delta_ms in delta_ms_list]
    missing_cols = [col for col in return_cols if col not in pooled_df.columns]
    
    if missing_cols:
        print("\nCalculating future returns...")
        pooled_df = (
            pooled_df
            .groupby('symbol', group_keys=False)
            .apply(lambda df: calculate_future_returns(df, delta_ms_list), include_groups=False)
        )
    
    df_valid = pooled_df.dropna(subset=return_cols).copy()
    
    if bootstrap_within_symbol:
        df_valid = df_valid.sort_values(['symbol', 'ts']).reset_index(drop=True)
    else:
        df_valid = df_valid.sort_values('ts').reset_index(drop=True)
    
    print(f"\nTotal observations: {len(df_valid):,}")
    print(f"Symbols: {', '.join(symbols)}")
    
    if bootstrap_within_symbol:
        for symbol in symbols:
            symbol_df = df_valid[df_valid['symbol'] == symbol]
            ts_diff = np.diff(symbol_df['ts'].values)
            avg_interval_ms = np.median(ts_diff) / 1_000_000 if len(ts_diff) > 0 else 0
            print(f"  {symbol}: {len(symbol_df):,} obs, median interval: {avg_interval_ms:.2f}ms")
    else:
        ts_diff = np.diff(df_valid['ts'].values)
        avg_interval_ms = np.median(ts_diff) / 1_000_000
        print(f"Median time between observations: {avg_interval_ms:.2f}ms")
    
    if bootstrap_within_symbol:
        symbol_df = df_valid[df_valid['symbol'] == symbols[0]]
        ts_diff = np.diff(symbol_df['ts'].values)
        avg_interval_ms = np.median(ts_diff) / 1_000_000 if len(ts_diff) > 0 else 1.0
    
    block_size = max(1, int(block_length_ms / avg_interval_ms))
    
    if not bootstrap_within_symbol:
        n_obs = len(df_valid)
        if use_moving_blocks:
            n_blocks_possible = n_obs - block_size + 1 if n_obs > block_size else 1
            print(f"Block size: {block_size:,} observations (~{block_length_ms}ms)")
            print(f"Possible block starting positions: {n_blocks_possible:,}")
        else:
            n_blocks = n_obs // block_size
            print(f"Block size: {block_size:,} observations (~{block_length_ms}ms)")
            print(f"Number of non-overlapping blocks: {n_blocks:,}")
    else:
        print(f"Block size: {block_size:,} observations (~{block_length_ms}ms)")
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1
    
    use_parallel = n_jobs > 1 and n_bootstrap >= 100
    
    if use_parallel:
        print(f"Using {n_jobs} parallel processes")
    else:
        print("Running sequentially")
        n_jobs = 1
    
    np.random.seed(random_seed)
    
    bootstrap_results = {'obi1': {}, 'obi3': {}}
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"BOOTSTRAPPING {obi_col.upper()}")
        print(f"{'='*80}")
        
        y_dict = {}
        X_dict = {}
        
        for delta_ms in delta_ms_list:
            col_name = f'future_return_{delta_ms}ms'
            y_dict[col_name] = df_valid[col_name].values
            X_dict[col_name] = df_valid[obi_col].values
        
        n_obs = len(df_valid)
        
        print(f"\nGenerating {n_bootstrap} bootstrap index sets...", end='', flush=True)
        bootstrap_indices_list = []
        
        for boot_iter in range(n_bootstrap):
            if use_moving_blocks:
                n_blocks_needed = int(np.ceil(n_obs / block_size))
                bootstrap_indices = []
                
                for _ in range(n_blocks_needed):
                    max_start = n_obs - block_size
                    if max_start <= 0:
                        start_idx = 0
                        block_indices = np.arange(n_obs)
                    else:
                        start_idx = np.random.randint(0, max_start + 1)
                        block_indices = np.arange(start_idx, start_idx + block_size)
                    
                    bootstrap_indices.extend(block_indices)
                
                bootstrap_indices = np.array(bootstrap_indices[:n_obs])
            else:
                n_blocks = n_obs // block_size
                block_idx = np.random.choice(n_blocks, size=n_blocks, replace=True)
                
                bootstrap_indices = np.concatenate([
                    np.arange(bi * block_size, min((bi + 1) * block_size, n_obs))
                    for bi in block_idx
                ])
            
            bootstrap_indices_list.append(bootstrap_indices)
        
        print(" Done!")
        
        print("\nCalculating original statistics...")
        for delta_ms in delta_ms_list:
            col_name = f'future_return_{delta_ms}ms'
            y_orig = y_dict[col_name]
            X_orig = X_dict[col_name]
            
            X_mean = np.mean(X_orig)
            y_mean = np.mean(y_orig)
            cov_xy = np.mean((X_orig - X_mean) * (y_orig - y_mean))
            var_x = np.mean((X_orig - X_mean) ** 2)
            beta_orig = cov_xy / var_x if var_x > 1e-10 else np.nan
            
            X_orig_const = sm.add_constant(X_orig)
            model_orig = sm.OLS(y_orig, X_orig_const).fit()
            beta_ols = model_orig.params[1]
            
            print(f"  Δ={delta_ms}ms: β (analytical)={beta_orig:.6f}, β (OLS)={beta_ols:.6f}")
            
            bootstrap_results[obi_col][delta_ms] = {
                'beta_orig': beta_orig,
                'beta_samples': []
            }
        
        print(f"\nRunning {n_bootstrap} bootstrap iterations (all horizons)...", end='', flush=True)
        
        bootstrap_args = [
            (i, bootstrap_indices_list, y_dict, X_dict)
            for i in range(n_bootstrap)
        ]
        
        if use_parallel:
            with Pool(processes=n_jobs) as pool:
                bootstrap_iter_results = pool.map(_bootstrap_iteration, bootstrap_args)
        else:
            bootstrap_iter_results = [_bootstrap_iteration(args) for args in bootstrap_args]
        
        print(" Done!")
        
        for delta_ms in delta_ms_list:
            col_name = f'future_return_{delta_ms}ms'
            beta_samples = np.array([r[col_name] for r in bootstrap_iter_results if not np.isnan(r[col_name])])
            bootstrap_results[obi_col][delta_ms]['beta_samples'] = beta_samples
        
        print("\nBootstrap Results:")
        for delta_ms in delta_ms_list:
            result = bootstrap_results[obi_col][delta_ms]
            beta_samples = result['beta_samples']
            
            beta_mean = np.mean(beta_samples)
            beta_std = np.std(beta_samples)
            beta_ci = np.percentile(beta_samples, [5, 95])
            
            result['beta_mean'] = beta_mean
            result['beta_std'] = beta_std
            result['beta_ci_5'] = beta_ci[0]
            result['beta_ci_95'] = beta_ci[1]
            result['n_samples'] = len(beta_samples)
            
            print(f"\n  Δ = {delta_ms}ms:")
            print(f"    β (original):      {result['beta_orig']:>10.6f}")
            print(f"    β (bootstrap):     {beta_mean:>10.6f} ± {beta_std:>10.6f}")
            print(f"    β [5%, 95%]:       [{beta_ci[0]:>10.6f}, {beta_ci[1]:>10.6f}]")
            print(f"    n samples:         {len(beta_samples):>10,}")
            
            beta_stable = (beta_ci[0] * beta_ci[1]) > 0
            print(f"    Stable (β):        {'✓' if beta_stable else '✗'}")
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"BOOTSTRAP SUMMARY: {obi_col.upper()}")
        print(f"{'='*80}")
        
        summary_data = []
        for delta_ms in delta_ms_list:
            result = bootstrap_results[obi_col][delta_ms]
            summary_data.append({
                'Δ (ms)': delta_ms,
                'β (orig)': result['beta_orig'],
                'β (mean)': result['beta_mean'],
                'β (std)': result['beta_std'],
                'β CI [5%]': result['beta_ci_5'],
                'β CI [95%]': result['beta_ci_95'],
                'n samples': result['n_samples'],
                'Stable': '✓' if (result['beta_ci_5'] * result['beta_ci_95']) > 0 else '✗'
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        bootstrap_type = 'moving' if use_moving_blocks else 'nonoverlap'
        scope = 'within_symbol' if bootstrap_within_symbol else 'pooled'
        output_file = output_path / f'bootstrap_results_{obi_col}_{bootstrap_type}_{scope}.csv'
        df_summary.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
    
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    bootstrap_type = 'moving' if use_moving_blocks else 'nonoverlap'
    scope = 'within_symbol' if bootstrap_within_symbol else 'pooled'
    
    for obi_col in ['obi1', 'obi3']:
        n_cols = 2
        n_rows = 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, delta_ms in enumerate(delta_ms_list):
            ax = axes[i]
            result = bootstrap_results[obi_col][delta_ms]
            
            ax.hist(result['beta_samples'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            
            ax.axvline(result['beta_orig'], color='red', linestyle='--', linewidth=2, label='Original β')
            ax.axvline(result['beta_mean'], color='darkgreen', linestyle='-', linewidth=2, label='Bootstrap mean')
            ax.axvline(result['beta_ci_5'], color='orange', linestyle=':', linewidth=2, label='5th %ile')
            ax.axvline(result['beta_ci_95'], color='orange', linestyle=':', linewidth=2, label='95th %ile')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('β coefficient', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Δ={delta_ms}ms: β={result["beta_mean"]:.4f} [{result["beta_ci_5"]:.4f}, {result["beta_ci_95"]:.4f}]',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
        
        block_type_label = 'Moving Block' if use_moving_blocks else 'Non-overlapping Block'
        plt.suptitle(f'{block_type_label} Bootstrap: β Distribution - {obi_col.upper()} - Pooled ({", ".join(symbols)})',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_file = output_path / f'bootstrap_beta_distributions_{obi_col}_{bootstrap_type}_{scope}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        horizons = []
        beta_orig_list = []
        beta_mean_list = []
        beta_ci_5_list = []
        beta_ci_95_list = []
        
        for delta_ms in delta_ms_list:
            result = bootstrap_results[obi_col][delta_ms]
            horizons.append(delta_ms)
            beta_orig_list.append(result['beta_orig'])
            beta_mean_list.append(result['beta_mean'])
            beta_ci_5_list.append(result['beta_ci_5'])
            beta_ci_95_list.append(result['beta_ci_95'])
        
        ax.plot(horizons, beta_mean_list, marker='o', linewidth=2.5, markersize=10, 
                color='steelblue', label='Bootstrap mean', zorder=3)
        ax.fill_between(horizons, beta_ci_5_list, beta_ci_95_list, 
                         alpha=0.3, color='steelblue', label='90% CI', zorder=2)
        ax.plot(horizons, beta_orig_list, marker='s', linewidth=2, markersize=8, 
                color='red', linestyle='--', alpha=0.7, label='Original', zorder=3)
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
        ax.set_xlabel('Time Horizon (ms)', fontsize=13)
        ax.set_ylabel('β coefficient', fontsize=13)
        ax.set_title(f'Bootstrap Stability: β - {obi_col.upper()}', fontsize=15, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        output_file = output_path / f'bootstrap_stability_summary_{obi_col}_{bootstrap_type}_{scope}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS: BOOTSTRAP STABILITY")
    print(f"{'='*80}")
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{obi_col.upper()}:")
        
        stable_count = sum(1 for delta_ms in delta_ms_list 
                          if (bootstrap_results[obi_col][delta_ms]['beta_ci_5'] * 
                              bootstrap_results[obi_col][delta_ms]['beta_ci_95']) > 0)
        
        print(f"  Stable β across horizons: {stable_count}/{len(delta_ms_list)}")
        
        within_ci = sum(1 for delta_ms in delta_ms_list
                       if (bootstrap_results[obi_col][delta_ms]['beta_ci_5'] <= 
                           bootstrap_results[obi_col][delta_ms]['beta_orig'] <= 
                           bootstrap_results[obi_col][delta_ms]['beta_ci_95']))
        
        print(f"  Original β within 90% CI: {within_ci}/{len(delta_ms_list)}")
        
        avg_ci_width = np.mean([bootstrap_results[obi_col][delta_ms]['beta_ci_95'] - 
                               bootstrap_results[obi_col][delta_ms]['beta_ci_5']
                               for delta_ms in delta_ms_list])
        print(f"  Average 90% CI width: {avg_ci_width:.6f}")
        
        avg_bias = np.mean([bootstrap_results[obi_col][delta_ms]['beta_mean'] - 
                           bootstrap_results[obi_col][delta_ms]['beta_orig']
                           for delta_ms in delta_ms_list])
        print(f"  Average bootstrap bias: {avg_bias:.6f}")

def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python block_bootstrap.py <output_dir> <csv1> <symbol1> [<csv2> <symbol2> ...]")
        print("\nExample:")
        print("  python block_bootstrap.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL")
        print("  python block_bootstrap.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL ../Data/Filtered/SPY_tops_filtered_output.csv SPY ../Data/Filtered/QQQ_tops_filtered_output.csv QQQ")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    csv_files = []
    symbols = []
    
    for i in range(2, len(sys.argv), 2):
        csv_files.append(sys.argv[i])
        symbols.append(sys.argv[i + 1])
    
    print(f"Running block bootstrap analysis for: {', '.join(symbols)}")
    print(f"Output directory: {output_dir}\n")
    
    print("Loading data...")
    dfs = []
    for csv_file, symbol in zip(csv_files, symbols):
        df = pd.read_csv(csv_file)
        df['symbol'] = symbol
        dfs.append(df)
        print(f"  {symbol}: {len(df):,} records")
    
    pooled_df = pd.concat(dfs, ignore_index=True)
    print(f"\nPooled total: {len(pooled_df):,} records")
    
    block_bootstrap_analysis(
        pooled_df, 
        symbols, 
        output_dir,
        delta_ms_list=[10, 100, 1000, 5000],
        block_length_ms=1000,
        n_bootstrap=300,
        n_jobs=-1,
        use_moving_blocks=True,
        bootstrap_within_symbol=False
    )
    
    print("\n" + "="*80)
    print("BLOCK BOOTSTRAP ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()