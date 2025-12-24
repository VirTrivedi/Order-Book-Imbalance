import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

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


def calculate_controls(df: pd.DataFrame, vol_window: int = 100) -> pd.DataFrame:
    """Calculate control variables: spread and volatility."""
    if 'mid_price' not in df.columns:
        df['mid_price'] = 0.5 * (df['bid1_nanos'] + df['ask1_nanos']) / PRICE_SCALE
    
    df['spread'] = (df['ask1_nanos'] - df['bid1_nanos']) / PRICE_SCALE
    df['spread_bps'] = 1e4 * df['spread'] / df['mid_price']
    df['mid_return'] = df['mid_price'].pct_change() * 1e4
    df['vol'] = df['mid_return'].rolling(window=vol_window, min_periods=20).std()    
    df['vol_abs'] = df['mid_return'].abs()
    
    return df


def run_control_regressions(pooled_df: pd.DataFrame, symbols: list, output_dir: str,
                            delta_ms_list: list = [10, 100, 1000, 5000],
                            use_abs_vol: bool = False):
    """Run control regressions."""
    print("\n" + "="*80)
    print("CONTROL REGRESSIONS: OBI + SPREAD + VOLATILITY")
    print("="*80)
    print(f"Volatility measure: {'Absolute returns' if use_abs_vol else 'Rolling std (window=100)'}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return_cols = [f'future_return_{delta_ms}ms' for delta_ms in delta_ms_list]
    missing_cols = [col for col in return_cols if col not in pooled_df.columns]
    
    if missing_cols:
        print("\nCalculating future returns...")
        dfs_with_returns = []
        for symbol in symbols:
            symbol_df = pooled_df[pooled_df['symbol'] == symbol].copy()
            symbol_df = calculate_future_returns(symbol_df, delta_ms_list)
            dfs_with_returns.append(symbol_df)
        pooled_df = pd.concat(dfs_with_returns, ignore_index=True)
    
    print("Calculating control variables (spread, volatility)...")
    dfs_with_controls = []
    for symbol in symbols:
        symbol_df = pooled_df[pooled_df['symbol'] == symbol].copy()
        symbol_df = calculate_controls(symbol_df)
        dfs_with_controls.append(symbol_df)
    pooled_df = pd.concat(dfs_with_controls, ignore_index=True)
    
    vol_col = 'vol_abs' if use_abs_vol else 'vol'
    
    control_cols = ['spread_bps', vol_col]
    df_valid = pooled_df.dropna(subset=return_cols + control_cols + ['obi1', 'obi3']).copy()
    
    print(f"\nTotal observations after filtering: {len(df_valid):,}")
    print(f"Symbols: {', '.join(symbols)}")
    
    results = {'obi1': {}, 'obi3': {}}    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"CONTROL REGRESSIONS: {obi_col.upper()}")
        print(f"{'='*80}")
        
        for delta_ms in delta_ms_list:
            col_name = f'future_return_{delta_ms}ms'
            y = df_valid[col_name].values
            
            # OBI only (baseline)
            X1 = df_valid[[obi_col]].values
            X1_const = sm.add_constant(X1)
            model1 = sm.OLS(y, X1_const).fit()
            
            # OBI + Spread
            X2 = df_valid[[obi_col, 'spread_bps']].values
            X2_const = sm.add_constant(X2)
            model2 = sm.OLS(y, X2_const).fit()
            
            # OBI + Vol
            X3 = df_valid[[obi_col, vol_col]].values
            X3_const = sm.add_constant(X3)
            model3 = sm.OLS(y, X3_const).fit()
            
            # OBI + Spread + Vol
            X4 = df_valid[[obi_col, 'spread_bps', vol_col]].values
            X4_const = sm.add_constant(X4)
            model4 = sm.OLS(y, X4_const).fit()
            
            beta_obi_only = model1.params[1]
            beta_obi_spread = model2.params[1]
            beta_obi_vol = model3.params[1]
            beta_obi_full = model4.params[1]
            
            gamma_spread = model4.params[2]
            delta_vol = model4.params[3]
            
            results[obi_col][delta_ms] = {
                # OBI only
                'beta_obi_only': beta_obi_only,
                'tstat_obi_only': model1.tvalues[1],
                'pval_obi_only': model1.pvalues[1],
                'r2_obi_only': model1.rsquared,
                
                # OBI + Spread
                'beta_obi_spread': beta_obi_spread,
                'tstat_obi_spread': model2.tvalues[1],
                'pval_obi_spread': model2.pvalues[1],
                'gamma_spread': model2.params[2],
                'tstat_spread': model2.tvalues[2],
                'pval_spread': model2.pvalues[2],
                'r2_obi_spread': model2.rsquared,
                
                # OBI + Vol
                'beta_obi_vol': beta_obi_vol,
                'tstat_obi_vol': model3.tvalues[1],
                'pval_obi_vol': model3.pvalues[1],
                'delta_vol_only': model3.params[2],
                'tstat_vol_only': model3.tvalues[2],
                'pval_vol_only': model3.pvalues[2],
                'r2_obi_vol': model3.rsquared,
                
                # OBI + Spread + Vol
                'beta_obi_full': beta_obi_full,
                'tstat_obi_full': model4.tvalues[1],
                'pval_obi_full': model4.pvalues[1],
                'gamma_spread_full': gamma_spread,
                'tstat_spread_full': model4.tvalues[2],
                'pval_spread_full': model4.pvalues[2],
                'delta_vol_full': delta_vol,
                'tstat_vol_full': model4.tvalues[3],
                'pval_vol_full': model4.pvalues[3],
                'r2_full': model4.rsquared,
                
                # Beta retention
                'beta_retention_spread': beta_obi_spread / beta_obi_only if beta_obi_only != 0 else np.nan,
                'beta_retention_vol': beta_obi_vol / beta_obi_only if beta_obi_only != 0 else np.nan,
                'beta_retention_full': beta_obi_full / beta_obi_only if beta_obi_only != 0 else np.nan,
            }
            
            print(f"\n{'─'*80}")
            print(f"Δ = {delta_ms}ms")
            print(f"{'─'*80}")
            
            print("\nModel 1: r ~ OBI")
            print(f"  β(OBI):        {beta_obi_only:>10.6f}  (t={model1.tvalues[1]:>7.2f}, p={model1.pvalues[1]:.4f})")
            print(f"  R²:            {model1.rsquared:>10.6f}")
            
            print("\nModel 2: r ~ OBI + Spread")
            print(f"  β(OBI):        {beta_obi_spread:>10.6f}  (t={model2.tvalues[1]:>7.2f}, p={model2.pvalues[1]:.4f})")
            print(f"  γ(Spread):     {model2.params[2]:>10.6f}  (t={model2.tvalues[2]:>7.2f}, p={model2.pvalues[2]:.4f})")
            print(f"  R²:            {model2.rsquared:>10.6f}  (Δ={model2.rsquared - model1.rsquared:>+.6f})")
            print(f"  β retention:   {beta_obi_spread/beta_obi_only:>10.2%}")
            
            print("\nModel 3: r ~ OBI + Vol")
            print(f"  β(OBI):        {beta_obi_vol:>10.6f}  (t={model3.tvalues[1]:>7.2f}, p={model3.pvalues[1]:.4f})")
            print(f"  δ(Vol):        {model3.params[2]:>10.6f}  (t={model3.tvalues[2]:>7.2f}, p={model3.pvalues[2]:.4f})")
            print(f"  R²:            {model3.rsquared:>10.6f}  (Δ={model3.rsquared - model1.rsquared:>+.6f})")
            print(f"  β retention:   {beta_obi_vol/beta_obi_only:>10.2%}")
            
            print("\nModel 4: r ~ OBI + Spread + Vol (FULL)")
            print(f"  β(OBI):        {beta_obi_full:>10.6f}  (t={model4.tvalues[1]:>7.2f}, p={model4.pvalues[1]:.4f})")
            print(f"  γ(Spread):     {gamma_spread:>10.6f}  (t={model4.tvalues[2]:>7.2f}, p={model4.pvalues[2]:.4f})")
            print(f"  δ(Vol):        {delta_vol:>10.6f}  (t={model4.tvalues[3]:>7.2f}, p={model4.pvalues[3]:.4f})")
            print(f"  R²:            {model4.rsquared:>10.6f}  (Δ={model4.rsquared - model1.rsquared:>+.6f})")
            print(f"  β retention:   {beta_obi_full/beta_obi_only:>10.2%}")
            
            sig_marker = '✓' if model4.pvalues[1] < 0.05 else '✗'
            print(f"\n  OBI significant in full model: {sig_marker}")
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE: {obi_col.upper()}")
        print(f"{'='*80}")
        
        summary_data = []
        for delta_ms in delta_ms_list:
            res = results[obi_col][delta_ms]
            summary_data.append({
                'Δ (ms)': delta_ms,
                'β(OBI) only': res['beta_obi_only'],
                't(OBI) only': res['tstat_obi_only'],
                'R² only': res['r2_obi_only'],
                'β(OBI) full': res['beta_obi_full'],
                't(OBI) full': res['tstat_obi_full'],
                'γ(Spread)': res['gamma_spread_full'],
                't(Spread)': res['tstat_spread_full'],
                'δ(Vol)': res['delta_vol_full'],
                't(Vol)': res['tstat_vol_full'],
                'R² full': res['r2_full'],
                'β retention': res['beta_retention_full'],
                'Sig': '✓' if res['pval_obi_full'] < 0.05 else '✗'
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.0f}'))
        
        vol_suffix = 'abs' if use_abs_vol else 'rolling'
        output_file = output_path / f'control_regressions_{obi_col}_{vol_suffix}.csv'
        df_summary.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
    
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    for obi_col in ['obi1', 'obi3']:
        # Beta coefficients across models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        horizons = delta_ms_list
        beta_only = [results[obi_col][d]['beta_obi_only'] for d in horizons]
        beta_spread = [results[obi_col][d]['beta_obi_spread'] for d in horizons]
        beta_vol = [results[obi_col][d]['beta_obi_vol'] for d in horizons]
        beta_full = [results[obi_col][d]['beta_obi_full'] for d in horizons]
        
        # Beta coefficients
        ax1.plot(horizons, beta_only, marker='o', linewidth=2.5, markersize=8, 
                label='OBI only', color='steelblue')
        ax1.plot(horizons, beta_spread, marker='s', linewidth=2, markersize=7, 
                label='OBI + Spread', color='orange', linestyle='--')
        ax1.plot(horizons, beta_vol, marker='^', linewidth=2, markersize=7, 
                label='OBI + Vol', color='green', linestyle='--')
        ax1.plot(horizons, beta_full, marker='D', linewidth=2.5, markersize=7, 
                label='OBI + Spread + Vol', color='red', linestyle=':')
        ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax1.set_xlabel('Time Horizon (ms)', fontsize=12)
        ax1.set_ylabel('β(OBI) coefficient', fontsize=12)
        ax1.set_title(f'β(OBI) Across Models - {obi_col.upper()}', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, which='both')
        
        # Beta retention %
        retention_spread = [results[obi_col][d]['beta_retention_spread'] * 100 for d in horizons]
        retention_vol = [results[obi_col][d]['beta_retention_vol'] * 100 for d in horizons]
        retention_full = [results[obi_col][d]['beta_retention_full'] * 100 for d in horizons]
        
        ax2.plot(horizons, retention_spread, marker='s', linewidth=2, markersize=7, 
                label='OBI + Spread', color='orange')
        ax2.plot(horizons, retention_vol, marker='^', linewidth=2, markersize=7, 
                label='OBI + Vol', color='green')
        ax2.plot(horizons, retention_full, marker='D', linewidth=2.5, markersize=7, 
                label='OBI + Spread + Vol', color='red')
        ax2.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (100%)')
        ax2.set_xlabel('Time Horizon (ms)', fontsize=12)
        ax2.set_ylabel('β retention (%)', fontsize=12)
        ax2.set_title(f'β(OBI) Retention - {obi_col.upper()}', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        vol_suffix = 'abs' if use_abs_vol else 'rolling'
        output_file = output_path / f'control_beta_comparison_{obi_col}_{vol_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # R² comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        r2_only = [results[obi_col][d]['r2_obi_only'] * 100 for d in horizons]
        r2_spread = [results[obi_col][d]['r2_obi_spread'] * 100 for d in horizons]
        r2_vol = [results[obi_col][d]['r2_obi_vol'] * 100 for d in horizons]
        r2_full = [results[obi_col][d]['r2_full'] * 100 for d in horizons]
        
        x = np.arange(len(horizons))
        width = 0.2
        
        ax.bar(x - 1.5*width, r2_only, width, label='OBI only', color='steelblue', alpha=0.8)
        ax.bar(x - 0.5*width, r2_spread, width, label='OBI + Spread', color='orange', alpha=0.8)
        ax.bar(x + 0.5*width, r2_vol, width, label='OBI + Vol', color='green', alpha=0.8)
        ax.bar(x + 1.5*width, r2_full, width, label='OBI + Spread + Vol', color='red', alpha=0.8)
        
        ax.set_xlabel('Time Horizon (ms)', fontsize=12)
        ax.set_ylabel('R² (%)', fontsize=12)
        ax.set_title(f'R² Comparison Across Models - {obi_col.upper()}', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}' for d in horizons])
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_path / f'control_r2_comparison_{obi_col}_{vol_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # t-statistics heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        tstat_data = []
        row_labels = []
        
        for delta_ms in delta_ms_list:
            res = results[obi_col][delta_ms]
            tstat_data.append([
                res['tstat_obi_only'],
                res['tstat_obi_full'],
                res['tstat_spread_full'],
                res['tstat_vol_full']
            ])
            row_labels.append(f'{delta_ms}ms')
        
        tstat_array = np.array(tstat_data)
        col_labels = ['β(OBI)\nonly', 'β(OBI)\nfull', 'γ(Spread)\nfull', 'δ(Vol)\nfull']
        
        im = ax.imshow(tstat_array, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=11)
        ax.set_yticklabels(row_labels, fontsize=11)
        
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{tstat_array[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax.axhline(y=-0.5, color='black', linewidth=1)
        ax.axvline(x=1.5, color='black', linewidth=2, linestyle='--', alpha=0.5)
        
        ax.set_title(f't-statistics: Control Regressions - {obi_col.upper()}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='t-statistic')
        plt.tight_layout()
        
        output_file = output_path / f'control_tstat_heatmap_{obi_col}_{vol_suffix}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS: CONTROL REGRESSIONS")
    print(f"{'='*80}")
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{obi_col.upper()}:")
        
        avg_retention_full = np.mean([results[obi_col][d]['beta_retention_full'] 
                                     for d in delta_ms_list])
        print(f"  Average β retention (full model): {avg_retention_full:.1%}")
        
        sig_count = sum(1 for d in delta_ms_list if results[obi_col][d]['pval_obi_full'] < 0.05)
        print(f"  OBI significant in full model: {sig_count}/{len(delta_ms_list)} horizons")
        
        avg_r2_increase = np.mean([results[obi_col][d]['r2_full'] - results[obi_col][d]['r2_obi_only']
                                   for d in delta_ms_list])
        print(f"  Average R² increase (controls): {avg_r2_increase:+.6f}")
        
        spread_sig = sum(1 for d in delta_ms_list if results[obi_col][d]['pval_spread_full'] < 0.05)
        vol_sig = sum(1 for d in delta_ms_list if results[obi_col][d]['pval_vol_full'] < 0.05)
        print(f"  Spread significant: {spread_sig}/{len(delta_ms_list)} horizons")
        print(f"  Vol significant: {vol_sig}/{len(delta_ms_list)} horizons")


def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python control_regressions.py <output_dir> <csv1> <symbol1> [<csv2> <symbol2> ...]")
        print("\nExample:")
        print("  python control_regressions.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL")
        print("  python control_regressions.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL ../Data/Filtered/SPY_tops_filtered_output.csv SPY ../Data/Filtered/QQQ_tops_filtered_output.csv QQQ")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    csv_files = []
    symbols = []
    
    for i in range(2, len(sys.argv), 2):
        csv_files.append(sys.argv[i])
        symbols.append(sys.argv[i + 1])
    
    print(f"Running control regressions for: {', '.join(symbols)}")
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
    
    run_control_regressions(
        pooled_df,
        symbols,
        output_dir,
        delta_ms_list=[10, 100, 1000, 5000],
        use_abs_vol=False
    )
    
    print("\n" + "="*80)
    print("CONTROL REGRESSIONS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()