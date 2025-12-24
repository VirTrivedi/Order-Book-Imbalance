import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time

sns.set_style("whitegrid")

# Price scale constant
PRICE_SCALE = 1e9

# Market hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Time periods
OPEN_START = time(9, 30)
OPEN_END = time(10, 0)
MIDDAY_START = time(10, 0)
MIDDAY_END = time(15, 30)
CLOSE_START = time(15, 30)
CLOSE_END = time(16, 0)


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


def assign_time_period(df: pd.DataFrame) -> pd.DataFrame:
    """Assign time period labels based on timestamp."""
    df['datetime'] = pd.to_datetime(df['ts'], unit='ns', utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert('US/Eastern')
    df['time'] = df['datetime'].dt.time
    
    def get_period(t):
        if OPEN_START <= t < OPEN_END:
            return 'Open'
        elif MIDDAY_START <= t < MIDDAY_END:
            return 'Midday'
        elif CLOSE_START <= t < CLOSE_END:
            return 'Close'
        else:
            return 'Other'
    
    df['period'] = df['time'].apply(get_period)
    
    return df


def time_of_day_analysis(pooled_df: pd.DataFrame, symbols: list, output_dir: str,
                        delta_ms_list: list = [10, 100, 1000, 5000]):
    """Analyze OBI predictive power across different time periods."""
    print("\n" + "="*80)
    print("TIME OF DAY ANALYSIS")
    print("="*80)
    print("Periods:")
    print(f"  Open:   {OPEN_START.strftime('%H:%M')} - {OPEN_END.strftime('%H:%M')}")
    print(f"  Midday: {MIDDAY_START.strftime('%H:%M')} - {MIDDAY_END.strftime('%H:%M')}")
    print(f"  Close:  {CLOSE_START.strftime('%H:%M')} - {CLOSE_END.strftime('%H:%M')}")
    
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
    
    print("Assigning time periods...")
    pooled_df = assign_time_period(pooled_df)
    
    df_valid = pooled_df[pooled_df['period'].isin(['Open', 'Midday', 'Close'])].copy()
    df_valid = df_valid.dropna(subset=return_cols + ['obi1', 'obi3'])
    
    print(f"\nTotal observations: {len(df_valid):,}")
    print("\nObservations by period:")
    for period in ['Open', 'Midday', 'Close']:
        count = len(df_valid[df_valid['period'] == period])
        pct = 100 * count / len(df_valid)
        print(f"  {period:8s}: {count:>10,} ({pct:>5.1f}%)")
    
    results = {'obi1': {}, 'obi3': {}}
    periods = ['Open', 'Midday', 'Close']
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"ANALYSIS: {obi_col.upper()}")
        print(f"{'='*80}")
        
        results[obi_col] = {period: {} for period in periods}
        
        for period in periods:
            print(f"\n{'-'*80}")
            print(f"PERIOD: {period.upper()}")
            print(f"{'-'*80}")
            
            period_df = df_valid[df_valid['period'] == period]
            
            for delta_ms in delta_ms_list:
                col_name = f'future_return_{delta_ms}ms'
                y = period_df[col_name].values
                X = period_df[[obi_col]].values
                X_const = sm.add_constant(X)
                
                model = sm.OLS(y, X_const).fit()
                
                ic = np.corrcoef(period_df[obi_col].values, y)[0, 1]
                
                results[obi_col][period][delta_ms] = {
                    'beta': model.params[1],
                    'tstat': model.tvalues[1],
                    'pval': model.pvalues[1],
                    'r2': model.rsquared,
                    'ic': ic,
                    'n_obs': len(y)
                }
                
                print(f"\nΔ = {delta_ms}ms:")
                print(f"  β:      {model.params[1]:>10.6f}  (t={model.tvalues[1]:>7.2f}, p={model.pvalues[1]:.4f})")
                print(f"  IC:     {ic:>10.6f}")
                print(f"  R²:     {model.rsquared:>10.6f}")
                print(f"  n:      {len(y):>10,}")
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE: {obi_col.upper()}")
        print(f"{'='*80}")
        
        for period in periods:
            print(f"\n{period.upper()}:")
            summary_data = []
            for delta_ms in delta_ms_list:
                res = results[obi_col][period][delta_ms]
                summary_data.append({
                    'Δ (ms)': delta_ms,
                    'β': res['beta'],
                    't-stat': res['tstat'],
                    'p-value': res['pval'],
                    'IC': res['ic'],
                    'R²': res['r2'],
                    'n': res['n_obs'],
                    'Sig': '✓' if res['pval'] < 0.05 else '✗'
                })
            
            df_summary = pd.DataFrame(summary_data)
            print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.6f}' if abs(x) < 1 else f'{x:.2f}'))
        
        output_file = output_path / f'time_of_day_summary_{obi_col}.csv'
        all_data = []
        for period in periods:
            for delta_ms in delta_ms_list:
                res = results[obi_col][period][delta_ms]
                all_data.append({
                    'Period': period,
                    'Δ (ms)': delta_ms,
                    'β': res['beta'],
                    't-stat': res['tstat'],
                    'p-value': res['pval'],
                    'IC': res['ic'],
                    'R²': res['r2'],
                    'n': res['n_obs']
                })
        df_all = pd.DataFrame(all_data)
        df_all.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
    
    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}")
    
    for obi_col in ['obi1', 'obi3']:
        # IC vs Δ across periods
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for period in periods:
            ic_values = [results[obi_col][period][d]['ic'] for d in delta_ms_list]
            marker_style = {'Open': 'o', 'Midday': 's', 'Close': '^'}
            color_style = {'Open': 'darkgreen', 'Midday': 'steelblue', 'Close': 'darkorange'}
            
            ax.plot(delta_ms_list, ic_values, 
                   marker=marker_style[period], 
                   linewidth=2.5, 
                   markersize=10,
                   label=period,
                   color=color_style[period])
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Time Horizon (ms)', fontsize=13)
        ax.set_ylabel('Information Coefficient (IC)', fontsize=13)
        ax.set_title(f'IC vs Time Horizon by Period - {obi_col.upper()}', fontsize=15, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        output_file = output_path / f'time_of_day_ic_{obi_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # β vs Δ across periods
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for period in periods:
            beta_values = [results[obi_col][period][d]['beta'] for d in delta_ms_list]
            marker_style = {'Open': 'o', 'Midday': 's', 'Close': '^'}
            color_style = {'Open': 'darkgreen', 'Midday': 'steelblue', 'Close': 'darkorange'}
            
            ax.plot(delta_ms_list, beta_values, 
                   marker=marker_style[period], 
                   linewidth=2.5, 
                   markersize=10,
                   label=period,
                   color=color_style[period])
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Time Horizon (ms)', fontsize=13)
        ax.set_ylabel('β coefficient', fontsize=13)
        ax.set_title(f'β vs Time Horizon by Period - {obi_col.upper()}', fontsize=15, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        output_file = output_path / f'time_of_day_beta_{obi_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # Combined IC and β
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        for period in periods:
            ic_values = [results[obi_col][period][d]['ic'] for d in delta_ms_list]
            marker_style = {'Open': 'o', 'Midday': 's', 'Close': '^'}
            color_style = {'Open': 'darkgreen', 'Midday': 'steelblue', 'Close': 'darkorange'}
            
            ax1.plot(delta_ms_list, ic_values, 
                    marker=marker_style[period], 
                    linewidth=2.5, 
                    markersize=10,
                    label=period,
                    color=color_style[period])
        
        ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax1.set_xlabel('Time Horizon (ms)', fontsize=12)
        ax1.set_ylabel('Information Coefficient (IC)', fontsize=12)
        ax1.set_title('IC vs Time Horizon', fontsize=13, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, which='both')
        
        for period in periods:
            beta_values = [results[obi_col][period][d]['beta'] for d in delta_ms_list]
            marker_style = {'Open': 'o', 'Midday': 's', 'Close': '^'}
            color_style = {'Open': 'darkgreen', 'Midday': 'steelblue', 'Close': 'darkorange'}
            
            ax2.plot(delta_ms_list, beta_values, 
                    marker=marker_style[period], 
                    linewidth=2.5, 
                    markersize=10,
                    label=period,
                    color=color_style[period])
        
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.set_xlabel('Time Horizon (ms)', fontsize=12)
        ax2.set_ylabel('β coefficient', fontsize=12)
        ax2.set_title('β vs Time Horizon', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, which='both')
        
        plt.suptitle(f'Time of Day Analysis - {obi_col.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_file = output_path / f'time_of_day_combined_{obi_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # R² comparison
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(delta_ms_list))
        width = 0.25
        
        for i, period in enumerate(periods):
            r2_values = [results[obi_col][period][d]['r2'] * 100 for d in delta_ms_list]
            color_style = {'Open': 'darkgreen', 'Midday': 'steelblue', 'Close': 'darkorange'}
            
            ax.bar(x + (i - 1) * width, r2_values, width, 
                  label=period, color=color_style[period], alpha=0.8)
        
        ax.set_xlabel('Time Horizon (ms)', fontsize=13)
        ax.set_ylabel('R² (%)', fontsize=13)
        ax.set_title(f'R² by Period - {obi_col.upper()}', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}' for d in delta_ms_list])
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = output_path / f'time_of_day_r2_{obi_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # t-statistics heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        tstat_data = []
        for period in periods:
            tstat_data.append([results[obi_col][period][d]['tstat'] for d in delta_ms_list])
        
        tstat_array = np.array(tstat_data)
        
        im = ax.imshow(tstat_array, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        ax.set_xticks(np.arange(len(delta_ms_list)))
        ax.set_yticks(np.arange(len(periods)))
        ax.set_xticklabels([f'{d}ms' for d in delta_ms_list], fontsize=11)
        ax.set_yticklabels(periods, fontsize=11)
        
        for i in range(len(periods)):
            for j in range(len(delta_ms_list)):
                text = ax.text(j, i, f'{tstat_array[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=11, fontweight='bold')
        
        ax.set_title(f't-statistics by Period - {obi_col.upper()}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='t-statistic')
        plt.tight_layout()
        
        output_file = output_path / f'time_of_day_tstat_heatmap_{obi_col}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS: TIME OF DAY PATTERNS")
    print(f"{'='*80}")
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{obi_col.upper()}:")
        
        # Average IC by period
        for period in periods:
            avg_ic = np.mean([results[obi_col][period][d]['ic'] for d in delta_ms_list])
            avg_beta = np.mean([results[obi_col][period][d]['beta'] for d in delta_ms_list])
            sig_count = sum(1 for d in delta_ms_list if results[obi_col][period][d]['pval'] < 0.05)
            
            print(f"\n  {period}:")
            print(f"    Average IC:   {avg_ic:>8.6f}")
            print(f"    Average β:    {avg_beta:>8.6f}")
            print(f"    Significant:  {sig_count}/{len(delta_ms_list)} horizons")
        
        # Ranking
        print(f"\n  Period strength (by avg IC):")
        period_ics = {period: np.mean([results[obi_col][period][d]['ic'] for d in delta_ms_list]) 
                     for period in periods}
        ranked = sorted(period_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        for i, (period, ic) in enumerate(ranked, 1):
            print(f"    {i}. {period:8s}  (IC = {ic:>8.6f})")
        

def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python time_of_day_splits.py <output_dir> <csv1> <symbol1> [<csv2> <symbol2> ...]")
        print("\nExample:")
        print("  python time_of_day_splits.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL")
        print("  python time_of_day_splits.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL ../Data/Filtered/SPY_tops_filtered_output.csv SPY ../Data/Filtered/QQQ_tops_filtered_output.csv QQQ")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    csv_files = []
    symbols = []
    
    for i in range(2, len(sys.argv), 2):
        csv_files.append(sys.argv[i])
        symbols.append(sys.argv[i + 1])
    
    print(f"Running time of day analysis for: {', '.join(symbols)}")
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
    
    time_of_day_analysis(
        pooled_df,
        symbols,
        output_dir,
        delta_ms_list=[10, 100, 1000, 5000]
    )
    
    print("\n" + "="*80)
    print("TIME OF DAY ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()