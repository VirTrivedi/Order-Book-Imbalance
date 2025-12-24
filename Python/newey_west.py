import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
from scipy import stats

# Price scale constant
PRICE_SCALE = 1e9


def calculate_future_returns(df: pd.DataFrame, delta_ms_list: list = [10, 50, 100, 250, 500, 1000, 2000, 5000]) -> pd.DataFrame:
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


def newey_west_analysis(pooled_df: pd.DataFrame, symbols: list, output_dir: str,
                        delta_ms_list: list = [10, 50, 100, 250, 500, 1000, 2000, 5000],
                        sampling_interval_ms: float = 1.0):
    """Perform OLS regression with Newey-West HAC standard errors."""
    print("\n" + "="*80)
    print("NEWEY-WEST REGRESSION ANALYSIS")
    print("="*80)
    
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
    print(f"\nTotal observations: {len(df_valid)}")
    print(f"Symbols: {', '.join(symbols)}")
    
    results_dict = {'obi1': [], 'obi3': []}
    
    for obi_col in ['obi1', 'obi3']:
        print(f"\n{'='*80}")
        print(f"ANALYZING {obi_col.upper()}")
        print(f"{'='*80}")
        
        results = []
        
        for delta_ms in delta_ms_list:
            col_name = f'future_return_{delta_ms}ms'
            
            y = df_valid[col_name].values
            X = df_valid[obi_col].values
            X = sm.add_constant(X)
            
            maxlags = max(1, int(delta_ms / sampling_interval_ms))
            
            model = sm.OLS(y, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': maxlags}
            )
            
            beta = model.params[1]
            se = model.bse[1]
            t_stat = model.tvalues[1]
            p_value = model.pvalues[1]
            r_squared = model.rsquared
            n_obs = int(model.nobs)
            
            model_naive = sm.OLS(y, X).fit()
            se_naive = model_naive.bse[1]
            t_stat_naive = model_naive.tvalues[1]
            
            inflation = se / se_naive if se_naive > 0 else np.nan
            
            results.append({
                'Δ (ms)': delta_ms,
                'β': beta,
                'SE (NW)': se,
                'SE (naive)': se_naive,
                'SE inflation': inflation,
                't-stat (NW)': t_stat,
                't-stat (naive)': t_stat_naive,
                'p-value': p_value,
                'R²': r_squared,
                'n': n_obs,
                'maxlags': maxlags,
                'significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            })
            
            print(f"\nΔ = {delta_ms}ms (maxlags = {maxlags}):")
            print(f"  β coefficient:      {beta:>10.6f}")
            print(f"  SE (Newey-West):    {se:>10.6f}")
            print(f"  SE (naive):         {se_naive:>10.6f}")
            print(f"  SE inflation:       {inflation:>10.2f}x")
            print(f"  t-stat (NW):        {t_stat:>10.3f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
            print(f"  t-stat (naive):     {t_stat_naive:>10.3f}")
            print(f"  p-value:            {p_value:>10.6f}")
            print(f"  R²:                 {r_squared:>10.6f}")
            print(f"  n:                  {n_obs:>10,}")
        
        results_dict[obi_col] = results
        df_results = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE: {obi_col.upper()}")
        print(f"{'='*80}")
        print(df_results[['Δ (ms)', 'β', 't-stat (NW)', 'SE inflation', 'R²', 'significant']].to_string(index=False))
    
    for obi_col, results in results_dict.items():
        df_results = pd.DataFrame(results)
        output_file = output_path / f'newey_west_results_{obi_col}.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
    
    print(f"\n{'='*80}")
    print("COMPARISON: OBI1 vs OBI3")
    print(f"{'='*80}")
    
    comparison_data = []
    for i, delta_ms in enumerate(delta_ms_list):
        obi1_result = results_dict['obi1'][i]
        obi3_result = results_dict['obi3'][i]
        
        comparison_data.append({
            'Δ (ms)': delta_ms,
            'β (OBI1)': obi1_result['β'],
            't-stat (OBI1)': obi1_result['t-stat (NW)'],
            'β (OBI3)': obi3_result['β'],
            't-stat (OBI3)': obi3_result['t-stat (NW)'],
            'Stronger': 'OBI1' if abs(obi1_result['t-stat (NW)']) > abs(obi3_result['t-stat (NW)']) else 'OBI3'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    output_file = output_path / 'newey_west_comparison.csv'
    df_comparison.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    for obi_col in ['obi1', 'obi3']:
        results = results_dict[obi_col]
        
        print(f"\n{obi_col.upper()}:")
        
        sig_001 = sum(1 for r in results if r['p-value'] < 0.001)
        sig_01 = sum(1 for r in results if r['p-value'] < 0.01)
        sig_05 = sum(1 for r in results if r['p-value'] < 0.05)
        
        print(f"  Significant at p<0.001: {sig_001}/{len(results)}")
        print(f"  Significant at p<0.01:  {sig_01}/{len(results)}")
        print(f"  Significant at p<0.05:  {sig_05}/{len(results)}")
        
        avg_inflation = np.mean([r['SE inflation'] for r in results])
        print(f"  Average SE inflation:   {avg_inflation:.2f}x")
        
        strongest = max(results, key=lambda x: abs(x['t-stat (NW)']))
        print(f"  Strongest at Δ={strongest['Δ (ms)']}ms: t={strongest['t-stat (NW)']:.2f}")
        
        t_stats = [r['t-stat (NW)'] for r in results]
        if len(t_stats) > 1:
            is_decaying = all(abs(t_stats[i]) >= abs(t_stats[i+1]) for i in range(len(t_stats)-1))
            print(f"  Predictive power decays monotonically: {is_decaying}")
    
    return df_valid


def main():
    if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
        print("Usage: python testing.py <output_dir> <csv1> <symbol1> [<csv2> <symbol2> ...]")
        print("\nExample:")
        print("  python testing.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL")
        print("  python testing.py ../Plots ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL ../Data/Filtered/SPY_tops_filtered_output.csv SPY")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    csv_files = []
    symbols = []
    
    for i in range(2, len(sys.argv), 2):
        csv_files.append(sys.argv[i])
        symbols.append(sys.argv[i + 1])
    
    print(f"Running statistical tests for: {', '.join(symbols)}")
    print(f"Output directory: {output_dir}\n")
    
    print("Loading data...")
    dfs = []
    for csv_file, symbol in zip(csv_files, symbols):
        df = pd.read_csv(csv_file)
        df['symbol'] = symbol
        dfs.append(df)
        print(f"  {symbol}: {len(df)} records")
    
    pooled_df = pd.concat(dfs, ignore_index=True)
    print(f"\nPooled total: {len(pooled_df)} records")
    
    newey_west_analysis(pooled_df, symbols, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()