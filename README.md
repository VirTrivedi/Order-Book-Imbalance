# Order-Book-Imbalance

Empirical study of whether order book imbalance (OBI) predicts short-horizon mid-price moves in US equities. The pipeline turns binary top-of-book, book-snapshot and fill files into imbalance-event datasets, then runs regression, robustness and time-of-day analyses and produces the plots in `Plots/`. A draft write-up is in `Paper/Order Book Imbalance Draft Paper.pdf`.

Symbols studied: AAPL, SPY, QQQ (filtered output is in `Data/Filtered/`). Timestamps in the data fall on 2022-08-01.

## Definitions

- **OBI1** = (bid1_qty - ask1_qty) / (bid1_qty + ask1_qty)
- **OBI3** = (bid_depth - ask_depth) / (bid_depth + ask_depth), where depth = 1.0 * L1 + 0.5 * L2 + 0.25 * L3 quantity
- **Future return** at horizon Δ: `1e4 * (mid(t + Δ) - mid(t)) / mid(t)`, in basis points, using the first observation at or after `t + Δ`
- Horizons: 10, 50, 100, 250, 500, 1000, 2000, 5000 ms (some scripts use a subset)

## Pipeline

```
book_tops.bin ──tops_filtering.py──> <SYM>_tops_filtered_output.csv   (imbalance events)
                                              │
book_snapshots.bin ──snapshots_filtering.py───┤  aligned to events within ±1 ms
book_fills.bin     ──fills_filtering.py───────┘  -> *_snapshots_filtered_output.csv, *_fills_filtered_output.csv

tops CSVs ──> newey_west.py, control_regressions.py, block_bootstrap.py,
              time_of_day_splits.py, plots.py   ──> Plots/
```

### 1. Filtering (`Python/`)

| Script | Input | What it does |
|---|---|---|
| `tops_filtering.py <tops.bin> <out.csv>` | Binary book tops | Keeps clean states (increasing timestamps, positive top-level quantities, uncrossed book). Computes OBI1 and OBI3, then keeps **imbalance events**: updates where the absolute OBI1 or OBI3 rises through its 80th percentile (computed on that file's own clean states), or where either one jumps by more than that threshold in a single update. |
| `snapshots_filtering.py <snapshots.bin> <events.csv> <out.csv>` | Binary full-book snapshots | Keeps snapshots with increasing timestamps, an uncrossed book and a positive spread. For each imbalance event, takes the nearest unused snapshot within ±1 ms. Adds best-level and total quantities, level-2 depth, order counts, spread, the mid difference versus the event, a flag for resting orders with remaining quantity below their original quantity, and `queue_imbalance` from the best-level quantities. |
| `fills_filtering.py <fills.bin> <events.csv> <out.csv>` | Binary trades | Drops hidden and invalid fills, then assigns each fill to the first imbalance event whose window (5 ms before to 100 ms after the event) contains it. Adds signed volume, aggressor side (opposite of the resting side), and price versus the event mid. |

### 2. Analysis

All take `<output_dir> <csv1> <symbol1> [<csv2> <symbol2> ...]` and pool the symbols.

| Script | Question | Outputs |
|---|---|---|
| `newey_west.py` | Does OBI predict returns once autocorrelation is accounted for? | OLS of future return on OBI with Newey-West standard errors, per horizon (lag count = horizon in ms, assuming 1 ms between observations); `newey_west_results_*.csv`, `newey_west_comparison.csv` |
| `control_regressions.py` | Does the effect survive controls for spread and recent volatility? | Adds spread (bps) and rolling volatility (std of the last 100 mid returns) as controls at 10, 100, 1000 and 5000 ms; `control_regressions_*_rolling.csv`, beta / R² / t-stat plots |
| `block_bootstrap.py` | Is the coefficient stable? | 300 moving-block bootstrap resamples with 1000 ms blocks at 10, 100, 1000 and 5000 ms, run in parallel; `bootstrap_results_*.csv`, distribution and stability plots |
| `time_of_day_splits.py` | Is it stronger at the open, midday or close? | Splits timestamps (converted to US/Eastern) into 09:30-10:00, 10:00-15:30 and 15:30-16:00; beta, R², IC and t-stat plots by period |
| `plots.py` | Descriptive views | OBI distributions, future return vs OBI, predictive decay (beta / R² / IC by horizon), and order-flow response around events. The last one reads `<SYMBOL>_fills_filtered_output.csv` from the same folder as the first tops CSV and is skipped if none are found. |

## Headline result

Pooled Newey-West slopes (basis points of future return per unit of OBI), from `Plots/newey_west_comparison.csv`:

| Δ (ms) | β OBI1 | t OBI1 | β OBI3 | t OBI3 |
|---|---|---|---|---|
| 10 | 0.103 | 210.7 | 0.159 | 144.1 |
| 100 | 0.117 | 120.6 | 0.185 | 80.2 |
| 1000 | 0.136 | 41.1 | 0.220 | 25.7 |
| 5000 | 0.132 | 12.7 | 0.211 | 7.6 |

The slope is positive and significant at every horizon and peaks around 1 s, while significance falls with horizon. OBI1 has the larger t-statistic at every horizon; OBI3 has the larger coefficient. See the paper for interpretation.

## Requirements

`numpy`, `pandas`, `statsmodels`, `matplotlib`, `seaborn`, `scipy`.

## Example

```bash
cd Python
python tops_filtering.py <AAPL_tops.bin> ../Data/Filtered/AAPL_tops_filtered_output.csv
python snapshots_filtering.py <AAPL_snapshots.bin> ../Data/Filtered/AAPL_tops_filtered_output.csv ../Data/Filtered/AAPL_snapshots_filtered_output.csv
python fills_filtering.py <AAPL_fills.bin> ../Data/Filtered/AAPL_tops_filtered_output.csv ../Data/Filtered/AAPL_fills_filtered_output.csv

python newey_west.py ../Plots \
  ../Data/Filtered/AAPL_tops_filtered_output.csv AAPL \
  ../Data/Filtered/SPY_tops_filtered_output.csv SPY \
  ../Data/Filtered/QQQ_tops_filtered_output.csv QQQ
```

## Data format

Binary files start with a 24-byte header (`feed_id`, `dateint`, count, `symbol_idx`), followed by 88-byte tops records, 90-byte fill records, or variable-length snapshots. Prices are integers in nanos (1e-9 dollars).
