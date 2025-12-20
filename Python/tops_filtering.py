import struct
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys


def read_bin_file(file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read binary file and parse book tops data."""
    records = []
    with open(file_path, 'rb') as f:
        hdr_data = f.read(24)
        if len(hdr_data) < 24:
            raise ValueError("File too short to contain header")
        
        feed_id, dateint, number_of_tops, symbol_idx = struct.unpack('<QIIQ', hdr_data)
                
        print(f"Header: feed_id={feed_id}, dateint={dateint}, number_of_tops={number_of_tops}, symbol_idx={symbol_idx}")
        
        for i in range(number_of_tops):
            data = f.read(88)
            if len(data) < 88:
                print(f"Warning: incomplete record at position {i}")
                break
            
            ts, seqno = struct.unpack('<QQ', data[0:16])
            levels = []
            for j in range(3):
                offset = 16 + j * 24
                bid_nanos, ask_nanos, bid_qty, ask_qty = struct.unpack('<qqII', data[offset:offset+24])
                levels.append({
                    'bid_nanos': bid_nanos,
                    'ask_nanos': ask_nanos,
                    'bid_qty': bid_qty,
                    'ask_qty': ask_qty
                })
            
            records.append({
                'ts': ts,
                'seqno': seqno,
                'bid1_nanos': levels[0]['bid_nanos'],
                'ask1_nanos': levels[0]['ask_nanos'],
                'bid1_qty': levels[0]['bid_qty'],
                'ask1_qty': levels[0]['ask_qty'],
                'bid2_nanos': levels[1]['bid_nanos'],
                'ask2_nanos': levels[1]['ask_nanos'],
                'bid2_qty': levels[1]['bid_qty'],
                'ask2_qty': levels[1]['ask_qty'],
                'bid3_nanos': levels[2]['bid_nanos'],
                'ask3_nanos': levels[2]['ask_nanos'],
                'bid3_qty': levels[2]['bid_qty'],
                'ask3_qty': levels[2]['ask_qty'],
            })
    
    return records


def calculate_obi(record: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate OBI_1 and OBI_3 for a record."""
    bid1_qty = record['bid1_qty']
    ask1_qty = record['ask1_qty']
    if bid1_qty + ask1_qty == 0:
        obi1 = 0.0
    else:
        obi1 = (bid1_qty - ask1_qty) / (bid1_qty + ask1_qty)
    
    w = [1.0, 0.5, 0.25]
    bid_depth = (w[0] * record['bid1_qty'] + 
                 w[1] * record['bid2_qty'] + 
                 w[2] * record['bid3_qty'])
    ask_depth = (w[0] * record['ask1_qty'] + 
                 w[1] * record['ask2_qty'] + 
                 w[2] * record['ask3_qty'])
    
    if bid_depth + ask_depth == 0:
        obi3 = 0.0
    else:
        obi3 = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    
    return obi1, obi3


def filter_clean_states(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter for clean, tradable states."""
    filtered = []
    last_ts = 0
    
    for record in records:
        if record['ts'] <= last_ts:
            continue
        
        if record['bid1_qty'] <= 0 or record['ask1_qty'] <= 0:
            continue
        
        if record['bid1_nanos'] >= record['ask1_nanos']:
            continue
        
        spread = record['ask1_nanos'] - record['bid1_nanos']
        if spread <= 0:
            continue
        
        last_ts = record['ts']
        filtered.append(record)
    
    return filtered


def compute_percentiles(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute OBI percentiles for threshold-based filtering."""
    obi1_values = []
    obi3_values = []
    
    for record in records:
        obi1, obi3 = calculate_obi(record)
        obi1_values.append(abs(obi1))
        obi3_values.append(abs(obi3))
    
    obi1_arr = np.array(obi1_values)
    obi3_arr = np.array(obi3_values)
    
    return {
        'obi1_p80': np.percentile(obi1_arr, 80),
        'obi1_p90': np.percentile(obi1_arr, 90),
        'obi3_p80': np.percentile(obi3_arr, 80),
        'obi3_p90': np.percentile(obi3_arr, 90),
    }


def filter_imbalance_events(records: List[Dict[str, Any]], 
                            percentiles: Dict[str, float],
                            use_p90: bool = False) -> List[Dict[str, Any]]:
    """Filter for significant imbalance events."""
    filtered = []
    prev_obi1, prev_obi3 = 0.0, 0.0
    
    threshold_key = 'p90' if use_p90 else 'p80'
    obi1_threshold = percentiles[f'obi1_{threshold_key}']
    obi3_threshold = percentiles[f'obi3_{threshold_key}']
    
    for record in records:
        obi1, obi3 = calculate_obi(record)
        
        crosses_threshold = (
            (abs(prev_obi1) < obi1_threshold and abs(obi1) >= obi1_threshold) or
            (abs(prev_obi3) < obi3_threshold and abs(obi3) >= obi3_threshold)
        )
        
        delta_obi1 = abs(obi1 - prev_obi1)
        delta_obi3 = abs(obi3 - prev_obi3)
        shock_event = delta_obi1 > obi1_threshold or delta_obi3 > obi3_threshold
        
        if crosses_threshold or shock_event:
            record['obi1'] = obi1
            record['obi3'] = obi3
            record['delta_obi1'] = delta_obi1
            record['delta_obi3'] = delta_obi3
            filtered.append(record)
        
        prev_obi1, prev_obi3 = obi1, obi3
    
    return filtered


def write_csv(records: List[Dict[str, Any]], output_path: str):
    """Write filtered records to CSV file."""
    if not records:
        print("No records to write")
        return
    
    fieldnames = records[0].keys()
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"CSV file created: {output_path} with {len(records)} records")


def main():
    if len(sys.argv) != 3:
        print("Usage: python tops_filtering.py <input_bin_file> <output_csv_file>")
        print("Example: python tops_filtering.py data.bin output.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    print(f"\nReading binary file: {input_file}")
    records = read_bin_file(input_file)
    print(f"Total records read: {len(records)}")
    
    print("\n1. Applying clean state filters...")
    clean_records = filter_clean_states(records)
    print(f"Clean records: {len(clean_records)}")
    
    if len(clean_records) == 0:
        print("No clean records found. Exiting.")
        sys.exit(1)
    
    print("\n2. Computing OBI percentiles...")
    percentiles = compute_percentiles(clean_records)
    print(f"OBI1 thresholds: 80th={percentiles['obi1_p80']:.4f}, 90th={percentiles['obi1_p90']:.4f}")
    print(f"OBI3 thresholds: 80th={percentiles['obi3_p80']:.4f}, 90th={percentiles['obi3_p90']:.4f}")
    
    print("\n3. Filtering imbalance events (80th percentile)...")
    filtered_records = filter_imbalance_events(clean_records, percentiles, use_p90=False)
    print(f"Imbalance events: {len(filtered_records)}")
    
    write_csv(filtered_records, output_file)
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()