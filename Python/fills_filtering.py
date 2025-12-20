import struct
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys


def read_fills_bin_file(file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read binary file and parse book fills data."""
    records = []
    with open(file_path, 'rb') as f:
        hdr_data = f.read(24)
        if len(hdr_data) < 24:
            raise ValueError("File too short to contain header")
        
        feed_id, dateint, number_of_fills, symbol_idx = struct.unpack('<QIIQ', hdr_data)
                
        print(f"Header: feed_id={feed_id}, dateint={dateint}, number_of_fills={number_of_fills}, symbol_idx={symbol_idx}")
        
        for i in range(number_of_fills):
            data = f.read(90)
            if len(data) < 90:
                print(f"Warning: incomplete record at position {i}")
                break
            
            unpacked = struct.unpack('<QQQ?qIQIIQ?qIqII', data)
            
            records.append({
                'ts': unpacked[0],
                'seq_no': unpacked[1],
                'resting_order_id': unpacked[2],
                'was_hidden': unpacked[3],
                'trade_price': unpacked[4],
                'trade_qty': unpacked[5],
                'execution_id': unpacked[6],
                'resting_original_qty': unpacked[7],
                'resting_order_remaining_qty': unpacked[8],
                'resting_order_last_update_ts': unpacked[9],
                'resting_side_is_bid': unpacked[10],
                'resting_side_price': unpacked[11],
                'resting_side_qty': unpacked[12],
                'opposing_side_price': unpacked[13],
                'opposing_side_qty': unpacked[14],
                'resting_side_number_of_orders': unpacked[15],
            })
    
    return records


def calculate_mid_price(record: Dict[str, Any]) -> float:
    """Calculate mid price from resting and opposing sides."""
    resting_price = record['resting_side_price']
    opposing_price = record['opposing_side_price']
    
    if resting_price == 0 or opposing_price == 0:
        return 0.0
    
    mid_price = (resting_price + opposing_price) / 2.0
    return mid_price


def filter_clean_fills(records: List[Dict[str, Any]], 
                       min_trade_qty: int = 1,
                       exclude_hidden: bool = True) -> List[Dict[str, Any]]:
    """Filter for clean, tradable fills."""
    filtered = []
    last_ts = 0
    for record in records:
        if record['ts'] <= last_ts:
            continue
        
        if exclude_hidden and record['was_hidden']:
            continue
        
        if record['trade_qty'] < min_trade_qty:
            continue
        
        if record['trade_price'] <= 0:
            continue
        
        if record['resting_side_price'] <= 0 or record['opposing_side_price'] <= 0:
            continue
        
        if record['resting_side_qty'] <= 0 or record['opposing_side_qty'] <= 0:
            continue
        
        last_ts = record['ts']
        filtered.append(record)
    
    return filtered


def align_fills_with_imbalance_events(
    fills: List[Dict[str, Any]],
    imbalance_events: List[Dict[str, Any]],
    window_before_ms: float = 5.0,
    window_after_ms: float = 100.0
) -> List[Dict[str, Any]]:
    """Align fills with imbalance events within specified time windows."""
    aligned = []
    used_fill_indices = set()
    
    window_before_ns = int(window_before_ms * 1_000_000)
    window_after_ns = int(window_after_ms * 1_000_000)
    
    fill_idx = 0
    for event in imbalance_events:
        event_ts = event['ts']
        event_mid = (event['bid1_nanos'] + event['ask1_nanos']) / 2.0
        
        window_start = event_ts - window_before_ns
        window_end = event_ts + window_after_ns
        
        while fill_idx < len(fills) and fills[fill_idx]['ts'] < window_start:
            fill_idx += 1
        
        temp_idx = fill_idx
        while temp_idx < len(fills) and fills[temp_idx]['ts'] <= window_end:
            if temp_idx in used_fill_indices:
                temp_idx += 1
                continue
                
            fill = fills[temp_idx].copy()
            
            fill['event_ts'] = event_ts
            fill['time_delta_ns'] = fill['ts'] - event_ts
            fill['time_delta_ms'] = fill['time_delta_ns'] / 1_000_000.0
            
            fill['mid_price'] = calculate_mid_price(fill)
            fill['event_mid_price'] = event_mid
            
            fill['price_vs_mid'] = fill['trade_price'] - event_mid
            
            aggressor_sign = -1 if fill['resting_side_is_bid'] else 1
            fill['signed_volume'] = aggressor_sign * fill['trade_qty']
            fill['aggressor_side'] = 'SELL' if fill['resting_side_is_bid'] else 'BUY'
            
            fill['event_obi1'] = event.get('obi1', 0.0)
            fill['event_obi3'] = event.get('obi3', 0.0)
            
            aligned.append(fill)
            used_fill_indices.add(temp_idx)
            temp_idx += 1
    
    return aligned


def calculate_order_flow_metrics(aligned_fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate order flow and adverse selection metrics."""
    if not aligned_fills:
        return {}
    
    total_buy_volume = sum(f['trade_qty'] for f in aligned_fills if f['aggressor_side'] == 'BUY')
    total_sell_volume = sum(f['trade_qty'] for f in aligned_fills if f['aggressor_side'] == 'SELL')
    
    total_signed_volume = sum(f['signed_volume'] for f in aligned_fills)
    
    total_volume = total_buy_volume + total_sell_volume
    ofi = total_signed_volume / total_volume if total_volume > 0 else 0.0
    
    avg_price_impact = np.mean([f['price_vs_mid'] for f in aligned_fills])
    
    total_notional = sum(f['trade_qty'] * f['price_vs_mid'] for f in aligned_fills)
    vwap_impact = total_notional / total_volume if total_volume > 0 else 0.0
    
    return {
        'total_fills': len(aligned_fills),
        'total_buy_volume': total_buy_volume,
        'total_sell_volume': total_sell_volume,
        'total_signed_volume': total_signed_volume,
        'order_flow_imbalance': ofi,
        'avg_price_impact': avg_price_impact,
        'vwap_price_impact': vwap_impact,
    }


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
    if len(sys.argv) != 4:
        print("Usage: python fills_filtering.py <fills_bin_file> <imbalance_events_csv_file> <output_csv_file>")
        print("Example: python fills_filtering.py fills.bin events.csv output.csv")
        sys.exit(1)
    
    fills_file = sys.argv[1]
    events_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if not Path(fills_file).exists():
        print(f"Error: Input file '{fills_file}' does not exist")
        sys.exit(1)
    
    if not Path(events_file).exists():
        print(f"Error: Events file '{events_file}' does not exist")
        sys.exit(1)
    
    print(f"\n1. Reading fills binary file: {fills_file}")
    fills = read_fills_bin_file(fills_file)
    print(f"Total fills read: {len(fills)}")
    
    print("\n2. Applying clean fill filters...")
    clean_fills = filter_clean_fills(fills, min_trade_qty=1, exclude_hidden=True)
    print(f"Clean fills: {len(clean_fills)}")
    
    if len(clean_fills) == 0:
        print("No clean fills found. Exiting.")
        sys.exit(1)
    
    print(f"\n3. Reading imbalance events from: {events_file}")
    imbalance_events = []
    with open(events_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = {
                'ts': int(row['ts']),
                'bid1_nanos': int(row['bid1_nanos']),
                'ask1_nanos': int(row['ask1_nanos']),
                'obi1': float(row.get('obi1', 0.0)),
                'obi3': float(row.get('obi3', 0.0)),
            }
            imbalance_events.append(event)
    
    print(f"Imbalance events loaded: {len(imbalance_events)}")
    
    print("\n4. Aligning fills with imbalance events...")
    aligned_fills = align_fills_with_imbalance_events(
        clean_fills, 
        imbalance_events,
        window_before_ms=5.0,
        window_after_ms=100.0
    )
    print(f"Aligned fills: {len(aligned_fills)}")
    
    print("\n5. Computing order flow metrics...")
    metrics = calculate_order_flow_metrics(aligned_fills)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    write_csv(aligned_fills, output_file)
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()