import struct
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys


def read_snapshots_bin_file(file_path: str) -> List[Dict[str, Any]]:
    """Read binary file and parse book snapshot data."""
    snapshots = []
    with open(file_path, 'rb') as f:
        hdr_data = f.read(24)
        if len(hdr_data) < 24:
            raise ValueError("File too short to contain header")
        
        feed_id, dateint, number_of_snapshots, symbol_idx = struct.unpack('<QIIQ', hdr_data)
        
        print(f"Header: feed_id={feed_id}, dateint={dateint}, number_of_snapshots={number_of_snapshots}, symbol_idx={symbol_idx}")
        
        for i in range(number_of_snapshots):
            hdr = f.read(38)
            if len(hdr) < 38:
                print(f"Warning: incomplete snapshot header at position {i}")
                break
            
            ts, seq_no, byte_offset, symbol_seqno, num_orders, snap_is_crossed, allow_cross_session = struct.unpack('<QQQQI??', hdr)
            
            orders = []
            for j in range(num_orders):
                order_data = f.read(33)
                if len(order_data) < 33:
                    print(f"Warning: incomplete order at snapshot {i}, order {j}")
                    break
                
                price, order_id, qty, original_qty, is_bid, order_ts = struct.unpack('<qQII?Q', order_data)
                
                orders.append({
                    'price': price,
                    'order_id': order_id,
                    'qty': qty,
                    'original_qty': original_qty,
                    'is_bid': is_bid,
                    'order_ts': order_ts
                })
            
            snapshots.append({
                'ts': ts,
                'seq_no': seq_no,
                'byte_offset': byte_offset,
                'symbol_seqno': symbol_seqno,
                'number_of_orders': num_orders,
                'snap_is_crossed': snap_is_crossed,
                'allow_cross_session': allow_cross_session,
                'orders': orders
            })
    
    return snapshots


def extract_snapshot_metrics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from a snapshot."""
    orders = snapshot['orders']
    if not orders:
        return {
            'ts': snapshot['ts'],
            'seq_no': snapshot['seq_no'],
            'snap_is_crossed': snapshot['snap_is_crossed'],
            'best_bid_price': 0,
            'best_ask_price': 0,
            'best_bid_qty': 0,
            'best_ask_qty': 0,
            'total_bid_qty': 0,
            'total_ask_qty': 0,
            'num_bid_orders': 0,
            'num_ask_orders': 0,
            'has_hidden_liquidity': False,
            'bid_depth_level2': 0,
            'ask_depth_level2': 0,
        }
    
    bids = [o for o in orders if o['is_bid']]
    asks = [o for o in orders if not o['is_bid']]
    
    bids.sort(key=lambda x: x['price'], reverse=True)
    asks.sort(key=lambda x: x['price'])
    
    best_bid_price = bids[0]['price'] if bids else 0
    best_ask_price = asks[0]['price'] if asks else 0
    
    best_bid_qty = sum(o['qty'] for o in bids if o['price'] == best_bid_price)
    best_ask_qty = sum(o['qty'] for o in asks if o['price'] == best_ask_price)
    
    total_bid_qty = sum(o['qty'] for o in bids)
    total_ask_qty = sum(o['qty'] for o in asks)
    
    num_bid_orders = len([o for o in bids if o['price'] == best_bid_price])
    num_ask_orders = len([o for o in asks if o['price'] == best_ask_price])
    
    has_hidden = any(o['qty'] < o['original_qty'] for o in orders)
    
    bid_prices = sorted(set(o['price'] for o in bids), reverse=True)
    ask_prices = sorted(set(o['price'] for o in asks))
    
    bid_depth_level2 = 0
    if len(bid_prices) >= 2:
        second_bid_price = bid_prices[1]
        bid_depth_level2 = sum(o['qty'] for o in bids if o['price'] == second_bid_price)
    
    ask_depth_level2 = 0
    if len(ask_prices) >= 2:
        second_ask_price = ask_prices[1]
        ask_depth_level2 = sum(o['qty'] for o in asks if o['price'] == second_ask_price)
    
    return {
        'ts': snapshot['ts'],
        'seq_no': snapshot['seq_no'],
        'snap_is_crossed': snapshot['snap_is_crossed'],
        'best_bid_price': best_bid_price,
        'best_ask_price': best_ask_price,
        'best_bid_qty': best_bid_qty,
        'best_ask_qty': best_ask_qty,
        'total_bid_qty': total_bid_qty,
        'total_ask_qty': total_ask_qty,
        'num_bid_orders': num_bid_orders,
        'num_ask_orders': num_ask_orders,
        'has_hidden_liquidity': has_hidden,
        'bid_depth_level2': bid_depth_level2,
        'ask_depth_level2': ask_depth_level2,
        'spread': best_ask_price - best_bid_price if best_bid_price and best_ask_price else 0,
    }


def filter_clean_snapshots(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter for clean, consistent snapshots."""
    filtered = []
    last_ts = 0
    for snapshot in snapshots:
        if snapshot['ts'] <= last_ts:
            continue
        
        if snapshot['snap_is_crossed']:
            continue
        
        if snapshot['number_of_orders'] == 0:
            continue
        
        metrics = extract_snapshot_metrics(snapshot)
        
        if metrics['best_bid_price'] <= 0 or metrics['best_ask_price'] <= 0:
            continue
        
        if metrics['best_bid_price'] >= metrics['best_ask_price']:
            continue
        
        if metrics['spread'] <= 0:
            continue
        
        last_ts = snapshot['ts']
        filtered.append(metrics)
    
    return filtered


def align_snapshots_with_imbalance_events(
    snapshots: List[Dict[str, Any]],
    imbalance_events: List[Dict[str, Any]],
    window_ms: float = 1.0
) -> List[Dict[str, Any]]:
    """Align snapshots with imbalance events within ±1ms window."""
    aligned = []
    used_snapshot_indices = set()
    
    window_ns = int(window_ms * 1_000_000)
    
    snap_idx = 0
    for event in imbalance_events:
        event_ts = event['ts']
        event_mid = (event['bid1_nanos'] + event['ask1_nanos']) / 2.0
        
        window_start = event_ts - window_ns
        window_end = event_ts + window_ns
        
        while snap_idx < len(snapshots) and snapshots[snap_idx]['ts'] < window_start:
            snap_idx += 1
        
        temp_idx = snap_idx
        closest_snapshot = None
        closest_delta = float('inf')
        closest_idx = -1
        
        while temp_idx < len(snapshots) and snapshots[temp_idx]['ts'] <= window_end:
            if temp_idx in used_snapshot_indices:
                temp_idx += 1
                continue
            
            delta = abs(snapshots[temp_idx]['ts'] - event_ts)
            if delta < closest_delta:
                closest_delta = delta
                closest_snapshot = snapshots[temp_idx].copy()
                closest_idx = temp_idx
            
            temp_idx += 1
        
        if closest_snapshot is not None:
            closest_snapshot['event_ts'] = event_ts
            closest_snapshot['time_delta_ns'] = closest_snapshot['ts'] - event_ts
            closest_snapshot['time_delta_ms'] = closest_snapshot['time_delta_ns'] / 1_000_000.0
            
            closest_snapshot['event_obi1'] = event.get('obi1', 0.0)
            closest_snapshot['event_obi3'] = event.get('obi3', 0.0)
            closest_snapshot['event_mid_price'] = event_mid
            
            snapshot_mid = (closest_snapshot['best_bid_price'] + closest_snapshot['best_ask_price']) / 2.0
            closest_snapshot['mid_price_diff'] = snapshot_mid - event_mid
            
            bid_qty = closest_snapshot['best_bid_qty']
            ask_qty = closest_snapshot['best_ask_qty']
            if bid_qty + ask_qty > 0:
                closest_snapshot['queue_imbalance'] = (bid_qty - ask_qty) / (bid_qty + ask_qty)
            else:
                closest_snapshot['queue_imbalance'] = 0.0
            
            aligned.append(closest_snapshot)
            used_snapshot_indices.add(closest_idx)
    
    return aligned


def calculate_snapshot_statistics(aligned_snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics from aligned snapshots."""
    if not aligned_snapshots:
        return {}
    
    hidden_liquidity_count = sum(1 for s in aligned_snapshots if s['has_hidden_liquidity'])
    
    avg_best_bid_qty = np.mean([s['best_bid_qty'] for s in aligned_snapshots])
    avg_best_ask_qty = np.mean([s['best_ask_qty'] for s in aligned_snapshots])
    avg_spread = np.mean([s['spread'] for s in aligned_snapshots])
    avg_queue_imbalance = np.mean([s['queue_imbalance'] for s in aligned_snapshots])
    
    avg_num_bid_orders = np.mean([s['num_bid_orders'] for s in aligned_snapshots])
    avg_num_ask_orders = np.mean([s['num_ask_orders'] for s in aligned_snapshots])
    
    return {
        'total_snapshots': len(aligned_snapshots),
        'hidden_liquidity_count': hidden_liquidity_count,
        'hidden_liquidity_pct': hidden_liquidity_count / len(aligned_snapshots) * 100,
        'avg_best_bid_qty': avg_best_bid_qty,
        'avg_best_ask_qty': avg_best_ask_qty,
        'avg_spread': avg_spread,
        'avg_queue_imbalance': avg_queue_imbalance,
        'avg_num_bid_orders': avg_num_bid_orders,
        'avg_num_ask_orders': avg_num_ask_orders,
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
        print("Usage: python snapshots_filtering.py <snapshots_bin_file> <imbalance_events_csv_file> <output_csv_file>")
        print("Example: python snapshots_filtering.py snapshots.bin events.csv output.csv")
        sys.exit(1)
    
    snapshots_file = sys.argv[1]
    events_file = sys.argv[2]
    output_file = sys.argv[3]
    
    if not Path(snapshots_file).exists():
        print(f"Error: Input file '{snapshots_file}' does not exist")
        sys.exit(1)
    
    if not Path(events_file).exists():
        print(f"Error: Events file '{events_file}' does not exist")
        sys.exit(1)
    
    print(f"\n1. Reading snapshots binary file: {snapshots_file}")
    snapshots = read_snapshots_bin_file(snapshots_file)
    print(f"Total snapshots read: {len(snapshots)}")
    
    print("\n2. Applying clean snapshot filters...")
    clean_snapshots = filter_clean_snapshots(snapshots)
    print(f"Clean snapshots: {len(clean_snapshots)}")
    
    if len(clean_snapshots) == 0:
        print("No clean snapshots found. Exiting.")
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
    
    print("\n4. Aligning snapshots with imbalance events (±1ms window)...")
    aligned_snapshots = align_snapshots_with_imbalance_events(
        clean_snapshots,
        imbalance_events,
        window_ms=1.0
    )
    print(f"Aligned snapshots: {len(aligned_snapshots)}")
    
    print("\n5. Computing snapshot statistics...")
    stats = calculate_snapshot_statistics(aligned_snapshots)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    write_csv(aligned_snapshots, output_file)
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()