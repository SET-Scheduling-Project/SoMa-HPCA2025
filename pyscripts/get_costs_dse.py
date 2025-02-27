import os, sys
from decimal import Decimal, getcontext
from math import fsum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io

if len(sys.argv) != 3:
    print("Usage: python get_costs_dse.py <path_to_dse_results_dir> <output_csv_file>")
    sys.exit(1)

getcontext().prec = 64

output = io.StringIO()
csv_writer = csv.writer(output)

result_dir = sys.argv[1]

def extract_results_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            
            hw_config = json_data['Config']['Hardware']
            dram_bw = int(hw_config['DRAM_BW'])
            buffer_size = int(hw_config['L2_BUFFER_SIZE']/1024/1024)
            
            net_config = json_data['Config']['Network']
            network = net_config['name']
            batch_size = int(net_config['batch_size'])
            
            search_config = json_data['Config']['Searcher']
            if int(search_config['baseline_type']) == 0:
                baseline_type = 'baseline'
            elif int(search_config['baseline_type']) == 2: 
                baseline_type = 'ours'
            else:
                assert False, "Unknown baseline type"
            
            result = json_data['Results'][0]
            s1_results = result['stage1']
            s1_real_time = Decimal(str(s1_results['real_cost']['time']))
            
            s2_real_time = 0
            if baseline_type == 'ours':
                s2_results = result['stage2']
                s2_real_time = Decimal(str(s2_results['real_cost']['time']))
                
            # We use a GPT-2-small block for evaluation, while the original model has 12 such blocks, 
            # so the results should scale by a factor of 12.
            if network == 'GPT_2_small_prefill': 
                s1_real_time *= Decimal('12')
                s2_real_time *= Decimal('12')
        return [network, baseline_type, buffer_size, batch_size, dram_bw, s1_real_time, s2_real_time]

    except FileNotFoundError:
        print(f"Error: Log file '{file_path}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to read log file '{file_path}'.")
        return None
    except UnicodeDecodeError:
        print(f"Error: Failed to decode log file '{file_path}' using UTF-8 encoding.")
        return None
    except KeyError:
        print(f"Error: Key error in log file '{file_path}'.")
        return None

def process_file(filename):
    file_path = os.path.join(result_dir, filename)
    data = extract_results_from_json(file_path)
    #print(file_path)
    return data

csv_heads = ['network', 'baseline_type', 'buffer_size', 'batch_size', 'dram_bandwidth', 's1_real_time', 's2_real_time']
csv_writer.writerow(csv_heads)
column_indices = {name: idx for idx, name in enumerate(csv_heads)}
column_order = ['network', 'baseline_type', 'buffer_size', 'batch_size', 'dram_bandwidth']
required_columns_indices = [column_indices[col] for col in column_order]

all_data = []

def parallel_processing():
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, filename): filename for filename in os.listdir(result_dir) if filename.endswith('.json')}
        for future in as_completed(futures):
            if future.result() is None:
                continue
            data_list = future.result()
            all_data.append(data_list)

def serial_processing():
    for filename in os.listdir(result_dir):
        if filename.endswith('.json'):
            data_list = process_file(filename)
            if data_list is None:
                continue
            all_data.append(data_list)

parallel_processing()
# serial_processing() # if line above does not work, use this line instead

sorted_data = sorted(
    all_data,
    key=lambda row: tuple(row[column_indices[col]] for col in column_order)
)

for row in sorted_data:
    sorted_row = [row[column_indices[col]] for col in csv_heads]
    csv_writer.writerow(sorted_row)

csv_string = output.getvalue().strip()
with open(sys.argv[2], 'w') as f:
    f.write(csv_string)
