import os, sys
from decimal import Decimal, getcontext
from math import fsum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io

if len(sys.argv) != 3:
    print("Usage: python get_costs_overall.py <path_to_overall_results_dir> <output_csv> > <statistics_log>")
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
            tops = hw_config['TOPS']
            dram_bw = hw_config['DRAM_BW']
            buffer_size = hw_config['L2_BUFFER_SIZE']/1024/1024
            
            net_config = json_data['Config']['Network']
            network = net_config['name']
            batch_size = net_config['batch_size']
            
            search_config = json_data['Config']['Searcher']
            if int(search_config['baseline_type']) == 0:
                baseline_type = 'baseline'
            elif int(search_config['baseline_type']) == 2: 
                baseline_type = 'ours'
            else:
                assert False, "Unknown baseline type"
            
            result = json_data['Results'][0]
            model_ideal_time = Decimal(str(result['dream_time']))
            s1_ideal_time = Decimal(str(result['ideal_cost']['time']))
            s1_ideal_util = model_ideal_time / s1_ideal_time * Decimal('100')
            s1_total_energy = Decimal(str(result['ideal_cost']['energy']))
            s1_core_energy = Decimal(str(result['ideal_cost_breakdown']['comp_energy']))
            s1_dram_energy = Decimal(str(result['ideal_cost_breakdown']['dram_energy']))
            
            
            s1_results = result['stage1']
            lg_num = Decimal(str(len(s1_results['enc']['layer_group_partition'])))
            slg_num = Decimal(str(len(s1_results['enc']['tile_numbers'])))
            comp_tile_num = Decimal(str(len(s1_results["run_graph"]["COMP_Tile_Info"])))
            dram_tensor_num = 0
            for entry in s1_results["run_graph"]["DRAM_Tensor_Info"]:
                if entry[1]["tensor_access_time"] > 0:
                    dram_tensor_num += 1
            dram_tensor_num = Decimal(str(dram_tensor_num))
            
            s1_real_util = Decimal(str(s1_results['dream_util']))
            s1_avg_buffer_usage = Decimal(str(s1_results['avg_buffer_usage']))/Decimal('1024')/Decimal('1024')
            
            s2_real_util = 0
            s2_avg_buffer_usage = 0
            
            if baseline_type == 'ours':
                s2_results = result['stage2']
                s2_real_util = Decimal(str(s2_results['dream_util']))
                s2_avg_buffer_usage = Decimal(str(s2_results['avg_buffer_usage']))/Decimal('1024')/Decimal('1024')
            
        return [network, baseline_type, buffer_size, batch_size, dram_bw, tops, s1_avg_buffer_usage, s2_avg_buffer_usage, s1_core_energy, s1_dram_energy, s1_total_energy, s1_real_util, s2_real_util, s1_ideal_util, lg_num, slg_num, comp_tile_num, dram_tensor_num]

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

csv_heads = ['network', 'baseline_type', 'buffer_size', 'batch_size', 'dram_bw', 'tops', 's1_avg_buffer_usage', 's2_avg_buffer_usage', 's1_core_energy', 's1_dram_energy', 's1_total_energy', 's1_real_util', 's2_real_util', 's1_ideal_util']
all_heads = csv_heads + ['lg_num', 'slg_num', 'comp_tile_num', 'dram_tensor_num']
csv_writer.writerow(csv_heads)
column_indices = {name: idx for idx, name in enumerate(all_heads)}
column_order = ['baseline_type', 'network', 'tops', 'buffer_size', 'dram_bw', 'batch_size']
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

def cal_mean(data:list):
    if any(isinstance(x, Decimal) for x in data):
        return sum(data) / Decimal(len(data))
    else:
        return fsum(data) / len(data)

baseline_lg_num = [row[column_indices['lg_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'baseline']
ours_lg_num = [row[column_indices['lg_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'ours']
ours_slg_num = [row[column_indices['slg_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'ours']

baseline_comp_tile_num = [row[column_indices['comp_tile_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'baseline']
ours_comp_tile_num = [row[column_indices['comp_tile_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'ours']

baseline_dram_tensor_num = [row[column_indices['dram_tensor_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'baseline']
ours_dram_tensor_num = [row[column_indices['dram_tensor_num']] for row in sorted_data if row[column_indices['baseline_type']] == 'ours']

mean_baseline_lg_num = cal_mean(baseline_lg_num)
mean_ours_lg_num = cal_mean(ours_lg_num)
mean_ours_slg_num = cal_mean(ours_slg_num)
mean_baseline_comp_tile_num = cal_mean(baseline_comp_tile_num)
mean_ours_comp_tile_num = cal_mean(ours_comp_tile_num)
mean_baseline_dram_tensor_num = cal_mean(baseline_dram_tensor_num)
mean_ours_dram_tensor_num = cal_mean(ours_dram_tensor_num)
print(f"Mean LG num (baseline) = {mean_baseline_lg_num.quantize(Decimal('0.0'))}")
print(f"Mean LG num (ours) = {mean_ours_lg_num.quantize(Decimal('0.0'))}")
print(f"Mean SLG num (ours) = {mean_ours_slg_num.quantize(Decimal('0.0'))}")
print(f"Mean Comp Tile num (baseline) = {mean_baseline_comp_tile_num.quantize(Decimal('0'))}")
print(f"Mean Comp Tile num (ours) = {mean_ours_comp_tile_num.quantize(Decimal('0'))}")
print(f"Mean DRAM Tensor num (baseline) = {mean_baseline_dram_tensor_num.quantize(Decimal('0'))}")
print(f"Mean DRAM Tensor num (ours) = {mean_ours_dram_tensor_num.quantize(Decimal('0'))}")
print()

def show_network_stats(network_list:list):
    if network_list == []:
        return
    baseline_data = [row for row in sorted_data if row[column_indices['baseline_type']] == 'baseline' and row[column_indices['network']] in network_list]
    our_data = [row for row in sorted_data if row[column_indices['baseline_type']] == 'ours' and row[column_indices['network']] in network_list]
    if baseline_data == [] or our_data == []:
        print("No baseline data found.")
        return
    
    baseline_s1_real_util = [row[column_indices['s1_real_util']] for row in baseline_data]
    ours_s1_real_util = [row[column_indices['s1_real_util']] for row in our_data]
    ours_s2_real_util = [row[column_indices['s2_real_util']] for row in our_data]
    ours_s1_ideal_util = [row[column_indices['s1_ideal_util']] for row in our_data]

    s1_perf_boost = [ours1/baseline for ours1, baseline in zip(ours_s1_real_util, baseline_s1_real_util)]
    s2_perf_boost = [ours2/ours1 for ours2, ours1 in zip(ours_s2_real_util, ours_s1_real_util)]
    total_perf_boost = [ours2/baseline for ours2, baseline in zip(ours_s2_real_util, baseline_s1_real_util)]
    s2_and_ideal_perf_diff = [abs(ours2-ideal)/ideal*Decimal('100') for ours2, ideal in zip(ours_s2_real_util, ours_s1_ideal_util)]
    if len(network_list) == 8:
        print(f"Mean Stage1 Perf Boost = {cal_mean(s1_perf_boost).quantize(Decimal('0.00'))}x")
        print(f"Mean Stage2 Perf Boost = {cal_mean(s2_perf_boost).quantize(Decimal('0.00'))}x")
    print(f"Mean Total Perf Boost = {cal_mean(total_perf_boost).quantize(Decimal('0.00'))}x")
    if len(network_list) == 8:
        print(f"Mean Stage2 and Ideal Perf Diff = {cal_mean(s2_and_ideal_perf_diff).quantize(Decimal('0.0'))}%")

    baseline_s1_total_energy = [row[column_indices['s1_total_energy']] for row in baseline_data]
    baseline_s1_core_energy = [row[column_indices['s1_core_energy']] for row in baseline_data]
    baseline_s1_dram_energy = [row[column_indices['s1_dram_energy']] for row in baseline_data]
    ours_s1_total_energy = [row[column_indices['s1_total_energy']] for row in our_data]
    ours_s1_core_energy = [row[column_indices['s1_core_energy']] for row in our_data]
    ours_s1_dram_energy = [row[column_indices['s1_dram_energy']] for row in our_data]

    core_energy_reduction = [(Decimal('1') - ours/baseline) * Decimal('100') for ours, baseline in zip(ours_s1_core_energy, baseline_s1_core_energy)]
    dram_energy_reduction = [(Decimal('1') - ours/baseline) * Decimal('100') for ours, baseline in zip(ours_s1_dram_energy, baseline_s1_dram_energy)]
    total_energy_reduction = [(Decimal('1') - ours/baseline) * Decimal('100') for ours, baseline in zip(ours_s1_total_energy, baseline_s1_total_energy)]
    if len(network_list) == 8:
        print(f"Mean Core Energy Reduction = {cal_mean(core_energy_reduction).quantize(Decimal('0.0'))}%")
        print(f"Mean DRAM Energy Reduction = {cal_mean(dram_energy_reduction).quantize(Decimal('0.0'))}%")
    print(f"Mean Total Energy Reduction = {cal_mean(total_energy_reduction).quantize(Decimal('0.0'))}%")
    print()

print("ResNet-50 Boosts:")
show_network_stats(['resnet'])
print("ResNet-101 Boosts:")
show_network_stats(['resnet101'])
print("Inception-ResNet-v1 Boosts:")
show_network_stats(['ires'])
print("RandWire Boosts:")
show_network_stats(['randwire_small'])
print("GPT2-Prefill Boosts:")
show_network_stats(['GPT_2_small_prefill', 'GPT_2_XL_prefill'])
print("GPT2-Decode Boosts:")
show_network_stats(['GPT_2_small_decode', 'GPT_2_XL_decode'])
print("Total Networks Boosts:")
show_network_stats(['resnet', 'resnet101', 'ires', 'randwire_small', 'GPT_2_small_prefill', 'GPT_2_XL_prefill', 'GPT_2_small_decode', 'GPT_2_XL_decode'])

for row in sorted_data:
    sorted_row = [row[column_indices[col]] for col in csv_heads]
    csv_writer.writerow(sorted_row)

csv_string = output.getvalue().strip()
with open(sys.argv[2], 'w') as f:
    f.write(csv_string)
