import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python Fig7_reproduce.py <path_to_dse_csv> <output_svg_file>")
    sys.exit(1)
file_path = sys.argv[1]
data = pd.read_csv(file_path, delimiter=",", skipinitialspace=True)
data.columns = [col.strip() for col in data.columns]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

buffer_sizes = sorted(data['buffer_size'].unique())
dram_bw_ratios = sorted(data['dram_bandwidth'].unique())
batch_sizes = [1, 4, 16]
network_names = {
    'resnet': 'ResNet-50',
    'GPT_2_small_prefill': 'GPT_2_Prefill'
}
baseline_types = ['baseline', 'ours']

fig, axes = plt.subplots(len(batch_sizes), len(network_names) * len(baseline_types), figsize=(20, 12), sharex=False, sharey=False)
plt.subplots_adjust(wspace=0.3, hspace=0.6, bottom=0.1)

for i, batch_size in enumerate(batch_sizes):
    for j, (network_key, network_name) in enumerate(network_names.items()):
        subset_cocco = data[(data['batch_size'] == batch_size) &
                            (data['network'] == network_key) &
                            (data['baseline_type'] == 'baseline')]
        subset_ours = data[(data['batch_size'] == batch_size) &
                           (data['network'] == network_key) &
                           (data['baseline_type'] == 'ours')]

        value_cocco = subset_cocco.pivot(index='dram_bandwidth', columns='buffer_size', values='s1_real_time')
        value_ours = subset_ours.pivot(index='dram_bandwidth', columns='buffer_size', values='s2_real_time')

        min_value = min(value_cocco.min().min(), value_ours.min().min())
        max_value = max(value_cocco.max().max(), value_ours.max().max())
        scaling_factor = 10 ** np.floor(np.log10(max_value))
        
        value_cocco /= scaling_factor
        value_ours /= scaling_factor
        
        vmin = min_value / scaling_factor
        vmax = max_value / scaling_factor

        for k, (subset, value_column) in enumerate([(subset_cocco, 's1_real_time'), (subset_ours, 's2_real_time')]):
            heatmap_data = subset.pivot(index='dram_bandwidth', columns='buffer_size', values=value_column)
            heatmap_data /= scaling_factor
            
            ax = axes[i, j * len(baseline_types) + k]
            sns.heatmap(heatmap_data, ax=ax, cmap="YlGnBu", annot=True, fmt=".2f", vmin=vmin, vmax=vmax,
                        cbar_kws={'label': f'x{scaling_factor:.0e} (ns)'})
            title_baseline = 'Cocco' if k == 0 else 'Ours'
            ax.text(0.5, -0.24, f"{title_baseline} - {network_name} - Batch size = {batch_size}", 
                    size=15, ha='center', transform=ax.transAxes, fontstyle='italic', color='darkred')
            cbar = ax.collections[0].colorbar

            cbar_ticks = np.linspace(vmin, vmax, 6)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([f'{x:.2f}' for x in cbar_ticks])
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_ylabel(cbar.ax.get_ylabel(), size=15)

            ax.set_xlabel('Buffer Size (MB)', fontsize=13, fontstyle='italic', color='darkblue')
            ax.set_ylabel('DRAM BW (GB/s)', fontsize=13, fontstyle='italic', color='darkblue')
            
            ax.xaxis.set_label_coords(0.5, -0.12)
            ax.yaxis.set_label_coords(-0.13, 0.5)

plt.tight_layout()
plt.savefig(sys.argv[2], format='svg', dpi=1200)

