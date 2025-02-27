# SOMA
This repository contains the code for "SoMa: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators". Please follow the instructions in the AE Appendix of our corresponding HPCA 2025 paper.

## Quick Start

1. First you will need to install some python libs: `pip install -r requirements.txt` or `conda install --file requirements.txt`
2. Optional: we use OpenMP to do multi-threaded search, if you do not want this, just comment out `-fopenmp` in the `Makefile`.

```shell
./build.sh
./run.sh --eta
./get_results.sh
```

## Console Arguments Explanation

After running `build.sh`, you can execute a single experiment using the command:

```shell
./build/soma 108 2 512 1 8 1 8 4 256 512 849779186 results/dse
```

### Argument Breakdown:
1. **Network (108)**: Specifies the neural network to be used. (Full list below)
2. **Baseline Type (2)**: Must be `2`, as other values are not supported.
3. **Sequence Length (512)**: Relevant for LLMs; ignored for CNNs.
4. **Number of Segments (1)**: Used when a network is too large and needs partitioning for scheduling.
5. **L2 Buffer Size (8 MB)**: Defines the L2 buffer size in megabytes.
6. **Batch Size (1)**: Specifies the number of input samples processed at once.
7. **DRAM Bandwidth Ratio (8)**: Ratio of DRAM bandwidth (GB/s) to computational power (TOPS). Default is `1`.
8. **PE Array Dimension (4)**: Typically ranges from `4` to `16`.
9. **L2 Buffer Bandwidth (256 GB/s)**: Specifies the bandwidth for L2 buffer.
10. **MAC Units per PE (512)**: Determines the number of multiply-accumulate (MAC) units per PE. TOPS is calculated as:
    ```
    TOPS = 2 * mac_num * PE_ARRAY_Dim^2 / 1024
    ```
11. **Random Seed (849779186)**: Used for the random number generator.
12. **Results Folder (`results/dse`)**: Specifies where the experiment results are stored.

### Supported Networks

#### Convolutional Neural Networks (CNNs):
- `0`: Darknet19
- `1`: VGG19
- `2`: ResNet50
- `3`: GoogLeNet
- `4`: ResNet101
- `5`: DenseNet
- `6`: Inception-ResNet-V1
- `7`: GNMT
- `8`: LSTM
- `9`: ZFNet
- `10`: Transformer
- `11`: Transformer Cell
- `12`: PNASNet
- `13`: ResNeXt50
- `14`: ResNet152
- `15`: Transformer Big Cell
- `16`: RetinaNet-ResNet50
- `17`: U-Net
- `18`: RandWire Small
- `19`: RandWire Large

#### Large Language Models (LLMs):
- `101`: GPT-J 6B (Decode)
- `102`: GPT-J 6B (Prefill)
- `103`: LLaMa 2 70B (Decode)
- `104`: LLaMa 2 70B (Prefill)
- `105`: BERT Base
- `106`: BERT Large
- `107`: GPT-2 Small (Decode)
- `108`: GPT-2 Small (Prefill)
- `109`: GPT-2 XL (Decode)
- `110`: GPT-2 XL (Prefill)

For unsupported models, the program will throw an error: `Model not supported.`

## Folder Overview

### `include/`
Contains header files for the project.

### `pyscripts/`
Holds Python scripts used for processing and analyzing experiment results.

### `src/`
Contains the source code (C++) for implementing the SoMa Framework.

> Note on GPT: Please note that for GPT-2, we actually explored only one block, so in the data processing script, the corresponding latency is multiplied by the number of blocks. For GPT-2-small in the DSE experiments, the number of blocks is 12.

## Contact Info
If you encounter any issues during the AE process, please contact:

Jingwei Cai (Tsinghua University) <1148821791@qq.com>  
Xuan Wang (Xi'an Jiaotong University, Institute for Interdisciplinary Information Core Technology) <wangxuanxjtu@163.com>
