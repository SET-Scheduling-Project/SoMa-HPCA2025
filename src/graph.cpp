#include "graph.h"
#include <algorithm>
#include <cassert>
#include <math.h>
#include <cstring>
#include <chrono>
#include <queue>
#include <sstream>

constexpr int GRAPH_LOG_LEVEL = 0;
constexpr int LG_LOG_LEVEL = 1;
constexpr int SLG_LOG_LEVEL = 2;
constexpr int SL_LOG_LEVEL = 3;

LayerEngine* Graph::layerMapper = nullptr;
thread_local vol_t Graph::Buffer::MAX_BUFFER_SIZE = 8*1024*1024;
thread_local double Graph::Buffer::STAGE_1_LIMIT_RATIO = 1;
thread_local bool Graph::Buffer::IS_STAGE_1 = true;
thread_local len_t Graph::num_tile_cost_cache_total = 0;
thread_local len_t Graph::num_tile_cost_cache_miss = 0;
thread_local len_t Graph::num_tile_cost_cache_hit = 0;
vector<lid_t /*layer_id*/> Graph::transpose_layer_ids = {};
vector<lid_t /*layer_id*/> Graph::fc_layer_ids = {};
bool Graph::has_prefetch = false;
constexpr int MAX_NODE_CNT = 1024;
#define INSERT_TENSOR_INFO(info, id)            \
    do {                                        \
        tensor_info_to_id.insert({ info, id }); \
        tensor_id_to_info.insert({ id, info }); \
    } while (false)

inline void insertChars(std::ostream& os, int count, string c="  ") {
    for (int i = 0; i < count; ++i) {
        os << c;
    }
}

Graph::Stage1Encoding parseInput(const std::vector<std::string>& input) 
{
    Graph::Stage1Encoding enc;

    // Read Layer order
    std::istringstream layerOrderStream(input[0].substr(input[0].find(":") + 1));
    int id;
    while (layerOrderStream >> id) {
        enc.layer_order_to_id.push_back(id);
    }

    // Read Layer group partition
    std::istringstream layerGroupPartitionStream(input[1].substr(input[1].find("(") + 1, input[1].find(")") - input[1].find("(") - 1));
    while (layerGroupPartitionStream >> id) {
        enc.layer_group_partition.set(id);
        if (layerGroupPartitionStream.peek() == ',') {
            layerGroupPartitionStream.ignore();
        }
    }

    // Read Sub layer group partition
    std::istringstream subLayerGroupPartitionStream(input[2].substr(input[2].find("(") + 1, input[2].find(")") - input[2].find("(") - 1));
    while (subLayerGroupPartitionStream >> id) {
        enc.sub_layer_group_partition.set(id);
        if (subLayerGroupPartitionStream.peek() == ',') {
            subLayerGroupPartitionStream.ignore();
        }
    }

    // Read Tile numbers
    std::istringstream tileNumbersStream(input[3].substr(input[3].find(":") + 1));
    int tileNumber;
    while (tileNumbersStream >> tileNumber) {
        enc.tile_numbers.push_back(tileNumber);
    }

    return enc;
}

std::vector< std::pair<double, Graph::Stage1Encoding> > parseAllInputs(std::istream& in)
{
    std::vector< std::pair<double, Graph::Stage1Encoding> > encodings;
    std::string line;
    std::vector<std::string> buffer;

    while (std::getline(in, line)) {
        buffer.push_back(line);

        if (buffer.size() == 5) {
            string ratio = buffer[0];
            buffer.erase(buffer.begin());
            encodings.push_back(std::make_pair(stod(ratio), parseInput(buffer)));
            buffer.clear();
        }
    }

    return encodings;
}

__uint128_t Graph::get_sum_buffer_usage() const
{
    __uint128_t sum_buffer_usage = 0;
    cycle_t last_tile_start_time = 0;
    for (auto it = buffer.buffer_usage_by_time.begin(); it != buffer.buffer_usage_by_time.end(); it++) { 
        sum_buffer_usage += (it->first - last_tile_start_time) * it->second;
        last_tile_start_time = it->first;
    }
    return sum_buffer_usage;
}

void print_avg_buffer_usage(const Graph& g, const CoreMapper::MapCost& cost)
{
    __uint128_t sum_buffer_usage = g.get_sum_buffer_usage();
    std::cout << "Average Buffer Usage = " << (double)sum_buffer_usage / (double)cost.time / (double)1024 << "KB" << std::endl;
}

inline len_t toIfm(const len_t& x, const len_t& stride, const len_t& kernel) {
    return kernel + (x - 1) * stride;
}
size_t Graph::TensorInfoHash::operator()(const TensorInfo& t) const
{
    std::hash<lid_t> lid_hasher;
    std::hash<len_t> len_hasher;
    std::hash<TensorType> type_hasher;
    std::hash<int> int_hasher;

    size_t seed = 0;
    seed ^= lid_hasher(t.layer_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= len_hasher(t.tile_id) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= type_hasher(t.tensor_type) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= int_hasher(t.source) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

    return seed;
}

bool Graph::Stage1Encoding::operator==(const Graph::Stage1Encoding& other) const {
    return layer_order_to_id == other.layer_order_to_id &&
           layer_group_partition == other.layer_group_partition &&
           sub_layer_group_partition == other.sub_layer_group_partition &&
           tile_numbers == other.tile_numbers;
}

bool Graph::Stage1Encoding_tile_sizes::operator==(const Graph::Stage1Encoding_tile_sizes& other) const {
    return layer_order_to_id == other.layer_order_to_id &&
        layer_group_partition == other.layer_group_partition &&
        sub_layer_group_partition == other.sub_layer_group_partition &&
        tile_sizes == other.tile_sizes;
}

bool Graph::Stage2Encoding::operator==(const Graph::Stage2Encoding& other) const {
    return tensor_times == other.tensor_times &&
           tile_tensor_order == other.tile_tensor_order;
}

std::ostream& operator<<(std::ostream& os, const Graph::Stage1Encoding& enc)
{
    os << "Layer order: ";
    for (auto& i : enc.layer_order_to_id) {
        os << i << " ";
    }
    os << std::endl;
    os << "Layer group partition: " << enc.layer_group_partition << std::endl;
    os << "Sub layer group partition: " << enc.sub_layer_group_partition << std::endl;
    os << "Tile numbers: ";
    for (auto& i : enc.tile_numbers) {
        os << i << " ";
    }
    os << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Graph::Stage2Encoding& enc)
{
    os << "Tensor ID: Tensor Time" << std::endl;
    for (int i = 0; i < enc.tensor_times.size(); i++) {
        os << i << ": " << enc.tensor_times[i] << std::endl;
    }
    os << "Tensor Order: Tensor ID@[Tensor Time]" << std::endl;
    for (int i = 0; i < enc.tile_tensor_order.size(); i++) {
        os << i << ": " << enc.tile_tensor_order[i] << "@" << enc.tensor_times[enc.tile_tensor_order[i]] << std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ErrorType& err_type)
{
    switch (err_type) {
    case ErrorType::SUCCESS:
        os << "SUCCESS";
        break;
    case ErrorType::LAYER_ORDER_DEPENDENCY:
        os << "LAYER_ORDER_DEPENDENCY";
        break;
    case ErrorType::BUFFER_OVERFLOW:
        os << "BUFFER_OVERFLOW";
        break;
    case ErrorType::SUBGRAPH_NOT_FULLY_CONNECTED:
        os << "SUBGRAPH_NOT_FULLY_CONNECTED";
        break;
    case ErrorType::TENSOR_ORDER_DEPENDENCY:
        os << "TENSOR_ORDER_DEPENDENCY";
        break;
    case ErrorType::TILE_NUMBER_TOO_LARGE:
        os << "TILE_NUMBER_TOO_LARGE";
        break;
    default:
        os << "Unknown Error";
        break;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::TensorType& t_type)
{
    switch (t_type) {
    case Graph::TensorType::IFM:
        os << "IFM";
        break;
    case Graph::TensorType::WGT:
        os << "WGT";
        break;
    case Graph::TensorType::OFM:
        os << "OFM";
        break;
    default:
        os << "Unknown";
        break;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::TensorInfoSetType& ti_set_type)
{
    switch (ti_set_type) {
    case Graph::TensorInfoSetType::ONLY_IFMs:
        os << "ONLY_IFMs";
        break;
    case Graph::TensorInfoSetType::ONLY_WGTs:
        os << "ONLY_WGTs";
        break;
    case Graph::TensorInfoSetType::OFM_TO_DRAM:
        os << "OFM_TO_DRAM";
        break;
    case Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs:
        os << "OFM_WITH_LOCAL_IFMs";
        break;
    default:
        os << "Unknown";
        break;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::TensorTime& t_time)
{
    os << "[" << t_time.start_time << "~" << t_time.end_time << "]";
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::TensorInfo& t_info)
{
    os  << "Layer ID: " << t_info.layer_id << ", "
        << "Tile ID: " << t_info.tile_id << ", "
        << "Type: " << t_info.tensor_type << ", "
        << "Source Layer ID: " << t_info.source << ", "
        << "Full Size: " << t_info.get_layer_size(SchNode::tot_batch);
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::Buffer& b)
{
    os  << "Buffer Size: " << b.get_max_buffer_size() << ", "
        << "Buffer Usage: " << b.cur_buffer_usage << std::endl;
    os << "Buffer Utilization: " << b.cur_buffer_usage << " / " << b.get_max_buffer_size() << std::endl;
    os << "Tensors in Buffer: " << std::endl;
    for (auto& i : b.tensors_in_buffer) {
        os  << "Tensor ID: " << i.first << ", "
            << "Size: " << i.second << std::endl;
    }
    os << "Tensor Ready: ";
    for (auto& i : b.tensor_ready) {
        os << "Tensor ID: " << i << std::endl;
    }
    os << "Buffer Usage by Time: " << std::endl;
    for (auto& i : b.buffer_usage_by_time) {
        os  << "Cycle: " << i.first << ", "
            << "Usage: " << i.second << std::endl;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::SubLayer& sl)
{
    insertChars(os, SL_LOG_LEVEL);
    os // << "Layer ID: " << sl.layer_id << ", "
        << "Tile Size: " << sl.tile_size << std::endl; // << ", "
        // << "Delta: (" << sl.delta[0] << ", " << sl.delta[1] << "), "
        // << "Step Num: (" << sl.step_num[0] << ", " << sl.step_num[1] << ")" << std::endl;
    insertChars(os, SL_LOG_LEVEL);
    os << "Local Inputs: ";
    for (auto& i : sl.local_inputs) {
        os << i << " ";
    }
    os << std::endl;
    insertChars(os, SL_LOG_LEVEL);
    os << "Local Outputs: ";
    for (auto& i : sl.local_outputs) {
        os << i << " ";
    }
    os << std::endl;
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::SubLayerGroup& slg)
{
    insertChars(os, SLG_LOG_LEVEL);
    os << "Tile Number: " << slg.tile_number << std::endl;
    insertChars(os, SLG_LOG_LEVEL);
    os << "SLG [Starts, End]: [" << slg.sub_layer_group_start << ", " << slg.sub_layer_group_end << "]" << std::endl;
    insertChars(os, SLG_LOG_LEVEL);
    os << "Input Nodes:" << std::endl;
    insertChars(os, SLG_LOG_LEVEL);
    for (auto& i : slg.dram_ifmaps) {
        os << "Node Id: " << i.first << std::endl << i.second;
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph::LayerGroup& lg)
{
    insertChars(os, LG_LOG_LEVEL);
    os << "LG [Starts, End]: [" << lg.layer_group_start << ", " << lg.layer_group_end << "]" << std::endl;
    insertChars(os, LG_LOG_LEVEL);
    os << "SLG [Starts, End]: [" << lg.slg_idx_start << ", " << lg.slg_idx_end << "]" << std::endl;
    os << std::endl;
    return os;
}
std::ostream& operator<<(std::ostream& os, const Graph g)
{
    os  << std::endl
        << "Order,\t"
        << "Layer ID: " << std::endl;
    for (lid_t i = 0; i < g.layer_order_to_id.size(); ++i) {
        os << i << ",\t" << g.layer_order_to_id[i] << std::endl;
    }
    os << "Layer Groups: " << std::endl;
    for (lid_t i = 0; i < g.layer_groups.size(); i++) {
        auto& lg = g.layer_groups[i];
        os << "Layer Group [" << i << "]; Layers: [";
        for (lid_t j = lg.layer_group_start; j <= lg.layer_group_end; ++j) {
            os << g.layer_order_to_id[j] << ", ";
        }
        os << "]" << std::endl;
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; j++) {
            auto& slg = g.all_slgs[j];
            os << "Sub Layer Group [" << j << "]; Layers: [";
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                os << g.layer_order_to_id[k] << ", ";
            }
            os << "]" << std::endl;
            os << slg;
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const Layer& l = network->getNode(g.layer_order_to_id[k]).layer();
                len_t cur_stride_h, cur_stride_w, cur_kernel_h, cur_kernel_w;
                getStride(l, cur_stride_h, cur_stride_w, cur_kernel_h, cur_kernel_w);
                insertChars(os, SLG_LOG_LEVEL);
                os  << "Layer " << g.layer_order_to_id[k] << ": " << l.get_name() 
                    << " tile_pos: " << g.layer_id_to_tile_pos[g.layer_order_to_id[k]]
                    << "\tStride: [" << cur_stride_h << ", " << cur_stride_w << "]; Kernel:[" << cur_kernel_h << ", " << cur_kernel_w << "]" << std::endl
                    << "Layer Ofmap Shape: " << l.ofmap_shape() << std::endl
                    << "Layer Tot Ifmap Shape: " << l.tot_ifmap_shape() << std::endl
                    << g.all_layers[g.layer_order_to_id[k]];
            }
        }
    }
    // os << "Tensor Times: " << std::endl;
    // for (len_t tt_idx = 0; tt_idx < g.tensor_times.size(); tt_idx++) {
    //     os << "TensorTime " << tt_idx << ": "
    //        << g.tensor_times[tt_idx] << std::endl;
    // }
    os << "Tensor ID to Info: " << std::endl;
    for (auto& i : g.tensor_id_to_info) {
        os << "Tensor ID: " << i.first << ", "
           << "Tensor Info: " << i.second << ", "
            << g.tensor_times[i.first] << ", "
            << "Set Type: " << g.tensor_info_set_types[i.first] << ", "
            << "Tile Size: " << g.tensor_id_to_size.at(i.first) << std::endl;
    }
    os << "Tensor Info to ID: " << std::endl;
    for (auto& i : g.tensor_info_to_id) {
        os  << "Tensor Info: " << i.first << ", "
            << "Tensor ID: " << i.second << ", "
            << g.tensor_times[i.second] << ", "
            << "Set Type: " << g.tensor_info_set_types[i.second] << ", "
            << "Tile Size: " << g.tensor_id_to_size.at(i.second) << std::endl;
    }
    os << "Tile Tensor Order: ";
    for (auto& i : g.tile_tensor_order) {
        // os << i << ", ";
        os << g.tensor_times[i] << ", ";
    }
    os  << std::endl
        << "Buffer: " << std::endl
        << g.buffer;
    std::cout << "Buffer Info: " << std::endl;
    for (auto& i : g.buffer.buffer_usage_by_time) {
        std::cout << "Cycle " << i.first << " Buffer Usage: " << i.second << std::endl;
    }
    // output map<cycle_t, DRAM_Tensor_Info> g.DRAM_Tensor_Info_by_time
    std::cout << "DRAM Tensor Info: " << std::endl;
    for (auto& i : g.DRAM_Tensor_Info_by_time) {
        std::cout << "Cycle " << i.first << " Layer ID : " << i.second.layer_id << "	Tile ID: " << i.second.tile_id;
        std::cout << " Tensor Type: " << i.second.tile_tensor_type << " " << i.second.tensor_access_time << std::endl;
    }
    // output map<cycle_t, COMP_Tile_Info> g.COMP_Tile_Info_by_time
    std::cout << "COMP Tile Info: " << std::endl;
    for (auto& i : g.COMP_Tile_Info_by_time) {
        std::cout << "Cycle " << i.first << " Layer ID : " << i.second.layer_id << "	Tile ID: " << i.second.tile_id;
        std::cout << " Tile Comp Time: " << i.second.tile_comp_time << std::endl;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const Graph::IdealCostResults& icr)
{
    os << "Ideal Comp Time: " << icr.ideal_comp << ", Ideal DRAM Time: " << icr.ideal_dram << std::endl;
    os  << "Comp Energy: " << icr.comp_energy << ", "
        << "UBuf Energy: " << icr.ubuf_energy << ", "
        << "Buffer Energy: " << icr.buffer_energy << ", "
        << "NoC Energy: " << icr.noc_energy << ", "
        << "MAC Energy: " << icr.mac_energy << ", "
        << "DRAM Energy: " << icr.dram_energy << std::endl;
    return os;
}

void Graph::print_graph_result() const
{
    std::cout << "Buffer Info: " << std::endl;
    for (auto& i : buffer.buffer_usage_by_time) {
        std::cout << "Cycle " << i.first << " Buffer Usage: " << i.second << std::endl;
    }
    // output map<cycle_t, DRAM_Tensor_Info> DRAM_Tensor_Info_by_time
    std::cout << "DRAM Tensor Info: " << std::endl;
    for (auto& i : DRAM_Tensor_Info_by_time) {
        std::cout << "Cycle " << i.first << " Layer ID : " << i.second.layer_id << "	Tile ID: " << i.second.tile_id;
        std::cout << " Tensor Type: " << i.second.tile_tensor_type << " " << i.second.tensor_access_time << std::endl;
    }
    // output map<cycle_t, COMP_Tile_Info> COMP_Tile_Info_by_time
    std::cout << "COMP Tile Info: " << std::endl;
    for (auto& i : COMP_Tile_Info_by_time) {
        std::cout << "Cycle " << i.first << " Layer ID : " << i.second.layer_id << "	Tile ID: " << i.second.tile_id;
        std::cout << " Tile Comp Time: " << i.second.tile_comp_time << std::endl;
    }
}

void Graph::print_intensity() const
{
    vector<vol_t> lg_comp_time, lg_dram_time, slg_comp_time, slg_dram_time;
    get_intensity(lg_comp_time, lg_dram_time, slg_comp_time, slg_dram_time);
    std::cout << "LG COMP-DRAM: ";
    for (auto i = 0; i < lg_comp_time.size(); i++) {
        std::cout << lg_comp_time[i] << ", " << lg_dram_time[i] << "; ";
    }
    std::cout << std::endl;
    std::cout << "SLG COMP-DRAM: ";
    for (auto i = 0; i < slg_comp_time.size(); i++) {
        std::cout << slg_comp_time[i] << ", " << slg_dram_time[i] << "; ";
    }
    std::cout << std::endl;
    std::cout << "Tile COMP-DRAM: ";
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        const LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const SubLayerGroup& slg = all_slgs[j];
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                const SubLayer& sl = all_layers[layer_id];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                const auto& ts =  sl.tile_size;
                
                const CoreMapper::MapCost& tile_cost = tile_costs.at(std::make_pair(layer_id, ts));

                vol_t tile0_sum_dram_size = 0;
                vol_t tile1_sum_dram_size = 0;
                if (!node.hasWgtPrevs() && layer.weight_size() > 0) {
                    tile0_sum_dram_size += layer.weight_size();
                }
                // OFM
                const TensorInfo first_info_ofm = { layer_id, 0, TensorType::OFM, 0 };
                auto range = tensor_info_to_id.equal_range(first_info_ofm);
                const len_t& ofm_tid = range.first->second;
                assert(distance(range.first, range.second) == 1);
                if (tensor_info_set_types[ofm_tid] == TensorInfoSetType::OFM_TO_DRAM) {
                    const vol_t ofm_tensor_size = ts.bk * ts.c * ts.h * ts.w;
                    tile1_sum_dram_size += ofm_tensor_size;
                }
                // IFM
                {
                    auto cur_ofm_range = fmap_range(fmap_shape(ts.c, ts.h, ts.w), ts.bk);
                    layer.ofm_to_ifm(cur_ofm_range);
                    tile1_sum_dram_size += cur_ofm_range.size();
                }
                if (node.hasWgtPrevs()) {
                    auto cur_ofm_range = fmap_range(fmap_shape(ts.c, ts.h, ts.w), ts.bk);
                    layer.ofm_to_wgt(cur_ofm_range);
                    tile1_sum_dram_size += cur_ofm_range.size();
                }
                tile0_sum_dram_size += tile1_sum_dram_size;
                std::cout << tile_cost.time << ", " << tile0_sum_dram_size << "; ";
                for (int tile_id = 1; tile_id < slg.tile_number; tile_id++) {
                    std::cout << tile_cost.time << ", " << tile1_sum_dram_size << "; ";
                }
            }
        }
    }
    std::cout << std::endl;
}

void Graph::print_tile_group_info() const
{
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        const LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const SubLayerGroup& slg = all_slgs[j];
            cycle_t tg_sum_comp_time = 0;
            cycle_t tg0_sum_dram_time = 0;
            cycle_t tg1_sum_dram_time = 0;
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                const SubLayer& sl = all_layers[layer_id];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                const auto& ts =  sl.tile_size;
                const CoreMapper::MapCost& tile_cost = tile_costs.at(std::make_pair(layer_id, ts));
                tg_sum_comp_time += tile_cost.time;

                // WGT
                if (!node.hasWgtPrevs() && layer.weight_size() > 0) {
                    tg0_sum_dram_time += DIVCEIL(layer.weight_size(), SchNode::DRAM_bw);
                }
                // OFM
                const TensorInfo first_info_ofm = { layer_id, 0, TensorType::OFM, 0 };
                auto range = tensor_info_to_id.equal_range(first_info_ofm);
                const len_t& ofm_tid = range.first->second;
                assert(distance(range.first, range.second) == 1);
                if (tensor_info_set_types[ofm_tid] == TensorInfoSetType::OFM_TO_DRAM) {
                    const vol_t ofm_tensor_size = ts.bk * ts.c * ts.h * ts.w;
                    tg1_sum_dram_time += DIVCEIL(ofm_tensor_size, SchNode::DRAM_bw);
                }
                // IFM
                {
                    auto cur_ofm_range = fmap_range(fmap_shape(ts.c, ts.h, ts.w), ts.bk);
                    layer.ofm_to_ifm(cur_ofm_range);
                    tg1_sum_dram_time += DIVCEIL(cur_ofm_range.size(), SchNode::DRAM_bw);
                }
                if (node.hasWgtPrevs()) {
                    auto cur_ofm_range = fmap_range(fmap_shape(ts.c, ts.h, ts.w), ts.bk);
                    layer.ofm_to_wgt(cur_ofm_range);
                    tg1_sum_dram_time += DIVCEIL(cur_ofm_range.size(), SchNode::DRAM_bw);
                }
            }
            tg0_sum_dram_time += tg1_sum_dram_time;
            std::cout << "LG " << i << " SLG " << j << ": " << std::endl;
            std::cout << "TG Sum COMP Time: " << tg_sum_comp_time << " TG_0 Sum DRAM Time: " << tg0_sum_dram_time << " TG_rest Sum DRAM Time: " << tg1_sum_dram_time << std::endl;
            for (int tile_id = 0; tile_id < slg.tile_number; tile_id++) {
                const auto& first_layer_id = layer_order_to_id[slg.sub_layer_group_start];
                const auto& first_layer_node = network->getNode(first_layer_id);
                const auto& first_layer = first_layer_node.layer();
                const auto& last_layer_id = layer_order_to_id[slg.sub_layer_group_end];
                const auto& last_layer_node = network->getNode(last_layer_id);
                const auto& last_layer = last_layer_node.layer();
                const auto tile_first_layer_start_cycle = tile_start_cycles.at(std::make_pair(first_layer_id, tile_id));
                const auto tile_last_layer_start_cycle = tile_start_cycles.at(std::make_pair(last_layer_id, tile_id));
                const auto tile_last_layer_end_cycle = tile_last_layer_start_cycle + tile_costs.at(std::make_pair(last_layer_id, all_layers[last_layer_id].tile_size)).time;
                std::cout << "TG " << tile_id << " COMP: [" << tile_first_layer_start_cycle << " ~ " << tile_last_layer_end_cycle << "] = " << tile_last_layer_end_cycle - tile_first_layer_start_cycle << " cycles" << std::endl;

                // check tile first layer IFM/WGT start cycle
                cycle_t tg_dram_start_cycle = tile_first_layer_start_cycle;
                if (!first_layer_node.hasWgtPrevs() && first_layer.weight_size() > 0) {
                    const auto& info_wgt = TensorInfo{ first_layer_id, tile_id, TensorType::WGT, 0 };
                    auto range = tensor_info_to_id.equal_range(info_wgt);
                    assert(distance(range.first, range.second) == 1);
                    const len_t& wgt_tid = range.first->second;
                    tg_dram_start_cycle = MIN(tg_dram_start_cycle, dram_tensor_start_cycles.at(wgt_tid));
                }
                FOR_BITSET(ifm_id, first_layer_node.getPrevs()) {
                    if (layer_id_to_order[ifm_id] < lg.layer_group_start) { // IFM from DRAM
                        const auto& info_ifm = TensorInfo{ first_layer_id, tile_id, TensorType::IFM, ifm_id };
                        auto range = tensor_info_to_id.equal_range(info_ifm);
                        assert(distance(range.first, range.second) == 1);
                        const len_t& ifm_tid = range.first->second;
                        tg_dram_start_cycle = MIN(tg_dram_start_cycle, dram_tensor_start_cycles.at(ifm_tid));
                    }
                }
                FOR_BITSET(ext_id, first_layer_node.getExtPrevs()) {
                    const auto& info_ifm = TensorInfo{ first_layer_id, tile_id, TensorType::IFM, -ext_id - 1 };
                    auto range = tensor_info_to_id.equal_range(info_ifm);
                    assert(distance(range.first, range.second) == 1);
                    const len_t& ifm_tid = range.first->second;
                    tg_dram_start_cycle = MIN(tg_dram_start_cycle, dram_tensor_start_cycles.at(ifm_tid));
                }
                cycle_t tg_dram_end_cycle = tile_last_layer_end_cycle;
                {
                    const auto& info_ofm = TensorInfo{ last_layer_id, tile_id, TensorType::OFM, 0 };
                    auto range = tensor_info_to_id.equal_range(info_ofm);
                    assert(distance(range.first, range.second) == 1);
                    const len_t& ofm_tid = range.first->second;
                    if (tensor_info_set_types.at(ofm_tid) == TensorInfoSetType::OFM_TO_DRAM) {
                        tg_dram_end_cycle = MIN(tg_dram_end_cycle, dram_tensor_start_cycles.at(ofm_tid) + DIVCEIL(tensor_id_to_size.at(ofm_tid), SchNode::DRAM_bw));
                    }
                }
                std::cout << "TG " << tile_id << " DRAM: [" << tg_dram_start_cycle << " ~ " << tg_dram_end_cycle << "] = " << tg_dram_end_cycle - tg_dram_start_cycle << " cycles" << std::endl;
            }
        }
    }
}

Graph::TensorTime::TensorTime(const int& start, const int& end)
{
    start_time = MAX(-1, start);
    end_time = end;
}

bool Graph::TensorTime::operator<(const TensorTime& other) const
{
    return (start_time != other.start_time) ? (start_time < other.start_time) : (end_time < other.end_time);
}

bool Graph::TensorTime::operator==(const TensorTime& other) const
{
    return start_time == other.start_time && end_time == other.end_time;
}

void Graph::TensorTime::expand(const int& start, const int& end)
{
    start_time = MAX(-1, MIN(start, start_time));
    end_time = MAX(end_time, end);
}

bool Graph::TensorInfo::operator<(const TensorInfo& b) const
{
    if (layer_id != b.layer_id) { return layer_id < b.layer_id;
    } else if (tile_id != b.tile_id) { return tile_id < b.tile_id;
    } else if (tensor_type != b.tensor_type) { return tensor_type < b.tensor_type;
    } else { return source < b.source; }
    assert(false && "Completely Identical TensorInfo Found!");
    return false;
}
bool Graph::TensorInfo::operator==(const TensorInfo& b) const
{
    return layer_id == b.layer_id && tile_id == b.tile_id && tensor_type == b.tensor_type && source == b.source;
}

vol_t Graph::TensorInfo::get_layer_size(const len_t& bk) const
{
    const Layer& l = network->getNode(layer_id).layer();
    switch (tensor_type) {
    case TensorType::IFM:
        return l.tot_ifmap_shape().tot_size(bk);
    case TensorType::WGT:
        return l.weight_size();
    case TensorType::OFM:
        return l.ofmap_shape().tot_size(bk);
    default:
        assert(false && "tensor_type invald!");
        return 0;
    }
}

bool Graph::Buffer::set_tensor_ready(const len_t& tensor_id)
{
    assert(buffer_has(tensor_id));
    tensor_ready.emplace(tensor_id);
    return true;
}
bool Graph::Buffer::ask_tensor_ready(const len_t& tensor_id) const
{
    return tensor_ready.find(tensor_id) != tensor_ready.end();
}
bool Graph::Buffer::buffer_has(const len_t& tensor_id) const
{
    return tensors_in_buffer.find(tensor_id) != tensors_in_buffer.end();
}
bool Graph::Buffer::buffer_add(const len_t& tensor_id, const vol_t& tensor_size)
{
    assert(!buffer_has(tensor_id));
    if (cur_buffer_usage + tensor_size > get_max_buffer_size()) {
        return false;
    }
    // emplace is faster than tensors_in_buffer[pair(tensor_id, tensor_type)] = tensor_size;
    tensors_in_buffer[tensor_id] = tensor_size;
    cur_buffer_usage += tensor_size;
    max_buffer_usage = MAX(max_buffer_usage, cur_buffer_usage);
    return true;
}
bool Graph::Buffer::buffer_del(const len_t& tensor_id)
{
    assert(buffer_has(tensor_id));
    cur_buffer_usage -= tensors_in_buffer[tensor_id];
    tensor_ready.erase(tensor_id);
    tensors_in_buffer.erase(tensor_id);
    return true;
}

bool Graph::Buffer::buffer_del(const list<len_t>& tensor_ids, len_t& del_size)
{
    // delete tile sizes in list until del_size is reached
    for (auto& i : tensor_ids) {
        assert(buffer_has(i));
        if (del_size <= tensors_in_buffer[i]) {
            cur_buffer_usage -= del_size;
            tensors_in_buffer[i] -= del_size; // still ready, just tile size reduced
            del_size = 0;
            break;
        } else {
            cur_buffer_usage -= tensors_in_buffer[i];
            del_size -= tensors_in_buffer[i];
            tensor_ready.erase(i);
            tensors_in_buffer.erase(i);
        }
    }
    return true;
}

void Graph::Buffer::clear()
{
    cur_buffer_usage = 0;
    max_buffer_usage = 0;
    tensors_in_buffer.clear();
    tensor_ready.clear();
    buffer_usage_by_time.clear();
}

#ifdef DEBUG
vol_t Graph::get_tensor_size(const len_t& tensor_id) const
{
    auto range = tensor_id_to_info.equal_range(tensor_id);
    vol_t tensor_size = 0;
    for (auto info_it = range.first; info_it != range.second; ++info_it) {
        // if there is 1 (and only 1) OFM, use the size of that OFM;
        // if there is NO OFM, but there is 1 (and only 1) WGT, then there shall be NO IFM and !hasWgtPrevs(), use the size of that WGT;
        // if there is NO OFM and NO WGT, which means IFM from DRAM, use the size of the IFM;
        if (info_it->second.tensor_type == TensorType::OFM) {
            const auto& ts = all_layers[info_it->second.layer_id].tile_size;
            tensor_size = ts.bk * ts.c * ts.h * ts.w;
            break;
        } else if (info_it->second.tensor_type == TensorType::WGT) { // CNN WGT
            const Node& node = network->getNode(info_it->second.layer_id);
            const Layer& layer = node.layer();
            assert(!node.hasWgtPrevs() && layer.weight_size() > 0); // See IFM part for Transformer WGT
            tensor_size = layer.weight_size();
            break;
        } else if (info_it->second.tensor_type == TensorType::IFM) {
            const auto& sl = all_layers[info_it->second.layer_id];
            // EXT DATA or IFM or Transformer WGT
            // Wrong: tensor_size = network->getExtInputs()[-info_it->second.source - 1].get_shape().tot_size(sl.tile_size.bk);
            // Because this calculation is based on the tile size of the layer, not the tile size of the tensor
            const Node& node = network->getNode(info_it->second.layer_id);
            const Layer& layer = node.layer();
            auto cur_ofm_range = fmap_range(fmap_shape(sl.tile_size.c, sl.tile_size.h, sl.tile_size.w), sl.tile_size.bk);
            if (node.hasWgtPrevs() && info_it->second.source >= 0 && node.getWgtPrevs().contains(info_it->second.source)) {
                layer.ofm_to_wgt(cur_ofm_range);
            } else {
                layer.ofm_to_ifm(cur_ofm_range);
            }
            auto input_node_shape = info_it->second.source >= 0 ? 
                                        network->getNode(info_it->second.source).layer().ofmap_shape() :
                                        network->getExtInputs()[-info_it->second.source - 1].get_shape();
            cur_ofm_range.c = {0, input_node_shape.c};
            tensor_size = cur_ofm_range.size();
        } else { assert(false && "TensorType invalid!"); }
    }
    return tensor_size;
}
#endif

bool Graph::check_layer_order_valid() const
{
    for (lid_t k = 0; k < network->len(); ++k) {
        const lid_t& layer_id = layer_order_to_id[k];
        const Node& node = network->getNode(layer_id);
        FOR_BITSET(ifm_or_wgt_id, node.getPrevs()) {
            if (layer_id_to_order[ifm_or_wgt_id] >= k)
                return false;
        }
        FOR_BITSET(ofm_id, node.get_nexts()) {
            if (layer_id_to_order[ofm_id] <= k)
                return false;
        }
    }
    return true;
}

bool Graph::cut_into_tiles(tensor_shape& tile_size, const len_t& tile_number, SLGTransposeType avoid_dims)
{
    auto tn = tile_number;
    // first cut as many N as possible
    int gcd = getGCD(tile_size.bk, tn);
    tn /= gcd;
    int factor_h, factor_w;

    if (avoid_dims == SLGTransposeType::CH_AND_CW) {
        // we can only cut N
        factor_h = 1, factor_w = 1;
    } else if (avoid_dims == SLGTransposeType::ONLY_CH) {
        // we can only cut N and W
        factor_h = 1, factor_w = tn;
    } else if (avoid_dims == SLGTransposeType::ONLY_CW) {
        // we can only cut N and H
        factor_h = tn, factor_w = 1;
    } else if (avoid_dims == SLGTransposeType::NONE) {
        // next cut H & W to similar size
        // divide tn into two factors which are as close as possible in O(1)
        double alpha = (double)tile_size.h/(double)tile_size.w;
        double ideal_best_w = sqrt((double)tn / alpha);
        int w_down = static_cast<int>(ideal_best_w);
        int w_up = w_down + 1;
        while (w_down > 0 && tn % w_down != 0) { w_down--; }
        while (w_up <= tn && tn % w_up != 0) { w_up++; }
        double distance_down = (w_down > 0) ? std::abs(static_cast<double>(tn / w_down) - alpha * w_down) : std::numeric_limits<double>::infinity();
        double distance_up = (w_up <= tn) ? std::abs(static_cast<double>(tn / w_up) - alpha * w_up) : std::numeric_limits<double>::infinity();
        
        if (distance_down < distance_up) {
            factor_h = tn / w_down, factor_w = w_down;
        } else {
            factor_h = tn / w_up, factor_w = w_up;
        }
    } else {
        assert(false && "SLGTransposeType invalid!");
    }

    if (tile_size.h < factor_h || tile_size.w < factor_w) {
        return false; // report error if the tile is too small to cut
    }
    tile_size.bk = tile_size.bk / gcd;
    tile_size.h = DIVCEIL(tile_size.h, factor_h);
    tile_size.w = DIVCEIL(tile_size.w, factor_w);
    
    return true;
}

ErrorType Graph::init_stage_1_order(const Graph::Stage1Encoding& enc)
{
#ifdef DEBUG
    std::cout << "Init Order" << std::endl;
#endif
    // 0. Clear some data
    layer_order_to_id.clear();
    layer_id_to_order.clear();
    layer_id_to_lg_slg.clear();
    all_layers.clear();
    // dirp_set.clear();
    // 1. Init layer order
    const lid_t num_layer = network->len();
    layer_order_to_id = enc.layer_order_to_id;
    layer_id_to_order.resize(num_layer);
    for (lid_t i = 0; i < num_layer; ++i) { // here i means order
        layer_id_to_order[layer_order_to_id[i]] = i;
    }
    // 2. Init all_layers
    all_layers.resize(num_layer);
    // dirp_set.resize(num_layer);
    for (lid_t i = 0; i < num_layer; ++i) { // here i means id
        auto& al = all_layers[i];
        // al.delta[0] = al.delta[1] = 1;
        // al.step_num[0] = al.step_num[1] = 1;
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_order_with_tile_sizes(const Stage1Encoding_tile_sizes& enc_tz)
{
#ifdef DEBUG
    std::cout << "Init Order" << std::endl;
#endif
    // 0. Clear some data
    layer_order_to_id.clear();
    layer_id_to_order.clear();
    all_layers.clear();
    // dirp_set.clear();
    // 1. Init layer order
    const lid_t num_layer = network->len();
    layer_order_to_id = enc_tz.layer_order_to_id;
    layer_id_to_order.resize(num_layer);
    for (lid_t i = 0; i < num_layer; ++i) { // here i means order
        layer_id_to_order[layer_order_to_id[i]] = i;
    }
    // 2. Init all_layers
    all_layers.resize(num_layer);
    // dirp_set.resize(num_layer);
    for (lid_t i = 0; i < num_layer; ++i) { // here i means id
        auto& al = all_layers[i];
        al.tile_size = enc_tz.tile_sizes[layer_id_to_order[i]];
        // al.delta[0] = al.delta[1] = 1;
        // al.step_num[0] = al.step_num[1] = 1;
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_partition(const Graph::Stage1Encoding& enc)
{
#ifdef DEBUG
    std::cout << "Init Partition" << std::endl;
#endif
    // 0. Clear some data
    layer_groups.clear();
    all_slgs.clear();
    layer_id_to_lg_slg.clear();

    // 3. Init layer_groups & sub_layer_groups
    // START is INCLUDED, but END is NOT INCLUDED
    const lid_t num_layer = network->len();
    const lid_t numLayerGroups = enc.layer_group_partition.count();
    layer_groups.reserve(numLayerGroups);
    all_slgs.resize(enc.tile_numbers.size());
    layer_id_to_lg_slg.resize(num_layer);
    lid_t lg_idx = 0;
    lid_t slg_idx = 0;
    for (lid_t i = 0; i < numLayerGroups; ++i) {
        LayerGroup lg;
        lid_t next_lg_idx = MIN(enc.layer_group_partition.next(lg_idx), num_layer);
        lg.layer_group_start = lg_idx;
        lg.layer_group_end = next_lg_idx - 1;
        assert(lg.layer_group_end >= lg.layer_group_start);
        lg.slg_idx_start = slg_idx;
        for (lid_t j = lg_idx; j < next_lg_idx; j = enc.sub_layer_group_partition.next(j)) {
            SubLayerGroup slg;
            slg.tile_number = enc.tile_numbers[slg_idx];
            slg.sub_layer_group_start = j;
            slg.sub_layer_group_end = MIN(next_lg_idx, enc.sub_layer_group_partition.next(j)) - 1;
            assert(slg.sub_layer_group_end >= slg.sub_layer_group_start);
            SLGTransposeType slg_tr_type = SLGTransposeType::NONE;
            // we do not need to consider Tranpose Layers whose prevs are all not in this slg
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                if (slg_tr_type == SLGTransposeType::CH_AND_CW)
                    break;
                const lid_t& layer_id = layer_order_to_id[k];
                if (std::binary_search(Graph::transpose_layer_ids.begin(), Graph::transpose_layer_ids.end(), layer_id)) {
                    bool prevs_in_slg = false;
                    FOR_BITSET(tr_p, network->getNode(layer_id).getPrevs())
                    {
                        if (layer_id_to_order[tr_p] >= slg.sub_layer_group_start) {
                            prevs_in_slg = true;
                            break;
                        }
                    }
                    if (!prevs_in_slg) {
                        continue;
                    }
                    const Node& node = network->getNode(layer_id);
                    const Layer& l = node.layer();
                    const TransposeLayer& tl = static_cast<const TransposeLayer &>(l);
                    // get to know the transpose type
                    const TransposeLayer::dim* tr_order = get_transpose_order(tl);
                    if (tr_order[TransposeLayer::dim::C] == TransposeLayer::dim::H) {
                        assert(tr_order[TransposeLayer::dim::H] == TransposeLayer::dim::C);
                        slg_tr_type = (slg_tr_type == SLGTransposeType::ONLY_CW) ? SLGTransposeType::CH_AND_CW : SLGTransposeType::ONLY_CH;
                    } else if (tr_order[TransposeLayer::dim::C] == TransposeLayer::dim::W) {
                        assert(tr_order[TransposeLayer::dim::W] == TransposeLayer::dim::C);
                        slg_tr_type = (slg_tr_type == SLGTransposeType::ONLY_CH) ? SLGTransposeType::CH_AND_CW : SLGTransposeType::ONLY_CW;
                    } else {
                        assert(tr_order[TransposeLayer::dim::H] == TransposeLayer::dim::W);
                        assert(tr_order[TransposeLayer::dim::W] == TransposeLayer::dim::H);
                    }
                }
            }
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                layer_id_to_lg_slg[layer_id] = make_pair(i, slg_idx);
                SubLayer& sl = all_layers[layer_id];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                if (node.hasOrderPrevs()) {
                    FOR_BITSET(order_id, node.getOrderPrevs())
                    {
                        if (layer_id_to_order[order_id] >= k)
                            return ErrorType::LAYER_ORDER_DEPENDENCY;
                    }
                }
                // if (node.hasExtPrevs()) {
                //     lg.input_idx.insert(layer_id);
                // }
                // dirp_set[layer_id].clear();
                FOR_BITSET(ext_id, node.getExtPrevs())
                {
                    if (slg.dram_ifmaps.count(-ext_id - 1) == 0) {
                        // auto ext_input_shape = network->getExtInputs().at(ext_id).get_shape();
                        // tensor_shape ext_tensor_shape = { SchNode::tot_batch, ext_input_shape.c, ext_input_shape.h, ext_input_shape.w };
                        tensor_shape ext_tensor_shape = { 0, 0, 0, 0 };
                        slg.dram_ifmaps[-ext_id - 1] = { ext_tensor_shape, set<int>(), set<int>() };
                    }
                    slg.dram_ifmaps[-ext_id - 1].local_outputs.insert(layer_id);
                    sl.local_inputs.insert(-ext_id - 1);
                }
                FOR_BITSET(ifm_or_wgt_id, node.getPrevs())
                {
                    if (layer_id_to_order[ifm_or_wgt_id] >= k) // k <= slg.sub_layer_group_end <= lg.layer_group_end
                        return ErrorType::LAYER_ORDER_DEPENDENCY;
                    // if (layer_id_to_order[ifm_or_wgt_id] >= lg.layer_group_start) // input IN local LG
                    //     dirp_set[layer_id].set(ifm_or_wgt_id);
                    if (layer_id_to_order[ifm_or_wgt_id] < slg.sub_layer_group_start) { // input NOT IN local SLG
                        if (slg.dram_ifmaps.count(ifm_or_wgt_id) == 0) {
                            // auto ifm_or_wgt_shape = network->getNode(ifm_or_wgt_id).layer().ofmap_shape();
                            // tensor_shape ifm_or_wgt_tile_size = { SchNode::tot_batch, ifm_or_wgt_shape.c, ifm_or_wgt_shape.h, ifm_or_wgt_shape.w };
                            tensor_shape ifm_or_wgt_tile_size = { 0, 0, 0, 0 };
                            slg.dram_ifmaps[ifm_or_wgt_id] = { ifm_or_wgt_tile_size, set<int>(), set<int>() };
                        }
                        slg.dram_ifmaps[ifm_or_wgt_id].local_outputs.insert(layer_id);
                    } else { // input IN local SLG
                        sl.local_inputs.insert(ifm_or_wgt_id);
                    }
                }
                FOR_BITSET(ofm_id, node.get_nexts())
                {
                    if (layer_id_to_order[ofm_id] <= k)  // k >= slg.sub_layer_group_start >= lg.layer_group_start
                        return ErrorType::LAYER_ORDER_DEPENDENCY;
                    if (layer_id_to_order[ofm_id] <= slg.sub_layer_group_end) // output IN local SLG
                        sl.local_outputs.insert(ofm_id);
                }
                { // init tile_sizes for pure output layers
                    if (sl.local_outputs.empty()) {
                        slg.pure_outputs_id.emplace_back(layer_id);
                        const auto ofm_shape = layer.ofmap_shape();
                        tensor_shape out_tensor_shape = { SchNode::tot_batch, ofm_shape.c, ofm_shape.h, ofm_shape.w };
                        bool cut_success = cut_into_tiles(out_tensor_shape, slg.tile_number, slg_tr_type);
                        if (!cut_success && sl.local_outputs.empty())
                            return ErrorType::TILE_NUMBER_TOO_LARGE;
                        sl.tile_size = out_tensor_shape;
                    } else {
                        sl.tile_size = { 0, 0, 0, 0 };
                    }
                }
            }
            all_slgs[slg_idx++] = slg;
        }
        lg.slg_idx_end = slg_idx - 1;
        assert(lg.slg_idx_end >= lg.slg_idx_start);
        layer_groups.push_back(lg);
        lg_idx = next_lg_idx;
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_partition_with_tile_sizes(const Stage1Encoding_tile_sizes& enc_tz)
{
#ifdef DEBUG
    std::cout << "Init Partition" << std::endl;
#endif
    // 0. Clear some data
    layer_groups.clear();
    all_slgs.clear();
    layer_id_to_lg_slg.clear();

    // 3. Init layer_groups & sub_layer_groups
    // START is INCLUDED, but END is NOT INCLUDED
    const lid_t num_layer = network->len();
    const lid_t numLayerGroups = enc_tz.layer_group_partition.count();
    layer_groups.reserve(numLayerGroups);
    all_slgs.resize((enc_tz.layer_group_partition | enc_tz.sub_layer_group_partition).count());
    layer_id_to_lg_slg.resize(num_layer);
    lid_t lg_idx = 0;
    lid_t slg_idx = 0;
    for (lid_t i = 0; i < numLayerGroups; ++i) {
        LayerGroup lg;
        lid_t next_lg_idx = MIN(enc_tz.layer_group_partition.next(lg_idx), num_layer);
        lg.layer_group_start = lg_idx;
        lg.layer_group_end = next_lg_idx - 1;
        lg.slg_idx_start = slg_idx;
        for (lid_t j = lg_idx; j < next_lg_idx; j = enc_tz.sub_layer_group_partition.next(j)) {
            SubLayerGroup slg;
            slg.sub_layer_group_start = j;
            slg.sub_layer_group_end = MIN(next_lg_idx, enc_tz.sub_layer_group_partition.next(j)) - 1;
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                layer_id_to_lg_slg[layer_id] = make_pair(i, slg_idx);
                SubLayer& sl = all_layers[layer_id];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                // if (node.hasExtPrevs()) {
                //     lg.input_idx.insert(layer_id);
                // }
                // dirp_set[layer_id].clear();
                FOR_BITSET(ext_id, node.getExtPrevs())
                {
                    if (slg.dram_ifmaps.count(-ext_id - 1) == 0) {
                        // auto ext_input_shape = network->getExtInputs()[ext_id].get_shape();
                        // tensor_shape ext_tensor_shape = { SchNode::tot_batch, ext_input_shape.c, ext_input_shape.h, ext_input_shape.w };
                        tensor_shape ext_tensor_shape = { 0, 0, 0, 0 };
                        slg.dram_ifmaps[-ext_id - 1] = { ext_tensor_shape, set<int>(), set<int>() };
                    }
                    slg.dram_ifmaps[-ext_id - 1].local_outputs.insert(layer_id);
                    sl.local_inputs.insert(-ext_id - 1);
                }
                FOR_BITSET(ifm_or_wgt_id, node.getPrevs())
                {
                    if (layer_id_to_order[ifm_or_wgt_id] >= k) // k <= slg.sub_layer_group_end <= lg.layer_group_end
                        return ErrorType::LAYER_ORDER_DEPENDENCY;
                    // if (layer_id_to_order[ifm_or_wgt_id] >= lg.layer_group_start) // input IN local LG
                    //    dirp_set[layer_id].set(ifm_or_wgt_id);
                    if (layer_id_to_order[ifm_or_wgt_id] < slg.sub_layer_group_start) { // input NOT IN local SLG
                        if (slg.dram_ifmaps.count(ifm_or_wgt_id) == 0) {
                            // slg.dram_ifmaps[ifm_or_wgt_id] = { enc_tz.tile_sizes[ifm_or_wgt_id], set<int>(), set<int>() };
                            slg.dram_ifmaps[ifm_or_wgt_id] = { {0, 0, 0, 0}, set<int>(), set<int>() };
                        }
                        slg.dram_ifmaps[ifm_or_wgt_id].local_outputs.insert(layer_id);
                    } else { // input IN local SLG
                        sl.local_inputs.insert(ifm_or_wgt_id);
                    }
                }
                FOR_BITSET(ofm_id, node.get_nexts())
                {
                    if (layer_id_to_order[ofm_id] <= k)  // k >= slg.sub_layer_group_start >= lg.layer_group_start
                        return ErrorType::LAYER_ORDER_DEPENDENCY;
                    if (layer_id_to_order[ofm_id] <= slg.sub_layer_group_end) // output IN local SLG
                        sl.local_outputs.insert(ofm_id);
                }
                if (sl.local_outputs.empty()) {
                    slg.pure_outputs_id.emplace_back(layer_id);
                }
            }
            all_slgs[slg_idx++] = slg;
        }
        lg.slg_idx_end = slg_idx - 1;
        layer_groups.push_back(lg);
        lg_idx = next_lg_idx;
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_back_cal()
{
    // 4. Back-Calculation
#ifdef DEBUG
    std::cout << "Backcal started" << std::endl;
#endif
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            SubLayerGroup& slg = all_slgs[j];
            if (!backcalc(slg)) {
                return ErrorType::SUBGRAPH_NOT_FULLY_CONNECTED;
            }
            // int k = slg.sub_layer_group_end;
            // for (; k >= slg.sub_layer_group_start; --k) {
            //     SubLayer& sl = all_layers[layer_order_to_id[k]];
            //     if (sl.local_outputs.empty()) {
            //         const Layer& layer = network->getNode(layer_order_to_id[k]).layer();
            //         slg.tile_number = DIVCEIL(layer.ofmap_shape().h, sl.tile_size.h)
            //                         * DIVCEIL(layer.ofmap_shape().w, sl.tile_size.w)
            //                         * SchNode::tot_batch / sl.tile_size.bk;
            //         break;
            //     }
            // }
            // assert(k >= slg.sub_layer_group_start);
        }
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_back_cal_and_change_tile_sizes()
{
    // 4. Back-Calculation
#ifdef DEBUG
    std::cout << "Backcal started" << std::endl;
#endif
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            SubLayerGroup& slg = all_slgs[j];
            if (!backcalc(slg)) {
                return ErrorType::SUBGRAPH_NOT_FULLY_CONNECTED;
            }
            int k = slg.sub_layer_group_end;
            for (; k >= slg.sub_layer_group_start; --k) {
                SubLayer& sl = all_layers[layer_order_to_id[k]];
                if (sl.local_outputs.empty()) {
                    const Layer& layer = network->getNode(layer_order_to_id[k]).layer();
                    slg.tile_number = DIVCEIL(layer.ofmap_shape().h, sl.tile_size.h)
                                    * DIVCEIL(layer.ofmap_shape().w, sl.tile_size.w)
                                    * SchNode::tot_batch / sl.tile_size.bk;
                    break;
                }
            }
            assert(k >= slg.sub_layer_group_start);
        }
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_tile_pos()
{
#ifdef DEBUG
    std::cout << "Init tile pos" << std::endl;
#endif
    // 0. Clear some data
    layer_id_to_tile_pos.clear();
    const lid_t num_layer = network->len();
    const lid_t numLayerGroups = layer_groups.size();

    tile_costs.reserve(100 * network->len());
    // 5. Init tile_pos after tile_number is known
    layer_id_to_tile_pos.resize(num_layer);
    layer_id_to_tile_pos[layer_order_to_id[0]].start_time = 0;
    for (lid_t i = 0; i < numLayerGroups; ++i) {
        LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            SubLayerGroup& slg = all_slgs[j];
            lid_t slg_len = slg.layer_num(); // slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                if (k != slg.sub_layer_group_end) {
                    layer_id_to_tile_pos[layer_order_to_id[k + 1]].start_time = layer_id_to_tile_pos[layer_id].start_time + 1;
                } else if (k != num_layer - 1) {
                    layer_id_to_tile_pos[layer_order_to_id[k + 1]].start_time = layer_id_to_tile_pos[layer_id].start_time + 1 + (slg.tile_number - 1) * slg_len;
                } else {}
                layer_id_to_tile_pos[layer_id].end_time = layer_id_to_tile_pos[layer_id].start_time + (slg.tile_number - 1) * slg_len;
            }
        }
    }
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1_tensor_times()
{
#ifdef DEBUG
    std::cout << "Init tensor times" << std::endl;
#endif
    // 0. Clear some data
    tensor_times.clear();
    tensor_info_to_id.clear();
    tensor_id_to_info.clear();
    tensor_info_set_types.clear();
    tensor_id_to_size.clear();

    const lid_t numLayerGroups = layer_groups.size();
    // 6. Init tensor_times and tensor_info_set_types
    for (lid_t i = 0; i < numLayerGroups; ++i) {
        LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            SubLayerGroup& slg = all_slgs[j];
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                // for every layer, in a topology order
                const lid_t& layer_id = layer_order_to_id[k];
                SubLayer& sl = all_layers[layer_id];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                
                // IFM & Transformer WGT TensorTimes init
                /*
                    Assumption: every tile of this layer share the same IFM
                    If not, we will have multiple OFM<->IFM relationships, hard to deal with their Ends beecause these OFM's End differs:
                */
                // assert(node.getWgtPrevs().count() <= 1);
                FOR_BITSET(ifm_id, node.getPrevs())
                { // must be ifm_id's OFM
                    assert(layer_id_to_order[ifm_id] < k); // k == layer_id_to_order[layer_id] <= slg.sub_layer_group_end <= lg.layer_group_end
                    if (layer_id_to_order[ifm_id] < lg.layer_group_start) { // input NOT IN local LG
                        /*
                            for layers NOT IN this LG: for each tile create tensor_time tt
                            TODO: support IFM overlap cache, rather than re-loading overlap through DRAM
                            start = MAX{start of the OFM tile, start of the previous tile} = start of the previous tile
                            end = end of the this tile
                            ready = cycle that this tensor's been loaded from DRAM
                            tensor_info_to_id[this tile IFM info, IFM order] = tt_id;
                        */
                        auto cur_ofm_range = fmap_range(fmap_shape(sl.tile_size.c, sl.tile_size.h, sl.tile_size.w), sl.tile_size.bk);
                        if (node.hasWgtPrevs() && node.getWgtPrevs().contains(ifm_id)) {
                            layer.ofm_to_wgt(cur_ofm_range);
                        } else {
                            layer.ofm_to_ifm(cur_ofm_range);
                        }
                        auto input_node_shape = network->getNode(ifm_id).layer().ofmap_shape();
                        cur_ofm_range.c = {0, input_node_shape.c};
                        assert(cur_ofm_range.size() > 0);
                        int pos = layer_id_to_tile_pos[layer_id].start_time;
                        for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                            TensorTime tt_ifm(pos - 1, pos + 1);
                            tensor_id_to_size.emplace_back(cur_ofm_range.size());
                            tensor_times.emplace_back(tt_ifm);
                            tensor_info_set_types.emplace_back(TensorInfoSetType::ONLY_IFMs);
                            TensorInfo info_ifm = { layer_id, tile_id, TensorType::IFM, ifm_id };
                            INSERT_TENSOR_INFO(info_ifm, tensor_times.size() - 1);
                            pos += slg.layer_num(); // slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
                        }
                    } else if (layer_id_to_order[ifm_id] < slg.sub_layer_group_start) { // input NOT IN local SLG
                        /*
                            for layers IN this LG but NOT IN this SLG: is other tile's OFM, should be DONE before, just check
                            for each tile: (Converge OFM with IFMs which generated this OFM)
                            assert(tensor_info_to_id[this tile IFM info, IFM order] == vector(tensor_id of OFM of layer that generated IFM));
                        */
                        auto ifm_it = tensor_info_to_id.find(TensorInfo {layer_id, 0, TensorType::IFM, ifm_id });
                        assert(ifm_it != tensor_info_to_id.end());
                        for (len_t tile_id = 1; tile_id < slg.tile_number; ++tile_id) {
                            TensorInfo info_ifm = { layer_id, tile_id, TensorType::IFM, ifm_id };
                            INSERT_TENSOR_INFO(info_ifm, ifm_it->second);
                        }
                    } else { // input IN local SLG
                        /*
                            for layers IN this SLG: is other tile's OFM, should be DONE before, just check
                            assert(tensor_info_to_id.count(this tile IFM info) > 0);
                            assert(tensor_info_to_id[this tile IFM info, IFM order] == tensor_id of OFM of layer that generated IFM);
                        */
                        for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                            assert(tensor_info_to_id.count(TensorInfo { ifm_id, tile_id, TensorType::OFM, 0 }) > 0);
                            assert(tensor_info_to_id.count(TensorInfo {layer_id, tile_id, TensorType::IFM, ifm_id }) > 0);
                        }
                    }
                }
                // EXT_IFM TensorTimes init
                FOR_BITSET(ext_id, node.getExtPrevs())
                { // code below same as 'input NOT IN local LG'
                    int pos = layer_id_to_tile_pos[layer_id].start_time;
                    auto cur_ofm_range = fmap_range(fmap_shape(sl.tile_size.c, sl.tile_size.h, sl.tile_size.w), sl.tile_size.bk);
                    layer.ofm_to_ifm(cur_ofm_range);
                    auto input_node_shape = network->getExtInputs()[ext_id].get_shape();
                    cur_ofm_range.c = {0, input_node_shape.c};
                    assert(cur_ofm_range.size() > 0);
                    for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                        TensorTime tt_ifm(pos - 1, pos + 1);
                        tensor_id_to_size.emplace_back(cur_ofm_range.size());
                        tensor_times.emplace_back(tt_ifm);
                        tensor_info_set_types.emplace_back(TensorInfoSetType::ONLY_IFMs);
                        TensorInfo info_ifm = { layer_id, tile_id, TensorType::IFM, -ext_id - 1 };
                        INSERT_TENSOR_INFO(info_ifm, tensor_times.size() - 1);
                        pos += slg.layer_num(); // slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
                    }
                }

                // CNN WGT TensorTimes init
                /*
                    NO or YES but NOT IN this LG: create tensor_time tt, every tile of this layer share the same weight
                    start = not first LG ? (hasWgtPres ? MAX(OFM_end_pos, last_lg_first_tile_pos) : last_lg_first_tile_pos ) : 0
                    end = end of the last tile of this layer
                    ready = cycle that this tensor's been loaded from DRAM
                    tensor_info_to_id[every tile WGT of this layer] = tt_id;
                */
                if (!node.hasWgtPrevs() && layer.weight_size() > 0) {
                    // Start = last layer group's first tile, End = this layer's last tile's pos + 1
                    auto last_lg_first_layer_id = layer_order_to_id[layer_groups[MAX(i - 1, 0)].layer_group_start];
                    auto this_lg_first_layer_id = layer_order_to_id[layer_groups[i].layer_group_start];
                    TensorTime tt_wgt(layer_id_to_tile_pos[layer_id].start_time - 1, layer_id_to_tile_pos[layer_id].end_time + 1);
                    // TensorTime tt_wgt(layer_id_to_tile_pos[(has_prefetch ? last_lg_first_layer_id : this_lg_first_layer_id)].start_time, layer_id_to_tile_pos[layer_id].end_time + 1);
                    if (i == 0 && j == 0 && layer_id_to_order[layer_id] == 0)
                        tt_wgt.start_time = -1;
                    tensor_id_to_size.emplace_back(layer.weight_size());
                    tensor_times.emplace_back(tt_wgt);
                    tensor_info_set_types.emplace_back(TensorInfoSetType::ONLY_WGTs);
                    // all tile of this layer share the same CNN WGT tensor
                    for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                        TensorInfo info_wgt = { layer_id, tile_id, TensorType::WGT, 0 };
                        INSERT_TENSOR_INFO(info_wgt, tensor_times.size() - 1);
                    }
                }

                // OFM TensorTimes init
                {
                    int pos = layer_id_to_tile_pos[layer_id].start_time;
                    for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                        /*
                            for each tile, generate new tensor_time tt:
                            start = pos (aka. start of this tile)
                            ready = cycle(start + 1)
                            end = MIN{MAX{end1, end2, end3}, next_lg_first_tile_pos}
                            tensor_info_to_id[this tile OFM info] = tt_id;
                        */
                        tensor_times.emplace_back(TensorTime(pos, pos + 1));
                        TensorInfoSetType ti_set_type;
                        TensorTime& tt_ofm = tensor_times.back();
                        const auto& ts = all_layers[layer_id].tile_size;
                        assert(ts.bk * ts.c * ts.h * ts.w > 0);
                        tensor_id_to_size.emplace_back(ts.bk * ts.c * ts.h * ts.w);
                        TensorInfo info_ofm = { layer_id, tile_id, TensorType::OFM, 0 };
                        INSERT_TENSOR_INFO(info_ofm, tensor_times.size() - 1);
                        if (/*node.get_nexts().count() == 0*/ node.get_nexts().empty()) {
                            ti_set_type = TensorInfoSetType::OFM_TO_DRAM;
                            if (k != lg.layer_group_end || tile_id != slg.tile_number - 1) { // not the last tile of this layer group
                                tt_ofm.expand(pos, pos + 2);
                            }
                        } else {
                            ti_set_type = TensorInfoSetType::OFM_WITH_LOCAL_IFMs;
                        }
                        FOR_BITSET(ofm_id, node.get_nexts())
                        {
                            const Node& ofm_node = network->getNode(ofm_id);
                            if (layer_id_to_order[ofm_id] > lg.layer_group_end) { // input NOT IN local LG
                                ti_set_type = TensorInfoSetType::OFM_TO_DRAM;
                                // end1 = MIN{pos+2(aka. end of NEXT TILE), lg_end}
                                if (k != lg.layer_group_end || tile_id != slg.tile_number - 1) { // not the last tile of this layer group
                                    tt_ofm.expand(pos, pos + 2);
                                }
                            } else if (layer_id_to_order[ofm_id] > slg.sub_layer_group_end) { // input NOT IN local SLG
                                // end2 = end of last layer tile which needs OFM
                                tt_ofm.expand(pos, layer_id_to_tile_pos[ofm_id].end_time + 1);
                                // tensor_info_to_id[every layer tile info which needs OFM] = tt_id;
                                // Here we only deal with the first tile of the layer, 
                                // the rest would be handled in the loop of that IFM && Transformer WGT
                                assert((ofm_node.hasWgtPrevs() && ofm_node.getWgtPrevs().contains(layer_id)) || ofm_node.getIfmPrevs().contains(layer_id)); 
                                // IFM or Transformer WGT
                                TensorInfo info_ifm = { ofm_id, 0, TensorType::IFM, layer_id };
                                INSERT_TENSOR_INFO(info_ifm, tensor_times.size() - 1);
                            } else { // input IN local SLG
                                // end3 = end of last layer tile (with same tile_id) which needs OFM
                                // tensor_info_to_id[every layer tile info (with same tile_id) which needs OFM] = tt_id;
                                tt_ofm.expand(pos, pos + layer_id_to_tile_pos[ofm_id].start_time - layer_id_to_tile_pos[layer_id].start_time + 1);
                                assert((ofm_node.hasWgtPrevs() && ofm_node.getWgtPrevs().contains(layer_id)) || ofm_node.getIfmPrevs().contains(layer_id));
                                // IFM or Transformer WGT
                                TensorInfo info_ifm = { ofm_id, tile_id, TensorType::IFM, layer_id };
                                INSERT_TENSOR_INFO(info_ifm, tensor_times.size() - 1);
                            }
                        }
                        tensor_info_set_types.emplace_back(ti_set_type);
                        pos += slg.layer_num(); // slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
                    }
                }
            }
        }
    }
    // now pos == total_tile_number
#ifdef DEBUG
    for (int i = 0; i<tensor_times.size(); i++) {
        assert(tensor_id_to_size.at(i) == get_tensor_size(i));
    }
#endif

    // 7. Init tile_tensor_order
    // tile_tensor_order = enc.tile_tensor_order;
    tile_tensor_order.clear();
    tile_tensor_order.reserve(tensor_times.size());
    for (len_t i = 0; i < tensor_times.size(); ++i) { // O(tensors)
        const TensorInfoSetType ti_set_type = tensor_info_set_types[i];
        if (ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs) {
            tile_tensor_order.emplace_back(i);
        }
    }
    std::sort(tile_tensor_order.begin(), tile_tensor_order.end(), [&](int a, int b) {
        if (tensor_times[a].start_time != tensor_times[b].start_time)
            return tensor_times[a].start_time < tensor_times[b].start_time;
        else if (bool a_is_wgt = (tensor_info_set_types[a] == TensorInfoSetType::ONLY_WGTs), 
                      b_is_wgt = (tensor_info_set_types[b] == TensorInfoSetType::ONLY_WGTs);
                      a_is_wgt ^ b_is_wgt)
            return a_is_wgt;
        else if (tensor_times[a].end_time != tensor_times[b].end_time)
            return tensor_times[a].end_time < tensor_times[b].end_time;
        else
            return tensor_info_set_types[a] < tensor_info_set_types[b];
    }); // O(nlogn), where n is # of tensors
    // 8. Init all slgs' BandWidth_allowence, Buffer_allowence, and req_lists
    /*
    for (len_t i = 0; i < all_slgs.size(); ++i) {
        SubLayerGroup& slg = all_slgs[i];
        slg.BandWidth_allowence = 0;
        slg.Buffer_allowence = 0;
        slg.req_list_f.clear();
        slg.req_list_b.clear();
    }
    */
    return ErrorType::SUCCESS;
}

ErrorType Graph::init_stage_1(const Graph::Stage1Encoding& s1enc)
{
    assert(s1enc.tile_numbers.size() == (s1enc.layer_group_partition | s1enc.sub_layer_group_partition).count());
    assert((s1enc.layer_group_partition & s1enc.sub_layer_group_partition).count() == 0);
    ErrorType err = ErrorType::SUCCESS;
    // std::chrono::steady_clock::time_point begin, end;
    // begin = std::chrono::steady_clock::now();
    err = init_stage_1_order(s1enc);
    // end = std::chrono::steady_clock::now();
    // std::cout << "\t\t\tOrder Init Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    if (err != ErrorType::SUCCESS) {return err;}
    // begin = std::chrono::steady_clock::now();
    err = init_stage_1_partition(s1enc);
    // end = std::chrono::steady_clock::now();
    // std::cout << "\t\t\tPartition Init Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    if (err != ErrorType::SUCCESS) {return err;}
    // begin = std::chrono::steady_clock::now();
    err = init_stage_1_back_cal();
    // end = std::chrono::steady_clock::now();
    // std::cout << "\t\t\tBackcal Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    if (err != ErrorType::SUCCESS) {return err;}
    // begin = std::chrono::steady_clock::now();
    err = init_stage_1_tile_pos();
    // end = std::chrono::steady_clock::now();
    // std::cout << "\t\t\tTile Pos Init Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    if (err != ErrorType::SUCCESS) {return err;}
    // begin = std::chrono::steady_clock::now();
    err = init_stage_1_tensor_times();
    // end = std::chrono::steady_clock::now();
    // std::cout << "\t\t\tTensor Times Init Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
    if (err != ErrorType::SUCCESS) {return err;}

    return ErrorType::SUCCESS; // This only means that it is valid up to the **Stage 1**.
}

ErrorType Graph::init_stage_1_with_tile_sizes(const Stage1Encoding_tile_sizes& s1enc_tz)
{
    assert(s1enc_tz.tile_sizes.size() == network->len());
    assert((s1enc_tz.layer_group_partition & s1enc_tz.sub_layer_group_partition).count() == 0);
    ErrorType err = ErrorType::SUCCESS;
    err = init_stage_1_order_with_tile_sizes(s1enc_tz);
    if (err != ErrorType::SUCCESS) {return err;}
    err = init_stage_1_partition_with_tile_sizes(s1enc_tz);
    if (err != ErrorType::SUCCESS) {return err;}
    err = init_stage_1_back_cal_and_change_tile_sizes(); // different from the previous one
    if (err != ErrorType::SUCCESS) {return err;}
    err = init_stage_1_tile_pos();
    if (err != ErrorType::SUCCESS) {return err;}
    err = init_stage_1_tensor_times();
    if (err != ErrorType::SUCCESS) {return err;}

    return ErrorType::SUCCESS; // This only means that it is valid up to the **Stage 1**.
}

ErrorType Graph::init_stage_2(const Graph::Stage2Encoding& s2enc, const bool update_dram_order, const bool update_tensor_time) 
{
    // 1. Init tensor_times
    if (update_tensor_time)
        tensor_times = s2enc.tensor_times;
    // 2. Init tile_tensor_order
    if (update_dram_order)
        tile_tensor_order = s2enc.tile_tensor_order;
    if (!check_buffer_valid())
        return ErrorType::BUFFER_OVERFLOW;
    return ErrorType::SUCCESS; // This only means that it is valid up to the **preliminary testing**.
}

pair<Graph::Stage1Encoding, Graph::Stage2Encoding> Graph::get_Encoding() const
{
    Stage1Encoding s1enc;
    Stage2Encoding s2enc;
    s1enc.layer_order_to_id = layer_order_to_id;
    s1enc.layer_group_partition.clear();
    s1enc.sub_layer_group_partition.clear();
    s1enc.tile_numbers.reserve(all_slgs.size());
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        const LayerGroup& lg = layer_groups[i];
        s1enc.layer_group_partition.set(lg.layer_group_start);
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const SubLayerGroup& slg = all_slgs[j];
            if (slg.sub_layer_group_start != lg.layer_group_start)
                s1enc.sub_layer_group_partition.set(slg.sub_layer_group_start);
            s1enc.tile_numbers.push_back(slg.tile_number);
        }
    }
    s2enc.tensor_times = tensor_times;
    s2enc.tile_tensor_order = tile_tensor_order;
    return make_pair(s1enc, s2enc);
}

void Graph::get_intensity(vector<vol_t>& lg_comp_time, vector<vol_t>& lg_dram_time, vector<vol_t>& slg_comp_time, vector<vol_t>& slg_dram_time) const
{
    vol_t sum_tile_comp_time = 0;
    vol_t sum_tile_dram_time = 0;
    energy_t sum_comp_energy = 0;
    energy_t sum_dram_energy = 0;
    lg_comp_time.clear();
    lg_dram_time.clear();
    slg_comp_time.clear();
    slg_dram_time.clear();
    lg_comp_time.resize(layer_groups.size(), 0);
    lg_dram_time.resize(layer_groups.size(), 0);
    slg_comp_time.resize(all_slgs.size(), 0);
    slg_dram_time.resize(all_slgs.size(), 0);
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        const LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const SubLayerGroup& slg = all_slgs[j];
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                const SubLayer& sl = all_layers[layer_id];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                const auto& ts =  all_layers[layer_id].tile_size;
                
                const CoreMapper::CoreMapping tile_cost = getTileCost(layer_id, sl.tile_size);
                cycle_t layer_comp_time = tile_cost.cost.time * slg.tile_number;
                energy_t layer_comp_energy = tile_cost.cost.energy * slg.tile_number;
                energy_t layer_ubuf_energy = tile_cost.ubuf * slg.tile_number;
                energy_t layer_buffer_energy = tile_cost.buffer * slg.tile_number;
                energy_t layer_noc_energy = tile_cost.noc * slg.tile_number;
                energy_t layer_mac_energy = tile_cost.mac * slg.tile_number;
                energy_t layer_dram_energy = 0;
                slg_comp_time[j] += layer_comp_time;
                sum_comp_energy += layer_comp_energy;

                std::cout << "Layer " << layer_id << " Sum COMP Time = " << layer_comp_time << ", ";
                
                int layer_dram_time = 0;
                // WGT
                if(!node.hasWgtPrevs() && layer.weight_size() > 0) {
                    layer_dram_time += DIVCEIL(layer.weight_size(), SchNode::DRAM_bw);
                    layer_dram_energy += layer.weight_size() * NoC::DRAM_acc_cost;
                }
                // OFM
                const TensorInfo first_info_ofm = { layer_id, 0, TensorType::OFM, 0 };
                auto range = tensor_info_to_id.equal_range(first_info_ofm);
                const len_t& ofm_tid = range.first->second;
                assert(distance(range.first, range.second) == 1);
                if (tensor_info_set_types[ofm_tid] == TensorInfoSetType::OFM_TO_DRAM) {
                    const vol_t ofm_tensor_size = ts.bk * ts.c * ts.h * ts.w;
                    layer_dram_time += DIVCEIL(ofm_tensor_size * slg.tile_number, SchNode::DRAM_bw);
                    layer_dram_energy += ofm_tensor_size * slg.tile_number * NoC::DRAM_acc_cost;
                }
                // IFM
                {
                    auto cur_ofm_range = fmap_range(fmap_shape(sl.tile_size.c, sl.tile_size.h, sl.tile_size.w), sl.tile_size.bk);
                    layer.ofm_to_ifm(cur_ofm_range);
                    layer_dram_time += DIVCEIL(cur_ofm_range.size() * slg.tile_number, SchNode::DRAM_bw);
                    layer_dram_energy += cur_ofm_range.size() * slg.tile_number * NoC::DRAM_acc_cost;
                }
                if (node.hasWgtPrevs()) {
                    auto cur_ofm_range = fmap_range(fmap_shape(sl.tile_size.c, sl.tile_size.h, sl.tile_size.w), sl.tile_size.bk);
                    layer.ofm_to_wgt(cur_ofm_range);
                    layer_dram_time += DIVCEIL(cur_ofm_range.size() * slg.tile_number, SchNode::DRAM_bw);
                    layer_dram_energy += cur_ofm_range.size() * slg.tile_number * NoC::DRAM_acc_cost;
                }
                slg_dram_time[j] += layer_dram_time;
                sum_dram_energy += layer_dram_energy;
                std::cout << "Sum DRAM Time = " << layer_dram_time << ", "
                          << "Sum COMP Energy = " << layer_comp_energy << ", "
                          << "Sum UBuf Energy = " << layer_ubuf_energy << ", "
                          << "Sum Buffer Energy = " << layer_buffer_energy << ", "
                          << "Sum NoC Energy = " << layer_noc_energy << ", "
                          << "Sum MAC Energy = " << layer_mac_energy << ", "
                          << "Sum DRAM Energy = " << layer_dram_energy << std::endl;
            }
            lg_comp_time[i] += slg_comp_time[j];
            lg_dram_time[i] += slg_dram_time[j];
        }
        sum_tile_comp_time += lg_comp_time[i];
        sum_tile_dram_time += lg_dram_time[i];
    }
    std::cout << "Network Sum Tile COMP Time = " << sum_tile_comp_time 
              << ", Sum DRAM Time = " << sum_tile_dram_time << std::endl;
    return ;
}

void Graph::initTileCosts() 
{
    num_tile_cost_cache_total += network->len();
    for (lid_t k = 0; k < network->len(); ++k) {
        const lid_t& layer_id = layer_order_to_id[k];
        const SubLayer& sl = all_layers[layer_id];
        if (tile_costs.count(std::make_pair(layer_id, sl.tile_size)) == 0) {
            num_tile_cost_cache_miss++;
            CoreMapper::MapCost tile_cost = getTileCost(layer_id, sl.tile_size).cost;
            tile_costs[std::make_pair(layer_id, sl.tile_size)] = tile_cost;
        } else {
            num_tile_cost_cache_hit++;
        }
    }
}

/*
void Graph::get_slg_req_lists() // UNUSED, UNFINISHED
{
    // Update all slgs' BandWidth_allowence, Buffer_allowence, and req_lists
    for (len_t i = 0; i < all_slgs.size(); ++i) {
        SubLayerGroup& slg = all_slgs[i];
        slg.BandWidth_allowence = 0;
        slg.Buffer_allowence = 0;
        slg.req_list_f.clear();
        slg.req_list_b.clear();
        vol_t slg_comp_time = 0;
        for (len_t j = slg.sub_layer_group_start; j <= slg.sub_layer_group_end; ++j) {
            const lid_t& layer_id = layer_order_to_id[j];
            const SubLayer& sl = all_layers[layer_id];
            const Node& node = network->getNode(layer_id);
            const Layer& layer = node.layer();
                
            const CoreMapper::MapCost& tile_cost = tile_costs.at(std::make_pair(layer_id, sl.tile_size));
            slg_comp_time += tile_cost.time * slg.tile_number;

            // pick the last OFM for req_list_b
            const TensorInfo last_info_ofm = { layer_id, slg.tile_number - 1, TensorType::OFM, 0 };
            auto range = tensor_info_to_id.equal_range(last_info_ofm);
            const len_t& ofm_tid = range.first->second;
            assert(distance(range.first, range.second) == 1);
            if (tensor_info_set_types[ofm_tid] == TensorInfoSetType::OFM_TO_DRAM) {
                slg.req_list_b.push_back(ofm_tid);
            }

        }
    }
}
*/

Graph::IdealCostResults Graph::getIdealCost(CoreMapper::MapCost& ideal_cost, const bool print_results) const
{
    // Stage 1
    // Step 1: Pre-Processing (Already Done in Constructor)
    // Step 2: cal COMP and DRAM L/S latency
#ifdef DEBUG
    len_t total_tile_number = 0;
#endif
    ideal_cost.time = ideal_cost.energy = 0;
    double ideal_comp = 0, ideal_dram = 0;
    energy_t comp_energy = 0;
    energy_t ubuf_energy = 0;
    energy_t buffer_energy = 0;
    energy_t noc_energy = 0;
    energy_t mac_energy = 0;
    energy_t dram_energy = 0;
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        const LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const SubLayerGroup& slg = all_slgs[j];
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = layer_order_to_id[k];
                const SubLayer& sl = all_layers[layer_id];
                const Node& cur_node = network->getNode(layer_id);
                const Layer& cur_layer = cur_node.layer();

                const CoreMapper::CoreMapping tile_cost = getTileCost(layer_id, sl.tile_size);

                ideal_comp += tile_cost.cost.time * slg.tile_number;
                comp_energy += tile_cost.cost.energy * slg.tile_number;
                ubuf_energy += tile_cost.ubuf * slg.tile_number;
                buffer_energy += tile_cost.buffer * slg.tile_number;
                noc_energy += tile_cost.noc * slg.tile_number;
                mac_energy += tile_cost.mac * slg.tile_number;
            }
#ifdef DEBUG
            total_tile_number += slg.tile_number * slg.layer_num(); // (slg.sub_layer_group_end - slg.sub_layer_group_start + 1);
#endif
        }
    }
    ideal_cost.energy = (ubuf_energy + buffer_energy) + (noc_energy + mac_energy);
    for (len_t tensor_id = 0; tensor_id < tensor_times.size(); ++tensor_id) {
        if (tensor_info_set_types[tensor_id] != TensorInfoSetType::OFM_WITH_LOCAL_IFMs) {
            const vol_t tensor_size = tensor_id_to_size.at(tensor_id);
            ideal_dram += DIVCEIL(tensor_size, SchNode::DRAM_bw);
            dram_energy += tensor_size * NoC::DRAM_acc_cost;
        }
    }
    ideal_cost.energy += dram_energy;
    ideal_cost.time = MAX(ideal_comp, ideal_dram);

    IdealCostResults icr = {ideal_comp, ideal_dram, comp_energy, ubuf_energy, buffer_energy, noc_energy, mac_energy, dram_energy};
    if (print_results) {
        #pragma omp critical
        {
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", " << icr;
        }
    }

#ifdef DEBUG
    assert(total_tile_number == layer_id_to_tile_pos[layer_order_to_id[network->len() - 1]].end_time + 1);
    std::cout << "total_tile_number: " << total_tile_number << std::endl; 
#endif
    return icr;
}

inline void Graph::getTensorsForNextTile(unordered_set<len_t /*tensor_id*/> &ts, const len_t& next_layer_id, const len_t& next_tile_id) const
{
    // 3.1. CNN WGT of NEXT TILE
    if (!network->getNode(next_layer_id).hasWgtPrevs() && network->getNode(next_layer_id).layer().weight_size() > 0) {
        TensorInfo next_tile_wgt_tensor_info = { next_layer_id, next_tile_id, TensorType::WGT, 0 };
        auto range = tensor_info_to_id.equal_range(next_tile_wgt_tensor_info); // Need to check whether it is from DRAM
        auto wgt_it = range.first;
        for (auto chk_wgt_it = range.first; chk_wgt_it != range.second; ++chk_wgt_it) {
            assert(chk_wgt_it->second == wgt_it->second);
        }
        const TensorInfoSetType ti_set_type = tensor_info_set_types[wgt_it->second];
        if (ti_set_type == TensorInfoSetType::ONLY_WGTs && !buffer.ask_tensor_ready(wgt_it->second)) {
            ts.insert(wgt_it->second);
        }
    }
    // 3.2. IFM & Transformer WGT of NEXT TILE
    {
        FOR_BITSET(it, network->getNode(next_layer_id).getPrevs()) { // it == layer_id
            // ONLY DRAM I/O tensors
            // LOCAL tensors are generated for buffer
            TensorInfo next_tile_ifm_tensor_info = { next_layer_id, next_tile_id, TensorType::IFM, it };
            auto range = tensor_info_to_id.equal_range(next_tile_ifm_tensor_info);
            for (auto ifm_it = range.first; ifm_it != range.second; ++ifm_it) {
                const TensorInfoSetType ti_set_type = tensor_info_set_types[ifm_it->second];
                if (ti_set_type == TensorInfoSetType::ONLY_IFMs && !buffer.ask_tensor_ready(ifm_it->second)) {
                    ts.insert(ifm_it->second);
                }
            }
        }
    }
    // 3.3. EXTERNAL IFM of NEXT TILE
    {
        FOR_BITSET(it, network->getNode(next_layer_id).getExtPrevs()) { // it == layer_id
            TensorInfo next_tile_ifm_tensor_info = { next_layer_id, next_tile_id, TensorType::IFM, -it - 1 };
            auto range = tensor_info_to_id.equal_range(next_tile_ifm_tensor_info);
            for (auto ifm_it = range.first; ifm_it != range.second; ++ifm_it) {
                const TensorInfoSetType ti_set_type = tensor_info_set_types[ifm_it->second];
                if (ti_set_type == TensorInfoSetType::ONLY_IFMs && !buffer.ask_tensor_ready(ifm_it->second)) {
                    ts.insert(ifm_it->second);
                }
            }
        }
    }
}

bool Graph::check_buffer_valid()
{
    const len_t& total_tile_number = layer_id_to_tile_pos[layer_order_to_id[network->len() - 1]].end_time + 1;
    // Stage 2

    // Step 1: Buffer Management
    // check buffer validility
    // this is right because all tensors in buffer will increase buffer usage
    // so we do not have to consider tensor load/store order
    buffer.buffer_usage_by_tile.assign(total_tile_number + 1, 0); // index == tile_id
    // buffer.buffer_usage_by_tile.resize(total_tile_number + 1); //for -1
    { // control varible scope
        auto& A = buffer.buffer_usage_by_tile;
        vector<int64_t /*usage*/> D(A.size(), 0); // index == tile_id
        for (len_t tensor_id = 0; tensor_id < tensor_times.size(); ++tensor_id) { // O(# of tensors)
            const TensorTime& tt = tensor_times[tensor_id];
            vol_t tensor_size = tensor_id_to_size.at(tensor_id);
            D[tt.start_time + 1] += tensor_size;
            if (tt.end_time + 1 < D.size())
                D[tt.end_time + 1] -= tensor_size;
        }
        A[0] = D[0];
        for (len_t i = 1; i < A.size(); i++) {
            A[i] = A[i - 1] + D[i];
        }
    }
#ifdef DEBUG
    for (int i = 0; i < buffer.buffer_usage_by_tile.size(); i++) {
        std::cout << "@Tile " << i-1 << " Buffer Size: " << buffer.buffer_usage_by_tile[i] << std::endl;
    }
#endif
    for (len_t i = 0; i < buffer.buffer_usage_by_tile.size(); i++) {
        if (buffer.buffer_usage_by_tile[i] > buffer.get_max_buffer_size()) {
            // std::cout << "early buffer invalid: @" << i << " with size: " << buffer.buffer_usage_by_tile[i] << std::endl;
            return false; // NOT VALID!
        }
    }
    return true;
}

ErrorType Graph::getRealCost(CoreMapper::MapCost& real_cost, bool record, bool do_check)
{
    buffer.clear();
    if (record) {
        DRAM_Tensor_Info_by_time.clear();
        COMP_Tile_Info_by_time.clear();
    }
    // std::chrono::steady_clock::time_point begin, end;
    if (do_check) {
        bool check_pass = false;
        // begin = std::chrono::steady_clock::now();
        check_pass = check_buffer_valid();
        // end = std::chrono::steady_clock::now();
        // std::cout << "\t\t\tCheck Buffer Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
        if (!check_pass)
            return ErrorType::BUFFER_OVERFLOW;
        /*
        // std::cout << "\t\t\tCheck Tensor Order Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;
        if (!check_pass) {
            assert(false && "Tensor Order Dependency Error");
            return ErrorType::TENSOR_ORDER_DEPENDENCY;
        }
        */
    }
    real_cost.energy = real_cost.time = 0;
    const len_t& total_tile_number = layer_id_to_tile_pos[layer_order_to_id[network->len() - 1]].end_time + 1;
    if (record) {
        tile_start_cycles.clear();
        tile_start_cycles.reserve(total_tile_number);
        dram_tensor_start_cycles.clear();
        dram_tensor_start_cycles.reserve(tile_tensor_order.size());
    }

    // Step 2: Tensor Time Calculation
    // SubStep 1. Sort tensor_time by start and by end separately, from small to large
    // begin = std::chrono::steady_clock::now();
    vector<int /* index of sorted tensor_times */> tt_index_sorted_by_start(tensor_times.size(), 0);
    vector<int /* index of sorted tensor_times */> tt_index_sorted_by_end(tensor_times.size(), 0);
    for (int i = 0; i != tensor_times.size(); i++) {
        tt_index_sorted_by_start[i] = i;
        tt_index_sorted_by_end[i] = i;
    }
    std::sort(tt_index_sorted_by_start.begin(), tt_index_sorted_by_start.end(), [&](int a, int b) {
        return tensor_times[a].start_time < tensor_times[b].start_time;
    });
    std::sort(tt_index_sorted_by_end.begin(), tt_index_sorted_by_end.end(), [&](int a, int b) {
        return tensor_times[a].end_time < tensor_times[b].end_time;
    });
    // end = std::chrono::steady_clock::now();
    // std::cout << "\t\t\tTensor Times Sort Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    cycle_t dram = 0, comp = 0;
    len_t tile_tensor_idx = 0; // tile_tensor_idx: index of reading tile_tensor_order
    len_t stt_idx = 0; // stt_idx: (last index with tensor_times[tt_index_sorted_by_start[stt_idx]].start_time == pos) + 1
    len_t ett_idx = 0; // ett_idx: (last index with tensor_times[tt_index_sorted_by_end[ett_idx]].end_time == pos + 1) + 1
    len_t last_tile_unloaded_tensor_idx = 0;
    // __uint128_t total_buffer_add_time = 0;
    // __uint128_t total_comp_add_time = 0;
    // __uint128_t total_dram_free_time = 0;
    // __uint128_t total_dependency_find_time = 0;
    // __uint128_t total_dram_force_time = 0;
    // __uint128_t total_buffer_del_time = 0;
    // SubStep 2. Special Run for the first tile
    {
        constexpr int pos = -1;
        constexpr lid_t layer_id = 0;
        constexpr len_t tile_id = 0;
#ifdef DEBUG
        std::cout << "pos = " << pos << std::endl;
#endif
        // begin = std::chrono::steady_clock::now();
        // 0. buffer_add all tensors in tensor_times with start == pos == -1
        for (; stt_idx < tensor_times.size(); stt_idx++) {
            const auto& tt_index = tt_index_sorted_by_start[stt_idx];
            if (tensor_times[tt_index].start_time < pos) { // There shall never be start < -1
                assert(false && "There shall never be start < -1");
            } else if (tensor_times[tt_index].start_time == pos) {
                vol_t tensor_size = tensor_id_to_size.at(tt_index);
#ifdef DEBUG
                std::cout << "Buffer Add: " << tt_index << "@" << tensor_times[tt_index];
#endif
                if (!buffer.buffer_add(tt_index, tensor_size)) {
#ifdef DEBUG
                    std::cout << " NOT Done!" << std::endl;
#endif
                    std::cout << "Buffer overflow at tile 0" << std::endl;
                    return ErrorType::BUFFER_OVERFLOW;
                } else {
#ifdef DEBUG
                    std::cout << " Done!" << std::endl;
#endif
                }
            } else { 
                break; 
            }
        }
        buffer.buffer_usage_by_time[comp] = buffer.cur_buffer_usage;
        // end = std::chrono::steady_clock::now();
        // total_buffer_add_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        // 1. Compute Time of THIS TILE
        // comp += 0;
        // begin = std::chrono::steady_clock::now();
        // 2. follow tile_tensor_order, do DRAM L/S one by one until one of the following conditions is met: 
        //      dram + tensor_access_time > comp
        //      current tensor is this/future tile's OFM
        while (tile_tensor_idx < tile_tensor_order.size()) {
            len_t& cur_tile_tensor_id = tile_tensor_order[tile_tensor_idx];
            if (!buffer.buffer_has(cur_tile_tensor_id)) {
                if (dram < comp) {
                    dram = comp;
                }
                break;
            }
            if (tensor_times[cur_tile_tensor_id].start_time > pos || pos > tensor_times[cur_tile_tensor_id].end_time) {
                // std::cout << "OutBound " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                return ErrorType::TENSOR_ORDER_DEPENDENCY;
            }
            vol_t tensor_size = tensor_id_to_size.at(cur_tile_tensor_id);
            cycle_t tensor_access_time = DIVCEIL(tensor_size, SchNode::DRAM_bw);
            const TensorInfoSetType ti_set_type = tensor_info_set_types[cur_tile_tensor_id];
            auto range = tensor_id_to_info.equal_range(cur_tile_tensor_id);
            auto info = range.first->second;
            // STALL for THIS TILE OFM
            if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM) {
                info = getOFMInfo(range);
                assert(info.tensor_type == TensorType::OFM && info.source == 0);
                // assert(distance(range.first, range.second) == 1); 
                if (info.layer_id == layer_id && info.tile_id == tile_id && dram < comp) {
                    dram = comp;
                    break;
                }
            }
            assert(ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
            if (dram + tensor_access_time > comp) {
                break;
            }
            if (record) {
                DRAM_Tensor_Info_by_time[dram] = DRAM_Tensor_Info({ info.layer_id, info.tile_id, info.tensor_type, tensor_access_time });
                dram_tensor_start_cycles[cur_tile_tensor_id] = dram;
            }
            dram += tensor_access_time;
            real_cost.energy += tensor_size * NoC::DRAM_acc_cost;
#ifdef DEBUG
            std::cout << "Buffer Ready: " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id];
#endif
            buffer.set_tensor_ready(cur_tile_tensor_id);
#ifdef DEBUG
            std::cout << " Done!" << std::endl;
#endif
            tile_tensor_idx++;
        }
        // end = std::chrono::steady_clock::now();
        // total_dram_free_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        // begin = std::chrono::steady_clock::now();
        // 3. Check if all tensors that the NEXT TILE needs are ready
        unordered_set<len_t /*tensor_id*/> tensors_for_next_tile;
        // 3.0. Get next_layer_id and next_tile_id
        constexpr lid_t next_layer_id = 0;
        constexpr len_t next_tile_id = 0;
        // 3.1~3.3. IFMs & WGTs for next tile

        getTensorsForNextTile(tensors_for_next_tile, next_layer_id, next_tile_id);


        // 3.4. OFM whose end == pos + 1
        for (int tt_idx = ett_idx; ett_idx < tensor_times.size(); tt_idx++) {
            if (tensor_times[tt_index_sorted_by_end[tt_idx]].end_time < pos + 1) {
            } else if (tensor_times[tt_index_sorted_by_end[tt_idx]].end_time == pos + 1) {
                assert(false && "There shall never be end == 0"); // There shall never be end == 0
                // const TensorInfoSetType ti_set_type = tensor_info_set_types[tt_index_sorted_by_end[tt_idx]];
                // if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM && !buffer.ask_tensor_ready(tt_index_sorted_by_end[tt_idx])) {
                //     tensors_for_next_tile.insert(tt_index_sorted_by_end[tt_idx]);
                // }
            } else {
                break;
            }
        }
        // 3.5. Get the last index of the tensor that we need to handle,
        // which is, the max index in array tile_tensor_order with tile_tensor_order[index] in tensors_for_next_tile
        // if last_tile_tensor_idx < tile_tensor_idx,
        //     then we do not need to handle any tensor
        // if last_tile_tensor_idx >= tile_tensor_idx,
        //     then we need to handle all tensors from tile_tensor_order[tile_tensor_idx ~ last_tile_tensor_idx]
        /* auto last_tile_tensor_idx = tile_tensor_order.rend() - 1 - find_if(tile_tensor_order.rbegin(), tile_tensor_order.rend() - tile_tensor_idx, [&](const len_t& x) {
            return tensors_for_next_tile.count(x) > 0;
        });*/
        auto last_tile_tensor_idx = tile_tensor_idx;
        if (tensors_for_next_tile.empty()) {
            last_tile_tensor_idx--;
        } else {
            auto temp_tensors_for_next_tile = tensors_for_next_tile;
            for (; last_tile_tensor_idx < tile_tensor_order.size(); ++last_tile_tensor_idx) {
                auto idx = temp_tensors_for_next_tile.find(tile_tensor_order[last_tile_tensor_idx]);
                if (idx != temp_tensors_for_next_tile.end()) {
                    temp_tensors_for_next_tile.erase(idx);
                }
                if (temp_tensors_for_next_tile.empty())
                    break;
            }
            assert(temp_tensors_for_next_tile.empty());
        }
        assert(last_tile_tensor_idx >= tile_tensor_idx || 
              (tensors_for_next_tile.empty() && last_tile_tensor_idx == tile_tensor_idx - 1));
#ifdef DEBUG
        std::cout << "Tensors for Next Tile: ";
        for (auto& t_id: tensors_for_next_tile) {
            std::cout << t_id << "@" << tensor_times[t_id] << ", ";
        }
        std::cout << std::endl;
#endif
        // end = std::chrono::steady_clock::now();
        // total_dependency_find_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        // begin = std::chrono::steady_clock::now();
        // 4. Then continue to load/store until all dependencies/end==pos tensors are resolved
        // ONLY those tensors that interact with DRAM might update dram time and envoke buffer_ready:
        //    - ONLY {1 OFM} tensors, ONLY {1/more WGT...} tensors, ONLY {1/more IFM...} tensors
        last_tile_unloaded_tensor_idx = last_tile_tensor_idx;
        while (tile_tensor_idx <= last_tile_tensor_idx) {
            len_t& cur_tile_tensor_id = tile_tensor_order[tile_tensor_idx];
            if (!buffer.buffer_has(cur_tile_tensor_id)) {
                // first tile self needed WGTs and IFMs should have start == -1,
                // so this situation should never happen:
                //    the first tile's dependency tensors have start == 0
                // std::cout << "Missing " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                return ErrorType::TENSOR_ORDER_DEPENDENCY;
            }
            if (tensor_times[cur_tile_tensor_id].start_time > pos || pos > tensor_times[cur_tile_tensor_id].end_time) {
                // std::cout << "OutBound " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                return ErrorType::TENSOR_ORDER_DEPENDENCY;
            }
            
            const TensorInfoSetType ti_set_type = tensor_info_set_types[cur_tile_tensor_id];
            auto range = tensor_id_to_info.equal_range(cur_tile_tensor_id);
            // STALL for THIS TILE OFM
            TensorInfo info = range.first->second;
            if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM) {
                info = getOFMInfo(range);
                assert(info.tensor_type == TensorType::OFM && info.source == 0);
                // assert(distance(range.first, range.second) == 1);
                assert(false && "First Tile's dependency tensor should never be an OFM"); // First Tile's dependency tensor should never be an OFM
                // if (info.layer_id == layer_id && info.tile_id == tile_id && dram < comp) {
                //     dram = comp;
                // }
            }
            vol_t tensor_size = tensor_id_to_size.at(cur_tile_tensor_id);
            cycle_t tensor_access_time = DIVCEIL(tensor_size, SchNode::DRAM_bw);
            assert(ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
            assert(ti_set_type != TensorInfoSetType::OFM_TO_DRAM && "First Tile's dependency tensor should never be an OFM");
            if (record) {
                DRAM_Tensor_Info_by_time[dram] = DRAM_Tensor_Info({ info.layer_id, info.tile_id, info.tensor_type, tensor_access_time });
                dram_tensor_start_cycles[cur_tile_tensor_id] = dram;
            }
            dram += tensor_access_time;
            real_cost.energy += tensor_size * NoC::DRAM_acc_cost;
#ifdef DEBUG
            std::cout << "Buffer Ready: " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id];
#endif
            buffer.set_tensor_ready(cur_tile_tensor_id);
#ifdef DEBUG
            std::cout << " Done!" << std::endl;
#endif
            tile_tensor_idx++;
        }
        // end = std::chrono::steady_clock::now();
        // total_dram_force_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        // begin = std::chrono::steady_clock::now();
        // 5. buffer_del ALL tensors with end == pos + 1 
        // Because all tile_tensor's start <= end - 1, if End == pos + 1, we have Start <= pos
        for (; ett_idx < tensor_times.size(); ett_idx++) {
            const auto& tt_index = tt_index_sorted_by_end[ett_idx];
            if (tensor_times[tt_index].end_time < pos + 1) {
                assert(false && "There shall never be end < 0"); // There shall never be end < 0
            } else if (tensor_times[tt_index].end_time == pos + 1) {
                // buffer.buffer_del(tt_index);
                assert(false && "There shall never be end == 0"); // There shall never be end == 0
            } else {
                break;
            }
        }
        // 6. Sync Here
        if (!tensors_for_next_tile.empty() && dram > comp) {
            /* this means comp stall:
                |      DRAM     |<- sync here
                | COMP |  STALL |?|   next COMP   |
            */
            comp = dram;
        }
        // end = std::chrono::steady_clock::now();
        // total_buffer_del_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    // SubStep 3. Run for the rest tiles
    len_t pos = 0;
    for (lid_t i = 0; i < layer_groups.size(); ++i) {
        LayerGroup& lg = layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            SubLayerGroup& slg = all_slgs[j];
            // TODO: special handle with the start tile of the Sub Layer Group
            for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) { // for each tile
                    const lid_t& layer_id = layer_order_to_id[k];
                    SubLayer& sl = all_layers[layer_id];
                    const Node& node = network->getNode(layer_id);
                    const Layer& layer = node.layer();

#ifdef DEBUG
                    std::cout << "pos = " << pos << std::endl;
#endif
                    // begin = std::chrono::steady_clock::now();
                    // 0. buffer_add all tensors in tensor_times with start == pos
                    for (; stt_idx < tensor_times.size(); stt_idx++) {
                        const auto& tt_index = tt_index_sorted_by_start[stt_idx];
                        if (tensor_times[tt_index].start_time < pos) {
                            assert((tensor_times[tt_index].end_time <= pos && !buffer.buffer_has(tt_index))
                                || (tensor_times[tt_index].end_time  > pos &&  buffer.buffer_has(tt_index)));
                            continue;
                        } else if (tensor_times[tt_index].start_time == pos) {
                            vol_t tensor_size = tensor_id_to_size.at(tt_index);
#ifdef DEBUG
                            std::cout << "Buffer Add: " << tt_index << "@" << tensor_times[tt_index];
#endif
                            if (!buffer.buffer_add(tt_index, tensor_size)) {
#ifdef DEBUG
                                std::cout << " NOT Done!" << std::endl;
#endif
                                std::cout << "Buffer overflow at tile " << tile_id << std::endl;
                                return ErrorType::BUFFER_OVERFLOW;
                            } else {
#ifdef DEBUG
                                std::cout << " Done!" << std::endl;
#endif
                            }
                        } else {
                            break;
                        }
                    }
                    if (stt_idx < tensor_times.size()) {
                        assert(tensor_times[tt_index_sorted_by_start[stt_idx]].start_time > pos);
                    }
                    // end = std::chrono::steady_clock::now();
                    // total_buffer_add_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    // begin = std::chrono::steady_clock::now();
                    // 0.5. if there are tensors that were not loaded in the last tile, then load them
                    if (tile_tensor_idx <= last_tile_unloaded_tensor_idx) {
#ifdef DEBUG
                        std::cout << "Handling Last Tile Unloaded Tensors" << std::endl;
#endif
                        while (tile_tensor_idx <= last_tile_unloaded_tensor_idx) {
                            len_t& cur_tile_tensor_id = tile_tensor_order[tile_tensor_idx];
                            if (!buffer.buffer_has(cur_tile_tensor_id)) {
                                // std::cout << "Missing " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                                return ErrorType::TENSOR_ORDER_DEPENDENCY;
                            }
                            if (tensor_times[cur_tile_tensor_id].start_time > pos || pos > tensor_times[cur_tile_tensor_id].end_time) {
                                // std::cout << "OutBound " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                                return ErrorType::TENSOR_ORDER_DEPENDENCY;
                            }
                            
                            const TensorInfoSetType ti_set_type = tensor_info_set_types[cur_tile_tensor_id];
                            auto range = tensor_id_to_info.equal_range(cur_tile_tensor_id);
                            // STALL for THIS TILE OFM
                            TensorInfo info = range.first->second;
                            if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM) {
                                info = getOFMInfo(range);
                                assert(info.tensor_type == TensorType::OFM && info.source == 0);
                                // assert(distance(range.first, range.second) == 1); 
                                if (info.layer_id == layer_id && info.tile_id == tile_id && dram < comp) {
                                    dram = comp;
                                }
                            }
                            vol_t tensor_size = tensor_id_to_size.at(cur_tile_tensor_id);
                            cycle_t tensor_access_time = DIVCEIL(tensor_size, SchNode::DRAM_bw);
                            assert(ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
                            if (record) {
                                DRAM_Tensor_Info_by_time[dram] = DRAM_Tensor_Info({ info.layer_id, info.tile_id, info.tensor_type, tensor_access_time });
                                dram_tensor_start_cycles[cur_tile_tensor_id] = dram;
                            }
                            dram += tensor_access_time;
                            real_cost.energy += tensor_size * NoC::DRAM_acc_cost;
#ifdef DEBUG
                            std::cout << "Buffer Ready: " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id];
#endif
                            buffer.set_tensor_ready(cur_tile_tensor_id);
#ifdef DEBUG
                            std::cout << " Done!" << std::endl;
#endif
                            tile_tensor_idx++;
                        }
                        // Loading unhandled tensors increase dram time, so we need to sync
                        if (dram > comp)
                            comp = dram;
                    }
                    // end = std::chrono::steady_clock::now();
                    // total_dram_force_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    // begin = std::chrono::steady_clock::now();
                    // 1. Compute Time of THIS TILE
                    const CoreMapper::MapCost& tile_cost = tile_costs.at(std::make_pair(layer_id, sl.tile_size)); // getTileCost(layer_id, sl.tile_size);
                    buffer.buffer_usage_by_time[comp] = buffer.cur_buffer_usage;
                    if (record) {
                        COMP_Tile_Info_by_time[comp] = COMP_Tile_Info({layer_id, (len_t)tile_id, tile_cost.time });
                        tile_start_cycles[std::make_pair(layer_id, (len_t)tile_id)] = comp;
                    }
                    comp += tile_cost.time;
                    real_cost.energy += tile_cost.energy;
                    // end = std::chrono::steady_clock::now();
                    // total_comp_add_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    // begin = std::chrono::steady_clock::now();
                    // 2. follow tile_tensor_order, do DRAM L/S one by one until one of the following conditions is met: 
                    //      dram + tensor_access_time > comp
                    //      current tensor is this/future tile's OFM
#ifdef DEBUG
                    std::cout << "DRAM Free L/S Stage" << std::endl;
#endif
                    while (tile_tensor_idx < tile_tensor_order.size()) {
                        len_t& cur_tile_tensor_id = tile_tensor_order[tile_tensor_idx];
                        if (!buffer.buffer_has(cur_tile_tensor_id)) {
                            if (dram < comp) {
                                dram = comp;
                            }
                            break;
                        }
                        if (tensor_times[cur_tile_tensor_id].start_time > pos || pos > tensor_times[cur_tile_tensor_id].end_time) {
                            // std::cout << "OutBound " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                            return ErrorType::TENSOR_ORDER_DEPENDENCY;
                        }
                        vol_t tensor_size = tensor_id_to_size.at(cur_tile_tensor_id);
                        cycle_t tensor_access_time = DIVCEIL(tensor_size, SchNode::DRAM_bw);
                        const TensorInfoSetType ti_set_type = tensor_info_set_types[cur_tile_tensor_id];
                        auto range = tensor_id_to_info.equal_range(cur_tile_tensor_id);
                        // STALL for THIS TILE OFM
                        TensorInfo info = range.first->second;
                        if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM) {
                            info = getOFMInfo(range);
                            assert(info.tensor_type == TensorType::OFM && info.source == 0);
                            // assert(distance(range.first, range.second) == 1); 
                            if (info.layer_id == layer_id && info.tile_id == tile_id && dram < comp) {
                                dram = comp;
                                break;
                            }
                        }
                        assert(ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
                        if (dram + tensor_access_time > comp) {
                            break;
                        }
                        if (record) {
                            DRAM_Tensor_Info_by_time[dram] = DRAM_Tensor_Info({ info.layer_id, info.tile_id, info.tensor_type, tensor_access_time });
                            dram_tensor_start_cycles[cur_tile_tensor_id] = dram;
                        }
                        dram += tensor_access_time;
                        real_cost.energy += tensor_size * NoC::DRAM_acc_cost;
#ifdef DEBUG
                        std::cout << "Buffer Ready: " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id];
#endif
                        buffer.set_tensor_ready(cur_tile_tensor_id);
#ifdef DEBUG
                        std::cout << " Done!" << std::endl;
#endif
                        tile_tensor_idx++;
                    }
                    // end = std::chrono::steady_clock::now();
                    // total_dram_free_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    // begin = std::chrono::steady_clock::now();
                    // 3. Check if all tensors that the NEXT TILE needs are ready
                    unordered_set<len_t /*tensor_id*/> tensors_for_next_tile;
                    int last_tile_tensor_idx = 0;
                    lid_t next_layer_id = layer_id;
                    len_t next_tile_id = tile_id;
                    if (i == layer_groups.size() - 1 && j == lg.slg_idx_end && tile_id == slg.tile_number - 1 && k == slg.sub_layer_group_end) {
                        assert(pos == total_tile_number - 1);
                        // last tile, next_layer_id == null, next_tile_id == null
                        // load/store all remaining tensors
                        last_tile_tensor_idx = tile_tensor_order.size() - 1;
                        assert(tensors_for_next_tile.empty());
                    } else {
                        // 3.0. Get next_layer_id and next_tile_id
                        // next_layer_pos == k + 1 == (k - slg.sub_layer_group_start + 1) % slg.layer_num() /*(slg.sub_layer_group_end - slg.sub_layer_group_start + 1)*/ + slg.sub_layer_group_start
                        if (k == slg.sub_layer_group_end) {
                            if (tile_id == slg.tile_number - 1) { // next slg's first tile
                                next_layer_id = layer_order_to_id.at(k + 1);
                                next_tile_id = 0;
                            } else {
                                next_layer_id = layer_order_to_id.at(slg.sub_layer_group_start);
                                next_tile_id = tile_id + 1;
                            }
                        } else {
                            next_layer_id = layer_order_to_id.at(k + 1);
                            next_tile_id = tile_id;
                        }
                        // 3.1~3.3. IFMs & WGTs for next tile
                        assert(next_layer_id < network->len());
                        getTensorsForNextTile(tensors_for_next_tile, next_layer_id, next_tile_id);

                        // 3.4. OFM whose end == pos + 1
                        for (int tt_idx = ett_idx; ett_idx < tensor_times.size(); tt_idx++) {
                            if(tt_idx >= tensor_times.size()) break;
                            if (tensor_times[tt_index_sorted_by_end[tt_idx]].end_time == pos + 1) {
                                const TensorInfoSetType ti_set_type = tensor_info_set_types[tt_index_sorted_by_end[tt_idx]];
                                if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM && !buffer.ask_tensor_ready(tt_index_sorted_by_end[tt_idx])) {
                                    tensors_for_next_tile.insert(tt_index_sorted_by_end[tt_idx]);
                                }
                            } else {
                                break;
                            }
                        }
                        // 3.5. Get the last index of the tensor that we need to handle,
                        // which is, the max index in array tile_tensor_order with tile_tensor_order[index] in tensors_for_next_tile
                        // if last_tile_tensor_idx < tile_tensor_idx,
                        //     then we do not need to handle any tensor
                        // if last_tile_tensor_idx >= tile_tensor_idx,
                        //     then we need to handle all tensors from tile_tensor_order[tile_tensor_idx ~ last_tile_tensor_idx]
                        /*last_tile_tensor_idx = tile_tensor_order.rend() - 1 - find_if(tile_tensor_order.rbegin(), tile_tensor_order.rend() - tile_tensor_idx, [&](const len_t& x) {
                            return tensors_for_next_tile.count(x) > 0;
                        });*/
                        last_tile_tensor_idx = tile_tensor_idx;
                        if (tensors_for_next_tile.empty()) {
                            last_tile_tensor_idx--;
                        } else {
                            auto temp_tensors_for_next_tile = tensors_for_next_tile;
                            for (; last_tile_tensor_idx < tile_tensor_order.size(); ++last_tile_tensor_idx) {
                                auto idx = temp_tensors_for_next_tile.find(tile_tensor_order[last_tile_tensor_idx]);
                                if (idx != temp_tensors_for_next_tile.end()) {
                                    temp_tensors_for_next_tile.erase(idx);
                                }
                                if (temp_tensors_for_next_tile.empty())
                                    break;
                            }
                            assert(temp_tensors_for_next_tile.empty());
                        }
                        assert(last_tile_tensor_idx >= tile_tensor_idx || 
                            (tensors_for_next_tile.empty() && last_tile_tensor_idx == tile_tensor_idx - 1));
                    }
#ifdef DEBUG
                    std::cout << "Tensors for Next Tile: ";
                    for (auto& t_id: tensors_for_next_tile) {
                        std::cout << t_id << "@" << tensor_times[t_id] << ", ";
                    }
                    std::cout << std::endl;
#endif
                    // end = std::chrono::steady_clock::now();
                    // total_dependency_find_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    // begin = std::chrono::steady_clock::now();
                    // 4. Then continue to load/store until all dependencies/end==pos tensors are resolved
                    // ONLY those tensors that interact with DRAM might update dram time and envoke buffer_ready:
                    //    - ONLY {1 OFM} tensors, ONLY {1/more WGT...} tensors, ONLY {1/more IFM...} tensors
#ifdef DEBUG
                    std::cout << "DRAM Dependency L/S Stage" << std::endl;
#endif
                    last_tile_unloaded_tensor_idx = last_tile_tensor_idx;
                    while (tile_tensor_idx <= last_tile_tensor_idx) {
                        len_t& cur_tile_tensor_id = tile_tensor_order[tile_tensor_idx];
                        if (!buffer.buffer_has(cur_tile_tensor_id)) {
                            // The rest tiles must have start == pos+1 && they are WGTs or IFMs
                            for (int mid_tt_idx = tile_tensor_idx; mid_tt_idx <= last_tile_tensor_idx; ++mid_tt_idx) {
                                len_t& mid_tile_tensor_id = tile_tensor_order[mid_tt_idx];
                                TensorInfoSetType& mid_ti_set_type = tensor_info_set_types[mid_tile_tensor_id];
                                auto range = tensor_id_to_info.equal_range(mid_tile_tensor_id);
                                bool not_next_tile_dep = false;
                                for (auto it = range.first; it != range.second; ++it) {
                                    if (it->second.layer_id != next_layer_id) {
                                        not_next_tile_dep = true;
                                        break;
                                    }
                                }
                                if (not_next_tile_dep || (tensor_times[mid_tile_tensor_id].start_time != pos + 1) ||
                                    (mid_ti_set_type != TensorInfoSetType::ONLY_WGTs && mid_ti_set_type != TensorInfoSetType::ONLY_IFMs)) {
                                    // std::cout << "Missing " << mid_tile_tensor_id << "@" << tensor_times[mid_tile_tensor_id] << std::endl;
                                    return ErrorType::TENSOR_ORDER_DEPENDENCY;
                                }
                            }
                            // we L/S the rest tiles at the beginning of the next cycle
                            break;
                        }
                        if (tensor_times[cur_tile_tensor_id].start_time > pos || pos > tensor_times[cur_tile_tensor_id].end_time) {
                            // std::cout << "OutBound " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id] << std::endl;
                            return ErrorType::TENSOR_ORDER_DEPENDENCY;
                        }
                        
                        const TensorInfoSetType ti_set_type = tensor_info_set_types[cur_tile_tensor_id];
                        auto range = tensor_id_to_info.equal_range(cur_tile_tensor_id);
                        // STALL for THIS TILE OFM
                        TensorInfo info = range.first->second;
                        if (ti_set_type == TensorInfoSetType::OFM_TO_DRAM) {
                            info = getOFMInfo(range);
                            assert(info.tensor_type == TensorType::OFM && info.source == 0);
                            // assert(distance(range.first, range.second) == 1); 
                            if (info.layer_id == layer_id && info.tile_id == tile_id && dram < comp) {
                                dram = comp;
                            }
                        }
                        vol_t tensor_size = tensor_id_to_size.at(cur_tile_tensor_id);
                        cycle_t tensor_access_time = DIVCEIL(tensor_size, SchNode::DRAM_bw);
                        assert(ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
                        if (record) {
                            DRAM_Tensor_Info_by_time[dram] = DRAM_Tensor_Info({ info.layer_id, info.tile_id, info.tensor_type, tensor_access_time });
                            dram_tensor_start_cycles[cur_tile_tensor_id] = dram;
                        }
                        dram += tensor_access_time;
                        real_cost.energy += tensor_size * NoC::DRAM_acc_cost;
#ifdef DEBUG
                        std::cout << "Buffer Ready: " << cur_tile_tensor_id << "@" << tensor_times[cur_tile_tensor_id];
#endif
                        buffer.set_tensor_ready(cur_tile_tensor_id);
#ifdef DEBUG
                        std::cout << " Done!" << std::endl;
#endif
                        tile_tensor_idx++;
                    }
#ifdef DEBUG
                    if (tile_tensor_idx <= last_tile_tensor_idx) {
                        std::cout << "There are unloaded WGTs/IFMs, typically at LG border" << std::endl;
                    } else {
                        assert(tile_tensor_idx == last_tile_tensor_idx + 1);
                    }
#endif
                    // end = std::chrono::steady_clock::now();
                    // total_dram_force_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    // begin = std::chrono::steady_clock::now();
                    // 5. buffer_del ALL tensors with end == pos + 1 
                    // Because all tile_tensor's start <= end - 1, if End == pos + 1, we have Start <= pos
                    for (; ett_idx < tensor_times.size(); ett_idx++) {
                        const auto& tt_index = tt_index_sorted_by_end[ett_idx];
                        if (tensor_times[tt_index].end_time < pos + 1) {
                            assert(!buffer.buffer_has(tt_index));
                            continue;
                        } else if (tensor_times[tt_index].end_time == pos + 1) {
#ifdef DEBUG
                            std::cout << "Buffer Del: " << tt_index << "@" << tensor_times[tt_index];
#endif
                            const TensorInfoSetType ti_set_type = tensor_info_set_types[tt_index];
                            // DRAM related tensors should be checked before being deleted
                            if (ti_set_type != TensorInfoSetType::OFM_WITH_LOCAL_IFMs && !buffer.ask_tensor_ready(tt_index)) {
#ifdef DEBUG
                                std::cout << " NOT Done!" << std::endl;
#endif
                                std::cout << "Tensor Order Dependency Error @ Tensor " << tt_index << std::endl;
                                return ErrorType::TENSOR_ORDER_DEPENDENCY;
                            } else {
                                buffer.buffer_del(tt_index);
#ifdef DEBUG
                                std::cout << " Done!" << std::endl;
#endif
                            }
                        } else {
                            break;
                        }
                    }
                    if (ett_idx < tensor_times.size()) {
                        assert(tensor_times[tt_index_sorted_by_end[ett_idx]].end_time > pos + 1);
                    }
                    // 6. Sync Here
                    if (!tensors_for_next_tile.empty() && dram > comp) {
                        /* this means comp stall:
                            |      DRAM     |<- sync here
                            | COMP |  STALL |?|   next COMP   |
                        */
                        comp = dram;
                    }
                    // end = std::chrono::steady_clock::now();
                    // total_buffer_del_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
                    pos++;
                }
            }
        }
    }
    // Step 3: Write Back
    assert(buffer.cur_buffer_usage == 0);
    buffer.buffer_usage_by_time[comp] = buffer.cur_buffer_usage;
    // std::cout << "\t\t\tTotal Buffer Add Time: " << (double)total_buffer_add_time / 1e6 << "ms" << std::endl;
    // std::cout << "\t\t\tTotal Comp Add Time: " << (double)total_comp_add_time / 1e6 << "ms" << std::endl;
    // std::cout << "\t\t\tTotal Dependency Find Time: " << (double)total_dependency_find_time / 1e6 << "ms" << std::endl;
    // std::cout << "\t\t\tTotal DRAM Free L/S Time: " << (double)total_dram_free_time / 1e6 << "ms" << std::endl;
    // std::cout << "\t\t\tTotal DRAM Dependency L/S Time: " << (double)total_dram_force_time / 1e6 << "ms" << std::endl;
    // std::cout << "\t\t\tTotal Buffer Del Time: " << (double)total_buffer_del_time / 1e6 << "ms" << std::endl;
    real_cost.time = MAX(comp, dram);
    return ErrorType::SUCCESS;
}

void getStride(const Layer& layer, len_t& stride_h, len_t& stride_w)
{
    if (REF_IS_INSTANCE(layer, ConvLayer)) {
        const ConvLayer& cl = static_cast<const ConvLayer&>(layer);
        stride_h = cl.get_workload().sH;
        stride_w = cl.get_workload().sW;
    } else if (REF_IS_INSTANCE(layer, LRLayer)) {
        const LRLayer& cl = static_cast<const LRLayer&>(layer);
        stride_h = cl.get_workload().sH;
        stride_w = cl.get_workload().sW;
    } else {
        stride_h = 1;
        stride_w = 1;
    }
}

void getStride(const Layer& layer, len_t& stride_h, len_t& stride_w, len_t& kernel_h, len_t& kernel_w)
{
    if (REF_IS_INSTANCE(layer, ConvLayer)) {
        const ConvLayer& cl = static_cast<const ConvLayer&>(layer);
        stride_h = cl.get_workload().sH;
        stride_w = cl.get_workload().sW;
        kernel_h = cl.get_workload().R;
        kernel_w = cl.get_workload().S;
    } else if (REF_IS_INSTANCE(layer, LRLayer)) {
        const LRLayer& cl = static_cast<const LRLayer&>(layer);
        stride_h = cl.get_workload().sH;
        stride_w = cl.get_workload().sW;
        kernel_h = cl.get_workload().R;
        kernel_w = cl.get_workload().S;
    } else {
        stride_h = 1;
        stride_w = 1;
        kernel_h = 1;
        kernel_w = 1;
    }
}

int Graph::check_subgraph_not_fully_connected(const SubLayerGroup& slg, int*& father, int& num_inputs, int& num_nodes) const
{
    auto& dram_ifmaps = slg.dram_ifmaps;
    const auto& sub_layer_group_start = slg.sub_layer_group_start;
    const auto& sub_layer_group_end = slg.sub_layer_group_end;
    unordered_map<int /*layer_id OR ext_id*/, lid_t /*index of dram_ifmaps*/> input_nodes_layer_id_to_idx;
    {
        lid_t kk = 0;
        for (const auto& in : dram_ifmaps) {
            input_nodes_layer_id_to_idx[in.first] = kk++;
        }
    }
    static int fa[MAX_NODE_CNT];
    const int input_cnt = dram_ifmaps.size();
    const int node_cnt = sub_layer_group_end - sub_layer_group_start + 1;
    for (int i = 0; i < input_cnt + node_cnt; i++) {
        fa[i] = i;
    }
    for (int i = 0; i < node_cnt; i++) {
        const auto cur_layer_id = layer_order_to_id[i + sub_layer_group_start];
        for (int input_layer_id : all_layers[cur_layer_id].local_inputs) {
            int _input_idx;
            if (input_layer_id < 0 || layer_id_to_order[input_layer_id] < sub_layer_group_start) {
                assert(input_nodes_layer_id_to_idx.count(input_layer_id));
                _input_idx = -input_nodes_layer_id_to_idx[input_layer_id] - 1;
            } else {
                _input_idx = layer_id_to_order[input_layer_id] - sub_layer_group_start;
            }
            unity(fa, i + input_cnt, _input_idx + input_cnt);
        }
    }
    {
        lid_t kk = 0;
        for (const auto& in : dram_ifmaps) {
            for (int output: in.second.local_outputs) {
                int _output_idx = layer_id_to_order[output] - sub_layer_group_start;
                unity(fa, _output_idx + input_cnt, kk);
            }
            kk++;
        }
    }
    int number_of_connected_components = 0;
    for (int i = 0; i < input_cnt + node_cnt; i++) {
        if (find_root(fa, i) == i) {
            number_of_connected_components++;
        }
    }
    father = fa;
    num_inputs = input_cnt;
    num_nodes = node_cnt;
    return number_of_connected_components;
}

bool Graph::backcalc(SubLayerGroup& slg)
{
    // pure_output_layers: out_degree == 0
    queue<lid_t /*layer_id*/> pure_output_layers;
    for (const auto& p_l_id : slg.pure_outputs_id) {
        pure_output_layers.push(p_l_id);
    }
    unordered_map<lid_t /*layer_id*/, set<int> > slg_local_outputs;
    for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; k++) {
        const auto& cur_layer_id = layer_order_to_id[k];
        slg_local_outputs[cur_layer_id] = all_layers[cur_layer_id].local_outputs;
    }
    int while_cnt = 0;
    while (!pure_output_layers.empty()) {
        while_cnt++;
        int cur_layer_id = pure_output_layers.front();
		const Node& cur_node = network->getNode(cur_layer_id);
		const Layer& cur_layer = cur_node.layer();
        pure_output_layers.pop();
        /*
		traverse cur_layer's all previous layers, may have multiple
		1. when conv's weight is some layer's output, use ofm_to_wgt
		2. multicast multi-Op VP Layer
		3. concat
		*/
	    //  traverse prev
		FOR_BITSET(prev_layer_id, cur_node.getIfmPrevs()) // NO external inputs
        {
            const lid_t& prev_layer_order = layer_id_to_order[prev_layer_id];
            const tensor_shape& prev_layer_ofmap_ts = all_layers[cur_layer_id].tile_size;
            fmap_range prev_layer_ofmap_range = tensor_shape_to_fmap_range(prev_layer_ofmap_ts);
            cur_layer.ofm_to_ifm(prev_layer_ofmap_range);
            if (prev_layer_order >= slg.sub_layer_group_start) { // prev IN local SLG
                all_layers[prev_layer_id].tile_size = fmap_range_to_tensor_shape(tensor_shape_to_fmap_range(all_layers[prev_layer_id].tile_size).combine(prev_layer_ofmap_range));
                slg_local_outputs[prev_layer_id].erase(cur_layer_id);
                // update prev's out_degree
                if (slg_local_outputs.at(prev_layer_id).empty()) {
                    pure_output_layers.push(prev_layer_id);
                }
            } else if (layer_id_to_lg_slg.at(prev_layer_id).first != layer_id_to_lg_slg.at(cur_layer_id).first) { // prev NOT IN local LG
                slg.dram_ifmaps[prev_layer_id].tile_size = fmap_range_to_tensor_shape(tensor_shape_to_fmap_range(slg.dram_ifmaps[prev_layer_id].tile_size).combine(prev_layer_ofmap_range));
            } else { // prev IN local LG but NOT IN local SLG
                ;
            }
        }
        if (cur_node.hasWgtPrevs()) {
            FOR_BITSET(prev_layer_id, cur_node.getWgtPrevs()) // NO external inputs
            {
                const lid_t& prev_layer_order = layer_id_to_order[prev_layer_id];
                const tensor_shape& prev_layer_ofmap_ts = all_layers[cur_layer_id].tile_size;
                fmap_range prev_layer_ofmap_range = tensor_shape_to_fmap_range(prev_layer_ofmap_ts);
                cur_layer.ofm_to_wgt(prev_layer_ofmap_range);
                if (prev_layer_order >= slg.sub_layer_group_start) { // prev IN local SLG
                    all_layers[prev_layer_id].tile_size = fmap_range_to_tensor_shape(tensor_shape_to_fmap_range(all_layers[prev_layer_id].tile_size).combine(prev_layer_ofmap_range));
                    int suc = slg_local_outputs[prev_layer_id].erase(cur_layer_id);
                    // update prev's out_degree
                    if (slg_local_outputs[prev_layer_id].empty()) {
                        pure_output_layers.push(prev_layer_id);
                    }
                } else if (layer_id_to_lg_slg.at(prev_layer_id).first != layer_id_to_lg_slg.at(cur_layer_id).first) { // prev NOT IN local LG
                    slg.dram_ifmaps[prev_layer_id].tile_size = fmap_range_to_tensor_shape(tensor_shape_to_fmap_range(slg.dram_ifmaps[prev_layer_id].tile_size).combine(prev_layer_ofmap_range));
                } else { // prev IN local LG but NOT IN local SLG
                    ;
                }
            }
        }
        if (cur_node.hasExtPrevs()) {
            FOR_BITSET(prev_layer_id, cur_node.getExtPrevs()) // NO external inputs
            {
                const lid_t& prev_layer_order = layer_id_to_order[prev_layer_id];
                const tensor_shape& prev_layer_ofmap_ts = all_layers[cur_layer_id].tile_size;
                fmap_range prev_layer_ofmap_range = tensor_shape_to_fmap_range(prev_layer_ofmap_ts);
                cur_layer.ofm_to_ifm(prev_layer_ofmap_range);
                slg.dram_ifmaps[prev_layer_id].tile_size = fmap_range_to_tensor_shape(tensor_shape_to_fmap_range(slg.dram_ifmaps[prev_layer_id].tile_size).combine(prev_layer_ofmap_range));
            }
        }
    }
    assert(while_cnt == slg.sub_layer_group_end - slg.sub_layer_group_start + 1);
    return true;
}

CoreMapper::CoreMapping Graph::getTileCost(lid_t layer_id, const tensor_shape& tensor) const
{
    const Node& layerT = network->getNode(layer_id);
    const Layer& layer = layerT.layer();
    return layerMapper->Core_tensor_explore_full(layer, tensor, layerT.hasWgtPrevs());
}
