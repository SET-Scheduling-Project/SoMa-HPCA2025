#ifndef GRAPH_H
#define GRAPH_H

#include "bitset.h"
#include "core.h"
#include "coremapping.h"
#include "layerengine.h"
#include "network.h"
#include "utils.h"
#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <functional>
#include <vector>

#define KB *1024
#define MB *1024 * 1024
using namespace std;

struct pair_hash
{
	template <class T1, class T2>
	std::size_t operator()(const std::pair<T1, T2> &p) const 
    {
        std::size_t seed = 0;
        seed ^= std::hash<T1>{}(p.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<T2>{}(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		// main hash function
        return seed;
	}
};
//
// struct pair_equal
// {
// 	template <class T1, class T2>
// 	bool operator()(const std::pair<T1, T2> &lhs, const std::pair<T1, T2> &rhs) const
// 	{
// 		return lhs.first == rhs.first && lhs.second == rhs.second;
// 	}
// };

template<>
struct std::hash<tensor_shape> 
{
    std::size_t operator()(const tensor_shape& shape) const 
    {
        std::size_t seed = 0;
        seed ^= std::hash<len_t>{}(shape.bk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<len_t>{}(shape.c) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<len_t>{}(shape.h) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<len_t>{}(shape.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

enum class ErrorType {
    SUCCESS,
    LAYER_ORDER_DEPENDENCY,
    SUBGRAPH_NOT_FULLY_CONNECTED,
    BUFFER_OVERFLOW,
    TENSOR_ORDER_DEPENDENCY,
    ADDRESS_OVERFLOW, 
    TILE_NUMBER_TOO_LARGE,
};

class Graph {
public:
    enum class TensorType {
        IFM,
        WGT,
        OFM
    };
    enum class TensorInfoSetType {
        ONLY_WGTs,
        ONLY_IFMs,
        OFM_TO_DRAM, // could have ONLY 1 OFM, OR could have some local IFMs, but still have to access DRAM due to out-of-LG IFMs
        OFM_WITH_LOCAL_IFMs
    };
    enum class SLGTransposeType {
        ONLY_CH, // slg only has CH transpose
        ONLY_CW, // slg only has CW transpose
        CH_AND_CW, // slg has both CH and CW transpose
        NONE // NONE means no transpose or slg only has HW transpose
    };
    struct TensorTime {
        int start_time;
        int end_time;
        TensorTime() : start_time(0), end_time(0) {}
        TensorTime(const int& start, const int& end);
        bool operator<(const TensorTime& other) const;
        bool operator==(const TensorTime& other) const;
        void expand(const int& start, const int& end);
    };
    struct TensorInfo {
        lid_t layer_id;
        len_t tile_id;
        TensorType tensor_type;
        int source;
        // NOTE on source:
        // for IFM: < 0 means -ext_id of external input, >=0 means layer_id of internal IFM/Transformer WGT
        // for CNN WGT and OFM: always == 0
        // may have multiple tensor_id for the same layer, tile, and type.
        vol_t get_layer_size(const len_t& bk) const;
        bool operator<(const TensorInfo&) const;
        bool operator==(const TensorInfo&) const;
    };
    class TensorInfoHash {
    public:
        size_t operator()(const TensorInfo& t) const;
    };
    struct Buffer {
        static thread_local vol_t MAX_BUFFER_SIZE;
        static thread_local double STAGE_1_LIMIT_RATIO;
        static thread_local bool IS_STAGE_1;
        vol_t cur_buffer_usage, max_buffer_usage;
        unordered_map<len_t /*tensor_id*/, vol_t /*tensor_size*/> tensors_in_buffer;
        unordered_set<len_t /*tensor_id*/> tensor_ready;
        map<cycle_t /*time*/, vol_t /*buffer_usage*/> buffer_usage_by_time;
        vector<vol_t /*usage*/> buffer_usage_by_tile;
        Buffer(): cur_buffer_usage(0), max_buffer_usage(0) {}
        ~Buffer() { }
        static void set_stage_1_limit_ratio(const double& ratio) { STAGE_1_LIMIT_RATIO = ratio; }
        static void set_stage_1(const bool& is_stage_1) { IS_STAGE_1 = is_stage_1; }
        static void set_max_buffer_size(const vol_t& size) { MAX_BUFFER_SIZE = size; }
        static vol_t get_max_buffer_size() { if (IS_STAGE_1) return MAX_BUFFER_SIZE * STAGE_1_LIMIT_RATIO; else return MAX_BUFFER_SIZE; }
        bool set_tensor_ready(const len_t& tensor_id);
        bool ask_tensor_ready(const len_t& tensor_id) const;
        bool buffer_has(const len_t& tensor_id) const;
        bool buffer_add(const len_t& tensor_id, const vol_t& tensor_size);
        bool buffer_del(const len_t& tensor_id);
        bool buffer_del(const list<len_t>& tensor_id, len_t& del_size);
        void clear();
    };
    // SubLayer: layer with a specific tile size
    struct SubLayer { // lid_t layer_id; is the index of all_layers
        tensor_shape tile_size;
        // len_t delta[2];
        // len_t step_num[2];
        set<int /*layer_id OR ext_id*/> local_inputs, local_outputs;
    };
    struct SubLayerGroup {
        len_t tile_number;
        lid_t sub_layer_group_start; // idx of the first sub_layer in layer order
        lid_t sub_layer_group_end; // idx of the last sub_layer in layer order
        map<int /*layer_id OR ext_id*/, SubLayer> dram_ifmaps; // dram ifmaps of this sub_layer_group
        // if dram_ifmaps.layer_id < 0, it is external input
        vector<lid_t /*layer_id*/> pure_outputs_id;
        // double BandWidth_allowence, Buffer_allowence;
        // vector<len_t /*tensor_id*/> req_list_f, req_list_b; // forward and backward
        // this func is just for convenience
        inline lid_t layer_num() const { return sub_layer_group_end - sub_layer_group_start + 1; }
    };
    struct LayerGroup {
        lid_t slg_idx_start, // == first index of this lg's slg in all_slgs
              slg_idx_end; // == last index of this lg's slg in all_slgs
        lid_t layer_group_start, // == all_slgs[slg_idx_start].sub_layer_group_start
              layer_group_end; // == all_slgs[slg_idx_end].sub_layer_group_end
    };
    struct Stage1Encoding {
        vector<lid_t /*idx order*/> layer_order_to_id;
        Bitset layer_group_partition;
        Bitset sub_layer_group_partition;
        vector<len_t> tile_numbers; // index == slg_id
        bool operator==(const Stage1Encoding& other) const;
    };
    struct Stage1Encoding_tile_sizes {
        vector<lid_t /*idx order*/> layer_order_to_id;
        Bitset layer_group_partition;
        Bitset sub_layer_group_partition;
        vector<tensor_shape> tile_sizes; // index == layer_id
        bool operator==(const Stage1Encoding_tile_sizes& other) const;
    };
    struct Stage2Encoding {
        vector<TensorTime> tensor_times; // order matters
        vector<len_t/*tensor_id*/> tile_tensor_order;
        bool operator==(const Stage2Encoding& other) const;
    };
    struct DRAM_Tensor_Info // for DRAM bandwidth
    {
        lid_t layer_id;
        len_t tile_id;
        TensorType tile_tensor_type;
        cycle_t tensor_access_time;
    };
    struct COMP_Tile_Info // for computing time
    {
        lid_t layer_id;
        len_t tile_id;
        cycle_t tile_comp_time;
    };
    struct IdealCostResults {
        double ideal_comp, ideal_dram, comp_energy, ubuf_energy, buffer_energy, noc_energy, mac_energy, dram_energy;
    };
    static LayerEngine* layerMapper;
    // FOR IR
    vector<LayerGroup> layer_groups;
    vector<lid_t /*idx order*/> layer_id_to_order;
    vector<lid_t /*layer_id*/> layer_order_to_id;
    vector< pair<lid_t /*LG id*/, lid_t /*SLG id*/> > layer_id_to_lg_slg; // layer_id to layer_group_id
    vector<TensorTime /*first_tile_pos, last_tile_pos*/> layer_id_to_tile_pos; // auxiliary data structure for layer order
    vector<SubLayer> all_layers; // idx == layer_id, size==network->len();
    // vector<Bitset> dirp_set; // Direct Prevs Set, idx == layer_id
    vector<SubLayerGroup> all_slgs; // idx == slg_id, following computing order
    vector<TensorTime> tensor_times; // vector index == lid_t tensor_id; // global tensor id, different data has different id
    unordered_multimap<TensorInfo, len_t /*tensor_id*/, TensorInfoHash> tensor_info_to_id; // This is NOT DONE in each TensorInfo
    unordered_multimap<len_t /*tensor_id*/, TensorInfo> tensor_id_to_info; // tensor_id == idx of tensor_times
    vector<vol_t /*tensor_size*/> tensor_id_to_size; // only used in one getRealCost(), next time in initTileCosts() it will be cleared
    
    vector<len_t /*tensor_id*/> tile_tensor_order; // Contains tensors that interact with DRAM
    // INTERNAL DATA
    vector<TensorInfoSetType> tensor_info_set_types; // idx == tensor_id
    Buffer buffer;
    map<cycle_t, DRAM_Tensor_Info> DRAM_Tensor_Info_by_time;
    map<cycle_t, COMP_Tile_Info> COMP_Tile_Info_by_time;
    unordered_map< pair<lid_t /*layer_id*/, len_t/*tile_id*/>, cycle_t /*tile COMP start cycle*/, pair_hash> tile_start_cycles;
    unordered_map<len_t /*tensor_id*/, cycle_t /*tensor DRAM L/S start cycle*/> dram_tensor_start_cycles;
    unordered_map< pair<lid_t /*layer_id*/, tensor_shape /*tile_size*/>, CoreMapper::MapCost, pair_hash> tile_costs; // must be initialized by initTileCosts first
    static thread_local len_t num_tile_cost_cache_hit, num_tile_cost_cache_miss, num_tile_cost_cache_total;
    static bool has_prefetch;
    static vector<lid_t /*layer_id*/> transpose_layer_ids;
    static vector<lid_t /*layer_id*/> fc_layer_ids;

    Graph() {};
    static void get_special_layer_ids() {
        transpose_layer_ids.clear();
        fc_layer_ids.clear();
        // must follow a layer_id increasing order, for binary search
        for (int layer_id = 0; layer_id < network->len(); layer_id++) {
            const Node& n = network->getNode(layer_id);
            const Layer& l = n.layer();
            if (REF_IS_INSTANCE(l, FCLayer)) {
                fc_layer_ids.emplace_back(layer_id);
            } else if (REF_IS_INSTANCE(l, TransposeLayer)) {
                transpose_layer_ids.emplace_back(layer_id);
            } else {
                continue;
            }
        }
    }
    static void resetCacheCounter() { num_tile_cost_cache_hit = num_tile_cost_cache_miss = num_tile_cost_cache_total = 0; }
    static void setHasPrefetch(bool _has_prefetch) { has_prefetch = _has_prefetch; }
    ErrorType init_stage_1(const Stage1Encoding& s1enc);
    ErrorType init_stage_1_with_tile_sizes(const Stage1Encoding_tile_sizes& s1enc_tz);
    ErrorType init_stage_1_order(const Stage1Encoding& s1enc);
    ErrorType init_stage_1_order_with_tile_sizes(const Stage1Encoding_tile_sizes& s1enc_tz);
    ErrorType init_stage_1_partition(const Stage1Encoding& s1enc);
    ErrorType init_stage_1_partition_with_tile_sizes(const Stage1Encoding_tile_sizes& s1enc_tz);
    ErrorType init_stage_1_back_cal();
    ErrorType init_stage_1_back_cal_and_change_tile_sizes();
    ErrorType init_stage_1_tile_pos();
    ErrorType init_stage_1_tensor_times();
    ErrorType init_stage_2(const Stage2Encoding& s2enc, const bool update_dram_order, const bool update_tensor_time);
    void initTileCosts();
    bool cut_into_tiles(tensor_shape& tile_size, const len_t& tile_number, SLGTransposeType avoid_dims); 
    bool check_layer_order_valid() const;
    bool check_buffer_valid();
    IdealCostResults getIdealCost(CoreMapper::MapCost& ideal_cost, const bool print_results) const;
    ErrorType getRealCost(CoreMapper::MapCost& real_cost, bool record, bool do_check=true);
    inline void getTensorsForNextTile(unordered_set<len_t /*tensor_id*/> &ts, const len_t& next_layer_id, const len_t& next_tile_id) const;
#ifdef DEBUG
    vol_t get_tensor_size(const len_t& tensor_id) const;
#endif
    CoreMapper::CoreMapping getTileCost(lid_t layer_id, const tensor_shape& tensor) const;
    int check_subgraph_not_fully_connected(const SubLayerGroup& slg, int*& father, int& num_inputs, int& num_nodes) const; 
    bool backcalc(SubLayerGroup& slg); // return false if not fully connected
    void get_intensity(vector<vol_t>& lg_comp_time, vector<vol_t>& lg_dram_time, vector<vol_t>& slg_comp_time, vector<vol_t>& slg_dram_time) const;
    void print_intensity() const;
    void print_tile_group_info() const;
    // for searcher
    pair<Stage1Encoding, Stage2Encoding> get_Encoding() const;
    template <typename Iterator>
    auto getOFMInfo(std::pair<Iterator, Iterator> range) const -> decltype(range.first->second) {
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second.tensor_type == TensorType::OFM) {
                return it->second;
            }
        }
        assert(false && "No OFM node found in range");
    }
    void print_graph_result() const;
    __uint128_t get_sum_buffer_usage() const;
};
std::ostream& operator<<(std::ostream& os, const Graph::Stage1Encoding& enc);
std::ostream& operator<<(std::ostream& os, const Graph::Stage2Encoding& enc);
std::ostream& operator<<(std::ostream& os, const ErrorType& err_type);
std::ostream& operator<<(std::ostream& os, const Graph::TensorType& t_type);
std::ostream& operator<<(std::ostream& os, const Graph::TensorInfoSetType& ti_set_type);
std::ostream& operator<<(std::ostream& os, const Graph::TensorTime& t_time);
std::ostream& operator<<(std::ostream& os, const Graph::TensorInfo& t_info);
std::ostream& operator<<(std::ostream& os, const Graph::Buffer& b);
std::ostream& operator<<(std::ostream& os, const Graph::SubLayer& sl);
std::ostream& operator<<(std::ostream& os, const Graph::SubLayerGroup& slg);
std::ostream& operator<<(std::ostream& os, const Graph::LayerGroup& lg);
std::ostream& operator<<(std::ostream& os, const Graph g);
std::ostream& operator<<(std::ostream& os, const Graph::IdealCostResults& icr);
Graph::Stage1Encoding parseInput(const std::vector<std::string>& input);
std::vector< std::pair<double, Graph::Stage1Encoding> > parseAllInputs(std::istream& in);
inline void insertChars(std::ostream& os, int count, string c);
void print_avg_buffer_usage(const Graph& g, const CoreMapper::MapCost& cost);

void getStride(const Layer& layer, len_t& stride_h, len_t& stride_w);
void getStride(const Layer& layer, len_t& stride_h, len_t& stride_w, len_t& kernel_h, len_t& kernel_w);
inline len_t toIfm(const len_t& x, const len_t& stride, const len_t& kernel);

#undef KB
#undef MB
#endif // GRAPH_H
