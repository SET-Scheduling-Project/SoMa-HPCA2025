#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <array>
#include <vector>
#include <functional>
#include <unordered_map>
#include "network.h"
#include "graph.h"
#include "bitset.h"
#include "core.h"
#include "coremapping.h"
#include "layerengine.h"
#include "network.h"
#include "utils.h"

// FROM DEUS GRAPH.H
#define MAX_NODE_CNT 1024  // for static arrays

// Encoding Format:
/*
    struct Encoding {
        vector<lid_t> layer_order_to_id;
        // Arranged by layer order |
        Bitset layer_group_partition;
        Bitset sub_layer_group_partition;
        vector<tensor_shape> tile_sizes; // 1 x 1 ~ full_height x full_width
    };
*/



template<>
struct std::hash<Bitset> {
    std::size_t operator()(const Bitset& bitset) const {
        std::size_t seed = 0;
        FOR_BITSET(i, bitset) {
            seed ^= std::hash<Bitset::bitlen_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// template<>
// struct std::hash<tensor_shape> {
//     std::size_t operator()(const tensor_shape& shape) const {
//         std::size_t seed = 0;
//         seed ^= std::hash<len_t>{}(shape.bk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         seed ^= std::hash<len_t>{}(shape.c) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         seed ^= std::hash<len_t>{}(shape.h) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         seed ^= std::hash<len_t>{}(shape.w) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         seed ^= std::hash<vol_t>{}(shape.size) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         return seed;
//     }
// };

template<>
struct std::hash<Graph::Stage1Encoding> {
    std::size_t operator()(const Graph::Stage1Encoding& encoding) const {
        std::size_t seed = 0;
        for (auto& i : encoding.layer_order_to_id) {
            seed ^= std::hash<lid_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        seed ^= std::hash<Bitset>{}(encoding.layer_group_partition) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<Bitset>{}(encoding.sub_layer_group_partition) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        for (auto& i : encoding.tile_numbers) {
            seed ^= std::hash<len_t>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct OptimizationStatistics {
    struct sa_cnt_type {
        unsigned long long better_cnt, acc_cnt, val_cnt, all_cnt;
    };
    struct cache_stat_type {
        unsigned long long hit_cnt, miss_cnt, total_cnt;
    };
    std::array<sa_cnt_type, 5> sa_cnts;
    cache_stat_type cache_stats;
    int64_t search_time;
    double avg_buffer_usage;
};
std::ostream& operator<<(std::ostream& os, const OptimizationStatistics& stats);

// Base class for optimization
class OptimizationStage
{
public:
    // Network structure
    const Network* network;

    OptimizationStage(const Network* _network);
    virtual ~OptimizationStage() = default;
    // SA accept with probability
    inline bool accept(const cost_t& old_cost, const cost_t& new_cost, const double& temperature) const;
    inline bool accept_by_progress(const cost_t& old_cost, const cost_t& new_cost, const double progress) const;
    // return an integer in the range [rand_min, rand_max] (both inclusive)
    int randint(const int& rand_min, const int& rand_max) const;
    // return an integer in the range [start, mid) U (mid, end]
    int randint_except(const int& start, const int& mid, const int& end) const;
    // return true with probability prob, else return false
    bool rand_prob(const double& prob) const;
    // return a random index i with probability prob[i], i = 0, 1, ..., num - 1
    int rand_multi_prob(const std::initializer_list<double> prob) const;
    int rand_multi_prob(const std::vector<double> prob) const;
    // return a random index i in [l, m) U (m, r] with 'semi-linear' probability, we want P(i) to be higher near m
    // 'semi-linear' means P(i) in [l, m) and (m, r] are both linear(decreasing) to i, and they virtually meet at m
    // here we use (x-l)/(m-l) in [l, m) and (x-r)/(m-r) in (m, r]
    int randint_except_semi_linear1(const int& l, const int& m, const int& r) const;
    // here we use a simple linear function: P(i) = 1 - abs(i - m) / (r - l)
    int randint_except_semi_linear2(const int& l, const int& m, const int& r) const;
    // here we use Aitken interpolation: P(i) = (x - l)(x - r)/((m - l)(m - r))
    int randint_except_semi_linear3(const int& l, const int& m, const int& r) const;

};


class Stage1SA: public OptimizationStage
{
public:
    struct changeable_layer {
        lid_t order;
        lid_t layer_id;
        lid_t prev;
        lid_t next;
    };
    Graph last_g;
    uint8_t baseline_type;
    Graph::Stage1Encoding enc;
    Graph::Stage1Encoding last_enc;
    Graph::Stage1Encoding best_enc;
    CoreMapper::MapCost best_cost;
    vector<changeable_layer> changeable_layers;
    void update_changeable_layers(const Graph& g);
    // Operators which applys to encodings
    // Change layer order
    bool change_layer_order(const Graph& g);
    // Change one slg's tile_number
    bool change_tile_number(const Graph& g);
    // Change layer group partition
    bool change_layer_group(const Graph& g);
    // Change sub layer group partition
    bool change_sub_layer_group(const Graph& g);
    // check if all slgs are fully connected, if not, cut them apart.
    bool make_sure_all_slgs_are_fully_connected(const Graph& g);
    // Change layer order without slg change
    bool change_layer_order_baseline(const Graph& g);
    // Change layer group partition by layer and no slg change
    bool change_layer_group_by_layer(const Graph& g);
    bool change_layer_group_by_layer_baseline(const Graph& g);
    // check if all lgs are fully connected, if not, cut them apart.
    bool make_sure_all_lgs_are_fully_connected(const Graph& g);


    Stage1SA(const Network* _network, const Graph::Stage1Encoding _default_encoding, const uint8_t _baseline_type);
    bool sa_init(const Graph::Stage1Encoding_tile_sizes& enc_tz, const bool use_tile_sizes, const bool print_results, const bool use_small_tile_nums=true);
    ~Stage1SA() = default;
    inline bool accept_by_progress(const cost_t& old_cost, const cost_t& new_cost, const bool let_dram_tensor_num_decrease, const len_t& dram_tensor_num, const double progress) const;

    // Search algorithm
    OptimizationStatistics solve(len_t num_rounds);

};

class Stage2SA: public OptimizationStage
{
public:
    struct tensor_constraint_range {
        int min;     // prev OR start/end's min range (contained)
        int max;     // next OR start/end's max range (contained)
        tensor_constraint_range intersect_with(const tensor_constraint_range& other) const;
    };
    struct change_scheme {
        int tensor_id;
        int old_sch;
        int new_sch;
    };
    /*     Auxiliary Data Structures     */
    // Dynamic during SA, need update after each change
    unordered_map<len_t /*tensor_id*/, len_t /*tensor_order*/> tile_tensor_id_to_order;
    std::vector<int> max_start;
    std::vector<int> min_end;
    change_scheme cur_change;
    long long buffer_time_saved;
    //  unordered_map<len_t /*tensor_id*/, tensor_range> tensor_ranges;
    //  map<int /*the ith start's value*/, len_t /*tile pos with ith start*/> start_val_to_pos;
    //  map<len_t /*tile pos with ith start*/, int /*the ith start's value*/> start_pos_to_val;
    
    // Static during SA, initialized only once at the beginning
    struct SimpleGraph {
        std::vector< set<len_t> > forward;   // forward graph
        std::vector< set<len_t> > reverse;   // reverse graph
        SimpleGraph() = default;
        ~SimpleGraph() = default;
        void init(int n);
        void add_edge(int from, int to);
        void del_edge(int from, int to);
        void del_single_direct_edge(int from, int to, bool is_forawrd);
        void print_graph() const;
        void print_nodes(const Graph& g) const;
        void print_edges() const;
    };
    struct SparseGraph {
        std::unordered_map<len_t, vector<len_t> > forward;   // forward graph
        std::unordered_map<len_t, vector<len_t> > reverse;   // reverse graph
        SparseGraph() = default;
        ~SparseGraph() = default;
        void add_edge(int from, int to);
        void print_graph() const;
    };
    SimpleGraph hasse_diag;
    SparseGraph across_lg_constraints;
    SparseGraph IFM_tiles_constraints;
    SparseGraph OFM_tiles_constraints;
    size_t num_dram_tiles, total_tile_number;
    std::unordered_map<len_t /*tensor_id*/, len_t/*tile_pos*/> dram_tensor_id_to_tile_pos;
    std::vector<double /*tensor_size*/> dram_tensor_id_to_size; // if not in DRAM, set to 0

    /*     SA variables     */
    Graph last_g;
    Graph::Stage2Encoding enc;
    Graph::Stage2Encoding last_enc;
    Graph::Stage2Encoding best_enc;
    CoreMapper::MapCost best_cost;
    CoreMapper::MapCost s1_ideal_cost;
    __uint128_t best_sum_buffer_usage;
    __uint128_t ideal_sum_buffer_usage;
    // const Stage1SA* cur_stage1;

    // SA accept with probability
    inline bool accept(const cost_t& old_cost, const cost_t& new_cost, const cost_t& ideal_cost, const double& temperature) const;
    inline bool accept_by_progress(const cost_t& old_cost, const cost_t& new_cost, const cost_t& ideal_cost, const double progress) const;
    inline bool accept_by_progress_and_buffer_usage(const cost_t& old_cost, const cost_t& new_cost, const __uint128_t& old_sum_buffer_usage, const __uint128_t& new_sum_buffer_usage,  const __uint128_t& ideal_sum_buffer_usage, const double progress) const;

    /*     Initialize functions     */
    Stage2SA(const Network* _network, const CoreMapper::MapCost& stage1_best_cost, const Graph& stage1_best_g);
    ~Stage2SA() = default;
    void init_Hasse_Diag(const Graph& g);
    void compact_Hasse_Diag(const Graph& g);

    /*     Update functions     */
    // Lazy get tensor_range after picking a tensor
    tensor_constraint_range get_tensor_order_range(const Graph& g, const len_t& tensor_id);
    tensor_constraint_range get_tensor_time_range(const Graph& g, const len_t& tensor_id);
    
    /*     Mutation methods     */
    // Randomly pick a tensor and change its schedule
    bool change_tensor_order_and_time(const Graph& g);
    // Randomly pick a tensor and insert it into a random position in the valid range
    bool change_tensor_order(const Graph& g);
    // Randomly pick a tensor and change its start/end time
    bool change_tensor_time(const Graph& g, const bool single_direction_change);
    
    /*     Search algorithm     */
    std::array<OptimizationStatistics, 2> solve(const len_t num_rounds, const bool opt_buffer_usage);

    /*     Statistics Results     */
    std::tuple<access_t, double, double> get_encoding_changes(const Graph::Stage2Encoding& original_enc);
};

std::ostream& operator<<(std::ostream& os, const Stage2SA::tensor_constraint_range& tr);

#endif // OPTIMIZATION_H