#include "optimization.h"
#include <ctime>
#include <array>
#include <chrono>
#define CMAX(x,y) do { \
    __typeof__(x) _x = (x); \
    __typeof__(y) _y = (y); \
    (x) = MAX(_x, _y); \
} while(0)

#define CMIN(x,y) do { \
    __typeof__(x) _x = (x); \
    __typeof__(y) _y = (y); \
    (x) = MIN(_x, _y); \
} while(0)

std::ostream& operator<<(std::ostream& os, const Stage2SA::tensor_constraint_range& tr) 
{
    os << "[" << tr.min << "~" << tr.max << "]";
    return os;

}

std::ostream& operator<<(std::ostream& os, const OptimizationStatistics& stats)
{
    os << "Search Time: " << stats.search_time << "s" << std::endl;
    // os << "Average Buffer Usage: " << stats.avg_buffer_usage << std::endl;
    os << "Tile Cost Cache Access:" << stats.cache_stats.total_cnt << 
            ", Hit: " << stats.cache_stats.hit_cnt << 
            ", Miss: " << stats.cache_stats.miss_cnt << 
            ", Miss Rate: " << stats.cache_stats.miss_cnt*100.0/(double)stats.cache_stats.total_cnt << "%" << std::endl;
    assert (stats.sa_cnts.size() <= 5);
    for (int i = 0; i < MIN(stats.sa_cnts.size(), 5); i++) {
        const auto & sa_cnt = stats.sa_cnts.at(i);
        if (sa_cnt.all_cnt == 0) {
            os << "Op" << i << ": No operation applied" << std::endl;
            continue;
        } else {
            os << "Op" << i 
                << ": Better=" << sa_cnt.better_cnt 
                << ", Accept=" << sa_cnt.acc_cnt 
                << ", Valid=" <<  sa_cnt.val_cnt 
                << ", All=" << sa_cnt.all_cnt 
                << "; Better/All=" << sa_cnt.better_cnt*100/sa_cnt.all_cnt 
                << "%, Acc/All=" << sa_cnt.acc_cnt*100.0/sa_cnt.all_cnt 
                << "%, Val/All=" << sa_cnt.val_cnt*100.0/sa_cnt.all_cnt << "%" << std::endl;
        }
    }
    return os;
}


int OptimizationStage::randint(const int& rand_min, const int& rand_max) const
{
    // assert(rand_max - rand_min + 1 > 0);
    std::uniform_int_distribution<> dist {rand_min, rand_max};
    return dist(gen);
}

int OptimizationStage::randint_except(const int& start, const int& mid, const int& end) const
{
    assert(start <= mid && mid <= end);
    int ret = randint(0, end - start - 1);
    if (ret >= (mid - start))
        ret++;
    return start + ret;
}

int OptimizationStage::randint_except_semi_linear1(const int& l, const int& m, const int& r) const
{
    // here we use (x-l)/(m-l) in [l, m) and (x-r)/(m-r) in (m, r]
    vector<double> prob(r - l + 1);
    for (int i = l; i < m; i++)
        prob[i - l] = (double)(i - (l-1)) / (double)(m - (l-1));
    for (int i = m; i <= r; i++)
        prob[i - l] = (double)((r+1) - i) / (double)((r+1) - m);
    prob[m - l] = 0.0;
    return l + rand_multi_prob(prob);
}

int OptimizationStage::randint_except_semi_linear2(const int& l, const int& m, const int& r) const
{
    // here we use a simple linear function: P(i) = 1 - abs(i - m) / (r - l)
    vector<double> prob(r - l + 1);
    for (int i = l; i <= r; i++)
        prob[i - l] = 1.0 - (double)abs(i - m) / (double)((r+1) - (l-1));
    prob[m - l] = 0.0;
    return l + rand_multi_prob(prob);
}

int OptimizationStage::randint_except_semi_linear3(const int& l, const int& m, const int& r) const
{
    // here we use Aitken interpolation: P(i) = (x - l)(x - r)/((m - l)(m - r))
    vector<double> prob(r - l + 1);
    for (int i = l; i <= r; i++)
        prob[i - l] = (double)(i - (l-1)) * (double)((r+1) - i) / (double)((m - (l-1)) * ((r+1) - m));
    prob[m - l] = 0.0;
    return l + rand_multi_prob(prob);
}

bool OptimizationStage::rand_prob(const double& prob) const
{
    auto bd = std::bernoulli_distribution{prob};
    return bd(gen);
}

int OptimizationStage::rand_multi_prob(const std::initializer_list<double> prob) const
{
    std::discrete_distribution<int> dd{prob};
    return dd(gen);
}

int OptimizationStage::rand_multi_prob(const std::vector<double> prob) const
{
    std::discrete_distribution<int> dd(prob.begin(), prob.end());
    return dd(gen);
}

OptimizationStage::OptimizationStage(const Network* _network)
: network(_network) {
}

inline bool OptimizationStage::accept(const cost_t& old_cost, const cost_t& new_cost, const double& temperature) const
{  // TODO
    if (new_cost < old_cost) {
        return true;
    } else {
        double prob = std::exp(((double)(old_cost - new_cost) / (double)old_cost) / temperature);
        return rand_prob(prob);
    }
}

inline bool OptimizationStage::accept_by_progress(const cost_t& old_cost, const cost_t& new_cost, const double progress) const
{
    if (new_cost < old_cost) {
        return true;
    } else {
        /*
		 * T(x) = a+c/(b+x)
		 * a + c/b = 0.1
		 * a + c/(b+0.5) = 0.01         # 0.001
		 * a + c/(b+1) = 0
		 * a = -1/80                    # -1/980
		 * b = 1/8                      # 1/98
		 * c = 9 / 640                  # 99/(980*98)
		 * T(x) = 1/10 * (1-x)/(1+8x)   # 1/10 * (1-x)/(1+98x)
		 */
        double T = 0.1 * (1.0 - progress) / (1.0 + 8.0 * progress);
        double prob = std::exp(((double)(old_cost - new_cost) / (double)old_cost) / T);
        return rand_prob(prob);
    }
}

inline bool Stage1SA::accept_by_progress(const cost_t& old_cost, const cost_t& new_cost, const bool let_dram_tensor_num_decrease, const len_t& dram_tensor_num, const double progress) const
{
    if (new_cost < old_cost) {
        return true;
    } else {
        /*
		 * T(x) = a+c/(b+x)
		 * a + c/b = 0.1
		 * a + c/(b+0.5) = 0.01         # 0.001
		 * a + c/(b+1) = 0
		 * a = -1/80                    # -1/980
		 * b = 1/8                      # 1/98
		 * c = 9 / 640                  # 99/(980*98)
		 * T(x) = 1/10 * (1-x)/(1+8x)   # 1/10 * (1-x)/(1+98x)
		 */
        /*if (let_dram_tensor_num_decrease) {
            if((double)abs(old_cost - new_cost) / (double)old_cost < 1e-4) {
                if (dram_tensor_num < last_g.tile_tensor_order.size()) {
                    return true;
                }
            }
        }*/
        double T = 0.1 * (1.0 - progress) / (1.0 + 8.0 * progress);
        double prob = std::exp(((double)(old_cost - new_cost) / (double)old_cost) / T);
        return rand_prob(prob);
    }
}

inline double flat(double x)
{
    if (x >= -1e-9)
        return 1.0;
    else
        return 1.0 - std::exp(-1.0 / (double)(120.0*120.0*x*x));
}

inline double sigmod(double x)
{
    return 1.0 / (1.0 + std::exp(- (double)100 * log(99.0) * x));
}

inline bool Stage2SA::accept(const cost_t& old_cost, const cost_t& new_cost, const cost_t& ideal_cost, const double& temperature) const
{  // TODO
    if (new_cost < old_cost) {
        return true;
    } else {
        /*if ((double)(old_cost - new_cost) / (double)old_cost <= 1e-4) {
            double buffer_prob = sigmod(((double)buffer_time_saved / (double)last_g.buffer.get_max_buffer_size()) / (double)(total_tile_number + 1));
            // std::cout << "\t\tbuffer_prob: " << buffer_prob << std::endl;
            return buffer_prob;
        }*/
        double prob = std::exp(((double)(old_cost - new_cost) / (double)(old_cost - ideal_cost)) / temperature);
        // std::cout << "\t\tprob: " << prob << std::endl;
        return rand_prob(prob);
    }
}

inline bool Stage2SA::accept_by_progress(const cost_t& old_cost, const cost_t& new_cost, const cost_t& ideal_cost, const double progress) const
{
    if (new_cost < old_cost) {
        return true;
    } else {
        /*if ((double)(old_cost - new_cost) / (double)old_cost <= 1e-5) {
            double buffer_prob = sigmod(((double)buffer_time_saved / (double)last_g.buffer.get_max_buffer_size()) / (double)(total_tile_number + 1));
            // std::cout << "\t\tbuffer_prob: " << buffer_prob << std::endl;
            return buffer_prob;
        }*/
        double T = 0.1 * (1.0 - progress) / (1.0 + 8.0 * progress);
        double prob = std::exp(((double)(old_cost - new_cost) / (double)(old_cost - ideal_cost)) / T);
        // std::cout << "\t\tprob: " << prob << std::endl;
        return rand_prob(prob);
    }
}

inline bool Stage2SA::accept_by_progress_and_buffer_usage(const cost_t& old_cost, const cost_t& new_cost, const __uint128_t& old_sum_buffer_usage, const __uint128_t& new_sum_buffer_usage, const __uint128_t& ideal_sum_buffer_usage, const double progress) const
{
    if (new_cost < old_cost) {
        return true;
    } else if (new_cost == old_cost){
        if (new_sum_buffer_usage < old_sum_buffer_usage) {
            return true;
        } else {
            double T = 0.1 * (1.0 - progress) / (1.0 + 8.0 * progress);
            double prob = std::exp((double)(old_sum_buffer_usage - new_sum_buffer_usage) / (double)(old_sum_buffer_usage - ideal_sum_buffer_usage) / T);
            return rand_prob(prob);
        }
    } else {
        return false;
    }
}

void Stage1SA::update_changeable_layers(const Graph& g)
{
    const auto num_layers = network->len();
    changeable_layers.clear();
    changeable_layers.reserve(num_layers);
    for (lid_t id = 0; id < num_layers; id++) {
        changeable_layer l;
        l.layer_id = id;
        l.order = g.layer_id_to_order[id];
        // NO limit at the beginning
        l.prev = 0;
        l.next = num_layers - 1;
        FOR_BITSET(p, network->getNode(id).getPrevs()) {
            l.prev = MAX(g.layer_id_to_order[p], l.prev);
        }
        FOR_BITSET(o_p, network->getNode(id).getOrderPrevs()) {
            l.prev = MAX(g.layer_id_to_order[o_p], l.prev);
        }

        FOR_BITSET(n, network->getNode(id).get_nexts()) {
            l.next = MIN(g.layer_id_to_order[n], l.next);
        }
        FOR_BITSET(o_n, network->getNode(id).getOrderNexts()) {
            l.next = MIN(g.layer_id_to_order[o_n], l.next);
        }
        if (l.prev >= l.next - 2)
            continue;
        else
            changeable_layers.emplace_back(l);
    }
}

bool Stage1SA::change_layer_order(const Graph& g)
{
    if (changeable_layers.size() == 0) {
        // std::cout << "Linear order, cannot change\n";
        return false;
    }
    
    // pick a random movable layer to move
    const int l_idx = randint(0, changeable_layers.size() - 1);
    auto& l = changeable_layers[l_idx];
    const int l_id = enc.layer_order_to_id[l.order];
    // pick a random movable position to move to
    const int random_range = l.next - l.prev - 2;
    const int pos_shift = randint(1, random_range); // [1, n-p-2]
    int pos = l.prev + pos_shift; // [p+1, n-2]
    if (pos >= l.order) pos++; // [p+1, m-1] U [m+1, n-1]
    assert(pos != l.order);
    // std::cout << "Layer " << l.layer_id << "@" << l.order << " insert at Order " << pos << std::endl;
    // std::cout << "l.prev=" << l.prev << ", l.next=" << l.next << std::endl;
    
    // insert l_idx to pos
    auto& lg_par = enc.layer_group_partition;
    auto& slg_par = enc.sub_layer_group_partition;
    // special handle with first par
    bool lg_is_10  = lg_par[l.order] & (~lg_par[l.order+1]); //[order] == 1, [order+1] == 0
    // bool lg_is_11  = lg_par[l.order] &   lg_par[l.order+1];  //[order] == 1, [order+1] == 1
    bool slg_is_10 = (~lg_par[l.order]) & (~lg_par[l.order+1]) & slg_par[l.order] & (~slg_par[l.order+1]); // in the same LG && [order] == 1, [order+1] == 0
    bool slg_is_11 = (lg_par[l.order] | slg_par[l.order]) & (lg_par[l.order+1] | slg_par[l.order+1]);  //[order] == 1, [order+1] == 1
    if (l.order == network->len() - 1) {
        // which means lg and slg are x1
        lg_is_10 = false;
        slg_is_10 = false;
        slg_is_11 = lg_par[l.order] | slg_par[l.order];
    }
    if (l.order < pos) {
        for (int i = l.order; i < pos; i++) {
            // Order [i] = [i+1]
            enc.layer_order_to_id[i] = enc.layer_order_to_id[i + 1];
            // LG    [i] = [i+1]
            lg_par[i] = lg_par[i+1];
            // if (lg_par[i+1])  { lg_par.set(i); } else { lg_par.reset(i); }
            // SLG   [i] = [i+1]
            slg_par[i] = slg_par[i+1];
            // if (slg_par[i+1]) {slg_par.set(i); } else {slg_par.reset(i); }
        }
        if (lg_is_10)  { lg_par.set(l.order); slg_par.reset(l.order); }
        if (slg_is_10) {slg_par.set(l.order); }
    } else {
        for (int i = l.order; i > pos; i--) {
            // Order [i] = [i-1]
            enc.layer_order_to_id[i] = enc.layer_order_to_id[i - 1];
            // LG    [i] = [i-1]
            lg_par[i] = lg_par[i-1];
            // if (lg_par.contains(i-1))  { lg_par.set(i); } else { lg_par.reset(i); }
            // SLG   [i] = [i-1]
            slg_par[i] = slg_par[i-1];
            // if (slg_par.contains(i-1)) {slg_par.set(i); } else {slg_par.reset(i); }
        }
        if (lg_is_10)  { lg_par.set(l.order+1); slg_par.reset(l.order+1);}
        if (slg_is_10) {slg_par.set(l.order+1); }
    }
    enc.layer_order_to_id[pos] = l_id;
    enc.layer_group_partition.reset(pos);       // [pos] = 0
    enc.sub_layer_group_partition.reset(pos);   // [pos] = 0
    if (slg_is_11) {
        int single_slg = g.layer_id_to_lg_slg[l.layer_id].second;
        enc.tile_numbers.erase(enc.tile_numbers.begin() + single_slg);
    }
    // Now enc ready
    // std::cout << enc << std::endl;
    // std::cout << "LG diff:" << (enc.layer_group_partition ^ last_enc.layer_group_partition) << ", SLG diff:" << (enc.sub_layer_group_partition ^ last_enc.sub_layer_group_partition) << std::endl;
    assert(enc.tile_numbers.size() == (enc.layer_group_partition | enc.sub_layer_group_partition).count());
    assert((enc.layer_group_partition & enc.sub_layer_group_partition).count() == 0);
    assert(!enc.layer_group_partition.contains(network->len()));
    
    return true;
}

bool Stage1SA::change_layer_order_baseline(const Graph& g)
{
    if (changeable_layers.size() == 0) {
        // std::cout << "Linear order, cannot change\n";
        return false;
    }
    
    // pick a random movable layer to move
    const int l_idx = randint(0, changeable_layers.size() - 1);
    auto& l = changeable_layers[l_idx];
    const int l_id = enc.layer_order_to_id[l.order];
    // pick a random movable position to move to
    const int random_range = l.next - l.prev - 2;
    const int pos_shift = randint(1, random_range); // [1, n-p-2]
    int pos = l.prev + pos_shift; // [p+1, n-2]
    if (pos >= l.order) pos++; // [p+1, m-1] U [m+1, n-1]
    assert(pos != l.order);
    // std::cout << "Layer " << l.layer_id << "@" << l.order << " insert at Order " << pos << std::endl;
    // std::cout << "l.prev=" << l.prev << ", l.next=" << l.next << std::endl;
    
    // insert l_idx to pos
    auto& lg_par = enc.layer_group_partition;
    // special handle with first par
    bool lg_is_10 = lg_par[l.order] & (~lg_par[l.order+1]); //[order] == 1, [order+1] == 0
    bool lg_is_11 = lg_par[l.order] &   lg_par[l.order+1];  //[order] == 1, [order+1] == 1
    
    if (l.order < pos) {
        for (int i = l.order; i < pos; i++) {
            // Order [i] = [i+1]
            enc.layer_order_to_id[i] = enc.layer_order_to_id[i + 1];
            // LG    [i] = [i+1]
            lg_par[i] = lg_par[i+1];
            // if (lg_par[i+1])  { lg_par.set(i); } else { lg_par.reset(i); }
        }
        if (lg_is_10)  { lg_par.set(l.order); }
    } else {
        for (int i = l.order; i > pos; i--) {
            // Order [i] = [i-1]
            enc.layer_order_to_id[i] = enc.layer_order_to_id[i - 1];
            // LG    [i] = [i-1]
            lg_par[i] = lg_par[i-1];
            // if (lg_par.contains(i-1))  { lg_par.set(i); } else { lg_par.reset(i); }
        }
        if (lg_is_10)  { lg_par.set(l.order+1); }
    }
    enc.layer_order_to_id[pos] = l_id;
    enc.layer_group_partition.reset(pos);       // [pos] = 0
    if (lg_is_11) {
        int single_slg = g.layer_id_to_lg_slg[l.layer_id].second;
        enc.tile_numbers.erase(enc.tile_numbers.begin() + single_slg);
    }
    // Now enc ready
    // std::cout << enc << std::endl;
    // std::cout << "LG diff:" << (enc.layer_group_partition ^ last_enc.layer_group_partition) << ", SLG diff:" << (enc.sub_layer_group_partition ^ last_enc.sub_layer_group_partition) << std::endl;
    assert(enc.tile_numbers.size() == enc.layer_group_partition.count());
    assert(enc.sub_layer_group_partition.empty());
    
    return true;
}

bool Stage1SA::change_tile_number(const Graph& g)
{
    // change the tile number of a random SLG
    const lid_t slg_id = randint(0, enc.tile_numbers.size() - 1);
    unsigned int new_tile_number = 0;

    // get max SLG output tile number, 'cause cutting more tiles than the total size is meaningless
    auto total_output_ofm_shape = network->getNode(g.all_slgs[slg_id].pure_outputs_id[0]).layer().ofmap_shape();
    uint32_t max_slg_tile_num = SchNode::tot_batch * total_output_ofm_shape.h * total_output_ofm_shape.w;

    // @attention tile_number update policy
    if (rand_prob(0.5))
        new_tile_number = MAX(1, enc.tile_numbers[slg_id]/2);
    else
        new_tile_number = MIN(1 << (31 - __builtin_clz(max_slg_tile_num)), enc.tile_numbers[slg_id]*2);
    // return false if no change
    if (new_tile_number == enc.tile_numbers[slg_id])
        return false;
    // std::cout << new_tile_number << std::endl;
    enc.tile_numbers[slg_id] = new_tile_number;
    // Now enc ready
    return true;
}

bool Stage1SA::change_layer_group(const Graph& g)
{
    constexpr int MAX_TRIAL = 30;
    // Randomly pick a method to change:
    // 1. fuse two layer groups
    // 2. split a layer group at a random position in its slg border, may fail if the layer group has only one slg
    // 3. change the border between 2 layer groups, shift it 
    int method = rand_multi_prob({0.3, 0.3, 0.4});
    auto numLayerGroups = enc.layer_group_partition.count();
    auto numSubLayerGroups = enc.tile_numbers.size();
    // std::cout << method << std::endl;
    // std::cout << enc.layer_group_partition << ", \t" << enc.sub_layer_group_partition << std::endl;
    if (method == 0) { // fuse two layer groups
        if (numLayerGroups <= 1) {
            // std::cout << "border limit\n";
            return false;
        }

        // pick a random layer group to fuse
        int lg_idx = randint(1, numLayerGroups - 1);
        int fuse_pos = g.layer_groups[lg_idx].layer_group_start;
        enc.layer_group_partition.reset(fuse_pos);
        enc.sub_layer_group_partition.set(fuse_pos);
    } else if (method == 1) { // split a layer group
        if (numLayerGroups >= numSubLayerGroups) {
            // std::cout << "border limit\n";
            return false;
        }
        
        // pick a random layer group to split. should have at least 2 slgs
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            lg_idx = randint(0, numLayerGroups - 1);
            if (g.layer_groups[lg_idx].slg_idx_end != g.layer_groups[lg_idx].slg_idx_start) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        
        // pick a random position to split
        int split_slg_idx = randint(g.layer_groups[lg_idx].slg_idx_start + 1, g.layer_groups[lg_idx].slg_idx_end);
        int split_slg_pos = g.all_slgs[split_slg_idx].sub_layer_group_start;
        enc.layer_group_partition.set(split_slg_pos);
        enc.sub_layer_group_partition.reset(split_slg_pos);
    } else { // change the border between 2 layer groups
        if (numLayerGroups <= 1 || numLayerGroups >= numSubLayerGroups) {
            // std::cout << "border limit\n";
            return false;
        }
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            // pick a random layer group to change the border
            lg_idx = randint(1, numLayerGroups - 1);
            // move the lg border to a adjacent slg
            if (g.layer_groups[lg_idx].slg_idx_start != g.layer_groups[lg_idx].slg_idx_end || 
                g.layer_groups[lg_idx-1].slg_idx_start != g.layer_groups[lg_idx-1].slg_idx_end) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }

        // pick a random position to move to
        int move_to_slg_idx = randint_except(g.layer_groups[lg_idx-1].slg_idx_start + 1, 
                                             g.layer_groups[lg_idx].slg_idx_start, 
                                             g.layer_groups[lg_idx].slg_idx_end);
        enc.layer_group_partition.reset(g.layer_groups[lg_idx].layer_group_start);
        enc.sub_layer_group_partition.set(g.layer_groups[lg_idx].layer_group_start);
        enc.layer_group_partition.set(g.all_slgs[move_to_slg_idx].sub_layer_group_start);
        enc.sub_layer_group_partition.reset(g.all_slgs[move_to_slg_idx].sub_layer_group_start);
    }
    // std::cout << enc.layer_group_partition << ", \t" << enc.sub_layer_group_partition << std::endl;
    // Now enc ready
    return true;
}

bool Stage1SA::change_layer_group_by_layer(const Graph& g)
{
    constexpr int MAX_TRIAL = 30;
    // Randomly pick a method to change:
    // 1. fuse two layer groups
    // 2. split a layer group at a random layer, may fail if the layer group has only one layer
    // 3. change the border between 2 layer groups, shift it 
    int method = rand_multi_prob({0.3, 0.3, 0.4});
    auto numLayerGroups = enc.layer_group_partition.count();
    // std::cout << method << std::endl;
    // std::cout << enc.layer_group_partition << std::endl;
    if (method == 0) { // fuse two layer groups
        if (numLayerGroups <= 1) {
            // std::cout << "border limit\n";
            return false;
        }
        // pick a random layer group to fuse
        int lg_idx = randint(1, numLayerGroups - 1);
        int fuse_pos = g.layer_groups[lg_idx].layer_group_start;
        enc.layer_group_partition.reset(fuse_pos);
        // slg number--
        assert(g.layer_groups[lg_idx].slg_idx_start - g.layer_groups[lg_idx-1].slg_idx_end == 1);
        double layer_num_proportion = (double)(g.all_slgs[g.layer_groups[lg_idx].slg_idx_start].layer_num()) / (double)(g.all_slgs[g.layer_groups[lg_idx].slg_idx_start].layer_num() + g.all_slgs[g.layer_groups[lg_idx-1].slg_idx_end].layer_num());
        if (/*rand_prob(0.5)*/rand_prob(layer_num_proportion)) {
            enc.tile_numbers.erase(enc.tile_numbers.begin() + g.layer_groups[lg_idx-1].slg_idx_end);
        } else {
            enc.tile_numbers.erase(enc.tile_numbers.begin() + g.layer_groups[lg_idx].slg_idx_start);
        }
    } else if (method == 1) { // split a layer group
        if (numLayerGroups >= network->len()) {
            // std::cout << "border limit\n";
            return false;
        }
        // pick a random layer group to split. should have at least 2 layers
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            lg_idx = randint(0, numLayerGroups - 1);
            if (g.layer_groups[lg_idx].layer_group_end != g.layer_groups[lg_idx].layer_group_start) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        // pick a random position to split
        int split_pos = randint(g.layer_groups[lg_idx].layer_group_start + 1, g.layer_groups[lg_idx].layer_group_end);
        enc.layer_group_partition.set(split_pos);
        if (enc.sub_layer_group_partition.contains(split_pos)) { // split at a slg border
            // slg number same
            enc.sub_layer_group_partition.reset(split_pos);
        } else { // cut one slg into two
            // slg number++
            const lid_t& split_layer_id = g.layer_order_to_id[split_pos];
            const lid_t& split_slg_idx = g.layer_id_to_lg_slg[split_layer_id].second;
            assert(g.layer_groups[lg_idx].slg_idx_start <= split_slg_idx && split_slg_idx <= g.layer_groups[lg_idx].slg_idx_end);
            enc.tile_numbers.insert(enc.tile_numbers.begin() + split_slg_idx, enc.tile_numbers[split_slg_idx]);
        }
    } else { // change the border between 2 layer groups
        if (numLayerGroups <= 1 || numLayerGroups >= network->len()) {
            // std::cout << "border limit\n";
            return false;
        }
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            // pick a random layer group to change the border
            lg_idx = randint(1, numLayerGroups - 1);
            // move the lg border to a adjacent layer
            if (g.layer_groups[lg_idx].layer_group_start != g.layer_groups[lg_idx].layer_group_end || 
                g.layer_groups[lg_idx-1].layer_group_start != g.layer_groups[lg_idx-1].layer_group_end) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        // pick a random position to move to
        int move_to_pos = randint_except(g.layer_groups[lg_idx-1].layer_group_start + 1, 
                                         g.layer_groups[lg_idx].layer_group_start, 
                                         g.layer_groups[lg_idx].layer_group_end);
        enc.layer_group_partition.reset(g.layer_groups[lg_idx].layer_group_start);
        enc.sub_layer_group_partition.set(g.layer_groups[lg_idx].layer_group_start);
        enc.layer_group_partition.set(move_to_pos);
        if (enc.sub_layer_group_partition.contains(move_to_pos)) { // move to a slg border
            // slg number same
            enc.sub_layer_group_partition.reset(move_to_pos);
        } else { // cut one slg into two
            // slg number++
            const lid_t& move_to_layer_id = g.layer_order_to_id[move_to_pos];
            const lid_t& move_to_slg_idx = g.layer_id_to_lg_slg[move_to_layer_id].second;
            assert(g.layer_groups[lg_idx-1].slg_idx_start <= move_to_slg_idx && move_to_slg_idx <= g.layer_groups[lg_idx].slg_idx_end);
            enc.tile_numbers.insert(enc.tile_numbers.begin() + move_to_slg_idx, enc.tile_numbers[move_to_slg_idx]);
        }
    }
    assert(enc.tile_numbers.size() == (enc.layer_group_partition | enc.sub_layer_group_partition).count());
    // std::cout << enc.layer_group_partition << std::endl;
    // std::cout << enc.sub_layer_group_partition << std::endl;
    // Now enc ready
    return true;
}

bool Stage1SA::change_layer_group_by_layer_baseline(const Graph & g)
{
    constexpr int MAX_TRIAL = 30;
    // Randomly pick a method to change:
    // 1. fuse two layer groups
    // 2. split a layer group at a random layer, may fail if the layer group has only one layer
    // 3. change the border between 2 layer groups, shift it 
    int method = rand_multi_prob({0.3, 0.3, 0.4});
    auto numLayerGroups = enc.layer_group_partition.count();
    // std::cout << method << std::endl;
    // std::cout << enc.layer_group_partition << std::endl;
    if (method == 0) { // fuse two layer groups
        if (numLayerGroups <= 1) {
            // std::cout << "border limit\n";
            return false;
        }
        // pick a random layer group to fuse
        int lg_idx = randint(1, numLayerGroups - 1);
        int fuse_pos = g.layer_groups[lg_idx].layer_group_start;
        enc.layer_group_partition.reset(fuse_pos);
        // enc.sub_layer_group_partition.set(fuse_pos);
        // slg number--
        double layer_num_proportion = (double)(g.layer_groups[lg_idx].layer_group_end - g.layer_groups[lg_idx].layer_group_start + 1) / (double)(g.layer_groups[lg_idx].layer_group_end - g.layer_groups[lg_idx-1].layer_group_start + 1);
        //  double tile_num_proportion = (double)enc.tile_numbers[lg_idx] / (double)(enc.tile_numbers[lg_idx] + enc.tile_numbers[lg_idx - 1]);
        //  if (tile_num_proportion > 0.5) {
        if (/*rand_prob(0.5)*/rand_prob(layer_num_proportion)) {
            enc.tile_numbers.erase(enc.tile_numbers.begin() + lg_idx - 1);
        } else {
            enc.tile_numbers.erase(enc.tile_numbers.begin() + lg_idx);
        }
    } else if (method == 1) { // split a layer group
        if (numLayerGroups >= network->len()) {
            // std::cout << "border limit\n";
            return false;
        }
        // pick a random layer group to split. should have at least 2 layers
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            lg_idx = randint(0, numLayerGroups - 1);
            if (g.layer_groups[lg_idx].layer_group_end != g.layer_groups[lg_idx].layer_group_start) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }

        // pick a random position to split
        int split_pos = randint(g.layer_groups[lg_idx].layer_group_start + 1, g.layer_groups[lg_idx].layer_group_end);
        enc.layer_group_partition.set(split_pos);
        // enc.sub_layer_group_partition.reset(split_pos);
        // slg number++
        enc.tile_numbers.insert(enc.tile_numbers.begin() + lg_idx + 1, enc.tile_numbers[lg_idx]);
    } else { // change the border between 2 layer groups
        if (numLayerGroups <= 1 || numLayerGroups >= network->len()) {
            // std::cout << "border limit\n";
            return false;
        }
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            // pick a random layer group to change the border
            lg_idx = randint(1, numLayerGroups - 1);
            // move the lg border to a adjacent layer
            if (g.layer_groups[lg_idx].layer_group_start != g.layer_groups[lg_idx].layer_group_end || 
                g.layer_groups[lg_idx-1].layer_group_start != g.layer_groups[lg_idx-1].layer_group_end) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        // pick a random position to move to
        int move_to_pos = randint_except(g.layer_groups[lg_idx-1].layer_group_start + 1, 
                                         g.layer_groups[lg_idx].layer_group_start, 
                                         g.layer_groups[lg_idx].layer_group_end);
        enc.layer_group_partition.reset(g.layer_groups[lg_idx].layer_group_start);
        enc.layer_group_partition.set(move_to_pos);
    }
    assert(enc.tile_numbers.size() == enc.layer_group_partition.count());
    // std::cout << enc.layer_group_partition << std::endl;
    // Now enc ready
    return true;
}

bool Stage1SA::change_sub_layer_group(const Graph& g)
{
    constexpr int MAX_TRIAL = 30;
    // Randomly pick a method to change:
    // 1. fuse two sub layer groups randomly in a random layer group
    // 2. split a sub layer group at a random layer pos, may fail if the sub layer group has only one layer
    // 3. change the border between 2 sub layer groups, shift it
    int method = rand_multi_prob({1, 0.0, 0.0});
    auto numLayerGroups = enc.layer_group_partition.count();
    auto numSubLayerGroups = enc.tile_numbers.size();
    // std::cout << method << std::endl;
    // std::cout << enc.sub_layer_group_partition << std::endl;
    if (method == 0) { // fuse two sub layer groups
        if (numSubLayerGroups <= 1) {
            // std::cout << "border limit\n";
            return false;
        }
        
        // pick a random layer group with at least 2 slgs
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            lg_idx = randint(0, numLayerGroups - 1);
            if (g.layer_groups[lg_idx].slg_idx_start != g.layer_groups[lg_idx].slg_idx_end) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        // pick a random slg pos to fuse
        int fuse_slg_idx = randint(g.layer_groups[lg_idx].slg_idx_start + 1, g.layer_groups[lg_idx].slg_idx_end);
        enc.sub_layer_group_partition.reset(g.all_slgs[fuse_slg_idx].sub_layer_group_start);
        // double layer_num_proportion = (double)(g.all_slgs[fuse_slg_idx].sub_layer_group_end - g.all_slgs[fuse_slg_idx].sub_layer_group_start + 1)/ 
        //                               (double)(g.all_slgs[fuse_slg_idx].sub_layer_group_end - g.all_slgs[fuse_slg_idx-1].sub_layer_group_start + 1);
        double layer_num_proportion = (double)(g.all_slgs[fuse_slg_idx].layer_num())/ 
                                      (double)(g.all_slgs[fuse_slg_idx].layer_num() + g.all_slgs[fuse_slg_idx-1].layer_num());
        if (/*rand_prob(0.5)*/rand_prob(layer_num_proportion)) { // we prone to use the tile_number of the slg with more layers
            enc.tile_numbers.erase(enc.tile_numbers.begin() + fuse_slg_idx - 1);
        } else {
            enc.tile_numbers.erase(enc.tile_numbers.begin() + fuse_slg_idx);
        }
    } else if (method == 1) { // split a sub layer group
        if (numSubLayerGroups >= network->len()) {
            // std::cout << "border limit\n";
            return false;
        }
        
        // pick a random sub layer group to split. should have at least 2 layers
        int slg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            slg_idx = randint(0, numSubLayerGroups - 1);
            // if (g.all_slgs[slg_idx].sub_layer_group_end - g.all_slgs[slg_idx].sub_layer_group_start >= 1) {
            if (g.all_slgs[slg_idx].layer_num() >= 2) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        // pick a random position to split
        assert(g.all_slgs.size() == enc.tile_numbers.size());
        int split_pos = randint(g.all_slgs[slg_idx].sub_layer_group_start + 1, g.all_slgs[slg_idx].sub_layer_group_end);
        enc.sub_layer_group_partition.set(split_pos);
        enc.tile_numbers.insert(enc.tile_numbers.begin() + slg_idx + 1, enc.tile_numbers[slg_idx]);
    } else { // change the border between 2 sub layer groups
        if (numSubLayerGroups <= 1 || numSubLayerGroups >= network->len()) {
            // std::cout << "border limit\n";
            return false;
        }
        
        // pick a random layer group with at least 2 slgs
        int lg_idx;
        bool success = false;
        for (int i = 0; i < MAX_TRIAL; i++) {
            lg_idx = randint(0, numLayerGroups - 1);
            if (g.layer_groups[lg_idx].slg_idx_start != g.layer_groups[lg_idx].slg_idx_end) {
                success = true;
                break;
            }
        }
        if (!success) {
            // std::cout << "no luck\n";
            return false;
        }
        
        // pick a random slg pos to move border
        int move_slg_idx = randint(g.layer_groups[lg_idx].slg_idx_start + 1, g.layer_groups[lg_idx].slg_idx_end);
        if (g.all_slgs[move_slg_idx].sub_layer_group_start == g.all_slgs[move_slg_idx].sub_layer_group_end && 
            g.all_slgs[move_slg_idx-1].sub_layer_group_start == g.all_slgs[move_slg_idx-1].sub_layer_group_end) {
            // std::cout << "no luck\n";
            return false;
        }
        // pick a random position to move to
        int move_to_pos = randint_except(g.all_slgs[move_slg_idx-1].sub_layer_group_start + 1, 
                                         g.all_slgs[move_slg_idx].sub_layer_group_start, 
                                         g.all_slgs[move_slg_idx].sub_layer_group_end);
        enc.sub_layer_group_partition.reset(g.all_slgs[move_slg_idx].sub_layer_group_start);
        enc.sub_layer_group_partition.set(move_to_pos);
    }
    assert(enc.tile_numbers.size() == (enc.layer_group_partition | enc.sub_layer_group_partition).count());
    // std::cout << enc.sub_layer_group_partition << std::endl;
    // Now enc ready
    return true;
}

bool Stage1SA::make_sure_all_slgs_are_fully_connected(const Graph& g)
{
    int num_inputs, num_nodes, num_slgs;
    int* fa;
    int k = 1; // inter-slg effect, k stands for #newly_added_slgs
    bool update = false;
    for (size_t i = 0; i < g.all_slgs.size(); i++) {
        num_slgs = g.check_subgraph_not_fully_connected(g.all_slgs[i], fa, num_inputs, num_nodes);
        assert(num_slgs > 0);
        if (num_slgs > 1) {
            update = true;
            // make them fully connected by cutting them apart
            enc.tile_numbers.insert(enc.tile_numbers.begin() + i + k, num_slgs - 1, enc.tile_numbers[i]);
            k += num_slgs - 1; 
            vector<lid_t /*layer_id*/> new_slg_layer_id[num_slgs];
            unordered_map<int, int> d;
            for (int j = 0; j < num_nodes; j++) {
                if (!d.count(fa[num_inputs + j])) { // a new slg
                    d[fa[num_inputs + j]] = d.size();
                }
                int slg_idx = d[fa[num_inputs + j]];
                int layer_order = g.all_slgs[i].sub_layer_group_start + j;
                new_slg_layer_id[slg_idx].emplace_back(g.layer_order_to_id[layer_order]);
            }
            int p = g.all_slgs[i].sub_layer_group_start;
            for (int j = 0; j < num_slgs; j++) {
                assert(new_slg_layer_id[j].size() > 0);
                for (const lid_t& l : new_slg_layer_id[j]) {
                    enc.layer_order_to_id[p++] = l;
                }
                if (j != num_slgs - 1)
                    enc.sub_layer_group_partition.set(p);
            }
            assert(p == g.all_slgs[i].sub_layer_group_end + 1);
        }
    }
    return update;
}

bool Stage1SA::make_sure_all_lgs_are_fully_connected(const Graph& g)
{
    assert(g.layer_groups.size() == g.all_slgs.size());
    int num_inputs, num_nodes, num_slgs;
    int* fa;
    int k = 1; // inter-slg effect, k stands for #newly_added_slgs
    bool update = false;
    for (size_t i = 0; i < g.all_slgs.size(); i++) {
        num_slgs = g.check_subgraph_not_fully_connected(g.all_slgs[i], fa, num_inputs, num_nodes);
        assert(num_slgs > 0);
        if (num_slgs > 1) {
            update = true;
            // make them fully connected by cutting them apart
            enc.tile_numbers.insert(enc.tile_numbers.begin() + i + k, num_slgs - 1, enc.tile_numbers[i]);
            k += num_slgs - 1; 
            vector<lid_t /*layer_id*/> new_slg_layer_id[num_slgs];
            unordered_map<int, int> d;
            for (int j = 0; j < num_nodes; j++) {
                if (!d.count(fa[num_inputs + j])) { // a new slg
                    d[fa[num_inputs + j]] = d.size();
                }
                int slg_idx = d[fa[num_inputs + j]];
                int layer_order = g.all_slgs[i].sub_layer_group_start + j;
                new_slg_layer_id[slg_idx].emplace_back(g.layer_order_to_id[layer_order]);
            }
            int p = g.all_slgs[i].sub_layer_group_start;
            for (int j = 0; j < num_slgs; j++) {
                assert(new_slg_layer_id[j].size() > 0);
                for (const lid_t& l : new_slg_layer_id[j]) {
                    enc.layer_order_to_id[p++] = l;
                }
                if (j != num_slgs - 1)
                    enc.layer_group_partition.set(p);
            }
            assert(p == g.all_slgs[i].sub_layer_group_end + 1);
        }
    }
    return update;
}

Stage1SA::Stage1SA(const Network* _network, Graph::Stage1Encoding _default_encoding, const uint8_t _baseline_type)
: OptimizationStage(_network)
{
    enc = _default_encoding;
    last_enc = _default_encoding;
    baseline_type = _baseline_type;
}

inline len_t nextPowerOfTwo(len_t x) {
    // If x is already a power of two, return x
    if (x && !(x & (x - 1)))
        return x;
    // Otherwise, compute the next power of two
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

bool Stage1SA::sa_init(const Graph::Stage1Encoding_tile_sizes& enc_tz, const bool use_tile_sizes, const bool print_results, const bool use_small_tile_nums)
{
    assert(baseline_type == 2 && "Only support baseline_type 2");
    // init graph
    Graph::resetCacheCounter();
    Graph::Buffer::set_stage_1(true);
    ErrorType err = ErrorType::SUCCESS;
    for (lid_t i = 0; i < network->len(); i++) {
        const Node& n = network->getNode(i);
        const Layer& l = n.layer();
        vol_t WGT_size = l.weight_size();
        vol_t TILE_size = l.ofmap_shape().c;
        vol_t IFM_size = 0;
        {
            auto ofm_range = fmap_range(fmap_shape(l.ofmap_shape().c, 1, 1), 1);
            l.ofm_to_ifm(ofm_range);
            IFM_size += ofm_range.size();
        }
        if (n.hasWgtPrevs()) {
            WGT_size = 0;
            auto ofm_range = fmap_range(fmap_shape(l.ofmap_shape().c, 1, 1), 1);
            l.ofm_to_wgt(ofm_range);
            IFM_size += ofm_range.size();
        }
        if(WGT_size + IFM_size + TILE_size > Graph::Buffer::get_max_buffer_size()) {
            err = ErrorType::BUFFER_OVERFLOW;
            std::cout << "Layer " << i << "(" << l.get_name() 
                 << ")'s WGT size " << WGT_size/1024.0 << "KB"
                 << " + Minimal OFM size " << TILE_size/1024.0 << "KB"
                 << " + ofm_to_ifm(OFM) size " << IFM_size/1024.0 << "KB"
                 << " = " << (WGT_size + IFM_size + TILE_size)/1024.0 << "KB"
                 << " > Buffer Size " << Graph::Buffer::get_max_buffer_size()/1024 << "KB" << std::endl;
        }
    }
    if (err != ErrorType::SUCCESS) {
        std::cout << "Default init solution INVALID: " << err << std::endl;
        return false;
    }
    Graph g;
    if (use_tile_sizes) {
        err = g.init_stage_1_with_tile_sizes(enc_tz);
    } else {
        err = g.init_stage_1(enc);
    }
    if (err != ErrorType::SUCCESS) {
        if (err == ErrorType::SUBGRAPH_NOT_FULLY_CONNECTED) {
            make_sure_all_lgs_are_fully_connected(g);
            err = g.init_stage_1(enc);
            assert(enc == g.get_Encoding().first);
            if (err != ErrorType::SUCCESS) {
                std::cout << "Default init solution INVALID: " << err << std::endl;
                return false;
            }        
        } else {
            std::cout << "Default init solution INVALID: " << err << std::endl;
            return false;
        }
    }
    if (use_tile_sizes) {
        enc = g.get_Encoding().first;
        // g.init_stage_1(enc);
    }
    for (int i = 0; i < enc.tile_numbers.size(); i++) {
        enc.tile_numbers[i] = nextPowerOfTwo(enc.tile_numbers[i]);
    }
    vector<len_t> old_tile_numbers;
    if (use_small_tile_nums) {
        old_tile_numbers = enc.tile_numbers;
        for (int i = 0; i < enc.tile_numbers.size(); i++) {
            if (enc.tile_numbers[i] > 16 * SchNode::tot_batch) {
                enc.tile_numbers[i] = SchNode::tot_batch;
            }
        }
    }
    err = g.init_stage_1(enc);
    if (err != ErrorType::SUCCESS) {
        std::cout << "Default init solution INVALID: " << err << std::endl;
        return false;
    }
    CoreMapper::MapCost ideal_cost, real_cost;
    g.initTileCosts();
    g.getIdealCost(ideal_cost, false);
    err = g.getRealCost(real_cost, false);
    // int tile_layer_order = 0;
    // for (size_t x = 0; x < g.buffer.buffer_usage_by_tile.size(); x++) {
    //     if (g.buffer.buffer_usage_by_tile[x] > Graph::Buffer::get_max_buffer_size()) {
    //         std::cout << "Buffer overflow @" << x << " : " << g.buffer.buffer_usage_by_tile[x] << std::endl;
    //         while (x > g.layer_id_to_tile_pos[tile_layer_order].end_time)
    //             tile_layer_order++;
    //         std::cout << "Layer " << g.layer_order_to_id[tile_layer_order] << " : " << network->getNode(g.layer_order_to_id[tile_layer_order]).layer().get_name() << std::endl;
    //     }
    // }
    // std::cout << g;
    if (err != ErrorType::SUCCESS) {
        // mitiagte the issue by dividing the tile size
        for (lid_t slg_id = 0; slg_id < g.all_slgs.size(); slg_id++) {
            auto total_output_ofm_shape = network->getNode(g.all_slgs[slg_id].pure_outputs_id[0]).layer().ofmap_shape();
            uint32_t max_slg_tile_num = SchNode::tot_batch * total_output_ofm_shape.h * total_output_ofm_shape.w;
            for (lid_t k = g.all_slgs[slg_id].sub_layer_group_start; k <= g.all_slgs[slg_id].sub_layer_group_end; k++) {
                const lid_t& layer_id = g.layer_order_to_id[k];
                const Node& n = network->getNode(layer_id);
                const Layer& l = n.layer();
                g.all_layers[layer_id].tile_size.update_size();
                const auto& ofm_shape = g.all_layers[layer_id].tile_size;
                
                auto getIFMSize = [&l](const tensor_shape& ofm_shape) {
                    auto ofm_range = fmap_range(fmap_shape(ofm_shape.c, ofm_shape.h, ofm_shape.w), ofm_shape.bk);
                    l.ofm_to_ifm(ofm_range);
                    return ofm_range.size();
                };

                auto getPrevWGTSize = [&l](const tensor_shape& ofm_shape) {
                    auto ofm_range = fmap_range(fmap_shape(ofm_shape.c, ofm_shape.h, ofm_shape.w), ofm_shape.bk);
                    l.ofm_to_wgt(ofm_range);
                    return ofm_range.size();
                };

                vol_t WGT_size = l.weight_size();
                vol_t TILE_size = ofm_shape.get_size();
                vol_t max_buffer_size = Graph::Buffer::get_max_buffer_size()/2;
                vol_t IFM_size = getIFMSize(ofm_shape);
                if (n.hasWgtPrevs()) {
                    WGT_size = 0;
                    IFM_size += getPrevWGTSize(ofm_shape);
                }
                if (WGT_size + IFM_size + TILE_size > max_buffer_size) {
                    // we do binary search
                    int tn = enc.tile_numbers[slg_id];
                    auto ts = ofm_shape;
                    int exceed = WGT_size + IFM_size + TILE_size - max_buffer_size;
                    while (exceed > 0 && tn <= max_slg_tile_num && (!use_small_tile_nums || tn < old_tile_numbers[slg_id]))
                    {
                        tn = tn * 2;
                        g.cut_into_tiles(ts, tn, Graph::SLGTransposeType::NONE); // Transpose is the first layer of this slg (slg only contains 1 layer)
                        exceed = WGT_size + ofm_shape.get_size() + getIFMSize(ofm_shape) - max_buffer_size;
                        if (n.hasWgtPrevs()) {
                            exceed += getPrevWGTSize(ofm_shape);
                        }
                    }
                    enc.tile_numbers[slg_id] = tn;
                }
            }
        }
        err = g.init_stage_1(enc);
        if (err != ErrorType::SUCCESS) {
            std::cout << "Default init solution INVALID: " << err << std::endl;
            return false;
        }
        g.initTileCosts();
        g.getIdealCost(ideal_cost, false);
        err = g.getRealCost(real_cost, false);
        if (err != ErrorType::SUCCESS) {
            // cannot mitigate the issue
            std::cout << "Default init solution INVALID: " << err << std::endl;
            return false;
        }
    }
    last_g = g;
    last_enc = enc;
    best_enc = enc;
    best_cost = real_cost;
    if (print_results) {
        std::cout << "Stage 1 Init Encoding: " << std::endl;
        std::cout << enc;
        std::cout << "Stage 1 Init Intensity: " << std::endl;
        g.print_intensity();
        print_avg_buffer_usage(g, real_cost);
        std::cout << "Stage 1 Init Results: " << std::endl;
        std::cout << "Real Time: " << real_cost.time << ", Energy: " << real_cost.energy << ", Cost: " << real_cost.cost() << std::endl;
        std::cout << "Ideal Time: " << ideal_cost.time << ", Energy: " << ideal_cost.energy << ", Cost: " << ideal_cost.cost() << std::endl;
        // std::cout << g;
        // g.print_graph_result(); // must be called after getRealCost(, true)
    }

    // init internal data structure
    update_changeable_layers(g);
    return true;
}

OptimizationStatistics Stage1SA::solve(len_t num_rounds)
{
    std::chrono::steady_clock::time_point sa_begin = std::chrono::steady_clock::now();
    bool using_best = false;
    CoreMapper::MapCost last_cost = best_cost;
    unsigned long long better_cnt[5] = {0, 0, 0, 0, 0};
    unsigned long long acc_cnt[5] = {0, 0, 0, 0, 0};
    unsigned long long val_cnt[5] = {0, 0, 0, 0, 0};
    unsigned long long all_cnt[5] = {0, 0, 0, 0, 0};
    Graph g = last_g;
    std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 1 SA Start! Seed: " << _seed_here << std::endl;
    for (int cur_round = 0; cur_round < num_rounds; cur_round++) {
        // report every 10% num_rounds
        if (cur_round % (num_rounds / 100) == 0) {
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 1 Progress: " << (double)cur_round / (double)num_rounds << std::endl;
        }
        // std::cout << "Round:" << cur_round << std::endl;
        if (cur_round >= 0.90 * num_rounds && !using_best) {
            using_best = true;
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 1 Switch to best solution." << std::endl;
            g.init_stage_1(best_enc);
            update_changeable_layers(g);
            enc = best_enc;
            last_g = g;
            last_enc = best_enc;
            last_cost = best_cost;
        }
        // Randomly pick an operator to apply
#ifdef PERF_TIME_SPY
        std::chrono::steady_clock::time_point method_begin = std::chrono::steady_clock::now();
#endif
        int method = rand_multi_prob({0.2, 0.35, 0, 0.15, 0.3});
        bool success = false;
        if (!(enc == g.get_Encoding().first)) {
            std::cout << "Encoding not match with graph" << std::endl;
            std::cout << "Enc: \n" << enc;
            std::cout << "g.Encoding(): \n" << g.get_Encoding().first;
            assert(false);
        }
        assert(!enc.layer_group_partition.contains(network->len()));
        all_cnt[method]++;
        // change enc & update graph
        switch (method) {
            case 0: success = change_layer_order(g); break;
            case 1: success = change_tile_number(g); break;
            case 2: 
                success = change_layer_group(g); 
                // if (!success) success = change_sub_layer_group(g);
                break;
            case 3: 
                success = change_sub_layer_group(g); 
                // if (!success) success = change_layer_group(g);
                break;
            case 4: success = change_layer_group_by_layer(g); break;
            default: break;
        }
        assert(!enc.layer_group_partition.contains(network->len()));
#ifdef PERF_TIME_SPY
        std::chrono::steady_clock::time_point method_end = std::chrono::steady_clock::now();
        std::cout << "\t\tMethod Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(method_end - method_begin).count() << "ms" << std::endl;
        std::chrono::steady_clock::time_point init_stage1_begin, init_stage1_end;
        std::chrono::steady_clock::time_point init_cost_begin, init_cost_end;
        std::chrono::steady_clock::time_point get_cost_begin, get_cost_end;
        std::chrono::steady_clock::time_point update_begin, update_end;
#endif
        // enc is updated
        {
            vector<lid_t> enc_layer_id_to_order(network->len());
            for (int k = 0; k < network->len(); k++) {
                enc_layer_id_to_order[enc.layer_order_to_id[k]] = k;
            }
            // check: FC and its previous Pooling layers should be isolated with other parts of the graph
            for (auto & fc_lid : Graph::fc_layer_ids) {
                const lid_t& k = enc_layer_id_to_order[fc_lid];
                bool ok = true;
                lid_t prev_order = MAX(0, k - 1);
                for (; prev_order >= 0; prev_order--) {
                    const Node& n = network->getNode(enc.layer_order_to_id[prev_order]);
                    const Layer& l = n.layer();
                    if (!REF_IS_INSTANCE(l, PoolingLayer)) {
                        break;
                    }
                }
                prev_order++;
                if (!enc.layer_group_partition.contains(prev_order))
                    ok = false;
                if (k < network->len() - 1 && !enc.layer_group_partition.contains(k + 1))
                    ok = false;
                if (!ok) {
                    // std::cout << "Invalid Solution: FC/Former Pooling Layer " << fc_lid << " not isolated" << std::endl;
                    success = false;
                    break;
                }
            }
            // check: Transpose Layer and its prevs should never be in the same SLG
            // for (auto& tr_lid : Graph::transpose_layer_ids) {
            //     const lid_t& k = enc_layer_id_to_order[tr_lid];
            //     lid_t closest_prev = 0;
            //     FOR_BITSET(p, network->getNode(tr_lid).getPrevs()) {
            //         CMAX(closest_prev, enc_layer_id_to_order[p]);
            //     }
            //     if ((enc.layer_group_partition|enc.sub_layer_group_partition).next(closest_prev) > k) {
            //         std::cout << "Invalid Solution: Transpose Layer " << tr_lid << " and its prevs in same SLG" << std::endl;
            //         success = false;
            //         break;
            //     }
            // }
        }
        assert(!enc.layer_group_partition.contains(network->len()));

        bool slg_changed_by_make_connected = false;
        // std::cout << "\t\tMethod: " << method << std::endl;
        if (success) {
#ifdef PERF_TIME_SPY
            init_stage1_begin = std::chrono::steady_clock::now();
#endif
            ErrorType err;
            err = g.init_stage_1(enc);
            if (err == ErrorType::SUBGRAPH_NOT_FULLY_CONNECTED) {
                make_sure_all_slgs_are_fully_connected(g);
                err = g.init_stage_1(enc);
                if (enc == last_enc)
                    continue;
                if (err != ErrorType::SUCCESS) {
                    if (err == ErrorType::LAYER_ORDER_DEPENDENCY) {
                        std::cout << "Invalid Solution: " << err << "after SUBGRAPH_NOT_FULLY_CONNECTED" << std::endl; 
                        std::cout << "last_enc\n" << last_enc;
                        std::cout << "enc\n" << enc;
#ifdef PERF_TIME_SPY
                        // std::cout << "last_g\n" << last_g;
                        // std::cout << "g\n" << g;
                        assert(false);
#endif
                    }
                    // don't update enc
                    g = last_g;
                    enc = last_enc;
                } else {
                    assert(enc == g.get_Encoding().first);
                    slg_changed_by_make_connected = true;
                }
            } else if (err != ErrorType::SUCCESS) {
                // std::cout << "\t\tInvalid Solution: " << err << std::endl;
                if (err == ErrorType::LAYER_ORDER_DEPENDENCY) {
                    std::cout << "last_enc\n" << last_enc;
                    std::cout << "enc\n" << enc;
                    for (auto& x: changeable_layers) {
                        std::cout << "Layer ID: " << x.layer_id 
                                  << ", Order: " << x.order
                                  << ", [" << x.prev
                                  << "~" << x.next << "]" << std::endl;
                    }
                    // std::cout << "last_g\n" << last_g;
                    // std::cout << "g\n" << g;
#ifdef PERF_TIME_SPY
                    assert(false);
#endif
                }
                // don't update enc
                g = last_g;
                enc = last_enc;
#ifdef PERF_TIME_SPY
                init_stage1_end = std::chrono::steady_clock::now();
                std::cout << "\t\tInit Stage1 Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(init_stage1_end - init_stage1_begin).count() << "ms" << std::endl;
#endif
                continue;
            }
            // init_stage_1 success
#ifdef PERF_TIME_SPY
            init_stage1_end = std::chrono::steady_clock::now();
            std::cout << "\t\tInit Stage1 Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(init_stage1_end - init_stage1_begin).count() << "ms" << std::endl;
            init_cost_begin = std::chrono::steady_clock::now();
#endif
            g.initTileCosts();
#ifdef PERF_TIME_SPY
            init_cost_end = std::chrono::steady_clock::now();
            std::cout << "\t\tInit Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(init_cost_end - init_cost_begin).count() << "ms" << std::endl;
            get_cost_begin = std::chrono::steady_clock::now();
#endif
            CoreMapper::MapCost ideal_cost, real_cost;
            g.getIdealCost(ideal_cost, false);
            err = g.getRealCost(real_cost, false);
#ifdef PERF_TIME_SPY
            get_cost_end = std::chrono::steady_clock::now();
            std::cout << "\t\tGet Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(get_cost_end - get_cost_begin).count() << "ms" << std::endl;
#endif
            // std::cout << "\t\tReal Cost: " << real_cost.cost() << ", Ideal Cost: " << ideal_cost.cost() << std::endl;
#ifdef PERF_TIME_SPY
            update_begin = std::chrono::steady_clock::now();
#endif
            if (err == ErrorType::SUCCESS) {
                // std::cout << "Real Time: " << real_cost.time << ", Energy: " << real_cost.energy << ", Cost: " << real_cost.cost() << std::endl;
                val_cnt[method]++;
                if (best_cost.cost() > real_cost.cost()) {
                    better_cnt[method]++;
                    best_enc = enc;
                    best_cost = real_cost;
                    // std::cout << "\t\tBest Cost Update: " << real_cost.cost() << std::endl;
                }
                if (OptimizationStage::accept_by_progress(last_cost.cost(), real_cost.cost(), (double)cur_round/(double)num_rounds)) {
                    acc_cnt[method]++;
                    last_cost = real_cost;
                    last_g = g;
                    last_enc = enc;
                    if (method == 0 || slg_changed_by_make_connected) {// order change
                    // Re-compute order range: prev and next
                        update_changeable_layers(g);
                    }
                    // std::cout << "\t\tNew Last Cost: " << real_cost.cost() << std::endl;
                    // std::cout << "#LG = " << enc.layer_group_partition.count() << std::endl;
                    // std::cout << "#SLG = " << enc.tile_numbers.size() << std::endl;
                    // update = true;
                } else {
                    // don't update enc
                    g = last_g;
                    enc = last_enc;
#ifdef PERF_TIME_SPY
                    update_end = std::chrono::steady_clock::now();
                    std::cout << "\t\tUpdate Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_begin).count() << "ms" << std::endl;
#endif
                    continue;
                }
            } else {
                // std::cout << "\t\tInvalid Solution: " << err << std::endl;
                // don't update enc
                g = last_g;
                enc = last_enc;
#ifdef PERF_TIME_SPY
                update_end = std::chrono::steady_clock::now();
                std::cout << "\t\tUpdate Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_begin).count() << "ms" << std::endl;
#endif
                continue;
            }
#ifdef PERF_TIME_SPY
            update_end = std::chrono::steady_clock::now();
            std::cout << "\t\tUpdate Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_begin).count() << "ms" << std::endl;
#endif
        } else {
            enc = last_enc;
            continue;
        }
    }
    std::chrono::steady_clock::time_point sa_end = std::chrono::steady_clock::now();
    OptimizationStatistics stats;
    stats.search_time = std::chrono::duration_cast<std::chrono::seconds>(sa_end - sa_begin).count();
    stats.cache_stats.hit_cnt = Graph::num_tile_cost_cache_hit;
    stats.cache_stats.miss_cnt = Graph::num_tile_cost_cache_miss;
    stats.cache_stats.total_cnt = Graph::num_tile_cost_cache_total;
    const int method_num = 5;
    for (int i = 0; i < MIN(method_num, 5); i++) {
        stats.sa_cnts[i].better_cnt = better_cnt[i];
        stats.sa_cnts[i].acc_cnt = acc_cnt[i];
        stats.sa_cnts[i].val_cnt = val_cnt[i];
        stats.sa_cnts[i].all_cnt = all_cnt[i];
    }
    for (int i = MIN(method_num, 5); i < 5; i++) {
        stats.sa_cnts[i].better_cnt = 0;
        stats.sa_cnts[i].acc_cnt = 0;
        stats.sa_cnts[i].val_cnt = 0;
        stats.sa_cnts[i].all_cnt = 0;
    }
    return stats;
}

Stage2SA::tensor_constraint_range Stage2SA::tensor_constraint_range::intersect_with(const tensor_constraint_range& other) const
{
    tensor_constraint_range intersect;
    if (other.max < min || other.min > max) {
        intersect.min = intersect.max = -2;
    } else {
        intersect.min = MAX(min, other.min);
        intersect.max = MIN(max, other.max);
    }
    return intersect;
}

void Stage2SA::SimpleGraph::init(int n)
{
    forward.resize(n);
    reverse.resize(n);
}

void Stage2SA::SimpleGraph::add_edge(int from, int to) 
{
    // forward[from].push_back(to);
    // reverse[to].push_back(from);
    forward[from].insert(to);
    reverse[to].insert(from);
}

void Stage2SA::SparseGraph::add_edge(int from, int to) 
{
    forward[from].push_back(to);
    reverse[to].push_back(from);
}

void Stage2SA::SimpleGraph::del_edge(int from, int to) 
{
    // we can do binary search here
    if (auto it = forward[from].erase(to); it == 0) {
        std::cerr << "Error: cannot find edge " << from << " -> " << to << std::endl;
        assert(false && "Error: cannot find edge");
    }
    if (auto it = reverse[to].erase(from); it == 0) {
        std::cerr << "Error: cannot find reverse edge " << to << " -> " << from << std::endl;
        assert(false && "Error: cannot find reverse edge");
    }
}

void Stage2SA::SimpleGraph::del_single_direct_edge(int from, int to, bool is_forawrd) 
{
    // we can do binary search here
    if(is_forawrd) {
        if (auto it = forward[from].erase(to); it == 0) {
            std::cerr << "Error: cannot find edge " << from << " -> " << to << std::endl;
            assert(false && "Error: cannot find edge");
        }
    } else {
        if (auto it = reverse[to].erase(from); it == 0) {
            std::cerr << "Error: cannot find reverse edge " << to << " -> " << from << std::endl;
            assert(false && "Error: cannot find reverse edge");
        }
    }
}

void Stage2SA::SimpleGraph::print_graph() const 
{
    for (size_t i = 0; i < forward.size(); ++i) {
        std::cout << "Node " << i << ":";
        for (const auto& x : forward[i]) {
            std::cout << ", " << x;
        }
        std::cout << std::endl;
    }
}

void Stage2SA::SimpleGraph::print_nodes(const Graph& g) const 
{
    // Id=tensor_id, Label=tensor_info_set, tensor_info_set_type
    std::cout << "Id, Label, tensor_infos, tensor_info_set_type" << std::endl; 
    for (size_t i = 0; i < forward.size(); ++i) {
        std::cout << i << ", " << i << ", [";
        auto range = g.tensor_id_to_info.equal_range(i);
        auto it = range.first;
        std::cout << it->second.layer_id << "|" << it->second.tile_id << "|" << it->second.tensor_type << "|" << it->second.source;
        it++;
        if (g.tensor_info_set_types[i] != Graph::TensorInfoSetType::ONLY_WGTs) {
            for (; it != range.second; ++it) {
                std::cout << "; " << it->second.layer_id << "|" << it->second.tile_id << "|" << it->second.tensor_type << "|" << it->second.source;
            }
        }

        std::cout << "]" << ", " << g.tensor_info_set_types[i] << std::endl;
    }
}

void Stage2SA::SimpleGraph::print_edges() const 
{
    std::cout << "Source, Target" << std::endl; 
    for (size_t i = 0; i < forward.size(); ++i) {
        for (const auto& x : forward[i]) {
            std::cout << i << ", " << x << std::endl;
        }
    }
}

void Stage2SA::SparseGraph::print_graph() const 
{
    for (auto& i : forward) {
        std::cout << "Node " << i.first << ":";
        for (const auto& x : i.second) {
            std::cout << " " << x;
        }
        std::cout << std::endl;
    }
}

Stage2SA::Stage2SA(const Network* _network, const CoreMapper::MapCost& stage1_best_cost, const Graph& stage1_best_g) : OptimizationStage(_network)
{
    /*
    Rule for Validation:    i, i+1, i+2, ..., j-1, j => i+1, i+2, ..., j-1, j, i
        RV1: ∀ i<j in dram tensor order, i.start < j.end
            - for a order range [prev, next] and current order i:
                - ∀ j ∈ [prev, i), i.start < j.end
                - ∀ j ∈ (i, next], i.end > j.start
            - for every tensor with order k, if it is 
                - IFM/WGT: start < min_end[k]
                - OFM: end > max_start[k]
        RV2: Dependencies
            RV2.1: Dependency of OFM & IFM 
                - across LG: 
                    A->B, OFM_A_last_tile's order < IFM_B_first_tile's order
                    OFM_A_last_tile's start <= IFM_B_first_tile's start
                - across SLG in the same LG: 
                    A->B, OFM_A_last_tile's order < OFM_B_first_tile's order
                    No constraints on start/end because OFMs' start are fixed
                - in the same SLG: 
                    A->B, OFM_A_each_tile's order < OFM_B_each_tile's order
                    No constraints on start/end because OFMs' start are fixed
                - of the same layer: 
                    IFM_this_layer_each_tile's order < OFM_this_layer_each_tile's order
                    No constraints because it is trivial (use RV3): 
                        IFM_this_layer_each_tile's start <= this_tile's pos < OFM_this_layer_each_tile's end
            RV2.2: Dependency of WGT (if this layer has weight) 
                - for every layer, 
                    WGT_this_layer's order < OFM_this_layer_first_tile's order
                    No constraints because it is trivial (use RV3): 
                        WGT_this_layer_each_tile's start <= this_tile's pos < OFM_this_layer_each_tile's end
        RV3: max of start and min of end
            - for every IFM, max of start = this layer this tile's pos
            - for every WGT, max of start = this layer first tile's pos
            - for every OFM, min of end = this layer this tile's pos + 1
    
    Rule for Pruning:
        RP1: different tiles of the same layer should follow comp order:
            - for every layer, tile T0, T1, ..., T_L:
                IFM_T0's order < IFM_T1's order < ... < IFM_T_L's order
                OFM_T0's order < OFM_T1's order < ... < OFM_T_L's order
                IFM_T0's start <= IFM_T1's start <= ... <= IFM_T_L's start
                OFM_T0's end <= OFM_T1's end <= ... <= OFM_T_L's end
        RP2: (optional) follow the dram tensor order, start should be non-decreasing
            - this rule overides RV1 (is stronger), because ∀ i<j in dram tensor order, i.start <= j.start < j.end
            - RP2 makes hasse_diag compacting unnecessary
            - RP2 only allow very locally change, which might not be good for SA
    */
    Graph::Buffer::set_stage_1(false);
    // 1. init last_g, last_enc, best_enc, best_cost from stage1
    // cur_stage1 = &stage1;
    last_g = stage1_best_g;
    last_g.getIdealCost(s1_ideal_cost, false);
    ideal_sum_buffer_usage = last_g.get_sum_buffer_usage();
    enc = last_g.get_Encoding().second; // TODO: examine that last_g is the best graph
    last_enc  = enc;
    best_enc  = enc;
    best_cost = stage1_best_cost;
    num_dram_tiles = last_g.tile_tensor_order.size();
    total_tile_number = last_g.layer_id_to_tile_pos[last_g.layer_order_to_id[network->len() - 1]].end_time + 1;
    // init tile_tensor_id_to_order
    tile_tensor_id_to_order.reserve(num_dram_tiles);
    dram_tensor_id_to_size.resize(last_g.tensor_times.size(), 0);
    double min_wgt_size = Graph::Buffer::get_max_buffer_size();
    vector<len_t /*tensor_id*/> zero_size_tensors;
    for (len_t i = 0; i < num_dram_tiles; ++i) {
        tile_tensor_id_to_order[last_g.tile_tensor_order[i]] = i;
        double size = last_g.tensor_id_to_size.at(last_g.tile_tensor_order[i]);
        if (size > 0) {
            if (last_g.tensor_info_set_types[last_g.tile_tensor_order[i]] == Graph::TensorInfoSetType::ONLY_WGTs) {
                CMIN(min_wgt_size, size);
            }
            dram_tensor_id_to_size[last_g.tile_tensor_order[i]] = size;
        } else {
            zero_size_tensors.push_back(last_g.tile_tensor_order[i]);
            // assert(false && "find zero size tensor");
        }
    }
    for (auto& x : zero_size_tensors) {
        dram_tensor_id_to_size[x] = min_wgt_size;
    }
    /*
    for (len_t i = 0; i < num_dram_tiles; ++i) {
        dram_tensor_id_to_size[last_g.tile_tensor_order[i]] = std::exp(dram_tensor_id_to_size[last_g.tile_tensor_order[i]]);
    }
    double min_dram_tensor_size = dram_tensor_id_to_size[last_g.tile_tensor_order[0]];
    for (len_t i = 1; i < num_dram_tiles; ++i) {
        CMIN(min_dram_tensor_size, dram_tensor_id_to_size[last_g.tile_tensor_order[i]]);
    }
    for (len_t i = 0; i < num_dram_tiles; ++i) {
        dram_tensor_id_to_size[last_g.tile_tensor_order[i]] /= min_dram_tensor_size;
    }
    */

    // 2. init star/end related internal data structures

    // 2.0. init ith_start_pos & ith_start_val
    // start_val_to_pos[-1] = 0;
    // start_pos_to_val[0] = -1;
    // for (len_t i = 0; i < num_dram_tiles; ++i) {
    //     const auto& tile_tensor_start = last_g.tensor_times[last_g.tile_tensor_order[i]].start_time;
    //     if (start_val_to_pos.find(tile_tensor_start) != start_val_to_pos.end()) {
    //         start_val_to_pos.emplace(tile_tensor_start, i);
    //         start_pos_to_val.emplace(i, tile_tensor_start);
    //     }
    // }

    // 2.1. init max start prefix array
    //      max_start = new int[num_dram_tiles];
    max_start.resize(num_dram_tiles);
    max_start[0] = last_g.tensor_times[last_g.tile_tensor_order[0]].start_time;
    for (int i = 1; i < num_dram_tiles; ++i) {
        max_start[i] = MAX(max_start[i - 1], last_g.tensor_times[last_g.tile_tensor_order[i]].start_time);
    }
    // 2.2. init min end suffix array
    //      min_end = new int[num_dram_tiles];
    min_end.resize(num_dram_tiles);
    min_end[num_dram_tiles - 1] = last_g.tensor_times[last_g.tile_tensor_order[num_dram_tiles - 1]].end_time;
    for (int i = num_dram_tiles - 2; i >= 0; --i) {
        min_end[i] = MIN(min_end[i + 1], last_g.tensor_times[last_g.tile_tensor_order[i]].end_time);
    }
    // 3. apply RV3 ONLY ONCE and store RV3 constraints in dram_tensor_id_to_tile_pos
    dram_tensor_id_to_tile_pos.reserve(num_dram_tiles);
    for (len_t k = 0; k < num_dram_tiles; ++k) {
        const len_t& t_id = last_g.tile_tensor_order[k];
        auto range = last_g.tensor_id_to_info.equal_range(t_id);
        assert(range.first != last_g.tensor_id_to_info.end());
        const Graph::TensorInfoSetType& ti_type = last_g.tensor_info_set_types[t_id];
        if (ti_type == Graph::TensorInfoSetType::ONLY_IFMs) {
            assert(distance(range.first, range.second) == 1);
            const Graph::TensorInfo& info = range.first->second;
            const auto& slg = last_g.all_slgs[last_g.layer_id_to_lg_slg[info.layer_id].second];
            // lid_t slg_len= slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
            len_t this_tile_actual_pos = last_g.layer_id_to_tile_pos[info.layer_id].start_time + info.tile_id * slg.layer_num();
            dram_tensor_id_to_tile_pos[t_id] = this_tile_actual_pos; // start max
        } else if (ti_type == Graph::TensorInfoSetType::ONLY_WGTs) {
            // assert(distance(range.first, range.second) == 1); // all tiles of the same layer have the same WGT tensor_id
            const Graph::TensorInfo& info = range.first->second;
            dram_tensor_id_to_tile_pos[t_id] = last_g.layer_id_to_tile_pos[info.layer_id].start_time; // start max
        } else if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
            const Graph::TensorInfo info = last_g.getOFMInfo(range);
            const auto& slg = last_g.all_slgs[last_g.layer_id_to_lg_slg[info.layer_id].second];
            // lid_t slg_len= slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
            len_t this_tile_actual_pos = last_g.layer_id_to_tile_pos[info.layer_id].start_time + info.tile_id * slg.layer_num();
            dram_tensor_id_to_tile_pos[t_id] = this_tile_actual_pos + 1; // end min
        } else {
            assert(false && "find OFM_WITH_LOCAL_IFMs in dram tensor order");
        }
    }
    // 4. init Tensor Hasse Diagram
    hasse_diag.init(last_g.tensor_times.size());
    init_Hasse_Diag(last_g);
    // hasse_diag.print_nodes(last_g);
    // hasse_diag.print_edges();
    compact_Hasse_Diag(last_g);
    // hasse_diag.print_nodes(last_g);
    // hasse_diag.print_edges();
    assert(hasse_diag.forward.size() == last_g.tensor_times.size());
#ifdef DEBUG
    for (int i = 0; i < hasse_diag.forward.size(); ++i) {
        if (last_g.tensor_info_set_types[i] == Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs) 
            if (hasse_diag.forward[i].size() > 0 || hasse_diag.reverse[i].size() > 0) {
                Graph::TensorInfo ofm_info = last_g.getOFMInfo(last_g.tensor_id_to_info.equal_range(i));
                std::cout << ofm_info;
                assert(false && "OFM_WITH_LOCAL_IFMs should not have any dependencies");
            }
    }
#endif
}

/*Stage2SA::~Stage2SA()
{
    // delete[] max_start;
    // delete[] min_end;
}*/

void Stage2SA::init_Hasse_Diag(const Graph& g)
{
    for (lid_t i = 0; i < g.layer_groups.size(); ++i) {
        const Graph::LayerGroup& lg = g.layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const Graph::SubLayerGroup& slg = g.all_slgs[j];
            lid_t slg_len = slg.layer_num(); // slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = g.layer_order_to_id[k];
                const Node& node = network->getNode(layer_id);
                const Layer& layer = node.layer();
                // 1. get every tensor_id of this layer
                // 1.1. get OFM tensor_id of each_tile (including internal OFM_WITH_IFMs)
                vector<len_t /*tensor_id*/> ofm_tensor_ids_by_tile(slg.tile_number);
                for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                    Graph::TensorInfo info_ofm = { layer_id, tile_id, Graph::TensorType::OFM, 0 };
                    const auto range = g.tensor_info_to_id.equal_range(info_ofm);
                    assert(range.first != g.tensor_info_to_id.end());
                    assert(distance(range.first, range.second) == 1);
                    ofm_tensor_ids_by_tile[tile_id] = range.first->second;
                }
                
                // 1.2. get DRAM-related IFM tensor_id of each_tile
                vector< map<int /*source*/, len_t /*tensor_id*/> > ifm_tensor_ids_by_source_by_tile(slg.tile_number);
                FOR_BITSET(ifm_id, node.getPrevs())
                {
                    if (g.layer_id_to_order[ifm_id] < lg.layer_group_start) { // input NOT IN local LG
                        for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                            Graph::TensorInfo info_ifm = { layer_id, tile_id, Graph::TensorType::IFM, ifm_id };
                            const auto range = g.tensor_info_to_id.equal_range(info_ifm);
                            assert(range.first != g.tensor_info_to_id.end());
                            assert(distance(range.first, range.second) == 1);
                            ifm_tensor_ids_by_source_by_tile[tile_id].emplace(ifm_id, range.first->second);
                        }
                    }
                }
                FOR_BITSET(ext_id, node.getExtPrevs())
                {
                    for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                        Graph::TensorInfo info_ifm = { layer_id, tile_id, Graph::TensorType::IFM, -ext_id - 1 };
                        const auto range = g.tensor_info_to_id.equal_range(info_ifm);
                        assert(range.first != g.tensor_info_to_id.end());
                        assert(distance(range.first, range.second) == 1);
                        ifm_tensor_ids_by_source_by_tile[tile_id].emplace(ext_id, range.first->second);
                    }
                }
                
                // 1.3. get WGT tensor_id
                len_t wgt_tensor_id;
                if (!node.hasWgtPrevs() && layer.weight_size() > 0)
                    wgt_tensor_id = g.tensor_info_to_id.equal_range({layer_id, 0, Graph::TensorType::WGT, 0}).first->second;

                // 2. set order dependency for each tensor
                // 2.1. IFM
                for (len_t tile_id = 0; tile_id < ifm_tensor_ids_by_source_by_tile.size(); ++tile_id) {
                    const auto& ifm_tensor_ids_by_source = ifm_tensor_ids_by_source_by_tile[tile_id];
                    len_t this_tile_actual_pos = g.layer_id_to_tile_pos[layer_id].start_time + tile_id * slg_len;
                    for (auto& source_and_ifm_or_ext_id : ifm_tensor_ids_by_source) {
                        const int& source = source_and_ifm_or_ext_id.first;
                        const len_t& t_id = source_and_ifm_or_ext_id.second;
                        // 2.1.1. IFM chain order
                        if (tile_id > 0) {
                            // IFM_this_tile's prev   > IFM_prev_tile's order 
                            len_t prev_tile_tensor_id = ifm_tensor_ids_by_source_by_tile[tile_id - 1].at(source);
                            hasse_diag.add_edge(prev_tile_tensor_id, t_id);
                            IFM_tiles_constraints.add_edge(prev_tile_tensor_id, t_id);
                        } else { // first tile of this layer
                            // Ext and Inter-LG IFMs are done in 2.3. OFM
                        }
                        // 2.1.2. IFM_this_tile's order < OFM_this_tile's order
                        hasse_diag.add_edge(t_id, ofm_tensor_ids_by_tile[tile_id]);
                    }
                }
                // 2.2. WGT
                // WGT_this_layer < OFM_this_layer_1st_tile
                if (!node.hasWgtPrevs() && layer.weight_size() > 0) {
                    hasse_diag.add_edge(wgt_tensor_id, ofm_tensor_ids_by_tile[0]); 
                }

                // 2.3. OFM
                for (len_t tile_id = 0; tile_id < ofm_tensor_ids_by_tile.size(); ++tile_id) {
                    const len_t& t_id = ofm_tensor_ids_by_tile[tile_id];

                    // 2.3.1. OFMs chain order
                    if (tile_id < slg.tile_number - 1) {
                        // OFM_this_tile's order < OFM_next_tile's order
                        len_t next_tile_tensor_id = ofm_tensor_ids_by_tile[tile_id + 1];
                        hasse_diag.add_edge(t_id, next_tile_tensor_id);
                        OFM_tiles_constraints.add_edge(t_id, next_tile_tensor_id);
                    } else { 
                        // 2.3.2. last tile's OFM of this layer
                        FOR_BITSET(ofm_id, node.get_nexts())
                        {
                            if (g.layer_id_to_order[ofm_id] > lg.layer_group_end) { // output NOT IN local LG
                                // OFM_this_layer_last_tile's order < IFM_next_layer_first_tile's order
                                Graph::TensorInfo info_ifm = { ofm_id, 0, Graph::TensorType::IFM, layer_id };
                                const auto range = g.tensor_info_to_id.equal_range(info_ifm);
                                assert(distance(range.first, range.second) == 1);
                                len_t next_layer_1st_tile_ifm_tensor_id = range.first->second;
                                hasse_diag.add_edge(t_id, next_layer_1st_tile_ifm_tensor_id);
                                across_lg_constraints.add_edge(t_id, next_layer_1st_tile_ifm_tensor_id);
                            } else if (g.layer_id_to_order[ofm_id] > slg.sub_layer_group_end) { // output IN local LG but NOT IN this SLG
                                // OFM_this_layer_last_tile's order < OFM_next_layer_first_tile's order
                                Graph::TensorInfo info_ofm = { ofm_id, 0, Graph::TensorType::OFM, 0 };
                                const auto range = g.tensor_info_to_id.equal_range(info_ofm);
                                assert(distance(range.first, range.second) == 1);
                                len_t next_layer_1st_tile_ofm_tensor_id = range.first->second;
                                hasse_diag.add_edge(t_id, next_layer_1st_tile_ofm_tensor_id);
                            } else { // output IN this SLG
                                // See 2.3.3.
                            }
                        }
                    }
                    // 2.3.3. Deal with layers in the same SLG
                    // OFM_this_layer_ith_tile's order < OFM_next_layer_ith_tile's order
                    FOR_BITSET(ofm_id, node.get_nexts())
                    {
                        if (g.layer_id_to_order[ofm_id] <= slg.sub_layer_group_end) { // output IN this SLG
                            Graph::TensorInfo info_ofm = { ofm_id, tile_id, Graph::TensorType::OFM, 0 };
                            const auto range = g.tensor_info_to_id.equal_range(info_ofm);
                            assert(distance(range.first, range.second) == 1);
                            len_t next_layer_ith_tile_ofm_tensor_id = range.first->second;
                            hasse_diag.add_edge(t_id, next_layer_ith_tile_ofm_tensor_id);
                        }
                    }
                }
            }
        }
    }
}

void Stage2SA::compact_Hasse_Diag(const Graph& g)
{
    for (lid_t i = 0; i < g.layer_groups.size(); ++i) {
        const Graph::LayerGroup& lg = g.layer_groups[i];
        for (lid_t j = lg.slg_idx_start; j <= lg.slg_idx_end; ++j) {
            const Graph::SubLayerGroup& slg = g.all_slgs[j];
            lid_t slg_len = slg.layer_num(); // slg.sub_layer_group_end - slg.sub_layer_group_start + 1;
            for (lid_t k = slg.sub_layer_group_start; k <= slg.sub_layer_group_end; ++k) {
                const lid_t& layer_id = g.layer_order_to_id[k];
                const Node& node = network->getNode(layer_id);

                Graph::TensorInfo first_info_ofm = { layer_id, 0, Graph::TensorType::OFM, 0 };
                const auto first_ofm_range = g.tensor_info_to_id.equal_range(first_info_ofm);
                assert(distance(first_ofm_range.first, first_ofm_range.second) == 1);
                const len_t& first_ofm_tensor_id = first_ofm_range.first->second;
                if (g.tensor_info_set_types[first_ofm_tensor_id] == Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs) {
                    // this mean we need to compact this layer's OFM
                    // 1. get every OFM tensor_id of this layer
                    vector<len_t /*tensor_id*/> ofm_tensor_ids_by_tile(slg.tile_number);
                    ofm_tensor_ids_by_tile[0] = first_ofm_tensor_id;
                    for (len_t tile_id = 1; tile_id < slg.tile_number; ++tile_id) {
                        Graph::TensorInfo info_ofm = { layer_id, tile_id, Graph::TensorType::OFM, 0 };
                        const auto range = g.tensor_info_to_id.equal_range(info_ofm);
                        assert(distance(range.first, range.second) == 1);
                        ofm_tensor_ids_by_tile[tile_id] = range.first->second;
                    }
                    // 2. link all its prevs layer's OFM(may in the form of IFMs) to next layer's OFM
                    // previous links may change, so directly reference to hasse_diag is necessary
                    for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                        const len_t& t_id = ofm_tensor_ids_by_tile[tile_id];
                        for (len_t u : hasse_diag.reverse[t_id]) {
                            if (tile_id > 0 && u == ofm_tensor_ids_by_tile[tile_id - 1])
                                continue;
                            for (len_t v : hasse_diag.forward[t_id]) {
                                if (tile_id < slg.tile_number - 1 && v == ofm_tensor_ids_by_tile[tile_id + 1])
                                    continue;
                                hasse_diag.add_edge(u, v);
                            }
                        }
                    }
                    if (slg.tile_number > 1) {
                        // Find tensors from same LG but diff SLG
                        set<len_t> first_tile_special_t_ids, last_tile_special_t_ids;
                        for (len_t u : hasse_diag.reverse[ofm_tensor_ids_by_tile[0]]) {
                            if (g.tensor_info_set_types[u] == Graph::TensorInfoSetType::ONLY_WGTs) {
                                first_tile_special_t_ids.insert(u);
                            } else { 
                                auto range = g.tensor_id_to_info.equal_range(u);
                                Graph::TensorInfo info;
                                if (g.tensor_info_set_types[u] == Graph::TensorInfoSetType::ONLY_IFMs) {
                                    assert(distance(range.first, range.second) == 1);
                                    info = range.first->second;
                                } else if (g.tensor_info_set_types[u] == Graph::TensorInfoSetType::OFM_TO_DRAM) {
                                    info = g.getOFMInfo(range);
                                } else { assert(false && "find OFM_WITH_LOCAL_IFMs in dram tensor order"); }
                                lid_t lg_id = g.layer_id_to_lg_slg[info.layer_id].first;
                                lid_t slg_id = g.layer_id_to_lg_slg[info.layer_id].second;
                                if (lg_id == i && slg_id != j) {
                                    assert(slg_id < j);
                                    first_tile_special_t_ids.insert(u);
                                }
                            }
                        }
                        assert(first_tile_special_t_ids.size() == hasse_diag.reverse[ofm_tensor_ids_by_tile[0]].size() - hasse_diag.reverse[ofm_tensor_ids_by_tile[1]].size() + 1);
                        for (len_t u : hasse_diag.forward[ofm_tensor_ids_by_tile[slg.tile_number - 1]]) {
                            assert(g.tensor_info_set_types[u] != Graph::TensorInfoSetType::ONLY_WGTs);
                            assert(g.tensor_info_set_types[u] != Graph::TensorInfoSetType::ONLY_IFMs);
                            auto range = g.tensor_id_to_info.equal_range(u);
                            Graph::TensorInfo info;
                            info = g.getOFMInfo(range);
                            lid_t lg_id = g.layer_id_to_lg_slg[info.layer_id].first;
                            lid_t slg_id = g.layer_id_to_lg_slg[info.layer_id].second;
                            if (lg_id == i && slg_id != j) {
                                assert(slg_id > j);
                                last_tile_special_t_ids.insert(u);
                            }
                        }
                        assert(last_tile_special_t_ids.size() == hasse_diag.forward[ofm_tensor_ids_by_tile[slg.tile_number - 1]].size() - hasse_diag.forward[ofm_tensor_ids_by_tile[slg.tile_number - 2]].size() + 1);

                        for (const auto& u : first_tile_special_t_ids) {
                            for (const auto& v : last_tile_special_t_ids) {
                                hasse_diag.add_edge(u, v);
                            }
                        }
                    }
                    // 3. clear all edges connected to this layer's OFM
                    for (len_t tile_id = 0; tile_id < slg.tile_number; ++tile_id) {
                        const len_t& t_id = ofm_tensor_ids_by_tile[tile_id];
                        for (int u : hasse_diag.reverse[t_id]) {
                            hasse_diag.del_single_direct_edge(u, t_id, true);
                            // hasse_diag.del_edge(u, t_id);
                        }
                        for (int v : hasse_diag.forward[t_id]) {
                            hasse_diag.del_single_direct_edge(t_id, v, false);
                            // hasse_diag.del_edge(t_id, v);
                        }
                        hasse_diag.forward[t_id].clear();
                        hasse_diag.reverse[t_id].clear();
                        assert(hasse_diag.reverse[t_id].empty());
                        assert(hasse_diag.forward[t_id].empty());
                    }
                }
            }
        }
    }
}

bool Stage2SA::change_tensor_order_and_time(const Graph& g)
{
    constexpr int MAX_TRIAL_FOR_ORDER = 3;
    constexpr int MAX_TRIAL_FOR_TIME = 10;
    // 0. randomly pick a tensor to change
    int tt_id;
    tensor_constraint_range tr_time;
    bool time_try_success = false;
    for (int time_try = 0; time_try < MAX_TRIAL_FOR_TIME; time_try++) {
        bool order_try_success = false;
        tensor_constraint_range tr_order;
        for (int order_try = 0; order_try < MAX_TRIAL_FOR_ORDER; order_try++) {
            tt_id = rand_multi_prob(dram_tensor_id_to_size);
            if (g.tensor_info_set_types[tt_id] == Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs)
                continue;
            {
                // Order Range
                // 0.0. init prev and next
                tr_order.min = 0;
                tr_order.max = num_dram_tiles - 1;
                // 0.1. apply RV2 & RP1 on order range
                for (auto prev_t_id : hasse_diag.reverse[tt_id])
                    CMAX(tr_order.min, tile_tensor_id_to_order.at(prev_t_id) + 1);
                for (auto next_t_id : hasse_diag.forward[tt_id])
                    CMIN(tr_order.max, tile_tensor_id_to_order.at(next_t_id) - 1);
                assert(tr_order.min <= tile_tensor_id_to_order.at(tt_id) && tile_tensor_id_to_order.at(tt_id) <= tr_order.max);
            }
            if (tr_order.min < tr_order.max) {
                order_try_success = true;
                break;
            }
        }
        if (!order_try_success) {
            continue;
        }
        cur_change.tensor_id = tt_id;
        const len_t tt_order = tile_tensor_id_to_order.at(tt_id);
        cur_change.old_sch = tt_order;
        const len_t new_order = randint_except(tr_order.min, tt_order, tr_order.max);
        cur_change.new_sch = new_order;
        enc.tile_tensor_order.erase(enc.tile_tensor_order.begin() + tt_order);
        enc.tile_tensor_order.insert(enc.tile_tensor_order.begin() + new_order, tt_id);

        // 1. pseudo-update internal data structures
        // @attetion maybe we dont need to update pseudo_tile_tensor_id_to_order
        // std::unordered_map<len_t, len_t> pseudo_tile_tensor_id_to_order = tile_tensor_id_to_order;
        std::vector<int> pseudo_max_start = max_start;
        std::vector<int> pseudo_min_end = min_end;
        {
            // 1.1. update tile_tensor_id_to_order
            const int former_pos = MIN(cur_change.old_sch, cur_change.new_sch);
            const int latter_pos = MAX(cur_change.old_sch, cur_change.new_sch);
            
            // for (len_t i = former_pos; i <= latter_pos; ++i) {
            //     pseudo_tile_tensor_id_to_order[enc.tile_tensor_order[i]] = i;
            // }
            // assert(num_dram_tiles == pseudo_tile_tensor_id_to_order.size());
            // 1.2. update max start and min end in order form
            {
                int i = former_pos;
                if (i == 0) {
                    pseudo_max_start[i] = g.tensor_times[enc.tile_tensor_order[i]].start_time;
                    i++;
                }
                for (; i <= latter_pos; ++i)
                    pseudo_max_start[i] = MAX(pseudo_max_start[i - 1], g.tensor_times[enc.tile_tensor_order[i]].start_time);
            }
            {
                int i = latter_pos;
                if (i == num_dram_tiles - 1) {
                    pseudo_min_end[i] = g.tensor_times[enc.tile_tensor_order[i]].end_time;
                    i--;
                }
                for (; i >= former_pos; --i)
                    pseudo_min_end[i] = MIN(pseudo_min_end[i + 1], g.tensor_times[enc.tile_tensor_order[i]].end_time);
            }
        }

        // 2. get Start/End range of this tensor
        const Graph::TensorInfoSetType& ti_type = g.tensor_info_set_types[tt_id];
        {
            // 2.0. init min and max
            tr_time.min = -1;
            tr_time.max = total_tile_number;
            // 2.1. apply RV2 on s/e range
            // - across LG: A->B, OFM_A_last_tile's start <= IFM_B_first_tile's start
            if (ti_type == Graph::TensorInfoSetType::ONLY_IFMs) {
                if (auto it = across_lg_constraints.reverse.find(tt_id); it != across_lg_constraints.reverse.end()) {
                    for (auto& prev_tid : it->second)
                        CMAX(tr_time.min, g.tensor_times[prev_tid].start_time);
                }
            }
            // 2.2. apply RP1 on s/e range
            if (ti_type == Graph::TensorInfoSetType::ONLY_IFMs) {
                if (auto it = IFM_tiles_constraints.reverse.find(tt_id); it != IFM_tiles_constraints.reverse.end()) {
                    for (auto& prev_tid : it->second)
                        CMAX(tr_time.min, g.tensor_times[prev_tid].start_time);
                }
                if (auto it = IFM_tiles_constraints.forward.find(tt_id); it != IFM_tiles_constraints.forward.end()) {
                    for (auto& next_tid : it->second)
                        CMIN(tr_time.max, g.tensor_times[next_tid].start_time);
                }
            } else if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
                if (auto it = OFM_tiles_constraints.reverse.find(tt_id); it != OFM_tiles_constraints.reverse.end()) {
                    for (auto& prev_tid : it->second)
                        CMAX(tr_time.min, g.tensor_times[prev_tid].end_time);
                }
                if (auto it = OFM_tiles_constraints.forward.find(tt_id); it != OFM_tiles_constraints.forward.end()) {
                    for (auto& next_tid : it->second)
                        CMIN(tr_time.max, g.tensor_times[next_tid].end_time);
                }
            }
            // 2.3. apply RV3 on s/e range
            if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
                CMAX(tr_time.min, dram_tensor_id_to_tile_pos[tt_id]);
            } else { // IFM or WGT
                CMIN(tr_time.max, dram_tensor_id_to_tile_pos[tt_id]);
            }
        }
        // 2.4. apply RV1 on s/e range
        // - IFM/WGT: start < min_end[k]
        // - OFM: end > max_start[k]
        if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
            CMAX(tr_time.min, pseudo_max_start[new_order] + 1); // end min
            if (tr_time.min < tr_time.max && g.tensor_times[tt_id].start_time < pseudo_min_end[new_order]) {
                time_try_success = true;
                break;
            } else {
                enc.tile_tensor_order.erase(enc.tile_tensor_order.begin() + new_order);
                enc.tile_tensor_order.insert(enc.tile_tensor_order.begin() + tt_order, tt_id);
            }
        } else { // IFM or WGT
            CMIN(tr_time.max, pseudo_min_end[new_order] - 1); // start max
            if (tr_time.min < tr_time.max && g.tensor_times[tt_id].end_time > pseudo_max_start[new_order]) {
                time_try_success = true;
                break;
            } else {
                enc.tile_tensor_order.erase(enc.tile_tensor_order.begin() + new_order);
                enc.tile_tensor_order.insert(enc.tile_tensor_order.begin() + tt_order, tt_id);
            }
        }
    }
    if (!time_try_success) {
        return false;
    }
    auto is_ofm = [&g](const len_t& t_id) -> int {
        return g.tensor_info_set_types[t_id] == Graph::TensorInfoSetType::OFM_TO_DRAM;
    };
    // 3. change the S/E time according to the range
    /*
    // try to not use local tensor
    if (is_ofm(tt_id)) {
        const auto old_end_time = enc.tensor_times[tt_id].end_time; 
        enc.tensor_times[tt_id].end_time = randint(tr_time.min, tr_time.max);
        if (old_end_time == enc.tensor_times[tt_id].end_time && tr_time.min < tr_time.max) {
            enc.tensor_times[tt_id].end_time = randint_except(tr_time.min, old_end_time, tr_time.max);
        } else {
            return false;
        }
    } else {
        const auto old_start_time = enc.tensor_times[tt_id].start_time;
        enc.tensor_times[tt_id].start_time = randint(tr_time.min, tr_time.max);
        if (old_start_time == enc.tensor_times[tt_id].start_time && tr_time.min < tr_time.max) {
            enc.tensor_times[tt_id].start_time = randint_except(tr_time.min, old_start_time, tr_time.max);
        } else {
            return false;
        }
    }
    return true;
    */
    const auto& cur_tt_order = cur_change.new_sch;
    tensor_constraint_range tr_local_se;

    if (cur_tt_order == 0) {
        tr_local_se.min = -1;
    } else {
        const auto& prev_tt_id = enc.tile_tensor_order[cur_tt_order - 1];
#ifdef DEBUG
        std::cout << "\t\t\tP: " << g.tensor_info_set_types[prev_tt_id] << "@" << g.tensor_times[prev_tt_id];
#endif
        if (is_ofm(tt_id)) {
            tr_local_se.min = g.tensor_times[prev_tt_id].end_time - 1 + is_ofm(prev_tt_id);
        } else {
            tr_local_se.min = g.tensor_times[prev_tt_id].start_time + is_ofm(prev_tt_id);
        }
    }
#ifdef DEBUG
    std::cout << "\t\t\tM: "  << g.tensor_info_set_types[tt_id] << "@" << g.tensor_times[tt_id];
#endif
    if (cur_tt_order == num_dram_tiles - 1) {
        tr_local_se.max = num_dram_tiles - 1;
    } else {
        const auto& next_tt_id = enc.tile_tensor_order[cur_tt_order + 1];
#ifdef DEBUG
        std::cout << "\t\t\tN: " << g.tensor_info_set_types[next_tt_id] << "@" << g.tensor_times[next_tt_id];
#endif
        if (is_ofm(tt_id)) {
            tr_local_se.max = g.tensor_times[next_tt_id].end_time - 1 + is_ofm(next_tt_id);
        } else {
            tr_local_se.max = g.tensor_times[next_tt_id].start_time + is_ofm(next_tt_id);
        }
    }
    tr_local_se.min = MIN(tr_local_se.min, tr_local_se.max);
    tr_local_se.max = MAX(tr_local_se.min, tr_local_se.max);
#ifdef DEBUG
    std::cout << std::endl;
    std::cout << "\t\t\ttr_time: " << tr_time << std::endl;
    std::cout << "\t\t\ttr_local_se: " << tr_local_se << std::endl;
#endif

    tensor_constraint_range tr_final = tr_time.intersect_with(tr_local_se);
    if (tr_final.max < -1) {
#ifdef DEBUG
        std::cout << "\t\t\ttr_final: ∅" << std::endl;
#endif
        if (is_ofm(tt_id)) { // use r of tr_time
            enc.tensor_times[tt_id].end_time = tr_time.max;
        } else { // use l of tr_time
            enc.tensor_times[tt_id].start_time = tr_time.min;
        }
    } else {
#ifdef DEBUG
        std::cout << "\t\t\ttr_final: " << tr_final << std::endl;
#endif
        // randomly pick from tr_final
        if (is_ofm(tt_id)) { // use r of tr_time
            const auto old_end_time = enc.tensor_times[tt_id].end_time; 
            enc.tensor_times[tt_id].end_time = randint(tr_final.min, tr_final.max);
            buffer_time_saved = -(enc.tensor_times[tt_id].end_time - old_end_time) * (int64_t)g.tensor_id_to_size.at(tt_id);
#ifdef DEBUG
            std::cout << "\t\t\tEnd " << old_end_time << " -> " << enc.tensor_times[tt_id].end_time;
            if (old_end_time == enc.tensor_times[tt_id].end_time) 
                std::cout << " (No Change)";
            std::cout << std::endl;
            std::cout << "\t\t\tBuffer Time Saved: " << buffer_time_saved << std::endl;
#endif
        } else { // use l of tr_time
            const auto old_start_time = enc.tensor_times[tt_id].start_time;
            enc.tensor_times[tt_id].start_time = randint(tr_final.min, tr_final.max);
            buffer_time_saved = (enc.tensor_times[tt_id].start_time - old_start_time) * (int64_t)g.tensor_id_to_size.at(tt_id);
#ifdef DEBUG
            std::cout << "\t\t\tStart " << old_start_time << " -> " << enc.tensor_times[tt_id].start_time;
            if (old_start_time == enc.tensor_times[tt_id].start_time) 
                std::cout << " (No Change)";
            std::cout << std::endl;
            std::cout << "\t\t\tBuffer Time Saved: " << buffer_time_saved << std::endl;
#endif
        }
    }
    return true;
}

bool Stage2SA::change_tensor_order(const Graph& g)
{
    constexpr int MAX_TRIAL = 30;
    // randomly pick a tensor to change
    int tt_id;
    tensor_constraint_range tr;
    bool success = false;
    for (int i = 0; i < MAX_TRIAL; i++) {
        tt_id = rand_multi_prob(dram_tensor_id_to_size);
        if (g.tensor_info_set_types[tt_id] == Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs)
            continue;
        tr = get_tensor_order_range(g, tt_id);
        if (tr.min < tr.max) {
            success = true;
            break;
        }
    }
    if (!success) {
        return false;
    }
    cur_change.tensor_id = tt_id;
    const len_t& tt_order = tile_tensor_id_to_order.at(tt_id);
    cur_change.old_sch = tt_order;
    const len_t pos = randint_except(tr.min, tt_order, tr.max);
    cur_change.new_sch = pos;
    enc.tile_tensor_order.erase(enc.tile_tensor_order.begin() + tt_order);
    enc.tile_tensor_order.insert(enc.tile_tensor_order.begin() + pos, tt_id);
    buffer_time_saved = 0;
    // check enc valid
    return true;
}

bool Stage2SA::change_tensor_time(const Graph& g, const bool single_direction_change)
{
    constexpr int MAX_TRIAL = 30;
    // randomly pick a tensor to change
    int tt_id;
    tensor_constraint_range tr;
    bool success = false;
    for (int i = 0; i < MAX_TRIAL; i++) {
        tt_id = rand_multi_prob(dram_tensor_id_to_size);
        if (g.tensor_info_set_types[tt_id] == Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs)
            continue;
        tr = get_tensor_time_range(g, tt_id);
        if (single_direction_change) {
            if (g.tensor_info_set_types[tt_id] == Graph::TensorInfoSetType::OFM_TO_DRAM) {
                if (tr.min < g.tensor_times[tt_id].end_time) {
                    success = true;
                    break;
                }
            } else {
                if (g.tensor_times[tt_id].start_time < tr.max) {
                    success = true;
                    break;
                }
            }
        } else {
            if (tr.min < tr.max) {
                success = true;
                break;
            }
        }
    }
    if (!success) {
        return false;
    }
    cur_change.tensor_id = tt_id;
    const Graph::TensorTime& tt = g.tensor_times[tt_id];
    const Graph::TensorInfoSetType& ti_type = g.tensor_info_set_types[tt_id];
    
    if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
        cur_change.old_sch = enc.tensor_times[tt_id].end_time;
        if (single_direction_change)
            enc.tensor_times[tt_id].end_time = randint(tr.min, enc.tensor_times[tt_id].end_time - 1);
        else
            enc.tensor_times[tt_id].end_time = randint_except(tr.min, enc.tensor_times[tt_id].end_time, tr.max);
        cur_change.new_sch = enc.tensor_times[tt_id].end_time;
        buffer_time_saved = -(cur_change.new_sch - cur_change.old_sch) * (int64_t)g.tensor_id_to_size.at(tt_id);
    } else { // ONLY_IFMs or ONLY_WGTs
        cur_change.old_sch = enc.tensor_times[tt_id].start_time;
        if (single_direction_change)
            enc.tensor_times[tt_id].start_time = randint(enc.tensor_times[tt_id].start_time + 1, tr.max);
        else
            enc.tensor_times[tt_id].start_time = randint_except(tr.min, enc.tensor_times[tt_id].start_time, tr.max);
        cur_change.new_sch = enc.tensor_times[tt_id].start_time;
        buffer_time_saved = (cur_change.new_sch - cur_change.old_sch) * (int64_t)g.tensor_id_to_size.at(tt_id);
    }
    // std::cout << "\t\t\tBuffer Time Saved: " << buffer_time_saved << std::endl;
    // check enc valid
    return true;
}

Stage2SA::tensor_constraint_range Stage2SA::get_tensor_order_range(const Graph& g, const len_t& tensor_id)
{
    const len_t& tt_order = tile_tensor_id_to_order.at(tensor_id);
    const Graph::TensorInfoSetType& ti_type = g.tensor_info_set_types[tensor_id];
    
    assert(ti_type != Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
    tensor_constraint_range tr;
    // 1. Order Range
    // 1.0. init prev and next
    tr.min = 0;
    tr.max = num_dram_tiles - 1;
    // 1.1. apply RV2 & RP1 on order range
    for (auto prev_t_id : hasse_diag.reverse[tensor_id])
        CMAX(tr.min, tile_tensor_id_to_order.at(prev_t_id) + 1);
    for (auto next_t_id : hasse_diag.forward[tensor_id])
        CMIN(tr.max, tile_tensor_id_to_order.at(next_t_id) - 1);
    // 1.2. apply RV1 on order range
    for (int insert_pos = tt_order - 1; insert_pos >= tr.min; --insert_pos) {
        const len_t& insert_t_id = g.tile_tensor_order[insert_pos];
        if (g.tensor_times[tensor_id].start_time >= g.tensor_times[insert_t_id].end_time) {
            CMAX(tr.min, insert_pos + 1);
            break;
        }
    }
    for (int insert_pos = tt_order + 1; insert_pos <= tr.max; ++insert_pos) {
        const len_t& insert_t_id = g.tile_tensor_order[insert_pos];
        if (g.tensor_times[tensor_id].end_time <= g.tensor_times[insert_t_id].start_time) {
            CMIN(tr.max, insert_pos - 1);
            break;
        }
    }
    // @note: RV3 is only applied in s/e range
    // Check tensor order range
    assert(tr.min <= tt_order && tt_order <= tr.max);
    return tr;
}

Stage2SA::tensor_constraint_range Stage2SA::get_tensor_time_range(const Graph& g, const len_t& tensor_id)
{
    const len_t& tt_order = tile_tensor_id_to_order.at(tensor_id);
    const Graph::TensorInfoSetType& ti_type = g.tensor_info_set_types[tensor_id];
    
    assert(ti_type != Graph::TensorInfoSetType::OFM_WITH_LOCAL_IFMs);
    tensor_constraint_range tr;
    // 2. Start/End Range
    // 2.0. init min and max
    tr.min = -1;
    tr.max = total_tile_number;
    // 2.1. apply RV2 on s/e range
    // - across LG: A->B, OFM_A_last_tile's start <= IFM_B_first_tile's start
    if (ti_type == Graph::TensorInfoSetType::ONLY_IFMs) {
        if (auto it = across_lg_constraints.reverse.find(tensor_id); it != across_lg_constraints.reverse.end()) {
            for (auto& prev_tid : it->second)
                CMAX(tr.min, g.tensor_times[prev_tid].start_time);
        }
    }
    // 2.2. apply RP1 on s/e range
    if (ti_type == Graph::TensorInfoSetType::ONLY_IFMs) {
        if (auto it = IFM_tiles_constraints.reverse.find(tensor_id); it != IFM_tiles_constraints.reverse.end()) {
            for (auto& prev_tid : it->second)
                CMAX(tr.min, g.tensor_times[prev_tid].start_time);
        }
        if (auto it = IFM_tiles_constraints.forward.find(tensor_id); it != IFM_tiles_constraints.forward.end()) {
            for (auto& next_tid : it->second)
                CMIN(tr.max, g.tensor_times[next_tid].start_time);
        }
    } else if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
        if (auto it = OFM_tiles_constraints.reverse.find(tensor_id); it != OFM_tiles_constraints.reverse.end()) {
            for (auto& prev_tid : it->second)
                CMAX(tr.min, g.tensor_times[prev_tid].end_time);
        }
        if (auto it = OFM_tiles_constraints.forward.find(tensor_id); it != OFM_tiles_constraints.forward.end()) {
            for (auto& next_tid : it->second)
                CMIN(tr.max, g.tensor_times[next_tid].end_time);
        }
    }
    // 2.3. apply RV3 on s/e range
    if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
        CMAX(tr.min, dram_tensor_id_to_tile_pos[tensor_id]);
    } else { // IFM or WGT
        CMIN(tr.max, dram_tensor_id_to_tile_pos[tensor_id]);
    }
    // 2.4. apply RV1 on s/e range
    if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
        CMAX(tr.min, max_start[tt_order] + 1); // end min
    } else { // IFM or WGT
        CMIN(tr.max, min_end[tt_order] - 1); // start max
    }
    // Check tensor start/end range
    if (ti_type == Graph::TensorInfoSetType::OFM_TO_DRAM) {
        assert(tr.min <= g.tensor_times[tensor_id].end_time && g.tensor_times[tensor_id].end_time <= tr.max);
    } else { // IFM or WGT
        assert(tr.min <= g.tensor_times[tensor_id].start_time && g.tensor_times[tensor_id].start_time <= tr.max);
    }
    return tr;
}

std::array<OptimizationStatistics, 2> Stage2SA::solve(const len_t num_rounds, const bool opt_buffer_usage)
{
    std::chrono::steady_clock::time_point sa2_begin = std::chrono::steady_clock::now();
    bool using_best = false;
    buffer_time_saved = 0;
    CoreMapper::MapCost last_cost = best_cost;
    std::array<unsigned long long, 3> better_cnt = {0, 0, 0};
    std::array<unsigned long long, 3> acc_cnt = {0, 0, 0};
    std::array<unsigned long long, 3> val_cnt = {0, 0, 0};
    std::array<unsigned long long, 3> all_cnt = {0, 0, 0};
    Graph g = last_g;
    std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 2 SA Start! Seed: " << _seed_here << std::endl;
    for (int cur_round = 0; cur_round < num_rounds; cur_round++) {
        // report every 10% num_rounds
        if (cur_round % (num_rounds / 100) == 0) {
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 2 Progress: " << (double)cur_round / (double)num_rounds << std::endl;
        }
        // std::cout << "Round:" << cur_round << std::endl;
        if (cur_round >= 0.90 * num_rounds && !using_best) {
            using_best = true;
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 2 Switch to best solution." << std::endl;
            // SA variables
            g.init_stage_2(best_enc, true, true);
            last_g = g;
            enc = best_enc;
            last_enc = best_enc;
            last_cost = best_cost;
            // Internal Data Structures
            buffer_time_saved = 0;
            for (len_t i = 0; i < num_dram_tiles; ++i) {
                tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
            }
            max_start[0] = g.tensor_times[g.tile_tensor_order[0]].start_time;
            for (int i = 1; i < num_dram_tiles; ++i) {
                max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
            }
            min_end[num_dram_tiles - 1] = g.tensor_times[g.tile_tensor_order[num_dram_tiles - 1]].end_time;
            for (int i = num_dram_tiles - 2; i >= 0; --i) {
                min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
            }
        }
        // Randomly pick an operator to apply
#ifdef PERF_TIME_SPY
        std::chrono::steady_clock::time_point method_begin = std::chrono::steady_clock::now();
#endif
        int method = rand_multi_prob({0.1, 0.4, 0.5});
        bool success = false;
        if (!(enc == g.get_Encoding().second)) {
            std::cout << "Encoding not match with graph" << std::endl;
            std::cout << "Enc: \n" << enc;
            std::cout << "g.Encoding(): \n" << g.get_Encoding().first;
            assert(false);
        }
        all_cnt[method]++;
        // change enc & update graph
        switch (method) {
            case 0: success = change_tensor_order(g); break;
            case 1: success = change_tensor_time(g, false); break;
            case 2: success = change_tensor_order_and_time(g); break;
            default: break;
        }
#ifdef PERF_TIME_SPY
        std::chrono::steady_clock::time_point method_end = std::chrono::steady_clock::now();
        std::cout << "\t\tMethod Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(method_end - method_begin).count() << "ms" << std::endl;
        std::chrono::steady_clock::time_point init_stage2_begin, init_stage2_end;
        std::chrono::steady_clock::time_point init_cost_begin, init_cost_end;
        std::chrono::steady_clock::time_point get_cost_begin, get_cost_end;
        std::chrono::steady_clock::time_point update_begin, update_end;
#endif
        // assert(g.get_Encoding().first == cur_stage1->best_enc);
        // std::cout << "\t\tMethod: " << method << std::endl;
        if (success) {
#ifdef PERF_TIME_SPY
            init_stage2_begin = std::chrono::steady_clock::now();
#endif
            ErrorType err;
            if (method == 0) {
                err = g.init_stage_2(enc, true, false);
            } else if (method == 1) {
                err = g.init_stage_2(enc, false, true);
            } else if (method == 2) {
                err = g.init_stage_2(enc, true, true);
            } else { assert(false); }
            if (err != ErrorType::SUCCESS) {
                // std::cout << "\t\tInvalid Solution: " << err << std::endl;
                // don't update enc
                g = last_g;
                enc = last_enc;
#ifdef PERF_TIME_SPY
                init_stage2_end = std::chrono::steady_clock::now();
                std::cout << "\t\tInit Stage2 Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(init_stage2_end - init_stage2_begin).count() << "ms" << std::endl;
#endif
                continue;
            }
            // init_stage_2 success
#ifdef PERF_TIME_SPY
            init_stage2_end = std::chrono::steady_clock::now();
            std::cout << "\t\tInit stage2 Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(init_stage2_end - init_stage2_begin).count() << "ms" << std::endl;
            init_cost_begin = std::chrono::steady_clock::now();
#endif
            // g.initTileCosts();
#ifdef PERF_TIME_SPY
            init_cost_end = std::chrono::steady_clock::now();
            std::cout << "\t\tInit Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(init_cost_end - init_cost_begin).count() << "ms" << std::endl;
            get_cost_begin = std::chrono::steady_clock::now();
#endif
            CoreMapper::MapCost real_cost;
            // g.getIdealCost(ideal_cost, false);
            err = g.getRealCost(real_cost, false, false);
#ifdef PERF_TIME_SPY
            get_cost_end = std::chrono::steady_clock::now();
            std::cout << "\t\tGet Cost Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(get_cost_end - get_cost_begin).count() << "ms" << std::endl;
#endif
            // std::cout << "\t\tReal Cost: " << real_cost.cost() << ", Ideal Cost: " << s1_ideal_cost.cost() << std::endl;
#ifdef PERF_TIME_SPY
            update_begin = std::chrono::steady_clock::now();
#endif
            if (err == ErrorType::SUCCESS) {
                // std::cout << "Real Time: " << real_cost.time << ", Energy: " << real_cost.energy << ", Cost: " << real_cost.cost() << std::endl;
                val_cnt[method]++;
                if (best_cost.cost() > real_cost.cost()) {
                    better_cnt[method]++;
                    best_enc = enc;
                    best_cost = real_cost;
                    // std::cout << "\t\tBest Cost Update: " << real_cost.cost() << std::endl;
                }
                if (accept_by_progress(last_cost.cost(), real_cost.cost(), s1_ideal_cost.cost(), (double)cur_round/(double)num_rounds)) {
                    acc_cnt[method]++;
                    last_cost = real_cost;
                    last_g = g;
                    last_enc = enc;
                    // std::cout << "\t\tMax Buffer Usage: " << g.buffer.max_buffer_usage << std::endl;
                    // update internal data structures
                    assert(cur_change.old_sch != cur_change.new_sch);

                    if (method == 0) {
                        const int former_pos = MIN(cur_change.old_sch, cur_change.new_sch);
                        const int latter_pos = MAX(cur_change.old_sch, cur_change.new_sch);
                        // update tile_tensor_id_to_order
                        assert(num_dram_tiles == tile_tensor_id_to_order.size());
                        for (len_t i = former_pos; i <= latter_pos; ++i) {
                            // assert(my_tile_tensor_id_to_order.find(g.tile_tensor_order[i]) != my_tile_tensor_id_to_order.end());
                            tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
                        }
                        assert(num_dram_tiles == tile_tensor_id_to_order.size());
                        // update max start and min end in order form
                        {
                            int i = former_pos;
                            if (i == 0) {
                                max_start[i] = g.tensor_times[g.tile_tensor_order[i]].start_time;
                                i++;
                            }
                            for (; i <= latter_pos; ++i)
                                max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
                        }
                        {
                            int i = latter_pos;
                            if (i == num_dram_tiles - 1) {
                                min_end[i] = g.tensor_times[g.tile_tensor_order[i]].end_time;
                                i--;
                            }
                            for (; i >= former_pos; --i)
                                min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
                        }
                    } else if (method == 1) {
                        // update max start and min end in s/e form
                        const auto& cur_tensor_order = tile_tensor_id_to_order.at(cur_change.tensor_id);
                        if (g.tensor_info_set_types[cur_change.tensor_id] == Graph::TensorInfoSetType::OFM_TO_DRAM) {
                            // only end_time is changed, so only need to update min_end
                            const int& cur_tensor_min_end = min_end[cur_tensor_order];
                            if ((cur_change.old_sch > cur_tensor_min_end && cur_change.new_sch >= cur_tensor_min_end) ||
                                (cur_change.old_sch == cur_tensor_min_end && 
                                 cur_change.new_sch > cur_tensor_min_end && 
                                 cur_tensor_order != num_dram_tiles - 1 && 
                                 min_end[cur_tensor_order + 1] == cur_tensor_min_end)) {
                                // no need
                            } else {
                                // update min_end till we meet a min_end[k] that: min_end[k] <= MIN(old_sch, new_sch) 
                                const int smaller_min_end = MIN(cur_change.old_sch, cur_change.new_sch);
                                {
                                    int i = cur_tensor_order;
                                    if (i == num_dram_tiles - 1) {
                                        assert(cur_change.tensor_id == g.tile_tensor_order[num_dram_tiles - 1]);
                                        min_end[i] = g.tensor_times[cur_change.tensor_id].end_time;
                                        i--;
                                    }
                                    for (; i >= 0; --i) {
                                        if (min_end[i] < smaller_min_end) 
                                            break;
                                        min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
                                    }
                                }
                            }
                        } else {
                            // only start_time is changed, so only need to update max_start
                            const int& cur_tensor_max_start = max_start[cur_tensor_order];
                            if ((cur_change.old_sch < cur_tensor_max_start && cur_change.new_sch <= cur_tensor_max_start) ||
                                (cur_change.old_sch == cur_tensor_max_start && 
                                 cur_change.new_sch < cur_tensor_max_start && 
                                 cur_tensor_order != 0 && 
                                 max_start[cur_tensor_order - 1] == cur_tensor_max_start)) {
                                // no need
                            } else {
                                // update max_start till we meet a max_start[k] that: max_start[k] >= MAX(old_sch, new_sch)
                                const int larger_max_start = MAX(cur_change.old_sch, cur_change.new_sch);
                                {
                                    int i = cur_tensor_order;
                                    if (i == 0) {
                                        assert(cur_change.tensor_id == g.tile_tensor_order[0]);
                                        max_start[i] = g.tensor_times[cur_change.tensor_id].start_time;
                                        i++;
                                    }
                                    for (; i < num_dram_tiles; ++i) {
                                        if (max_start[i] > larger_max_start) 
                                            break;
                                        max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
                                    }
                                }
                            }
                        }
                    } else if (method == 2) {
                        const int former_pos = MIN(cur_change.old_sch, cur_change.new_sch);
                        const int latter_pos = MAX(cur_change.old_sch, cur_change.new_sch);
                        // update tile_tensor_id_to_order
                        for (len_t i = former_pos; i <= latter_pos; ++i) {
                            tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
                        }
                        assert(num_dram_tiles == tile_tensor_id_to_order.size());
                        // update max start and min end in order form
                        {
                            int i = former_pos;
                            if (i == 0) {
                                max_start[i] = g.tensor_times[g.tile_tensor_order[i]].start_time;
                                i++;
                            }
                            for (; i <= num_dram_tiles - 1; ++i)
                                max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
                        }
                        {
                            int i = latter_pos;
                            if (i == num_dram_tiles - 1) {
                                min_end[i] = g.tensor_times[g.tile_tensor_order[i]].end_time;
                                i--;
                            }
                            for (; i >= 0; --i)
                                min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
                        }
                    } else {
                        assert(false);
                    }
                    // std::cout << "\t\tNew Last Cost: " << real_cost.cost() << std::endl;
                    // update = true;
                } else {
                    // std::cout << "Acceptance: " << exp((last_cost.cost() - real_cost.cost()) / T) << std::endl;
                    // don't update enc
                    g = last_g;
                    enc = last_enc;
#ifdef PERF_TIME_SPY
                    update_end = std::chrono::steady_clock::now();
                    std::cout << "\t\tUpdate Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_begin).count() << "ms" << std::endl;
#endif
                    continue;
                }
            } else {
                // std::cout << "\t\tInvalid Solution: " << err << std::endl;
                // don't update enc
                g = last_g;
                enc = last_enc;
#ifdef PERF_TIME_SPY
                update_end = std::chrono::steady_clock::now();
                std::cout << "\t\tUpdate Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_begin).count() << "ms" << std::endl;
#endif
                continue;
            }
#ifdef PERF_TIME_SPY
            update_end = std::chrono::steady_clock::now();
            std::cout << "\t\tUpdate Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(update_end - update_begin).count() << "ms" << std::endl;
#endif
        } else {
            enc = last_enc;
            continue;
        }
    }
    std::chrono::steady_clock::time_point sa2_end = std::chrono::steady_clock::now();
    std::array<OptimizationStatistics, 2> stats;
    stats[0].search_time = std::chrono::duration_cast<std::chrono::seconds>(sa2_end - sa2_begin).count();
    stats[0].avg_buffer_usage = (double)best_sum_buffer_usage / (double)best_cost.time;
    stats[0].cache_stats.hit_cnt = Graph::num_tile_cost_cache_hit;
    stats[0].cache_stats.miss_cnt = Graph::num_tile_cost_cache_miss;
    stats[0].cache_stats.total_cnt = Graph::num_tile_cost_cache_total;
    for (int i = 0; i < 3; i++) {
        stats[0].sa_cnts[i].better_cnt = better_cnt[i];
        stats[0].sa_cnts[i].acc_cnt = acc_cnt[i];
        stats[0].sa_cnts[i].val_cnt = val_cnt[i];
        stats[0].sa_cnts[i].all_cnt = all_cnt[i];
    }
    for (int i = 3; i < 5; i++) {
        stats[0].sa_cnts[i].better_cnt = 0;
        stats[0].sa_cnts[i].acc_cnt = 0;
        stats[0].sa_cnts[i].val_cnt = 0;
        stats[0].sa_cnts[i].all_cnt = 0;
    }
    if (!opt_buffer_usage) {
        return stats;
    }
    std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 3 Start! Seed: " << _seed_here << std::endl;

    g.init_stage_2(best_enc, true, true);
    {
        // g.initTileCosts();
        CoreMapper::MapCost real_cost;
        // g.getIdealCost(ideal_cost, false);
        ErrorType err = g.getRealCost(real_cost, false);
        if (err != ErrorType::SUCCESS) {
            std::cout << "Stage 3 Init Failed: " << err << std::endl;
            assert(false);
        }
    }
    last_g = g;
    enc = best_enc;
    assert(enc == g.get_Encoding().second);
    last_enc = best_enc;
    last_cost = best_cost;
    using_best = false;
    buffer_time_saved = 0;
    // Internal Data Structures
    buffer_time_saved = 0;
    for (len_t i = 0; i < num_dram_tiles; ++i) {
        tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
    }
    max_start[0] = g.tensor_times[g.tile_tensor_order[0]].start_time;
    for (int i = 1; i < num_dram_tiles; ++i) {
        max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
    }
    min_end[num_dram_tiles - 1] = g.tensor_times[g.tile_tensor_order[num_dram_tiles - 1]].end_time;
    for (int i = num_dram_tiles - 2; i >= 0; --i) {
        min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
    }
    __uint128_t last_sum_buffer_usage = g.get_sum_buffer_usage();
    better_cnt = {0, 0, 0};
    acc_cnt = {0, 0, 0};
    val_cnt = {0, 0, 0};
    all_cnt = {0, 0, 0};
    len_t buffer_num_rounds = num_rounds / 10;
    for (len_t i = 0; i < num_dram_tiles; ++i) {
        dram_tensor_id_to_size[last_g.tile_tensor_order[i]] = 1;
    }
    best_sum_buffer_usage = last_sum_buffer_usage;
    for (int cur_round = 0; cur_round < buffer_num_rounds; cur_round++) {
        if (cur_round % (buffer_num_rounds / 100) == 0) {
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 3 Progress: " << (double)cur_round / (double)buffer_num_rounds << std::endl;
        }
        if (cur_round >= 0.90 * buffer_num_rounds && !using_best) {
            using_best = true;
            std::cout << "Buffer Limit Ratio: " << Graph::Buffer::STAGE_1_LIMIT_RATIO << ", Stage 3 Switch to best solution." << std::endl;
            // SA variables
            g.init_stage_2(best_enc, true, true);
            last_g = g;
            enc = best_enc;
            last_enc = best_enc;
            last_cost = best_cost;
            last_sum_buffer_usage = best_sum_buffer_usage;
            // Internal Data Structures
            buffer_time_saved = 0;
            for (len_t i = 0; i < num_dram_tiles; ++i) {
                tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
            }
            max_start[0] = g.tensor_times[g.tile_tensor_order[0]].start_time;
            for (int i = 1; i < num_dram_tiles; ++i) {
                max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
            }
            min_end[num_dram_tiles - 1] = g.tensor_times[g.tile_tensor_order[num_dram_tiles - 1]].end_time;
            for (int i = num_dram_tiles - 2; i >= 0; --i) {
                min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
            }
        }
        int method = rand_multi_prob({0.4, 0.6, 0});
        bool success = false;
        if (!(enc == g.get_Encoding().second)) {
            std::cout << "Encoding not match with graph" << std::endl;
            std::cout << "Enc: \n" << enc;
            std::cout << "g.Encoding(): \n" << g.get_Encoding().first;
            assert(false);
        }
        all_cnt[method]++;
        // change enc & update graph
        switch (method) {
            case 0: success = change_tensor_order(g); break;
            case 1: success = change_tensor_time(g, true); break;
            case 2: success = change_tensor_order_and_time(g); break;
            default: break;
        }
        if (success) {
            ErrorType err;
            if (method == 0) {
                err = g.init_stage_2(enc, true, false);
            } else if (method == 1) {
                err = g.init_stage_2(enc, false, true);
            } else if (method == 2) {
                err = g.init_stage_2(enc, true, true);
            } else { assert(false); }
            if (err != ErrorType::SUCCESS) {
                // std::cout << "\t\tInvalid Solution: " << err << std::endl;
                // don't update enc
                g = last_g;
                enc = last_enc;
                continue;
            }
            // init_stage_2 success
            // g.initTileCosts();
            CoreMapper::MapCost real_cost;
            // g.getIdealCost(ideal_cost, false);
            err = g.getRealCost(real_cost, false, false);
            
            if (err == ErrorType::SUCCESS) {
                // std::cout << "Real Time: " << real_cost.time << ", Energy: " << real_cost.energy << ", Cost: " << real_cost.cost() << ", Ideal Cost: " << s1_ideal_cost.cost() << std::endl;
                __uint128_t cur_sum_buffer_usage = g.get_sum_buffer_usage();
                val_cnt[method]++;
                if (best_cost.cost() > real_cost.cost() || (best_cost.cost() == real_cost.cost() && best_sum_buffer_usage > cur_sum_buffer_usage)) {
                    better_cnt[method]++;
                    best_enc = enc;
                    best_cost = real_cost;
                    best_sum_buffer_usage = cur_sum_buffer_usage;
                    // std::cout << "\t\tBest Cost Update: " << real_cost.cost() << ", with Average Buffer Usage = " << (double)cur_sum_buffer_usage / (double)real_cost.time / (double)1024 << "KB" << std::endl;
                }
                if (accept_by_progress_and_buffer_usage(last_cost.cost(), real_cost.cost(), last_sum_buffer_usage, cur_sum_buffer_usage, ideal_sum_buffer_usage, (double)cur_round/(double)buffer_num_rounds)) {
                    acc_cnt[method]++;
                    last_cost = real_cost;
                    last_sum_buffer_usage = cur_sum_buffer_usage;
                    last_g = g;
                    last_enc = enc;
                    // std::cout << "\t\tMax Buffer Usage: " << g.buffer.max_buffer_usage << std::endl;
                    // update internal data structures
                    assert(cur_change.old_sch != cur_change.new_sch);

                    if (method == 0) {
                        const int former_pos = MIN(cur_change.old_sch, cur_change.new_sch);
                        const int latter_pos = MAX(cur_change.old_sch, cur_change.new_sch);
                        // update tile_tensor_id_to_order
                        assert(num_dram_tiles == tile_tensor_id_to_order.size());
                        for (len_t i = former_pos; i <= latter_pos; ++i) {
                            tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
                        }
                        assert(num_dram_tiles == tile_tensor_id_to_order.size());
                        // update max start and min end in order form
                        {
                            int i = former_pos;
                            if (i == 0) {
                                max_start[i] = g.tensor_times[g.tile_tensor_order[i]].start_time;
                                i++;
                            }
                            for (; i <= latter_pos; ++i)
                                max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
                        }
                        {
                            int i = latter_pos;
                            if (i == num_dram_tiles - 1) {
                                min_end[i] = g.tensor_times[g.tile_tensor_order[i]].end_time;
                                i--;
                            }
                            for (; i >= former_pos; --i)
                                min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
                        }
                    } else if (method == 1) {
                        // update max start and min end in s/e form
                        const auto& cur_tensor_order = tile_tensor_id_to_order.at(cur_change.tensor_id);
                        if (g.tensor_info_set_types[cur_change.tensor_id] == Graph::TensorInfoSetType::OFM_TO_DRAM) {
                            // only end_time is changed, so only need to update min_end
                            const int& cur_tensor_min_end = min_end[cur_tensor_order];
                            if ((cur_change.old_sch > cur_tensor_min_end && cur_change.new_sch >= cur_tensor_min_end) ||
                                (cur_change.old_sch == cur_tensor_min_end && 
                                 cur_change.new_sch > cur_tensor_min_end && 
                                 cur_tensor_order != num_dram_tiles - 1 && 
                                 min_end[cur_tensor_order + 1] == cur_tensor_min_end)) {
                                // no need
                            } else {
                                // update min_end till we meet a min_end[k] that: min_end[k] <= MIN(old_sch, new_sch) 
                                const int smaller_min_end = MIN(cur_change.old_sch, cur_change.new_sch);
                                {
                                    int i = cur_tensor_order;
                                    if (i == num_dram_tiles - 1) {
                                        assert(cur_change.tensor_id == g.tile_tensor_order[num_dram_tiles - 1]);
                                        min_end[i] = g.tensor_times[cur_change.tensor_id].end_time;
                                        i--;
                                    }
                                    for (; i >= 0; --i) {
                                        if (min_end[i] < smaller_min_end) 
                                            break;
                                        min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
                                    }
                                }
                            }
                        } else {
                            // only start_time is changed, so only need to update max_start
                            const int& cur_tensor_max_start = max_start[cur_tensor_order];
                            if ((cur_change.old_sch < cur_tensor_max_start && cur_change.new_sch <= cur_tensor_max_start) ||
                                (cur_change.old_sch == cur_tensor_max_start && 
                                 cur_change.new_sch < cur_tensor_max_start && 
                                 cur_tensor_order != 0 && 
                                 max_start[cur_tensor_order - 1] == cur_tensor_max_start)) {
                                // no need
                            } else {
                                // update max_start till we meet a max_start[k] that: max_start[k] >= MAX(old_sch, new_sch)
                                const int larger_max_start = MAX(cur_change.old_sch, cur_change.new_sch);
                                {
                                    int i = cur_tensor_order;
                                    if (i == 0) {
                                        assert(cur_change.tensor_id == g.tile_tensor_order[0]);
                                        max_start[i] = g.tensor_times[cur_change.tensor_id].start_time;
                                        i++;
                                    }
                                    for (; i < num_dram_tiles; ++i) {
                                        if (max_start[i] > larger_max_start) 
                                            break;
                                        max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
                                    }
                                }
                            }
                        }
                    } else if (method == 2) {
                        const int former_pos = MIN(cur_change.old_sch, cur_change.new_sch);
                        const int latter_pos = MAX(cur_change.old_sch, cur_change.new_sch);
                        // update tile_tensor_id_to_order
                        for (len_t i = former_pos; i <= latter_pos; ++i) {
                            tile_tensor_id_to_order[g.tile_tensor_order[i]] = i;
                        }
                        assert(num_dram_tiles == tile_tensor_id_to_order.size());
                        // update max start and min end in order form
                        {
                            int i = former_pos;
                            if (i == 0) {
                                max_start[i] = g.tensor_times[g.tile_tensor_order[i]].start_time;
                                i++;
                            }
                            for (; i <= num_dram_tiles - 1; ++i)
                                max_start[i] = MAX(max_start[i - 1], g.tensor_times[g.tile_tensor_order[i]].start_time);
                        }
                        {
                            int i = latter_pos;
                            if (i == num_dram_tiles - 1) {
                                min_end[i] = g.tensor_times[g.tile_tensor_order[i]].end_time;
                                i--;
                            }
                            for (; i >= 0; --i)
                                min_end[i] = MIN(min_end[i + 1], g.tensor_times[g.tile_tensor_order[i]].end_time);
                        }
                    } else {
                        assert(false);
                    }
                    // std::cout << "\t\tNew Last Cost: " << real_cost.cost() << std::endl;
                    // update = true;
                } else {
                    // std::cout << "Acceptance: " << exp((last_cost.cost() - real_cost.cost()) / T) << std::endl;
                    // don't update enc
                    g = last_g;
                    enc = last_enc;
                    continue;
                }
            } else {
                // std::cout << "\t\tInvalid Solution: " << err << std::endl;
                // don't update enc
                g = last_g;
                enc = last_enc;
                continue;
            }
        } else {
            enc = last_enc;
            continue;
        }
    }
    std::chrono::steady_clock::time_point sa3_end = std::chrono::steady_clock::now();
    stats[1].search_time = std::chrono::duration_cast<std::chrono::seconds>(sa3_end - sa2_end).count();
    stats[1].avg_buffer_usage = (double)best_sum_buffer_usage / (double)best_cost.time;
    stats[1].cache_stats.hit_cnt = Graph::num_tile_cost_cache_hit;
    stats[1].cache_stats.miss_cnt = Graph::num_tile_cost_cache_miss;
    stats[1].cache_stats.total_cnt = Graph::num_tile_cost_cache_total;
    for (int i = 0; i < 3; i++) {
        stats[1].sa_cnts[i].better_cnt = better_cnt[i];
        stats[1].sa_cnts[i].acc_cnt = acc_cnt[i];
        stats[1].sa_cnts[i].val_cnt = val_cnt[i];
        stats[1].sa_cnts[i].all_cnt = all_cnt[i];
    }
    for (int i = 3; i < 5; i++) {
        stats[1].sa_cnts[i].better_cnt = 0;
        stats[1].sa_cnts[i].acc_cnt = 0;
        stats[1].sa_cnts[i].val_cnt = 0;
        stats[1].sa_cnts[i].all_cnt = 0;
    }
    return stats;
}

int merge(vector<int>& nums, vector<int>& temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left, inv_count = 0;
    while (i <= mid && j <= right) {
        if (nums[i] <= nums[j])
            temp[k++] = nums[i++];
        else
            temp[k++] = nums[j++], inv_count += (mid - i + 1);
    }
    while (i <= mid)
        temp[k++] = nums[i++];
    while (j <= right)
        temp[k++] = nums[j++];
    for (i = left; i <= right; i++)
        nums[i] = temp[i];
    return inv_count;
}

std::tuple<access_t, double, double> Stage2SA::get_encoding_changes(const Graph::Stage2Encoding& old_enc)
{
    vector<int> best_enc_order_to_original_enc_order(num_dram_tiles);
    for (int i = 0; i < num_dram_tiles; ++i) {
        best_enc_order_to_original_enc_order[tile_tensor_id_to_order[old_enc.tile_tensor_order[i]]] = i;
    }
    vector<int> temp(num_dram_tiles);
    access_t inv_count = 0;

    for (int curr_size = 1; curr_size < num_dram_tiles; curr_size *= 2) {
        for (int left_start = 0; left_start < num_dram_tiles - 1; left_start += 2 * curr_size) {
            int mid = MIN(left_start + curr_size - 1, num_dram_tiles - 1);
            int right_end = MIN(left_start + 2 * curr_size - 1, num_dram_tiles - 1);
            inv_count += merge(best_enc_order_to_original_enc_order, temp, left_start, mid, right_end);
        }
    }
    double inv_ratio = (double)inv_count / (double)(num_dram_tiles * (num_dram_tiles - 1) / 2);
    cycle_t total_tiles_changes = 0;
    __uint128_t total_dram_tensor_size = 0;
    for (int i = 0; i < num_dram_tiles; ++i) {
        const len_t& tt_id = old_enc.tile_tensor_order[i];
        total_dram_tensor_size += last_g.tensor_id_to_size.at(tt_id);
        if (last_g.tensor_info_set_types[tt_id] == Graph::TensorInfoSetType::OFM_TO_DRAM)
            total_tiles_changes += last_g.tensor_id_to_size.at(tt_id) * abs(best_enc.tensor_times[tt_id].end_time - old_enc.tensor_times[tt_id].end_time);
        else
            total_tiles_changes += last_g.tensor_id_to_size.at(tt_id) * abs(best_enc.tensor_times[tt_id].start_time - old_enc.tensor_times[tt_id].start_time);
    }
    double avg_tile_change = (double)total_tiles_changes / (double)total_dram_tensor_size;
    return {inv_count, inv_ratio, avg_tile_change};
}