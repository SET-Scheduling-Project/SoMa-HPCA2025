#include "bitset.h"
#include "cluster.h"
#include "debug.h"
#include "layerengine.h"
#include "noc.h"
#include "spatial_mapping/segmentation.h"
#include "utils.h"
#include "optimization.h"
#include <fstream>
#include <iostream>

#include "graph.h"
#include "ltreenode.h"
#include "nns/nns.h"

#include <cassert>
#include <chrono>
#include <math.h>
#include <random>
#include <string.h>
#include <time.h>

#include <map>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <omp.h>

// #include <mutex>
#include <thread>

#include "schnode.h"
#include "json/json.hpp"
using ordered_json = nlohmann::ordered_json;

inline void to_json(ordered_json& j, const CoreMapper::MapCost& cost) {
    j = ordered_json{ {"time", cost.time}, {"energy", cost.energy}, {"cost", cost.cost()} };
}
inline void from_json(const ordered_json& j, CoreMapper::MapCost& cost) {
    j.at("time").get_to(cost.time);
    j.at("energy").get_to(cost.energy);
}
inline void to_json(ordered_json& oj, const tensor_shape& nlohmann_json_t) {
    oj["bk"] = nlohmann_json_t.bk; 
    oj["c"] = nlohmann_json_t.c; 
    oj["h"] = nlohmann_json_t.h; 
    oj["w"] = nlohmann_json_t.w; 
    oj["size"] = nlohmann_json_t.size; 
} 
inline void from_json(const ordered_json& oj, tensor_shape& nlohmann_json_t) {
    oj.at("bk").get_to(nlohmann_json_t.bk); 
    oj.at("c").get_to(nlohmann_json_t.c); 
    oj.at("h").get_to(nlohmann_json_t.h); 
    oj.at("w").get_to(nlohmann_json_t.w); 
    oj.at("size").get_to(nlohmann_json_t.size);
}
inline void to_json(ordered_json& oj, const Graph::TensorInfo& nlohmann_json_t) {
    oj["layer_id"] = nlohmann_json_t.layer_id; 
    oj["tile_id"] = nlohmann_json_t.tile_id; 
    oj["tensor_type"] = nlohmann_json_t.tensor_type; 
    oj["source"] = nlohmann_json_t.source; 
}
inline void from_json(const ordered_json& oj, Graph::TensorInfo& nlohmann_json_t) {
    oj.at("layer_id").get_to(nlohmann_json_t.layer_id); 
    oj.at("tile_id").get_to(nlohmann_json_t.tile_id); 
    oj.at("tensor_type").get_to(nlohmann_json_t.tensor_type); 
    oj.at("source").get_to(nlohmann_json_t.source); 
}
inline void to_json(ordered_json& oj, const Graph::TensorTime& nlohmann_json_t) {
    oj.push_back(nlohmann_json_t.start_time);
    oj.push_back(nlohmann_json_t.end_time);
}
inline void from_json(const ordered_json& oj, Graph::TensorTime& nlohmann_json_t) {
    oj.at(0).get_to(nlohmann_json_t.start_time);
    oj.at(1).get_to(nlohmann_json_t.end_time);
}
inline void to_json(ordered_json& oj, const Bitset& nlohmann_json_t) {
    FOR_BITSET(i, nlohmann_json_t) {
        oj.push_back(i);
    }
}
inline void from_json(const ordered_json& oj, Bitset& nlohmann_json_t) {
    nlohmann_json_t.clear();
    for (auto& e : oj) {
        nlohmann_json_t.set(e);
    }
}
inline void to_json(ordered_json& oj, const Graph::Stage1Encoding& nlohmann_json_t) {
    oj["layer_order_to_id"] = nlohmann_json_t.layer_order_to_id; 
    oj["layer_group_partition"] = nlohmann_json_t.layer_group_partition; 
    oj["sub_layer_group_partition"] = nlohmann_json_t.sub_layer_group_partition; 
    oj["tile_numbers"] = nlohmann_json_t.tile_numbers;
}
inline void from_json(const ordered_json& oj, Graph::Stage1Encoding& nlohmann_json_t) {
    oj.at("layer_order_to_id").get_to(nlohmann_json_t.layer_order_to_id); 
    oj.at("layer_group_partition").get_to(nlohmann_json_t.layer_group_partition); 
    oj.at("sub_layer_group_partition").get_to(nlohmann_json_t.sub_layer_group_partition); 
    oj.at("tile_numbers").get_to(nlohmann_json_t.tile_numbers);
}
inline void to_json(ordered_json& oj, const Graph::Stage2Encoding& nlohmann_json_t) {
    oj["tensor_times"] = nlohmann_json_t.tensor_times;
    oj["tile_tensor_order"] = nlohmann_json_t.tile_tensor_order;
}
inline void from_json(const ordered_json& oj, Graph::Stage2Encoding& nlohmann_json_t) {
    oj.at("tensor_times").get_to(nlohmann_json_t.tensor_times);
    oj.at("tile_tensor_order").get_to(nlohmann_json_t.tile_tensor_order);
}
inline void to_json(ordered_json& oj, const Graph::DRAM_Tensor_Info& nlohmann_json_t) {
    oj["layer_id"] = nlohmann_json_t.layer_id; 
    oj["tile_id"] = nlohmann_json_t.tile_id; 
    oj["tile_tensor_type"] = nlohmann_json_t.tile_tensor_type; 
    oj["tensor_access_time"] = nlohmann_json_t.tensor_access_time; 
}
inline void from_json(const ordered_json& oj, Graph::DRAM_Tensor_Info& nlohmann_json_t) {
    oj.at("layer_id").get_to(nlohmann_json_t.layer_id); 
    oj.at("tile_id").get_to(nlohmann_json_t.tile_id); 
    oj.at("tile_tensor_type").get_to(nlohmann_json_t.tile_tensor_type); 
    oj.at("tensor_access_time").get_to(nlohmann_json_t.tensor_access_time); 
}
inline void to_json(ordered_json& oj, const Graph::COMP_Tile_Info& nlohmann_json_t) {
    oj["layer_id"] = nlohmann_json_t.layer_id; 
    oj["tile_id"] = nlohmann_json_t.tile_id; 
    oj["tile_comp_time"] = nlohmann_json_t.tile_comp_time; 
}
inline void from_json(const ordered_json& oj, Graph::COMP_Tile_Info& nlohmann_json_t) {
    oj.at("layer_id").get_to(nlohmann_json_t.layer_id); 
    oj.at("tile_id").get_to(nlohmann_json_t.tile_id); 
    oj.at("tile_comp_time").get_to(nlohmann_json_t.tile_comp_time); 
}
inline void to_json(ordered_json& oj, const Graph::IdealCostResults& nlohmann_json_t) {
    oj["ideal_comp"] = nlohmann_json_t.ideal_comp; 
    oj["ideal_dram"] = nlohmann_json_t.ideal_dram; 
    oj["comp_energy"] = nlohmann_json_t.comp_energy; 
    oj["ubuf_energy"] = nlohmann_json_t.ubuf_energy; 
    oj["buffer_energy"] = nlohmann_json_t.buffer_energy; 
    oj["noc_energy"] = nlohmann_json_t.noc_energy; 
    oj["mac_energy"] = nlohmann_json_t.mac_energy; 
    oj["dram_energy"] = nlohmann_json_t.dram_energy; 
}
inline void from_json(const ordered_json& oj, Graph::IdealCostResults& nlohmann_json_t) {
    oj.at("ideal_comp").get_to(nlohmann_json_t.ideal_comp); 
    oj.at("ideal_dram").get_to(nlohmann_json_t.ideal_dram); 
    oj.at("comp_energy").get_to(nlohmann_json_t.comp_energy); 
    oj.at("ubuf_energy").get_to(nlohmann_json_t.ubuf_energy); 
    oj.at("buffer_energy").get_to(nlohmann_json_t.buffer_energy); 
    oj.at("noc_energy").get_to(nlohmann_json_t.noc_energy); 
    oj.at("mac_energy").get_to(nlohmann_json_t.mac_energy); 
    oj.at("dram_energy").get_to(nlohmann_json_t.dram_energy); 
}

#define KB *1024
#define MB *1024 * 1024
#define DIVIDE_USING_DOUBLE(a, b) ((double)(a) / (double)(b))
#define TRAUNCATE(x) (x > 1 ? 1 : x)
#define DIVIDE_AND_TRUNCATE(a, b) TRAUNCATE(DIVIDE_USING_DOUBLE(a, b))
#define MEASURE_TIME(func)                                                                      \
    do {                                                                                        \
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();         \
        func;                                                                                   \
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();           \
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin); \
        std::cout << "Elapsed Time: " << elapsed_time.count() << " us" << std::endl;            \
    } while (0)
using namespace std;
std::vector<double> buffer(int width, vol_t size)
{
    vector<double> access; // per byte 0 is read, 1 is write
    access.resize(2);
    vol_t size_;
    if (width >= 512) {
        size_ = size / (width / 256);
        access = buffer(256, MIN(size_, 32 KB));
        access[0] *= pow(1.1, size / size_); // sqrt(size / size_);
        access[1] *= pow(1.1, size / size_); // sqrt(size / size_);
        return access;
    }
    if (width == 256) {
        if (size == 32 KB) {
            access[0] = 0.065678125;
            access[1] = 0.0641375;
        } else if (size == 16 KB) {
            access[0] = 0.06311875;
            access[1] = 0.05563125;
        } else if (size == 8 KB) {
            access[0] = 0.051290625;
            access[1] = 0.04560625;
        } else if (size == 4 KB) {
            access[0] = 0.049;
            access[1] = 0.064;
        } else {
            throw runtime_error("Cannot find buffer energy.");
        }
    }
    if (width == 128) {
        if (size == 32 KB) {
            access[0] = 0.106675;
            access[1] = 0.10539375;
        } else if (size == 16 KB) {
            access[0] = 0.06848125;
            access[1] = 0.06684375;
        } else if (size == 8 KB) {
            access[0] = 0.06553125;
            access[1] = 0.057975;
        } else if (size == 4 KB) {
            access[0] = 0.05336875;
            access[1] = 0.0476625;
        } else if (size == 2 KB) {
            access[0] = 0.06025625;
            access[1] = 0.06025625;
        } else {
            throw runtime_error("Cannot find buffer energy.");
        }
    }
    if (width == 64) {
        if (size == 32 KB) {
            access[0] = 0.194775;
            access[1] = 0.1923;
        } else if (size == 16 KB) {
            access[0] = 0.112;
            access[1] = 0.110675;
        } else if (size == 8 KB) {
            access[0] = 0.0740875;
            access[1] = 0.0722625;
        } else if (size == 4 KB || size == 4.5 KB) {
            access[0] = 0.0703625;
            access[1] = 0.0626625;
        } else if (size == 2 KB) {
            access[0] = 0.057525;
            access[1] = 0.051775;
        } else {
            throw runtime_error("Cannot find buffer energy.");
        }
    }
    if (width == 32) {
        if (size == 32 KB) {
            access[0] = 0.194775;
            access[1] = 0.1923;
        } else if (size == 16 KB) {
            access[0] = 0.112;
            access[1] = 0.110675;
        } else if (size == 8 KB) {
            access[0] = 0.11608;
            access[1] = 0.13898;
        } else if (size == 4 KB) {
            access[0] = 0.07422;
            access[1] = 0.09414;
        } else if (size == 2 KB) {
            access[0] = 0.06326;
            access[1] = 0.08034;
        } else if (size == 1 KB) {
            access[0] = 0.07952;
            access[1] = 0.0981;
        } else {
            throw runtime_error("Cannot find buffer energy.");
        }
    }
    return access;
}
uint16_t PE_ARRAY_X, PE_ARRAY_Y;
void handcraft_opt_tile_size(tensor_shape& ts, const int idx, const vol_t& vector_len, const vol_t& lane_len)
{
    uint8_t MIN_TILE_SIZE = 4;
    const fmap_shape ofm_shape = network->getNode(idx).layer().ofmap_shape();
    const fmap_shape ifm_shape = network->getNode(idx).layer().tot_ifmap_shape();
    double util = 0;
    double best_util = 0;
    double C1 = (double)ifm_shape.c/(double)vector_len;
    double K1 = (double)ofm_shape.c/(double)lane_len;
    // CK
    util = DIVIDE_AND_TRUNCATE(C1, PE_ARRAY_X) * DIVIDE_AND_TRUNCATE(K1, PE_ARRAY_Y);
    if (util >= 1) {
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, MIN_TILE_SIZE) };
        return;
    } else if (util > best_util) {
        best_util = util;
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, MIN_TILE_SIZE) };
    } else {}
    // CB 
    util = DIVIDE_AND_TRUNCATE(C1, PE_ARRAY_X) * DIVIDE_AND_TRUNCATE(SchNode::tot_batch, PE_ARRAY_Y) * TRAUNCATE(K1) ;
    if (util >= 1) {
        ts = { MIN(SchNode::tot_batch, PE_ARRAY_Y), ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, MIN_TILE_SIZE) };
        return;
    } else if (util > best_util) {
        best_util = util;
        ts = { MIN(SchNode::tot_batch, PE_ARRAY_Y), ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, MIN_TILE_SIZE) };
    } else {}
    // CW
    util = DIVIDE_AND_TRUNCATE(C1, PE_ARRAY_X) * DIVIDE_AND_TRUNCATE(ofm_shape.w, PE_ARRAY_Y) * TRAUNCATE(K1);
    if (util >= 1) {
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, PE_ARRAY_Y) };
        return;
    } else if (util > best_util) {
        best_util = util;
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, PE_ARRAY_Y) };
    } else {}
    // BK
    util = TRAUNCATE(C1) * DIVIDE_AND_TRUNCATE(SchNode::tot_batch, PE_ARRAY_X) * DIVIDE_AND_TRUNCATE(K1, PE_ARRAY_Y);
    if (util >= 1) {
        ts = { MIN(SchNode::tot_batch, PE_ARRAY_X), ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, MIN_TILE_SIZE) };
        return;
    } else if (util > best_util) {
        best_util = util;
        ts = { MIN(SchNode::tot_batch, PE_ARRAY_X), ofm_shape.c, MIN(ofm_shape.h, MIN_TILE_SIZE), MIN(ofm_shape.w, MIN_TILE_SIZE) };
    } else {}
    // HK
    util = TRAUNCATE(C1) * DIVIDE_AND_TRUNCATE(ofm_shape.h, PE_ARRAY_X) * DIVIDE_AND_TRUNCATE(K1, PE_ARRAY_Y);
    if (util >= 1) {
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, PE_ARRAY_X), MIN(ofm_shape.w, MIN_TILE_SIZE) };
        return;
    } else if (util > best_util) {
        best_util = util;
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, PE_ARRAY_X), MIN(ofm_shape.w, MIN_TILE_SIZE) };
    } else {}
    // HW
    util = TRAUNCATE(C1) * DIVIDE_AND_TRUNCATE(ofm_shape.h, PE_ARRAY_X) * DIVIDE_AND_TRUNCATE(ofm_shape.w, PE_ARRAY_Y);
    if (util >= 1) {
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, PE_ARRAY_X), MIN(ofm_shape.w, PE_ARRAY_Y) };
        return;
    } else if (util > best_util) {
        best_util = util;
        ts = { 1, ofm_shape.c, MIN(ofm_shape.h, PE_ARRAY_X), MIN(ofm_shape.w, PE_ARRAY_Y) };
    } else {}

    return;
}

inline bool write_to_json(const ordered_json& o_j, const std::string& json_filename)
{
    std::ofstream o_j_out;
    o_j_out.open(json_filename, std::ios::out);
    if (!o_j_out.is_open()) {
        std::cerr << "Error: failed to open file " << json_filename << std::endl;
        return false;
    }
    o_j_out << o_j.dump(4) << std::endl;
    o_j_out.close();
    if (!o_j_out.good()) {
        std::cerr << "Error: failed to write to file " << json_filename << std::endl;
        return false;
    }
    return true;
}

inline void record_error_type(const ErrorType& error_type, ordered_json& o_j)
{
    std::stringstream ss;
    ss << "Error: " << error_type;
    std::cout << ss.str() << std::endl;
    o_j["Error"] = ss.str();
    return ;
}

int main(int argc, char** argv)
{
    assert(argc == 13);
    {
        // std::random_device rd;
        _seed_here = static_cast<std::random_device::result_type>(atoll(argv[11])); // rd();
        set_gen_seed();
    }
    ordered_json o_j;
    uint8_t baseline_type = atoi(argv[2]);
    assert(baseline_type == 2 && "Only support baseline type 2");
    std::string result_dir = argv[12];
    std::ostringstream oss;
    for (int i = 1; i <= 10; ++i) {
        if (i > 1) {
            oss << "_";
        }
        oss << argv[i];
    }
    std::string filename = result_dir + "/log/" + oss.str() + ".log";
    std::string enc_filename = result_dir + "/enc/" + oss.str() + "_enc.log";
    std::string json_filename = result_dir + "/json/" + oss.str() + ".json";
#ifdef RESULT_LOG_FILE
    FILE* file = freopen(filename.c_str(), "w", stdout);
    if (!file) {
        std::cerr << "Error: failed to open file " << filename << std::endl;
        return 1;
    }
    std::cout << "Output File: " << filename << std::endl;
    std::cout << "Enc File: " << enc_filename << std::endl;
    std::cout << "Json File: " << json_filename << std::endl;
#endif
    std::cout.precision(4);
    // Get input parameters.
    int mm, nn, xx, yy, ss, bb, rr, ff, _DRAM_bw, _NoC_bw, _mac_dim, total_tops;
    // cin >> mm >> nn >> xx >> yy >> ss >> bb >> rr >> ff >> _DRAM_bw >> _NoC_bw >> _mac_dim >> _ul3 >> total_tops;
    mm = 0, nn = atoi(argv[1]), xx = yy = 1, ss = 1, bb = atoi(argv[6]), rr = 150, ff = 1;
    _NoC_bw = 64, _mac_dim = atol(argv[10]);
    PE_ARRAY_X = PE_ARRAY_Y = atoi(argv[8]);
    total_tops = 2 * xx * yy * _mac_dim * PE_ARRAY_X * PE_ARRAY_Y / 1024;
    _DRAM_bw = total_tops * atof(argv[7]);
    core_buffer_read_bandwidth = atoi(argv[9]);
    std::cout << "Computing Power = " << total_tops << "TOPS" << std::endl;
    std::cout << "DRAM Bandwidth = " << _DRAM_bw << "GB/s" << std::endl;
    std::cout << "L2 Bandwidth = " << core_buffer_read_bandwidth << "GB/s" << std::endl;
    o_j["Config"]["Hardware"]["TOPS"] = total_tops;
    o_j["Config"]["Hardware"]["PE_ARRAY_X"] = PE_ARRAY_X;
    o_j["Config"]["Hardware"]["PE_MAC_NUM"] = _mac_dim;
    o_j["Config"]["Hardware"]["DRAM_BW"] = _DRAM_bw;
    o_j["Config"]["Hardware"]["L2_BW"] = core_buffer_read_bandwidth;
    llm_seq_len = atoi(argv[3]);
    // Graph::setHasPrefetch((argv[9][0] == '1'));
    Graph::Buffer::set_max_buffer_size(atol(argv[5]) MB);
    // std::cout << Graph::Buffer::get_max_buffer_size()/1024/1024 << std::endl;
    // assert(total_tops == 2*xx*yy*_mac_dim*_mac_dim);
    /********************* INPUT *********************/
    NoC::DRAM_bw = _DRAM_bw;
    SchNode::DRAM_bw = _DRAM_bw;
    double DRAM_bw_each = _DRAM_bw / 1024 / 4;
    NoC::NoC_bw = _NoC_bw;
    NoC::soc = true;
    Cluster::xlen = xx;
    Cluster::ylen = yy;
    std::uint16_t mac_dim = _mac_dim;
    vol_t ul3_ = atol(argv[5]) MB;
    o_j["Config"]["Hardware"]["L2_BUFFER_SIZE"] = ul3_;

    std::uint16_t vector_len;
    std::uint16_t lane_len;
    if (mac_dim == 32) {
        vector_len = 4;
        lane_len = 8;
    } else if (mac_dim == 64) {
        vector_len = 8;
        lane_len = 8;
    } else if (mac_dim == 128) {
        vector_len = 8;
        lane_len = 16;
    } else if (mac_dim == 256) {
        vector_len = 16;
        lane_len = 16;
    } else if (mac_dim == 512) {
        vector_len = 16;
        lane_len = 32;
    } else if (mac_dim == 4096) {
        vector_len = 64;
        lane_len = 64;
    } else {
        throw runtime_error("MAC scale not supported·");
    }
    NoC::seperate_IO = true;
    double turnover_factor = 0.3 / 0.5;
    ofm_ubuf_vol = 10 KB;
    NoC::NoC_hop_cost = 0.8 * 8;
    NoC::DRAM_acc_cost = 10.5 * 8;
    energy_t LR_mac_cost = 0.0873; // IEEE FP16
    Core::numMac_t LR_mac_num = vector_len * lane_len * PE_ARRAY_X * PE_ARRAY_Y / 16;
    PolarCore::Buffer al1, wl1, ol1, al2, wl2, ol2, ul3;
    PolarCore::PESetting s(vector_len, lane_len, 0.018);
    PolarCore::Bus bus(PE_ARRAY_X, PE_ARRAY_Y, 0.018, 64);
    al1.Size = 8 * vector_len / 8 KB;
    ol1.Size = 2 * lane_len / 8 KB;
    wl1.Size = 4 * lane_len * vector_len / 64 KB;
    ol2.Size = 28 * vector_len * lane_len / 64 KB;
    wl2.Size = 0; // 256 KB;
    ul3.Size = ul3_; // 16 64-bit IO 64KB 1-port MBSRAM
    SchNode::ubuf = ul3.Size;

    double tops = 16 * vector_len * lane_len * Cluster::xlen * Cluster::ylen * 2;
    tops /= 1024;

    al2.Size = 0;
    al1.RCost = (buffer(vector_len * 8, al1.Size)[0] + 0.1 * lane_len / 8) * 8 * turnover_factor;
    al1.WCost = buffer(vector_len * 8, al1.Size)[1] * 8 * turnover_factor;
    wl1.RCost = buffer(vector_len * 8, wl1.Size)[0] * 8 * turnover_factor;
    wl1.WCost = buffer(vector_len * 8, wl1.Size)[1] * 8 * turnover_factor;
    ol1.RCost = buffer(lane_len * 16, ol1.Size)[0] * 8 * turnover_factor;
    ol1.WCost = buffer(lane_len * 16, ol1.Size)[1] * 8 * turnover_factor;
    ol2.RCost = 0.07648125 * 8 * pow(1.1, ul3_ / (2048 KB)) * turnover_factor; // ol2 is a small part of ul3
    ol2.WCost = 0.0989875 * 8 * pow(1.1, ul3_ / (2048 KB)) * turnover_factor;
    ul3.RCost = 0.217125 * 8 * pow(1.1, ul3_ / (2048 KB)) * turnover_factor;
    ul3.WCost = 0.234025 * 8 * pow(1.1, ul3_ / (2048 KB)) * turnover_factor;
    al2.RCost = al2.WCost = 0;
    wl2.RCost = wl2.WCost = 0;

    PolarCore core(s, bus, al1, wl1, ol1, al2, wl2, ol2, ul3, LR_mac_num, LR_mac_cost);
    PolarMapper mapper(core);

    EyerissCore::Buffer _al1, _wl1, pl1, ul2;
    EyerissCore::PESetting s2(mac_dim, mac_dim, 0.018);
    EyerissCore::Bus ibus(0.018, 64);
    EyerissCore::Bus wbus(0.018, 64);
    EyerissCore::Bus pbus(0.018, 64); // ifmap RC, weight RCK, psum RK

    _al1.Size = 32;
    pl1.Size = 1;
    _wl1.Size = 128;
    ul2.Size = 1024 KB;

    _al1.RCost = 0.0509 * 8 * turnover_factor; // 8bit IO single port
    _al1.WCost = 0.0506 * 8 * turnover_factor; // 0.045;
    _wl1.RCost = 0.0545 * 8 * turnover_factor; // Using 2 banks of 64
    _wl1.WCost = 0.054 * 8 * turnover_factor; // 0.090;
    pl1.RCost = pl1.WCost = 0.0 * turnover_factor;
    ul2.RCost = 0.1317125 * 8 * turnover_factor;
    ul2.WCost = 0.234025 * 8 * turnover_factor;

    EyerissCore core2(s2, ibus, wbus, pbus, _al1, _wl1, pl1, ul2, LR_mac_num, LR_mac_cost);
    EyerissMapper mapper2(core2);

    CoreMapper* cMapper;
    if (mm == 0) {
        cMapper = &mapper;
    } else {
        cMapper = &mapper2;
    }
    // TOPS

    // 2 GB/TOPS
    if (NoC::DRAM_bw == 0) {
        NoC::DRAM_bw = 0.75 * (tops / 4);
        SchNode::DRAM_bw = 0.75 * (tops / 4);
        NoC::NoC_bw = 4; // NoC::DRAM_bw / 4;
    }
    NoC::interleave = false;

    StdLayerEngine engine(cMapper);
    Graph::layerMapper = &engine;

    len_t tot_batch = bb;
    SchNode::tot_batch = tot_batch;

    Cluster::stride = ss;
    Cluster::min_util = 0.75;

    switch (ff) {
    case 1:
        // This is the default cost_func.
        cost_func = [](energy_t e, cycle_t t) { return e * t; };
        break;
    case 0:
        cost_func = [](energy_t, cycle_t t) { return t; };
        break;
    case -1:
        cost_func = [](energy_t e, cycle_t) { return e; };
        break;
    default:
        if (ff > 0) {
            cost_func = [=](energy_t e, cycle_t t) -> cost_t { return pow(e, ff) * t; };
        } else {
            cost_func = [=](energy_t e, cycle_t t) -> cost_t { return e * pow(t, -ff); };
        }
    }
    std::string net_name_all_layers;
    const Network* network_all_layers;
    switch (nn) {
    case 0:
        network_all_layers = &darknet19;
        net_name_all_layers = "darknet19";
        break;
    case 1:
        network_all_layers = &vgg19;
        net_name_all_layers = "vgg";
        break;
    case 2:
        network_all_layers = &resnet50;
        net_name_all_layers = "resnet";
        break;
    case 3:
        network_all_layers = &googlenet;
        net_name_all_layers = "goog";
        break;
    case 4:
        network_all_layers = &resnet101;
        net_name_all_layers = "resnet101";
        break;
    case 5:
        network_all_layers = &densenet;
        net_name_all_layers = "densenet";
        break;
    case 6:
        network_all_layers = &inception_resnet_v1;
        net_name_all_layers = "ires";
        break;
    case 7:
        network_all_layers = &gnmt;
        net_name_all_layers = "gnmt";
        break;
    case 8:
        network_all_layers = &lstm;
        net_name_all_layers = "lstm";
        break;
    case 9:
        network_all_layers = &zfnet;
        net_name_all_layers = "zfnet";
        break;
    case 10:
        network_all_layers = &transformer;
        net_name_all_layers = "trans";
        break;
    case 11:
        network_all_layers = &transformer_cell;
        net_name_all_layers = "trans_cell";
        break;
    case 12:
        network_all_layers = &PNASNet;
        net_name_all_layers = "pnas";
        break;
    case 13:
        network_all_layers = &resnext50;
        net_name_all_layers = "resnext50";
        break;
    case 14:
        network_all_layers = &resnet152;
        net_name_all_layers = "resnet152";
        break;
    case 15:
        network_all_layers = &transformer_big_cell;
        net_name_all_layers = "transformer_big_cell";
        break;
    case 16:
        network_all_layers = &retinanet_resnet50;
        net_name_all_layers = "retinanet_resnet50";
        break;
    case 17:
        network_all_layers = &unet;
        net_name_all_layers = "unet";
        break;
    case 18:
        network_all_layers = &randwire_small;
        net_name_all_layers = "randwire_small";
        break;
    case 19:
        network_all_layers = &randwire_large;
        net_name_all_layers = "randwire_large";
        break;
    // LLMs
    case 101:
        network_all_layers = get_GPT_J_6B_decode();
        net_name_all_layers = "GPT_J_6B_decode";
        break;
    case 102:
        network_all_layers = get_GPT_J_6B_prefill();
        net_name_all_layers = "GPT_J_6B_prefill";
        break;
    case 103:
        network_all_layers = get_LLaMa_2_70B_decode();
        net_name_all_layers = "LLaMa_2_70B_decode";
        break;
    case 104:
        network_all_layers = get_LLaMa_2_70B_prefill();
        net_name_all_layers = "LLaMa_2_70B_prefill";
        break;
    case 105:
        network_all_layers = get_BERT_base();
        net_name_all_layers = "BERT_base";
        break;
    case 106:
        network_all_layers = get_BERT_large();
        net_name_all_layers = "BERT_large";
        break;
    case 107:
        network_all_layers = get_GPT_2_small_decode();
        net_name_all_layers = "GPT_2_small_decode";
        break;
    case 108:
        network_all_layers = get_GPT_2_small_prefill();
        net_name_all_layers = "GPT_2_small_prefill";
        break;
    case 109:
        network_all_layers = get_GPT_2_XL_decode();
        net_name_all_layers = "GPT_2_XL_decode";
        break;
    case 110:
        network_all_layers = get_GPT_2_XL_prefill();
        net_name_all_layers = "GPT_2_XL_prefill";
        break;
    // TODO: Support more DNNs.
    default:
        throw runtime_error("Model not supported.");
    }
    
    const lid_t num_all_layers = network_all_layers->len();
    network_all_layers->set_utime(*cMapper);
    const int num_segments = atoi(argv[4]);
    const int segment_len = num_all_layers / num_segments;
    const int segment_res = num_all_layers % num_segments;

    o_j["Config"]["Network"]["name"] = net_name_all_layers;
    o_j["Config"]["Network"]["seq_len"] = llm_seq_len;
    o_j["Config"]["Network"]["batch_size"] = tot_batch;
    o_j["Config"]["Network"]["num_segments"] = num_segments;

    CoreMapper::MapCost total_cost;
    total_cost.time = 0;
    total_cost.energy = 0;
    std::vector<CoreMapper::MapCost> seg_costs;
    seg_costs.reserve(num_segments);
    for (lid_t par_i = 0; par_i < num_segments; ++par_i) {
        Network sub_network;
        {
            const lid_t layer_start = par_i * segment_len + MIN(par_i, segment_res);
            const lid_t layer_end = layer_start + segment_len - 1 + (par_i < segment_res ? 1 : 0);
            Network::layer_set layer_partition(layer_end - layer_start + 1);
            for (lid_t i = 0; i < layer_end - layer_start + 1; ++i) {
                layer_partition[i] = i + layer_start;
            }
            gen_sub_network(sub_network, network_all_layers, layer_partition);
        }
        network = &sub_network;
        std::cout << "Network: " << net_name_all_layers + " \tPart " + to_string(par_i) << std::endl;
        if (nn == 10 || nn == 11 || nn == 15) {
            std::cout << "seq_len = 1024" << std::endl;
        } else if (nn >= 16 && nn <= 21) {
            std::cout << "seq_len = " << llm_seq_len << std::endl;
        } else {
            ;
        }
        network->set_utime(*cMapper);
        cycle_t total_utime = 0;
        cycle_t total_dream_time = 0;
        { // caculate total utime & dream time
            total_utime += network->getNode(0).get_utime() * SchNode::tot_batch;
            std::cout << "Layer Utime: " << total_utime;
            for (lid_t i = 1; i < network->len(); ++i) {
                utime_t layer_utime = network->getNode(i).get_utime() * SchNode::tot_batch;
                total_utime += layer_utime;
                std::cout << ", " << layer_utime;
            }
            std::cout << std::endl;
            std::cout << "Total Utime: " << total_utime << std::endl;
            
            std::cout << "Layer Dream time: ";
            for (lid_t i = 0; i < network->len(); ++i) {
                const Layer& l = network->getNode(i).layer();
                utime_t t = l.get_num_op(SchNode::tot_batch);
                if (REF_IS_INSTANCE(l, ConvLayer)) {
                    t /= core.mac_num;
                } else {
                    t /= core.LR_mac_num;
                }
                std::cout << t;
                if (i != network->len() - 1)
                    std::cout << ", ";
                total_dream_time += t;
            }
            std::cout << std::endl;
            std::cout << "Total Dream Time: " << total_dream_time << std::endl;
        }
        Graph::get_special_layer_ids();
        const lid_t num_layer = network->len();
        Graph::Stage1Encoding enc;
        enc.layer_order_to_id.resize(num_layer);
        enc.layer_group_partition.clear();
        enc.sub_layer_group_partition.clear();
        Graph::Stage1Encoding_tile_sizes enc_tz;
        enc_tz.layer_order_to_id.resize(num_layer);
        enc_tz.layer_group_partition.clear();
        enc_tz.sub_layer_group_partition.clear();
        enc_tz.tile_sizes.resize(num_layer);
        for (auto i = 0; i < num_layer; i++) {
            enc.layer_order_to_id[i] = i;
            enc_tz.layer_order_to_id[i] = i;
            const fmap_shape ofm_shape = network->getNode(i).layer().ofmap_shape();
            const fmap_shape ifm_shape = network->getNode(i).layer().tot_ifmap_shape();
            // enc.tile_sizes[i] = {SchNode::tot_batch, ofm_shape.c, ofm_shape.h, ofm_shape.w};
            // enc.tile_sizes[i] = {SchNode::tot_batch, ofm_shape.c, (ofm_shape.h+1)/2, (ofm_shape.w+1)/2};
            handcraft_opt_tile_size(enc_tz.tile_sizes[i], i, vector_len, lane_len);
            enc.layer_group_partition.set(i);     // 1111111111111111   LG 100011
            enc_tz.layer_group_partition.set(i);  // 1111111111111111  SLG 001000
        }
        Bitset slgs = enc.layer_group_partition|enc.sub_layer_group_partition;
        enc.tile_numbers.resize(slgs.count(), 32);

        std::cout << "Enc OK" << std::endl;

        double stage1_buffer_ratio_100;
        CoreMapper::MapCost best_seg_stage1_real_cost, best_seg_stage2_real_cost, best_seg_ideal_cost;
        Graph best_g_stage2, best_g_stage1;
        Graph::IdealCostResults best_ideal_cost_breakdown;
        size_t stage1_rounds = 0, stage2_rounds = 0;
        int64_t stage1_time = 0, stage2_time = 0;
        
        // Stage 1 with 100% buffer limit
        {
            Graph::Buffer::set_stage_1_limit_ratio(1);
            std::cout << "Stage 1 With 100% Buffer Limit" << std::endl;
            Stage1SA sa1(network, enc, baseline_type);
            if (!sa1.sa_init(enc_tz, true, false)) {
                std::cout << "SA Init Failed" << std::endl;
                o_j["Status"] = "First SA Init Failed";
                write_to_json(o_j, json_filename);
                return 1;
            }
            stage1_rounds = 100 * network->len() * sqrt(SchNode::tot_batch);
            OptimizationStatistics stage1_stats = sa1.solve(stage1_rounds);
            std::cout << "Stage 1 " << stage1_stats;
            stage1_time = stage1_stats.search_time;
            
        #ifdef RESULT_LOG_FILE
            ofstream enc_out;
            enc_out.open(enc_filename, ios::out | ios::app);
            enc_out << "1" << std::endl;
            enc_out << sa1.best_enc;
            enc_out.close();
        #endif
            Graph g;
            CoreMapper::MapCost real_cost, ideal_cost;
            std::cout << "Stage 1 Final Encoding: " << std::endl;
            std::cout << sa1.best_enc;
            g.init_stage_1(sa1.best_enc);
            g.initTileCosts();
            best_ideal_cost_breakdown = g.getIdealCost(ideal_cost, true);
            if (auto err_type = g.getRealCost(real_cost, true); err_type != ErrorType::SUCCESS) {
                record_error_type(err_type, o_j);
                write_to_json(o_j, json_filename);
                return 1;
            }
            // std::cout << "Stage 1 Final Intensity: " << std::endl;
            // g.print_intensity();
            // std::cout << "Stage 1 Final TG Info: " << std::endl;
            // g.print_tile_group_info();
            std::cout << "Stage 1 Final Results: " << std::endl;
            std::cout << "Real Time: " << real_cost.time << ", Energy: " << real_cost.energy << ", Cost: " << real_cost.cost() << std::endl;
            std::cout << "Ideal Time: " << ideal_cost.time << ", Energy: " << ideal_cost.energy << ", Cost: " << ideal_cost.cost() << std::endl;
            std::cout << "Architecture Util: " << (double)total_utime * 100 /(double)real_cost.time << "%" << std::endl;
            std::cout << "Real Util: " << (double)total_dream_time * 100 / (double)real_cost.time << "%" << std::endl;
            print_avg_buffer_usage(g, real_cost);
            std::cout << "Max Buffer Usage: " << g.buffer.max_buffer_usage << std::endl;
            const auto dram_tile_num = g.tile_tensor_order.size();
            std::cout << "DRAM Tile Num: " << dram_tile_num << std::endl;
            // g.print_graph_result();
            best_seg_stage1_real_cost = real_cost;
            best_seg_ideal_cost = ideal_cost;
            best_g_stage1 = g;

            stage1_buffer_ratio_100 = (double)g.buffer.max_buffer_usage / (double)Graph::Buffer::get_max_buffer_size();
            double prob_points;
            // if (dram_tile_num <= 1e2)
            //     prob_points = 1250;
            // else if (dram_tile_num > 1e2 && dram_tile_num <= 1e3)
            //     prob_points = 400;
            // else if (dram_tile_num > 1e3 && dram_tile_num <= 5e3)
            //     prob_points = 200;
            // else {
            //     std::cout << "Too many dram tiles!!!" << std::endl;
            //     std::cout << "Now Stage2 is meaningless, quiting." << std::endl;
            //     return 0;
            // }
            if (dram_tile_num <= 3e3)
                prob_points = 1000;
            else {
                std::cout << "Too many dram tiles!!!" << std::endl;
                std::cout << "Stage2 will only do 3e6 rounds" << std::endl;
                prob_points = 3e6/(double)dram_tile_num;
            }
            // Stage2 try 1 using 100% buffer limit stage1 result
            std::cout << "Stage 2 after Stage 1 with 100% buffer limit" << std::endl;
            std::cout << "Stage 2 SA Iteration Num: " << (len_t)(prob_points * dram_tile_num) << std::endl;
            Stage2SA sa2(network, sa1.best_cost, g);
            stage2_rounds = prob_points * dram_tile_num;
            std::array<OptimizationStatistics, 2> stage2_stats = sa2.solve((len_t)stage2_rounds, true);
            std::cout << "Stage 2 " << stage2_stats[0];
            std::cout << "Stage 3 " << stage2_stats[1];
            stage2_time = stage2_stats[0].search_time + stage2_stats[1].search_time;
            
            // std::cout << "Stage 2 Final Encoding: " << std::endl;
            // std::cout << sa2.best_enc;
            if (auto err_type = g.init_stage_2(sa2.best_enc, true, true); err_type != ErrorType::SUCCESS) {
                record_error_type(err_type, o_j);
                write_to_json(o_j, json_filename);
                return 1;
            }
            g.initTileCosts();
            g.getIdealCost(ideal_cost, false);
            assert(sa1.best_enc == g.get_Encoding().first);
            assert(sa2.best_enc == g.get_Encoding().second);
            if (auto err_type = g.getRealCost(real_cost, true); err_type != ErrorType::SUCCESS) {
                record_error_type(err_type, o_j);
                write_to_json(o_j, json_filename);
                return 1;
            }
            std::cout << "Stage 2 Final Results @" << (len_t)(prob_points * dram_tile_num) << ": " << std::endl;
            std::cout << "Real Time: " << real_cost.time << ", Energy: " << real_cost.energy << ", Cost: " << real_cost.cost() << std::endl;
            std::cout << "Ideal Time: " << ideal_cost.time << ", Energy: " << ideal_cost.energy << ", Cost: " << ideal_cost.cost() << std::endl;
            std::cout << "Architecture Util: " << (double)total_utime * 100 /(double)real_cost.time << "%" << std::endl;
            std::cout << "Real Util: " << (double)total_dream_time * 100 / (double)real_cost.time << "%" << std::endl;
            print_avg_buffer_usage(g, real_cost);
            std::cout << "Max Buffer Usage: " << g.buffer.max_buffer_usage << std::endl;
            // g.print_graph_result();
            best_seg_stage2_real_cost = real_cost;
            best_g_stage2 = g;
        }
        if (par_i == 0) {
            o_j["Config"]["Searcher"]["baseline_type"] = baseline_type;
            o_j["Config"]["Searcher"]["seed"] = _seed_here;
            o_j["Config"]["Searcher"]["stage1_rounds"] = stage1_rounds;
            o_j["Config"]["Searcher"]["stage2_rounds"] = stage2_rounds;
        }
        // Stage 2 try 2 using x-10% buffer limit stage1 result, 
        // x = actual max buffer usage ratio of that 100% buffer limit stage1 result
        struct iter_results {
            double buffer_limit_ratio;
            CoreMapper::MapCost stage1_real_cost, stage2_real_cost, ideal_cost;
            Graph g_stage1, g_stage2;
            int64_t stage1_time, stage2_time;
            bool sa_success;
            Graph::Stage1Encoding stage1_enc;
            Graph::Stage1Encoding stage2_enc;
            len_t stage1_rounds, stage2_rounds;
            OptimizationStatistics stage1_stats;
            std::array<OptimizationStatistics, 2> stage2_stats;
            Graph::IdealCostResults ideal_cost_breakdown;
        };

        double best_buffer_limit_ratio = stage1_buffer_ratio_100;
        const int _NUM_ITERATIONS = 5;
        iter_results thread_results[_NUM_ITERATIONS];
        #pragma omp parallel for 
        for (int iterations = 0; iterations < _NUM_ITERATIONS; ++iterations) {
            iter_results& thread_res = thread_results[iterations];
            double buffer_limit_ratio = stage1_buffer_ratio_100 * pow(0.9, iterations); 
            assert(buffer_limit_ratio > 0);
            thread_res.buffer_limit_ratio = buffer_limit_ratio;
            Graph::Buffer::set_stage_1_limit_ratio(buffer_limit_ratio);

            Stage1SA sa1(network, enc, baseline_type);
            thread_res.sa_success = sa1.sa_init(enc_tz, true, false);
            if (!thread_res.sa_success) {
                continue;
            }
            thread_res.stage1_rounds = 100 * network->len() * sqrt(SchNode::tot_batch);
            thread_res.stage1_stats = sa1.solve(100 * network->len() * sqrt(SchNode::tot_batch));
            thread_res.stage1_time = thread_res.stage1_stats.search_time;
            thread_res.stage1_enc = sa1.best_enc;
            Graph g;
            g.init_stage_1(sa1.best_enc);
            g.initTileCosts();
            thread_res.ideal_cost_breakdown = g.getIdealCost(thread_res.ideal_cost, false);
            if (auto err_type = g.getRealCost(thread_res.stage1_real_cost, true); err_type != ErrorType::SUCCESS) {
                record_error_type(err_type, o_j);
                write_to_json(o_j, json_filename);
                thread_res.sa_success = false;
                continue;
            }
            thread_res.g_stage1 = g;
            const auto dram_tile_num = g.tile_tensor_order.size();
            double prob_points;
            if (dram_tile_num <= 5e3)
                prob_points = 1000;
            else {
                std::cout << "Too many dram tiles!!!" << std::endl;
                std::cout << "Stage2 will only do 5e6 rounds" << std::endl;
                prob_points = 5e6/(double)dram_tile_num;
            }
            thread_res.stage2_rounds = (len_t)(prob_points * dram_tile_num);
            Stage2SA sa2(network, sa1.best_cost, g);
            thread_res.stage2_stats = sa2.solve((len_t)(prob_points * dram_tile_num), true);
            thread_res.stage2_time = thread_res.stage2_stats[0].search_time + thread_res.stage2_stats[1].search_time;
            if (auto err_type = g.init_stage_2(sa2.best_enc, true, true); err_type != ErrorType::SUCCESS) {
                record_error_type(err_type, o_j);
                write_to_json(o_j, json_filename);
                thread_res.sa_success = false;
                continue;
            }
            g.initTileCosts();
            g.getIdealCost(thread_res.ideal_cost, false);
            assert(sa1.best_enc == g.get_Encoding().first);
            assert(sa2.best_enc == g.get_Encoding().second);
            if (auto err_type = g.getRealCost(thread_res.stage2_real_cost, true); err_type != ErrorType::SUCCESS) {
                record_error_type(err_type, o_j);
                write_to_json(o_j, json_filename);
                thread_res.sa_success = false;
                continue;
            }
            thread_res.g_stage2 = g;
            thread_res.stage1_enc = sa1.best_enc;
        }
        
        for (int iterations = 0; iterations < _NUM_ITERATIONS; ++iterations) {
            const auto& thread_res = thread_results[iterations];
            std::cout << "Iteration: " << iterations << " Buffer Limit Ratio: " << thread_res.buffer_limit_ratio << std::endl;
            std::cout << "Stage 1 With " << thread_res.buffer_limit_ratio * 100 << "% Buffer Limit" << std::endl;
            if (!thread_res.sa_success) {
                std::cout << "SA Init Failed or Final Graph Invalid" << std::endl;
                continue;
            }
            std::cout << "Stage 1 SA Iteration Num: " << thread_res.stage1_rounds << std::endl;
            std::cout << "Stage 1 " << thread_res.stage1_stats;
#ifdef RESULT_LOG_FILE
            ofstream enc_out;
            enc_out.open(enc_filename, ios::out | ios::app);
            enc_out << thread_res.buffer_limit_ratio << std::endl;
            enc_out << thread_res.stage1_enc;
            enc_out.close();
#endif
            std::cout << "Stage 1 Final Encoding: " << std::endl;
            std::cout << thread_res.stage1_enc;
            std::cout << thread_res.ideal_cost_breakdown << std::endl;
            std::cout << "Stage 1 Final Results: " << std::endl;
            std::cout << "Real Time: " << thread_res.stage1_real_cost.time << ", Energy: " << thread_res.stage1_real_cost.energy << ", Cost: " << thread_res.stage1_real_cost.cost() << std::endl;
            std::cout << "Ideal Time: " << thread_res.ideal_cost.time << ", Energy: " << thread_res.ideal_cost.energy << ", Cost: " << thread_res.ideal_cost.cost() << std::endl;
            std::cout << "Architecture Util: " << (double)total_utime * 100 /(double)thread_res.stage1_real_cost.time << "%" << std::endl;
            std::cout << "Real Util: " << (double)total_dream_time * 100 / (double)thread_res.stage1_real_cost.time << "%" << std::endl;
            print_avg_buffer_usage(thread_res.g_stage1, thread_res.stage1_real_cost);
            std::cout << "Max Buffer Usage: " << thread_res.g_stage1.buffer.max_buffer_usage << std::endl;
            std::cout << "DRAM Tile Num: " << thread_res.g_stage1.tile_tensor_order.size() << std::endl;
            // thread_res.g_stage1.print_graph_result();
            // Stage2 try using buffer_limit_ratio-buffer-limit-stage1 result
            std::cout << "Stage 2 After Stage 1 With " << thread_res.buffer_limit_ratio << " Buffer Limit" << std::endl;
            std::cout << "Stage 2 SA Iteration Num: " << thread_res.stage2_rounds << std::endl;
            std::cout << "Stage 2 " << thread_res.stage2_stats[0];
            std::cout << "Stage 3 " << thread_res.stage2_stats[1];
            std::cout << "Stage 2 Final Results @" << thread_res.stage2_rounds << ": " << std::endl;
            std::cout << "Real Time: " << thread_res.stage2_real_cost.time << ", Energy: " << thread_res.stage2_real_cost.energy << ", Cost: " << thread_res.stage2_real_cost.cost() << std::endl;
            std::cout << "Ideal Time: " << thread_res.ideal_cost.time << ", Energy: " << thread_res.ideal_cost.energy << ", Cost: " << thread_res.ideal_cost.cost() << std::endl;
            std::cout << "Architecture Util: " << (double)total_utime * 100 /(double)thread_res.stage2_real_cost.time << "%" << std::endl;
            std::cout << "Real Util: " << (double)total_dream_time * 100 / (double)thread_res.stage2_real_cost.time << "%" << std::endl;
            print_avg_buffer_usage(thread_res.g_stage2, thread_res.stage2_real_cost);
            std::cout << "Max Buffer Usage: " << thread_res.g_stage2.buffer.max_buffer_usage << std::endl;
            std::cout << "DRAM Tile Num: " << thread_res.g_stage2.tile_tensor_order.size() << std::endl;
            // g.print_graph_result();
            if (thread_res.stage2_real_cost.time < best_seg_stage2_real_cost.time) {
                best_seg_stage1_real_cost = thread_res.stage1_real_cost;
                best_seg_stage2_real_cost = thread_res.stage2_real_cost;
                best_seg_ideal_cost = thread_res.ideal_cost;
                best_g_stage1 = thread_res.g_stage1;
                best_g_stage2 = thread_res.g_stage2;
                stage1_time += thread_res.stage1_time;
                stage2_time += thread_res.stage2_time;
                best_buffer_limit_ratio = thread_res.buffer_limit_ratio;
                best_ideal_cost_breakdown = thread_res.ideal_cost_breakdown;
            }
        }

        // json only save the best result
        seg_costs.push_back(best_seg_stage2_real_cost);
        total_cost.time += best_seg_stage2_real_cost.time;
        total_cost.energy += best_seg_stage2_real_cost.energy;
        
        o_j["Config"]["Searcher"]["total_stage1_time"] = stage1_time;
        o_j["Config"]["Searcher"]["total_stage2_time"] = stage2_time;
        o_j["Config"]["Searcher"]["mean_stage1_time"] = stage1_time / (_NUM_ITERATIONS + 1);
        o_j["Config"]["Searcher"]["mean_stage2_time"] = stage2_time / (_NUM_ITERATIONS + 1);

        ordered_json o_j_seg;
        o_j_seg["buffer_ratio"] = best_buffer_limit_ratio;
        o_j_seg["DRAM_tile_num"] = best_g_stage2.tile_tensor_order.size();
        o_j_seg["arch_time"] = total_utime;
        o_j_seg["dream_time"] = total_dream_time;
        o_j_seg["ideal_cost"] = best_seg_ideal_cost;
        o_j_seg["ideal_cost_breakdown"] = best_ideal_cost_breakdown;

        o_j_seg["stage1"]["enc"] = best_g_stage1.get_Encoding().first;
        o_j_seg["stage1"]["real_cost"] = best_seg_stage1_real_cost;
        o_j_seg["stage1"]["arch_util"] = (double)total_utime * 100 /(double)best_seg_stage1_real_cost.time;
        o_j_seg["stage1"]["dream_util"] = (double)total_dream_time * 100 / (double)best_seg_stage1_real_cost.time;
        o_j_seg["stage1"]["avg_buffer_usage"] = (double)best_g_stage1.get_sum_buffer_usage() / (double)best_seg_stage1_real_cost.time;
        o_j_seg["stage1"]["max_buffer_usage"] = best_g_stage1.buffer.max_buffer_usage;
        o_j_seg["stage1"]["run_graph"]["buffer_usage"] = best_g_stage1.buffer.buffer_usage_by_time;
        o_j_seg["stage1"]["run_graph"]["DRAM_Tensor_Info"] = best_g_stage1.DRAM_Tensor_Info_by_time;
        o_j_seg["stage1"]["run_graph"]["COMP_Tile_Info"] = best_g_stage1.COMP_Tile_Info_by_time;

        assert(best_g_stage1.get_Encoding().first == best_g_stage2.get_Encoding().first);
        o_j_seg["stage2"]["enc"] = best_g_stage2.get_Encoding().second;
        o_j_seg["stage2"]["real_cost"] = best_seg_stage2_real_cost;
        o_j_seg["stage2"]["arch_util"] = (double)total_utime * 100 /(double)best_seg_stage2_real_cost.time;
        o_j_seg["stage2"]["dream_util"] = (double)total_dream_time * 100 / (double)best_seg_stage2_real_cost.time;
        o_j_seg["stage2"]["avg_buffer_usage"] = (double)best_g_stage2.get_sum_buffer_usage() / (double)best_seg_stage2_real_cost.time;
        o_j_seg["stage2"]["max_buffer_usage"] = best_g_stage2.buffer.max_buffer_usage;
        o_j_seg["stage2"]["run_graph"]["buffer_usage"] = best_g_stage2.buffer.buffer_usage_by_time;
        o_j_seg["stage2"]["run_graph"]["DRAM_Tensor_Info"] = best_g_stage2.DRAM_Tensor_Info_by_time;
        o_j_seg["stage2"]["run_graph"]["COMP_Tile_Info"] = best_g_stage2.COMP_Tile_Info_by_time;

        o_j["Results"].push_back(o_j_seg);
        // for continuous save
        write_to_json(o_j, json_filename);
    }
    std::cout << "Segment Costs: " << std::endl;
    for (int i = 0; i < seg_costs.size(); ++i) {
        std::cout << "Segment [" << i << "] Time: " << seg_costs[i].time << ", Energy: " << seg_costs[i].energy << ", Cost: " << seg_costs[i].cost() << std::endl;
    }
    std::cout << "Total Costs: " << std::endl;
    std::cout << "Time: " << total_cost.time << ", Energy: " << total_cost.energy << ", Cost: " << total_cost.cost() << std::endl;

    o_j["Summary"]["Segment"] = seg_costs;
    o_j["Summary"]["Total"] = total_cost;
    write_to_json(o_j, json_filename);

#ifdef RESULT_LOG_FILE
    fclose(file);
#endif
    return 0;
}