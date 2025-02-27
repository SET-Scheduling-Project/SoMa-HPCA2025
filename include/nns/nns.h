#ifndef NNS_H
#define NNS_H

#include "network.h"
#include "utils.h"

extern const Network darknet19;

extern const Network googlenet;
extern const Network googlenet_no_fc;

extern const Network inception_resnet_v1;
extern const Network inception_resnet_v1_no_fc;

extern const Network resnet50;
extern const Network resnet101;
extern const Network resnet152;
extern const Network resnet50_no_fc;
extern const Network resnet101_no_fc;
extern const Network resnet152_no_fc;

// extern Network vgg16;
extern const Network vgg19;

extern const Network zfnet;
extern const Network alexnet;
extern const Network densenet;
extern const Network densenet_no_fc;

extern const Network gnmt;
extern const Network lstm;

extern const Network transformer;
extern const Network transformer_cell;
extern const Network transformer_big_cell;

extern const Network PNASNet;
extern const Network PNASNet_no_fc;
extern const Network resnext50;
extern const Network resnext50_no_fc;
extern const Network retinanet_resnet50;
extern const Network unet;
extern const Network randwire_small;
extern const Network randwire_large;
static const Network create_LLM_block_decode(len_t num_heads, len_t d_head, len_t group_num, len_t seq_len);
static const Network create_LLM_block_prefill(len_t num_heads, len_t d_head, len_t group_num, len_t seq_len);
extern len_t llm_seq_len;
const Network* get_GPT_J_6B_decode();
const Network* get_GPT_J_6B_prefill();
const Network* get_LLaMa_2_70B_decode();
const Network* get_LLaMa_2_70B_prefill();
const Network* get_BERT_large();
const Network* get_BERT_base();
const Network* get_GPT_2_small_decode();
const Network* get_GPT_2_small_prefill();
const Network* get_GPT_2_XL_decode();
const Network* get_GPT_2_XL_prefill();

Network gen_network(std::string model_name);
#endif // NNS_H
