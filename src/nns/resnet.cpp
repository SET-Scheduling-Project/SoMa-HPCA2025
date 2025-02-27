#include "nns/nns.h"
#include <cassert>

static constexpr int seg_lens_50[] = {3,4,6,3};
static constexpr int seg_lens_101[] = {3,4,23,3};
static constexpr int seg_lens_152[] = {3,8,36,3};

// Bottleneck block.
static void btn_blk(
	Network& n, Network::lid_t& prev,
	int seg, int blk, bool has_br,
	len_t in_C, len_t mid_C, len_t out_C,
	len_t H_len, len_t strd=1
){
	std::string name = "conv_"+std::to_string(seg)+"_"+std::to_string(blk)+"_";
	n.add(NLAYER(name+"a", Conv, C=in_C, K=mid_C, H=H_len, sH=strd), {prev});
	n.add(NLAYER(name+"b", Conv, C=mid_C, K=mid_C, H=H_len, R=3));
	auto layer_c = n.add(NLAYER(name+"c", Conv, C=mid_C, K=out_C, H=H_len));
	if(has_br){
		prev = n.add(NLAYER("conv_"+std::to_string(seg)+"_br", Conv,
				C=in_C, K=out_C, H=H_len, sH=strd), {prev});
	}
	prev = n.add(NLAYER(name+"res", Eltwise, K=out_C, H=H_len, N=2), {prev, layer_c});
}

static void group_btn_blk(
	Network& n, Network::lid_t& prev,
	int seg, int blk, bool has_br,
	len_t in_C, len_t mid_C, len_t out_C,
	len_t H_len, len_t strd=1
){
	std::string name = "conv_"+std::to_string(seg)+"_"+std::to_string(blk)+"_";
	n.add(NLAYER(name+"a", Conv, C=in_C, K=mid_C, H=H_len, sH=strd), {prev});
	n.add(NLAYER(name+"b", GroupConv, G=32, C=mid_C, K=mid_C, H=H_len, R=3));
	auto layer_c = n.add(NLAYER(name+"c", Conv, C=mid_C, K=out_C, H=H_len));
	if(has_br){
		prev = n.add(NLAYER("conv_"+std::to_string(seg)+"_br", Conv,
				C=in_C, K=out_C, H=H_len, sH=strd), {prev});
	}
	prev = n.add(NLAYER(name+"res", Eltwise, K=out_C, H=H_len, N=2), {prev, layer_c});
}

static Network gen_resnet(const int* seg_lens, bool no_fc=false){
	len_t H_lens[] = {56,28,14,7};
	len_t mid_Cs[] = {64,128,256,512};
	len_t out_Cs[] = {256,512,1024,2048};
	len_t init_C = 64;

	Network::lid_t prev;
	Network n;
	InputData input("input", fmap_shape(3,224));

	n.add(NLAYER("conv1", Conv, C=3, K=init_C, H=112, R=7, sH=2), {}, 0, {input});
	prev = n.add(NLAYER("pool1", Pooling, K=init_C, H=56, R=3, sH=2));

	len_t strd = 1;
	len_t in_C = init_C;
	len_t H_len, mid_C, out_C;
	for(int seg=2; seg<6; ++seg){
		H_len = H_lens[seg-2];
		mid_C = mid_Cs[seg-2];
		out_C = out_Cs[seg-2];

		for(int blk=0; blk<seg_lens[seg-2]; ++blk){
			btn_blk(n, prev, seg, blk, blk==0,
					in_C, mid_C, out_C, H_len, strd);
			strd = 1;
			in_C = out_C;
		}
		strd = 2;
	}
	if (!no_fc) {
		n.add(NLAYER("pool5", Pooling, K=in_C, H=1, R=H_len), {prev});
		n.add(NLAYER("fc", FC, C=in_C, K=1000));
	}
	return n;
}

static Network gen_resnext(const int* seg_lens, bool no_fc=false){
	len_t H_lens[] = {56,28,14,7};
	len_t mid_Cs[] = {64,128,256,512};
	len_t out_Cs[] = {256,512,1024,2048};
	len_t init_C = 64;

	Network::lid_t prev;
	Network n;
	InputData input("input", fmap_shape(3,224));

	n.add(NLAYER("conv1", Conv, C=3, K=init_C, H=112, R=7, sH=2), {}, 0, {input});
	prev = n.add(NLAYER("pool1", Pooling, K=init_C, H=56, R=3, sH=2));

	len_t strd = 1;
	len_t in_C = init_C;
	len_t H_len, mid_C, out_C;
	for(int seg=2; seg<6; ++seg){
		H_len = H_lens[seg-2];
		mid_C = mid_Cs[seg-2];
		out_C = out_Cs[seg-2];

		for(int blk=0; blk<seg_lens[seg-2]; ++blk){
			group_btn_blk(n, prev, seg, blk, blk==0,
					in_C, mid_C, out_C, H_len, strd);
			strd = 1;
			in_C = out_C;
		}
		strd = 2;
	}
	if (!no_fc) {
		n.add(NLAYER("pool5", Pooling, K=in_C, H=1, R=H_len), {prev});
		n.add(NLAYER("fc", FC, C=in_C, K=1000));
	}
	return n;
}

static void retinanet_head(Network& n, 
	const Network::lid_t prev, 
	const std::string head_type, 
	const len_t num_features_in, 
	const len_t H_len, 
	const len_t feature_size=256, 
	const len_t num_anchors=9, 
	const len_t num_class=80)
{
	const bool is_regression = head_type == "regression_head";
	const bool is_classification = head_type == "classification_head";
	assert(is_regression ^ is_classification);
	Network::lid_t out;
	out = n.add(NLAYER(head_type + "_conv_1", Conv, C=num_features_in, K=feature_size, H=H_len, R=3), {prev});
	for (int i = 2; i < 5; i++) {
		out = n.add(NLAYER(head_type + "_conv_" + std::to_string(i), Conv, C=feature_size, K=feature_size, H=H_len, R=3), {out});
	}
	out = n.add(NLAYER(head_type + "_out", Conv, C=feature_size, K=num_anchors*(is_regression ? 4 : num_class), H=H_len, R=3), {out});
}

static std::array<Network::lid_t, 5> 
pyramid_features(Network& n, 
	const Network::lid_t& C3, 
	const Network::lid_t& C4, 
	const Network::lid_t& C5, 
	const len_t C3_size, 
	const len_t C4_size, 
	const len_t C5_size, 
	const len_t H_len,
	const len_t feature_size=256) 
{
	Network::lid_t P3, P4, P4_upsampled, P5, P5_upsampled, P6, P7;

	const len_t P5_H = H_len;
	const len_t P4_H = 2 * H_len;
	const len_t P3_H = 4 * H_len;
	const len_t P6_H = (P5_H - 1) / 2 + 1;
	const len_t P7_H = (P6_H - 1) / 2 + 1;
    P5 = n.add(NLAYER("P5_1", Conv, C=C5_size, K=feature_size, H=P5_H, R=1), {C5});
	P5_upsampled = n.add(NLAYER("P5_upsampled", Upsample, K=feature_size, H=P4_H, W=P4_H, sK=1, sH=2, sW=2), {P5});
	P5 = n.add(NLAYER("P5_2", Conv, C=feature_size, K=feature_size, H=P5_H, R=3), {P5});

	P4 = n.add(NLAYER("P4_1", Conv, C=C4_size, K=feature_size, H=P4_H, R=1), {C4});
	P4 = n.add(NLAYER("P4_add", Eltwise, K=feature_size, H=P4_H, N=2), {P4, P5_upsampled});
	P4_upsampled = n.add(NLAYER("P4_upsampled", Upsample, K=feature_size, H=P3_H, W=P3_H, sK=1, sH=2, sW=2), {P4});
    P4 = n.add(NLAYER("P4_2", Conv, C=feature_size, K=feature_size, H=P4_H, R=3), {P4});

    P3 = n.add(NLAYER("P3_1", Conv, C=C3_size, K=feature_size, H=P3_H, R=1), {C3});
	P3 = n.add(NLAYER("P3_add", Eltwise, K=feature_size, H=P3_H, N=2), {P3, P4_upsampled});
	P3 = n.add(NLAYER("P3_2", Conv, C=feature_size, K=feature_size, H=P3_H, R=3), {P3});

    // P6: Conv2d(C5_size -> feature_size, kernel=3, stride=2, padding=1)
	P6 = n.add(NLAYER("P6", Conv, C=C5_size, K=feature_size, H=P6_H, R=3, sH=2), {C5});
    // P7: ReLU activation followed by Conv2d(feature_size -> feature_size, kernel=3, stride=2, padding=1)
	P7= n.add(NLAYER("P7_1", PTP, K=feature_size, H=P6_H), {P6});
	P7 = n.add(NLAYER("P7_2", Conv, C=feature_size, K=feature_size, H=P7_H, R=3, sH=2), {P7});
	return {P3, P4, P5, P6, P7};
}

const Network resnet50 = gen_resnet(seg_lens_50);
const Network resnet101 = gen_resnet(seg_lens_101);
const Network resnet152 = gen_resnet(seg_lens_152);
const Network resnext50 = gen_resnext(seg_lens_50);

const Network resnet50_no_fc = gen_resnet(seg_lens_50, true);
const Network resnet101_no_fc = gen_resnet(seg_lens_101, true);
const Network resnet152_no_fc = gen_resnet(seg_lens_152, true);
const Network resnext50_no_fc = gen_resnext(seg_lens_50, true);

static Network gen_retinanet(const int* seg_lens, bool no_fc=false){
	len_t H_lens[] = {200,100,50,25,13,7};
	len_t mid_Cs[] = {64,128,256,512};
	len_t out_Cs[] = {256,512,1024,2048};
	len_t init_C = 64;
	Network::lid_t prev;
	Network n;
	InputData input("input", fmap_shape(3,800));

	prev = n.add(NLAYER("conv1", Conv, C=3, K=init_C, H=400, R=7, sH=2), {}, 0, {input});
	prev = n.add(NLAYER("pool1", Pooling, K=init_C, H=200, R=3, sH=2), {prev});

	len_t strd = 1;
	len_t in_C = init_C;
	len_t H_len, mid_C, out_C;
	Network::lid_t C_ids[3];
	len_t C_sizes[3];
	for(int seg=2; seg<6; ++seg){
		H_len = H_lens[seg-2];
		mid_C = mid_Cs[seg-2];
		out_C = out_Cs[seg-2];

		for(int blk=0; blk<seg_lens[seg-2]; ++blk){
			btn_blk(n, prev, seg, blk, blk==0,
					in_C, mid_C, out_C, H_len, strd);
			strd = 1;
			in_C = out_C;
		}
		strd = 2;
		if (seg > 2) {
			C_ids[seg-3] = prev;
			C_sizes[seg-3] = out_C;
		}
	}
	auto features = pyramid_features(n, C_ids[0], C_ids[1], C_ids[2], C_sizes[0], C_sizes[1], C_sizes[2], H_lens[3]);
	for (int i = 0; i < 5; i++) {
		retinanet_head(n, features[i], "classification_head", 256, H_lens[i+1]);
		retinanet_head(n, features[i], "regression_head", 256, H_lens[i+1]);
	}

	return n;
}

const Network retinanet_resnet50 = gen_retinanet(seg_lens_50);