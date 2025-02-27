#include "nns/nns.h"
#include <cassert>
#include <vector>

Network::lid_t double_conv(Network &n, const std::string name, Network::lid_t prev, const len_t in_C, const len_t mid_C, const len_t out_C, const len_t out_H, const std::vector<InputData> inputs={}) {
	Network::lid_t conv1;
	if (inputs.size() > 0) {
		conv1 = n.add(NLAYER(name + "_conv_1", Conv, C=in_C, K=mid_C, H=out_H, R=3), {}, 0, inputs);
	} else {
    	conv1 = n.add(NLAYER(name + "_conv_1", Conv, C=in_C, K=mid_C, H=out_H, R=3), {prev});
	}
    return n.add(NLAYER(name + "_conv_2", Conv, C=mid_C, K=out_C, H=out_H, R=3), {conv1});
}

Network::lid_t down(Network &n, const std::string name, Network::lid_t prev, const len_t in_C, const len_t out_C, const len_t out_H) {
	// in_H = out_H * 2
    auto maxpool = n.add(NLAYER(name + "_DOWN_maxpool", Pooling, K=in_C, H=out_H, R=2, sH=2), {prev});
    return double_conv(n, name + "_DOWN", maxpool, in_C, out_C, out_C, out_H);
}
Network::lid_t up(Network &n, const std::string name, const Network::lid_t prev1, const Network::lid_t prev2, const len_t in_C, const len_t out_C, const len_t out_H, const bool bilinear) {
	// in_H = out_H / 2
    Network::lid_t up = n.add(NLAYER(name + "_UP_upsample", Upsample, K=in_C, H=out_H, W=out_H, sK=1, sH=2, sW=2), {prev1});
	len_t mid_C;
    if (bilinear) {
		mid_C = in_C / 2;
    } else {
        up = n.add(NLAYER(name + "_UP_conv_transpose", Conv, C=in_C, K=in_C / 2, H=out_H, R=2), {up});
		mid_C = out_C;
    }
	const len_t concat_C = n.getNode(up).layer().ofmap_shape().c + n.getNode(prev2).layer().ofmap_shape().c;
    auto cat = n.add(NLAYER(name + "_UP_concat", PTP, K=concat_C, H=out_H, W=out_H), {up, prev2});
    return double_conv(n, name + "_UP", cat, concat_C, mid_C, out_C, out_H);
}

// UNet model definition
Network::lid_t gen_unet(Network &n, const InputData& input, const len_t in_H, const len_t init_C=1, const len_t n_classes=2, const bool bilinear = false) {
	auto x1 = double_conv(n, "x0", 0, init_C, 64, 64, in_H, {input});
    auto x2 = down(n, "x1", x1, 64, 128, in_H/2);
    auto x3 = down(n, "x2", x2, 128, 256, in_H/4);
    auto x4 = down(n, "x3", x3, 256, 512, in_H/8);
    const len_t factor = bilinear ? 2 : 1;
    auto x5 = down(n, "x4", x4, 512, 1024 / factor, in_H/16);
    auto x = up(n, "x4", x5, x4, 1024, 512 / factor, in_H/8, bilinear);
    x = up(n, "x3", x, x3, 512, 256 / factor, in_H/4, bilinear);
    x = up(n, "x2", x, x2, 256, 128 / factor, in_H/2, bilinear);
    x = up(n, "x1", x, x1, 128, 64, in_H, bilinear);
    return n.add(NLAYER("out_conv", Conv, C=64, K=n_classes, H=in_H, R=1), {x});
}

const Network unet = []{
	Network n;
	InputData input("input", fmap_shape(1, 512, 512));
	gen_unet(n, input, 512);
	return n;
}();