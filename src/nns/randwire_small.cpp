#include "nns/nns.h"

const Network randwire_small = []{
    Network n;
    // block 0
    InputData input("input", fmap_shape(3, 224, 224));
    auto conv2d_1 = n.add(NLAYER("conv2d_1", Conv, C=3, K=32, H=112, W=112, R=3, S=3, sH=2, sW=2), {}, 0, {input});
    auto conv2d_2 = n.add(NLAYER("conv2d_2", Conv, C=32, K=64, H=56, W=56, R=3, S=3, sH=2, sW=2), {conv2d_1});
    auto conv2d_3 = n.add(NLAYER("conv2d_3", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=2, sW=2), {conv2d_2});
    auto conv2d_4 = n.add(NLAYER("conv2d_4", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_3});
    auto conv2d_5 = n.add(NLAYER("conv2d_5", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {conv2d_4});
    auto conv2d_6 = n.add(NLAYER("conv2d_6", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_5});
    auto conv2d_7 = n.add(NLAYER("conv2d_7", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=2, sW=2), {conv2d_2});
    auto conv2d_8 = n.add(NLAYER("conv2d_8", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_7});
    auto element_wise1 = n.add(NLAYER("element_wise1", Eltwise, K=64, H=28, W=28, N=2), {conv2d_4, conv2d_8});
    auto conv2d_9 = n.add(NLAYER("conv2d_9", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise1});
    auto conv2d_10 = n.add(NLAYER("conv2d_10", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_9});
    auto element_wise2 = n.add(NLAYER("element_wise2", Eltwise, K=64, H=28, W=28, N=2), {conv2d_4, conv2d_8});
    auto conv2d_11 = n.add(NLAYER("conv2d_11", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise2});
    auto conv2d_12 = n.add(NLAYER("conv2d_12", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_11});
    auto element_wise3 = n.add(NLAYER("element_wise3", Eltwise, K=64, H=28, W=28, N=2), {conv2d_4, conv2d_10});
    auto conv2d_13 = n.add(NLAYER("conv2d_13", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise3});
    auto conv2d_14 = n.add(NLAYER("conv2d_14", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_13});
    auto element_wise4 = n.add(NLAYER("element_wise4", Eltwise, K=64, H=28, W=28, N=4), {conv2d_6, conv2d_10, conv2d_12, conv2d_14});
    auto conv2d_15 = n.add(NLAYER("conv2d_15", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise4});
    auto conv2d_16 = n.add(NLAYER("conv2d_16", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_15});
    auto element_wise5 = n.add(NLAYER("element_wise5", Eltwise, K=64, H=28, W=28, N=2), {conv2d_10, conv2d_14});
    auto conv2d_17 = n.add(NLAYER("conv2d_17", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise5});
    auto conv2d_18 = n.add(NLAYER("conv2d_18", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_17});
    auto element_wise6 = n.add(NLAYER("element_wise6", Eltwise, K=64, H=28, W=28, N=4), {conv2d_8, conv2d_10, conv2d_16, conv2d_18});
    auto conv2d_19 = n.add(NLAYER("conv2d_19", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise6});
    auto conv2d_20 = n.add(NLAYER("conv2d_20", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_19});
    auto element_wise7 = n.add(NLAYER("element_wise7", Eltwise, K=64, H=28, W=28, N=3), {conv2d_4, conv2d_10, conv2d_20});
    // block 1
    auto conv2d_21 = n.add(NLAYER("conv2d_21", Conv, C=64, K=64, H=28, W=28, R=3, S=3, sH=1, sW=1), {element_wise7});
    auto conv2d_22 = n.add(NLAYER("conv2d_22", Conv, C=64, K=64, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_21});
    auto element_wise8 = n.add(NLAYER("element_wise8", PTP, K=64, H=28, W=28), {conv2d_22});
    auto conv2d_23 = n.add(NLAYER("conv2d_23", Conv, C=64, K=64, H=14, W=14, R=3, S=3, sH=2, sW=2), {element_wise8});
    auto conv2d_24 = n.add(NLAYER("conv2d_24", Conv, C=64, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_23});
    auto conv2d_25 = n.add(NLAYER("conv2d_25", Conv, C=64, K=64, H=14, W=14, R=3, S=3, sH=2, sW=2), {element_wise8});
    auto conv2d_26 = n.add(NLAYER("conv2d_26", Conv, C=64, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_25});
    auto conv2d_27 = n.add(NLAYER("conv2d_27", Conv, C=64, K=64, H=14, W=14, R=3, S=3, sH=2, sW=2), {element_wise8});
    auto conv2d_28 = n.add(NLAYER("conv2d_28", Conv, C=64, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_27});
    auto element_wise9 = n.add(NLAYER("element_wise9", Eltwise, K=128, H=14, W=14, N=2), {conv2d_24, conv2d_26});
    auto conv2d_29 = n.add(NLAYER("conv2d_29", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {element_wise9});
    auto conv2d_30 = n.add(NLAYER("conv2d_30", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_29});
    auto conv2d_31 = n.add(NLAYER("conv2d_31", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {conv2d_24});
    auto conv2d_32 = n.add(NLAYER("conv2d_32", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_31});
    auto element_wise10 = n.add(NLAYER("element_wise10", Eltwise, K=128, H=14, W=14, N=3), {conv2d_24, conv2d_26, conv2d_28});
    auto conv2d_33 = n.add(NLAYER("conv2d_33", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {element_wise10});
    auto conv2d_34 = n.add(NLAYER("conv2d_34", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_33});
    auto element_wise11 = n.add(NLAYER("element_wise11", Eltwise, K=128, H=14, W=14, N=2), {conv2d_32, conv2d_34});
    auto conv2d_35 = n.add(NLAYER("conv2d_35", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {element_wise11});
    auto conv2d_36 = n.add(NLAYER("conv2d_36", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_35});
    auto element_wise12 = n.add(NLAYER("element_wise12", Eltwise, K=128, H=14, W=14, N=5), {conv2d_24, conv2d_26, conv2d_28, conv2d_30, conv2d_36});
    auto conv2d_37 = n.add(NLAYER("conv2d_37", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {element_wise12});
    auto conv2d_38 = n.add(NLAYER("conv2d_38", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_37});
    auto element_wise13 = n.add(NLAYER("element_wise13", Eltwise, K=128, H=14, W=14, N=3), {conv2d_30, conv2d_36, conv2d_38});
    auto conv2d_39 = n.add(NLAYER("conv2d_39", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {element_wise13});
    auto conv2d_40 = n.add(NLAYER("conv2d_40", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_39});
    auto element_wise14 = n.add(NLAYER("element_wise14", Eltwise, K=128, H=14, W=14, N=4), {conv2d_30, conv2d_32, conv2d_34, conv2d_40});
    // block 2
    auto conv2d_41 = n.add(NLAYER("conv2d_41", Conv, C=128, K=128, H=14, W=14, R=3, S=3, sH=1, sW=1), {element_wise14});
    auto conv2d_42 = n.add(NLAYER("conv2d_42", Conv, C=128, K=128, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_41});
    auto element_wise15 = n.add(NLAYER("element_wise15", PTP, K=128, H=14, W=14), {conv2d_42});
    auto conv2d_43 = n.add(NLAYER("conv2d_43", Conv, C=128, K=128, H=7, W=7, R=3, S=3, sH=2, sW=2), {element_wise15});
    auto conv2d_44 = n.add(NLAYER("conv2d_44", Conv, C=128, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_43});
    auto conv2d_45 = n.add(NLAYER("conv2d_45", Conv, C=128, K=128, H=7, W=7, R=3, S=3, sH=2, sW=2), {element_wise15});
    auto conv2d_46 = n.add(NLAYER("conv2d_46", Conv, C=128, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_45});
    auto conv2d_47 = n.add(NLAYER("conv2d_47", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {conv2d_46});
    auto conv2d_48 = n.add(NLAYER("conv2d_48", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_47});
    auto conv2d_49 = n.add(NLAYER("conv2d_49", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {conv2d_48});
    auto conv2d_50 = n.add(NLAYER("conv2d_50", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_49});
    auto element_wise16 = n.add(NLAYER("element_wise16", Eltwise, K=256, H=7, W=7, N=2), {conv2d_44, conv2d_50});
    auto conv2d_51 = n.add(NLAYER("conv2d_51", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {element_wise16});
    auto conv2d_52 = n.add(NLAYER("conv2d_52", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_51});
    auto element_wise17 = n.add(NLAYER("element_wise17", Eltwise, K=256, H=7, W=7, N=3), {conv2d_44, conv2d_46, conv2d_52});
    auto conv2d_53 = n.add(NLAYER("conv2d_53", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {element_wise17});
    auto conv2d_54 = n.add(NLAYER("conv2d_54", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_53});
    auto element_wise18 = n.add(NLAYER("element_wise18", Eltwise, K=256, H=7, W=7, N=3), {conv2d_50, conv2d_52, conv2d_54});
    auto conv2d_55 = n.add(NLAYER("conv2d_55", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {element_wise18});
    auto conv2d_56 = n.add(NLAYER("conv2d_56", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_55});
    auto element_wise19 = n.add(NLAYER("element_wise19", Eltwise, K=256, H=7, W=7, N=4), {conv2d_44, conv2d_48, conv2d_52, conv2d_54});
    auto conv2d_57 = n.add(NLAYER("conv2d_57", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {element_wise19});
    auto conv2d_58 = n.add(NLAYER("conv2d_58", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_57});
    auto element_wise20 = n.add(NLAYER("element_wise20", Eltwise, K=256, H=7, W=7, N=3), {conv2d_46, conv2d_48, conv2d_52});
    auto conv2d_59 = n.add(NLAYER("conv2d_59", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {element_wise20});
    auto conv2d_60 = n.add(NLAYER("conv2d_60", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_59});
    auto element_wise21 = n.add(NLAYER("element_wise21", Eltwise, K=256, H=7, W=7, N=3), {conv2d_46, conv2d_50, conv2d_58});
    auto conv2d_61 = n.add(NLAYER("conv2d_61", Conv, C=256, K=256, H=7, W=7, R=3, S=3, sH=1, sW=1), {element_wise21});
    auto conv2d_62 = n.add(NLAYER("conv2d_62", Conv, C=256, K=256, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_61});
    auto element_wise22 = n.add(NLAYER("element_wise22", Eltwise, K=256, H=7, W=7, N=3), {conv2d_56, conv2d_60, conv2d_62});
    auto conv2d_63 = n.add(NLAYER("conv2d_63", Conv, C=256, K=1024, H=7, W=7, R=1, S=1, sH=1, sW=1), {element_wise22});
    // auto conv2d_64 = n.add(NLAYER("conv2d_64", GroupConv, C=1024, K=1024, H=1, W=1, R=7, S=7, sH=1, sW=1, G=1024), {conv2d_63});
    // auto conv2d_65 = n.add(NLAYER("conv2d_65", Conv, C=1024, K=1000, H=1, W=1, R=1, S=1, sH=1, sW=1), {conv2d_64});
    return n;
}();