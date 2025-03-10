#include "nns/nns.h"

const Network randwire_large = []{
    Network n;
    //block 0
    InputData input("input", fmap_shape(3, 224, 224));
    auto conv2d_1 = n.add(NLAYER("conv2d_1", Conv, C=3, K=39, H=112, W=112, R=3, S=3, sH=2, sW=2), {}, 0, {input});
    auto conv2d_2 = n.add(NLAYER("conv2d_2", Conv, C=39, K=78, H=56, W=56, R=3, S=3, sH=2, sW=2), {conv2d_1});
    auto conv2d_3 = n.add(NLAYER("conv2d_3", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=2, sW=2, G=78), {conv2d_2});
    auto conv2d_4 = n.add(NLAYER("conv2d_4", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_3});
    auto conv2d_5 = n.add(NLAYER("conv2d_5", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=2, sW=2, G=78), {conv2d_2});
    auto conv2d_6 = n.add(NLAYER("conv2d_6", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_5});
    auto element_wise1 = n.add(NLAYER("element_wise1", Eltwise, K=78, H=28, W=28, N=2), {conv2d_4, conv2d_6});
    auto conv2d_7 = n.add(NLAYER("conv2d_7", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise1});
    auto conv2d_8 = n.add(NLAYER("conv2d_8", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_7});
    auto conv2d_9 = n.add(NLAYER("conv2d_9", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_8});
    auto conv2d_10 = n.add(NLAYER("conv2d_10", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_9});
    auto conv2d_11 = n.add(NLAYER("conv2d_11", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_10});
    auto conv2d_12 = n.add(NLAYER("conv2d_12", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_11});
    auto conv2d_13 = n.add(NLAYER("conv2d_13", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_8});
    auto conv2d_14 = n.add(NLAYER("conv2d_14", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_13});
    auto conv2d_15 = n.add(NLAYER("conv2d_15", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_14});
    auto conv2d_16 = n.add(NLAYER("conv2d_16", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_15});
    auto conv2d_17 = n.add(NLAYER("conv2d_17", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=2, sW=2, G=78), {conv2d_2});
    auto conv2d_18 = n.add(NLAYER("conv2d_18", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_17});
    auto element_wise2 = n.add(NLAYER("element_wise2", Eltwise, K=78, H=28, W=28, N=2), {conv2d_12, conv2d_16});
    auto conv2d_19 = n.add(NLAYER("conv2d_19", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise2});
    auto conv2d_20 = n.add(NLAYER("conv2d_20", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_19});
    auto conv2d_21 = n.add(NLAYER("conv2d_21", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_20});
    auto conv2d_22 = n.add(NLAYER("conv2d_22", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_21});
    auto element_wise3 = n.add(NLAYER("element_wise3", Eltwise, K=78, H=28, W=28, N=3), {conv2d_10, conv2d_14, conv2d_18});
    auto conv2d_23 = n.add(NLAYER("conv2d_23", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise3});
    auto conv2d_24 = n.add(NLAYER("conv2d_24", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_23});
    auto conv2d_25 = n.add(NLAYER("conv2d_25", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_24});
    auto conv2d_26 = n.add(NLAYER("conv2d_26", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_25});
    auto conv2d_27 = n.add(NLAYER("conv2d_27", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=2, sW=2, G=78), {conv2d_2});
    auto conv2d_28 = n.add(NLAYER("conv2d_28", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_27});
    auto element_wise4 = n.add(NLAYER("element_wise4", Eltwise, K=78, H=28, W=28, N=2), {conv2d_12, conv2d_16});
    auto conv2d_29 = n.add(NLAYER("conv2d_29", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise4});
    auto conv2d_30 = n.add(NLAYER("conv2d_30", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_29});
    auto element_wise5 = n.add(NLAYER("element_wise5", Eltwise, K=78, H=28, W=28, N=4), {conv2d_4, conv2d_6, conv2d_26, conv2d_30});
    auto conv2d_31 = n.add(NLAYER("conv2d_31", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise5});
    auto conv2d_32 = n.add(NLAYER("conv2d_32", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_31});
    auto element_wise6 = n.add(NLAYER("element_wise6", Eltwise, K=78, H=28, W=28, N=2), {conv2d_4, conv2d_32});
    auto conv2d_33 = n.add(NLAYER("conv2d_33", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise6});
    auto conv2d_34 = n.add(NLAYER("conv2d_34", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_33});
    auto element_wise7 = n.add(NLAYER("element_wise7", Eltwise, K=78, H=28, W=28, N=3), {conv2d_12, conv2d_16, conv2d_34});
    auto conv2d_35 = n.add(NLAYER("conv2d_35", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise7});
    auto conv2d_36 = n.add(NLAYER("conv2d_36", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_35});
    auto element_wise8 = n.add(NLAYER("element_wise8", Eltwise, K=78, H=28, W=28, N=3), {conv2d_10, conv2d_22, conv2d_36});
    auto conv2d_37 = n.add(NLAYER("conv2d_37", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise8});
    auto conv2d_38 = n.add(NLAYER("conv2d_38", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_37});
    auto element_wise9 = n.add(NLAYER("element_wise9", Eltwise, K=78, H=28, W=28, N=2), {conv2d_30, conv2d_38});
    auto conv2d_39 = n.add(NLAYER("conv2d_39", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise9});
    auto conv2d_40 = n.add(NLAYER("conv2d_40", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_39});
    auto element_wise10 = n.add(NLAYER("element_wise10", Eltwise, K=78, H=28, W=28, N=2), {conv2d_18, conv2d_26});
    auto conv2d_41 = n.add(NLAYER("conv2d_41", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise10});
    auto conv2d_42 = n.add(NLAYER("conv2d_42", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_41});
    auto element_wise11 = n.add(NLAYER("element_wise11", Eltwise, K=78, H=28, W=28, N=3), {conv2d_4, conv2d_22, conv2d_28});
    auto conv2d_43 = n.add(NLAYER("conv2d_43", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise11});
    auto conv2d_44 = n.add(NLAYER("conv2d_44", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_43});
    auto element_wise12 = n.add(NLAYER("element_wise12", Eltwise, K=78, H=28, W=28, N=2), {conv2d_4, conv2d_28});
    auto conv2d_45 = n.add(NLAYER("conv2d_45", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise12});
    auto conv2d_46 = n.add(NLAYER("conv2d_46", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_45});
    auto conv2d_47 = n.add(NLAYER("conv2d_47", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_22});
    auto conv2d_48 = n.add(NLAYER("conv2d_48", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_47});
    auto element_wise13 = n.add(NLAYER("element_wise13", Eltwise, K=78, H=28, W=28, N=2), {conv2d_12, conv2d_28});
    auto conv2d_49 = n.add(NLAYER("conv2d_49", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise13});
    auto conv2d_50 = n.add(NLAYER("conv2d_50", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_49});
    auto element_wise14 = n.add(NLAYER("element_wise14", Eltwise, K=78, H=28, W=28, N=5), {conv2d_4, conv2d_10, conv2d_14, conv2d_18, conv2d_40});
    auto conv2d_51 = n.add(NLAYER("conv2d_51", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise14});
    auto conv2d_52 = n.add(NLAYER("conv2d_52", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_51});
    auto element_wise15 = n.add(NLAYER("element_wise15", Eltwise, K=78, H=28, W=28, N=4), {conv2d_12, conv2d_28, conv2d_42, conv2d_48});
    auto conv2d_53 = n.add(NLAYER("conv2d_53", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise15});
    auto conv2d_54 = n.add(NLAYER("conv2d_54", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_53});
    auto element_wise16 = n.add(NLAYER("element_wise16", Eltwise, K=78, H=28, W=28, N=3), {conv2d_10, conv2d_16, conv2d_36});
    auto conv2d_55 = n.add(NLAYER("conv2d_55", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise16});
    auto conv2d_56 = n.add(NLAYER("conv2d_56", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_55});
    auto element_wise17 = n.add(NLAYER("element_wise17", Eltwise, K=78, H=28, W=28, N=4), {conv2d_38, conv2d_40, conv2d_48, conv2d_54});
    auto conv2d_57 = n.add(NLAYER("conv2d_57", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise17});
    auto conv2d_58 = n.add(NLAYER("conv2d_58", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_57});
    auto conv2d_59 = n.add(NLAYER("conv2d_59", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {conv2d_32});
    auto conv2d_60 = n.add(NLAYER("conv2d_60", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_59});
    auto element_wise18 = n.add(NLAYER("element_wise18", Eltwise, K=78, H=28, W=28, N=4), {conv2d_6, conv2d_50, conv2d_56, conv2d_60});
    auto conv2d_61 = n.add(NLAYER("conv2d_61", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise18});
    auto conv2d_62 = n.add(NLAYER("conv2d_62", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_61});
    auto element_wise19 = n.add(NLAYER("element_wise19", Eltwise, K=78, H=28, W=28, N=2), {conv2d_8, conv2d_52});
    auto conv2d_63 = n.add(NLAYER("conv2d_63", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise19});
    auto conv2d_64 = n.add(NLAYER("conv2d_64", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_63});
    auto element_wise20 = n.add(NLAYER("element_wise20", Eltwise, K=78, H=28, W=28, N=2), {conv2d_26, conv2d_30});
    auto conv2d_65 = n.add(NLAYER("conv2d_65", GroupConv, C=78, K=78, H=28, W=28, R=3, S=3, sH=1, sW=1, G=78), {element_wise20});
    auto conv2d_66 = n.add(NLAYER("conv2d_66", Conv, C=78, K=78, H=28, W=28, R=1, S=1, sH=1, sW=1), {conv2d_65});
    auto element_wise21 = n.add(NLAYER("element_wise21", Eltwise, K=78, H=28, W=28, N=6), {conv2d_44, conv2d_46, conv2d_58, conv2d_62, conv2d_64, conv2d_66});
    // block 1
    auto conv2d_67 = n.add(NLAYER("conv2d_67", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_68 = n.add(NLAYER("conv2d_68", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_67});
    auto conv2d_69 = n.add(NLAYER("conv2d_69", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_68});
    auto conv2d_70 = n.add(NLAYER("conv2d_70", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_69});
    auto conv2d_71 = n.add(NLAYER("conv2d_71", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_72 = n.add(NLAYER("conv2d_72", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_71});
    auto conv2d_73 = n.add(NLAYER("conv2d_73", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_74 = n.add(NLAYER("conv2d_74", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_73});
    auto conv2d_75 = n.add(NLAYER("conv2d_75", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_68});
    auto conv2d_76 = n.add(NLAYER("conv2d_76", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_75});
    auto conv2d_77 = n.add(NLAYER("conv2d_77", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_78 = n.add(NLAYER("conv2d_78", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_77});
    auto conv2d_79 = n.add(NLAYER("conv2d_79", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_74});
    auto conv2d_80 = n.add(NLAYER("conv2d_80", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_79});
    auto conv2d_81 = n.add(NLAYER("conv2d_81", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_82 = n.add(NLAYER("conv2d_82", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_81});
    auto conv2d_83 = n.add(NLAYER("conv2d_83", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_74});
    auto conv2d_84 = n.add(NLAYER("conv2d_84", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_83});
    auto element_wise22 = n.add(NLAYER("element_wise22", Eltwise, K=156, H=14, W=14, N=2), {conv2d_82, conv2d_84});
    auto conv2d_85 = n.add(NLAYER("conv2d_85", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise22});
    auto conv2d_86 = n.add(NLAYER("conv2d_86", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_85});
    auto conv2d_87 = n.add(NLAYER("conv2d_87", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_84});
    auto conv2d_88 = n.add(NLAYER("conv2d_88", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_87});
    auto conv2d_89 = n.add(NLAYER("conv2d_89", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_90 = n.add(NLAYER("conv2d_90", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_89});
    auto conv2d_91 = n.add(NLAYER("conv2d_91", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_76});
    auto conv2d_92 = n.add(NLAYER("conv2d_92", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_91});
    auto element_wise23 = n.add(NLAYER("element_wise23", Eltwise, K=156, H=14, W=14, N=4), {conv2d_68, conv2d_76, conv2d_90, conv2d_92});
    auto conv2d_93 = n.add(NLAYER("conv2d_93", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise23});
    auto conv2d_94 = n.add(NLAYER("conv2d_94", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_93});
    auto element_wise24 = n.add(NLAYER("element_wise24", Eltwise, K=156, H=14, W=14, N=3), {conv2d_80, conv2d_88, conv2d_92});
    auto conv2d_95 = n.add(NLAYER("conv2d_95", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise24});
    auto conv2d_96 = n.add(NLAYER("conv2d_96", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_95});
    auto element_wise25 = n.add(NLAYER("element_wise25", Eltwise, K=156, H=14, W=14, N=4), {conv2d_70, conv2d_76, conv2d_82, conv2d_96});
    auto conv2d_97 = n.add(NLAYER("conv2d_97", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise25});
    auto conv2d_98 = n.add(NLAYER("conv2d_98", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_97});
    auto conv2d_99 = n.add(NLAYER("conv2d_99", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_100 = n.add(NLAYER("conv2d_100", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_99});
    auto conv2d_101 = n.add(NLAYER("conv2d_101", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {conv2d_100});
    auto conv2d_102 = n.add(NLAYER("conv2d_102", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_101});
    auto conv2d_103 = n.add(NLAYER("conv2d_103", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_104 = n.add(NLAYER("conv2d_104", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_103});
    auto element_wise26 = n.add(NLAYER("element_wise26", Eltwise, K=156, H=14, W=14, N=2), {conv2d_68, conv2d_102});
    auto conv2d_105 = n.add(NLAYER("conv2d_105", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise26});
    auto conv2d_106 = n.add(NLAYER("conv2d_106", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_105});
    auto element_wise27 = n.add(NLAYER("element_wise27", Eltwise, K=156, H=14, W=14, N=3), {conv2d_82, conv2d_104, conv2d_106});
    auto conv2d_107 = n.add(NLAYER("conv2d_107", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise27});
    auto conv2d_108 = n.add(NLAYER("conv2d_108", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_107});
    auto element_wise28 = n.add(NLAYER("element_wise28", Eltwise, K=156, H=14, W=14, N=3), {conv2d_100, conv2d_106, conv2d_108});
    auto conv2d_109 = n.add(NLAYER("conv2d_109", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise28});
    auto conv2d_110 = n.add(NLAYER("conv2d_110", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_109});
    auto element_wise29 = n.add(NLAYER("element_wise29", Eltwise, K=156, H=14, W=14, N=3), {conv2d_68, conv2d_74, conv2d_78});
    auto conv2d_111 = n.add(NLAYER("conv2d_111", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise29});
    auto conv2d_112 = n.add(NLAYER("conv2d_112", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_111});
    auto element_wise30 = n.add(NLAYER("element_wise30", Eltwise, K=156, H=14, W=14, N=5), {conv2d_70, conv2d_72, conv2d_74, conv2d_86, conv2d_110});
    auto conv2d_113 = n.add(NLAYER("conv2d_113", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise30});
    auto conv2d_114 = n.add(NLAYER("conv2d_114", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_113});
    auto element_wise31 = n.add(NLAYER("element_wise31", Eltwise, K=156, H=14, W=14, N=5), {conv2d_76, conv2d_86, conv2d_88, conv2d_90, conv2d_98});
    auto conv2d_115 = n.add(NLAYER("conv2d_115", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise31});
    auto conv2d_116 = n.add(NLAYER("conv2d_116", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_115});
    auto element_wise32 = n.add(NLAYER("element_wise32", Eltwise, K=156, H=14, W=14, N=3), {conv2d_72, conv2d_74, conv2d_104});
    auto conv2d_117 = n.add(NLAYER("conv2d_117", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise32});
    auto conv2d_118 = n.add(NLAYER("conv2d_118", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_117});
    auto element_wise33 = n.add(NLAYER("element_wise33", Eltwise, K=156, H=14, W=14, N=3), {conv2d_70, conv2d_106, conv2d_114});
    auto conv2d_119 = n.add(NLAYER("conv2d_119", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise33});
    auto conv2d_120 = n.add(NLAYER("conv2d_120", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_119});
    auto conv2d_121 = n.add(NLAYER("conv2d_121", GroupConv, C=78, K=78, H=14, W=14, R=3, S=3, sH=2, sW=2, G=78), {element_wise21});
    auto conv2d_122 = n.add(NLAYER("conv2d_122", Conv, C=78, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_121});
    auto element_wise34 = n.add(NLAYER("element_wise34", Eltwise, K=156, H=14, W=14, N=5), {conv2d_70, conv2d_78, conv2d_82, conv2d_98, conv2d_122});
    auto conv2d_123 = n.add(NLAYER("conv2d_123", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise34});
    auto conv2d_124 = n.add(NLAYER("conv2d_124", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_123});
    auto element_wise35 = n.add(NLAYER("element_wise35", Eltwise, K=156, H=14, W=14, N=4), {conv2d_102, conv2d_118, conv2d_122, conv2d_124});
    auto conv2d_125 = n.add(NLAYER("conv2d_125", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise35});
    auto conv2d_126 = n.add(NLAYER("conv2d_126", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_125});
    auto element_wise36 = n.add(NLAYER("element_wise36", Eltwise, K=156, H=14, W=14, N=4), {conv2d_86, conv2d_88, conv2d_108, conv2d_118});
    auto conv2d_127 = n.add(NLAYER("conv2d_127", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise36});
    auto conv2d_128 = n.add(NLAYER("conv2d_128", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_127});
    auto element_wise37 = n.add(NLAYER("element_wise37", Eltwise, K=156, H=14, W=14, N=4), {conv2d_102, conv2d_114, conv2d_122, conv2d_126});
    auto conv2d_129 = n.add(NLAYER("conv2d_129", GroupConv, C=156, K=156, H=14, W=14, R=3, S=3, sH=1, sW=1, G=156), {element_wise37});
    auto conv2d_130 = n.add(NLAYER("conv2d_130", Conv, C=156, K=156, H=14, W=14, R=1, S=1, sH=1, sW=1), {conv2d_129});
    auto element_wise38 = n.add(NLAYER("element_wise38", Eltwise, K=156, H=14, W=14, N=6), {conv2d_94, conv2d_112, conv2d_116, conv2d_120, conv2d_128, conv2d_130});
    // block 2
    auto conv2d_131 = n.add(NLAYER("conv2d_131", GroupConv, C=156, K=156, H=7, W=7, R=3, S=3, sH=2, sW=2, G=156), {element_wise38});
    auto conv2d_132 = n.add(NLAYER("conv2d_132", Conv, C=156, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_131});
    auto conv2d_133 = n.add(NLAYER("conv2d_133", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_132});
    auto conv2d_134 = n.add(NLAYER("conv2d_134", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_133});
    auto conv2d_135 = n.add(NLAYER("conv2d_135", GroupConv, C=156, K=156, H=7, W=7, R=3, S=3, sH=2, sW=2, G=156), {element_wise38});
    auto conv2d_136 = n.add(NLAYER("conv2d_136", Conv, C=156, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_135});
    auto conv2d_137 = n.add(NLAYER("conv2d_137", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_136});
    auto conv2d_138 = n.add(NLAYER("conv2d_138", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_137});
    auto conv2d_139 = n.add(NLAYER("conv2d_139", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_138});
    auto conv2d_140 = n.add(NLAYER("conv2d_140", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_139});
    auto conv2d_141 = n.add(NLAYER("conv2d_141", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_140});
    auto conv2d_142 = n.add(NLAYER("conv2d_142", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_141});
    auto conv2d_143 = n.add(NLAYER("conv2d_143", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_136});
    auto conv2d_144 = n.add(NLAYER("conv2d_144", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_143});
    auto conv2d_145 = n.add(NLAYER("conv2d_145", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_134});
    auto conv2d_146 = n.add(NLAYER("conv2d_146", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_145});
    auto conv2d_147 = n.add(NLAYER("conv2d_147", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_136});
    auto conv2d_148 = n.add(NLAYER("conv2d_148", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_147});
    auto element_wise39 = n.add(NLAYER("element_wise39", Eltwise, K=312, H=7, W=7, N=2), {conv2d_144, conv2d_146});
    auto conv2d_149 = n.add(NLAYER("conv2d_149", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise39});
    auto conv2d_150 = n.add(NLAYER("conv2d_150", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_149});
    auto element_wise40 = n.add(NLAYER("element_wise40", Eltwise, K=312, H=7, W=7, N=2), {conv2d_146, conv2d_150});
    auto conv2d_151 = n.add(NLAYER("conv2d_151", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise40});
    auto conv2d_152 = n.add(NLAYER("conv2d_152", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_151});
    auto element_wise41 = n.add(NLAYER("element_wise41", Eltwise, K=312, H=7, W=7, N=2), {conv2d_138, conv2d_142});
    auto conv2d_153 = n.add(NLAYER("conv2d_153", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise41});
    auto conv2d_154 = n.add(NLAYER("conv2d_154", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_153});
    auto conv2d_155 = n.add(NLAYER("conv2d_155", GroupConv, C=156, K=156, H=7, W=7, R=3, S=3, sH=2, sW=2, G=156), {element_wise38});
    auto conv2d_156 = n.add(NLAYER("conv2d_156", Conv, C=156, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_155});
    auto conv2d_157 = n.add(NLAYER("conv2d_157", GroupConv, C=156, K=156, H=7, W=7, R=3, S=3, sH=2, sW=2, G=156), {element_wise38});
    auto conv2d_158 = n.add(NLAYER("conv2d_158", Conv, C=156, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_157});
    auto element_wise42 = n.add(NLAYER("element_wise42", Eltwise, K=312, H=7, W=7, N=2), {conv2d_150, conv2d_158});
    auto conv2d_159 = n.add(NLAYER("conv2d_159", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise42});
    auto conv2d_160 = n.add(NLAYER("conv2d_160", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_159});
    auto conv2d_161 = n.add(NLAYER("conv2d_161", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_134});
    auto conv2d_162 = n.add(NLAYER("conv2d_162", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_161});
    auto conv2d_163 = n.add(NLAYER("conv2d_163", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_148});
    auto conv2d_164 = n.add(NLAYER("conv2d_164", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_163});
    auto element_wise43 = n.add(NLAYER("element_wise43", Eltwise, K=312, H=7, W=7, N=3), {conv2d_140, conv2d_146, conv2d_152});
    auto conv2d_165 = n.add(NLAYER("conv2d_165", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise43});
    auto conv2d_166 = n.add(NLAYER("conv2d_166", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_165});
    auto element_wise44 = n.add(NLAYER("element_wise44", Eltwise, K=312, H=7, W=7, N=2), {conv2d_138, conv2d_164});
    auto conv2d_167 = n.add(NLAYER("conv2d_167", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise44});
    auto conv2d_168 = n.add(NLAYER("conv2d_168", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_167});
    auto conv2d_169 = n.add(NLAYER("conv2d_169", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {conv2d_152});
    auto conv2d_170 = n.add(NLAYER("conv2d_170", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_169});
    auto element_wise45 = n.add(NLAYER("element_wise45", Eltwise, K=312, H=7, W=7, N=3), {conv2d_158, conv2d_164, conv2d_170});
    auto conv2d_171 = n.add(NLAYER("conv2d_171", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise45});
    auto conv2d_172 = n.add(NLAYER("conv2d_172", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_171});
    auto element_wise46 = n.add(NLAYER("element_wise46", Eltwise, K=312, H=7, W=7, N=5), {conv2d_144, conv2d_154, conv2d_156, conv2d_166, conv2d_168});
    auto conv2d_173 = n.add(NLAYER("conv2d_173", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise46});
    auto conv2d_174 = n.add(NLAYER("conv2d_174", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_173});
    auto element_wise47 = n.add(NLAYER("element_wise47", Eltwise, K=312, H=7, W=7, N=5), {conv2d_134, conv2d_156, conv2d_160, conv2d_164, conv2d_166});
    auto conv2d_175 = n.add(NLAYER("conv2d_175", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise47});
    auto conv2d_176 = n.add(NLAYER("conv2d_176", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_175});
    auto element_wise48 = n.add(NLAYER("element_wise48", Eltwise, K=312, H=7, W=7, N=3), {conv2d_148, conv2d_154, conv2d_176});
    auto conv2d_177 = n.add(NLAYER("conv2d_177", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise48});
    auto conv2d_178 = n.add(NLAYER("conv2d_178", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_177});
    auto element_wise49 = n.add(NLAYER("element_wise49", Eltwise, K=312, H=7, W=7, N=2), {conv2d_152, conv2d_178});
    auto conv2d_179 = n.add(NLAYER("conv2d_179", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise49});
    auto conv2d_180 = n.add(NLAYER("conv2d_180", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_179});
    auto element_wise50 = n.add(NLAYER("element_wise50", Eltwise, K=312, H=7, W=7, N=6), {conv2d_132, conv2d_142, conv2d_144, conv2d_154, conv2d_162, conv2d_178});
    auto conv2d_181 = n.add(NLAYER("conv2d_181", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise50});
    auto conv2d_182 = n.add(NLAYER("conv2d_182", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_181});
    auto element_wise51 = n.add(NLAYER("element_wise51", Eltwise, K=312, H=7, W=7, N=3), {conv2d_132, conv2d_150, conv2d_170});
    auto conv2d_183 = n.add(NLAYER("conv2d_183", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise51});
    auto conv2d_184 = n.add(NLAYER("conv2d_184", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_183});
    auto element_wise52 = n.add(NLAYER("element_wise52", Eltwise, K=312, H=7, W=7, N=2), {conv2d_158, conv2d_184});
    auto conv2d_185 = n.add(NLAYER("conv2d_185", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise52});
    auto conv2d_186 = n.add(NLAYER("conv2d_186", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_185});
    auto conv2d_187 = n.add(NLAYER("conv2d_187", GroupConv, C=156, K=156, H=7, W=7, R=3, S=3, sH=2, sW=2, G=156), {element_wise38});
    auto conv2d_188 = n.add(NLAYER("conv2d_188", Conv, C=156, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_187});
    auto element_wise53 = n.add(NLAYER("element_wise53", Eltwise, K=312, H=7, W=7, N=4), {conv2d_168, conv2d_178, conv2d_186, conv2d_188});
    auto conv2d_189 = n.add(NLAYER("conv2d_189", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise53});
    auto conv2d_190 = n.add(NLAYER("conv2d_190", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_189});
    auto element_wise54 = n.add(NLAYER("element_wise54", Eltwise, K=312, H=7, W=7, N=4), {conv2d_132, conv2d_142, conv2d_158, conv2d_174});
    auto conv2d_191 = n.add(NLAYER("conv2d_191", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise54});
    auto conv2d_192 = n.add(NLAYER("conv2d_192", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_191});
    auto element_wise55 = n.add(NLAYER("element_wise55", Eltwise, K=312, H=7, W=7, N=4), {conv2d_132, conv2d_134, conv2d_180, conv2d_188});
    auto conv2d_193 = n.add(NLAYER("conv2d_193", GroupConv, C=312, K=312, H=7, W=7, R=3, S=3, sH=1, sW=1, G=312), {element_wise55});
    auto conv2d_194 = n.add(NLAYER("conv2d_194", Conv, C=312, K=312, H=7, W=7, R=1, S=1, sH=1, sW=1), {conv2d_193});
    auto element_wise56 = n.add(NLAYER("element_wise56", Eltwise, K=312, H=7, W=7, N=5), {conv2d_172, conv2d_182, conv2d_190, conv2d_192, conv2d_194});
    // block 3
    auto conv2d_195 = n.add(NLAYER("conv2d_195", Conv, C=312, K=1280, H=7, W=7, R=1, S=1, sH=1, sW=1), {element_wise56});
    auto conv2d_196 = n.add(NLAYER("conv2d_196", GroupConv, C=1280, K=1280, H=1, W=1, R=7, S=7, sH=1, sW=1, G=1280), {conv2d_195});
    auto conv2d_197 = n.add(NLAYER("conv2d_197", Conv, C=1280, K=1000, H=1, W=1, R=1, S=1, sH=1, sW=1), {conv2d_196});
    return n;
}();