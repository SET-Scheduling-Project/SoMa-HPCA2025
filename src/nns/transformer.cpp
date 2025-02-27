#include "nns/nns.h"
#include <cassert>
typedef TransposeLayer::dim Ldims;

static Network::lid_t add_attention(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize,
		Network::lid_t prevQ, Network::lid_t prevK, Network::lid_t prevV, len_t kv_head_num=1){


	Network::lid_t Q, K, Kt, V, QK, QK_elt, QKV;
	if (kv_head_num == 1) {
		Network::layer_set Ks;
		Q = n.add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prevQ});
		for(len_t i = 0; i < numG; ++i){
			K = n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
			Kt = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {K});
			Ks.push_back(Kt);
		}
		K = n.add(NLAYER(name + "_K", PTP, K=numG*len, H=gSize, W=1), Ks);
		V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, H=len, W=1), {prevV});
		QK = n.add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K=numG*len, G=numG), {Q}, 0, {}, {K});
		QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=numG*len, H=len, W=1), {QK});
		QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V});
		return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
	} else { 
		/*
		// too many layers
		Network::layer_set QKVs;
		// number of K, V == kv_head_num
		for(len_t i = 0; i < kv_head_num; ++i){
			const std::string suffix = "_group" + std::to_string(i);
			K = n.add(NLAYER(name + "_K" + suffix, Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
			Kt = n.add(NLAYER(name + "_Kt" + suffix, Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {K});
			V = n.add(NLAYER(name + "_V" + suffix, Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevV});
			for (len_t j = 0; j < numG/kv_head_num; ++j) {
				Q = n.add(NLAYER(name + "_Q" + suffix, Conv, H=len, W=1, C=numG*gSize, K=gSize), {prevQ});
				QK = n.add(NLAYER(name + "_QK" + suffix, Conv, H=len, W=1, C=gSize, K=len), {Q}, 0, {}, {Kt});
				QK_elt = n.add(NLAYER(name + "_QK_elt" + suffix, PTP, K=len, H=len, W=1), {QK});
				QKV = n.add(NLAYER(name + "_QKV" + suffix, Conv, H=len, W=1, C=len, K=gSize), {QK_elt}, 0, {}, {V});
				QKVs.push_back(QKV);
			}
		}
		return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), QKVs);
		*/
		// number of K, V == kv_head_num
		Network::lid_t K_expand, V_expand;
		Network::layer_set Ks;
		Q = n.add(NLAYER(name + "_Q", Conv, H=len, W=1, C=numG*gSize), {prevQ});
		for(len_t i = 0; i < kv_head_num; ++i){
			K = n.add(NLAYER(name + "_K" + std::to_string(i), Conv, C=numG*gSize, K=gSize, H=len, W=1), {prevK});
			Kt = n.add(NLAYER(name + "_Kt" + std::to_string(i), Transpose, K=len, H=gSize, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {K});
			Ks.push_back(Kt);
		}
		K = n.add(NLAYER(name + "_K", PTP, K=kv_head_num*len, H=gSize, W=1), Ks);
		K_expand = n.add(NLAYER(name + "_K_expand", Upsample, K=numG*len, H=gSize, W=1, sK=numG/kv_head_num, sH=1, sW=1), {K});
		V = n.add(NLAYER(name + "_V", Conv, C=numG*gSize, K = kv_head_num*gSize, H=len, W=1), {prevV});
		V_expand = n.add(NLAYER(name + "_V_expand", Upsample, K=numG*gSize, H=len, W=1, sK=numG/kv_head_num, sH=1, sW=1), {V});
		QK = n.add(NLAYER(name + "_QK", GroupConv, H=len, W=1, C=numG*gSize, K=numG*len, G=numG), {Q}, 0, {}, {K_expand});
		std::cout << n.getNode(QK).layer().weight_shape() << std::endl;
		QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=numG*len, H=len, W=1), {QK});
		QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=len, W=1, C=numG*len, K=numG*gSize, G=numG), {QK_elt}, 0, {}, {V_expand});
		std::cout << n.getNode(QKV).layer().weight_shape() << std::endl;
		return n.add(NLAYER(name + "_FC", Conv, H=len, W=1, C=numG*gSize), {QKV});
	}	
}

static Network::lid_t add_encoder(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len,
		Network::lid_t prev, bool dropout = false){
	Network::lid_t next_prev;
	next_prev = add_attention(n, name, len, numG, gSize, prev, prev, prev);
	if (dropout)
		next_prev = n.add(NLAYER(name + "_dropout", PTP, K=numG*gSize, H=len, W=1), {next_prev});
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

static Network::lid_t add_decoder(
		Network& n, const std::string& name,
		len_t len, len_t numG, len_t gSize, len_t ff_len,
		Network::lid_t prev, Network::lid_t enc_prev){
	Network::lid_t next_prev;
	next_prev = add_attention(n, name+"_1", len, numG, gSize, prev, prev, prev);
	prev = n.add(NLAYER(name + "_elt1", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	next_prev = add_attention(n, name+"_2", len, numG, gSize, prev, enc_prev, enc_prev);
	prev = n.add(NLAYER(name + "_elt2", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
	n.add(NLAYER(name + "_feedfwd1", Conv, C=numG*gSize, K=ff_len, H=len, W=1));
	next_prev = n.add(NLAYER(name + "_feedfwd2", Conv, C=ff_len, K=numG*gSize, H=len, W=1));
	return n.add(NLAYER(name + "_elt3", Eltwise, K=numG*gSize, H=len, W=1, N=2), {prev, next_prev});
}

const Network transformer = []{
	Network::lid_t enc_prev, dec_prev;
	Network::layer_set prevs;
	Network n;
	len_t numG = 8;
	len_t gSize = 64;
	len_t len = 512;
	len_t ff_len = 2048;
	len_t vocab_len = 1000;
	int nEncoder = 6;
	int nDecoder = 6;

	InputData input_enc("input_enc", fmap_shape(numG*gSize, len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_enc});
	for(int i=1; i<=nEncoder; ++i){
		enc_prev = add_encoder(n, "encoder"+std::to_string(i), len, numG, gSize, ff_len, enc_prev);
	}

	InputData input_dec("input_dec", fmap_shape(numG*gSize, len, 1));
	dec_prev = n.add(NLAYER("word_embed_dec", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_dec});
	for(int i=1; i<=nDecoder; ++i){
		dec_prev = add_decoder(n, "decoder"+std::to_string(i), len, numG, gSize, ff_len, dec_prev, enc_prev);
	}
	n.add(NLAYER("proj", Conv, C=numG*gSize, K=vocab_len, H=len, W=1), {dec_prev});
	return n;
}();

const Network transformer_cell = []{
	Network::lid_t enc_prev, dec_prev;
	Network::layer_set prevs;
	Network n;
	len_t numG = 8;
	len_t gSize = 64;
	len_t len = 512;
	len_t ff_len = 2048;
	len_t vocab_len = 1000;

	InputData input_enc("input_enc", fmap_shape(numG*gSize, len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_enc});
	enc_prev = add_encoder(n, "encoder", len, numG, gSize, ff_len, enc_prev);

	InputData input_dec("input_dec", fmap_shape(numG*gSize, len, 1));
	dec_prev = n.add(NLAYER("word_embed_dec", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_dec});
	dec_prev = add_decoder(n, "decoder", len, numG, gSize, ff_len, dec_prev, enc_prev);
	n.add(NLAYER("proj", Conv, C=numG*gSize, K=vocab_len, H=len, W=1), {dec_prev});
	return n;
}();

static const Network create_transformer_cell(len_t numG, len_t gSize, len_t len, len_t ff_len, len_t vocab_len)
{ 
	Network::lid_t enc_prev, dec_prev;
	Network::layer_set prevs;
	Network n;

	InputData input_enc("input_enc", fmap_shape(numG * gSize, len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=numG * gSize, H=len, W=1), {}, 0, {input_enc});
	enc_prev = add_encoder(n, "encoder", len, numG, gSize, ff_len, enc_prev);

	InputData input_dec("input_dec", fmap_shape(numG * gSize, len, 1));
	dec_prev = n.add(NLAYER("word_embed_dec", PTP, K=numG * gSize, H=len, W=1), {}, 0, {input_dec});
	dec_prev = add_decoder(n, "decoder", len, numG, gSize, ff_len, dec_prev, enc_prev);
	n.add(NLAYER("proj", Conv, C=numG * gSize, K=vocab_len, H=len, W=1), {dec_prev});

	return n;
};

static const Network create_BERT_block(len_t num_heads, len_t d_head, len_t seq_len)
{
	Network::lid_t enc_prev;
	Network::layer_set prevs;
	Network n;
	const len_t ff_len = 4 * num_heads * d_head;

	InputData input_enc("input_enc", fmap_shape(num_heads * d_head, seq_len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=num_heads * d_head, H=seq_len, W=1), {}, 0, {input_enc});
	enc_prev = add_encoder(n, "encoder", seq_len, num_heads, d_head, ff_len, enc_prev, true);
	return n;
}

static const Network create_LLM_block_decode(len_t num_heads, len_t d_head, len_t kv_head_num, len_t seq_len)
{
	assert(seq_len >= 1);
	assert(kv_head_num >= 1 && num_heads % kv_head_num == 0);
	const len_t d_model = num_heads * d_head;
	Network n;
	// Input layer for the new token
	InputData input_q("input_q", fmap_shape(num_heads * d_head, 1, 1));
	
	// LayerNorm on the new token
	Network::lid_t norm1 = n.add(NLAYER("norm1", PTP, K=num_heads * d_head, H=1, W=1), {}, 0, {input_q});

	// Attention with KV-cache
    Network::lid_t attn_output;
	{
		// Network::lid_t Q, K, V, newK, newKt, Kt, newV, newVt, Vt,  QK, QK_elt, QKV;
		Network::lid_t Q, K, V, newK, newK_one_head, newKt_one_head, newV,  QK, QK_elt, QKV;
		Network::layer_set newKs;
		const std::string name = "attention";

		Q = n.add(NLAYER(name + "_Q", Conv, C=d_model, H=1, W=1), {norm1});

		for(len_t i = 0; i < kv_head_num; ++i){
			newK_one_head = n.add(NLAYER(name + "_newK_head_" + std::to_string(i), Conv, C=d_model, K=d_head, H=1, W=1), {norm1});
			newKt_one_head = n.add(NLAYER(name + "_newKt_head_" + std::to_string(i), Transpose, K=1, H=d_head, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {newK_one_head});
			newKs.push_back(newKt_one_head);
		}
		newK = n.add(NLAYER(name + "_newK", PTP, K=kv_head_num, H=d_head, W=1), newKs); // concat only on C channel
		newV = n.add(NLAYER(name + "_newV", Conv, C=d_model, K=kv_head_num*d_head, H=1, W=1), {norm1});
		
		// all K and V from DRAM after calculating newK and newV
		InputData k_cache("k_cache", fmap_shape(kv_head_num*(seq_len-1), d_head, 1)); // transposed K cache
		InputData v_cache("v_cache", fmap_shape(seq_len-1, kv_head_num*d_head, 1)); // V cache

		if (kv_head_num != num_heads) {
			Network::lid_t newK_expand, newV_expand, k_cache_expand, v_cache_expand, Vt, newV_expand_t;
			newK_expand = n.add(NLAYER(name + "_newK_expand", Upsample, K=num_heads, H=d_head, W=1, sK=num_heads/kv_head_num, sH=1, sW=1), {newK});
			k_cache_expand = n.add(NLAYER(name + "_k_cache_expand", Upsample, K=num_heads*(seq_len-1), H=d_head, W=1, sK=num_heads/kv_head_num, sH=1, sW=1), {}, 0, {k_cache});
			K = n.add(NLAYER(name + "_K", PTP, K=num_heads*seq_len, H=d_head, W=1), {newK_expand, k_cache_expand});

			newV_expand = n.add(NLAYER(name + "_newV_expand", Upsample, K=num_heads*d_head, H=1, W=1, sK=num_heads/kv_head_num, sH=1, sW=1), {newV});
			newV_expand_t = n.add(NLAYER(name + "_newV_expand_t", Transpose, K=1, H=num_heads*d_head, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {newV_expand});
			v_cache_expand = n.add(NLAYER(name + "_v_cache_expand", Upsample, K=seq_len-1, H=num_heads*d_head, W=1, sK=1, sH=num_heads/kv_head_num, sW=1), {}, 0, {v_cache});
			Vt = n.add(NLAYER(name + "_Vt", PTP, K=seq_len, H=num_heads*d_head, W=1), {newV_expand_t, v_cache_expand});
			V = n.add(NLAYER(name + "_V", Transpose, K=num_heads*d_head, H=seq_len, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {Vt});
		} else {
			Network::lid_t newVt, Vt; 
			K = n.add(NLAYER(name + "_K", PTP, K=num_heads*seq_len, H=d_head, W=1), {newK}, 0, {k_cache});
			newVt = n.add(NLAYER(name + "_newVt", Transpose, K=1, H=kv_head_num*d_head, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {newV});
			Vt = n.add(NLAYER(name + "_Vt_concat", PTP, K=seq_len, H=kv_head_num*d_head, W=1), {newVt}, 0, {v_cache});
			V = n.add(NLAYER(name + "_V", Transpose, K=kv_head_num*d_head, H=seq_len, W=1, order[Ldims::C]=Ldims::H, order[Ldims::H]=Ldims::C), {Vt});
		}
		QK = n.add(NLAYER(name + "_QK", GroupConv, H=1, W=1, C=num_heads*d_head, K=num_heads*seq_len, G=num_heads), {Q}, 0, {}, {K}); // K_cache is the weight(from DRAM) of this layer
		QK_elt = n.add(NLAYER(name + "_QK_elt", PTP, K=num_heads*seq_len, H=1, W=1), {QK});
		QKV = n.add(NLAYER(name + "_QKV", GroupConv, H=1, W=1, C=num_heads*seq_len, K=num_heads*d_head, G=num_heads), {QK_elt}, 0, {}, {V}); // V_cache is the weight(from DRAM) of this layer
		// Output projection
		attn_output = n.add(NLAYER(name + "_Out_proj", Conv, C=d_model, H=1, W=1), {QKV});
	}
	// Residual connection after attention
    Network::lid_t res1 = n.add(NLAYER("res1", Eltwise, K=num_heads * d_head, H=1, W=1, N=2), {attn_output}, 0, {input_q});
	// Network::lid_t res1 = n.add(NLAYER("res1", Eltwise, K=num_heads * d_head, H=1, W=1, N=2), {attn_output, norm1});
	// LayerNorm after first residual
    Network::lid_t norm2 = n.add(NLAYER("norm2", PTP, K=num_heads * d_head, H=1, W=1), {res1});
    // Feed-Forward Network
	const len_t d_ff = 4 * d_model;
    Network::lid_t ff1 = n.add(NLAYER("ffn1", Conv, C=d_model, K=d_ff, H=1, W=1), {norm2});
    Network::lid_t ff2 = n.add(NLAYER("GeLU", PTP, K=d_ff, H=1, W=1), {ff1});
    Network::lid_t ff3 = n.add(NLAYER("ffn2", Conv, C=d_ff, K=d_model, H=1, W=1), {ff2});

    // Residual connection after FFN
    Network::lid_t res2 = n.add(NLAYER("res2", Eltwise, K=d_model, H=1, W=1, N=2), {res1, ff3});

	return n;
};

static const Network create_LLM_block_prefill(len_t num_heads, len_t d_head, len_t group_num, len_t seq_len)
{
	assert(seq_len >= 1);
	assert(group_num >= 1 && num_heads % group_num == 0);
	const len_t d_model = num_heads * d_head;
	Network n;
	// Input layer for the new token
	InputData input_qkv("input_qkv", fmap_shape(num_heads * d_head, seq_len, 1));
	
	// LayerNorm on the new token
	Network::lid_t norm1 = n.add(NLAYER("norm1", PTP, K=num_heads * d_head, H=seq_len, W=1), {}, 0, {input_qkv});

	// Attention with KV-cache
    Network::lid_t attn_output = add_attention(n, "attention", seq_len, num_heads, d_head, norm1, norm1, norm1,  group_num);
	// Residual connection after attention
    Network::lid_t res1 = n.add(NLAYER("res1", Eltwise, K=d_model, H=seq_len, W=1, N=2), {attn_output}, 0, {input_qkv});
	//Network::lid_t res1 = n.add(NLAYER("res1", Eltwise, K=d_model, H=seq_len, W=1, N=2), {attn_output, norm1});
	// LayerNorm after first residual
    Network::lid_t norm2 = n.add(NLAYER("norm2", PTP, K=num_heads * d_head, H=seq_len, W=1), {res1});
    // Feed-Forward Network
	const len_t d_ff = 4 * d_model;
    Network::lid_t ff1 = n.add(NLAYER("ffn1", Conv, C=d_model, K=d_ff, H=seq_len, W=1), {norm2});
    Network::lid_t ff2 = n.add(NLAYER("GeLU", PTP, K=d_ff, H=seq_len, W=1), {ff1});
    Network::lid_t ff3 = n.add(NLAYER("ffn2", Conv, C=d_ff, K=d_model, H=seq_len, W=1), {ff2});

    // Residual connection after FFN
    Network::lid_t res2 = n.add(NLAYER("res2", Eltwise, K=d_model, H=seq_len, W=1, N=2), {res1, ff3});

	return n;
};

// seq_len starts from 1
len_t llm_seq_len = 0;
const Network* get_GPT_J_6B_decode() {
	// "num_hidden_layers": 28, "hidden_size": 4096, "num_attention_heads": 16
    static const Network GPT_J_6B = create_LLM_block_decode(16, 256, 16, llm_seq_len);
    return &GPT_J_6B;
}
const Network* get_GPT_J_6B_prefill() {
	// "num_hidden_layers": 28, "hidden_size": 4096, "num_attention_heads": 16
    static const Network GPT_J_6B = create_LLM_block_prefill(16, 256, 1, llm_seq_len);
    return &GPT_J_6B;
}
const Network* get_LLaMa_2_70B_decode() {
	// "num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64, "n_kv_heads": 8
	static const Network LLaMa_2_70B = create_LLM_block_decode(64, 128, 8, llm_seq_len);
	return &LLaMa_2_70B;
}
const Network* get_LLaMa_2_70B_prefill() {
	// "num_hidden_layers": 80, "hidden_size": 8192, "num_attention_heads": 64, "n_kv_heads": 8
	static const Network LLaMa_2_70B = create_LLM_block_prefill(64, 128, 8, llm_seq_len);
	return &LLaMa_2_70B;
}
const Network* get_BERT_large() {
	// "num_hidden_layers": 24, "hidden_size": 1024, "num_attention_heads": 16
	static const Network BERT_large = create_BERT_block(16, 64, llm_seq_len);
	return &BERT_large;
}
const Network* get_BERT_base() {
	// "num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12
	static const Network BERT_base = create_BERT_block(12, 64, llm_seq_len);
	return &BERT_base;
}
const Network* get_GPT_2_small_decode() {
	// "num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12
	static const Network GPT_2_small = create_LLM_block_decode(12, 64, 12, llm_seq_len);
	return &GPT_2_small;
}
const Network* get_GPT_2_small_prefill() {
	// "num_hidden_layers": 12, "hidden_size": 768, "num_attention_heads": 12
	static const Network GPT_2_small = create_LLM_block_prefill(12, 64, 1, llm_seq_len);
	return &GPT_2_small;
}
const Network* get_GPT_2_XL_decode() {
	// "num_hidden_layers": 48, "hidden_size": 1600, "num_attention_heads": 25
	static const Network GPT_2_XL = create_LLM_block_decode(25, 64, 25, llm_seq_len);
	return &GPT_2_XL;
}
const Network* get_GPT_2_XL_prefill() {
	// "num_hidden_layers": 48, "hidden_size": 1600, "num_attention_heads": 25
	static const Network GPT_2_XL = create_LLM_block_prefill(25, 64, 1, llm_seq_len);
	return &GPT_2_XL;
}

const Network transformer_big_cell = []{
	Network::lid_t enc_prev, dec_prev;
	Network::layer_set prevs;
	Network n;
	len_t numG = 16;
	len_t gSize = 64;
	len_t len = 1024;
	len_t ff_len = 4096;
	len_t vocab_len = 1000;

	InputData input_enc("input_enc", fmap_shape(numG*gSize, len, 1));
	enc_prev = n.add(NLAYER("word_embed_enc", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_enc});
	enc_prev = add_encoder(n, "encoder", len, numG, gSize, ff_len, enc_prev);

	InputData input_dec("input_dec", fmap_shape(numG*gSize, len, 1));
	dec_prev = n.add(NLAYER("word_embed_dec", PTP, K=numG*gSize, H=len, W=1), {}, 0, {input_dec});
	dec_prev = add_decoder(n, "decoder", len, numG, gSize, ff_len, dec_prev, enc_prev);
	n.add(NLAYER("proj", Conv, C=numG*gSize, K=vocab_len, H=len, W=1), {dec_prev});
	return n;
}();