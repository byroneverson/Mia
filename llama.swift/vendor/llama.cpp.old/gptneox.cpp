#include "gptneox.h"

#include "ggml.h" //-gptneox.h"

#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <regex>
#include <cassert>
#include <cstring>

#define GPTNEOX_USE_SCRATCH
#define GPTNEOX_MAX_SCRATCH_BUFFERS 16

#define GPTNEOX_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "GPTNEOX_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifdef __GNUC__
#ifdef __MINGW32__
__attribute__((format(gnu_printf, 1, 2)))
#else
__attribute__((format(printf, 1, 2)))
#endif
#endif
static std::string format(const char * fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GPTNEOX_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GPTNEOX_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

static const size_t MB = 1024*1024;

// Only 12B Open-Assistant model for now
// These values are rough for bootstrapping
// Compared to llama 13B, oasst 12B has:
// same max length
// same hidden size
// same heads count
// 36 layers instead of 40
// 50288 vocab size instead of 32000
// 20480 intermediate instead of 13824
// gelu instead of silu (almost the same, in theory, the weights/biases can be pre-scaled to use one or the other)
static const int GPTNEOX_N_PARTS = 1;
static const size_t MEM_REQ_SCRATCH0 = 512ull*MB;
static const size_t MEM_REQ_SCRATCH1 = 512ull*MB;
static const size_t MEM_REQ_KV_SELF = 1608ull*MB;
static const size_t MEM_REQ_EVAL = 1024ull*MB;


/*
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif
 */

// default hparams (GPT-NeoX oasst 12B)
struct gptneox_hparams {
    int32_t n_vocab = 50288;
    int32_t n_ctx   = 2048; // full max length
    int32_t n_embd  = 5120;
    int32_t n_head  = 40;
    int32_t n_layer = 36;
    int32_t n_rot   = 32;
    int32_t use_parallel_residual = 1; // 1 = true, 0 = false
    int32_t f16     = 1;
};

struct gptneox_layer {
    // input_layernorm
    struct ggml_tensor * input_layernorm_weight;
    struct ggml_tensor * input_layernorm_bias;

    // post_attention_layernorm
    struct ggml_tensor * post_attention_layernorm_weight;
    struct ggml_tensor * post_attention_layernorm_bias;

    // attention
    struct ggml_tensor * c_attn_qkv_proj_w;
    //struct ggml_tensor * c_attn_q_proj_w;
    //struct ggml_tensor * c_attn_k_proj_w;
    //struct ggml_tensor * c_attn_v_proj_w;

    struct ggml_tensor * c_attn_qkv_proj_bias;
    //struct ggml_tensor * c_attn_q_proj_bias;
    //struct ggml_tensor * c_attn_k_proj_bias;
    //struct ggml_tensor * c_attn_v_proj_bias;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_bias;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w_trans;
    struct ggml_tensor * c_mlp_proj_b;
};

// From llama_util.cpp
// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct gptneox_buffer {
    uint8_t * addr = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] addr;
        addr = new uint8_t[size];
        this->size = size;
    }

    ~gptneox_buffer() {
        delete[] addr;
    }
};

struct gptneox_kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = NULL;

    gptneox_buffer buf;

    int n; // number of tokens currently in the cache

    ~gptneox_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct gptneox_model {
    gptneox_hparams hparams;

    // final normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte; // word embedding

    struct ggml_tensor * lmh_g; // language model head
    // struct ggml_tensor * lmh_b; // language model bias

    std::vector<gptneox_layer> layers;

    // context
    struct ggml_context * ctx = NULL;
    
    // key + value cache for the self attention
    // TODO: move to llama_state
    struct gptneox_kv_cache kv_self;
    
    // the model memory buffer
    gptneox_buffer buf;
    
    std::map<std::string, struct ggml_tensor *> tensors;
};

// from cformers/utils.h
struct gptneox_vocab {
    using id    = int32_t;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

// llama.cpp version
/*
struct gptneox_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};
 */

struct gptneox_context {
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    gptneox_model model;
    gptneox_vocab vocab;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in llama_state
    std::vector<uint8_t> buf_compute;
    std::vector<uint8_t> buf_scratch[GPTNEOX_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[GPTNEOX_MAX_SCRATCH_BUFFERS] = { 0 };

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(GPTNEOX_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size(), buf.data(), });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(GPTNEOX_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

//
// kv cache
//

static bool kv_cache_init(
        const struct gptneox_hparams & hparams,
             struct gptneox_kv_cache & cache,
                         ggml_type   wtype,
                               int   n_ctx) {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = (int64_t)n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.addr;
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    return true;
}


struct gptneox_context_params gptneox_context_default_params() {
    struct gptneox_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_parts                     =*/ -1,
        /*.seed                        =*/ 0,
        /*.f16_kv                      =*/ false,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
    };

    return result;
}

//
// model loading
//

// load the model's weights from a file
bool gptneox_model_load(
                        const std::string & fname,
                        gptneox_context & lctx,
                        int n_ctx,
                        int n_parts,
                        ggml_type memory_type,
                        //bool use_mmap, // implement me
                        //bool use_mlock, // implement me
                        bool vocab_only,
                        gptneox_progress_callback progress_callback,
                        void *progress_callback_user_data) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());
    
    const int64_t t_start_us = ggml_time_us();

    lctx.t_start_us = t_start_us;

    //std::vector<char> f_buf(1024*1024);

    auto & model = lctx.model;
    auto & vocab = lctx.vocab;

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic == GPTNEOX_FILE_MAGIC_UNVERSIONED) {
            fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files or convert them with convert-unversioned-ggml-to-ggml.py!)\n",
                    __func__, fname.c_str());
            return false;
        }
        if (magic != GPTNEOX_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }

        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != GPTNEOX_FILE_VERSION) {
            fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                    __func__, fname.c_str(), format_version, GPTNEOX_FILE_VERSION);
            return false;
        }
    }
    
    //int n_ff = 0;

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        // fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
        fin.read((char *) &hparams.use_parallel_residual,     sizeof(hparams.use_parallel_residual));
        fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

        hparams.n_ctx = n_ctx;
        
        // Must convert from llama to gptneox (n_mult?)
        //n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

        if (n_parts < 1) {
            n_parts = GPTNEOX_N_PARTS; //.at(hparams.n_embd);
        }
        
        // temp warning to tell the user to use "--n_parts"
        if (hparams.f16 == 4 && n_parts != 1) {
            fprintf(stderr, "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
            fprintf(stderr, "%s: use '--n_parts 1' if necessary\n", __func__);
        }
        
        // We will try to prevent hard-coded model specs
        /*
        if (hparams.n_layer == 32) {
            model.type = e_model::MODEL_7B;
        }

        if (hparams.n_layer == 40) {
            model.type = e_model::MODEL_13B;
        }

        if (hparams.n_layer == 60) {
            model.type = e_model::MODEL_30B;
        }

        if (hparams.n_layer == 80) {
            model.type = e_model::MODEL_65B;
        }
         */
        
        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: use_parallel_residual = %d\n", __func__, hparams.use_parallel_residual);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
        //printf("%s: n_ff    = %d\n", __func__, n_ff);
        printf("%s: n_parts = %d\n", __func__, n_parts);
        //printf("%s: type    = %d\n", __func__, model.type);
    }

    // load vocab
    {
        std::string word;
        int32_t n_vocab = model.hparams.n_vocab;
        // fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }
        
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }
    
    if (vocab_only) {
        return true;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    // GPTQ (case 4) might not be completed
    // 1d tensors are not quantized (GGML_TYPE_F32 size)
    ggml_type wtype, vtype;
    switch (model.hparams.f16) {
        case 0: wtype = vtype = GGML_TYPE_F32;  break;
        case 1: wtype = vtype = GGML_TYPE_F16;  break;
        case 2: wtype = vtype = GGML_TYPE_Q4_0; break;
        case 3: wtype = vtype = GGML_TYPE_Q4_1; break;
        case 4: wtype = GGML_TYPE_Q4_1; vtype = GGML_TYPE_F16; break;
        default:
                {
                    fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                            __func__, fname.c_str(), model.hparams.f16);
                    return false;
                }
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd*n_vocab*ggml_type_sizef(vtype); // wte
        
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_embd*n_vocab*ggml_type_sizef(vtype);         // lmh_g
        // ctx_size +=        n_vocab*ggml_type_sizef(GGML_TYPE_F32); // lmh_b

        { // Transformer layers
            // Input norm
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // input_layernorm_weight
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // input_layernorm_bias

            // Attention
            ctx_size += n_layer*(n_embd*3*n_embd*ggml_type_sizef(wtype)); // c_attn_qkv_proj_w
            //ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_q_proj_w
            //ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_k_proj_w
            //ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_v_proj_w

            ctx_size += n_layer*(3*n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_qkv_proj_bias
            //ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_q_proj_bias
            //ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_k_proj_bias
            //ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_v_proj_bias

            ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // c_attn_proj_w
            ctx_size += n_layer*(n_embd*       ggml_type_sizef(GGML_TYPE_F32)); // c_attn_proj_bias
            
            // FF Norm
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // post_attention_layernorm_weight
            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // post_attention_layernorm_bias

            // Feedforward layer
            ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_fc_w
            ctx_size += n_layer*(       4*n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

            ctx_size += n_layer*(4*n_embd*n_embd*ggml_type_sizef(wtype));         // c_mlp_proj_w_trans
            ctx_size += n_layer*(         n_embd*ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b
        }

        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(memory_type); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(memory_type); // memory_v

        // How is this determined?
        ctx_size += (6 + 16*n_layer)*256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }
    
    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
            ctx_size +
            MEM_REQ_SCRATCH0 + //.at(model.type) +
            MEM_REQ_SCRATCH1 + //.at(model.type) +
            MEM_REQ_EVAL; //.at    (model.type);

        // this is the memory required by one llama_state
        const size_t mem_required_state =
            scale*MEM_REQ_KV_SELF; //.at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
    }

    // create the ggml context
    {
        lctx.model.buf.resize(ctx_size);
        /*if (use_mlock) {
            lctx.model.mlock_buf.init(lctx.model.buf.addr);
            lctx.model.mlock_buf.grow_to(lctx.model.buf.size);
        }*/
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ lctx.model.buf.size,
            /*.mem_buffer =*/ lctx.model.buf.addr,
            /*.no_alloc   =*/ //ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        //const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.wte    = ggml_new_tensor_2d(ctx, vtype,         n_embd, n_vocab);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.lmh_g  = ggml_new_tensor_2d(ctx, vtype,         n_embd, n_vocab);
        // model.lmh_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

        // map by name
        model.tensors["gpt_neox.embed_in.weight"] = model.wte;

        model.tensors["gpt_neox.final_layer_norm.weight"] = model.ln_f_g;
        model.tensors["gpt_neox.final_layer_norm.bias"]   = model.ln_f_b;

        model.tensors["embed_out.weight"] = model.lmh_g;
        // model.tensors["lm_head.bias"]   = model.lmh_b;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            // Input norm
            layer.input_layernorm_weight          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.input_layernorm_bias            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            
            // Attention
            layer.c_attn_qkv_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            //layer.c_attn_q_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            //layer.c_attn_k_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            //layer.c_attn_v_proj_w       = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);

            layer.c_attn_qkv_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);
            //layer.c_attn_q_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            //layer.c_attn_k_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            //layer.c_attn_v_proj_bias    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_proj_w         = ggml_new_tensor_2d(ctx, wtype,           n_embd,   n_embd);
            layer.c_attn_proj_bias      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            
            // FF norm
            layer.post_attention_layernorm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.post_attention_layernorm_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // Feedforward
            layer.c_mlp_fc_w            = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b            = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w_trans    = ggml_new_tensor_2d(ctx, wtype,         4*n_embd,   n_embd);
            layer.c_mlp_proj_b          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name
            // Input norm
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.weight"]          = layer.input_layernorm_weight;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".input_layernorm.bias"]            = layer.input_layernorm_bias;
            // Attention
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.weight"]   = layer.c_attn_qkv_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query_key_value.bias"]     = layer.c_attn_qkv_proj_bias;
            //model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query.weight"]   = layer.c_attn_q_proj_w;
            //model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.query.bias"]     = layer.c_attn_q_proj_bias;
            //model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.key.weight"]     = layer.c_attn_k_proj_w;
            //model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.key.bias"]       = layer.c_attn_k_proj_bias;
            //model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.value.weight"]   = layer.c_attn_v_proj_w;
            //model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.value.bias"]     = layer.c_attn_v_proj_bias;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.weight"]   = layer.c_attn_proj_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".attention.dense.bias"]     = layer.c_attn_proj_bias;
            // FF norm
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.weight"] = layer.post_attention_layernorm_weight;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".post_attention_layernorm.bias"]   = layer.post_attention_layernorm_bias;
            // Feedforward
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.weight"]    = layer.c_mlp_fc_w;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_h_to_4h.bias"]      = layer.c_mlp_fc_b;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.weight"]    = layer.c_mlp_proj_w_trans;
            model.tensors["gpt_neox.layers." + std::to_string(i) + ".mlp.dense_4h_to_h.bias"]      = layer.c_mlp_proj_b;
        }
    }

    // old cformers version
    // key + value memory
    /*{
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        model.memory_k = ggml_new_tensor_1d(ctx, memory_type, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, memory_type, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }*/
    
    if (progress_callback) {
        progress_callback(0.0, progress_callback_user_data);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            if (0) {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            size_t bpe = 0;

            switch (ftype) {
                case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                default:
                        {
                            fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                            return false;
                        }
            };

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }
        
        // progress
        /*
        if (progress_callback) {
            float current_file_progress = float(size_t(fin.tellg()) - file_offset) / float(file_size - file_offset);
            float current_progress = (float(i) + current_file_progress) / float(n_parts);
            progress_callback(current_progress, progress_callback_user_data);
        }
         */

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, n_tensors);
    }

    fin.close();
    
    lctx.t_load_us = ggml_time_us() - t_start_us;

    if (progress_callback) {
        progress_callback(1.0, progress_callback_user_data);
    }

    return true;
}


// Temp targetted debug print
/*
bool debugPrintEnabled = false;
static inline int dbg_printf(const char * __restrict p) {
    if(!debugPrintEnabled) return 0;
    return printf(p);
}
static inline int dbg_printf(const char * __restrict p, int d) {
    if(!debugPrintEnabled) return 0;
    return printf(p, d);
}
static inline void dbg_print_tensor(ggml_tensor * t, const char * n) {
    if(!debugPrintEnabled) return;
    printf("%s %s\n", n, t->type == 0 ? "f32" : t->type == 1 ? "f16" : t->type == 2 ? "q4_0" : "unused");
    printf("%s ne [%d, %d, %d, %d]\n", n, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    printf("%s nb [%d, %d, %d, %d]\n", n, t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
}
 */

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-NeoX model requires about 16MB of memory per input token.
// Keep this is close to the llama.cpp impl for simplicity sake
bool gptneox_eval_internal(
        gptneox_context & lctx,
        const gptneox_token * tokens,
        const int   n_tokens,
        const int   n_past,
        const int   n_threads) {
    const int64_t t_start_us = ggml_time_us();
    
    const int N = n_tokens; //embd_inp.size();

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
              
    auto & kv_self = model.kv_self;

    GPTNEOX_ASSERT(!!kv_self.ctx);

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;
    //const int d_key = n_embd/n_head;
              
    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size(),
        /*.mem_buffer =*/ buf_compute.data(),
        //.mem_size   = buf_size,
        //.mem_buffer = buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
              
    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    ggml_cgraph gf = {};
    gf.n_threads = N >= 32 && ggml_cpu_has_blas() ? 1 : n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, tokens /*embd_inp.data()*/, N*ggml_element_size(embd));

    // wte
    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.wte, embd);
    //print_tensor(inpL, "  inpL");

    // for (int il = 0; il < 1; ++il) {
    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;
        
        lctx.use_buf(ctx0, 0);

        // input norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = input_layernorm_weight*cur + input_layernorm_bias
            cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].input_layernorm_weight, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].input_layernorm_bias, cur));
        }
        
        // self-attention
        {
            // attn
            // [3*n_embd, n_embd] - model.layers[il].c_attn_attn_w
            // [3*n_embd,      1] - model.layers[il].c_attn_attn_b
            // [  n_embd,      N] - cur (in)
            // [3*n_embd,      N] - cur (out)
            //
            // cur = attn_w*cur + attn_b
            // [3*n_embd, N]
            {
                cur = ggml_mul_mat(ctx0, model.layers[il].c_attn_qkv_proj_w, cur);
                cur = ggml_add(ctx0,
                        ggml_repeat(ctx0,
                                    model.layers[il].c_attn_qkv_proj_bias, cur),
                        cur);
            }
             
            // Split QKV and make contiguous
            struct ggml_tensor * Qcur = ggml_view_3d(ctx0, cur,
                                            n_embd/n_head,
                                            n_head,
                                            N,
                                            ggml_element_size(cur) * 3 * n_embd/n_head,
                                            ggml_element_size(cur) * 3 * n_embd,
                                            ggml_element_size(cur) * n_embd/n_head * 0);
            struct ggml_tensor * Kcur = ggml_view_3d(ctx0, cur,
                                            n_embd/n_head,
                                            n_head,
                                            N,
                                            ggml_element_size(cur) * 3 * n_embd/n_head,
                                            ggml_element_size(cur) * 3 * n_embd,
                                            ggml_element_size(cur) * n_embd/n_head * 1);
            struct ggml_tensor * Vcur = ggml_view_3d(ctx0, cur,
                                            n_embd/n_head,
                                            n_head,
                                            N,
                                            ggml_element_size(cur) * 3 * n_embd/n_head,
                                            ggml_element_size(cur) * 3 * n_embd,
                                            ggml_element_size(cur) * n_embd/n_head * 2);
            Qcur = ggml_cpy(ctx0, Qcur,
                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N));
            Kcur = ggml_cpy(ctx0, Kcur,
                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N));
            Vcur = ggml_cpy(ctx0, Vcur,
                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N));
            
            // MARK: gptneox RoPE Q and K, before cache
            // Bit 2 for gptneox style (2)
            // Bit 1 is zero for dont skip n_past +(0), use (2+1) = (3) if rope is applied to cache of k (after cache only)
            Qcur = ggml_rope(ctx0, Qcur, n_past, n_rot, 2);
            Kcur = ggml_rope(ctx0, Kcur, n_past, n_rot, 2); //3);

            // store key and value to memory, not required if prompt if only a single token (not practical or likely)
            //if (N >= 1) {
                // Each entry in kv_self has byte size of (ggml_element_size * n_embd * n_ctx * n_layer)
                Vcur = ggml_view_2d(ctx0, Vcur,
                            n_embd,
                            N,
                            ggml_element_size(Vcur) * n_embd,
                            0);
                Vcur = ggml_transpose(ctx0, Vcur);
            
                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k,
                                            n_embd * N, // num elements in current context (up to n_embd*n_ctx but usually less)
                                            ggml_element_size(kv_self.k) * n_embd * (il * n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v,
                                            N,
                                            n_embd,
                                            ggml_element_size(kv_self.v) * n_ctx,
                                            ggml_element_size(kv_self.v) * ((il * n_ctx * n_embd) + n_past));
            
                // important: storing RoPE-ed version of K in the KV cache!
                // TODO: Is this same for neox rope? Or should rope happen after cache? Seems correct based on orig impl
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            //}
            
            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kv_self.k,
                                (n_past + N) * n_embd,
                                ggml_element_size(kv_self.k) * il * n_ctx * n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // K * Q
            // Will use internally ggml_compute_forward_mul_mat_f16_f32 because K is f16 (cache) and Q is f32 (from q4_0)
            // Outputs [N, N, H, B], so it seems like this is correct for "scores"
            // K is internally transposed by ggml_mul_mat
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0, KQ,
                                                ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head)));
            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
            
            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans = ggml_view_3d(ctx0, kv_self.v,
                                                n_past + N,
                                                n_embd/n_head,
                                                n_head,
                                                ggml_element_size(kv_self.v) * n_ctx,
                                                ggml_element_size(kv_self.v) * n_ctx * n_embd/n_head,
                                                ggml_element_size(kv_self.v) * il * n_ctx * n_embd);

            // KQV = transpose(V) * KQ_soft_max
            // MARK: V input must be pre-transposed because ggml_mul_mat will transpose it again, we ultimately want KQ*V not KQ*V_T
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, model.layers[il].c_attn_proj_w, cur);

            // projection (then bias)
            cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_attn_proj_bias, cur), cur);
        }
        
        lctx.use_buf(ctx0, 1);

        if (hparams.use_parallel_residual == 1) {
            //printf("use_parallel_residual == 1\n");
            
            // This is independent of the self-attention result, so it could be done in parallel to the self-attention
            struct ggml_tensor * outAttn = cur;

            // post attention layer norm
            {
                cur = ggml_norm(ctx0, inpL);

                // cur = input_layernorm_weight*inpFF + input_layernorm_bias
                cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_weight, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_bias, cur));
            }


            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                cur = ggml_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, cur);

                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur), cur);

                // GELU activation
                cur = ggml_gelu(ctx0, cur);

                // projection
                // cur = proj_w*inpFF + proj_b
                cur = ggml_mul_mat(ctx0, model.layers[il].c_mlp_proj_w_trans, cur);

                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur), cur);
            }
            //# pseudocode:
            //# x = x + attn(ln1(x)) + mlp(ln2(x))
            // inpL = inpL + outAttn + cur
            cur = ggml_add(ctx0, outAttn, cur);
            inpL = ggml_add(ctx0, inpL, cur);
        } else if (hparams.use_parallel_residual == 0) {
            //printf("use_parallel_residual == 0\n");
            
            // This takes the self-attention residual output as input to Feedforward
            struct ggml_tensor * outAttn = cur;
            struct ggml_tensor * inpFF = ggml_add(ctx0, outAttn, inpL);

            // post attention layer norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // inpFF = input_layernorm_weight*inpFF + input_layernorm_bias
                cur = ggml_add(ctx0,
                    ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_weight, cur),
                        cur),
                    ggml_repeat(ctx0, model.layers[il].post_attention_layernorm_bias, cur));
            }

            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                cur = ggml_mul_mat(ctx0, model.layers[il].c_mlp_fc_w, cur);

                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur), cur);

                cur = ggml_gelu(ctx0, cur);

                cur = ggml_mul_mat(ctx0, model.layers[il].c_mlp_proj_w_trans, cur);

                cur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur), cur);
            }

            //# pseudocode:
            //# x = x + attn(ln1(x)) (residual above as input to mlp)
            //# x = x + mlp(ln2(x)) (residual after mlp aka inpL + cur)
            inpL = ggml_add(ctx0, inpL, cur);
        } else {
            printf("use_parallel_residual == %d\n", hparams.use_parallel_residual);
            assert(0);
        }
    }
              
    // used at the end to optionally extract the embeddings
    struct ggml_tensor * embeddings = NULL;

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.ln_f_g, inpL),
                    inpL),
                ggml_repeat(ctx0, model.ln_f_b, inpL));
        
        embeddings = inpL;
    }

    // lm_head
    inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);
              
    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    //embd_w.resize(n_vocab);
    //memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
    
    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }
              
    // extract embeddings
    if (lctx.embedding.size()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
              
#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

    ggml_free(ctx0);
    
    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (N > 1) {
        lctx.t_p_eval_us += ggml_time_us() - t_start_us;
        lctx.n_p_eval += N;
    }

    return true;
}


//
// sampling
//

static void sample_top_k(std::vector<std::pair<float, gptneox_vocab::id>> & logits_id, int top_k) {
    // find the top k tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<float, gptneox_vocab::id> & a, const std::pair<float, gptneox_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

static gptneox_vocab::id gptneox_sample_top_p_top_k(
        gptneox_context & lctx,
        const std::vector<gptneox_vocab::id> & last_n_tokens,
        int top_k,
        float top_p,
        float temp,
        float repeat_penalty) {
    auto & rng = lctx.rng;

    const int n_logits = lctx.model.hparams.n_vocab;

    const auto & logits = lctx.logits;
    const auto * plogits = logits.data() + logits.size() - n_logits;

    std::vector<std::pair<float, gptneox_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    float maxl = -std::numeric_limits<float>::infinity();
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top k tokens
    std::vector<float> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        const float p = expf(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0) {
        double cumsum = 0.0;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) 10; i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //printf("\n\n");
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}


//
// quantization
//

// quantize a model
static void gptneox_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, enum gptneox_ftype ftype, int nthread) {

    ggml_type quantized_type;
    switch (ftype) {
        case GPTNEOX_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case GPTNEOX_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        //case GPTNEOX_FTYPE_MOSTLY_Q4_2: quantized_type = GGML_TYPE_Q4_2; break;
        //case GPTNEOX_FTYPE_MOSTLY_Q4_3: quantized_type = GGML_TYPE_Q4_3; break;
        default: throw format("invalid output file type %d\n", ftype);
    };

    gptneox_vocab vocab;

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        throw format("failed to open '%s' for reading\n", fname_inp.c_str());
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        throw format("failed to open '%s' for writing\n", fname_out.c_str());
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char *) &magic, sizeof(magic));
        if (magic == GPTNEOX_FILE_MAGIC_UNVERSIONED) {
            throw format("invalid model file '%s' (too old, regenerate your model files!)\n", fname_inp.c_str());
        }
        if (magic != GPTNEOX_FILE_MAGIC) {
            throw format("invalid model file '%s' (bad magic)\n", fname_inp.c_str());
        }

        fout.write((char *) &magic, sizeof(magic));

        uint32_t format_version;
        finp.read((char *) &format_version, sizeof(format_version));

        if (format_version != GPTNEOX_FILE_VERSION) {
            throw format("invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n", fname_inp.c_str(), format_version, GPTNEOX_FILE_VERSION);
        }

        fout.write((char *) &format_version, sizeof(format_version));
    }

    gptneox_hparams hparams;

    // load hparams
    {
        finp.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //finp.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        finp.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        finp.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        finp.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        finp.read((char *) &hparams.n_rot,  sizeof(hparams.n_rot));
        finp.read((char *) &hparams.use_parallel_residual, sizeof(hparams.use_parallel_residual));
        finp.read((char *) &hparams.f16,     sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        // printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: use_parallel_residual = %d\n", __func__, hparams.use_parallel_residual);
        printf("%s: f16     = %d\n", __func__, hparams.f16);

        fout.write((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        //fout.write((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fout.write((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fout.write((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fout.write((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fout.write((char *) &hparams.n_rot,  sizeof(hparams.n_rot));
        fout.write((char *) &hparams.use_parallel_residual, sizeof(hparams.use_parallel_residual));
        fout.write((char *) &ftype,           sizeof(hparams.f16));
    }

    // load vocab
    {
        const int32_t n_vocab = hparams.n_vocab;

        if (n_vocab != hparams.n_vocab) {
            throw format("invalid model file '%s' (bad vocab size %d != %d)\n", fname_inp.c_str(), n_vocab, hparams.n_vocab);
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read ((char *) &len, sizeof(len));
            fout.write((char *) &len, sizeof(len));

            word.resize(len);
            finp.read ((char *) word.data(), len);
            fout.write((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // load weights
    {
        size_t total_size_org = 0;
        size_t total_size_new = 0;

        std::vector<float> work;

        std::vector<uint8_t>     data_u8;
        std::vector<ggml_fp16_t> data_f16;
        std::vector<float>       data_f32;

        std::vector<int64_t> hist_all(1 << 4, 0);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            finp.read(reinterpret_cast<char *>(&length), sizeof(length));
            finp.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

            if (finp.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            finp.read (&name[0], length);

            {
                static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                printf("%48s - [%5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ftype_str[ftype]);
            }

            // regexes of tensor names to be quantized
            const std::vector<std::string> k_names = {
                ".*weight",
            };

            bool quantize = false;
            for (const auto & s : k_names) {
                if (std::regex_match(name, std::regex(s))) {
                    quantize = true;
                    break;
                }
            }

            // quantize only 2D tensors
            quantize &= (n_dims == 2);

            if (quantize) {
                if (ftype != 0 && ftype != 1) {
                    throw format("unsupported ftype %d for integer quantization\n", ftype);
                }

                if (ftype == 1) {
                    data_f16.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                    data_f32.resize(nelements);
                    for (int i = 0; i < nelements; ++i) {
                        data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                    }
                    // DEBUG
                    /*
                    // if name is "gpt_neox.layers.0.attention.query.weight", then print first 10 float values.
                    if (name.find("gpt_neox.layers.0.attention") != std::string::npos) {
                        // if query.weight or key.weight or value.weight, print first 10 float values.
                        if (name.find("query.") != std::string::npos ||
                            name.find("key.") != std::string::npos ||
                            name.find("value.") != std::string::npos) {
                            printf("\n\nfirst 10 values: ");
                            for (int i = 0; i < 10; ++i) {
                                printf("%f ", data_f32[i]);
                            }
                            printf("\n");
                        }
                    }
                     */
                } else {
                    data_f32.resize(nelements);
                    finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
                }

                ftype = quantized_type; //itype;
            } else {
                const int bpe = (ftype == 0) ? sizeof(float) : sizeof(uint16_t);

                data_u8.resize(nelements*bpe);
                finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
            }

            // DEBUG
            /*
            {
                // if name is "gpt_neox.layers.0.attention.query.weight", then print first 10 float values.
                if (name.find("gpt_neox.layers.0.attention") != std::string::npos) {
                    // if query.weight or key.weight or value.weight, print first 10 float values.
                    if (name.find("query.bias") != std::string::npos ||
                        name.find("key.bias") != std::string::npos ||
                        name.find("value.bias") != std::string::npos) {
                        printf("\n\nfirst 10 values %s: ", name.data());
                        // combine four consecutive uint8_t to one float
                        for (int i = 0; i < 10; ++i) {
                            uint8_t *p = &data_u8[i*4];
                            float f = *(float *)p;
                            printf("%f ", f);
                        }
                        printf("\n");
                    }
                }
            }
             */

            fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fout.write(reinterpret_cast<char *>(&length), sizeof(length));
            fout.write(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
            for (int i = 0; i < n_dims; ++i) {
                fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            }
            fout.write(&name[0], length);

            if (quantize) {
                printf("quantizing .. ");
                work.resize(nelements); // for quantization

                size_t cur_size = 0;
                std::vector<int64_t> hist_cur(1 << 4, 0);

                switch (quantized_type) {
                    case GGML_TYPE_Q4_0:
                        {
                            cur_size = ggml_quantize_q4_0(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        } break;
                    case GGML_TYPE_Q4_1:
                        {
                            cur_size = ggml_quantize_q4_1(data_f32.data(), work.data(), nelements, ne[0], hist_cur.data());
                        } break;
                    default:
                        {
                            throw format("unsupported quantization type %d\n", quantized_type);
                        }
                }
                // // if name is "gpt_neox.layers.0.attention.query.weight", then print first 32 quantized values from work.data()
                // if (name.find("layers.0.attention.query.weight") != std::string::npos) {
                //     printf("\n\n\n");
                //     // first value is a fp32 scale followed by 32 "int4" values.
                //     printf("scale = %f\n", work[0]);
                //     void *p = &work[1];
                //     // Since int4_t is not defined, we use int8_t to print the values two at a time, offset by 4 bits.
                //     int8_t *p8 = (int8_t *)p;
                //     for (int i = 0; i < 16; i++) {
                //         // print first 4 bits
                //         printf("%d ", (p8[i] >> 4) & 0xf);
                //         // print last 4 bits
                //         printf("%d ", p8[i] & 0xf);
                //     }
                //     printf("\n\n\n");
                // }

                fout.write(reinterpret_cast<char *>(work.data()), cur_size);
                total_size_new += cur_size;

                printf("size = %8.2f MB -> %8.2f MB | hist: ", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
                for (int i = 0; i < hist_cur.size(); ++i) {
                    hist_all[i] += hist_cur[i];
                }

                for (int i = 0; i < hist_cur.size(); ++i) {
                    printf("%5.3f ", hist_cur[i] / (float)nelements);
                }
                printf("\n");
            } else {
                printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
                fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
                total_size_new += data_u8.size();
            }

            total_size_org += nelements * sizeof(float);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
        printf("%s: quant size  = %8.2f MB\n", __func__, total_size_new/1024.0/1024.0);

        {
            int64_t sum_all = 0;
            for (int i = 0; i < hist_all.size(); ++i) {
                sum_all += hist_all[i];
            }

            printf("%s: hist: ", __func__);
            for (int i = 0; i < hist_all.size(); ++i) {
                printf("%5.3f ", hist_all[i] / (float)sum_all);
            }
            printf("\n");
        }
    }

    finp.close();
    fout.close();
}



/*
int main_gptneox(gptneox_params params) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();
    int64_t t_load_us = 0;

    std::mt19937 rng(params.seed);

    gptneox_vocab vocab;
    gptneox_model model;
    // load the model
    {
        const int64_t t_start_us = ggml_time_us();
        const int n_ctx = 512; // TODO: set context from user input ??
        if (!gptneox_model_load(params.model, model, vocab, n_ctx)) {  // TODO: set context from user input ??
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gptneox_vocab::id> embd_inp = ::whitespace_tokenize(params.prompt); //TODO: set bos to true?

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("\n");
    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    // for (int i = 0; i < (int) embd_inp.size(); i++) {
    //     printf("%6d -> '%s'\n", embd_inp[i], vocab.id_to_token.at(embd_inp[i]).c_str());
    // }
    printf("\n");
    printf("sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    printf("\n\n");

    std::vector<gptneox_vocab::id> embd;

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gptneox_eval(model, params.n_threads, 0, { 1, 2, 3, 4, 5 }, logits, mem_per_token);

    int last_n_size = params.repeat_last_n;
    std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    printf("\n<|BEGIN> ");
    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gptneox_eval(model, params.n_threads, n_past, embd, logits, mem_per_token)) { // update logits
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const float top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;
            const float repeat_penalty = params.repeat_penalty;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            if (params.return_logits) {
                printf("logits: ");
                for (int i = 0; i < n_vocab; i++) {
                    // Upto 8 decimal places
                    printf("%.8f ", logits[i]);
                }
                printf(" <END|>\n");
                // Stdout should flush before returning
                fflush(stdout);
                return 0;
            }

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = sample_top_p_top_k_repeat_penalty(
                        vocab,
                        logits.data() + (logits.size() - n_vocab),
                        last_n_tokens,
                        repeat_penalty,
                        top_k,
                        top_p,
                        temp,
                        rng);

                // // print
                // printf("\ngenerated token: '%s' (%d)\n", vocab.id_to_token[id].c_str(), id);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                if (params.return_logits) {
                    printf("logits: ");
                    for (int i = 0; i < model.hparams.n_vocab; i++) {
                        // Upto 8 decimal places
                        printf("%.8f ", logits[i]);
                    }
                    printf("\n");
                }
                embd.push_back(embd_inp[k]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            if (!params.return_logits) {
                printf(" %d ", id);
            }
            // printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 2) {
            break;
        }
    }
    printf(" <END|>\n");

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
 */



//
// interface implementation
//

struct gptneox_context * gptneox_init_from_file(
                             const char * path_model,
            struct gptneox_context_params   params) {
    ggml_time_init();

    gptneox_context * ctx = new gptneox_context;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!gptneox_model_load(path_model, *ctx, params.n_ctx, params.n_parts, memory_type,
                          params.vocab_only, params.progress_callback,
                          params.progress_callback_user_data)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        gptneox_free(ctx);
        return nullptr;
    }

    // TODO: Update this file to match newest llama.cpp, hopefully the newest ggml.c can support most of gptneox
    /*
    if (params.use_mlock) {
        char *err;
        if (!ggml_mlock(ctx->model.ctx, &err)) {
            fprintf(stderr, "%s\n", err);
            free(err);
            gptneox_free(ctx);
            return nullptr;
        }
    }
     */

    // reserve memory for context buffers
    {
        if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->model.hparams.n_ctx)) {
            fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
            gptneox_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->model.kv_self.k) + ggml_nbytes(ctx->model.kv_self.v);
            fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }
        
        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_ctx);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        ctx->buf_compute.resize(MEM_REQ_EVAL); //.at(ctx->model.type));

        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0); //.at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1); //.at(ctx->model.type));
    }

    return ctx;
}

void gptneox_free(struct gptneox_context * ctx) {
    delete ctx;
}

int gptneox_model_quantize(
        const char * fname_inp,
        const char * fname_out,
enum gptneox_ftype   ftype,
        int          nthread) {
    try {
        gptneox_model_quantize_internal(fname_inp, fname_out, ftype, nthread);
        return 0;
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
        return 1;
    }
}

int gptneox_eval(
        struct gptneox_context * ctx,
           const gptneox_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!gptneox_eval_internal(*ctx, tokens, n_tokens, n_past, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }

    return 0;
}

//
// tokenizer
//

void replace(std::string & str, const std::string & needle, const std::string & replacement) {
    size_t pos = 0;
    while ((pos = str.find(needle, pos)) != std::string::npos) {
        str.replace(pos, needle.length(), replacement);
        pos += replacement.length();
    }
}

// from cformers/utils.cpp
std::map<std::string, int32_t> json_parse(const std::string & fname) {
    std::map<std::string, int32_t> result;

    // read file into string
    std::string json;
    {
        std::ifstream ifs(fname);
        if (!ifs) {
            fprintf(stderr, "Failed to open %s\n", fname.c_str());
            exit(1);
        }

        json = std::string((std::istreambuf_iterator<char>(ifs)),
                (std::istreambuf_iterator<char>()));
    }

    if (json[0] != '{') {
        return result;
    }

    // parse json
    {
        bool has_key  = false;
        bool in_token = false;

        std::string str_key = "";
        std::string str_val = "";

        int n = json.size();
        for (int i = 1; i < n; ++i) {
            if (!in_token) {
                if (json[i] == ' ') continue;
                if (json[i] == '"') {
                    in_token = true;
                    continue;
                }
            } else {
                if (json[i] == '\\' && i+1 < n) {
                    if (has_key == false) {
                        str_key += json[i];
                    } else {
                        str_val += json[i];
                    }
                    ++i;
                } else if (json[i] == '"') {
                    if (has_key == false) {
                        has_key = true;
                        ++i;
                        while (json[i] == ' ') ++i;
                        ++i; // :
                        while (json[i] == ' ') ++i;
                        if (json[i] != '\"') {
                            while (json[i] != ',' && json[i] != '}') {
                                str_val += json[i++];
                            }
                            has_key = false;
                        } else {
                            in_token = true;
                            continue;
                        }
                    } else {
                        has_key = false;
                    }

                    ::replace(str_key, "\\u0120", " " ); // \u0120 -> space
                    ::replace(str_key, "\\u010a", "\n"); // \u010a -> new line
                    ::replace(str_key, "\\\"",    "\""); // \\\"   -> "

                    try {
                        result[str_key] = std::stoi(str_val);
                    } catch (...) {
                        //fprintf(stderr, "%s: ignoring key '%s' with value '%s'\n", fname.c_str(), str_key.c_str(), str_val.c_str());

                    }
                    str_key = "";
                    str_val = "";
                    in_token = false;
                    continue;
                }
                if (has_key == false) {
                    str_key += json[i];
                } else {
                    str_val += json[i];
                }
            }
        }
    }

    return result;
}

// from cformers/utils.cpp gpt_vocab_init
bool gptneox_vocab_init(const std::string & fname, gptneox_vocab & vocab) {
    printf("%s: loading vocab from '%s'\n", __func__, fname.c_str());

    vocab.token_to_id = ::json_parse(fname);

    for (const auto & kv : vocab.token_to_id) {
        vocab.id_to_token[kv.second] = kv.first;
    }

    printf("%s: vocab size = %d\n", __func__, (int) vocab.token_to_id.size());

    // print the vocabulary
    //for (auto kv : vocab.token_to_id) {
    //    printf("'%s' -> %d\n", kv.first.data(), kv.second);
    //}

    return true;
}

// from cformers/utils.cpp gpt_tokenize
static std::vector<gptneox_vocab::id> gptneox_tokenize(const gptneox_vocab & vocab, const std::string & text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;
        std::string pat = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";

        std::regex re(pat);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }

    // find the longest tokens that form the words:
    std::vector<gptneox_vocab::id> tokens;
    for (const auto & word : words) {
        if (word.size() == 0) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            while (j > i) {
                auto it = vocab.token_to_id.find(word.substr(i, j-i));
                if (it != vocab.token_to_id.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    break;
                }
                --j;
            }
            if (i == n) {
                break;
            }
            if (j == i) {
                auto sub = word.substr(i, 1);
                if (vocab.token_to_id.find(sub) != vocab.token_to_id.end()) {
                    tokens.push_back(vocab.token_to_id.at(sub));
                } else {
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                }
                ++i;
            }
        }
    }

    return tokens;
}


int gptneox_tokenize(
        struct gptneox_context * ctx,
                  const char * text,
                     gptneox_token * tokens,
                     int   n_max_tokens) { //,
                        //bool   add_bos) {
    auto res = gptneox_tokenize(ctx->vocab, text); //, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int gptneox_n_vocab(struct gptneox_context * ctx) {
    return ctx->vocab.id_to_token.size();
}

int gptneox_n_ctx(struct gptneox_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int gptneox_n_embd(struct gptneox_context * ctx) {
    return ctx->model.hparams.n_embd;
}

float * gptneox_get_logits(struct gptneox_context * ctx) {
    return ctx->logits.data();
}

float * gptneox_get_embeddings(struct gptneox_context * ctx) {
    return ctx->embedding.data();
}

const char * gptneox_token_to_str(struct gptneox_context * ctx, gptneox_token token) {
    if (token >= gptneox_n_vocab(ctx)) {
        return nullptr;
    }

    return ctx->vocab.id_to_token[token]./*tok.*/c_str();
}

gptneox_token gptneox_token_bos() {
    return 0;
}

gptneox_token gptneox_token_eos() {
    return 0;
}

gptneox_token gptneox_sample_top_p_top_k(
        gptneox_context * ctx,
      const gptneox_token * last_n_tokens_data,
                    int   last_n_tokens_size,
                    int   top_k,
                  float   top_p,
                  float   temp,
                  float   repeat_penalty) {
    const int64_t t_start_sample_us = ggml_time_us();

    gptneox_token result = 0;

    // TODO: avoid this ...
    const auto last_n_tokens = std::vector<gptneox_token>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_size);

    result = gptneox_sample_top_p_top_k(
            *ctx,
            last_n_tokens,
            top_k,
            top_p,
            temp,
            repeat_penalty);

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;

    return result;
}

void gptneox_print_timings(struct gptneox_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    const int32_t n_sample = std::max(1, ctx->n_sample);
    const int32_t n_eval   = std::max(1, ctx->n_eval);
    const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
    fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
    fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
    fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_eval_us,   n_eval,   1e-3 * ctx->t_eval_us   / n_eval);
    fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0);
}

void gptneox_reset_timings(struct gptneox_context * ctx) {
    ctx->t_start_us = ggml_time_us();

    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * gptneox_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}

