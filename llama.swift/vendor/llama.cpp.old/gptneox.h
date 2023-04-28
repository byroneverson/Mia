#ifndef gptneox_h
#define gptneox_h

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef GPTNEOX_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GPTNEOX_BUILD
#            define GPTNEOX_API __declspec(dllexport)
#        else
#            define GPTNEOX_API __declspec(dllimport)
#        endif
#    else
#        define GPTNEOX_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define GPTNEOX_API
#endif

#define GPTNEOX_FILE_VERSION 0x00000001
#define GPTNEOX_FILE_MAGIC 0x67676d66 // 'ggmf' in hex
#define GPTNEOX_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

#ifdef __cplusplus
extern "C" {
#endif

//
// C interface
//
// TODO: show sample usage
//

struct gptneox_context;

typedef int gptneox_token;

typedef struct gptneox_token_data {
    gptneox_token id;  // token id

    float p;     // probability of the token
    float plog;  // log probability of the token

} gptneox_token_data;

typedef void (*gptneox_progress_callback)(float progress, void *ctx);

struct gptneox_context_params {
    int n_ctx;   // text context
    int n_parts; // -1 for default
    int seed;    // RNG seed, 0 for random

    bool f16_kv;     // use fp16 for KV cache
    bool logits_all; // the gptneox_eval() call computes all logits, not just the last one
    bool vocab_only; // only load the vocabulary, no weights
    bool use_mlock;  // force system to keep model in RAM
    bool embedding;  // embedding mode only

    // called with a progress value between 0 and 1, pass NULL to disable
    gptneox_progress_callback progress_callback;
    // context pointer passed to the progress callback
    void * progress_callback_user_data;
};

// model file types
enum gptneox_ftype {
    GPTNEOX_FTYPE_ALL_F32     = 0,
    GPTNEOX_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
    GPTNEOX_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    GPTNEOX_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
    //GPTNEOX_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
    //GPTNEOX_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
    //GPTNEOX_FTYPE_MOSTLY_Q4_3 = 6,  // except 1d tensors
};

GPTNEOX_API struct gptneox_context_params gptneox_context_default_params();

// Various functions for loading a ggml gptneox model.
// Allocate (almost) all memory needed for the model.
// Return NULL on failure
GPTNEOX_API struct gptneox_context * gptneox_init_from_file(
                         const char * path_model,
        struct gptneox_context_params   params);

// Frees all allocated memory
GPTNEOX_API void gptneox_free(struct gptneox_context * ctx);

// TODO: not great API - very likely to change
// nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
GPTNEOX_API int gptneox_model_quantize(
        const char * fname_inp,
        const char * fname_out,
  enum gptneox_ftype   ftype,
        int          nthread);

// Run the gptneox inference to obtain the logits and probabilities for the next token.
// tokens + n_tokens is the provided batch of new tokens to process
// n_past is the number of tokens to use from previous eval calls
// Returns 0 on success
GPTNEOX_API int gptneox_eval(
        struct gptneox_context * ctx,
           const gptneox_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads);

// Convert the provided text into tokens.
// The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success, no more than n_max_tokens
// Returns a negative number on failure - the number of tokens that would have been returned
// TODO: not sure if correct
GPTNEOX_API int gptneox_tokenize(
        struct gptneox_context * ctx,
                  const char * text,
                                 gptneox_token * tokens,
                                 int   n_max_tokens); //,
                        //bool   add_bos);

GPTNEOX_API int gptneox_n_vocab(struct gptneox_context * ctx);
GPTNEOX_API int gptneox_n_ctx  (struct gptneox_context * ctx);
GPTNEOX_API int gptneox_n_embd (struct gptneox_context * ctx);

// Token logits obtained from the last call to gptneox_eval()
// The logits for the last token are stored in the last row
// Can be mutated in order to change the probabilities of the next token
// Rows: n_tokens
// Cols: n_vocab
GPTNEOX_API float * gptneox_get_logits(struct gptneox_context * ctx);

// Get the embeddings for the input
// shape: [n_embd] (1-dimensional)
GPTNEOX_API float * gptneox_get_embeddings(struct gptneox_context * ctx);

// Token Id -> String. Uses the vocabulary in the provided context
GPTNEOX_API const char * gptneox_token_to_str(struct gptneox_context * ctx, gptneox_token token);

// Special tokens
GPTNEOX_API gptneox_token gptneox_token_bos();
GPTNEOX_API gptneox_token gptneox_token_eos();

// TODO: improve the last_n_tokens interface ?
GPTNEOX_API gptneox_token gptneox_sample_top_p_top_k(
   struct gptneox_context * ctx,
      const gptneox_token * last_n_tokens_data,
                    int   last_n_tokens_size,
                    int   top_k,
                  float   top_p,
                  float   temp,
                  float   repeat_penalty);

// Performance information
GPTNEOX_API void gptneox_print_timings(struct gptneox_context * ctx);
GPTNEOX_API void gptneox_reset_timings(struct gptneox_context * ctx);

// Print system information
GPTNEOX_API const char * gptneox_print_system_info(void);

#ifdef __cplusplus
}
#endif


#endif /* gptneox_h */
