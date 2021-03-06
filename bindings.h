#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * Loss type.
 */
typedef enum {
  /*
   * Bayesian Personalised Ranking.
   */
  BPR,
  /*
   * Pairwise hinge loss.
   */
  Hinge,
  /*
   * Weighted Approximate Pairwise loss. This is likely
   * to have the best accuracy at the expense of some speed.
   */
  WARP,
} Loss;

/*
 * Optimizer type.
 */
typedef enum {
  /*
   * Adagrad.
   */
  Adagrad,
  /*
   * Adam.
   */
  Adam,
} Optimizer;

/*
 * Opaque struct for the underlying object.
 */
typedef struct {

} ImplicitEWMAModelPointer;

/*
 * Result type for $type.
 *
 * One of `value`, `error` is always set; it's null otherwise.
 * The error string should never be freed; the value object
 * should be freed with the appropriate `free` function.
 */
typedef struct {
  ImplicitEWMAModelPointer *value;
  /*
   * Do not attempt to free the error string.
   */
  const char *error;
} ImplicitEWMAModelResult;

/*
 * Result type for $type.
 *
 * One of `value`, `error` is always set; it's null otherwise.
 * The error string should never be freed; the value object
 * should be freed with the appropriate `free` function.
 */
typedef struct {
  float *value;
  /*
   * Do not attempt to free the error string.
   */
  const char *error;
} FloatResult;

/*
 * Opaque struct for the underlying object.
 */
typedef struct {

} InteractionsPointer;

/*
 * FFI-compatible object for building hyperparameters
 * for `sbr::models::ewma::ImplictEWMAModel`.
 */
typedef struct {
  /*
   * Number of items in the dataset.
   */
  size_t num_items;
  /*
   * Maximum sequence length to consider when
   * computing a user representation.
   */
  size_t max_sequence_length;
  /*
   * Internal embedding dimensionality.
   */
  size_t item_embedding_dim;
  /*
   * Initial learning rate.
   */
  float learning_rate;
  /*
   * L2 penalty.
   */
  float l2_penalty;
  /*
   * Loss: one of 'hinge', 'bpr', 'warp'.
   */
  Loss loss;
  /*
   * Optimizer: one of 'adagrad', 'adam'.
   */
  Optimizer optimizer;
  /*
   * Number of threads to use when fitting.
   */
  size_t num_threads;
  /*
   * Number of epochs to run.
   */
  size_t num_epochs;
  /*
   * Random seed to use.
   */
  unsigned char random_seed[16];
} EWMAHyperparameters;

/*
 * Opaque struct for the underlying object.
 */
typedef struct {

} ImplicitLSTMModelPointer;

/*
 * Result type for $type.
 *
 * One of `value`, `error` is always set; it's null otherwise.
 * The error string should never be freed; the value object
 * should be freed with the appropriate `free` function.
 */
typedef struct {
  ImplicitLSTMModelPointer *value;
  /*
   * Do not attempt to free the error string.
   */
  const char *error;
} ImplicitLSTMModelResult;

/*
 * FFI-compatible object for building hyperparameters
 * for `sbr::models::lstm::ImplictLSTMModel`.
 */
typedef struct {
  /*
   * Number of items in the dataset.
   */
  size_t num_items;
  /*
   * Maximum sequence length to consider when
   * computing a user representation.
   */
  size_t max_sequence_length;
  /*
   * Internal embedding dimensionality.
   */
  size_t item_embedding_dim;
  /*
   * Initial learning rate.
   */
  float learning_rate;
  /*
   * L2 penalty.
   */
  float l2_penalty;
  /*
   * Coupled: should we use coupled forget and update gates. 0 for false,
   * 1 for true.
   */
  size_t coupled;
  /*
   * Loss: one of 'hinge', 'bpr', 'warp'.
   */
  Loss loss;
  /*
   * Optimizer: one of 'adagrad', 'adam'.
   */
  Optimizer optimizer;
  /*
   * Number of threads to use when fitting.
   */
  size_t num_threads;
  /*
   * Number of epochs to run.
   */
  size_t num_epochs;
  /*
   * Random seed to use.
   */
  unsigned char random_seed[16];
} LSTMHyperparameters;

/*
 * Result type for $type.
 *
 * One of `value`, `error` is always set; it's null otherwise.
 * The error string should never be freed; the value object
 * should be freed with the appropriate `free` function.
 */
typedef struct {
  InteractionsPointer *value;
  /*
   * Do not attempt to free the error string.
   */
  const char *error;
} InteractionsResult;

/*
 * Deserialize the model from a byte array.
 */
ImplicitEWMAModelResult implicit_ewma_deserialize(unsigned char *data, size_t len);

/*
 * Fit the model.
 */
FloatResult implicit_ewma_fit(ImplicitEWMAModelPointer *model, const InteractionsPointer *data);

/*
 * Free the data behind the input pointer.
 */
void implicit_ewma_free(ImplicitEWMAModelPointer *x);

/*
 * Get the size (in bytes) of the serialized model.
 */
size_t implicit_ewma_get_serialized_size(ImplicitEWMAModelPointer *model);

/*
 * Compute MRR score for a fitted model.
 */
FloatResult implicit_ewma_mrr_score(const ImplicitEWMAModelPointer *model,
                                    const InteractionsPointer *data);

/*
 * Build a new model from hyperparameters.
 * The caller owns the returned objects and should free
 * it with the corresponding `free` function.
 */
ImplicitEWMAModelResult implicit_ewma_new(EWMAHyperparameters hyperparameters);

/*
 * Get predictions out of the model.
 *
 * The returned string is non-null if an error occurred.
 * It must not be freed.
 */
const char *implicit_ewma_predict(ImplicitEWMAModelPointer *model,
                                  const size_t *user_history,
                                  size_t history_len,
                                  const size_t *item_ids,
                                  float *out,
                                  size_t predictions_len);

/*
 * Serialize the model to the provided pointer.
 *
 * Returns an error message if there was an error.
 */
const char *implicit_ewma_serialize(ImplicitEWMAModelPointer *model,
                                    unsigned char *out,
                                    size_t len);

/*
 * Deserialize the model from a byte array.
 */
ImplicitLSTMModelResult implicit_lstm_deserialize(unsigned char *data, size_t len);

/*
 * Fit the model.
 */
FloatResult implicit_lstm_fit(ImplicitLSTMModelPointer *model, const InteractionsPointer *data);

/*
 * Free the data behind the input pointer.
 */
void implicit_lstm_free(ImplicitLSTMModelPointer *x);

/*
 * Get the size (in bytes) of the serialized model.
 */
size_t implicit_lstm_get_serialized_size(ImplicitLSTMModelPointer *model);

/*
 * Compute MRR score for a fitted model.
 */
FloatResult implicit_lstm_mrr_score(const ImplicitLSTMModelPointer *model,
                                    const InteractionsPointer *data);

/*
 * Build a new model from hyperparameters.
 * The caller owns the returned objects and should free
 * it with the corresponding `free` function.
 */
ImplicitLSTMModelResult implicit_lstm_new(LSTMHyperparameters hyperparameters);

/*
 * Get predictions out of the model.
 *
 * The returned string is non-null if an error occurred.
 * It must not be freed.
 */
const char *implicit_lstm_predict(ImplicitLSTMModelPointer *model,
                                  const size_t *user_history,
                                  size_t history_len,
                                  const size_t *item_ids,
                                  float *out,
                                  size_t predictions_len);

/*
 * Serialize the model to the provided pointer.
 *
 * Returns an error message if there was an error.
 */
const char *implicit_lstm_serialize(ImplicitLSTMModelPointer *model,
                                    unsigned char *out,
                                    size_t len);

/*
 * Free the data behind the input pointer.
 */
void interactions_free(InteractionsPointer *x);

/*
 * Create an interaction dataset from input arrays.
 */
InteractionsResult interactions_new(size_t num_users,
                                    size_t num_items,
                                    size_t len,
                                    const size_t *users,
                                    const size_t *items,
                                    const size_t *timestamps);
