#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

/*
 * Loss type.
 */
typedef enum {
  BPR,
  Hinge,
} Loss;

/*
 * Optimizer type.
 */
typedef enum {
  Adagrad,
  Adam,
} Optimizer;

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
   * Do not free attempt to free the error string.
   */
  const char *error;
} FloatResult;

/*
 * Opaque struct for the underlying object.
 */
typedef struct {

} ImplicitLSTMModelPointer;

/*
 * Opaque struct for the underlying object.
 */
typedef struct {

} InteractionsPointer;

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
   * Do not free attempt to free the error string.
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
  uint64_t num_items;
  /*
   * Maximum sequence lenght to consider when
   * computing a user representation.
   */
  uint64_t max_sequence_length;
  /*
   * Internal embedding dimensionality.
   */
  uint64_t item_embedding_dim;
  /*
   * Initial learning rate.
   */
  float learning_rate;
  /*
   * L2 penalty.
   */
  float l2_penalty;
  /*
   * Loss: one of 'hinge', 'bpr'.
   */
  Loss loss;
  /*
   * Optimizer: one of 'adagrad', 'adam'.
   */
  Optimizer optimizer;
  /*
   * Number of threads to use when fitting.
   */
  uint64_t num_threads;
  /*
   * Number of epochs to run.
   */
  uint64_t num_epochs;
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
   * Do not free attempt to free the error string.
   */
  const char *error;
} InteractionsResult;

/*
 * Free the data behind the input pointer.
 */
void float_free(float *model);

/*
 * Fit an ImplicitLSTMModel.
 */
FloatResult implicit_lstm_fit(ImplicitLSTMModelPointer *model, const InteractionsPointer *data);

/*
 * Free the data behind the input pointer.
 */
void implicit_lstm_free(ImplicitLSTMModelPointer *model);

/*
 * Compute MRR score for a fitted model.
 */
FloatResult implicit_lstm_mrr_score(const ImplicitLSTMModelPointer *model,
                                    const InteractionsPointer *data);

/*
 * Build a new implicit LSTM model from hyperparameters.
 * The caller owns the returned objects and should free
 * it with [implicit_lstm_free].
 */
ImplicitLSTMModelResult implicit_lstm_new(LSTMHyperparameters hyperparameters);

/*
 * Free the data behind the input pointer.
 */
void interactions_free(InteractionsPointer *model);

/*
 * Create an interaction dataset from input arrays.
 */
InteractionsResult interactions_new(size_t num_users,
                                    size_t num_items,
                                    size_t len,
                                    const int32_t *users,
                                    const int32_t *items,
                                    const int32_t *timestamps);
