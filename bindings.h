#include <cstdint>
#include <cstdlib>

// Result type for $type.
//
// One of `value`, `error` is always set; it's null otherwise.
// The error string should never be freed; the value object
// should be freed with the appropriate `free` function.
struct FloatResult {
  float *value;
  const char *error;
};

// Result type for $type.
//
// One of `value`, `error` is always set; it's null otherwise.
// The error string should never be freed; the value object
// should be freed with the appropriate `free` function.
struct ImplicitLSTMModelResult {
  ImplicitLSTMModel *value;
  const char *error;
};

// FFI-compatible object for building hyperparameters
// for `sbr::models::lstm::ImplictLSTMModel`.
struct LSTMHyperparameters {
  // Number of items in the dataset.
  uint64_t num_items;
  // Maximum sequence lenght to consider when
  // computing a user representation.
  uint64_t max_sequence_length;
  // Internal embedding dimensionality.
  uint64_t item_embedding_dim;
  // Initial learning rate.
  float learning_rate;
  // L2 penalty.
  float l2_penalty;
  // Loss: one of 'hinge', 'bpr'.
  const char *loss;
  // Optimizer: one of 'adagrad', 'adam'.
  const char *optimizer;
  // Number of threads to use when fitting.
  uint64_t num_threads;
  // Number of epochs to run.
  uint64_t num_epochs;
  // Random seed to use.
  unsigned char random_seed[16];
};

// Result type for $type.
//
// One of `value`, `error` is always set; it's null otherwise.
// The error string should never be freed; the value object
// should be freed with the appropriate `free` function.
struct InteractionsResult {
  Interactions *value;
  const char *error;
};

extern "C" {

// Free the data behind the input pointer.
void float_free(float *model);

// Fit an ImplicitLSTMModel.
FloatResult implicit_lstm_fit(ImplicitLSTMModel *model, const Interactions *data);

// Free the data behind the input pointer.
void implicit_lstm_free(ImplicitLSTMModel *model);

// Compute MRR score for a fitted model.
FloatResult implicit_lstm_mrr_score(ImplicitLSTMModel *model, const Interactions *data);

// Build a new implicit LSTM model from hyperparameters.
// The caller owns the returned objects and should free
// it with [implicit_lstm_free].
ImplicitLSTMModelResult implicit_lstm_new(LSTMHyperparameters hyperparameters);

// Free the data behind the input pointer.
void interactions_free(Interactions *model);

// Create an interaction dataset from input arrays.
InteractionsResult interactions_new(size_t num_users,
                                    size_t num_items,
                                    size_t len,
                                    const int *users,
                                    const int *items,
                                    const int *timestamps);

} // extern "C"
