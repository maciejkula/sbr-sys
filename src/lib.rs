#![deny(missing_docs)]
//! C bindings for [Sbr](https://github.com/maciejkula/sbr-rs).
#[macro_use]
extern crate const_cstr;
#[macro_use]
extern crate itertools;
extern crate bincode;
extern crate libc;
extern crate sbr;

mod ffi_results;

#[macro_use]
mod ffi_macros;

use std::os::raw::{c_char, c_uchar};

use sbr::OnlineRankingModel;

pub trait Opaque<T> {
    unsafe fn into_box(self) -> Box<T>;
}

/// Loss type.
#[repr(C)]
#[derive(Clone, Debug)]
pub enum Loss {
    /// Bayesian Personalised Ranking.
    BPR,
    /// Pairwise hinge loss.
    Hinge,
    /// Weighted Approximate Pairwise loss. This is likely
    /// to have the best accuracy at the expense of some speed.
    WARP,
}

/// Optimizer type.
#[repr(C)]
#[derive(Clone, Debug)]
pub enum Optimizer {
    /// Adagrad.
    Adagrad,
    /// Adam.
    Adam,
}

/// FFI-compatible object for building hyperparameters
/// for `sbr::models::lstm::ImplictLSTMModel`.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct LSTMHyperparameters {
    /// Number of items in the dataset.
    num_items: libc::size_t,
    /// Maximum sequence lenght to consider when
    /// computing a user representation.
    max_sequence_length: libc::size_t,
    /// Internal embedding dimensionality.
    item_embedding_dim: libc::size_t,
    /// Initial learning rate.
    learning_rate: f32,
    /// L2 penalty.
    l2_penalty: f32,
    /// Coupled: should we use coupled forget and update gates. 0 for false,
    /// 1 for true.
    coupled: libc::size_t,
    /// Loss: one of 'hinge', 'bpr', 'warp'.
    loss: Loss,
    /// Optimizer: one of 'adagrad', 'adam'.
    optimizer: Optimizer,
    /// Number of threads to use when fitting.
    num_threads: libc::size_t,
    /// Number of epochs to run.
    num_epochs: libc::size_t,
    /// Random seed to use.
    random_seed: [c_uchar; 16],
}

impl LSTMHyperparameters {
    /// Convert to the actual hyperparameters object.
    unsafe fn convert(&self) -> Result<sbr::models::lstm::Hyperparameters, const_cstr::ConstCStr> {
        let optimizer = match self.optimizer {
            Optimizer::Adam => Ok(sbr::models::Optimizer::Adam),
            Optimizer::Adagrad => Ok(sbr::models::Optimizer::Adagrad),
        }?;

        let loss = match self.loss {
            Loss::BPR => Ok(sbr::models::Loss::BPR),
            Loss::Hinge => Ok(sbr::models::Loss::Hinge),
            Loss::WARP => Ok(sbr::models::Loss::WARP),
        }?;

        Ok(
            sbr::models::lstm::Hyperparameters::new(self.num_items, self.max_sequence_length)
                .learning_rate(self.learning_rate)
                .embedding_dim(self.item_embedding_dim)
                .l2_penalty(self.l2_penalty)
                .num_epochs(self.num_epochs)
                .num_threads(self.num_threads)
                .lstm_variant(if self.coupled == 1 {
                    sbr::models::lstm::LSTMVariant::Coupled
                } else {
                    sbr::models::lstm::LSTMVariant::Normal
                })
                .parallelism(sbr::models::Parallelism::Synchronous)
                .optimizer(optimizer)
                .loss(loss)
                .from_seed(self.random_seed),
        )
    }
}

impl_model!(
    hyperparameters = LSTMHyperparameters,
    model = sbr::models::lstm::ImplicitLSTMModel,
    new_name = implicit_lstm_new,
    free_name = implicit_lstm_free,
    fit_name = implicit_lstm_fit,
    predict_name = implicit_lstm_predict,
    result_name = ImplicitLSTMModelResult,
    opaque_name = ImplicitLSTMModelPointer,
);

ffi_result!(
    InteractionsResult,
    sbr::data::Interactions,
    InteractionsPointer
);
ffi_result!(FloatResult, f32);

/// Create an interaction dataset from input arrays.
#[no_mangle]
pub extern "C" fn interactions_new(
    num_users: libc::size_t,
    num_items: libc::size_t,
    len: libc::size_t,
    users: *const libc::size_t,
    items: *const libc::size_t,
    timestamps: *const libc::size_t,
) -> InteractionsResult {
    let (users, items, timestamps) = unsafe {
        (
            std::slice::from_raw_parts(users, len),
            std::slice::from_raw_parts(items, len),
            std::slice::from_raw_parts(timestamps, len),
        )
    };

    let mut interactions = sbr::data::Interactions::new(num_users, num_items);

    izip!(users.iter(), items.iter(), timestamps.iter())
        .map(|(&uid, &iid, &time)| sbr::data::Interaction::new(uid, iid, time))
        .for_each(|interaction| interactions.push(interaction));

    Ok(interactions).into()
}

free!(interactions_free, ffi_results::InteractionsPointer);
