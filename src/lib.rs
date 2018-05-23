#[macro_use]
extern crate const_cstr;
#[macro_use]
extern crate itertools;
extern crate libc;
extern crate sbr;

mod errors;

use std::ffi;
use std::os::raw::{c_char, c_uchar};

/// Loss type.
#[repr(C)]
#[derive(Clone, Debug)]
pub enum Loss {
    BPR,
    Hinge,
}

/// Optimizer type.
#[repr(C)]
#[derive(Clone, Debug)]
pub enum Optimizer {
    Adagrad,
    Adam,
}

/// FFI-compatible object for building hyperparameters
/// for `sbr::models::lstm::ImplictLSTMModel`.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct LSTMHyperparameters {
    /// Number of items in the dataset.
    num_items: u64,
    /// Maximum sequence lenght to consider when
    /// computing a user representation.
    max_sequence_length: u64,
    /// Internal embedding dimensionality.
    item_embedding_dim: u64,
    /// Initial learning rate.
    learning_rate: f32,
    /// L2 penalty.
    l2_penalty: f32,
    /// Loss: one of 'hinge', 'bpr'.
    loss: Loss,
    /// Optimizer: one of 'adagrad', 'adam'.
    optimizer: Optimizer,
    /// Number of threads to use when fitting.
    num_threads: u64,
    /// Number of epochs to run.
    num_epochs: u64,
    /// Random seed to use.
    random_seed: [c_uchar; 16],
}

impl LSTMHyperparameters {
    /// Convert to the actual hyperparameters object.
    unsafe fn convert(&self) -> Result<sbr::models::lstm::Hyperparameters, const_cstr::ConstCStr> {
        let optimizer = match self.optimizer {
            Optimizer::Adam => Ok(sbr::models::lstm::Optimizer::Adam),
            Optimizer::Adagrad => Ok(sbr::models::lstm::Optimizer::Adagrad),
        }?;

        let loss = match self.loss {
            Loss::BPR => Ok(sbr::models::lstm::Loss::BPR),
            Loss::Hinge => Ok(sbr::models::lstm::Loss::Hinge),
        }?;

        Ok(sbr::models::lstm::Hyperparameters::new(
            self.num_items as usize,
            self.max_sequence_length as usize,
        ).learning_rate(self.learning_rate)
            .embedding_dim(self.item_embedding_dim as usize)
            .l2_penalty(self.l2_penalty)
            .num_epochs(self.num_epochs as usize)
            .num_threads(self.num_threads as usize)
            .optimizer(optimizer)
            .loss(loss)
            .from_seed(self.random_seed))
    }
}

/// Build a new implicit LSTM model from hyperparameters.
/// The caller owns the returned objects and should free
/// it with [implicit_lstm_free].
#[no_mangle]
pub extern "C" fn implicit_lstm_new(
    hyperparameters: LSTMHyperparameters,
) -> errors::ImplicitLSTMModelResult {
    unsafe { hyperparameters.convert().map(|hyper| hyper.build()).into() }
}

macro_rules! free {
    ($name:ident, $type:ty) => {
        /// Free the data behind the input pointer.
        #[no_mangle]
        pub extern "C" fn $name(model: *mut $type) {
            unsafe {
                Box::from_raw(model);
            }
        }
    };
}

free!(implicit_lstm_free, errors::ImplicitLSTMModelPointer);

/// Create an interaction dataset from input arrays.
#[no_mangle]
pub extern "C" fn interactions_new(
    num_users: libc::size_t,
    num_items: libc::size_t,
    len: libc::size_t,
    users: *const libc::int32_t,
    items: *const libc::int32_t,
    timestamps: *const libc::int32_t,
) -> errors::InteractionsResult {
    let (users, items, timestamps) = unsafe {
        (
            std::slice::from_raw_parts(users, len),
            std::slice::from_raw_parts(items, len),
            std::slice::from_raw_parts(timestamps, len),
        )
    };

    let mut interactions = sbr::data::Interactions::new(num_users, num_items);

    izip!(users.iter(), items.iter(), timestamps.iter())
        .map(|(&uid, &iid, &time)| {
            sbr::data::Interaction::new(uid as usize, iid as usize, time as usize)
        })
        .for_each(|interaction| interactions.push(interaction));

    Ok(interactions).into()
}

free!(interactions_free, errors::InteractionsPointer);

/// Fit an ImplicitLSTMModel.
#[no_mangle]
pub extern "C" fn implicit_lstm_fit(
    model: *mut errors::ImplicitLSTMModelPointer,
    data: *const errors::InteractionsPointer,
) -> errors::FloatResult {
    let result = unsafe {
        (*(model as *mut sbr::models::lstm::ImplicitLSTMModel))
            .fit(&(*(data as *const sbr::data::Interactions)).to_compressed())
    };

    result.map_err(|_| errors::messages::FITTING_FAILED).into()
}

/// Compute MRR score for a fitted model.
#[no_mangle]
pub extern "C" fn implicit_lstm_mrr_score(
    model: *const errors::ImplicitLSTMModelPointer,
    data: *const errors::InteractionsPointer,
) -> errors::FloatResult {
    let result = unsafe {
        sbr::evaluation::mrr_score(
            &(*(model as *const sbr::models::lstm::ImplicitLSTMModel)),
            &(*(data as *const sbr::data::Interactions)).to_compressed(),
        )
    };

    result
        .map_err(|e| match e {
            sbr::PredictionError::InvalidPredictionValue => errors::messages::BAD_PREDICTION,
        })
        .into()
}

free!(float_free, f32);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
