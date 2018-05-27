//! C bindings for [Sbr](https://github.com/maciejkula/sbr-rs).
#[macro_use]
extern crate const_cstr;
#[macro_use]
extern crate itertools;
extern crate bincode;
extern crate libc;
extern crate sbr;

mod ffi_results;

use std::os::raw::{c_char, c_uchar};

use sbr::OnlineRankingModel;

use ffi_results::Opaque;

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
    /// Loss: one of 'hinge', 'bpr'.
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
            Optimizer::Adam => Ok(sbr::models::lstm::Optimizer::Adam),
            Optimizer::Adagrad => Ok(sbr::models::lstm::Optimizer::Adagrad),
        }?;

        let loss = match self.loss {
            Loss::BPR => Ok(sbr::models::lstm::Loss::BPR),
            Loss::Hinge => Ok(sbr::models::lstm::Loss::Hinge),
        }?;

        Ok(
            sbr::models::lstm::Hyperparameters::new(self.num_items, self.max_sequence_length)
                .learning_rate(self.learning_rate)
                .embedding_dim(self.item_embedding_dim)
                .l2_penalty(self.l2_penalty)
                .num_epochs(self.num_epochs)
                .num_threads(self.num_threads)
                .optimizer(optimizer)
                .loss(loss)
                .from_seed(self.random_seed),
        )
    }
}

/// Build a new implicit LSTM model from hyperparameters.
/// The caller owns the returned objects and should free
/// it with [implicit_lstm_free].
#[no_mangle]
pub extern "C" fn implicit_lstm_new(
    hyperparameters: LSTMHyperparameters,
) -> ffi_results::ImplicitLSTMModelResult {
    unsafe { hyperparameters.convert().map(|hyper| hyper.build()).into() }
}

macro_rules! free {
    ($name:ident, $type:ty) => {
        /// Free the data behind the input pointer.
        #[no_mangle]
        pub extern "C" fn $name(x: *mut $type) {
            unsafe {
                x.into_box();
            }
        }
    };
}

free!(implicit_lstm_free, ffi_results::ImplicitLSTMModelPointer);

/// Create an interaction dataset from input arrays.
#[no_mangle]
pub extern "C" fn interactions_new(
    num_users: libc::size_t,
    num_items: libc::size_t,
    len: libc::size_t,
    users: *const libc::size_t,
    items: *const libc::size_t,
    timestamps: *const libc::size_t,
) -> ffi_results::InteractionsResult {
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

/// Fit an ImplicitLSTMModel.
#[no_mangle]
pub extern "C" fn implicit_lstm_fit(
    model: *mut ffi_results::ImplicitLSTMModelPointer,
    data: *const ffi_results::InteractionsPointer,
) -> ffi_results::FloatResult {
    let result = unsafe {
        (*(model as *mut sbr::models::lstm::ImplicitLSTMModel))
            .fit(&(*(data as *const sbr::data::Interactions)).to_compressed())
    };

    result
        .map_err(|_| ffi_results::errors::FITTING_FAILED)
        .into()
}

/// Get predictions out of an ImplicitLSTMModel.
///
/// The returned string is non-null if an error occurred.
/// It must not be freed.
#[no_mangle]
pub extern "C" fn implicit_lstm_predict(
    model: *mut ffi_results::ImplicitLSTMModelPointer,
    user_history: *const libc::size_t,
    history_len: libc::size_t,
    item_ids: *const libc::size_t,
    out: *mut f32,
    predictions_len: libc::size_t,
) -> *const c_char {
    let (model, history, item_ids, out): (
        &sbr::models::lstm::ImplicitLSTMModel,
        &[libc::size_t],
        &[libc::size_t],
        &mut [f32],
    ) = unsafe {
        (
            &(*(model as *mut sbr::models::lstm::ImplicitLSTMModel)),
            std::slice::from_raw_parts(user_history, history_len as usize),
            std::slice::from_raw_parts(item_ids, predictions_len as usize),
            std::slice::from_raw_parts_mut(out, predictions_len as usize),
        )
    };

    let user_repr = if let Ok(repr) = model.user_representation(&history) {
        repr
    } else {
        return ffi_results::errors::BAD_REPRESENTATION.as_ptr();
    };

    if let Ok(predictions) = model.predict(&user_repr, &item_ids) {
        for (&prediction, out_val) in predictions.iter().zip(out.iter_mut()) {
            *out_val = prediction;
        }

        ::std::ptr::null::<c_char>()
    } else {
        ffi_results::errors::BAD_PREDICTION.as_ptr()
    }
}

/// Get the size (in bytes) of the serialized model.
#[no_mangle]
pub extern "C" fn implicit_lstm_get_serialized_size(
    model: *mut ffi_results::ImplicitLSTMModelPointer,
) -> libc::size_t {
    let model = unsafe { &(*(model as *mut sbr::models::lstm::ImplicitLSTMModel)) };
    bincode::serialized_size(model).expect("Unable to get serialized size") as usize
}

/// Serialize the model to the provided pointer.
///
/// Returns an error message if there was an error.
#[no_mangle]
pub extern "C" fn implicit_lstm_serialize(
    model: *mut ffi_results::ImplicitLSTMModelPointer,
    out: *mut c_uchar,
    len: libc::size_t,
) -> *const c_char {
    let model = unsafe { &(*(model as *mut sbr::models::lstm::ImplicitLSTMModel)) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, len) };

    if len < bincode::serialized_size(model).expect("Unable to get serialized size") as usize {
        return ffi_results::errors::SERIALIZATION_TOO_SMALL.as_ptr();
    }

    if let Ok(_) = bincode::serialize_into(out, model) {
        ::std::ptr::null::<c_char>()
    } else {
        ffi_results::errors::BAD_SERIALIZATION.as_ptr()
    }
}

/// Deserialize the LSTM model from a byte array.
#[no_mangle]
pub extern "C" fn implicit_lstm_deserialize(
    data: *mut c_uchar,
    len: libc::size_t,
) -> ffi_results::ImplicitLSTMModelResult {
    let data = unsafe { std::slice::from_raw_parts_mut(data, len) };

    bincode::deserialize::<sbr::models::lstm::ImplicitLSTMModel>(data)
        .map_err(|_| ffi_results::errors::BAD_DESERIALIZATION)
        .into()
}

/// Compute MRR score for a fitted model.
#[no_mangle]
pub extern "C" fn implicit_lstm_mrr_score(
    model: *const ffi_results::ImplicitLSTMModelPointer,
    data: *const ffi_results::InteractionsPointer,
) -> ffi_results::FloatResult {
    let result = unsafe {
        sbr::evaluation::mrr_score(
            &(*(model as *const sbr::models::lstm::ImplicitLSTMModel)),
            &(*(data as *const sbr::data::Interactions)).to_compressed(),
        )
    };

    result
        .map_err(|e| match e {
            sbr::PredictionError::InvalidPredictionValue => ffi_results::errors::BAD_PREDICTION,
        })
        .into()
}
