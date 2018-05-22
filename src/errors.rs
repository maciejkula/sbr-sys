use const_cstr;
use sbr;

use std::os::raw::c_char;

pub mod messages {
    const_cstr! {
        pub OPTIMIZER_BAD_PARSE = "Unable to parse optimizer";
        pub LOSS_BAD_PARSE = "Unable to parse loss";
        pub OPTIMIZER_NOT_RECOGNIZED = "Optimizer not recognized";
        pub LOSS_NOT_RECOGNIZED = "Loss not recognized";
        pub FITTING_FAILED = "Failure fitting model";
        pub BAD_PREDICTION = "Invalid prediction value: NaN or +/- inifinity";
    }
}

macro_rules! ffi_result {
    ($name:ident, $type:ty) => {
        /// Result type for $type.
        ///
        /// One of `value`, `error` is always set; it's null otherwise.
        /// The error string should never be freed; the value object
        /// should be freed with the appropriate `free` function.
        #[repr(C)]
        #[derive(Clone, Debug)]
        pub struct $name {
            value: *mut $type,
            error: *const c_char,
        }

        impl From<Result<$type, const_cstr::ConstCStr>> for $name {
            fn from(x: Result<$type, const_cstr::ConstCStr>) -> Self {
                match x {
                    Ok(val) => Self {
                        value: Box::into_raw(Box::new(val)),
                        error: ::std::ptr::null::<c_char>(),
                    },
                    Err(err) => Self {
                        value: ::std::ptr::null::<$type>() as *mut $type,
                        error: err.as_ptr(),
                    },
                }
            }
        }
    };
}

ffi_result!(
    ImplicitLSTMModelResult,
    sbr::models::lstm::ImplicitLSTMModel
);
ffi_result!(InteractionsResult, sbr::data::Interactions);
ffi_result!(FloatResult, f32);
