use const_cstr;
use sbr;

use std::os::raw::c_char;

pub mod messages {
    const_cstr! {
        pub FITTING_FAILED = "Failure fitting model";
        pub BAD_PREDICTION = "Invalid prediction value: NaN or +/- inifinity";
        pub BAD_REPRESENTATION = "Unable to compute user representation.";
        pub BAD_SERIALIZATION = "Unable to serialize model.";
        pub SERIALIZATION_TOO_SMALL = "Not enough space allocated for serialization";
        pub BAD_DESERIALIZATION = "Unable to deserialize model.";
    }
}

pub trait Opaque<T> {
    unsafe fn into_box(self) -> Box<T>;
}

macro_rules! ffi_result {
    ($name:ident, $type:ty,opaque_name = $opaque_name:ident) => {
        /// Opaque struct for the underlying object.
        #[repr(C)]
        pub struct $opaque_name {}

        /// Calling `into_box` will create a box of the underlying
        /// type from a pointer to its opaque wrapper.
        impl Opaque<$type> for *mut $opaque_name {
            unsafe fn into_box(self) -> Box<$type> {
                Box::from_raw(self as *mut $type)
            }
        }

        /// Result type for $type.
        ///
        /// One of `value`, `error` is always set; it's null otherwise.
        /// The error string should never be freed; the value object
        /// should be freed with the appropriate `free` function.
        #[repr(C)]
        #[derive(Clone, Debug)]
        pub struct $name {
            value: *mut $opaque_name,
            /// Do not free attempt to free the error string.
            error: *const c_char,
        }

        impl From<Result<$type, const_cstr::ConstCStr>> for $name {
            fn from(x: Result<$type, const_cstr::ConstCStr>) -> Self {
                match x {
                    Ok(val) => Self {
                        value: Box::into_raw(Box::new(val)) as *mut $opaque_name,
                        error: ::std::ptr::null::<c_char>(),
                    },
                    Err(err) => Self {
                        value: ::std::ptr::null::<$type>() as *mut $opaque_name,
                        error: err.as_ptr(),
                    },
                }
            }
        }
    };
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
            /// Do not free attempt to free the error string.
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
    sbr::models::lstm::ImplicitLSTMModel,
    opaque_name = ImplicitLSTMModelPointer
);
ffi_result!(
    InteractionsResult,
    sbr::data::Interactions,
    opaque_name = InteractionsPointer
);
ffi_result!(FloatResult, f32);
