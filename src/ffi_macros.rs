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

macro_rules! ffi_result {
    ($name:ident, $type:ty, $opaque_name:ident) => {
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
            /// Do not attempt to free the error string.
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
            /// Do not attempt to free the error string.
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

macro_rules! impl_model {
    (
        hyperparameters =
        $hyperparameters:ty,model =
        $model:ty,new_name =
        $new_name:ident,free_name =
        $free_name:ident,fit_name =
        $fit_name:ident,predict_name =
        $predict_name:ident,get_serialized_size_name =
        $get_serialized_size_name:ident,serialize_name =
        $serialize_name:ident,deserialize_name =
        $deserialize_name:ident,mrr_score_name =
        $mrr_score_name:ident,result_name =
        $result_name:ident,opaque_name =
        $opaque_name:ident,
    ) => {
        ffi_result!($result_name, $model, $opaque_name);

        /// Build a new model from hyperparameters.
        /// The caller owns the returned objects and should free
        /// it with the corresponding `free` function.
        #[no_mangle]
        pub extern "C" fn $new_name(hyperparameters: $hyperparameters) -> $result_name {
            unsafe { hyperparameters.convert().map(|hyper| hyper.build()).into() }
        }

        /// Fit the model.
        #[no_mangle]
        pub extern "C" fn $fit_name(
            model: *mut $opaque_name,
            data: *const InteractionsPointer,
        ) -> FloatResult {
            let result = unsafe {
                (*(model as *mut $model))
                    .fit(&(*(data as *const sbr::data::Interactions)).to_compressed())
            };

            result
                .map_err(|_| ffi_results::errors::FITTING_FAILED)
                .into()
        }

        /// Get predictions out of the model.
        ///
        /// The returned string is non-null if an error occurred.
        /// It must not be freed.
        #[no_mangle]
        pub extern "C" fn $predict_name(
            model: *mut $opaque_name,
            user_history: *const libc::size_t,
            history_len: libc::size_t,
            item_ids: *const libc::size_t,
            out: *mut f32,
            predictions_len: libc::size_t,
        ) -> *const c_char {
            let (model, history, item_ids, out): (
                &$model,
                &[libc::size_t],
                &[libc::size_t],
                &mut [f32],
            ) = unsafe {
                (
                    &(*(model as *mut $model)),
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
        pub extern "C" fn $get_serialized_size_name(model: *mut $opaque_name) -> libc::size_t {
            let model = unsafe { &(*(model as *mut $model)) };
            bincode::serialized_size(model).expect("Unable to get serialized size") as usize
        }

        /// Serialize the model to the provided pointer.
        ///
        /// Returns an error message if there was an error.
        #[no_mangle]
        pub extern "C" fn $serialize_name(
            model: *mut $opaque_name,
            out: *mut c_uchar,
            len: libc::size_t,
        ) -> *const c_char {
            let model = unsafe { &(*(model as *mut $model)) };
            let out = unsafe { std::slice::from_raw_parts_mut(out, len) };

            if len
                < bincode::serialized_size(model).expect("Unable to get serialized size") as usize
            {
                return ffi_results::errors::SERIALIZATION_TOO_SMALL.as_ptr();
            }

            if let Ok(_) = bincode::serialize_into(out, model) {
                ::std::ptr::null::<c_char>()
            } else {
                ffi_results::errors::BAD_SERIALIZATION.as_ptr()
            }
        }

        /// Deserialize the model from a byte array.
        #[no_mangle]
        pub extern "C" fn $deserialize_name(data: *mut c_uchar, len: libc::size_t) -> $result_name {
            let data = unsafe { std::slice::from_raw_parts_mut(data, len) };

            bincode::deserialize::<$model>(data)
                .map_err(|_| ffi_results::errors::BAD_DESERIALIZATION)
                .into()
        }

        /// Compute MRR score for a fitted model.
        #[no_mangle]
        pub extern "C" fn $mrr_score_name(
            model: *const $opaque_name,
            data: *const InteractionsPointer,
        ) -> FloatResult {
            let result = unsafe {
                sbr::evaluation::mrr_score(
                    &(*(model as *const $model)),
                    &(*(data as *const sbr::data::Interactions)).to_compressed(),
                )
            };

            result
                .map_err(|e| match e {
                    sbr::PredictionError::InvalidPredictionValue => {
                        ffi_results::errors::BAD_PREDICTION
                    }
                })
                .into()
        }

        free!($free_name, $opaque_name);
    };
}
