/// Errors messages as static C strings.
pub mod errors {
    const_cstr! {
        pub FITTING_FAILED = "Failure fitting model";
        pub BAD_PREDICTION = "Invalid prediction value: NaN or +/- inifinity";
        pub BAD_REPRESENTATION = "Unable to compute user representation.";
        pub BAD_SERIALIZATION = "Unable to serialize model.";
        pub SERIALIZATION_TOO_SMALL = "Not enough space allocated for serialization";
        pub BAD_DESERIALIZATION = "Unable to deserialize model.";
    }
}
