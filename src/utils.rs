// When writing application code, you rarely care what the error is, just that one happened.
// I could also just return Option everywhere, but I like this better.
pub type Res<T> = Result<T, ()>;
