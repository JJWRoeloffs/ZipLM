// When writing application courn Option everywhere, but I like this better.
pub type Res<T> = Result<T, ()>;

// The most simple type of corpus there could be, just a list of sentences.
#[derive(Debug, Default)]
pub struct Corpus<T: Default> {
    pub items: T,
}
