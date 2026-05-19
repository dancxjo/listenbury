#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OrthographicWord {
    pub text: String,
}

impl OrthographicWord {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}
