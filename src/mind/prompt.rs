#[derive(Debug, Clone)]
pub struct PromptBuilder {
    pub system: String,
}

impl PromptBuilder {
    pub fn build(&self, user_text: &str) -> String {
        format!("{}\nUser: {user_text}", self.system)
    }
}
