use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stress {
    Primary,
    Secondary,
    Unstressed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Phone {
    pub ipa: String,
    pub source_symbol: Option<String>,
    pub status: PhoneStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhoneStatus {
    Mapped,
    UnknownSymbol,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoneString {
    pub phones: Vec<Phone>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WordPosition {
    Singleton,
    WordInitial,
    WordMedial,
    WordFinal,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Environment {
    pub left_phone: Option<String>,
    pub right_phone: Option<String>,
    pub left_class: Option<String>,
    pub right_class: Option<String>,
    pub word_position: Option<WordPosition>,
    pub syllable_position: Option<String>,
    pub stress_context: Option<String>,
    pub phrase_position: Option<String>,
    pub language: Option<String>,
    pub dialect: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AllophoneRule {
    pub id: String,
    pub applies_to_symbols: Vec<String>,
    pub output_ipa: String,
    pub environment_hint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealizationMethod {
    Default,
    AllophoneRule,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Realization {
    pub ipa: String,
    pub method: RealizationMethod,
    pub rule: Option<String>,
    pub environment: Option<Environment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Phoneme {
    pub symbol: String,
    pub source_symbol: String,
    pub source: String,
    pub stress: Option<Stress>,
    pub default_phone_string: PhoneString,
    pub realization: Realization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizationConfig {
    pub enable_allophone_rules: bool,
    pub language: String,
    pub dialect: String,
}

impl Default for RealizationConfig {
    fn default() -> Self {
        Self {
            enable_allophone_rules: false,
            language: "en".to_string(),
            dialect: "american_english".to_string(),
        }
    }
}

pub fn phoneme_from_arpabet(symbol: &str, source: &str) -> Phoneme {
    let (base, stress) = split_arpabet_symbol(symbol);
    let phone = default_phone_for_arpabet(&base, symbol);
    let default_phone_string = PhoneString {
        phones: vec![phone],
    };
    let ipa = default_phone_string.phones[0].ipa.clone();
    Phoneme {
        symbol: base,
        source_symbol: symbol.to_string(),
        source: source.to_string(),
        stress,
        default_phone_string,
        realization: Realization {
            ipa,
            method: RealizationMethod::Default,
            rule: None,
            environment: None,
        },
    }
}

pub fn realize_sequence(sequence: &[Phoneme], config: &RealizationConfig) -> Vec<Phoneme> {
    if !config.enable_allophone_rules {
        return sequence.to_vec();
    }
    let mut realized: Vec<Phoneme> = sequence.to_vec();
    for i in 1..realized.len().saturating_sub(1) {
        let cur = &realized[i];
        if !(cur.symbol == "T" || cur.symbol == "D") {
            continue;
        }
        let left = &realized[i - 1];
        let right = &realized[i + 1];
        let left_is_vowel = is_vowel_symbol(&left.symbol);
        let right_is_vowel = is_vowel_symbol(&right.symbol);
        let left_stressed = matches!(left.stress, Some(Stress::Primary | Stress::Secondary));
        let right_unstressed = matches!(right.stress, Some(Stress::Unstressed));
        if !(left_is_vowel && right_is_vowel && left_stressed && right_unstressed) {
            continue;
        }
        let env = Environment {
            left_phone: Some(left.realization.ipa.clone()),
            right_phone: Some(right.realization.ipa.clone()),
            left_class: Some("vowel".to_string()),
            right_class: Some("vowel".to_string()),
            word_position: Some(word_position(i, realized.len())),
            syllable_position: None,
            stress_context: Some("between stressed vowel and unstressed vowel".to_string()),
            phrase_position: None,
            language: Some(config.language.clone()),
            dialect: Some(config.dialect.clone()),
        };
        realized[i].realization = Realization {
            ipa: "ɾ".to_string(),
            method: RealizationMethod::AllophoneRule,
            rule: Some("american_english_intervocalic_flapping".to_string()),
            environment: Some(env),
        };
    }
    realized
}

fn word_position(index: usize, len: usize) -> WordPosition {
    if len <= 1 {
        WordPosition::Singleton
    } else if index == 0 {
        WordPosition::WordInitial
    } else if index == len - 1 {
        WordPosition::WordFinal
    } else {
        WordPosition::WordMedial
    }
}

fn split_arpabet_symbol(symbol: &str) -> (String, Option<Stress>) {
    match symbol.chars().last() {
        Some('1') => (
            symbol[..symbol.len() - 1].to_string(),
            Some(Stress::Primary),
        ),
        Some('2') => (
            symbol[..symbol.len() - 1].to_string(),
            Some(Stress::Secondary),
        ),
        Some('0') => (
            symbol[..symbol.len() - 1].to_string(),
            Some(Stress::Unstressed),
        ),
        _ => (symbol.to_string(), None),
    }
}

fn default_phone_for_arpabet(base: &str, source_symbol: &str) -> Phone {
    let mapped = match base {
        "AA" => Some("ɑ"),
        "AE" => Some("æ"),
        "AH" => Some("ʌ"),
        "AO" => Some("ɔ"),
        "AW" => Some("aʊ"),
        "AY" => Some("aɪ"),
        "B" => Some("b"),
        "CH" => Some("tʃ"),
        "D" => Some("d"),
        "DH" => Some("ð"),
        "EH" => Some("ɛ"),
        "ER" => Some("ɝ"),
        "EY" => Some("eɪ"),
        "F" => Some("f"),
        "G" => Some("ɡ"),
        "HH" => Some("h"),
        "IH" => Some("ɪ"),
        "IY" => Some("iː"),
        "JH" => Some("dʒ"),
        "K" => Some("k"),
        "L" => Some("l"),
        "M" => Some("m"),
        "N" => Some("n"),
        "NG" => Some("ŋ"),
        "OW" => Some("oʊ"),
        "OY" => Some("ɔɪ"),
        "P" => Some("p"),
        "R" => Some("ɹ"),
        "S" => Some("s"),
        "SH" => Some("ʃ"),
        "T" => Some("t"),
        "TH" => Some("θ"),
        "UH" => Some("ʊ"),
        "UW" => Some("uː"),
        "V" => Some("v"),
        "W" => Some("w"),
        "Y" => Some("j"),
        "Z" => Some("z"),
        "ZH" => Some("ʒ"),
        _ => None,
    };
    match mapped {
        Some(ipa) => Phone {
            ipa: ipa.to_string(),
            source_symbol: Some(source_symbol.to_string()),
            status: PhoneStatus::Mapped,
        },
        None => Phone {
            ipa: format!("?{base}"),
            source_symbol: Some(source_symbol.to_string()),
            status: PhoneStatus::UnknownSymbol,
        },
    }
}

fn is_vowel_symbol(base: &str) -> bool {
    matches!(
        base,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arpabet_to_ipa_mapping_preserves_stress_metadata() {
        let phoneme = phoneme_from_arpabet("IY1", "cmudict");
        assert_eq!(phoneme.symbol, "IY");
        assert_eq!(phoneme.source_symbol, "IY1");
        assert_eq!(phoneme.stress, Some(Stress::Primary));
        assert_eq!(phoneme.default_phone_string.phones[0].ipa, "iː");
        assert_eq!(phoneme.realization.ipa, "iː");
        assert_eq!(phoneme.realization.method, RealizationMethod::Default);
    }

    #[test]
    fn unknown_symbol_falls_back_safely() {
        let phoneme = phoneme_from_arpabet("QH9", "cmudict");
        assert_eq!(phoneme.symbol, "QH9");
        assert_eq!(phoneme.stress, None);
        assert_eq!(phoneme.default_phone_string.phones[0].ipa, "?QH9");
        assert_eq!(
            phoneme.default_phone_string.phones[0].status,
            PhoneStatus::UnknownSymbol
        );
    }

    #[test]
    fn opt_in_flapping_rule_realizes_t_between_stressed_and_unstressed_vowels() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "ɾ");
        assert_eq!(
            realized[1].realization.method,
            RealizationMethod::AllophoneRule
        );
        assert_eq!(
            realized[1].realization.rule.as_deref(),
            Some("american_english_intervocalic_flapping")
        );
        assert_eq!(
            realized[1]
                .realization
                .environment
                .as_ref()
                .and_then(|env| env.stress_context.as_deref()),
            Some("between stressed vowel and unstressed vowel")
        );
    }

    #[test]
    fn flapping_rule_requires_following_unstressed_vowel() {
        let seq = vec![
            phoneme_from_arpabet("AH0", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("IH2", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn allophone_rules_are_opt_in() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(&seq, &RealizationConfig::default());
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }
}
