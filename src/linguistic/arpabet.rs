use crate::linguistic::inventory::{
    FeatureBundle, MajorClass, Manner, Place, Roundedness, Voicing, VowelBackness, VowelHeight,
};
use crate::linguistic::phone::{Phone, PhoneStatus, PhoneString, Stress};
use crate::linguistic::phoneme::Phoneme;
use crate::linguistic::realization::{Realization, RealizationMethod};

pub fn phoneme_from_arpabet(symbol: &str, source: &str) -> Phoneme {
    let (base, stress) = split_arpabet_symbol(symbol);
    let default_phone_string = default_phone_string_for_arpabet(&base, symbol);
    let ipa = default_phone_string.to_ipa();
    let features = feature_bundle_for_arpabet(&base);
    Phoneme {
        symbol: base,
        source_symbol: symbol.to_string(),
        source: source.to_string(),
        stress,
        features,
        default_phone_string: default_phone_string.clone(),
        realization: Realization {
            phone_string: default_phone_string.clone(),
            ipa,
            method: RealizationMethod::Default,
            rule: None,
            environment: None,
            environment_match: None,
        },
    }
}
pub fn feature_bundle_for_arpabet(base: &str) -> FeatureBundle {
    match base {
        "AA" => vowel_features(
            VowelHeight::Low,
            VowelBackness::Back,
            Roundedness::Unrounded,
        ),
        "AE" | "AY" => vowel_features(
            VowelHeight::Low,
            VowelBackness::Front,
            Roundedness::Unrounded,
        ),
        "AH" => vowel_features(
            VowelHeight::Mid,
            VowelBackness::Central,
            Roundedness::Unrounded,
        ),
        "AO" => vowel_features(
            VowelHeight::Low,
            VowelBackness::Back,
            Roundedness::Rounded,
        ),
        "AW" => vowel_features(
            VowelHeight::Low,
            VowelBackness::Central,
            Roundedness::Unrounded,
        ),
        "EH" | "EY" => vowel_features(
            VowelHeight::Mid,
            VowelBackness::Front,
            Roundedness::Unrounded,
        ),
        "ER" => vowel_features(
            VowelHeight::Rhotic,
            VowelBackness::Central,
            Roundedness::Unrounded,
        ),
        "IH" | "IY" => vowel_features(
            VowelHeight::High,
            VowelBackness::Front,
            Roundedness::Unrounded,
        ),
        "OW" => vowel_features(
            VowelHeight::Mid,
            VowelBackness::Back,
            Roundedness::Rounded,
        ),
        "OY" => vowel_features(
            VowelHeight::Mid,
            VowelBackness::Back,
            Roundedness::Rounded,
        ),
        "UH" | "UW" => vowel_features(
            VowelHeight::High,
            VowelBackness::Back,
            Roundedness::Rounded,
        ),
        "P" => consonant_features(Place::Bilabial, Manner::Stop, Voicing::Voiceless),
        "B" => consonant_features(Place::Bilabial, Manner::Stop, Voicing::Voiced),
        "M" => consonant_features(Place::Bilabial, Manner::Nasal, Voicing::Voiced),
        "F" => consonant_features(Place::Labiodental, Manner::Fricative, Voicing::Voiceless),
        "V" => consonant_features(Place::Labiodental, Manner::Fricative, Voicing::Voiced),
        "TH" => consonant_features(Place::Dental, Manner::Fricative, Voicing::Voiceless),
        "DH" => consonant_features(Place::Dental, Manner::Fricative, Voicing::Voiced),
        "T" => consonant_features(Place::Alveolar, Manner::Stop, Voicing::Voiceless),
        "D" => consonant_features(Place::Alveolar, Manner::Stop, Voicing::Voiced),
        "N" => consonant_features(Place::Alveolar, Manner::Nasal, Voicing::Voiced),
        "S" => consonant_features(Place::Alveolar, Manner::Fricative, Voicing::Voiceless),
        "Z" => consonant_features(Place::Alveolar, Manner::Fricative, Voicing::Voiced),
        "L" | "R" => consonant_features(Place::Alveolar, Manner::Liquid, Voicing::Voiced),
        "CH" => consonant_features(Place::Postalveolar, Manner::Affricate, Voicing::Voiceless),
        "JH" => consonant_features(Place::Postalveolar, Manner::Affricate, Voicing::Voiced),
        "SH" => consonant_features(Place::Postalveolar, Manner::Fricative, Voicing::Voiceless),
        "ZH" => consonant_features(Place::Postalveolar, Manner::Fricative, Voicing::Voiced),
        "Y" => consonant_features(Place::Palatal, Manner::Glide, Voicing::Voiced),
        "K" => consonant_features(Place::Velar, Manner::Stop, Voicing::Voiceless),
        "G" => consonant_features(Place::Velar, Manner::Stop, Voicing::Voiced),
        "NG" => consonant_features(Place::Velar, Manner::Nasal, Voicing::Voiced),
        "W" => consonant_features(Place::Velar, Manner::Glide, Voicing::Voiced),
        "HH" => consonant_features(Place::Glottal, Manner::Fricative, Voicing::Voiceless),
        _ => FeatureBundle {
            major: MajorClass::Consonant,
            place: None,
            vowel_height: None,
            vowel_backness: None,
            roundedness: None,
            manner: None,
            voicing: None,
            syllabic: false,
        },
    }
}

fn vowel_features(
    height: VowelHeight,
    backness: VowelBackness,
    roundedness: Roundedness,
) -> FeatureBundle {
    FeatureBundle {
        major: MajorClass::Vowel,
        place: None,
        vowel_height: Some(height),
        vowel_backness: Some(backness),
        roundedness: Some(roundedness),
        manner: Some(Manner::Vowel),
        voicing: Some(Voicing::Voiced),
        syllabic: true,
    }
}

fn consonant_features(place: Place, manner: Manner, voicing: Voicing) -> FeatureBundle {
    FeatureBundle {
        major: MajorClass::Consonant,
        place: Some(place),
        vowel_height: None,
        vowel_backness: None,
        roundedness: None,
        manner: Some(manner),
        voicing: Some(voicing),
        syllabic: false,
    }
}
pub(crate) fn split_arpabet_symbol(symbol: &str) -> (String, Option<Stress>) {
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

pub(crate) fn default_phone_string_for_arpabet(base: &str, source_symbol: &str) -> PhoneString {
    let mapped: Option<&[&str]> = match base {
        "AA" => Some(&["ɑ"]),
        "AE" => Some(&["æ"]),
        "AH" => Some(&["ʌ"]),
        "AO" => Some(&["ɔ"]),
        "AW" => Some(&["a", "ʊ"]),
        "AY" => Some(&["a", "ɪ"]),
        "B" => Some(&["b"]),
        "CH" => Some(&["tʃ"]),
        "D" => Some(&["d"]),
        "DH" => Some(&["ð"]),
        "EH" => Some(&["ɛ"]),
        "ER" => Some(&["ɝ"]),
        "EY" => Some(&["e", "ɪ"]),
        "F" => Some(&["f"]),
        "G" => Some(&["ɡ"]),
        "HH" => Some(&["h"]),
        "IH" => Some(&["ɪ"]),
        "IY" => Some(&["iː"]),
        "JH" => Some(&["dʒ"]),
        "K" => Some(&["k"]),
        "L" => Some(&["l"]),
        "M" => Some(&["m"]),
        "N" => Some(&["n"]),
        "NG" => Some(&["ŋ"]),
        "OW" => Some(&["o", "ʊ"]),
        "OY" => Some(&["ɔ", "ɪ"]),
        "P" => Some(&["p"]),
        "R" => Some(&["ɹ"]),
        "S" => Some(&["s"]),
        "SH" => Some(&["ʃ"]),
        "T" => Some(&["t"]),
        "TH" => Some(&["θ"]),
        "UH" => Some(&["ʊ"]),
        "UW" => Some(&["uː"]),
        "V" => Some(&["v"]),
        "W" => Some(&["w"]),
        "Y" => Some(&["j"]),
        "Z" => Some(&["z"]),
        "ZH" => Some(&["ʒ"]),
        _ => None,
    };
    match mapped {
        Some(ipa_segments) => PhoneString {
            phones: ipa_segments
                .iter()
                .map(|ipa| Phone {
                    ipa: (*ipa).to_string(),
                    source_symbol: Some(source_symbol.to_string()),
                    status: PhoneStatus::Mapped,
                })
                .collect(),
        },
        None => PhoneString {
            phones: vec![Phone {
                ipa: format!("?{base}"),
                source_symbol: Some(source_symbol.to_string()),
                status: PhoneStatus::UnknownSymbol,
            }],
        },
    }
}
