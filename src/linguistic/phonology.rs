//! Compatibility facade for phonology types split into focused modules.

pub use crate::linguistic::arpabet::{feature_bundle_for_arpabet, phoneme_from_arpabet};
pub use crate::linguistic::environment::*;
pub use crate::linguistic::inventory::*;
pub use crate::linguistic::phone::*;
pub use crate::linguistic::phoneme::Phoneme;
pub use crate::linguistic::realization::{
    AllophoneRule, Realization, RealizationConfig, RealizationMethod, realize_sequence,
    realize_sequence_as_schema,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::realization::phoneme_class_matches;

    #[test]
    fn arpabet_to_ipa_mapping_preserves_stress_metadata() {
        let phoneme = phoneme_from_arpabet("IY1", "cmudict");
        assert_eq!(phoneme.symbol, "IY");
        assert_eq!(phoneme.source_symbol, "IY1");
        assert_eq!(phoneme.stress, Some(Stress::Primary));
        assert_eq!(phoneme.default_phone_string.phones[0].ipa, "iː");
        assert_eq!(phoneme.realization.phone_string.to_ipa(), "iː");
        assert_eq!(phoneme.realization.ipa, "iː");
        assert_eq!(phoneme.realization.method, RealizationMethod::Default);
    }

    #[test]
    fn arpabet_multi_phone_defaults_are_structural() {
        let phoneme = phoneme_from_arpabet("CH", "cmudict");
        let segments = phoneme.default_phone_string.ipa_segments();

        assert_eq!(segments, vec!["tʃ"]);
        assert_eq!(phoneme.realization.phone_string.ipa_segments(), segments);
        assert_eq!(phoneme.realization.ipa, "tʃ");
        assert_eq!(
            PhoneString::from_realized(&phoneme).ipa_segments(),
            segments
        );
    }

    #[test]
    fn realized_phone_tokens_preserve_source_metadata() {
        let phonemes = vec![
            phoneme_from_arpabet("IY1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
        ];

        let realized = RealizedPhone::from_phoneme_slice(&phonemes);

        assert_eq!(realized.len(), 2);
        assert_eq!(realized[0].phone.ipa, "iː");
        assert_eq!(realized[0].source_phoneme_index, 0);
        assert_eq!(realized[0].source_symbol, "IY1");
        assert_eq!(realized[0].stress, Some(Stress::Primary));
        assert_eq!(realized[1].phone.ipa, "t");
        assert_eq!(realized[1].source_phoneme_index, 1);
        assert_eq!(realized[1].source_symbol, "T");
        assert_eq!(realized[1].stress, None);
    }

    #[test]
    fn realized_phone_tokens_expand_multi_phone_phonemes() {
        let phonemes = vec![phoneme_from_arpabet("CH", "cmudict")];

        let realized = RealizedPhone::from_phoneme_slice(&phonemes);

        assert_eq!(realized.len(), 1);
        assert_eq!(realized[0].phone.ipa, "tʃ");
        assert_eq!(realized[0].source_phoneme_index, 0);
        assert_eq!(realized[0].source_symbol, "CH");
    }

    #[test]
    fn acoustic_decomposition_splits_affricates_late() {
        let phonemes = vec![phoneme_from_arpabet("CH", "cmudict")];

        let broad = RealizedPhone::from_phoneme_slice_with_policy(
            &phonemes,
            PhoneDecompositionPolicy::KeepPhonemic,
        );
        assert_eq!(
            broad
                .iter()
                .map(|p| p.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["tʃ"]
        );

        let acoustic = RealizedPhone::from_phoneme_slice_with_policy(
            &phonemes,
            PhoneDecompositionPolicy::SplitForAcoustics,
        );
        assert_eq!(
            acoustic
                .iter()
                .map(|p| p.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["t", "ʃ"]
        );
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
        assert!(
            realized[1]
                .realization
                .environment_match
                .as_ref()
                .is_some_and(|m| m.commitment == MatchCommitment::Provisional)
        );
    }

    #[test]
    fn central_phoneme_presents_realized_tap_by_requested_schema() {
        let seq = vec![
            phoneme_from_arpabet("EY2", "morphophonology"),
            phoneme_from_arpabet("T", "morphophonology"),
            phoneme_from_arpabet("IH0", "morphophonology"),
        ];

        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );

        assert_eq!(realized[1].symbol, "T");
        assert_eq!(realized[1].realization.ipa, "ɾ");
        assert_eq!(
            realized[1].realization.rule.as_deref(),
            Some("american_english_intervocalic_flapping")
        );
        assert_eq!(realized[1].symbol_in_schema(PhonemeSchema::Arpabet), "T");
        assert_eq!(
            realized[1].symbol_in_schema(PhonemeSchema::ArpabetSurface),
            "DX"
        );
        assert_eq!(realized[1].symbol_in_schema(PhonemeSchema::Ipa), "ɾ");
        assert_eq!(
            realize_sequence_as_schema(
                &seq,
                &RealizationConfig {
                    enable_allophone_rules: true,
                    ..RealizationConfig::default()
                },
                PhonemeSchema::ArpabetSurface,
            ),
            vec!["EY2", "DX", "IH0"]
        );
    }

    #[test]
    fn flapping_rule_does_not_apply_to_d() {
        let seq = vec![
            phoneme_from_arpabet("EH1", "cmudict"),
            phoneme_from_arpabet("D", "cmudict"),
            phoneme_from_arpabet("IY0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].symbol, "D");
        assert_eq!(realized[1].realization.ipa, "d");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
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
    fn allophone_rules_are_enabled_by_default() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(&seq, &RealizationConfig::default());
        assert_eq!(realized[1].realization.ipa, "ɾ");
        assert_eq!(
            realized[1].realization.method,
            RealizationMethod::AllophoneRule
        );
    }

    #[test]
    fn allophone_rules_can_be_disabled() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: false,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn nasal_assimilation_realizes_n_before_velars() {
        let seq = vec![
            phoneme_from_arpabet("IH0", "cmudict"),
            phoneme_from_arpabet("N", "cmudict"),
            phoneme_from_arpabet("K", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "ŋ");
        assert_eq!(
            realized[1].realization.rule.as_deref(),
            Some("alveolar_nasal_velar_assimilation")
        );
        assert_eq!(
            realized[1]
                .realization
                .environment
                .as_ref()
                .and_then(|env| env.right_class.as_deref()),
            Some("velar_stop")
        );
    }

    #[test]
    fn nasal_assimilation_does_not_apply_before_non_velars() {
        let seq = vec![
            phoneme_from_arpabet("IH0", "cmudict"),
            phoneme_from_arpabet("N", "cmudict"),
            phoneme_from_arpabet("D", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "n");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn feature_bundle_supports_environment_classes_beyond_labels() {
        let nasal = phoneme_from_arpabet("M", "cmudict");
        let sibilant = phoneme_from_arpabet("SH", "cmudict");
        let high_vowel = phoneme_from_arpabet("IY0", "cmudict");

        assert!(phoneme_class_matches(PhonemeClass::Sonorant, &nasal));
        assert!(phoneme_class_matches(PhonemeClass::Labial, &nasal));
        assert!(phoneme_class_matches(PhonemeClass::Sibilant, &sibilant));
        assert!(phoneme_class_matches(PhonemeClass::Coronal, &sibilant));
        assert!(phoneme_class_matches(PhonemeClass::HighVowel, &high_vowel));
        assert!(phoneme_class_matches(
            PhonemeClass::UnstressedVowel,
            &high_vowel
        ));
    }

    #[test]
    fn commitment_follows_span_state() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                span_state: SpanState::Committed,
                ..RealizationConfig::default()
            },
        );
        assert!(
            realized[1]
                .realization
                .environment_match
                .as_ref()
                .is_some_and(|m| m.commitment == MatchCommitment::Committed)
        );
    }

    #[test]
    fn flapping_rule_is_blocked_in_careful_style() {
        let seq = vec![
            phoneme_from_arpabet("AE1", "cmudict"),
            phoneme_from_arpabet("T", "cmudict"),
            phoneme_from_arpabet("ER0", "cmudict"),
        ];
        let realized = realize_sequence(
            &seq,
            &RealizationConfig {
                enable_allophone_rules: true,
                careful_style: true,
                ..RealizationConfig::default()
            },
        );
        assert_eq!(realized[1].realization.ipa, "t");
        assert_eq!(realized[1].realization.method, RealizationMethod::Default);
    }

    #[test]
    fn phone_decomposition_policy_keeps_or_splits_diphthongs() {
        let seq = vec![phoneme_from_arpabet("OW1", "cmudict")];

        let broad = RealizedPhone::from_phoneme_slice_with_policy(
            &seq,
            PhoneDecompositionPolicy::KeepPhonemic,
        );
        assert_eq!(
            broad
                .iter()
                .map(|p| p.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["oʊ"]
        );

        let singing = RealizedPhone::from_phoneme_slice_with_policy(
            &seq,
            PhoneDecompositionPolicy::SplitForSinging,
        );
        assert_eq!(
            singing
                .iter()
                .map(|p| p.phone.ipa.as_str())
                .collect::<Vec<_>>(),
            vec!["o", "ʊ"]
        );
    }
}
