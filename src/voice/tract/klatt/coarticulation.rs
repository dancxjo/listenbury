use crate::linguistic::phonology::{FeatureBundle, MajorClass, Place, general_american_english};
use crate::voice::tract::targets::PhoneRenderTarget;
use crate::voice::tract::targets::default_english_phone_targets;

// Empirical blend factors tuned to stay audible but stable across the current
// English inventory without over-bending vowel targets.
const LEFT_CONSONANT_INFLUENCE: f32 = 0.35;
const RIGHT_CONSONANT_INFLUENCE: f32 = 0.25;
const INTERVOCALIC_CONSONANT_F2_BLEND: f32 = 0.4;

pub(crate) fn apply_neighbor_influence(targets: &[PhoneRenderTarget]) -> Vec<PhoneRenderTarget> {
    let inventory = general_american_english();
    let fallback_table = default_english_phone_targets();
    let mut adjusted = targets.to_vec();
    for idx in 0..adjusted.len() {
        let phone = adjusted[idx].phone.clone();
        let f0_hz = adjusted[idx].f0_hz;
        let left_phone = idx.checked_sub(1).map(|p| adjusted[p].phone.clone());
        let right_phone = adjusted.get(idx + 1).map(|target| target.phone.clone());
        let left_vowel_f2 = idx
            .checked_sub(1)
            .and_then(|p| adjusted[p].filter.as_ref().map(|f| f.f2_hz));
        let right_vowel_f2 = adjusted
            .get(idx + 1)
            .and_then(|next| next.filter.as_ref().map(|f| f.f2_hz));
        let features = inventory.features_for_phone(&phone);

        let Some(filter) = adjusted[idx].filter.as_mut() else {
            continue;
        };

        if is_vowel_features(features)
            || fallback_table
                .get(phone.ipa.as_str())
                .is_some_and(|target| target.is_vowel)
        {
            let left_bias = left_phone
                .as_ref()
                .map(|candidate| inventory.features_for_phone(candidate))
                .and_then(consonant_f2_bias_from_features)
                .or_else(|| {
                    left_phone
                        .as_ref()
                        .and_then(|candidate| consonant_f2_bias_from_symbol(candidate.ipa.as_str()))
                })
                .unwrap_or(0.0);
            let right_bias = right_phone
                .as_ref()
                .map(|candidate| inventory.features_for_phone(candidate))
                .and_then(consonant_f2_bias_from_features)
                .or_else(|| {
                    right_phone
                        .as_ref()
                        .and_then(|candidate| consonant_f2_bias_from_symbol(candidate.ipa.as_str()))
                })
                .unwrap_or(0.0);
            filter.f2_hz +=
                left_bias * LEFT_CONSONANT_INFLUENCE + right_bias * RIGHT_CONSONANT_INFLUENCE;
            continue;
        }

        if f0_hz.is_none() {
            if let (Some(left), Some(right)) = (left_vowel_f2, right_vowel_f2) {
                let center = (left + right) * 0.5;
                filter.f2_hz =
                    filter.f2_hz + (center - filter.f2_hz) * INTERVOCALIC_CONSONANT_F2_BLEND;
            }
        }
    }
    adjusted
}

fn is_vowel_features(features: FeatureBundle) -> bool {
    features.major == MajorClass::Vowel || features.syllabic
}

fn consonant_f2_bias_from_features(features: FeatureBundle) -> Option<f32> {
    if features.major != MajorClass::Consonant {
        return None;
    }
    Some(match features.place {
        Some(Place::Palatal) => 150.0,
        Some(Place::Velar) => -160.0,
        Some(Place::Bilabial | Place::Labiodental) => -80.0,
        Some(Place::Dental | Place::Alveolar | Place::Postalveolar) => 110.0,
        Some(Place::Glottal) => 60.0,
        _ => return None,
    })
}

fn consonant_f2_bias_from_symbol(phone: &str) -> Option<f32> {
    Some(match phone {
        "k" | "ɡ" | "ŋ" => -160.0,
        "p" | "b" | "m" | "w" => -80.0,
        "t" | "d" | "n" | "s" | "z" | "l" | "ɹ" => 110.0,
        "f" | "v" | "θ" | "ð" | "h" => 60.0,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linguistic::phonology::Phone;
    use crate::linguistic::phonology::PhoneString;
    use crate::voice::tract::targets::{
        default_english_phone_targets, phone_render_targets_from_string,
    };

    #[test]
    fn vowel_f2_moves_toward_neighboring_consonants() {
        let table = default_english_phone_targets();
        let ps = PhoneString {
            phones: vec![
                Phone::new_ipa("k"),
                Phone::new_ipa("i"),
                Phone::new_ipa("t"),
            ],
        };
        let baseline = phone_render_targets_from_string(&ps, Some(150.0), 0.7, &table);
        let adjusted = apply_neighbor_influence(&baseline);
        let original_f2 = baseline[1].filter.as_ref().unwrap().f2_hz;
        let adjusted_f2 = adjusted[1].filter.as_ref().unwrap().f2_hz;
        assert_ne!(
            original_f2, adjusted_f2,
            "coarticulation should adjust vowel F2"
        );
    }
}
