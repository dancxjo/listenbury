use std::collections::{HashMap, VecDeque};
use std::fmt;

use chrono::{DateTime, Utc};
use serde_json::Value;
use uuid::Uuid;

pub const IMAGE_DESCRIPTION_IMPRESSION_PROMPT: &str = "Describe the image as Pete's first-person visual impression. Write direct captions such as `I see Travis sitting at a desk.` Do not write hedged third-person captions such as `The image appears to show Travis sitting at a desk.` If a person is visible, do not assume Pete is looking at himself unless the image clearly shows a mirror, screen reflection, camera preview, or another explicit self-view cue.";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SensationId(String);

impl SensationId {
    pub fn new() -> Self {
        Self(format!("sens_{}", Uuid::new_v4().simple()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SensationId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for SensationId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for SensationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for SensationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImpressionId(String);

impl ImpressionId {
    pub fn new() -> Self {
        Self(format!("imp_{}", Uuid::new_v4().simple()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ImpressionId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for ImpressionId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for ImpressionId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for ImpressionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorRef {
    pub backend: String,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sensation {
    pub id: SensationId,
    pub kind: String,
    pub source: String,
    /// When the represented event/data occurred in the world or sensor stream.
    pub occurred_at: DateTime<Utc>,
    /// When this sensation record was created inside the system.
    pub created_at: DateTime<Utc>,
    /// Optional parent sensation for derived sensations, such as face crops from an image.
    pub parent: Option<SensationId>,
    /// Blob/file/object reference or inline payload descriptor.
    pub payload_ref: Option<String>,
    /// Optional natural-language handle for quick timeline display.
    pub natural_handle: Option<String>,
    /// Optional structured metadata, e.g. bbox, sample rate, dimensions, MIME type.
    pub metadata: Value,
    /// Zero or more vector references, not necessarily inline vectors.
    pub vectors: Vec<VectorRef>,
}

impl Sensation {
    pub fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        occurred_at: DateTime<Utc>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: SensationId::new(),
            kind: kind.into(),
            source: source.into(),
            occurred_at,
            created_at,
            parent: None,
            payload_ref: None,
            natural_handle: None,
            metadata: Value::Null,
            vectors: Vec::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<SensationId>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_parent(mut self, parent: impl Into<SensationId>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_payload_ref(mut self, payload_ref: impl Into<String>) -> Self {
        self.payload_ref = Some(payload_ref.into());
        self
    }

    pub fn with_natural_handle(mut self, natural_handle: impl Into<String>) -> Self {
        self.natural_handle = Some(natural_handle.into());
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_vectors(mut self, vectors: Vec<VectorRef>) -> Self {
        self.vectors = vectors;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Impression {
    pub id: ImpressionId,
    pub about: Vec<SensationId>,
    pub kind: String,
    /// When this interpretation/claim was produced.
    pub made_at: DateTime<Utc>,
    pub made_by: String,
    pub text: String,
    pub confidence: Option<f32>,
    pub structured: Value,
    pub vectors: Vec<VectorRef>,
}

impl Impression {
    pub fn new(
        about: Vec<SensationId>,
        kind: impl Into<String>,
        made_at: DateTime<Utc>,
        made_by: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: ImpressionId::new(),
            about,
            kind: kind.into(),
            made_at,
            made_by: made_by.into(),
            text: text.into(),
            confidence: None,
            structured: Value::Null,
            vectors: Vec::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<ImpressionId>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_structured(mut self, structured: Value) -> Self {
        self.structured = structured;
        self
    }

    pub fn with_vectors(mut self, vectors: Vec<VectorRef>) -> Self {
        self.vectors = vectors;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChildSensationInput {
    pub id: Option<SensationId>,
    pub kind: String,
    pub source: String,
    pub occurred_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub payload_ref: Option<String>,
    pub natural_handle: Option<String>,
    pub metadata: Value,
    pub vectors: Vec<VectorRef>,
}

impl ChildSensationInput {
    pub fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: None,
            kind: kind.into(),
            source: source.into(),
            occurred_at: None,
            created_at,
            payload_ref: None,
            natural_handle: None,
            metadata: Value::Null,
            vectors: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineQuery {
    pub since: Option<DateTime<Utc>>,
    pub max_sensations: usize,
    pub max_impressions: usize,
}

impl Default for TimelineQuery {
    fn default() -> Self {
        Self {
            since: None,
            max_sensations: 12,
            max_impressions: 24,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimelineWindow {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
}

impl TimelineWindow {
    pub fn is_empty(&self) -> bool {
        self.sensations.is_empty() && self.impressions.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ShortTermTimeline {
    sensations: VecDeque<Sensation>,
    impressions: VecDeque<Impression>,
    max_sensations: usize,
    max_impressions: usize,
}

impl ShortTermTimeline {
    pub fn new(max_sensations: usize, max_impressions: usize) -> Self {
        Self {
            sensations: VecDeque::new(),
            impressions: VecDeque::new(),
            max_sensations: max_sensations.max(1),
            max_impressions: max_impressions.max(1),
        }
    }

    pub fn record_sensation(&mut self, sensation: Sensation) -> SensationId {
        let id = sensation.id.clone();
        self.sensations.push_back(sensation);
        while self.sensations.len() > self.max_sensations {
            self.sensations.pop_front();
        }
        id
    }

    pub fn record_impression(&mut self, impression: Impression) -> ImpressionId {
        let id = impression.id.clone();
        self.impressions.push_back(impression);
        while self.impressions.len() > self.max_impressions {
            self.impressions.pop_front();
        }
        id
    }

    pub fn sensation(&self, id: &SensationId) -> Option<&Sensation> {
        self.sensations.iter().find(|sensation| &sensation.id == id)
    }

    pub fn impressions_about(&self, id: &SensationId) -> Vec<&Impression> {
        self.impressions
            .iter()
            .filter(|impression| impression.about.iter().any(|about| about == id))
            .collect()
    }

    pub fn derive_child_sensation(
        &mut self,
        parent: &SensationId,
        input: ChildSensationInput,
    ) -> Option<SensationId> {
        let parent_sensation = self.sensation(parent)?.clone();
        let mut sensation = Sensation::new(
            input.kind,
            input.source,
            input.occurred_at.unwrap_or(parent_sensation.occurred_at),
            input.created_at,
        )
        .with_parent(parent.clone())
        .with_metadata(input.metadata)
        .with_vectors(input.vectors);
        sensation.payload_ref = input.payload_ref;
        sensation.natural_handle = input.natural_handle;
        if let Some(id) = input.id {
            sensation.id = id;
        }
        Some(self.record_sensation(sensation))
    }

    pub fn window(&self, query: TimelineQuery) -> TimelineWindow {
        let mut sensations = self
            .sensations
            .iter()
            .filter(|sensation| {
                query
                    .since
                    .is_none_or(|since| sensation.occurred_at >= since)
            })
            .cloned()
            .collect::<Vec<_>>();
        sort_sensations(&mut sensations);
        if sensations.len() > query.max_sensations {
            sensations =
                sensations.split_off(sensations.len().saturating_sub(query.max_sensations));
        }

        let selected_ids = sensations
            .iter()
            .map(|sensation| sensation.id.clone())
            .collect::<Vec<_>>();
        let mut impressions = self
            .impressions
            .iter()
            .filter(|impression| {
                impression
                    .about
                    .iter()
                    .any(|id| selected_ids.iter().any(|selected| selected == id))
                    || query.since.is_none_or(|since| impression.made_at >= since)
            })
            .cloned()
            .collect::<Vec<_>>();
        sort_impressions(&mut impressions, &self.sensation_times());
        if impressions.len() > query.max_impressions {
            impressions =
                impressions.split_off(impressions.len().saturating_sub(query.max_impressions));
        }

        TimelineWindow {
            sensations,
            impressions,
        }
    }

    fn sensation_times(&self) -> HashMap<SensationId, DateTime<Utc>> {
        self.sensations
            .iter()
            .map(|sensation| (sensation.id.clone(), sensation.occurred_at))
            .collect()
    }
}

impl Default for ShortTermTimeline {
    fn default() -> Self {
        Self::new(128, 256)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TimelineFormatter {
    pub max_text_chars: usize,
}

impl Default for TimelineFormatter {
    fn default() -> Self {
        Self {
            max_text_chars: 220,
        }
    }
}

impl TimelineFormatter {
    pub fn format_prompt_section(&self, window: &TimelineWindow) -> String {
        if window.is_empty() {
            return "Recent events:\n(no recent sensations or impressions)".to_string();
        }

        let mut sensations = window.sensations.clone();
        sort_sensations(&mut sensations);
        let sensation_times = sensations
            .iter()
            .map(|sensation| (sensation.id.clone(), sensation.occurred_at))
            .collect::<HashMap<_, _>>();
        let mut impressions = window.impressions.clone();
        sort_impressions(&mut impressions, &sensation_times);

        let mut impressions_by_sensation = HashMap::<SensationId, Vec<Impression>>::new();
        let mut standalone = Vec::new();
        for impression in impressions {
            let mut attached = false;
            for id in &impression.about {
                if sensation_times.contains_key(id) {
                    impressions_by_sensation
                        .entry(id.clone())
                        .or_default()
                        .push(impression.clone());
                    attached = true;
                }
            }
            if !attached {
                standalone.push(impression);
            }
        }

        let mut lines = vec!["Recent events:".to_string()];
        for sensation in &sensations {
            lines.push(format!(
                "- [{}] {} {}",
                time_of_day(sensation.occurred_at),
                sensation.id,
                self.sensation_text(sensation, &sensation_times)
            ));
            if let Some(items) = impressions_by_sensation.get(&sensation.id) {
                for impression in items {
                    lines.push(format!(
                        "  - [interpreted {}] {} by {}: {}{}",
                        time_of_day(impression.made_at),
                        impression.id,
                        impression.made_by,
                        compact_line(&impression.text, self.max_text_chars),
                        confidence_suffix(impression.confidence)
                    ));
                }
            }
        }

        for impression in standalone {
            let about = if impression.about.is_empty() {
                "no sensation id".to_string()
            } else {
                impression
                    .about
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            };
            lines.push(format!(
                "- [{}] {} by {} interpreted {}: {}{}",
                time_of_day(impression_moment(&impression, &sensation_times)),
                impression.id,
                impression.made_by,
                about,
                compact_line(&impression.text, self.max_text_chars),
                confidence_suffix(impression.confidence)
            ));
        }

        lines.join("\n")
    }

    fn sensation_text(
        &self,
        sensation: &Sensation,
        sensation_times: &HashMap<SensationId, DateTime<Utc>>,
    ) -> String {
        if let Some(handle) = sensation.natural_handle.as_deref() {
            return punctuated(compact_line(handle, self.max_text_chars));
        }

        let kind = sensation.kind.replace('_', " ");
        let mut text = format!("{} produced {}.", sensation.source, article_phrase(&kind));
        if let Some(parent) = &sensation.parent
            && sensation_times.contains_key(parent)
        {
            text = format!(
                "{} produced {} derived from {}.",
                sensation.source,
                article_phrase(&kind),
                parent
            );
        }
        compact_line(&text, self.max_text_chars)
    }
}

fn sort_sensations(sensations: &mut [Sensation]) {
    sensations.sort_by(|left, right| {
        left.occurred_at
            .cmp(&right.occurred_at)
            .then_with(|| left.created_at.cmp(&right.created_at))
            .then_with(|| left.parent.is_some().cmp(&right.parent.is_some()))
            .then_with(|| left.id.as_str().cmp(right.id.as_str()))
    });
}

fn sort_impressions(
    impressions: &mut [Impression],
    sensation_times: &HashMap<SensationId, DateTime<Utc>>,
) {
    impressions.sort_by(|left, right| {
        impression_moment(left, sensation_times)
            .cmp(&impression_moment(right, sensation_times))
            .then_with(|| left.made_at.cmp(&right.made_at))
            .then_with(|| left.id.as_str().cmp(right.id.as_str()))
    });
}

fn impression_moment(
    impression: &Impression,
    sensation_times: &HashMap<SensationId, DateTime<Utc>>,
) -> DateTime<Utc> {
    impression
        .about
        .iter()
        .filter_map(|id| sensation_times.get(id).copied())
        .min()
        .unwrap_or(impression.made_at)
}

fn time_of_day(value: DateTime<Utc>) -> String {
    value.format("%H:%M:%S").to_string()
}

fn confidence_suffix(confidence: Option<f32>) -> String {
    match confidence {
        Some(value) => format!(" confidence={:.2}", value),
        None => " confidence=unknown".to_string(),
    }
}

fn compact_line(text: &str, max_chars: usize) -> String {
    let mut line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= max_chars {
        return line;
    }
    line = line.chars().take(max_chars.saturating_sub(3)).collect();
    line.push_str("...");
    line
}

fn punctuated(text: String) -> String {
    if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
        text
    } else {
        format!("{text}.")
    }
}

fn article_phrase(kind: &str) -> String {
    let article = kind
        .chars()
        .next()
        .map(|c| {
            if matches!(c.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u') {
                "an"
            } else {
                "a"
            }
        })
        .unwrap_or("a");
    format!("{article} {kind}")
}

// TODO: Add a persistence/selection trait here when short-term sensations and
// impressions need promotion into long-term memory. The timeline buffer is only
// immediate scene context; retaining an item here does not make it durable memory.

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ts(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).expect("valid timestamp")
    }

    #[test]
    fn timestamp_separation_preserves_event_record_and_interpretation_times() {
        let t0 = ts(1);
        let t1 = ts(2);
        let t2 = ts(3);
        let sensation = Sensation::new("image_frame", "camera0", t0, t1).with_id("sens_image");
        let impression = Impression::new(
            vec![sensation.id.clone()],
            "image_caption",
            t2,
            "vision_llm",
            "I see Travis sitting at a desk.",
        )
        .with_id("imp_caption");

        assert_eq!(sensation.occurred_at, t0);
        assert_eq!(sensation.created_at, t1);
        assert_eq!(impression.made_at, t2);
    }

    #[test]
    fn child_sensation_derivation_inherits_parent_occurrence_by_default() {
        let mut timeline = ShortTermTimeline::default();
        let parent = Sensation::new("image_frame", "camera0", ts(10), ts(11)).with_id("sens_image");
        let parent_id = timeline.record_sensation(parent);

        let child_id = timeline
            .derive_child_sensation(
                &parent_id,
                ChildSensationInput {
                    id: Some("sens_face".into()),
                    kind: "face_crop".to_string(),
                    source: "face_detector".to_string(),
                    occurred_at: None,
                    created_at: ts(12),
                    payload_ref: Some("blob://face".to_string()),
                    natural_handle: Some("face crop derived from camera0 image frame".to_string()),
                    metadata: json!({ "bbox": [1, 2, 3, 4] }),
                    vectors: Vec::new(),
                },
            )
            .expect("parent exists");

        let child = timeline.sensation(&child_id).expect("child exists");
        assert_eq!(child.parent.as_ref(), Some(&parent_id));
        assert_eq!(child.occurred_at, ts(10));
        assert_eq!(child.created_at, ts(12));
    }

    #[test]
    fn multiple_impressions_are_returned_for_same_sensation() {
        let mut timeline = ShortTermTimeline::default();
        let sensation =
            Sensation::new("image_frame", "camera0", ts(20), ts(21)).with_id("sens_image");
        let id = timeline.record_sensation(sensation);

        for (suffix, kind, text) in [
            (
                "caption",
                "image_caption",
                "I see Travis sitting at a desk.",
            ),
            ("objects", "object_detection", "I see a desk and a monitor."),
            (
                "recognition",
                "face_recognition",
                "I recognize this as Travis.",
            ),
        ] {
            timeline.record_impression(
                Impression::new(vec![id.clone()], kind, ts(22), suffix, text)
                    .with_id(format!("imp_{suffix}")),
            );
        }

        assert_eq!(timeline.impressions_about(&id).len(), 3);
        assert_eq!(
            timeline.window(TimelineQuery::default()).impressions.len(),
            3
        );
    }

    #[test]
    fn timeline_formatting_is_stable_and_chronological() {
        let mut timeline = ShortTermTimeline::default();
        let image = Sensation::new("image_frame", "camera0", ts(43_201), ts(43_202))
            .with_id("sens_image")
            .with_natural_handle("camera0 produced an image frame");
        let image_id = timeline.record_sensation(image);
        timeline.record_impression(
            Impression::new(
                vec![image_id.clone()],
                "image_caption",
                ts(43_203),
                "vision_llm",
                "I see Travis sitting at a desk.",
            )
            .with_id("imp_caption"),
        );
        let face = timeline
            .derive_child_sensation(
                &image_id,
                ChildSensationInput {
                    id: Some("sens_face".into()),
                    kind: "face_crop".to_string(),
                    source: "face_detector".to_string(),
                    occurred_at: None,
                    created_at: ts(43_204),
                    payload_ref: None,
                    natural_handle: Some("face crop derived from camera0 image frame".to_string()),
                    metadata: Value::Null,
                    vectors: Vec::new(),
                },
            )
            .expect("child");
        timeline.record_impression(
            Impression::new(
                vec![face],
                "face_recognition",
                ts(43_205),
                "face_recognizer",
                "I recognize this as George.",
            )
            .with_id("imp_face")
            .with_confidence(0.82),
        );

        let formatted =
            TimelineFormatter::default().format_prompt_section(&timeline.window(TimelineQuery {
                since: Some(ts(43_200)),
                max_sensations: 8,
                max_impressions: 8,
            }));

        assert_eq!(
            formatted,
            "Recent events:\n- [12:00:01] sens_image camera0 produced an image frame.\n  - [interpreted 12:00:03] imp_caption by vision_llm: I see Travis sitting at a desk. confidence=unknown\n- [12:00:01] sens_face face crop derived from camera0 image frame.\n  - [interpreted 12:00:05] imp_face by face_recognizer: I recognize this as George. confidence=0.82"
        );
    }

    #[test]
    fn image_description_prompt_prefers_first_person_direct_captioning() {
        assert!(IMAGE_DESCRIPTION_IMPRESSION_PROMPT.contains("I see Travis sitting at a desk."));
        assert!(IMAGE_DESCRIPTION_IMPRESSION_PROMPT.contains("Do not write"));
        assert!(IMAGE_DESCRIPTION_IMPRESSION_PROMPT.contains("unless the image clearly shows"));
    }
}
