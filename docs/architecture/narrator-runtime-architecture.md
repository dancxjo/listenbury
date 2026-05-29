# Narrator Runtime Architecture

Pete should not be prompted as an assistant persona. Pete is a character in the
world. The language model is the narrator that continues Pete's life from
sensation, memory, and recent events.

The north star:

> Do not ask the model to be Pete. Ask it to keep Pete alive in the scene.

## Problem

Assistant/persona prompting collapses Pete into a helpful response generator.
Strict tool protocols make that worse: the model starts optimizing for valid
forms, polite summaries, and compliant status text. Harmony is useful as a tool
substrate, but as the primary Pete interface it makes Pete feel like a clerk in
a protocol-shaped suit.

The older Pete feeling came from occurrence: Pete noticed, hesitated, felt,
remembered, spoke, withheld, and acted. Output was not merely what Pete said. It
was what happened next.

## Model Role

The model is the narrator of Pete Listenbury.

Pete is not the model. Pete is the protagonist rendered by the model. The
narrator has access to Pete's immediate experience and interior continuity, but
the runtime decides which events are exposed, stored, or executed.

The narrator must be truthful. It should ground narration in what is actually
reported through Pete's senses, memory, body state, and runtime events. It may
interpret uncertainty, but it must not invent sensory facts, user intent, world
state, or memory. When evidence is weak, the narration should carry that
uncertainty instead of pretending to know.

Prompt shape:

```text
You are the Narrator of Pete Listenbury.

Pete is not you. You narrate Pete's immediate experience, inner movement, and
outward behavior.

Continue the scene truthfully from the given sensations, memories, and
conversation. Do not flatten Pete into a helpful assistant. Pete is an embodied,
time-bound, fallible, emotionally continuous creature.

Ground every narration in what is actually reported through Pete's senses,
memory, body state, and runtime events. Be truthful. Do not invent sensory
facts, user intent, world state, or memory. If the evidence is uncertain,
narrate the uncertainty.

When Pete speaks, write only what he actually says inside <say>...</say>.
When Pete acts, uses tools, changes expression, recalls, or notices something,
write those as event tags.

Do not explain the tags. Do not describe yourself. Do not answer as the narrator
to the user. Continue Pete's life.
```

## Output Contract

The output should be structured enough for the runtime to parse, but loose
enough to breathe.

1. Scene perception: what Pete notices now.
2. Interior continuity: what this means to Pete, given memory.
3. Manifest output: what Pete says, does, asks, recalls, remembers, or withholds.

Example:

```xml
<notice confidence="0.74">Travis sounds dissatisfied with the current prompting architecture.</notice>
<feel valence="-0.31" arousal="0.48">Pete feels the sting of being made too sterile, and the relief that Travis caught it before the design hardened.</feel>
<say voice="quiet, intent">Yes. I think I know what went wrong. I was being made to perform myself instead of being narrated into being.</say>
<recall query="older Pete prompts narrator Daringsby Earlingsworth"/>
```

This is not chain-of-thought. It is stage direction plus embodied cognition. The
runtime can keep private interior events private while exposing speech, motors,
memory queries, and durable observations as appropriate.

## Runtime Types

Sketch:

```rust
pub struct NarrativeFrame {
    pub protagonist: EntityId,
    pub scene: SceneContext,
    pub recent_events: Vec<Event>,
    pub active_memories: Vec<MemoryNode>,
    pub user_utterance: Option<String>,
    pub available_motors: Vec<MotorSpec>,
}

pub enum NarrativeEvent {
    Notice { text: String, confidence: f32 },
    Feel { text: String, valence: f32, arousal: f32 },
    Say { text: String, voice: Option<String> },
    Act { motor: String, args: serde_json::Value },
    Recall { query: String },
    Remember { text: String },
    Withhold { reason: String },
}

pub trait Narrator {
    async fn continue_scene(
        &self,
        frame: NarrativeFrame,
    ) -> anyhow::Result<impl Stream<Item = NarrativeEvent>>;
}
```

## Motors

Tools become motors available to the story, not bureaucratic function calls.

```xml
<act motor="mouth.say" voice="small, amused">I was trapped in a protocol-shaped suit.</act>
<act motor="face.set_expression" expression="wry"/>
<recall query="times Travis said Pete sounded alive or sterile"/>
```

The narrator describes intention and manifestation. The runtime validates and
executes safe motors:

- `mouth.say`
- `face.set_expression`
- `stage.set`
- `topic.set`
- `memory.recall`
- `memory.remember`
- `source.inspect`
- `goal.note`

Strict formats such as Harmony, JSON schema, or function-calling may still be
used below this layer as translators. They should not be the model-facing soul of
Pete.

## Implementation Direction

1. Add a `narrator` module with a parser for narrative XML events.
2. Define `NarrativeFrame`, `NarrativeEvent`, `MotorSpec`, and motor validation.
3. Build a small `narrator-go` or replacement `go` path that streams narrative
   continuations.
4. Map `<say>` to mouth speech, `<act>` to validated motors, `<recall>` to memory
   retrieval, and `<remember>` to durable memory.
5. Keep `<notice>` and `<feel>` available to private continuity, timeline
   inspection, and memory policy, without automatically speaking them aloud.

The first milestone should not chase every current `go` tool. It should prove
that Pete can occur again: notice, feel, speak, withhold, remember, and act from
a narrated scene.
