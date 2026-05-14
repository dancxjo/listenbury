# Listenbury

Listenbury is a single-binary, low-latency Pete implementation focused on real-time embodied conversation: hearing, turn-taking, local inference, speech planning, and speaking.

Part of [Project PETE](https://dancxjo.github.io/project-pete.html): the Pseudoconscious Experiment in Thought and Emotions. Listenbury explores the low-latency continuation of PETE, where listening, inference, speech planning, TTS, and self-hearing can overlap in real time.

Listenbury does not replace Daringsby. Daringsby remains the distributed combobulating Pete implementation. Listenbury explores a tighter real-time organism: one process, bounded queues, realtime-safe audio paths, and local model backends behind traits.

All frame structs carry an exact wall-clock capture timestamp so timing stays traceable across the pipeline.

## Design mantra

Detect speech before understanding speech.
Segment breath groups before sentences.
Stream tokens before monologues.
Speak only when the turn-taking state permits it.
