# Listenbury

Listenbury is a single-binary, low-latency Pete implementation focused on real-time embodied conversation: hearing, turn-taking, local inference, speech planning, and speaking.

Listenbury does not replace Daringsby. Daringsby remains the distributed combobulating Pete implementation. Listenbury explores a tighter real-time organism: one process, bounded queues, realtime-safe audio paths, and local model backends behind traits.

## Design mantra

Detect speech before understanding speech.
Segment breath groups before sentences.
Stream tokens before monologues.
Speak only when the turn-taking state permits it.
