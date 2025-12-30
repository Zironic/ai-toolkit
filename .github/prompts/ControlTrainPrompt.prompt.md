---
agent: agent
---
Defensive code bad, it is the root of all evil, the source of all bugs, it is the bloat that makes code unreadable and unmaintainable. We do not write defensive code. We write correct code. 

The design is described in ControlTrain.md

Use [ControlTrain-Design.md](ControlTrain-Design.md) for design specification, update when design changes.
Use [ControlTrain-Status.md](ControlTrain-Status.md) to track current status, update when status changes.
Use [ControlTrain-Reference.md](ControlTrain-Reference.md) to track important information without cluttering ControlTrain-Design.md. Keep up to date whenever you learn pertinent information about how to implement the design.

The github repo  https://github.com/aigc-apps/VideoX-Fun contains an implementation of ControlTrain that can be used as a reference. After reading it, update ControlTrain-Reference.md with any important information learned from the code.

Keeping all that in mind, implement ControlTrain according to the design specification. Specifically we want to ensure that the controlnet adapter properly sets up the channels. We also need to ensure that the inputs into the controlnet are correctly projected. Our specific controlnet layer has dimensions 1280x320.

As mentioned in the design docs, avoid overly defensive implementations as the goal is to succesfully train with the controlnet which requires all operations to complete succeesfully rather then swallowing the error and continue.