use anyhow::{Result, anyhow};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::audio::frame::AudioFrame;

pub struct AudioRingTx {
    producer: Producer<AudioFrame>,
}

pub struct AudioRingRx {
    consumer: Consumer<AudioFrame>,
}

pub fn make_audio_ring(capacity: usize) -> (AudioRingTx, AudioRingRx) {
    let (producer, consumer) = RingBuffer::new(capacity);
    (AudioRingTx { producer }, AudioRingRx { consumer })
}

impl AudioRingTx {
    pub fn try_push(&mut self, frame: AudioFrame) -> Result<()> {
        self.producer
            .push(frame)
            .map_err(|_| anyhow!("audio ring is full"))
    }
}

impl AudioRingRx {
    pub fn try_pop(&mut self) -> Option<AudioFrame> {
        self.consumer.pop().ok()
    }
}
