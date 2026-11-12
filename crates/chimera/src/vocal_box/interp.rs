//! Zero-allocation audio-rate $C^1$ frame interpolator.
//!
//! Eliminates 64-sample control-block staircase modulation buzz by ramping all continuous
//! parameters smoothly per sample with $C^1$ Hermite smoothstep trajectories.

use super::vowels::{VocalFrame, lerp_frame};

/// Audio-rate $C^1$-continuous interpolator between [`VocalFrame`] targets.
#[derive(Clone, Debug)]
pub struct FrameInterpolator {
    start: VocalFrame,
    current: VocalFrame,
    target: VocalFrame,
    pos: usize,
    len: usize,
}

impl Default for FrameInterpolator {
    fn default() -> Self {
        Self::new(VocalFrame::default())
    }
}

impl FrameInterpolator {
    pub fn new(initial: VocalFrame) -> Self {
        let s = initial.sanitized();
        Self {
            start: s,
            current: s,
            target: s,
            pos: 0,
            len: 0,
        }
    }

    /// Snap immediately to `frame` with zero ramp.
    pub fn reset(&mut self, frame: VocalFrame) {
        let s = frame.sanitized();
        self.start = s;
        self.current = s;
        self.target = s;
        self.pos = 0;
        self.len = 0;
    }

    /// Set a new target frame to reach smoothly over `ramp_samples` output samples.
    pub fn set_target(&mut self, target: VocalFrame, ramp_samples: usize) {
        let t = target.sanitized();
        if ramp_samples == 0 {
            self.reset(t);
            return;
        }
        self.start = self.current;
        self.target = t;
        self.pos = 0;
        self.len = ramp_samples;
    }

    #[inline]
    pub fn current(&self) -> VocalFrame {
        self.current
    }

    /// Advance one sample along the $C^1$ smoothstep trajectory and return the interpolated frame.
    #[inline]
    pub fn step(&mut self) -> VocalFrame {
        if self.pos >= self.len {
            self.current = self.target;
            return self.current;
        }
        self.pos += 1;
        if self.pos >= self.len {
            self.current = self.target;
            return self.current;
        }
        let u = (self.pos as f64) / (self.len as f64);
        // Cubic Hermite smoothstep u^2 * (3 - 2u) gives zero slope jump at ramp boundaries.
        let s = u * u * (3.0 - 2.0 * u);
        self.current = lerp_frame(&self.start, &self.target, s);
        self.current
    }
}
