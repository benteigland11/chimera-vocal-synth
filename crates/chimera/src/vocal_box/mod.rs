//! First-principles physiological AI Vocal Box (`VocalBoxCore`).
//!
//! Headless, `Send + Sync`, zero-allocation, deterministic pure-Rust physical voice core:
//! - Continuous `BodyMix` anatomy decoupling `vtl_scale` from laryngeal `fold_area_scale`,
//! - Continuous Fant $R_d$ transformed LF source + coupled self-oscillating Two-Mass vocal folds,
//! - Aerodynamic transglottal pressure gating ($\Delta P = P_{\text{sub}} - P_{\text{oral}}$), whisper & fry,
//! - 5-formant coupled supraglottal tract with causal open-phase glottal impedance coupling,
//!   soprano *aggiustamento* ($F_1 \ge 1.08 f_0$), epilarynx singer's ring, piriform notch,
//!   lateral `/l/` anti-formant, and rhotic $F_3$ lowering,
//! - Zero-leak coupled nasal tract with place-ordered `/m/ < /n/ < /ɲ/ < /ŋ/` shunt anti-formants,
//! - $N=8$ Chebyshev Type-II oversampled anti-aliasing decimator supporting any host sample rate.

pub mod glottal;
pub mod interp;
pub mod nasal;
pub mod resampler;
pub mod tract;
pub mod vowels;

pub use glottal::{GlottalEngine, GlottalStep};
pub use interp::FrameInterpolator;
pub use nasal::{NasalBranch, NasalOutput};
pub use resampler::{Cheby2Decimator, oversample_factor_for};
pub use tract::ResonantTract;
pub use vowels::{
    BodyPreset, ConstrictionPlace, VocalFrame, VocalParams, body_vtl_scale,
    closure_sharpness_from_frame, lerp_frame, soft_max_c1, vowel_frame, vowel_targets,
    vowel_targets_with_vtl,
};

pub use crate::anatomy::human::{BodyMix, BodyName};
use crate::math::{PI, Rng, TAU, clamp, cos, exp, sin, tanh};
use vowels::{flush_subnormal, sanitize_clamped};

/// Instantaneous physical telemetry from [`VocalBoxCore`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VocalBoxTelemetry {
    pub f0_now_hz: f64,
    pub voicing: f64,
    pub glottal_area_cm2: f64,
    pub glottal_flow_cm3_s: f64,
    pub effective_f1_hz: f64,
    pub effective_b1_hz: f64,
    pub raw_tract_out: f64,
    pub nasal_out: f64,
    pub velum_port_cm2: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct FricResonator {
    z1: f64,
    z2: f64,
}

impl FricResonator {
    #[inline(always)]
    fn step_bandpass(&mut self, x: f64, f_hz: f64, q: f64, fs: f64) -> f64 {
        let w0 = TAU * f_hz / fs;
        let alpha = sin(w0) / (2.0 * q.max(0.4));
        let cw = cos(w0);
        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        let b2 = -alpha / a0;
        let a1 = (-2.0 * cw) / a0;
        let a2 = (1.0 - alpha) / a0;

        let y = b0 * x + self.z1;
        let z1n = -a1 * y + self.z2;
        let z2n = b2 * x - a2 * y;
        self.z1 = flush_subnormal(z1n);
        self.z2 = flush_subnormal(z2n);
        flush_subnormal(y)
    }
}

/// Complete zero-allocation first-principles vocal synthesis core.
#[derive(Clone, Debug)]
pub struct VocalBoxCore {
    sample_rate: f64,
    decimator: Cheby2Decimator,
    interp: FrameInterpolator,
    glottis: GlottalEngine,
    tract: ResonantTract,
    nasal: NasalBranch,
    fric_res: FricResonator,
    fric_rng: Rng,
    dc_x: f64,
    dc_y: f64,
    dc_coef: f64,
    last_telemetry: VocalBoxTelemetry,
}

impl VocalBoxCore {
    /// Construct a warm, allocation-free [`VocalBoxCore`] at `sample_rate` (Hz).
    pub fn new(sample_rate: f64) -> Self {
        let fs_out = sanitize_clamped(sample_rate, 48_000.0, 4_000.0, 384_000.0);
        let decimator = Cheby2Decimator::new(fs_out);
        let fs_int = decimator.fs_int();
        let init_frame = VocalFrame::default();
        Self {
            sample_rate: fs_out,
            decimator,
            interp: FrameInterpolator::new(init_frame),
            glottis: GlottalEngine::new(fs_int),
            tract: ResonantTract::new(fs_int),
            nasal: NasalBranch::new(fs_int),
            fric_res: FricResonator::default(),
            fric_rng: Rng::new(0x243f_6a88),
            dc_x: 0.0,
            dc_y: 0.0,
            dc_coef: exp((-2.0 * PI * 22.0) / fs_out),
            last_telemetry: VocalBoxTelemetry::default(),
        }
    }

    /// Construct a [`VocalBoxCore`] pre-configured to a continuous [`BodyMix`].
    pub fn with_body(sample_rate: f64, body: impl Into<BodyMix>) -> Self {
        let mut core = Self::new(sample_rate);
        let frame = VocalParams::from_body(body).build();
        core.interp.reset(frame);
        core
    }

    #[inline]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    #[inline]
    pub fn internal_sample_rate(&self) -> f64 {
        self.decimator.fs_int()
    }

    #[inline]
    pub fn telemetry(&self) -> VocalBoxTelemetry {
        self.last_telemetry
    }

    #[inline]
    pub fn current_frame(&self) -> VocalFrame {
        self.interp.current()
    }

    /// Reconfigure output sample rate on the fly with **zero heap allocations**.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        let fs_out = sanitize_clamped(sample_rate, 48_000.0, 4_000.0, 384_000.0);
        if (fs_out - self.sample_rate).abs() < 1e-6 {
            return;
        }
        self.sample_rate = fs_out;
        self.decimator.set_sample_rate(fs_out);
        let fs_int = self.decimator.fs_int();
        self.glottis.set_sample_rate(fs_int);
        self.tract.set_sample_rate(fs_int);
        self.nasal.set_sample_rate(fs_int);
        self.dc_coef = exp((-2.0 * PI * 22.0) / fs_out);
    }

    /// Deterministically reset all internal states, phases, PRNGs, and filters to power-on state.
    pub fn reset(&mut self) {
        let fs = self.sample_rate;
        *self = Self::new(fs);
    }

    /// Snap immediately to `frame` with zero ramp.
    pub fn set_frame_immediate(&mut self, frame: VocalFrame) {
        self.interp.reset(frame);
    }

    /// Ramp smoothly from the current frame to `target` over `ramp_seconds` at audio rate.
    pub fn set_target_frame(&mut self, target: VocalFrame, ramp_seconds: f64) {
        let r_s = sanitize_clamped(ramp_seconds, 0.0, 0.0, 10.0);
        let n = (r_s * self.sample_rate).round() as usize;
        self.interp.set_target(target, n);
    }

    /// Step one output sample using the internal audio-rate [`FrameInterpolator`].
    #[inline]
    pub fn step(&mut self) -> f32 {
        let frame = self.interp.step();
        self.step_frame_inner(&frame)
    }

    /// Step one output sample driven directly by caller-supplied `frame` (also updates interpolator state).
    #[inline]
    pub fn step_frame(&mut self, frame: &VocalFrame) -> f32 {
        let clean = frame.sanitized();
        self.interp.reset(clean);
        self.step_frame_inner(&clean)
    }

    /// Render a block of samples; bit-identical regardless of `out.len()` block partitioning.
    pub fn render(&mut self, out: &mut [f32]) {
        for sample in out.iter_mut() {
            *sample = self.step();
        }
    }

    #[inline]
    fn step_frame_inner(&mut self, f: &VocalFrame) -> f32 {
        let m = self.decimator.oversample();
        let fs_int = self.decimator.fs_int();
        let nyq = 0.465 * fs_int;

        let mut dec_out = 0.0;
        let mut last_glot = GlottalStep::default();
        let mut last_nasal = NasalOutput::default();

        for _ in 0..m {
            let glot = self.glottis.step(f);
            let oral_raw = self.tract.step(f, glot.excitation, glot.area_cm2);
            let nasal = self.nasal.step(f, glot.excitation, oral_raw);

            // Supraglottal obstacle / sibilant frication jet noise shaped by constriction place & VTL
            let fric = if f.frication_gain > 1e-5 && f.subglottal_pressure_cm_h2o > 0.4 {
                let n = self.fric_rng.gauss();
                let (f_loc_base, q_loc, gain_loc) = match f.constriction_place {
                    ConstrictionPlace::Alveolar => (6900.0, 3.2, 0.28),
                    ConstrictionPlace::Postalveolar => (3450.0, 2.6, 0.30),
                    ConstrictionPlace::Dental | ConstrictionPlace::Labiodental => (4800.0, 0.85, 0.14),
                    ConstrictionPlace::Bilabial => (1900.0, 0.9, 0.12),
                    ConstrictionPlace::Palatal => (4200.0, 2.2, 0.22),
                    ConstrictionPlace::Velar | ConstrictionPlace::VelarBack => (2350.0, 1.8, 0.18),
                    ConstrictionPlace::Uvular => (1750.0, 1.5, 0.18),
                    ConstrictionPlace::None | ConstrictionPlace::Glottal => (2600.0, 0.9, 0.12),
                };
                let f_loc = clamp(f_loc_base / f.vtl_scale, 400.0, nyq);
                let shaped = self.fric_res.step_bandpass(n, f_loc, q_loc, fs_int);
                let p_scale = clamp(f.subglottal_pressure_cm_h2o / 8.0, 0.0, 2.0);
                shaped * gain_loc * f.frication_gain * p_scale
            } else {
                let _ = self.fric_res.step_bandpass(0.0, 3000.0, 1.0, fs_int);
                0.0
            };

            dec_out = self.decimator.push(nasal.combined + fric);
            last_glot = glot;
            last_nasal = nasal;
        }

        // Smooth C^1 soft-knee safety limiter BEFORE the DC blocker so asymmetric high-pitch peaks
        // never rectify into DC offset, followed by 28 Hz highpass DC blocker and ±0.999 clamp.
        let dec_lim = soft_limit_099(dec_out);
        let y_dc = self.dc_coef * (self.dc_y + dec_lim - self.dc_x);
        self.dc_x = flush_subnormal(dec_lim);
        self.dc_y = flush_subnormal(y_dc);
        let y_lim = clamp(self.dc_y, -0.999, 0.999);

        self.last_telemetry = VocalBoxTelemetry {
            f0_now_hz: last_glot.f0_now_hz,
            voicing: last_glot.voicing,
            glottal_area_cm2: last_glot.area_cm2,
            glottal_flow_cm3_s: last_glot.flow_cm3_s,
            effective_f1_hz: self.tract.last_effective_f1_hz(),
            effective_b1_hz: self.tract.last_effective_b1_hz(),
            raw_tract_out: last_nasal.combined,
            nasal_out: last_nasal.nasal_out,
            velum_port_cm2: f.velum_port_cm2,
        };

        if y_lim.is_finite() { y_lim as f32 } else { 0.0 }
    }
}

/// Transparent below `|x| = 0.65`, $C^1$-smoothly saturating to `±0.85` for extreme inputs.
#[inline(always)]
fn soft_limit_099(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    let ax = x.abs();
    if ax <= 0.65 {
        flush_subnormal(x)
    } else {
        let excess = (ax - 0.65) / 0.20;
        let mag = 0.65 + 0.20 * tanh(excess);
        if x.is_sign_negative() { -mag } else { mag }
    }
}
