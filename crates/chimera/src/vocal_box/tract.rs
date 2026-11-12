//! Coupled vocal-tract resonator engine (`ResonantTract`):
//! - 5-formant cascade with Fant higher-pole behavior and physiological radiation damping,
//! - Causal open-phase glottal impedance coupling ($\Delta F_1(A_g) > 0$, $\Delta B_1(A_g) > 0$),
//! - Soprano *aggiustamento* ($F_1 \ge 1.08 f_0$ via [`soft_max_c1`]) with high-$f_0$ headroom safety,
//! - Continuous epilarynx singer's formant ring (`2.85–3.35 kHz`),
//! - Continuous piriform sinus side-branch anti-resonance (`~4600 Hz / vtl_scale`),
//! - Lateral `/l/` anti-formant notch (`~2550 Hz / vtl_scale`) and rhotic `/ɹ, ɝ/` $F_3$ lowering.

use super::vowels::{VocalFrame, flush_subnormal, sanitize_clamped, soft_max_c1};
use crate::math::{PI, TAU, clamp, cos, exp, lerp, pow, sin};

#[derive(Clone, Copy, Debug, Default)]
struct PolePair {
    z1: f64,
    z2: f64,
}

impl PolePair {
    #[inline(always)]
    fn step(&mut self, x: f64, f_hz: f64, bw_hz: f64, fs: f64) -> f64 {
        let r = exp(-PI * bw_hz / fs);
        let theta = TAU * f_hz / fs;
        let a1 = -2.0 * r * cos(theta);
        let a2 = r * r;
        let g = 1.0 + a1 + a2; // Exact unity DC gain
        let y = g * x - a1 * self.z1 - a2 * self.z2;
        let y_clean = flush_subnormal(y);
        self.z2 = self.z1;
        self.z1 = y_clean;
        y_clean
    }

    #[inline]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct BiquadState {
    z1: f64,
    z2: f64,
}

impl BiquadState {
    #[inline(always)]
    fn step(&mut self, x: f64, b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> f64 {
        let y = b0 * x + self.z1;
        let z1n = b1 * x - a1 * y + self.z2;
        let z2n = b2 * x - a2 * y;
        self.z1 = flush_subnormal(z1n);
        self.z2 = flush_subnormal(z2n);
        flush_subnormal(y)
    }

    #[inline]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Zero-allocation coupled supraglottal oral & pharyngeal resonator cascade.
#[derive(Clone, Debug)]
pub struct ResonantTract {
    fs: f64,
    poles: [PolePair; 8],
    epilarynx: BiquadState,
    piriform: BiquadState,
    lateral: BiquadState,
    hpc: BiquadState,
    cross_hp: BiquadState,
    cross_hp2: BiquadState,
    cross_hp3: BiquadState,
    last_f1_hz: f64,
    last_b1_hz: f64,
}

impl ResonantTract {
    pub fn new(fs: f64) -> Self {
        Self {
            fs: sanitize_clamped(fs, 96_000.0, 8_000.0, 384_000.0),
            poles: [PolePair::default(); 8],
            epilarynx: BiquadState::default(),
            piriform: BiquadState::default(),
            lateral: BiquadState::default(),
            hpc: BiquadState::default(),
            cross_hp: BiquadState::default(),
            cross_hp2: BiquadState::default(),
            cross_hp3: BiquadState::default(),
            last_f1_hz: 500.0,
            last_b1_hz: 76.0,
        }
    }

    #[inline]
    pub fn last_effective_f1_hz(&self) -> f64 {
        self.last_f1_hz
    }

    #[inline]
    pub fn last_effective_b1_hz(&self) -> f64 {
        self.last_b1_hz
    }

    pub fn set_sample_rate(&mut self, fs: f64) {
        self.fs = sanitize_clamped(fs, 96_000.0, 8_000.0, 384_000.0);
    }

    pub fn reset(&mut self) {
        for p in &mut self.poles {
            p.reset();
        }
        self.epilarynx.reset();
        self.piriform.reset();
        self.lateral.reset();
        self.hpc.reset();
        self.cross_hp.reset();
        self.cross_hp2.reset();
        self.cross_hp3.reset();
        self.last_f1_hz = 500.0;
        self.last_b1_hz = 76.0;
    }

    /// Advance one sample of the coupled supraglottal vocal tract.
    ///
    /// `glottal_area_cm2` is the instantaneous glottal opening $A_g(t) \ge 0$, which causally
    /// couples subglottal impedance into $F_1$ and $B_1$ during the open phase.
    #[inline]
    pub fn step(&mut self, raw_frame: &VocalFrame, excitation: f64, glottal_area_cm2: f64) -> f64 {
        let f = raw_frame.sanitized();
        let fs = self.fs;
        let nyq_guard = 0.465 * fs;

        let exc = sanitize_clamped(excitation, 0.0, -2.5, 2.5);
        let ag = sanitize_clamped(glottal_area_cm2, 0.0, 0.0, 1.5);

        // 1. Soprano aggiustamento: smoothly raise F1 so F1 >= 1.08 * f0
        let f1_base = soft_max_c1(f.formants_hz[0], 1.08 * f.f0_hz, 45.0);

        // 2. Causal open-phase glottal impedance coupling (Barney & Shadle 2000; Stevens 1998):
        // Finite glottal opening Ag shifts F1 upward via parallel glottal inertance and widens B1
        // via resistive transglottal damping.
        let ag_shift = ag / (ag + 0.14);
        let f1_coupled = f1_base * (1.0 + 0.030 * ag_shift);
        let db1_glottal = 82.0 * (ag / (ag + 0.16));

        // 3. Physiological frequency-squared lip-radiation damping & high-f0 resonance damping:
        // Prevents unnatural high-Q resonance overflow when soprano aggiustamento locks F1 onto H1!
        let agg_lift_hz = (f1_base - f.formants_hz[0]).max(0.0);
        let db1_soprano = 0.24 * agg_lift_hz
            + if f.f0_hz > 280.0 {
                55.0 * pow((f1_coupled / 1000.0).max(0.0), 2.0) + 0.12 * (f.f0_hz - 280.0)
            } else {
                10.0 * pow((f1_coupled / 1000.0).max(0.0), 2.0)
            };

        let mut freqs = f.formants_hz;
        let mut bws = f.bandwidths_hz;
        freqs[0] = f1_coupled;
        bws[0] = clamp(bws[0] + db1_glottal + db1_soprano, 35.0, 650.0);
        bws[1] = clamp(bws[1] + 0.22 * db1_glottal, 45.0, 700.0);

        // 4. Rhotic F3 lowering (/ɹ, ɝ/)
        if f.rhotic_mix > 0.0 {
            let rhotic_target = (1680.0 / f.vtl_scale).max(freqs[1] + 220.0);
            freqs[2] = lerp(freqs[2], freqs[2].min(rhotic_target), f.rhotic_mix);
        }

        // Enforce monotonic formant spacing and Nyquist safety at any sample rate
        for i in 0..5 {
            if i > 0 && freqs[i] < freqs[i - 1] + 140.0 {
                freqs[i] = freqs[i - 1] + 140.0;
            }
            freqs[i] = clamp(freqs[i], 140.0, nyq_guard);
        }
        self.last_f1_hz = freqs[0];
        self.last_b1_hz = bws[0];

        // 5. Step 5 controllable formants + 3 higher distributed-tube modes (F6..F8) when below Nyquist
        let mut y = exc;
        for i in 0..5 {
            y = self.poles[i].step(y, freqs[i], bws[i], fs);
        }
        let higher_modes = [
            (5500.0 / f.vtl_scale, 320.0),
            (6550.0 / f.vtl_scale, 410.0),
            (7650.0 / f.vtl_scale, 520.0),
        ];
        let mut prev_hi = freqs[4];
        for (idx, &(fh_raw, bwh)) in higher_modes.iter().enumerate() {
            let fh = fh_raw.max(prev_hi + 350.0);
            if fh < nyq_guard {
                y = self.poles[5 + idx].step(y, fh, bwh, fs);
                prev_hi = fh;
            }
        }

        // 6. Epilarynx narrow-tube singer's formant resonance (2.85–3.35 kHz cluster)
        if f.epilarynx_ring > 1e-5 {
            let f_epi = clamp(3060.0 / pow(f.vtl_scale, 0.30), 2400.0, nyq_guard);
            // Peaking EQ biquad with up to +16 dB boost at ring = 1.0
            let gain_lin = pow(10.0, (16.0 * f.epilarynx_ring) / 40.0); // sqrt(10^(dB/20))
            let w0 = TAU * f_epi / fs;
            let alpha = sin(w0) / (2.0 * 4.8);
            let cw = cos(w0);
            let a0 = 1.0 + alpha / gain_lin;
            let b0 = (1.0 + alpha * gain_lin) / a0;
            let b1 = (-2.0 * cw) / a0;
            let b2 = (1.0 - alpha * gain_lin) / a0;
            let a1 = b1;
            let a2 = (1.0 - alpha / gain_lin) / a0;
            y = self.epilarynx.step(y, b0, b1, b2, a1, a2);
        } else {
            // Step unity pass-through so filter memory stays warm without discontinuity
            y = self.epilarynx.step(y, 1.0, 0.0, 0.0, 0.0, 0.0);
        }

        // 7. Piriform sinus side-branch spectral zero (~4600 Hz / vtl_scale -> 5066 Hz @ vtl=0.908)
        let f_pir = 4600.0 / f.vtl_scale;
        if f.piriform_depth > 1e-5 && f_pir < nyq_guard {
            y = step_pole_zero_notch(&mut self.piriform, y, f_pir, 760.0, f.piriform_depth * 0.92, fs);
        } else {
            y = self.piriform.step(y, 1.0, 0.0, 0.0, 0.0, 0.0);
        }

        // 8. Lateral /l/ tongue-side channel anti-formant (~2550 Hz / vtl_scale)
        let f_lat = 2550.0 / f.vtl_scale;
        if f.lateral_mix > 1e-5 && f_lat < nyq_guard {
            y = step_pole_zero_notch(&mut self.lateral, y, f_lat, 640.0, f.lateral_mix * 0.92, fs);
        } else {
            y = self.lateral.step(y, 1.0, 0.0, 0.0, 0.0, 0.0);
        }

        // 9. Fant (1959) Higher-Pole Correction (HPC) + spherical-baffle radiation equalization:
        let fz_hpc = (1020.0 / pow(f.vtl_scale, 0.35)).min(0.35 * fs);
        let fp_hpc = (9200.0 / pow(f.vtl_scale, 0.35)).min(0.44 * fs);
        let wz = crate::math::tan(PI * fz_hpc / fs);
        let wp = crate::math::tan(PI * fp_hpc / fs);
        let qz = 0.60;
        let qp = 0.72;
        let az0 = 1.0 + wz / qz + wz * wz;
        let ap0 = 1.0 + wp / qp + wp * wp;
        let g_dc = (wp * wp) / (wz * wz);
        let b0_hpc = (az0 / ap0) * g_dc;
        let b1_hpc = (2.0 * (wz * wz - 1.0) / ap0) * g_dc;
        let b2_hpc = ((1.0 - wz / qz + wz * wz) / ap0) * g_dc;
        let a1_hpc = 2.0 * (wp * wp - 1.0) / ap0;
        let a2_hpc = (1.0 - wp / qp + wp * wp) / ap0;
        let out_1d = self.hpc.step(y, b0_hpc, b1_hpc, b2_hpc, a1_hpc, a2_hpc);

        // 10. 6th-order highpass above ~7.2 kHz transverse cross-mode cut-on: transmits voiced
        // 8 kHz shimmer (matching Byrne et al. 1994 -20 dB norm) with >23 dB isolation from the
        // 4.6 kHz piriform sinus trough and zero effect on whisper.
        let f_cm = (7200.0 / pow(f.vtl_scale, 0.5)).min(0.43 * fs);
        let cross = if !f.whisper && f_cm < nyq_guard && fs >= 32_000.0 {
            let wc = crate::math::tan(PI * f_cm / fs);
            let s2_const = core::f64::consts::SQRT_2;
            let ac0 = 1.0 + s2_const * wc + wc * wc;
            let bc0 = 1.0 / ac0;
            let bc1 = -2.0 / ac0;
            let bc2 = 1.0 / ac0;
            let ac1 = 2.0 * (wc * wc - 1.0) / ac0;
            let ac2 = (1.0 - s2_const * wc + wc * wc) / ac0;
            let s1 = self.cross_hp.step(exc, bc0, bc1, bc2, ac1, ac2);
            let s2 = self.cross_hp2.step(s1, bc0, bc1, bc2, ac1, ac2);
            clamp(7.6 * self.cross_hp3.step(s2, bc0, bc1, bc2, ac1, ac2), -0.060, 0.060)
        } else {
            let _ = self.cross_hp.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let _ = self.cross_hp2.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let _ = self.cross_hp3.step(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            0.0
        };

        // Soprano / high-energy resonator headroom guard so even pathological standalone calls stay <= 1.85
        let scaled = (out_1d + cross) * 0.34;
        clamp(flush_subnormal(scaled), -1.85, 1.85)
    }
}

/// Pole-zero acoustic side-branch notch with unity DC gain and sample-rate-invariant zero depth.
#[inline]
fn step_pole_zero_notch(
    st: &mut BiquadState,
    x: f64,
    f_notch_hz: f64,
    bw_pole_hz: f64,
    depth: f64,
    fs: f64,
) -> f64 {
    let w0 = TAU * f_notch_hz / fs;
    let cw = cos(w0);
    let r_p = exp(-PI * bw_pole_hz / fs);
    let bw_zero_hz = lerp(bw_pole_hz, 42.0, clamp(depth, 0.0, 0.98));
    let r_z = exp(-PI * bw_zero_hz / fs);

    let b0_raw = 1.0;
    let b1_raw = -2.0 * r_z * cw;
    let b2_raw = r_z * r_z;
    let a1 = -2.0 * r_p * cw;
    let a2 = r_p * r_p;

    let dc_num = (b0_raw + b1_raw + b2_raw).max(1e-6);
    let dc_den = 1.0 + a1 + a2;
    let g = dc_den / dc_num;

    st.step(x, b0_raw * g, b1_raw * g, b2_raw * g, a1, a2)
}
