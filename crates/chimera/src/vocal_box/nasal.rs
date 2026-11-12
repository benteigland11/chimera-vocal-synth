//! Coupled velopharyngeal nasal tract (`NasalBranch`):
//! - Exact `0.0` nasal leakage when `velum_port_cm2 <= 0.0` (with click-free ring-down when closing),
//! - Fixed nasal cavity murmur resonances ($F_{N1} \approx 260\text{ Hz}$, $F_{N2} \approx 1020\text{ Hz}$, $F_{N3} \approx 2150\text{ Hz}$),
//! - Place-ordered oral-shunt side-branch anti-formant zeros (`/m/ < /n/ < /ɲ/ < /ŋ/`),
//! - Full standalone sanitization against `NaN`, `±Inf`, and extreme float inputs.

use super::vowels::{ConstrictionPlace, VocalFrame, flush_subnormal, sanitize, sanitize_clamped};
use crate::math::{PI, TAU, clamp, cos, exp, lerp, pow};

/// Radiated oral, nasal, and summed lip+nostril acoustic pressure from [`NasalBranch`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NasalOutput {
    pub oral_out: f64,
    pub nasal_out: f64,
    pub combined: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct NasalPole {
    z1: f64,
    z2: f64,
}

impl NasalPole {
    #[inline(always)]
    fn step(&mut self, x: f64, f_hz: f64, bw_hz: f64, fs: f64) -> f64 {
        let r = exp(-PI * bw_hz / fs);
        let a1 = -2.0 * r * cos(TAU * f_hz / fs);
        let a2 = r * r;
        let g = 1.0 + a1 + a2;
        let y = g * x - a1 * self.z1 - a2 * self.z2;
        let yc = flush_subnormal(y);
        self.z2 = self.z1;
        self.z1 = yc;
        yc
    }

    #[inline(always)]
    fn energy(&self) -> f64 {
        self.z1.abs() + self.z2.abs()
    }

    #[inline]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ShuntZero {
    z1: f64,
    z2: f64,
}

impl ShuntZero {
    #[inline(always)]
    fn step(&mut self, x: f64, f_zero_hz: f64, depth: f64, fs: f64) -> f64 {
        let w0 = TAU * f_zero_hz / fs;
        let cw = cos(w0);
        let r_p = exp(-PI * 420.0 / fs);
        let r_z = lerp(r_p, 0.985, clamp(depth, 0.0, 0.98));

        let b0_raw = 1.0;
        let b1_raw = -2.0 * r_z * cw;
        let b2_raw = r_z * r_z;
        let a1 = -2.0 * r_p * cw;
        let a2 = r_p * r_p;

        let dc_num = (b0_raw + b1_raw + b2_raw).max(1e-6);
        let dc_den = 1.0 + a1 + a2;
        let g = dc_den / dc_num;

        let y = (b0_raw * g) * x + self.z1;
        let z1n = (b1_raw * g) * x - a1 * y + self.z2;
        let z2n = (b2_raw * g) * x - a2 * y;
        self.z1 = flush_subnormal(z1n);
        self.z2 = flush_subnormal(z2n);
        flush_subnormal(y)
    }

    #[inline(always)]
    fn energy(&self) -> f64 {
        self.z1.abs() + self.z2.abs()
    }

    #[inline]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Zero-allocation coupled nasal branch & oral-shunt anti-formant filter.
#[derive(Clone, Debug)]
pub struct NasalBranch {
    fs: f64,
    velum_sm: f64,
    oral_gate_sm: f64,
    poles: [NasalPole; 3],
    shunt: ShuntZero,
}

impl NasalBranch {
    pub fn new(fs: f64) -> Self {
        Self {
            fs: sanitize_clamped(fs, 96_000.0, 8_000.0, 384_000.0),
            velum_sm: 0.0,
            oral_gate_sm: 1.0,
            poles: [NasalPole::default(); 3],
            shunt: ShuntZero::default(),
        }
    }

    pub fn set_sample_rate(&mut self, fs: f64) {
        self.fs = sanitize_clamped(fs, 96_000.0, 8_000.0, 384_000.0);
    }

    pub fn reset(&mut self) {
        self.velum_sm = 0.0;
        self.oral_gate_sm = 1.0;
        for p in &mut self.poles {
            p.reset();
        }
        self.shunt.reset();
    }

    /// Place-specific oral cul-de-sac quarter-wave anti-formant frequency (Hz, adult male base).
    /// Shorter anterior oral cavities produce higher shunt zero frequencies:
    /// `/m/` (~790 Hz) < `/n/` (~1530 Hz) < `/ɲ/` (~2170 Hz) < `/ŋ/` (~2810 Hz).
    #[inline]
    fn shunt_zero_base_hz(place: ConstrictionPlace) -> Option<f64> {
        match place {
            ConstrictionPlace::None | ConstrictionPlace::Glottal => None,
            ConstrictionPlace::Bilabial => Some(790.0),
            ConstrictionPlace::Labiodental => Some(1060.0),
            ConstrictionPlace::Dental => Some(1340.0),
            ConstrictionPlace::Alveolar => Some(1530.0),
            ConstrictionPlace::Postalveolar => Some(1850.0),
            ConstrictionPlace::Palatal => Some(2170.0),
            ConstrictionPlace::Velar | ConstrictionPlace::VelarBack => Some(2810.0),
            ConstrictionPlace::Uvular => Some(3250.0),
        }
    }

    /// Step coupled nasal & oral radiation. Sanitizes all inputs so standalone calls with `NaN`,
    /// `±Inf`, or extreme amplitudes remain strictly finite and bounded (`<= 1.9`).
    #[inline]
    pub fn step(&mut self, raw_frame: &VocalFrame, raw_exc: f64, raw_oral: f64) -> NasalOutput {
        let f = raw_frame.sanitized();
        let fs = self.fs;
        let nyq = 0.465 * fs;

        let exc = sanitize_clamped(raw_exc, 0.0, -1.9, 1.9);
        let oral_in = sanitize_clamped(raw_oral, 0.0, -1.8, 1.8);

        let velum_raw = sanitize(raw_frame.velum_port_cm2, 0.0);
        let velum_target = if velum_raw <= 1e-7 { 0.0 } else { clamp(velum_raw, 0.0, 2.5) };

        // Oral occlusion gate: full oral closure (area -> 0) blocks oral radiation during stops/nasals
        let area_norm = clamp(f.constriction_area_cm2 / 0.35, 0.0, 1.0);
        let oral_target_gate = if f.constriction_place != ConstrictionPlace::None {
            area_norm
        } else {
            1.0
        };
        let alpha = 1.0 - exp(-1.0 / (0.0022 * fs));
        self.oral_gate_sm = flush_subnormal(self.oral_gate_sm + alpha * (oral_target_gate - self.oral_gate_sm));
        let oral_out = clamp(flush_subnormal(oral_in * self.oral_gate_sm), -1.85, 1.85);

        // Smooth velum port opening/closing (tau ~ 2.2 ms) and snap bit-exact to 0.0 once ring-down completes
        self.velum_sm = flush_subnormal(self.velum_sm + alpha * (velum_target - self.velum_sm));
        let state_e = self.poles[0].energy()
            + self.poles[1].energy()
            + self.poles[2].energy()
            + self.shunt.energy();

        if velum_target == 0.0 && self.velum_sm < 1e-4 && state_e < 1e-6 {
            self.velum_sm = 0.0;
            for p in &mut self.poles {
                p.reset();
            }
            self.shunt.reset();
            return NasalOutput {
                oral_out,
                nasal_out: 0.0,
                combined: oral_out,
            };
        }

        if self.velum_sm == 0.0 && velum_target == 0.0 {
            return NasalOutput {
                oral_out,
                nasal_out: 0.0,
                combined: oral_out,
            };
        }

        // Port coupling conductance gain (monotonic in velum_port_cm2)
        let port_gain = clamp(self.velum_sm / (self.velum_sm + 0.42), 0.0, 0.95);
        let drive = exc * port_gain;

        let vtl_nasal = pow(f.vtl_scale, 0.65);
        let fn1 = clamp(265.0 / vtl_nasal, 180.0, nyq);
        let fn2 = clamp(1020.0 / vtl_nasal, 500.0, nyq);
        let fn3 = clamp(2180.0 / vtl_nasal, 1100.0, nyq);

        let mut n = self.poles[0].step(drive, fn1, 72.0, fs);
        n = self.poles[1].step(n, fn2, 135.0, fs);
        n = self.poles[2].step(n, fn3, 210.0, fs);

        // Oral cul-de-sac shunt anti-formant notch
        if let Some(base_zero_hz) = Self::shunt_zero_base_hz(f.constriction_place) {
            let f_zero = (base_zero_hz / f.vtl_scale).min(nyq);
            let closure_depth = 1.0 - clamp(f.constriction_area_cm2 / 1.2, 0.0, 0.75);
            n = self.shunt.step(n, f_zero, 0.92 * closure_depth, fs);
        } else {
            // Nasalized vowel: mild oral-nasal coupling anti-resonance near 820 Hz
            let f_zero = (820.0 / f.vtl_scale).min(nyq);
            n = self.shunt.step(n, f_zero, 0.35 * port_gain, fs);
        }

        // Attenuate ring-down tail smoothly so closing velum_port 0.65 -> 0.0 settles to bit-exact 0.0 within ~20 ms
        let tail_gate = clamp(self.velum_sm * 200.0, 0.0, 1.0);
        let nasal_out = clamp(flush_subnormal(n * 0.62 * tail_gate), -1.85, 1.85);
        // Slightly reduce oral radiation as acoustic flow diverts into open velopharyngeal port
        let oral_diverted = oral_out * (1.0 - 0.28 * port_gain);
        let combined = clamp(flush_subnormal(oral_diverted + nasal_out), -1.90, 1.90);

        NasalOutput {
            oral_out: flush_subnormal(oral_diverted),
            nasal_out,
            combined,
        }
    }
}
