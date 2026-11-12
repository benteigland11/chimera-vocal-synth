//! Zero-allocation $N=8$ Chebyshev Type-II oversampled anti-aliasing decimator.
//!
//! Synthesizes internally at $f_{s,\text{int}} = M \cdot f_{s,\text{out}} \ge 96\text{ kHz}$
//! ($M = \lceil 96000 / f_{s,\text{out}} \rceil \in [2, 32]$) and decimates to any arbitrary
//! device rate (`8 kHz .. 192 kHz`) with $0.00\text{ dB}$ DC/passband ripple and $\ge 80\text{ dB}$
//! stopband rejection above $0.49 \cdot f_{s,\text{out}}$.

use super::vowels::{flush_subnormal, sanitize_clamped};
use crate::math::{PI, cos, cosh, ln, sin, sinh, sqrt, tan};

const N_SECTIONS: usize = 4; // Order N = 8

#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    #[inline(always)]
    fn step(&mut self, x: f64) -> f64 {
        let x = if x.is_finite() { x } else { 0.0 };
        let y = self.b0 * x + self.z1;
        let z1_next = self.b1 * x - self.a1 * y + self.z2;
        let z2_next = self.b2 * x - self.a2 * y;
        self.z1 = flush_subnormal(z1_next);
        self.z2 = flush_subnormal(z2_next);
        flush_subnormal(y)
    }

    #[inline]
    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

/// Stack-resident $N=8$ Chebyshev Type-II anti-aliasing decimator.
#[derive(Clone, Debug)]
pub struct Cheby2Decimator {
    oversample: usize,
    fs_out: f64,
    fs_int: f64,
    sections: [Biquad; N_SECTIONS],
}

/// Choose integer oversampling factor $M \in [2, 32]$ placing $f_{s,\text{int}} \ge 96\text{ kHz}$.
#[inline]
pub fn oversample_factor_for(fs_out: f64) -> usize {
    let fs = sanitize_clamped(fs_out, 48_000.0, 4_000.0, 384_000.0);
    let m = (96_000.0 / fs).ceil() as usize;
    m.clamp(2, 32)
}

impl Cheby2Decimator {
    pub fn new(fs_out: f64) -> Self {
        let fs = sanitize_clamped(fs_out, 48_000.0, 4_000.0, 384_000.0);
        let m = oversample_factor_for(fs);
        let mut dec = Self {
            oversample: m,
            fs_out: fs,
            fs_int: fs * (m as f64),
            sections: [Biquad::default(); N_SECTIONS],
        };
        dec.design_coeffs(m);
        dec
    }

    #[inline]
    pub fn oversample(&self) -> usize {
        self.oversample
    }

    #[inline]
    pub fn fs_out(&self) -> f64 {
        self.fs_out
    }

    #[inline]
    pub fn fs_int(&self) -> f64 {
        self.fs_int
    }

    /// Reconfigure output sample rate with zero heap allocations.
    pub fn set_sample_rate(&mut self, fs_out: f64) {
        let fs = sanitize_clamped(fs_out, 48_000.0, 4_000.0, 384_000.0);
        let m = oversample_factor_for(fs);
        let m_changed = m != self.oversample;
        self.fs_out = fs;
        self.oversample = m;
        self.fs_int = fs * (m as f64);
        if m_changed {
            self.design_coeffs(m);
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.sections {
            s.reset();
        }
    }

    /// Exact analytic bilinear design for $N=8$ Chebyshev Type-II lowpass with
    /// $R_s = 80\text{ dB}$ stopband attenuation at $f_{\text{stop}} = 0.49 \cdot f_{s,\text{out}}$.
    fn design_coeffs(&mut self, m: usize) {
        // 80 dB stopband attenuation -> 10^(80/10) - 1 = 1e8 - 1 -> inv_eps = sqrt(1e8 - 1)
        let inv_eps = sqrt(1.0e8 - 1.0);
        // arsinh(inv_eps) / N
        let mu = ln(inv_eps + sqrt(inv_eps * inv_eps + 1.0)) / 8.0;
        let sh = sinh(mu);
        let ch = cosh(mu);

        // Digital stopband edge: w_s = 2 * pi * (0.49 * fs_out) / (M * fs_out) = 2 * pi * 0.49 / M
        let omega_s = tan(PI * 0.49 / (m as f64));

        for (k, sec) in self.sections.iter_mut().enumerate() {
            let theta = PI * ((2 * k + 1) as f64) / 16.0;
            let st = sin(theta);
            let ct = cos(theta);

            // Prototype Chebyshev-I pole
            let sig = -sh * st;
            let om = ch * ct;
            let r2 = sig * sig + om * om;

            // Inverse Chebyshev (Type-II) analog pole & jw-axis transmission zero
            let p_re = omega_s * (sig / r2);
            let p_im = omega_s * (om / r2);
            let p_mag2 = p_re * p_re + p_im * p_im;
            let z_im = omega_s / ct;
            let z_im2 = z_im * z_im;

            // Bilinear transform s = (1 - z^-1) / (1 + z^-1)
            let a0 = 1.0 - 2.0 * p_re + p_mag2;
            let a1 = 2.0 * (p_mag2 - 1.0) / a0;
            let a2 = (1.0 + 2.0 * p_re + p_mag2) / a0;

            let b0_raw = 1.0 + z_im2;
            let b1_raw = 2.0 * (z_im2 - 1.0);
            let b2_raw = 1.0 + z_im2;

            // Exact unity DC gain per biquad section (0.00 dB passband ripple at DC)
            let dc_num = b0_raw + b1_raw + b2_raw;
            let dc_den = 1.0 + a1 + a2;
            let g = dc_den / dc_num;

            sec.b0 = b0_raw * g;
            sec.b1 = b1_raw * g;
            sec.b2 = b2_raw * g;
            sec.a1 = a1;
            sec.a2 = a2;
        }
    }

    /// Push one internal-rate sample through the 8th-order cascade.
    #[inline(always)]
    pub fn push(&mut self, mut x: f64) -> f64 {
        for sec in &mut self.sections {
            x = sec.step(x);
        }
        x
    }
}
