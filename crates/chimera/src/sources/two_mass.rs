//! Coupled two-mass self-oscillating vocal-fold source (Ishizaka & Flanagan 1972; Titze 1988).
//!
//! Models the lower ($m_1$) and upper ($m_2$) vocal-fold masses coupled by shear stiffness $k_c$
//! and driven by transglottal pressure $\Delta P = P_{\text{sub}} - P_{\text{supra}}$.
//! Self-sustained oscillation emerges from mucosal-wave phase delay ($x_1$ leads $x_2$):
//! the glottal channel is convergent during opening (high intraglottal pressure pushing folds
//! apart) and divergent during closing (flow separation / Bernoulli drop letting recoil snap them
//! shut). Below phonation threshold pressure or under wide abduction, tissue damping dominates and
//! vibration decays smoothly to rest.
//!
//! Real-time contract: zero heap allocations, deterministic across platforms via [`crate::math`].

use crate::math::{
    Drift, Rng, TAU, atan2, clamp, exp, js_max, js_min, lerp, pow, sin, smoothstep, sqrt,
};
use crate::physics::CM_H2O;
use crate::sources::glottis::{DEFAULT_GLOTTIS, GlottisParams};

/// Coupled two-mass vocal-fold model producing physical glottal area ($\text{cm}^2$).
#[derive(Clone, Debug)]
pub struct TwoMassGlottis {
    pub fs: f64,
    pub params: GlottisParams,
    /// Smoothed voicing level `0..1` (telemetry).
    pub voicing: f64,
    /// Instantaneous glottal orifice area ($\text{cm}^2$).
    pub area: f64,
    /// Instantaneous fundamental frequency (Hz, telemetry).
    pub f0_now: f64,
    /// Nominal phonation threshold pressure ($\text{dyn/cm}^2$).
    pub threshold: f64,

    // Lower (1) and upper (2) lateral displacements from pre-phonatory position (cm) & velocities (cm/s)
    x1: f64,
    v1: f64,
    x2: f64,
    v2: f64,

    // Cycle & pitch-servo tracking
    prev_open: bool,
    prev_amin: f64,
    period_acc: f64,
    q_trim: f64,
    measured_f0: f64,
    phase_est: f64,
    phase_lag_sm: f64,
    cycle_index: u64,
    cycle_amp: f64,
    cycle_period_scale: f64,
    chaos_x: f64,

    // Slow modulations
    vib_phase: f64,
    rng: Rng,
    drift: Drift,
    drift_value: f64,
    drift_counter: i32,
    pitch_factor: f64,

    // Smoothed aerodynamic & control states
    dp_sm: f64,
    dp_coef: f64,
    voicing_sm: f64,
    voice_coef: f64,
    abduct_sm: f64,
    closure_sm: f64,
    amp_sm: f64,
    sm_coef: f64,
}

impl TwoMassGlottis {
    /// Create a coupled two-mass vocal-fold oscillator running at sample rate `fs` (Hz).
    pub fn new(fs: f64) -> Self {
        let fs_clean = if fs.is_finite() && fs >= 4_000.0 {
            fs
        } else {
            48_000.0
        };
        Self {
            fs: fs_clean,
            params: DEFAULT_GLOTTIS,
            voicing: 0.0,
            area: 0.0,
            f0_now: DEFAULT_GLOTTIS.f0,
            threshold: 2.2 * CM_H2O,
            x1: 0.0020,
            v1: 0.0,
            x2: 0.0,
            v2: 0.0,
            prev_open: true,
            prev_amin: 0.015,
            period_acc: 0.0,
            q_trim: 1.0,
            measured_f0: DEFAULT_GLOTTIS.f0,
            phase_est: 0.0,
            phase_lag_sm: 0.85,
            cycle_index: 0,
            cycle_amp: 1.0,
            cycle_period_scale: 1.0,
            chaos_x: 0.37,
            vib_phase: 0.0,
            rng: Rng::new(0x51ed_2042),
            drift: Drift::new(fs_clean / 32.0, 0.3, 4242),
            drift_value: 0.0,
            drift_counter: 0,
            pitch_factor: 1.0,
            dp_sm: 0.0,
            dp_coef: 1.0 - exp(-TAU * 45.0 / fs_clean),
            voicing_sm: 0.0,
            voice_coef: 1.0 - exp(-TAU * 25.0 / fs_clean),
            abduct_sm: 0.0,
            closure_sm: 0.0,
            amp_sm: DEFAULT_GLOTTIS.amplitude,
            sm_coef: 1.0 - exp(-1.0 / (0.004 * fs_clean)),
        }
    }

    /// Reset displacements, velocities, and smoothers to initial rest state.
    pub fn reset(&mut self) {
        self.x1 = 0.0020;
        self.v1 = 0.0;
        self.x2 = 0.0;
        self.v2 = 0.0;
        self.prev_open = true;
        self.prev_amin = 0.015;
        self.period_acc = 0.0;
        self.q_trim = 1.0;
        self.measured_f0 = self.params.f0;
        self.phase_est = 0.0;
        self.phase_lag_sm = 0.85;
        self.cycle_index = 0;
        self.cycle_amp = 1.0;
        self.cycle_period_scale = 1.0;
        self.voicing = 0.0;
        self.voicing_sm = 0.0;
        self.dp_sm = 0.0;
        self.area = 0.0;
    }

    /// Lower and upper vocal-fold mass displacements `(x1_cm, x2_cm)` relative to pre-phonatory rest.
    #[inline]
    pub fn displacements_cm(&self) -> (f64, f64) {
        (self.x1, self.x2)
    }

    /// Smoothed mucosal-wave phase lag (radians in `0..TAU`) by which lower mass $x_1$ leads upper mass $x_2$.
    #[inline]
    pub fn phase_lag_rad(&self) -> f64 {
        self.phase_lag_sm
    }

    /// Instantaneous cycle phase `0..1` for scope synchronization.
    #[inline]
    pub fn cycle_phase(&self) -> f64 {
        self.phase_est
    }

    /// Alias matching [`crate::sources::glottis::ParametricGlottis::next`].
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn next(&mut self, dp: f64) -> f64 {
        self.step(dp)
    }

    /// Advance one sample given transglottal pressure `dp` ($\text{dyn/cm}^2$) and return glottal area ($\text{cm}^2$).
    #[inline]
    pub fn step(&mut self, dp: f64) -> f64 {
        let p = self.params;
        let dp_clean = if dp.is_finite() { dp } else { 0.0 };
        self.dp_sm += self.dp_coef * (dp_clean - self.dp_sm);

        // Slow pitch modulations (vibrato + drift) updated every 32 samples
        self.drift_counter -= 1;
        if self.drift_counter <= 0 {
            self.drift_counter = 32;
            self.drift_value = self.drift.step();
            self.vib_phase +=
                ((p.vibrato_rate * 32.0) / self.fs) * (1.0 + 0.04 * self.drift_value);
            if self.vib_phase >= 1.0 {
                self.vib_phase -= 1.0;
            }
            let cents = p.vibrato_depth * sin(TAU * self.vib_phase) + p.drift * self.drift_value;
            self.pitch_factor = pow(2.0, cents / 1200.0);
        }

        let mut f0_target = clamp(p.f0 * self.pitch_factor, 24.0, 1600.0);
        if p.fry > 0.0 {
            let fry_amt = clamp(p.fry, 0.0, 1.0);
            f0_target =
                f0_target * (1.0 - 0.62 * fry_amt) + js_min(f0_target, 52.0) * 0.62 * fry_amt;
        }

        // Smooth control parameters to prevent click discontinuities
        let sc = self.sm_coef;
        let effort_scale = 0.45 + 0.75 * clamp(p.effort, 0.0, 1.5);
        self.amp_sm += sc * (js_max(0.0, p.amplitude) * effort_scale - self.amp_sm);
        self.abduct_sm += sc * (js_max(0.0, p.abduction) - self.abduct_sm);
        self.closure_sm += sc * (clamp(p.closure, 0.0, 1.0) - self.closure_sm);

        let dp_cm_h2o = self.dp_sm / CM_H2O;
        let pth_cm_h2o = self.threshold / CM_H2O;

        // Aerodynamic & laryngeal voicing gates
        let p_gate = smoothstep(0.75 * pth_cm_h2o, 1.45 * pth_cm_h2o, dp_cm_h2o);
        let abd_gate = 1.0 - smoothstep(0.08, 0.22, self.abduct_sm);
        let drive_gate = p_gate * abd_gate * (1.0 - self.closure_sm);

        // Pre-phonatory neutral half-width x0 (cm)
        let oq_clamped = clamp(p.open_quotient, 0.30, 0.95);
        let x0_modal = 0.0115 + 0.012 * (oq_clamped - 0.35);
        let lg_cm = 1.4;
        let x0_abduct = self.abduct_sm / (2.0 * lg_cm);
        let x0 = clamp(x0_modal + x0_abduct - 0.025 * self.closure_sm, -0.020, 0.35);

        // Ishizaka & Flanagan (1972) natural-frequency detuning (omega_1 > omega_2 -> x1 leads x2!)
        let f0_inst = clamp(f0_target / self.cycle_period_scale, 22.0, 1800.0);
        let omega = TAU * f0_inst * self.q_trim / 0.985;
        let k1 = (1.16 * omega) * (1.16 * omega);
        let k2 = (0.84 * omega) * (0.84 * omega);
        let kc = 0.54 * omega * omega;

        // Dissipative aerodynamic vena-contracta saturation stabilizes a pure period-1 limit cycle
        let p_norm = clamp((dp_cm_h2o - 0.72 * pth_cm_h2o) / 6.5, 0.0, 2.5)
            * drive_gate
            * if self.amp_sm > 1e-4 { 1.0 } else { 0.0 };

        let n_sub = if self.fs < 70_000.0 { 2usize } else { 1usize };
        let dt = 1.0 / (self.fs * n_sub as f64);

        let c1_open = 2.0 * 0.075 * (1.16 * omega);
        let c2_open = 2.0 * 0.090 * (0.84 * omega);

        for _ in 0..n_sub {
            // Re-seed tiny symmetry-breaking perturbation when phonation turns back on from exact rest
            if p_norm > 0.08 && self.x1.abs() + self.x2.abs() < 1e-5 {
                self.x1 = 0.0015;
            }

            let a1 = x0 + self.x1;
            let a2 = x0 + self.x2;

            let v1_norm = self.v1 / js_max(omega, 60.0);
            let e_fold = self.x1 * self.x1 + v1_norm * v1_norm;
            let r_target = 0.024 * clamp(p_norm, 0.15, 1.6) * self.cycle_amp;
            let e_target = js_max(r_target * r_target, 1e-8);
            let sat = 1.0 / (1.0 + e_fold / e_target);

            let gamma = 1.32 * clamp(p_norm, 0.0, 1.6) * sat * omega * omega;
            let conv_diff = clamp(self.x1 - self.x2, -0.055, 0.055);
            let p_drive1 = gamma * conv_diff;
            let p_drive2 = -0.14 * gamma * conv_diff;

            let col1 = if a1 < 0.0 { 0.85 * k1 * a1 } else { 0.0 };
            let col2 = if a2 < 0.0 { 0.85 * k2 * a2 } else { 0.0 };
            let d1 = if a1 < 0.0 { c1_open * 2.2 } else { c1_open };
            let d2 = if a2 < 0.0 { c2_open * 2.2 } else { c2_open };

            let cubic1 = 110.0 * k1 * self.x1 * self.x1 * self.x1;
            let cubic2 = 110.0 * k2 * self.x2 * self.x2 * self.x2;

            let f1 =
                p_drive1 - k1 * self.x1 - cubic1 - col1 - kc * (self.x1 - self.x2) - d1 * self.v1;
            let f2 =
                p_drive2 - k2 * self.x2 - cubic2 - col2 - kc * (self.x2 - self.x1) - d2 * self.v2;

            self.v1 = flush_subnormal(clamp(self.v1 + f1 * dt, -500.0, 500.0));
            self.v2 = flush_subnormal(clamp(self.v2 + f2 * dt, -500.0, 500.0));
            self.x1 = flush_subnormal(clamp(self.x1 + self.v1 * dt, -0.10, 0.14));
            self.x2 = flush_subnormal(clamp(self.x2 + self.v2 * dt, -0.10, 0.14));
        }

        let a1 = x0 + self.x1;
        let a2 = x0 + self.x2;
        let diff_a = a1 - a2;
        let a_min_smooth = 0.5 * (a1 + a2 - sqrt(diff_a * diff_a + 1.0e-8));
        let h_open = js_max(0.0, a_min_smooth);

        // Sub-sample opening-crossing pitch servo & per-cycle perturbations
        let thresh = 0.001;
        let is_open = a_min_smooth > thresh;
        self.period_acc += 1.0;
        if is_open && !self.prev_open && self.period_acc > 0.55 * self.fs / js_max(f0_target, 24.0)
        {
            let denom = js_max((a_min_smooth - self.prev_amin).abs(), 1e-12);
            let frac = clamp((a_min_smooth - thresh) / denom, 0.0, 1.0);
            let exact_samples = self.period_acc - frac;
            let inst_f0 = self.fs / js_max(exact_samples, 1.0);
            let expected = f0_target / self.cycle_period_scale;
            if inst_f0 > 0.65 * expected && inst_f0 < 1.55 * expected {
                self.measured_f0 = lerp(self.measured_f0, inst_f0, 0.45);
                let ratio = clamp(expected / inst_f0, 0.85, 1.18);
                self.q_trim = clamp(self.q_trim * pow(ratio, 0.22), 0.80, 1.25);
            }
            self.period_acc = frac;

            // Per-cycle jitter, shimmer, diplophonia, and logistic bifurcation chaos
            self.cycle_index = self.cycle_index.wrapping_add(1);
            let alt = if self.cycle_index & 1 == 1 { 1.0 } else { -1.0 };
            let mut chaos_p = 0.0;
            let mut chaos_a = 0.0;
            if p.chaos > 0.0 {
                let r = 2.9 + 1.1 * js_min(1.0, p.chaos);
                self.chaos_x = r * self.chaos_x * (1.0 - self.chaos_x);
                if !(self.chaos_x > 0.0 && self.chaos_x < 1.0) {
                    self.chaos_x = 0.37;
                }
                let dev = self.chaos_x - 0.5;
                chaos_p = 0.40 * p.chaos * dev;
                chaos_a = 0.95 * p.chaos * dev;
            }
            let fry_sub = if p.fry > 0.25 {
                0.22 * smoothstep(0.25, 0.85, p.fry) * alt
            } else {
                0.0
            };
            self.cycle_period_scale = clamp(
                1.0 + p.jitter * self.rng.gauss() + p.diplophonia * 0.08 * alt + chaos_p,
                0.58,
                1.75,
            );
            self.cycle_amp = clamp(
                1.0 + p.shimmer * self.rng.gauss()
                    - (p.diplophonia * 0.32 + fry_sub.abs()) * (if alt > 0.0 { 1.0 } else { 0.0 })
                    + chaos_a,
                0.15,
                1.85,
            );
        } else if drive_gate < 0.05 {
            self.measured_f0 = f0_target;
        }
        self.prev_open = is_open;
        self.prev_amin = a_min_smooth;
        self.f0_now = self.measured_f0;

        // Instantaneous analytic phase & mucosal phase lag (x1 leads x2 -> positive lag in 0..TAU)
        let inv_w = 1.0 / js_max(omega, 1.0);
        let amp1_sq = self.x1 * self.x1 + (self.v1 * inv_w) * (self.v1 * inv_w);
        let amp2_sq = self.x2 * self.x2 + (self.v2 * inv_w) * (self.v2 * inv_w);
        if amp1_sq > 1e-8 && amp2_sq > 1e-8 {
            let phi1 = atan2(-self.v1 * inv_w, self.x1);
            let phi2 = atan2(-self.v2 * inv_w, self.x2);
            let mut lag = phi1 - phi2;
            while lag < 0.0 {
                lag += TAU;
            }
            while lag >= TAU {
                lag -= TAU;
            }
            self.phase_lag_sm += 0.015 * (lag - self.phase_lag_sm);
            let mut norm_ph = phi1 / TAU;
            if norm_ph < 0.0 {
                norm_ph += 1.0;
            }
            self.phase_est = clamp(norm_ph, 0.0, 1.0);
        }

        // Voicing telemetry smoothed from oscillation envelope vs threshold
        let vib_target = if drive_gate > 0.25 && amp1_sq > 1e-6 {
            1.0
        } else {
            0.0
        };
        self.voicing_sm += self.voice_coef * (vib_target - self.voicing_sm);
        self.voicing = clamp(self.voicing_sm, 0.0, 1.0);

        // Membranous glottal area + static abduction + posterior chink
        let static_abduct = self.abduct_sm * (1.0 - 0.92 * self.closure_sm);
        let h_vib = js_max(0.0, h_open - x0_abduct);
        let dyn_gain = self.amp_sm / 0.036;
        let chink = js_max(0.0, p.chink) * (1.0 - 0.92 * self.closure_sm);

        let total_area = js_max(0.0, static_abduct + h_vib * dyn_gain + chink);
        self.area = total_area;
        total_area
    }
}

#[inline(always)]
fn flush_subnormal(x: f64) -> f64 {
    if !x.is_finite() || x.abs() < 1e-24 {
        0.0
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_oscillates_above_threshold_with_mucosal_phase_lag_and_stops_when_abducted() {
        let fs = 96_000.0;
        let mut g = TwoMassGlottis::new(fs);
        g.params.f0 = 120.0;
        g.params.jitter = 0.0;
        g.params.shimmer = 0.0;
        g.params.drift = 0.0;

        // 1. Sub-threshold subglottal pressure (0.6 cmH2O < ~2.2 cmH2O): vibration decays cleanly to rest
        for _ in 0..(0.25 * fs) as usize {
            let a = g.step(0.6 * CM_H2O);
            assert!(a.is_finite() && a >= 0.0);
        }
        let (x1_sub, x2_sub) = g.displacements_cm();
        assert!(
            x1_sub.abs() < 1e-3 && x2_sub.abs() < 1e-3,
            "expected rest below phonation threshold, got ({x1_sub}, {x2_sub})"
        );
        assert!(g.voicing < 0.05, "voicing below threshold = {}", g.voicing);

        // 2. Supra-threshold pressure (8.0 cmH2O): self-sustained oscillation with mucosal wave phase lag
        let n_settle = (0.25 * fs) as usize;
        for _ in 0..n_settle {
            let _ = g.step(8.0 * CM_H2O);
        }
        let n_meas = (0.20 * fs) as usize;
        let mut area_min = f64::INFINITY;
        let mut area_max = 0.0f64;
        let mut x1_sq = 0.0;
        let mut x2_sq = 0.0;
        for _ in 0..n_meas {
            let a = g.step(8.0 * CM_H2O);
            area_min = area_min.min(a);
            area_max = area_max.max(a);
            let (x1, x2) = g.displacements_cm();
            x1_sq += x1 * x1;
            x2_sq += x2 * x2;
        }
        let x1_rms = sqrt(x1_sq / n_meas as f64);
        let x2_rms = sqrt(x2_sq / n_meas as f64);
        let lag = g.phase_lag_rad();

        assert!(
            x1_rms > 0.008 && x2_rms > 0.005,
            "expected self-sustained fold oscillation: x1_rms={x1_rms}, x2_rms={x2_rms}"
        );
        assert!(
            area_max - area_min > 0.08,
            "expected clear glottal area modulation: min={area_min}, max={area_max}"
        );
        assert!(
            area_min <= g.params.chink + 1e-3,
            "expected complete membranous closure down to posterior chink: area_min={area_min}"
        );
        assert!(
            (0.25..2.35).contains(&lag),
            "expected lower mass x1 to lead upper mass x2 in phase (0.25..2.35 rad), got {lag:.3} rad"
        );
        assert!(
            (g.f0_now - 120.0).abs() < 3.5,
            "expected pitch lock near 120 Hz, got {:.2} Hz",
            g.f0_now
        );
        assert!(
            g.voicing > 0.90,
            "expected strong voicing gate, got {}",
            g.voicing
        );

        // 3. Wide glottal abduction (/s/, /h/): oscillation quenches cleanly into static open glottis
        g.params.abduction = 0.28;
        for _ in 0..(0.25 * fs) as usize {
            let _ = g.step(8.0 * CM_H2O);
        }
        let mut abd_min = f64::INFINITY;
        let mut abd_max = 0.0f64;
        for _ in 0..(0.05 * fs) as usize {
            let a = g.step(8.0 * CM_H2O);
            abd_min = abd_min.min(a);
            abd_max = abd_max.max(a);
        }
        assert!(
            abd_max - abd_min < 1e-3 && abd_min > 0.20,
            "expected static abducted opening without vibration: min={abd_min}, max={abd_max}"
        );
        assert!(
            g.voicing < 0.05,
            "expected voicing < 0.05 when abducted, got {}",
            g.voicing
        );
    }
}
