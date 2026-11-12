//! Coupled two-mass self-oscillating vocal-fold source (Ishizaka & Flanagan 1972; Titze 1988).
//!
//! Models the lower ($m_1$) and upper ($m_2$) vocal-fold masses coupled by shear stiffness $k_c$
//! and driven by real transglottal pressure $\Delta P = P_{\text{sub}} - P_{\text{supra}}$ across
//! the airway's glottal port.
//!
//! Key physical properties:
//! - Self-sustained oscillation emerges from mucosal-wave phase delay ($x_1$ leads $x_2$),
//! - Supraglottal inertive pressure ripple ($\Delta P_{\text{ac}}$) feeds back onto the folds,
//! - Progressive anterior-to-posterior mucosal zipper closure ([`zipper_area_c1`]) eliminates
//!   artificial slope-discontinuity buzz while sealing cleanly to the posterior chink,
//! - Aerodynamic Bernoulli re-adduction entrainment gives natural `< 10 ms` post-consonant VOT
//!   onset so running speech never stutters on short syllables.
//!
//! Real-time contract: zero heap allocations, deterministic across platforms via [`crate::math`].

use crate::math::{
    Drift, Rng, TAU, atan2, clamp, cos, exp, js_max, js_min, lerp, pow, sin, smoothstep, sqrt,
};
use crate::physics::CM_H2O;
use crate::sources::glottis::{DEFAULT_GLOTTIS, GlottisParams};

/// Exact $C^1$ parabolic progressive-zipper contact law for a bowed lenticular glottal slit
/// (Titze 1989):
/// - $a \le -\delta \implies 0$ (complete membranous seal),
/// - $a \ge +\delta \implies a$ (fully parted edges),
/// - $-\delta < a < +\delta \implies \frac{(a + \delta)^2}{4\delta}$ (smooth anterior-to-posterior
///   zipper return phase with continuous value and continuous first derivative everywhere).
#[inline(always)]
pub fn zipper_area_c1(a: f64, delta: f64) -> f64 {
    if !a.is_finite() {
        return 0.0;
    }
    let d = if delta.is_finite() {
        js_max(delta, 1e-5)
    } else {
        0.004
    };
    if a <= -d {
        0.0
    } else if a >= d {
        a
    } else {
        let u = a + d;
        (u * u) / (4.0 * d)
    }
}

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
    jitter_lp: f64,
    shimmer_lp: f64,
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
            threshold: 2.1 * CM_H2O,
            x1: 0.0020,
            v1: 0.0,
            x2: 0.0,
            v2: 0.0,
            prev_open: true,
            prev_amin: 0.014,
            period_acc: 0.0,
            q_trim: 1.0,
            measured_f0: DEFAULT_GLOTTIS.f0,
            phase_est: 0.0,
            phase_lag_sm: 0.90,
            cycle_index: 0,
            cycle_amp: 1.0,
            cycle_period_scale: 1.0,
            jitter_lp: 0.0,
            shimmer_lp: 0.0,
            chaos_x: 0.37,
            vib_phase: 0.0,
            rng: Rng::new(777),
            drift: Drift::new(fs_clean / 32.0, 0.3, 4242),
            drift_value: 0.0,
            drift_counter: 0,
            pitch_factor: 1.0,
            dp_sm: 0.0,
            dp_coef: 1.0 - exp(-TAU * 35.0 / fs_clean),
            voicing_sm: 0.0,
            voice_coef: 1.0 - exp(-TAU * 48.0 / fs_clean),
            abduct_sm: 0.0,
            closure_sm: 0.0,
            amp_sm: 0.0,
            sm_coef: 1.0 - exp(-1.0 / (0.0035 * fs_clean)),
        }
    }

    /// Deterministically reset internal oscillator, phase, and filter state without allocating.
    pub fn reset(&mut self) {
        let fs = self.fs;
        let params = self.params;
        *self = Self::new(fs);
        self.params = params;
        self.f0_now = params.f0;
        self.measured_f0 = params.f0;
    }

    /// Synchronize mucosal-wave displacement & velocity to normalized glottal phase `phi` (`0..1`).
    pub fn sync_to_phase(&mut self, phi: f64, f0_hz: f64) {
        let f0 = clamp(f0_hz, 24.0, 1600.0);
        let omega = TAU * f0;
        let ang = TAU * clamp(phi, 0.0, 1.0);
        self.x1 = 0.018 * cos(ang);
        self.v1 = -0.018 * omega * sin(ang);
        self.x2 = 0.016 * cos(ang - 1.00);
        self.v2 = -0.016 * omega * sin(ang - 1.00);
        self.phase_est = clamp(phi, 0.0, 1.0);
        self.measured_f0 = f0;
        self.f0_now = f0;
        self.q_trim = 1.0;
        self.period_acc = 0.0;
    }

    /// Current lower and upper mass lateral displacements `(x1_cm, x2_cm)` from pre-phonatory neutral.
    #[inline]
    pub fn displacements_cm(&self) -> (f64, f64) {
        (self.x1, self.x2)
    }

    /// Smoothed mucosal-wave phase lag of upper mass behind lower mass (radians, `0..TAU`).
    #[inline]
    pub fn phase_lag_rad(&self) -> f64 {
        self.phase_lag_sm
    }

    /// Normalized glottal cycle phase `0..1` for scope synchronization.
    #[inline]
    pub fn cycle_phase(&self) -> f64 {
        self.phase_est
    }

    /// Alias matching [`crate::sources::glottis::ParametricGlottis::next`].
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
                f0_target * (1.0 - 0.62 * fry_amt) + js_min(f0_target, 50.0) * 0.62 * fry_amt;
        }
        if (self.measured_f0 - f0_target).abs() > 0.28 * f0_target {
            self.measured_f0 = f0_target;
            self.q_trim = 1.0;
        }

        // Smooth control parameters to prevent click discontinuities
        let sc = self.sm_coef;
        let effort_scale = 0.45 + 0.75 * clamp(p.effort, 0.0, 1.5);
        self.amp_sm += sc * (js_max(0.0, p.amplitude) * effort_scale - self.amp_sm);
        self.abduct_sm += sc * (js_max(0.0, p.abduction) - self.abduct_sm);
        self.closure_sm += sc * (clamp(p.closure, 0.0, 1.0) - self.closure_sm);

        let dp_cm_h2o = self.dp_sm / CM_H2O;
        let dp_ac_cm_h2o = clamp((dp_clean - self.dp_sm) / CM_H2O, -3.5, 3.5);
        let pth_cm_h2o = self.threshold / CM_H2O;

        // Aerodynamic & laryngeal voicing gates:
        // `abduct_sm` is scaled against physiological glottal size (`p.amplitude + 0.06`) so adult
        // male, female, and child larynges abduct and re-adduct at identical phonetic proportions.
        let abd_ref = js_max(0.075, 0.85 * p.amplitude + 0.045);
        let p_gate = smoothstep(0.68 * pth_cm_h2o, 1.35 * pth_cm_h2o, dp_cm_h2o);
        let abd_gate = 1.0 - smoothstep(0.45 * abd_ref, 1.25 * abd_ref, self.abduct_sm);
        let drive_gate = p_gate * abd_gate * (1.0 - self.closure_sm);

        // Membranous pre-phonatory neutral half-width x0 (cm), kept distinct from cartilaginous
        // abduction so mild breathiness never prevents vocal-fold collision.
        let oq_clamped = clamp(p.open_quotient, 0.30, 0.92);
        let x0_modal = 0.0080 + 0.015 * (oq_clamped - 0.35);
        let x0 = clamp(x0_modal - 0.022 * self.closure_sm, -0.015, 0.045);

        // Ishizaka & Flanagan (1972) natural-frequency detuning (omega_1 > omega_2 -> x1 leads x2!)
        let f0_inst = clamp(f0_target / self.cycle_period_scale, 22.0, 1800.0);
        let omega = TAU * f0_inst * self.q_trim / 0.985;
        let k1 = (1.16 * omega) * (1.16 * omega);
        let k2 = (0.84 * omega) * (0.84 * omega);
        let kc = 0.54 * omega * omega;

        // Dissipative aerodynamic vena-contracta drive with real supraglottal acoustic reactance:
        // once transglottal pressure crosses `pth_cm_h2o`, `0.58 + 0.52 * p_mean` keeps the 2-mass
        // ODE above its Hopf bifurcation on soft and phrase-final syllables (~2.5-4.5 cmH2O).
        let p_mean = clamp((dp_cm_h2o - 0.65 * pth_cm_h2o) / 6.2, 0.0, 2.4);
        let p_norm = clamp(0.58 + 0.52 * p_mean + 0.08 * dp_ac_cm_h2o, 0.0, 2.5)
            * drive_gate
            * if self.amp_sm > 1e-4 { 1.0 } else { 0.0 };

        let n_sub = if self.fs < 70_000.0 || f0_target > 420.0 {
            2usize
        } else {
            1usize
        };
        let dt = 1.0 / (self.fs * n_sub as f64);

        let c1_open = 2.0 * 0.075 * (1.16 * omega);
        let c2_open = 2.0 * 0.090 * (0.84 * omega);
        // `self.amp_sm` already incorporates primary respiratory effort; `r_target` adds gentle
        // subglottal excursion growth without double-attenuating soft or phrase-final syllables.
        let r_target = 0.025
            * (0.76 + 0.30 * clamp(p_mean, 0.0, 1.6))
            * smoothstep(0.0, 0.18, drive_gate)
            * self.cycle_amp;

        // Aerodynamic Bernoulli re-adduction impulse: when folds re-adduct into an active airstream
        // after a voiceless consonant, suction across the narrowing glottis entrains the mucosal
        // wave within 1 cycle (< 8 ms VOT) rather than waiting 80 ms from dead rest.
        if p_norm > 0.06 {
            let inv_w0 = 1.0 / js_max(omega, 60.0);
            let r_now = sqrt(self.x1 * self.x1 + (self.v1 * inv_w0) * (self.v1 * inv_w0));
            if r_now < 0.54 * r_target {
                let r_seed = 0.66 * r_target;
                let ang = TAU * self.phase_est;
                self.x1 = r_seed * cos(ang);
                self.v1 = -r_seed * omega * sin(ang);
                self.x2 = 0.88 * r_seed * cos(ang - 0.98);
                self.v2 = -0.88 * r_seed * omega * sin(ang - 0.98);
            }
        }

        for _ in 0..n_sub {
            let a1 = x0 + self.x1;
            let a2 = x0 + self.x2;

            let v1_norm = self.v1 / js_max(omega, 60.0);
            let e_fold = self.x1 * self.x1 + v1_norm * v1_norm;
            let e_target = js_max(r_target * r_target, 1e-8);
            let sat = 1.0 / (1.0 + e_fold / e_target);
            let onset_boost = if p_norm > 0.06 && e_fold < 0.70 * e_target {
                1.50
            } else {
                1.0
            };

            let gamma = 1.32 * onset_boost * clamp(p_norm, 0.0, 1.65) * sat * omega * omega;
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

            self.v1 = flush_subnormal(clamp(self.v1 + f1 * dt, -600.0, 600.0));
            self.v2 = flush_subnormal(clamp(self.v2 + f2 * dt, -600.0, 600.0));
            self.x1 = flush_subnormal(clamp(self.x1 + self.v1 * dt, -0.10, 0.14));
            self.x2 = flush_subnormal(clamp(self.x2 + self.v2 * dt, -0.10, 0.14));
        }

        let a1 = x0 + self.x1;
        let a2 = x0 + self.x2;
        let diff_a = a1 - a2;
        let a_min_smooth = 0.5 * (a1 + a2 - sqrt(diff_a * diff_a + 1.0e-6));

        // Sub-sample opening-crossing pitch servo & correlated neuromuscular perturbations
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
                self.q_trim = clamp(self.q_trim * pow(ratio, 0.24), 0.80, 1.25);
            }
            self.period_acc = frac;

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
            // Low-pass correlated neuromuscular tremor avoids harsh white AM fuzz on vowels
            self.jitter_lp = 0.62 * self.jitter_lp + 0.38 * self.rng.gauss();
            self.shimmer_lp = 0.62 * self.shimmer_lp + 0.38 * self.rng.gauss();

            let fry_sub = if p.fry > 0.25 {
                0.26 * smoothstep(0.25, 0.85, p.fry) * alt
            } else {
                0.0
            };
            self.cycle_period_scale = clamp(
                1.0 + p.jitter * self.jitter_lp + p.diplophonia * 0.08 * alt + chaos_p,
                0.58,
                1.75,
            );
            self.cycle_amp = clamp(
                1.0 + p.shimmer * self.shimmer_lp
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

        // Voicing envelope smoothed from active drive & oscillation radius
        let vib_target = if drive_gate > 0.03 && amp1_sq > 1e-7 {
            drive_gate
        } else {
            0.0
        };
        self.voicing_sm += self.voice_coef * (vib_target - self.voicing_sm);
        if vib_target == 0.0 && self.voicing_sm < 1e-5 {
            self.voicing_sm = 0.0;
        }
        self.voicing = clamp(self.voicing_sm, 0.0, 1.0);

        // Progressive anterior-to-posterior mucosal zipper contact + cartilaginous abduction + chink
        let sharp = clamp(p.closure_sharpness, 0.0, 1.0);
        let zipper_delta = lerp(0.0085, 0.0014, sharp);
        let h_vib = zipper_area_c1(a_min_smooth, zipper_delta) * self.voicing;
        let dyn_gain = self.amp_sm / 0.028;

        let static_abduct = self.abduct_sm * (1.0 - 0.92 * self.closure_sm);
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
    fn self_oscillates_above_threshold_with_mucosal_phase_lag_and_rapid_vot_reonset() {
        let fs = 96_000.0;
        let mut g = TwoMassGlottis::new(fs);
        g.params.f0 = 120.0;
        g.params.jitter = 0.0;
        g.params.shimmer = 0.0;
        g.params.drift = 0.0;

        // 1. Sub-threshold subglottal pressure (0.6 cmH2O < ~2.1 cmH2O): vibration decays cleanly to rest
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
        for _ in 0..(0.20 * fs) as usize {
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
        assert!(g.voicing > 0.90);

        // 3. Wide glottal abduction (/s/, /p/, /h/): oscillation quenches cleanly into static open glottis
        g.params.abduction = 0.28;
        for _ in 0..(0.20 * fs) as usize {
            let _ = g.step(8.0 * CM_H2O);
        }
        let mut abd_min = f64::INFINITY;
        let mut abd_max = 0.0f64;
        for _ in 0..(0.04 * fs) as usize {
            let a = g.step(8.0 * CM_H2O);
            abd_min = abd_min.min(a);
            abd_max = abd_max.max(a);
        }
        assert!(
            abd_max - abd_min < 1e-3 && abd_min > 0.20,
            "expected static abducted opening without vibration: min={abd_min}, max={abd_max}"
        );
        assert!(g.voicing < 0.02);

        // 4. Re-adduction into vowel: voicing recovers within 15 ms (no syllabic dropout!)
        g.params.abduction = 0.0;
        let mut on_min = f64::INFINITY;
        let mut on_max = 0.0f64;
        for _ in 0..(0.015 * fs) as usize {
            let a = g.step(8.0 * CM_H2O);
            on_min = on_min.min(a);
            on_max = on_max.max(a);
        }
        assert!(
            on_max - on_min > 0.05 && g.voicing > 0.50,
            "expected rapid <15 ms VOT re-onset, got swing={} voicing={}",
            on_max - on_min,
            g.voicing
        );
    }
}
