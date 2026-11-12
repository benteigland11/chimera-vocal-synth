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
    q_trim_sm: f64,
    measured_f0: f64,
    phase_est: f64,
    entrain_phase: f64,
    phase_lag_sm: f64,
    cycle_index: u64,
    cycle_amp: f64,
    cycle_amp_sm: f64,
    cycle_period_scale: f64,
    period_scale_sm: f64,
    jitter_lp: f64,
    shimmer_lp: f64,
    chaos_x: f64,

    // Slow modulations
    vib_phase: f64,
    rng: Rng,
    drift: Drift,
    drift_value: f64,
    drift_sm: f64,
    drift_counter: i32,

    // Smoothed aerodynamic & control states
    f0_sm: f64,
    oq_sm: f64,
    sharp_sm: f64,
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
            q_trim_sm: 1.0,
            measured_f0: DEFAULT_GLOTTIS.f0,
            phase_est: 0.0,
            entrain_phase: 0.0,
            phase_lag_sm: 0.90,
            cycle_index: 0,
            cycle_amp: 1.0,
            cycle_amp_sm: 1.0,
            cycle_period_scale: 1.0,
            period_scale_sm: 1.0,
            jitter_lp: 0.0,
            shimmer_lp: 0.0,
            chaos_x: 0.37,
            vib_phase: 0.0,
            rng: Rng::new(777),
            drift: Drift::new(fs_clean / 32.0, 0.3, 4242),
            drift_value: 0.0,
            drift_sm: 0.0,
            drift_counter: 0,
            f0_sm: DEFAULT_GLOTTIS.f0,
            oq_sm: DEFAULT_GLOTTIS.open_quotient,
            sharp_sm: DEFAULT_GLOTTIS.closure_sharpness,
            dp_sm: 0.0,
            dp_coef: 1.0 - exp(-TAU * 42.0 / fs_clean),
            voicing_sm: 0.0,
            voice_coef: 1.0 - exp(-TAU * 65.0 / fs_clean),
            abduct_sm: 0.0,
            closure_sm: 0.0,
            amp_sm: 0.0,
            sm_coef: 1.0 - exp(-1.0 / (0.0030 * fs_clean)),
        }
    }

    /// Deterministically reset internal oscillator, phase, and filter state without allocating.
    pub fn reset(&mut self) {
        let fs = self.fs;
        let params = self.params;
        *self = Self::new(fs);
        self.params = params;
        self.f0_now = params.f0;
        self.f0_sm = params.f0;
        self.oq_sm = params.open_quotient;
        self.sharp_sm = params.closure_sharpness;
        self.measured_f0 = params.f0;
    }

    /// Snap smoothed idle parameters at `clock == 0` so initial state matches `reset()` bit-for-bit.
    #[inline]
    pub fn sync_idle(&mut self) {
        self.f0_now = self.params.f0;
        self.f0_sm = self.params.f0;
        self.measured_f0 = self.params.f0;
        self.oq_sm = clamp(self.params.open_quotient, 0.30, 0.92);
        self.sharp_sm = clamp(self.params.closure_sharpness, 0.0, 1.0);
    }

    /// Synchronize mucosal-wave displacement & velocity to normalized glottal phase `phi` (`0..1`).
    pub fn sync_to_phase(&mut self, phi: f64, f0_hz: f64) {
        let f0 = clamp(f0_hz, 24.0, 1600.0);
        let omega = TAU * f0;
        let ph = clamp(phi, 0.0, 1.0);
        let ang = TAU * ph;
        self.x1 = 0.018 * cos(ang);
        self.v1 = -0.018 * omega * sin(ang);
        self.x2 = 0.016 * cos(ang - 1.00);
        self.v2 = -0.016 * omega * sin(ang - 1.00);
        self.phase_est = ph;
        self.entrain_phase = ph;
        self.measured_f0 = f0;
        self.f0_now = f0;
        self.f0_sm = f0;
        self.oq_sm = self.params.open_quotient;
        self.sharp_sm = self.params.closure_sharpness;
        self.q_trim = 1.0;
        self.q_trim_sm = 1.0;
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

        // Per-sample continuous vibrato & smoothed pitch drift (eliminates 32-sample FM staircases)
        self.drift_counter -= 1;
        if self.drift_counter <= 0 {
            self.drift_counter = 32;
            self.drift_value = self.drift.step();
        }
        let sc = self.sm_coef;
        self.drift_sm += sc * (self.drift_value - self.drift_sm);
        self.vib_phase += (p.vibrato_rate / self.fs) * (1.0 + 0.04 * self.drift_sm);
        if self.vib_phase >= 1.0 {
            self.vib_phase -= 1.0;
        }
        let cents = p.vibrato_depth * sin(TAU * self.vib_phase) + p.drift * self.drift_sm;
        let pitch_factor = pow(2.0, cents / 1200.0);

        let mut f0_raw = clamp(p.f0 * pitch_factor, 24.0, 1600.0);
        if p.fry > 0.0 {
            let fry_amt = clamp(p.fry, 0.0, 1.0);
            f0_raw = clamp(f0_raw * (1.0 - 0.18 * fry_amt), 24.0, 1600.0);
        }
        if (f0_raw - self.f0_sm).abs() > 0.35 * self.f0_sm {
            self.measured_f0 = f0_raw;
            self.q_trim = 1.0;
        }
        self.f0_sm += sc * (f0_raw - self.f0_sm);
        let f0_target = self.f0_sm;

        // Smooth all control & cycle parameters per sample to eliminate control-rate step kinks
        let effort_scale = 0.45 + 0.75 * clamp(p.effort, 0.0, 1.5);
        let sc_abd = js_min(1.0, 1.75 * sc);
        self.amp_sm += sc * (js_max(0.0, p.amplitude) * effort_scale - self.amp_sm);
        self.abduct_sm += sc_abd * (js_max(0.0, p.abduction) - self.abduct_sm);
        self.closure_sm += sc * (clamp(p.closure, 0.0, 1.0) - self.closure_sm);
        self.oq_sm += sc * (clamp(p.open_quotient, 0.30, 0.92) - self.oq_sm);
        self.sharp_sm += sc * (clamp(p.closure_sharpness, 0.0, 1.0) - self.sharp_sm);
        self.q_trim_sm += sc * (self.q_trim - self.q_trim_sm);
        self.period_scale_sm += sc * (self.cycle_period_scale - self.period_scale_sm);
        self.cycle_amp_sm += sc * (self.cycle_amp - self.cycle_amp_sm);

        let dp_cm_h2o = self.dp_sm / CM_H2O;
        let dp_ac_cm_h2o = clamp((dp_clean - self.dp_sm) / CM_H2O, -3.5, 3.5);
        let pth_cm_h2o = self.threshold / CM_H2O;

        // Aerodynamic & laryngeal voicing gates:
        // Cartilaginous arytenoid abduction scales with vocal-tract cross-section k_vtl^2 ≈ sqrt(kf),
        // devoicing cleanly during voiceless stops/fricatives while re-adducting promptly across VOT.
        let k2_est = clamp(sqrt(js_max(p.amplitude, 0.008) / 0.17), 0.52, 1.05);
        let p_gate = smoothstep(0.65 * pth_cm_h2o, 1.30 * pth_cm_h2o, dp_cm_h2o);
        let abd_gate = 1.0 - smoothstep(0.10 * k2_est, 0.205 * k2_est, self.abduct_sm);
        let drive_gate = p_gate * abd_gate * (1.0 - self.closure_sm);

        // Membranous pre-phonatory neutral half-width x0 (cm):
        // Calibrated so modal speech (OQ ~ 0.58) achieves a complete ~40% closed phase decoupling the
        // subglottal trachea, pressed phonation closes ~50%, and breathy phonation stays softly open.
        let oq_u = clamp((self.oq_sm - 0.34) / 0.52, 0.0, 1.0);
        let x0_modal = lerp(0.0046, 0.0128, oq_u);
        let x0 = clamp(x0_modal - 0.018 * self.closure_sm, -0.010, 0.045);

        // Collision stiffness & CT/TA pitch coordination (Titze 1994):
        // Stronger collision shortens the closed half-period by ~1/sqrt(1 + col_k); scaling open
        // stiffness by the closed-fraction duty factor keeps F0 true to target across adduction modes.
        let col_k = 0.78 + 0.50 * self.sharp_sm;
        let skew_close = 1.0 + 0.26 * clamp(p.skew - 0.50, 0.0, 0.40);
        let phi_col = clamp((0.025 - x0_modal) / 0.040, 0.0, 0.52);
        let col_comp = ((1.0 - phi_col) + phi_col / sqrt(1.0 + 0.92 * col_k)) / sqrt(0.5 + 0.5 * skew_close);

        // Ishizaka & Flanagan (1972) natural-frequency detuning (omega_1 > omega_2 -> x1 leads x2!)
        let f0_inst = clamp(f0_target / self.period_scale_sm, 22.0, 1800.0);
        let omega = TAU * f0_inst * self.q_trim_sm * col_comp / 0.915;
        let k1 = (1.16 * omega) * (1.16 * omega);
        let k2 = (0.84 * omega) * (0.84 * omega);
        let kc = 0.54 * omega * omega;

        // Supra-threshold dissipative vena-contracta drive
        let p_mean = clamp((dp_cm_h2o - 0.65 * pth_cm_h2o) / 6.2, 0.0, 2.4);
        let p_norm = clamp(0.62 + 0.52 * p_mean + 0.03 * dp_ac_cm_h2o, 0.0, 2.5)
            * smoothstep(0.0, 0.42, drive_gate)
            * if self.amp_sm > 1e-4 { 1.0 } else { 0.0 };

        let n_sub = if self.fs < 70_000.0 || f0_target > 420.0 {
            2usize
        } else {
            1usize
        };
        let dt = 1.0 / (self.fs * n_sub as f64);

        let c1_open = 2.0 * 0.074 * (1.16 * omega);
        let c2_open = 2.0 * 0.088 * (0.84 * omega);
        let r_target = 0.0265
            * (0.78 + 0.28 * clamp(p_mean, 0.0, 1.6))
            * smoothstep(0.02, 0.26, drive_gate)
            * self.cycle_amp_sm;

        // Continuous C^1 aerodynamic Bernoulli VOT entrainment:
        // Advances smoothly inside the acceleration ODE (never teleporting x1/x2 or v1/v2!),
        // so glottal area A_g(t) has zero step/slope clicks while entraining within <8 ms.
        let drive_on = smoothstep(0.02, 0.14, p_norm);
        let e_target = js_max(r_target * r_target, 1e-8);

        for _ in 0..n_sub {
            self.entrain_phase += f0_inst * dt;
            if self.entrain_phase >= 1.0 {
                self.entrain_phase -= 1.0;
            }

            let a1 = x0 + self.x1;
            let a2 = x0 + self.x2;

            let v1_norm = self.v1 / js_max(omega, 60.0);
            let e_fold = self.x1 * self.x1 + v1_norm * v1_norm;
            let e_ratio = e_fold / e_target;
            let sat = 1.0 / (1.0 + e_ratio);

            let deficit = drive_on * (1.0 - smoothstep(0.08, 0.78, e_ratio));
            let onset_boost = 1.0 + 0.90 * deficit;

            let gamma = 1.36 * onset_boost * clamp(p_norm, 0.0, 1.70) * sat * omega * omega;
            let conv_diff = clamp(self.x1 - self.x2, -0.055, 0.055);
            let asym = if conv_diff < 0.0 { skew_close } else { 1.0 };
            let p_drive1 = gamma * asym * conv_diff;
            let p_drive2 = -0.14 * gamma * conv_diff;

            // Smooth resonant aerodynamic suction entrainment when waking from rest/abduction
            let ang = TAU * self.entrain_phase;
            let f_entrain = deficit * (1.10 * r_target * omega * omega);
            let f_seed1 = -f_entrain * sin(ang);
            let f_seed2 = -0.86 * f_entrain * sin(ang - 0.98);

            let col1 = if a1 < 0.0 { col_k * k1 * a1 } else { 0.0 };
            let col2 = if a2 < 0.0 { col_k * k2 * a2 } else { 0.0 };
            let d1 = if a1 < 0.0 { c1_open * 2.2 } else { c1_open };
            let d2 = if a2 < 0.0 { c2_open * 2.2 } else { c2_open };

            let cubic1 = 105.0 * k1 * self.x1 * self.x1 * self.x1;
            let cubic2 = 105.0 * k2 * self.x2 * self.x2 * self.x2;

            let f1 = p_drive1 + f_seed1
                - k1 * self.x1
                - cubic1
                - col1
                - kc * (self.x1 - self.x2)
                - d1 * self.v1;
            let f2 = p_drive2 + f_seed2
                - k2 * self.x2
                - cubic2
                - col2
                - kc * (self.x2 - self.x1)
                - d2 * self.v2;

            self.v1 = flush_subnormal(clamp(self.v1 + f1 * dt, -600.0, 600.0));
            self.v2 = flush_subnormal(clamp(self.v2 + f2 * dt, -600.0, 600.0));
            self.x1 = flush_subnormal(clamp(self.x1 + self.v1 * dt, -0.10, 0.14));
            self.x2 = flush_subnormal(clamp(self.x2 + self.v2 * dt, -0.10, 0.14));
        }

        let a1 = x0 + self.x1;
        let a2 = x0 + self.x2;
        let diff_a = a1 - a2;
        let a_min_smooth = 0.5 * (a1 + a2 - sqrt(diff_a * diff_a + 4.0e-7));

        // Sub-sample opening-crossing pitch servo & correlated neuromuscular perturbations
        let thresh = 0.0008;
        let is_open = a_min_smooth > thresh;
        self.period_acc += 1.0;
        if is_open && !self.prev_open && self.period_acc > 0.50 * self.fs / js_max(f0_target, 24.0)
        {
            let denom = js_max((a_min_smooth - self.prev_amin).abs(), 1e-12);
            let frac = clamp((a_min_smooth - thresh) / denom, 0.0, 1.0);
            let exact_samples = self.period_acc - frac;
            let inst_f0 = self.fs / js_max(exact_samples, 1.0);
            let expected = f0_target / self.cycle_period_scale;
            if inst_f0 > 0.55 * expected && inst_f0 < 1.85 * expected {
                self.measured_f0 = lerp(self.measured_f0, inst_f0, 0.45);
                let ratio = clamp(expected / inst_f0, 0.80, 1.24);
                self.q_trim = clamp(self.q_trim * pow(ratio, 0.30), 0.68, 1.32);
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
            self.jitter_lp = 0.72 * self.jitter_lp + 0.28 * self.rng.gauss();
            self.shimmer_lp = 0.72 * self.shimmer_lp + 0.28 * self.rng.gauss();

            let fry_sub = if p.fry > 0.15 {
                smoothstep(0.15, 0.85, p.fry)
            } else {
                0.0
            };
            self.cycle_period_scale = clamp(
                1.0 + p.jitter * self.jitter_lp
                    + (p.diplophonia * 0.08 + 0.11 * fry_sub) * alt
                    + chaos_p,
                0.62,
                1.65,
            );
            self.cycle_amp = clamp(
                1.0 + p.shimmer * self.shimmer_lp
                    - (p.diplophonia * 0.30 + 0.32 * fry_sub) * (if alt > 0.0 { 1.0 } else { 0.0 })
                    + chaos_a,
                0.20,
                1.75,
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
            let ph_clean = clamp(norm_ph, 0.0, 1.0);
            self.phase_est = ph_clean;
            if amp1_sq > 1e-5 {
                self.entrain_phase = ph_clean;
            }
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

        // Progressive anterior-to-posterior mucosal zipper contact + cartilaginous abduction + chink:
        // osc_env zeroes the resting x0 offset during voiceless segments without double-suppressing
        // vibration amplitude during breathy VOT onset (since x1/x2 already scale dynamically).
        let zipper_delta = lerp(0.0046, 0.0007, self.sharp_sm);
        let osc_env = smoothstep(0.015, 0.22, self.voicing) * smoothstep(0.0008, 0.0045, sqrt(amp1_sq));
        let h_vib = zipper_area_c1(a_min_smooth, zipper_delta) * osc_env;
        let dyn_gain = self.amp_sm / 0.0235;

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
