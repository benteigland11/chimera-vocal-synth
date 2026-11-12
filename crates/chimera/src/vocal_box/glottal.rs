//! Myoelastic-aerodynamic glottal valve (`GlottalValve`) and decoupled laryngeal anatomy (`BodyAnatomy`).
//!
//! Couples directly across `glottis_port` of [`crate::voice::human_airway::HumanAirway`]:
//! receives instantaneous transglottal pressure $\Delta P = P_{\text{trachea}} - P_{\text{epilarynx}}$
//! ($\text{dyn/cm}^2$) and returns instantaneous glottal orifice area $A_g(t)$ ($\text{cm}^2$).
//! Flow, open-phase formant damping, pulse skewing, and stop devoicing emerge directly from
//! conservation of mass and momentum in the airway network.

use crate::anatomy::human::{BodyMix, BodyName, HumanBody};
use crate::math::{
    Drift, PI, Rng, TAU, clamp, cos, exp, js_max, js_min, lerp, log2, pow, sin, smoothstep, sqrt,
};
use crate::physics::CM_H2O;
use crate::sources::glottis::{DEFAULT_GLOTTIS, GlottisParams};
use crate::sources::two_mass::TwoMassGlottis;

/// Decoupled vocal-tract and laryngeal anatomy derived from a continuous [`BodyMix`].
///
/// Unlike isotropic scaling ($A_{\text{fold}} \propto L_{\text{tract}}^2$), human laryngeal
/// dimorphism shrinks adult female membranous fold length (~10 mm vs ~16 mm) much more than
/// vocal-tract length (~0.89×), while preserving a small posterior arytenoid chink
/// (Södersten & Lindestad 1990).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyAnatomy {
    pub mix: BodyMix,
    pub body: HumanBody,
    pub vtl_scale: f64,
    pub fold_area_scale: f64,
    pub base_chink_cm2: f64,
    pub speaking_f0_hz: f64,
}

impl BodyAnatomy {
    pub fn from_mix(mix: impl Into<BodyMix>) -> Self {
        let raw: BodyMix = mix.into();
        let clean = BodyMix::new(raw.female, raw.child);
        let body = clean.body();
        let fold_area_scale = clean.blend(|b| match b {
            BodyName::Male => 1.00,
            BodyName::Female => 0.60,
            BodyName::Child => 0.36,
        });
        let base_chink_cm2 = clean.blend(|b| match b {
            BodyName::Male => 0.0068,
            BodyName::Female => 0.0076,
            BodyName::Child => 0.0046,
        });
        let speaking_f0_hz = clean.blend(|b| match b {
            BodyName::Male => 110.0,
            BodyName::Female => 205.0,
            BodyName::Child => 275.0,
        });
        Self {
            mix: clean,
            body,
            vtl_scale: body.scale,
            fold_area_scale,
            base_chink_cm2,
            speaking_f0_hz,
        }
    }
}

/// Continuous cricothyroid (CT) vs thyroarytenoid (TA) pitch registration across 5 octaves.
#[derive(Clone, Copy, Debug)]
pub struct Registration {
    /// `0.0` in modal/chest register, rising smoothly toward `1.0` in upper head/soprano register.
    pub head_mix: f64,
    /// Vibratory fold cross-section multiplier (`1.0` chest -> `~0.68` high soprano).
    pub excursion_scale: f64,
    /// Open-quotient shift due to longitudinal CT fold thinning.
    pub oq_shift: f64,
    /// Closure-sharpness softening in thin-edge head register.
    pub sharpness_scale: f64,
}

#[inline]
pub fn register_state(f0_hz: f64, speaking_f0_hz: f64, tension: f64) -> Registration {
    let f_ref = js_max(speaking_f0_hz, 60.0);
    let oct_above = log2(clamp(f0_hz, 24.0, 2000.0) / f_ref);
    // Belt / high medial tension delays head-voice thinning upward in pitch
    let shift = 0.30 * clamp(tension - 0.50, -0.4, 0.5);
    let head_mix = smoothstep(0.45 + shift, 1.75 + shift, oct_above);
    Registration {
        head_mix,
        excursion_scale: 1.0 - 0.32 * head_mix,
        oq_shift: 0.11 * head_mix * (1.0 - 0.45 * clamp(tension, 0.0, 1.0)),
        sharpness_scale: 1.0 - 0.24 * head_mix,
    }
}

/// $C^1$-continuous normalized glottal area pulse (`0..1`) with progressive mucosal-zipper return phase.
///
/// Both value and first derivative vanish smoothly at cycle start (`ph = 0`), peak opening (`t = s`),
/// and complete closure (`t = 1`), eliminating the artificial `2–8 kHz` slope-corner buzz of piecewise
/// cosine-quarter pulses while preserving asymmetric Fant-LF closing snap.
#[inline]
pub fn ribbon_pulse_c1(ph: f64, oq: f64, skew: f64, sharpness: f64) -> f64 {
    if ph <= 0.0 || ph >= oq {
        return 0.0;
    }
    let t = ph / oq;
    let s = clamp(skew, 0.34, 0.88);
    if t < s {
        return 0.5 - 0.5 * cos((PI * t) / s);
    }
    let u = clamp((t - s) / (1.0 - s), 0.0, 1.0);
    let soft = 0.5 + 0.5 * cos(PI * u);
    let u2 = u * u;
    // (1 - u^2.5)^2: steepest closing velocity at u = 0.68, smooth 32% zipper return tail to 0
    let a_mid = 1.0 - u2 * sqrt(u);
    let mid = a_mid * a_mid;
    // (1 - u^4)^2: steepest closing velocity at u = 0.81, crisp 19% zipper return tail for pressed voice
    let a_crisp = 1.0 - u2 * u2;
    let crisp = a_crisp * a_crisp;

    let sh = clamp(sharpness, 0.0, 1.0);
    if sh <= 0.55 {
        lerp(soft, mid, sh / 0.55)
    } else {
        lerp(mid, crisp, (sh - 0.55) / 0.45)
    }
}

/// Analytical myoelastic-aerodynamic ribbon glottis with progressive zipper return phase and
/// correlated neuromuscular tremor.
#[derive(Clone, Debug)]
pub struct RibbonGlottis {
    pub fs: f64,
    pub params: GlottisParams,
    pub voicing: f64,
    pub area: f64,
    pub f0_now: f64,
    pub threshold: f64,

    phase: f64,
    cycle_amp: f64,
    cycle_period_scale: f64,
    cycle_oq: f64,
    cycle_skew: f64,
    cycle_sharp: f64,
    cycle_index: u64,
    jitter_lp: f64,
    shimmer_lp: f64,
    chaos_x: f64,

    vib_phase: f64,
    rng: Rng,
    drift: Drift,
    drift_value: f64,
    drift_sm: f64,
    drift_counter: i32,

    f0_sm: f64,
    oq_sm: f64,
    skew_sm: f64,
    sharp_sm: f64,
    voicing_sm: f64,
    dp_sm: f64,
    dp_coef: f64,
    voice_coef: f64,
    static_sm: f64,
    closure_sm: f64,
    amp_sm: f64,
    sm_coef: f64,
}

impl RibbonGlottis {
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
            phase: 0.0,
            cycle_amp: 1.0,
            cycle_period_scale: 1.0,
            cycle_oq: DEFAULT_GLOTTIS.open_quotient,
            cycle_skew: DEFAULT_GLOTTIS.skew,
            cycle_sharp: DEFAULT_GLOTTIS.closure_sharpness,
            cycle_index: 0,
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
            skew_sm: DEFAULT_GLOTTIS.skew,
            sharp_sm: DEFAULT_GLOTTIS.closure_sharpness,
            voicing_sm: 0.0,
            dp_sm: 0.0,
            dp_coef: 1.0 - exp(-TAU * 32.0 / fs_clean),
            voice_coef: 1.0 - exp(-TAU * 36.0 / fs_clean),
            static_sm: 0.0,
            closure_sm: 0.0,
            amp_sm: 0.0,
            sm_coef: 1.0 - exp(-1.0 / (0.0032 * fs_clean)),
        }
    }

    pub fn reset(&mut self) {
        let fs = self.fs;
        let params = self.params;
        *self = Self::new(fs);
        self.params = params;
        self.f0_now = params.f0;
        self.f0_sm = params.f0;
        self.oq_sm = params.open_quotient;
        self.skew_sm = params.skew;
        self.sharp_sm = params.closure_sharpness;
        self.cycle_oq = params.open_quotient;
        self.cycle_skew = params.skew;
        self.cycle_sharp = params.closure_sharpness;
    }

    #[inline]
    pub fn cycle_phase(&self) -> f64 {
        self.phase
    }

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

        // Melody-preserving vocal-fry register sag (keeps natural intonation contour intact)
        let fry_amt = clamp(p.fry, 0.0, 1.0);
        let mut f0_raw = clamp(p.f0 * pitch_factor, 24.0, 1800.0);
        if fry_amt > 0.0 {
            f0_raw = clamp(f0_raw * (1.0 - 0.18 * fry_amt), 24.0, 1800.0);
        }
        self.f0_sm += sc * (f0_raw - self.f0_sm);
        let f0 = self.f0_sm;
        self.f0_now = f0;

        // Smooth pulse-shape targets continuously
        let mut oq_target = clamp(p.open_quotient - 0.16 * (p.effort - 0.5), 0.26, 0.96);
        if fry_amt > 0.0 {
            oq_target = lerp(oq_target, 0.34, 0.72 * fry_amt);
        }
        self.oq_sm += sc * (oq_target - self.oq_sm);
        self.skew_sm += sc * (clamp(p.skew, 0.34, 0.88) - self.skew_sm);
        self.sharp_sm += sc * (clamp(p.closure_sharpness, 0.0, 1.0) - self.sharp_sm);

        // Aerodynamic & proportional adduction voicing gate (scales cleanly across male/female/child)
        let abd_ref = js_max(0.075, 0.85 * p.amplitude + 0.045);
        let pressure_gate = smoothstep(self.threshold * 0.75, self.threshold * 1.50, self.dp_sm);
        let adduct_gate =
            1.0 - smoothstep(0.45 * abd_ref, 1.25 * abd_ref, self.static_sm - p.chink);
        let target = pressure_gate
            * adduct_gate
            * (1.0 - self.closure_sm)
            * if p.amplitude > 1e-5 { 1.0 } else { 0.0 };
        self.voicing_sm += self.voice_coef * (target - self.voicing_sm);
        if target == 0.0 && self.voicing_sm < 1e-5 {
            self.voicing_sm = 0.0;
        }
        self.voicing = self.voicing_sm;

        if self.voicing_sm == 0.0 || self.phase >= self.cycle_oq {
            // Closed phase or unvoiced: safe to track smoothed pulse shape with zero waveform jump
            self.cycle_oq = self.oq_sm;
            self.cycle_skew = self.skew_sm;
            self.cycle_sharp = self.sharp_sm;
        }

        self.phase += (f0 / self.fs) / self.cycle_period_scale;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
            self.cycle_oq = self.oq_sm;
            self.cycle_skew = self.skew_sm;
            self.cycle_sharp = self.sharp_sm;
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
                chaos_p = 0.45 * p.chaos * dev;
                chaos_a = 1.05 * p.chaos * dev;
            }
            // Correlated 1/f-like neuromuscular tremor (avoids harsh white cycle-step noise)
            self.jitter_lp = 0.72 * self.jitter_lp + 0.28 * self.rng.gauss();
            self.shimmer_lp = 0.72 * self.shimmer_lp + 0.28 * self.rng.gauss();

            // Dicrotic subharmonic alternation for organic human vocal fry (Keating et al. 2015)
            let fry_sub = if fry_amt > 0.15 {
                smoothstep(0.15, 0.85, fry_amt)
            } else {
                0.0
            };
            self.cycle_period_scale = clamp(
                1.0 + p.jitter * self.jitter_lp
                    + (p.diplophonia * 0.08 + 0.13 * fry_sub) * alt
                    + chaos_p,
                0.60,
                1.70,
            );
            self.cycle_amp = js_max(
                0.18,
                1.0 + p.shimmer * self.shimmer_lp
                    - (p.diplophonia * 0.32 + 0.38 * fry_sub) * (if alt > 0.0 { 1.0 } else { 0.0 })
                    + chaos_a,
            );
        }

        let pulse = ribbon_pulse_c1(
            self.phase,
            self.cycle_oq,
            self.cycle_skew,
            self.cycle_sharp,
        );

        let effort_scale = 0.45 + 0.75 * p.effort;
        self.amp_sm += sc * (p.amplitude * effort_scale - self.amp_sm);
        self.closure_sm += sc * (p.closure - self.closure_sm);
        self.static_sm += sc * (p.abduction + p.chink - self.static_sm);

        let vib = self.amp_sm * self.cycle_amp * self.voicing_sm * pulse;
        let statics = self.static_sm * (1.0 - 0.92 * self.closure_sm);
        let a = js_max(0.0, vib + statics);
        self.area = a;
        a
    }
}

/// Unified physical glottal valve supporting both progressive-zipper [`RibbonGlottis`] and
/// coupled self-oscillating [`TwoMassGlottis`].
#[derive(Clone, Debug)]
pub struct GlottalValve {
    pub ribbon: RibbonGlottis,
    pub two_mass: TwoMassGlottis,
    use_two_mass: bool,
}

impl GlottalValve {
    pub fn new(fs: f64) -> Self {
        Self {
            ribbon: RibbonGlottis::new(fs),
            two_mass: TwoMassGlottis::new(fs),
            use_two_mass: false,
        }
    }

    #[inline]
    pub fn is_two_mass(&self) -> bool {
        self.use_two_mass
    }

    pub fn set_two_mass(&mut self, enabled: bool) {
        if enabled && !self.use_two_mass {
            let ph = self.ribbon.cycle_phase();
            let f0 = if self.ribbon.voicing > 0.0 || ph > 0.0 {
                self.ribbon.f0_now
            } else {
                self.ribbon.params.f0
            };
            self.two_mass.params = self.ribbon.params;
            self.two_mass.sync_to_phase(ph, f0);
        }
        self.use_two_mass = enabled;
    }

    #[inline]
    pub fn params_mut(&mut self) -> (&mut GlottisParams, &mut GlottisParams) {
        (&mut self.ribbon.params, &mut self.two_mass.params)
    }

    pub fn set_params(&mut self, params: GlottisParams) {
        self.ribbon.params = params;
        self.two_mass.params = params;
        if self.ribbon.voicing == 0.0 && self.ribbon.cycle_phase() == 0.0 {
            self.ribbon.f0_now = params.f0;
        }
    }

    pub fn sync_idle(&mut self) {
        let rp = self.ribbon.params;
        self.ribbon.f0_now = rp.f0;
        self.ribbon.f0_sm = rp.f0;
        self.ribbon.oq_sm = rp.open_quotient;
        self.ribbon.skew_sm = rp.skew;
        self.ribbon.sharp_sm = rp.closure_sharpness;
        self.ribbon.cycle_oq = rp.open_quotient;
        self.ribbon.cycle_skew = rp.skew;
        self.ribbon.cycle_sharp = rp.closure_sharpness;
        let tp = self.two_mass.params;
        self.two_mass.f0_now = tp.f0;
        if self.use_two_mass {
            self.two_mass.sync_to_phase(0.0, tp.f0);
        } else {
            self.two_mass.reset();
        }
    }

    pub fn reset(&mut self) {
        self.ribbon.reset();
        self.two_mass.reset();
        if self.use_two_mass {
            let ph = self.ribbon.cycle_phase();
            let f0 = self.ribbon.params.f0;
            self.two_mass.params = self.ribbon.params;
            self.two_mass.sync_to_phase(ph, f0);
        }
    }

    #[inline]
    pub fn step(&mut self, dp: f64) -> f64 {
        if self.use_two_mass {
            self.two_mass.step(dp)
        } else {
            self.ribbon.step(dp)
        }
    }

    #[inline]
    pub fn cycle_phase(&self) -> f64 {
        if self.use_two_mass {
            self.two_mass.cycle_phase()
        } else {
            self.ribbon.cycle_phase()
        }
    }

    #[inline]
    pub fn f0_now(&self) -> f64 {
        if self.use_two_mass {
            self.two_mass.f0_now
        } else {
            self.ribbon.f0_now
        }
    }

    #[inline]
    pub fn voicing(&self) -> f64 {
        if self.use_two_mass {
            self.two_mass.voicing
        } else {
            self.ribbon.voicing
        }
    }

    #[inline]
    pub fn area(&self) -> f64 {
        if self.use_two_mass {
            self.two_mass.area
        } else {
            self.ribbon.area
        }
    }
}
