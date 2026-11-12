//! Simplified 4-organ physical human voice (`VocalBoxVoice`) and 13 physiological [`EmotionPreset`]s.
//!
//! Built strictly on the four irreducible physical elements of human voice production:
//! 1. **Subglottal Supply** (`lungs` + `trachea` tube with yielding tracheal walls),
//! 2. **Myoelastic-Aerodynamic Glottal Valve** ([`GlottalValve`]: progressive-zipper [`super::RibbonGlottis`]
//!    or coupled 2-DOF [`crate::sources::two_mass::TwoMassGlottis`] across `glottis_port`),
//! 3. **Articulated Supraglottal Waveguide** ([`HumanAirway`]: 2D midsagittal geometry $\to$ soft-wall
//!    `pharynx` + `oral` tubes, inelastic palate contact mechanics, Reynolds constriction & teeth jets),
//! 4. **Velopharyngeal Bifurcation** (`velum_port` $\to$ `nasal` tube + maxillary sinus $\to$ nostrils).
//!
//! No formant biquad stacks, vowel lookup tables, locus overrides, oral-occlusion gates, or RC stop
//! timers: all formants, consonant transitions, stop pressure buildup, voice bars, release bursts,
//! sibilants, nasal anti-formants, and singer's-ring inertance emerge directly from conservation of
//! mass and momentum in the airway.

use super::glottal::{BodyAnatomy, GlottalValve, register_state};
use crate::airway::decimator::Decimator;
use crate::anatomy::geom::Vec2;
use crate::anatomy::human::{
    ArticKey, ArticPatch, BODY_CHILD, BODY_FEMALE, BODY_MALE, BodyMix, HumanBody, N_ARTIC,
};
use crate::control::articulators::ArticulatorBank;
use crate::control::score::{Blended, ControlKey, Gesture, GestureScore, N_CONTROL};
use crate::math::{
    PI, clamp, exp, js_max, js_min, js_round, js_sign, js_trunc_i32, log2, pow, smoothstep, sqrt,
    tanh,
};
use crate::physics::CM_H2O;
use crate::voice::human_airway::HumanAirway;
use crate::voice::human_voice::{
    DEFAULT_SOURCE, SCOPE_POINTS, SourcePatch, VoiceSource, VoiceTelemetry, oversample_for,
};
use serde::{Deserialize, Serialize};

const SCOPE_LEN: usize = 4096;

/// Instantaneous physical state snapshot from [`VocalBoxVoice`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VocalBoxTelemetry {
    pub f0_now_hz: f64,
    pub voicing: f64,
    pub glottal_area_cm2: f64,
    pub glottal_flow_cm3_s: f64,
    pub velum_port_cm2: f64,
    pub nasal_flow_cm3_s: f64,
    pub p_lung_cm_h2o: f64,
}

/// 13 canonical expressive vocal emotion & phonation presets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmotionPreset {
    #[default]
    Natural,
    Warm,
    Tender,
    Joyful,
    Dramatic,
    Melancholic,
    Angry,
    Pressed,
    Breathy,
    Whisper,
    Fry,
    Operatic,
    Pop,
}

impl EmotionPreset {
    /// All 13 canonical presets in deterministic order.
    pub const ALL: [EmotionPreset; 13] = [
        EmotionPreset::Natural,
        EmotionPreset::Warm,
        EmotionPreset::Tender,
        EmotionPreset::Joyful,
        EmotionPreset::Dramatic,
        EmotionPreset::Melancholic,
        EmotionPreset::Angry,
        EmotionPreset::Pressed,
        EmotionPreset::Breathy,
        EmotionPreset::Whisper,
        EmotionPreset::Fry,
        EmotionPreset::Operatic,
        EmotionPreset::Pop,
    ];

    /// Canonical lowercase CLI names for all 13 presets.
    pub const NAMES: [&'static str; 13] = [
        "natural",
        "warm",
        "tender",
        "joyful",
        "dramatic",
        "melancholic",
        "angry",
        "pressed",
        "breathy",
        "whisper",
        "fry",
        "operatic",
        "pop",
    ];

    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Natural => "natural",
            Self::Warm => "warm",
            Self::Tender => "tender",
            Self::Joyful => "joyful",
            Self::Dramatic => "dramatic",
            Self::Melancholic => "melancholic",
            Self::Angry => "angry",
            Self::Pressed => "pressed",
            Self::Breathy => "breathy",
            Self::Whisper => "whisper",
            Self::Fry => "fry",
            Self::Operatic => "operatic",
            Self::Pop => "pop",
        }
    }

    /// Parse a preset name or common synonym (case-insensitive).
    pub fn from_name(raw: &str) -> Result<Self, String> {
        let mut norm = String::with_capacity(raw.len());
        for ch in raw.trim().chars() {
            if ch != '-' && ch != '_' && ch != ' ' {
                norm.push(ch.to_ascii_lowercase());
            }
        }
        match norm.as_str() {
            "natural" | "neutral" | "modal" | "default" => Ok(Self::Natural),
            "warm" | "mellow" => Ok(Self::Warm),
            "tender" | "soft" | "gentle" | "intimate" => Ok(Self::Tender),
            "joyful" | "happy" | "bright" | "cheerful" => Ok(Self::Joyful),
            "dramatic" | "heroic" | "theatrical" => Ok(Self::Dramatic),
            "melancholic" | "sad" | "somber" | "sorrowful" => Ok(Self::Melancholic),
            "angry" | "fierce" | "harsh" => Ok(Self::Angry),
            "pressed" | "tense" | "strained" => Ok(Self::Pressed),
            "breathy" | "airy" => Ok(Self::Breathy),
            "whisper" | "whispered" => Ok(Self::Whisper),
            "fry" | "creak" | "creaky" | "vocalfry" => Ok(Self::Fry),
            "operatic" | "classical" | "opera" | "lyric" => Ok(Self::Operatic),
            "pop" | "belt" | "contemporary" => Ok(Self::Pop),
            _ => Err(format!(
                "unknown emotion preset {raw:?}; valid presets: {}",
                Self::NAMES.join(", ")
            )),
        }
    }

    /// Physiological respiratory, laryngeal, prosodic, and anatomical posture parameters.
    ///
    /// Note: `epilarynx` is bounded to `<= 0.62` and `ventricular` to `<= 0.05` so the
    /// epilaryngeal exit remains strictly above `0.90 * k^2` (`w_area == 0` in `HumanAirway`),
    /// producing pure acoustic singer's-formant impedance ring without false-positive 2600 Hz
    /// oral-constriction hiss on open vowels.
    #[inline]
    pub const fn profile(self) -> EmotionProfile {
        match self {
            Self::Natural => EmotionProfile {
                pressure_cm_h2o: 8.0,
                tension: 0.50,
                brightness: 0.30,
                breathiness: 0.13,
                roughness: 0.05,
                fry: 0.0,
                whisper: false,
                larynx_cm: 0.0,
                epilarynx: 0.14,
                ventricular: 0.0,
                glottis_noise_scale: 0.95,
                f0_scale: 1.00,
                rate_scale: 1.00,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.8,
                drift_cents: 4.0,
            },
            Self::Warm => EmotionProfile {
                pressure_cm_h2o: 7.6,
                tension: 0.42,
                brightness: 0.16,
                breathiness: 0.20,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                larynx_cm: -0.28,
                epilarynx: 0.06,
                ventricular: 0.0,
                glottis_noise_scale: 1.00,
                f0_scale: 0.95,
                rate_scale: 0.95,
                vibrato_depth_cents: 8.0,
                vibrato_rate_hz: 5.4,
                drift_cents: 4.0,
            },
            Self::Tender => EmotionProfile {
                pressure_cm_h2o: 5.8,
                tension: 0.32,
                brightness: 0.10,
                breathiness: 0.40,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                larynx_cm: -0.20,
                epilarynx: 0.04,
                ventricular: 0.0,
                glottis_noise_scale: 1.15,
                f0_scale: 0.97,
                rate_scale: 0.90,
                vibrato_depth_cents: 6.0,
                vibrato_rate_hz: 5.3,
                drift_cents: 3.5,
            },
            Self::Joyful => EmotionProfile {
                pressure_cm_h2o: 9.5,
                tension: 0.60,
                brightness: 0.52,
                breathiness: 0.09,
                roughness: 0.05,
                fry: 0.0,
                whisper: false,
                larynx_cm: 0.20,
                epilarynx: 0.32,
                ventricular: 0.0,
                glottis_noise_scale: 0.90,
                f0_scale: 1.18,
                rate_scale: 1.08,
                vibrato_depth_cents: 10.0,
                vibrato_rate_hz: 6.0,
                drift_cents: 5.0,
            },
            Self::Dramatic => EmotionProfile {
                pressure_cm_h2o: 10.6,
                tension: 0.66,
                brightness: 0.54,
                breathiness: 0.07,
                roughness: 0.06,
                fry: 0.0,
                whisper: false,
                larynx_cm: -0.34,
                epilarynx: 0.50,
                ventricular: 0.02,
                glottis_noise_scale: 0.92,
                f0_scale: 1.08,
                rate_scale: 0.93,
                vibrato_depth_cents: 22.0,
                vibrato_rate_hz: 5.7,
                drift_cents: 5.0,
            },
            Self::Melancholic => EmotionProfile {
                pressure_cm_h2o: 6.2,
                tension: 0.35,
                brightness: 0.12,
                breathiness: 0.34,
                roughness: 0.06,
                fry: 0.06,
                whisper: false,
                larynx_cm: -0.26,
                epilarynx: 0.04,
                ventricular: 0.0,
                glottis_noise_scale: 1.10,
                f0_scale: 0.88,
                rate_scale: 0.86,
                vibrato_depth_cents: 6.0,
                vibrato_rate_hz: 5.0,
                drift_cents: 6.0,
            },
            Self::Angry => EmotionProfile {
                pressure_cm_h2o: 10.8,
                tension: 0.78,
                brightness: 0.68,
                breathiness: 0.04,
                roughness: 0.09,
                fry: 0.0,
                whisper: false,
                larynx_cm: 0.22,
                epilarynx: 0.48,
                ventricular: 0.04,
                glottis_noise_scale: 0.95,
                f0_scale: 1.15,
                rate_scale: 1.08,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 6.2,
                drift_cents: 6.5,
            },
            Self::Pressed => EmotionProfile {
                pressure_cm_h2o: 9.6,
                tension: 0.88,
                brightness: 0.86,
                breathiness: 0.02,
                roughness: 0.06,
                fry: 0.0,
                whisper: false,
                larynx_cm: 0.18,
                epilarynx: 0.42,
                ventricular: 0.04,
                glottis_noise_scale: 0.85,
                f0_scale: 1.03,
                rate_scale: 1.00,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.8,
                drift_cents: 3.5,
            },
            Self::Breathy => EmotionProfile {
                pressure_cm_h2o: 7.2,
                tension: 0.20,
                brightness: 0.04,
                breathiness: 0.76,
                roughness: 0.05,
                fry: 0.0,
                whisper: false,
                larynx_cm: -0.10,
                epilarynx: 0.02,
                ventricular: 0.0,
                glottis_noise_scale: 1.38,
                f0_scale: 0.96,
                rate_scale: 0.95,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.5,
                drift_cents: 4.5,
            },
            Self::Whisper => EmotionProfile {
                pressure_cm_h2o: 9.2,
                tension: 0.34,
                brightness: 0.22,
                breathiness: 0.92,
                roughness: 0.04,
                fry: 0.0,
                whisper: true,
                larynx_cm: 0.0,
                epilarynx: 0.24,
                ventricular: 0.04,
                glottis_noise_scale: 3.20,
                f0_scale: 1.00,
                rate_scale: 0.95,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.5,
                drift_cents: 3.0,
            },
            Self::Fry => EmotionProfile {
                pressure_cm_h2o: 5.8,
                tension: 0.56,
                brightness: 0.34,
                breathiness: 0.06,
                roughness: 0.06,
                fry: 0.80,
                whisper: false,
                larynx_cm: -0.16,
                epilarynx: 0.14,
                ventricular: 0.03,
                glottis_noise_scale: 0.90,
                f0_scale: 0.72,
                rate_scale: 0.93,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.0,
                drift_cents: 4.5,
            },
            Self::Operatic => EmotionProfile {
                pressure_cm_h2o: 11.0,
                tension: 0.64,
                brightness: 0.60,
                breathiness: 0.05,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                larynx_cm: -0.46,
                epilarynx: 0.62,
                ventricular: 0.0,
                glottis_noise_scale: 0.88,
                f0_scale: 1.06,
                rate_scale: 0.92,
                vibrato_depth_cents: 40.0,
                vibrato_rate_hz: 5.8,
                drift_cents: 3.5,
            },
            Self::Pop => EmotionProfile {
                pressure_cm_h2o: 10.0,
                tension: 0.68,
                brightness: 0.64,
                breathiness: 0.10,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                larynx_cm: 0.18,
                epilarynx: 0.46,
                ventricular: 0.0,
                glottis_noise_scale: 0.95,
                f0_scale: 1.04,
                rate_scale: 1.02,
                vibrato_depth_cents: 18.0,
                vibrato_rate_hz: 5.9,
                drift_cents: 3.5,
            },
        }
    }
}

/// Physiological muscle, respiratory, and vocal-tract setting profile for an [`EmotionPreset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmotionProfile {
    pub pressure_cm_h2o: f64,
    pub tension: f64,
    pub brightness: f64,
    pub breathiness: f64,
    pub roughness: f64,
    pub fry: f64,
    pub whisper: bool,
    /// Vertical larynx offset (cm; negative lowers larynx & lengthens pharynx, positive raises).
    pub larynx_cm: f64,
    /// Epilaryngeal tube narrowing `0..1` (creates the physical ~3 kHz singer's formant & twang).
    pub epilarynx: f64,
    /// Ventricular (false vocal-fold) constriction `0..1`.
    pub ventricular: f64,
    /// Glottal Reynolds aspiration turbulence multiplier.
    pub glottis_noise_scale: f64,
    pub f0_scale: f64,
    pub rate_scale: f64,
    pub vibrato_depth_cents: f64,
    pub vibrato_rate_hz: f64,
    pub drift_cents: f64,
}

impl EmotionProfile {
    /// Convert this profile into a [`SourcePatch`] compatible with [`VoiceSource`].
    pub fn to_source_patch(&self, base_f0_hz: f64) -> SourcePatch {
        SourcePatch {
            f0: Some(base_f0_hz * self.f0_scale),
            glide: None,
            pressure: Some(self.pressure_cm_h2o),
            breathiness: Some(self.breathiness),
            tension: Some(self.tension),
            fry: Some(self.fry),
            roughness: Some(self.roughness),
            vibrato_depth: Some(self.vibrato_depth_cents),
            vibrato_rate: Some(self.vibrato_rate_hz),
            drift: Some(self.drift_cents),
            voiced: Some(!self.whisper),
            brightness: Some(self.brightness),
        }
    }
}

/// Real-time & offline 4-organ physical human voice.
///
/// Allocates nothing on [`VocalBoxVoice::render`] or [`VocalBoxVoice::telemetry`] once warm.
#[derive(Clone, Debug)]
pub struct VocalBoxVoice {
    pub fs_out: f64,
    pub m: usize,
    pub fs: f64,
    pub body: HumanBody,
    pub body_mix: BodyMix,
    pub anatomy: BodyAnatomy,
    pub airway: HumanAirway,
    pub valve: GlottalValve,
    pub artic: ArticulatorBank,
    pub source: VoiceSource,
    pub score: GestureScore,
    pub clock: f64,
    pub gate: f64,
    /// When `true` (melodies in `song`/`notes`), `ControlKey::F0` renders at exact concert pitch.
    pub musical_pitch_lock: bool,
    pub output_gain: f64,
    pub max_penetration: f64,

    emotion: EmotionPreset,
    dec: Decimator,
    ctrl_every: usize,
    ctrl_counter: i64,
    gate_target: f64,
    score_breath_hold: f64,
    p_lung: f64,
    p_lung_now: f64,
    p_lung_step: f64,
    f0: f64,
    prev_f0_target: f64,
    steady_voiced_s: f64,
    score_f0: Option<f64>,
    base_abduction: f64,
    base_closure: f64,
    base_glottis_noise: f64,
    blend: [Blended; N_CONTROL],

    dc_x: f64,
    dc_y: f64,
    dc_coef: f64,
    lim_peak: f64,
    lim_env: f64,
    lim_attack: f64,
    lim_hold: i64,
    lim_hold_n: i64,
    lim_release: f64,

    // Preallocated telemetry buffers (zero audio-thread heap allocations)
    rms_acc: Vec<f64>,
    rms_count: u32,
    flow_sm: Vec<f64>,
    scope_area: Vec<f32>,
    scope_flow: Vec<f32>,
    scope_pos: usize,
    cycle_starts: [usize; 8],
    cycle_idx: usize,
    last_phase: f64,
    out_acc: f64,
    out_count: u32,
    sub: u32,
}

impl VocalBoxVoice {
    /// Construct a warm, zero-allocation 4-organ [`VocalBoxVoice`] at output sample rate `fs_out` (Hz).
    pub fn new(fs_out: f64, body_mix: impl Into<BodyMix>) -> Self {
        crate::phonetics::planner::init();
        let fs_clean = if fs_out.is_finite() && fs_out >= 4_000.0 {
            fs_out
        } else {
            48_000.0
        };
        let anatomy = BodyAnatomy::from_mix(body_mix);
        let m = oversample_for(fs_clean);
        let fs = fs_clean * m as f64;
        let mut airway = HumanAirway::new(fs, anatomy.body);
        // Pre-warm the airway's slew-limiter scratch buffer at construction so `render()` allocates 0 bytes!
        let init_artic = ArticulatorBank::new();
        airway.apply(&init_artic.x, 1.0 / 1000.0);
        airway.apply(&init_artic.x, 0.0);
        airway.net.snap_to_targets();

        let base_glottis_noise = airway.noise_gain.glottis;
        let n_tract = airway.n_ph + airway.n_or;

        let mut source = DEFAULT_SOURCE;
        source.f0 = anatomy.speaking_f0_hz;
        source.roughness = 0.05;

        let sc = anatomy.vtl_scale;
        let body_gain = if sc >= 1.0 {
            1.0
        } else if sc <= 0.9 {
            0.58
        } else {
            0.58 + 0.42 * (sc - 0.9) / 0.1
        };

        let mut v = Self {
            fs_out: fs_clean,
            m,
            fs,
            body: anatomy.body,
            body_mix: anatomy.mix,
            anatomy,
            airway,
            valve: GlottalValve::new(fs),
            artic: init_artic,
            source,
            score: GestureScore::new(),
            clock: 0.0,
            gate: 0.0,
            musical_pitch_lock: false,
            output_gain: 4.65e-7 * body_gain,
            max_penetration: 0.1,
            emotion: EmotionPreset::Natural,
            dec: Decimator::new(m),
            ctrl_every: 64 * m,
            ctrl_counter: 0,
            gate_target: 0.0,
            score_breath_hold: 0.0,
            p_lung: 0.0,
            p_lung_now: 0.0,
            p_lung_step: 0.0,
            f0: anatomy.speaking_f0_hz,
            prev_f0_target: anatomy.speaking_f0_hz,
            steady_voiced_s: 0.0,
            score_f0: None,
            base_abduction: 0.0,
            base_closure: 0.0,
            base_glottis_noise,
            blend: [Blended::default(); N_CONTROL],
            dc_x: 0.0,
            dc_y: 0.0,
            dc_coef: exp((-2.0 * PI * 25.0) / fs_clean),
            lim_peak: 0.0,
            lim_env: 0.0,
            lim_attack: 1.0 - exp(-1.0 / (0.0025 * fs_clean)),
            lim_hold: 0,
            lim_hold_n: js_round(0.035 * fs_clean) as i64,
            lim_release: exp(-1.0 / (0.18 * fs_clean)),
            rms_acc: vec![0.0; n_tract],
            rms_count: 0,
            flow_sm: vec![0.0; n_tract],
            scope_area: vec![0.0; SCOPE_LEN],
            scope_flow: vec![0.0; SCOPE_LEN],
            scope_pos: 0,
            cycle_starts: [0; 8],
            cycle_idx: 0,
            last_phase: 0.0,
            out_acc: 0.0,
            out_count: 0,
            sub: 0,
        };
        v.apply_emotion_posture(true);
        v.apply_source();
        v
    }

    /// Construct from a [`HumanBody`] (matching `HumanVoice::new`).
    pub fn from_human_body(fs_out: f64, body: HumanBody) -> Self {
        let mix = if (body.scale - BODY_FEMALE.scale).abs() < 0.02 {
            BodyMix::FEMALE
        } else if (body.scale - BODY_CHILD.scale).abs() < 0.02 {
            BodyMix::CHILD
        } else if (body.scale - BODY_MALE.scale).abs() < 0.02 {
            BodyMix::MALE
        } else {
            let f = clamp((1.0 - body.scale) / (1.0 - BODY_CHILD.scale), 0.0, 1.0);
            BodyMix::new(0.0, f)
        };
        let mut v = Self::new(fs_out, mix);
        v.body = body;
        v
    }

    /// Default adult-male [`VocalBoxVoice`] at `fs_out`.
    pub fn with_defaults(fs_out: f64) -> Self {
        Self::new(fs_out, BodyMix::MALE)
    }

    #[inline]
    pub fn emotion(&self) -> EmotionPreset {
        self.emotion
    }

    /// Set expressive [`EmotionPreset`], updating respiratory, laryngeal, and anatomical posture.
    pub fn set_emotion(&mut self, preset: EmotionPreset) {
        self.emotion = preset;
        let p = preset.profile();
        self.source.pressure = p.pressure_cm_h2o;
        self.source.breathiness = p.breathiness;
        self.source.tension = p.tension;
        self.source.brightness = p.brightness;
        self.source.fry = p.fry;
        self.source.roughness = p.roughness;
        self.source.vibrato_depth = p.vibrato_depth_cents;
        self.source.vibrato_rate = p.vibrato_rate_hz;
        self.source.drift = p.drift_cents;
        self.source.voiced = !p.whisper;
        if self.clock == 0.0 {
            let f0_scale_eff = if self.musical_pitch_lock {
                1.0
            } else {
                p.f0_scale
            };
            self.prev_f0_target = self.source.f0 * f0_scale_eff;
        }
        self.apply_emotion_posture(self.clock == 0.0);
        self.apply_source();
        self.ctrl_counter = 0;
    }

    #[inline]
    pub fn two_mass(&self) -> bool {
        self.valve.is_two_mass()
    }

    /// Enable or disable coupled self-oscillating Two-Mass vocal folds across `glottis_port`.
    pub fn set_two_mass(&mut self, enabled: bool) {
        self.apply_source();
        self.valve.set_two_mass(enabled);
        self.ctrl_counter = 0;
    }

    pub fn set_articulation(&mut self, values: &ArticPatch, tip_free: Option<bool>) {
        if let Some(t) = tip_free {
            self.artic.tip_free = t;
        }
        self.artic.set_target(values);
    }

    pub fn set_source(&mut self, values: &SourcePatch) {
        values.apply(&mut self.source);
        if let Some(f0) = values.f0 {
            self.f0 = clamp(f0, 24.0, 2000.0);
            if self.clock == 0.0 {
                let f0_scale_eff = if self.musical_pitch_lock {
                    1.0
                } else {
                    self.emotion.profile().f0_scale
                };
                self.prev_f0_target = self.source.f0 * f0_scale_eff;
            }
        }
        self.apply_source();
        self.ctrl_counter = 0;
    }

    pub fn schedule(&mut self, gestures: &[Gesture]) {
        self.score.add_all(gestures);
    }

    pub fn release(&mut self, tag: f64) {
        self.score.release(tag, self.clock);
    }

    pub fn cancel_score(&mut self) {
        self.score.clear();
    }

    pub fn set_gate(&mut self, level: f64) {
        self.gate_target = clamp(level, 0.0, 1.5);
    }

    /// Instantaneous physical telemetry snapshot without mutating accumulators.
    #[inline]
    pub fn instant_telemetry(&self) -> VocalBoxTelemetry {
        let gp = self.airway.glottis_port;
        let rn = self.airway.rad_nose;
        VocalBoxTelemetry {
            f0_now_hz: self.valve.f0_now(),
            voicing: self.valve.voicing(),
            glottal_area_cm2: self.valve.area(),
            glottal_flow_cm3_s: self.airway.net.u[gp],
            velum_port_cm2: self.airway.frame.as_ref().map_or(0.0, |f| f.velum_port),
            nasal_flow_cm3_s: self.airway.net.radiated_flow(rn),
            p_lung_cm_h2o: self.p_lung_now / CM_H2O,
        }
    }

    /// Deterministically reset airway, glottal valve, articulators, and clock with zero allocations.
    pub fn reset(&mut self) {
        self.artic = ArticulatorBank::new();
        self.clock = 0.0;
        self.gate = 0.0;
        self.gate_target = 0.0;
        self.score_breath_hold = 0.0;
        self.p_lung = 0.0;
        self.p_lung_now = 0.0;
        self.p_lung_step = 0.0;
        self.f0 = self.source.f0;
        let f0_scale_eff = if self.musical_pitch_lock {
            1.0
        } else {
            self.emotion.profile().f0_scale
        };
        self.prev_f0_target = self.source.f0 * f0_scale_eff;
        self.steady_voiced_s = 0.0;
        self.score_f0 = None;
        self.ctrl_counter = 0;
        self.dc_x = 0.0;
        self.dc_y = 0.0;
        self.lim_peak = 0.0;
        self.lim_env = 0.0;
        self.lim_hold = 0;
        self.rms_acc.fill(0.0);
        self.rms_count = 0;
        self.flow_sm.fill(0.0);
        self.scope_area.fill(0.0);
        self.scope_flow.fill(0.0);
        self.scope_pos = 0;
        self.cycle_starts = [0; 8];
        self.cycle_idx = 0;
        self.last_phase = 0.0;
        self.out_acc = 0.0;
        self.out_count = 0;
        self.sub = 0;
        self.dec.reset();
        self.score.clear();
        self.apply_emotion_posture(true);
        self.apply_source();
        self.valve.reset();
    }

    fn apply_emotion_posture(&mut self, snap_now: bool) {
        let p = self.emotion.profile();
        self.airway.noise_gain.glottis = self.base_glottis_noise * p.glottis_noise_scale;
        let mut patch = ArticPatch::default();
        patch.set(ArticKey::Larynx, p.larynx_cm);
        patch.set(ArticKey::Epilarynx, p.epilarynx);
        patch.set(ArticKey::Ventricular, p.ventricular);
        if snap_now {
            self.artic.snap(&patch);
            let x = self.artic.x;
            self.airway.apply(&x, 0.0);
            self.airway.apply(&x, 1.0 / 1000.0);
            self.airway.apply(&x, 0.0);
            self.airway.net.snap_to_targets();
            self.airway.net.reset_deterministic();
        } else {
            self.artic.set_target(&patch);
        }
    }

    fn apply_source(&mut self) {
        let s = self.source;
        let kf = self.anatomy.fold_area_scale;
        let reg = register_state(self.f0, self.anatomy.speaking_f0_hz, s.tension);

        let amp = 0.17 * kf * reg.excursion_scale * if s.voiced { 1.0 } else { 0.0 };
        // In human whispering (Rubin et al. 2006), membranous folds adduct while a supercritical
        // arytenoid whisper aperture (~0.17-0.25 cm^2) drives a turbulent glottal jet (Re >> Re_crit)
        // that excites F1-F3 vowel resonances without periodic oscillation.
        let base_abd = if s.voiced {
            0.10 * s.breathiness * s.breathiness
        } else {
            0.19 + 0.06 * s.breathiness
        } * kf;
        self.base_abduction = base_abd;
        self.base_closure = 0.0;

        let chink = self.anatomy.base_chink_cm2 + 0.032 * s.breathiness * kf;
        let oq = clamp(0.76 - 0.34 * s.tension + reg.oq_shift, 0.28, 0.94);
        let bright = s.brightness - DEFAULT_SOURCE.brightness;
        let by_tension = 0.62 * s.tension + 0.90 * js_max(0.0, s.tension - 0.56);
        let closure_sharpness =
            clamp((by_tension + 0.80 * bright) * reg.sharpness_scale, 0.0, 1.0);
        let skew = clamp(0.57 + 0.16 * s.tension + 0.18 * bright, 0.38, 0.86);

        let jitter = 0.0016 + 0.018 * s.roughness;
        let shimmer = 0.010 + 0.14 * s.roughness;
        let diplophonia = if s.roughness > 0.5 {
            (s.roughness - 0.5) * 1.30
        } else {
            0.0
        };
        let effort = clamp((s.pressure.abs() - 3.0) / 10.0, 0.0, 1.0);

        let idle = self.clock == 0.0;
        let f0 = self.f0;
        let (g1, g2) = self.valve.params_mut();
        for g in [g1, g2] {
            g.f0 = f0;
            g.amplitude = amp;
            g.abduction = base_abd;
            g.chink = chink;
            g.open_quotient = oq;
            g.closure_sharpness = closure_sharpness;
            g.skew = skew;
            g.fry = s.fry;
            g.jitter = jitter;
            g.shimmer = shimmer;
            g.diplophonia = diplophonia;
            g.vibrato_depth = s.vibrato_depth;
            g.vibrato_rate = s.vibrato_rate;
            g.drift = s.drift;
            g.effort = effort;
        }
        if idle {
            self.valve.sync_idle();
        }
    }

    /// Render `out.len()` output samples with **zero heap allocations**.
    pub fn render(&mut self, out: &mut [f32]) {
        let m = self.m;
        let lungs = self.airway.lungs;
        let sub_node = self.airway.trachea.last_node();
        let supra = self.airway.pharynx.first_node();
        let gp = self.airway.glottis_port;
        let n_ph = self.airway.n_ph;
        let n_or = self.airway.n_or;
        let ph_first = self.airway.pharynx.first_node();
        let or_first = self.airway.oral.first_node();

        for o in out.iter_mut() {
            for _ in 0..m {
                if self.ctrl_counter <= 0 {
                    self.control();
                }
                self.ctrl_counter -= 1;

                let net = &mut self.airway.net;
                self.p_lung_now += self.p_lung_step;
                net.set_fixed_pressure(lungs, self.p_lung_now);
                let a = self.valve.step(net.p[sub_node] - net.p[supra]);
                net.set_port_area(gp, a);
                let y = net.step();

                if self.dec.push(y) {
                    let mut v = self.dec.value * self.output_gain;
                    // 25 Hz highpass DC blocker
                    let hp = v - self.dc_x + self.dc_coef * self.dc_y;
                    self.dc_x = v;
                    self.dc_y = hp;
                    v = hp;

                    // Two-stage peak-hold (35 ms hold, 180 ms release) + 2.5 ms smoothed gain-rider:
                    // - Normal speech sits below 0.76 at exact unity gain.
                    // - Sustained loud singing (e.g. Operatic high notes) converges to a steady linear
                    //   gain over 2.5 ms, preserving full waveform crest factor without saturation.
                    // - Rising peaks never evaluate `v * thresh / |v| = thresh`, preventing flat-topping.
                    let av = v.abs();
                    if av >= self.lim_peak {
                        self.lim_peak = av;
                        self.lim_hold = self.lim_hold_n;
                    } else if self.lim_hold > 0 {
                        self.lim_hold -= 1;
                    } else {
                        self.lim_peak *= self.lim_release;
                    }
                    if self.lim_peak > self.lim_env {
                        self.lim_env += self.lim_attack * (self.lim_peak - self.lim_env);
                    } else {
                        self.lim_env = self.lim_peak;
                    }
                    if self.lim_env > 0.76 {
                        v *= 0.76 / self.lim_env;
                    }
                    let al = v.abs();
                    if al > 0.86 {
                        v = js_sign(v) * (0.86 + 0.12 * tanh((al - 0.86) / 0.12));
                    }
                    let vf = if v.is_finite() { v as f32 } else { 0.0 };
                    *o = vf;
                    self.out_acc += v * v;
                    self.out_count = self.out_count.wrapping_add(1);

                    // Phase-locked glottal scope at output rate
                    self.scope_area[self.scope_pos] = a as f32;
                    self.scope_flow[self.scope_pos] = net.u[gp] as f32;
                    let ph = self.valve.cycle_phase();
                    if ph < self.last_phase {
                        self.cycle_starts[self.cycle_idx & 7] = self.scope_pos;
                        self.cycle_idx = self.cycle_idx.wrapping_add(1);
                    }
                    self.last_phase = ph;
                    self.scope_pos = (self.scope_pos + 1) % SCOPE_LEN;
                }

                self.sub = self.sub.wrapping_add(1);
                if self.sub & 3 == 0 {
                    let p = &self.airway.net.p;
                    for k in 0..n_ph {
                        let v = p[ph_first + k];
                        self.rms_acc[k] += v * v;
                    }
                    for k in 0..n_or {
                        let v = p[or_first + k];
                        self.rms_acc[n_ph + k] += v * v;
                    }
                    self.rms_count = self.rms_count.wrapping_add(1);
                }
            }
        }
    }

    fn control(&mut self) {
        let dt = self.ctrl_every as f64 / self.fs;
        let t = self.clock;
        let emo = self.emotion.profile();
        let f0_scale_eff = if self.musical_pitch_lock {
            1.0
        } else {
            emo.f0_scale
        };

        let mut breath_act = 0.0;
        let mut breath_val = 0.0;
        let mut f0_target = self.source.f0 * f0_scale_eff;

        if self.score.size() > 0 {
            self.score.blend_all(t, &mut self.blend);
            let b = &self.blend;
            self.artic.step(dt, Some(&b[..]), 0.05);

            let kf = self.anatomy.fold_area_scale;
            let ab = b[ControlKey::ABDUCTION.index()];
            let ab_act = if ab.weight > 0.0 {
                js_min(1.0, ab.weight / 2.0)
            } else {
                0.0
            };
            let abd = if self.source.voiced {
                self.base_abduction + ab.value * ab_act * kf
            } else {
                js_max(self.base_abduction, ab.value * ab_act * kf)
            };

            let cl = b[ControlKey::CLOSURE.index()];
            let clos = js_max(
                self.base_closure,
                if cl.weight > 0.0 {
                    cl.value * js_min(1.0, cl.weight / 2.0)
                } else {
                    0.0
                },
            );

            let br = b[ControlKey::BREATH.index()];
            if br.weight > 0.0 {
                breath_act = js_min(1.0, br.weight);
                breath_val = br.value;
            }

            let pf = b[ControlKey::F0.index()];
            if pf.weight > 0.0 {
                let raw_f0 = (self.source.f0 * 0.02 + pf.value * pf.weight) / (0.02 + pf.weight);
                f0_target = raw_f0 * f0_scale_eff;
                self.score_f0 = Some(f0_target);
            } else {
                self.score_f0 = None;
            }

            // Sustained-pitch vibrato bloom (Sundberg 1987): suppresses synthetic LFO warble across
            // rapid spoken syllables while blooming organically on held sung notes & long vowels.
            let rel_df0 = (f0_target - self.prev_f0_target).abs() / js_max(f0_target, 50.0);
            self.prev_f0_target = f0_target;
            if self.valve.voicing() > 0.40 && rel_df0 < 1.8 * dt {
                self.steady_voiced_s = js_min(1.0, self.steady_voiced_s + dt);
            } else {
                self.steady_voiced_s = js_max(0.0, self.steady_voiced_s - 3.0 * dt);
            }
            let vib_bloom = smoothstep(0.16, 0.42, self.steady_voiced_s);

            let vb = b[ControlKey::VIBRATO.index()];
            let vib_d = if vb.weight > 0.0 {
                let blended = (self.source.vibrato_depth * vib_bloom * 0.05 + vb.value * vb.weight)
                    / (0.05 + vb.weight);
                if emo.vibrato_depth_cents > 0.0 {
                    js_max(blended, 0.55 * emo.vibrato_depth_cents * vib_bloom)
                } else {
                    blended
                }
            } else {
                self.source.vibrato_depth * vib_bloom
            };

            let (g1, g2) = self.valve.params_mut();
            for g in [g1, g2] {
                g.abduction = abd;
                g.closure = clos;
                g.vibrato_depth = vib_d;
            }

            if js_trunc_i32(self.clock * 10.0) != js_trunc_i32((self.clock - dt) * 10.0) {
                self.score.prune(t - 0.2);
            }
        } else {
            self.artic.step(dt, None, 1.0);
            let rel_df0 = (f0_target - self.prev_f0_target).abs() / js_max(f0_target, 50.0);
            self.prev_f0_target = f0_target;
            if self.valve.voicing() > 0.40 && rel_df0 < 1.8 * dt {
                self.steady_voiced_s = js_min(1.0, self.steady_voiced_s + dt);
            } else {
                self.steady_voiced_s = js_max(0.0, self.steady_voiced_s - 3.0 * dt);
            }
            let vib_bloom = smoothstep(0.16, 0.42, self.steady_voiced_s);
            let (g1, g2) = self.valve.params_mut();
            for g in [g1, g2] {
                g.abduction = self.base_abduction;
                g.closure = self.base_closure;
                g.vibrato_depth = self.source.vibrato_depth * vib_bloom;
            }
            self.score_f0 = None;
        }
        self.clock += dt;

        // Apply 2D midsagittal geometry -> airway area targets with inelastic palate contact
        let x = self.artic.x;
        self.airway.apply(&x, dt);
        if self.enforce_contacts() {
            let x = self.artic.x;
            self.airway.reapply(&x);
        }
        self.airway.net.begin_block(self.ctrl_every);
        self.ctrl_counter = self.ctrl_every as i64;

        // Respiratory breath envelope with prosodic intercostal stress coupling (Ladefoged 1967):
        // stressed syllables naturally receive fuller subglottal support and brighter closure.
        let score_breath = breath_val * breath_act;
        let gate_goal = if score_breath < 0.0 {
            score_breath
        } else {
            js_max(self.gate_target, score_breath)
        };
        self.score_breath_hold = if breath_act > 0.0 {
            0.15
        } else {
            js_max(0.0, self.score_breath_hold - dt)
        };
        let tau = if gate_goal > self.gate {
            0.025
        } else if self.score_breath_hold > 0.0 {
            0.020
        } else {
            0.070
        };
        self.gate += (gate_goal - self.gate) * (1.0 - exp(-dt / tau));

        let f0_base = js_max(self.source.f0 * f0_scale_eff, 40.0);
        let prosodic_p = if !self.musical_pitch_lock && self.score_f0.is_some() {
            let oct = log2(clamp(f0_target / f0_base, 0.5, 2.0));
            clamp(1.0 + 0.34 * oct, 0.90, 1.18)
        } else {
            1.0
        };
        let p_eff_cm_h2o = self.source.pressure * prosodic_p;
        self.p_lung = p_eff_cm_h2o * CM_H2O * self.gate;
        self.p_lung_step = (self.p_lung - self.p_lung_now) / self.ctrl_every as f64;

        // Smooth pitch glide & continuous 5-octave CT/TA registration update
        let g_tau = js_max(0.002, if self.score_f0.is_some() { 0.03 } else { self.source.glide });
        self.f0 *= pow(clamp(f0_target, 24.0, 2000.0) / self.f0, 1.0 - exp(-dt / g_tau));

        let s = self.source;
        let kf = self.anatomy.fold_area_scale;
        let reg = register_state(self.f0, self.anatomy.speaking_f0_hz, s.tension);
        let effort = clamp(((p_eff_cm_h2o * self.gate).abs() - 3.0) / 10.0, 0.0, 1.0);
        let amp = 0.17 * kf * reg.excursion_scale * if s.voiced { 1.0 } else { 0.0 };
        let oq = clamp(0.76 - 0.34 * s.tension + reg.oq_shift, 0.28, 0.94);
        let bright = s.brightness - DEFAULT_SOURCE.brightness;
        let by_tension = 0.62 * s.tension + 0.90 * js_max(0.0, s.tension - 0.56);
        let sharp = clamp((by_tension + 0.80 * bright) * reg.sharpness_scale, 0.0, 1.0);

        let (g1, g2) = self.valve.params_mut();
        for g in [g1, g2] {
            g.f0 = self.f0;
            g.effort = effort;
            g.amplitude = amp;
            g.open_quotient = oq;
            g.closure_sharpness = sharp;
        }

        // Smoothed mean volume velocity per tract section for telemetry visuals
        let a_flow = 1.0 - exp(-dt / 0.03);
        let n_ph = self.airway.n_ph;
        for k in 0..n_ph + self.airway.n_or {
            let br = if k < n_ph {
                self.airway.branch_of(self.airway.pharynx, k as isize)
            } else {
                self.airway.branch_of(self.airway.oral, (k - n_ph) as isize)
            };
            self.flow_sm[k] += a_flow * (self.airway.net.u[br] - self.flow_sm[k]);
        }
    }

    fn enforce_contacts(&mut self) -> bool {
        let tract = &self.airway.tract;
        let k = tract.scale();
        let lim = -self.max_penetration * k;
        let mut changed = false;
        let mut fix = |artic: &mut ArticulatorBank, d: f64, n: Vec2, kx: ArticKey, ky: ArticKey| {
            if d >= lim {
                return;
            }
            changed = true;
            let depth = lim - d;
            let sx = (if artic.x.get(kx) > 0.5 { 1.0 / 0.86 } else { 1.0 }) / k;
            let sy = 1.0 / k;
            *artic.x.get_mut(kx) += n.x * depth * sx;
            *artic.x.get_mut(ky) += n.y * depth * sy;
            let (ix, iy) = (kx.index(), ky.index());
            let vn = artic.v[ix] * n.x / sx + artic.v[iy] * n.y / sy;
            if vn < 0.0 {
                artic.v[ix] -= vn * n.x * sx;
                artic.v[iy] -= vn * n.y * sy;
            }
        };
        let (ct, ctn, cb, cbn) = (
            tract.contact_tip,
            tract.contact_tip_n,
            tract.contact_body,
            tract.contact_body_n,
        );
        fix(&mut self.artic, ct, ctn, ArticKey::TtX, ArticKey::TtY);
        fix(&mut self.artic, cb, cbn, ArticKey::TbX, ArticKey::TbY);
        changed
    }

    /// Fill a [`VoiceTelemetry`] snapshot with **zero heap allocations** once `out` is warm.
    pub fn telemetry(&mut self, out: &mut VoiceTelemetry) {
        let n_t = self.airway.n_ph + self.airway.n_or;
        out.artic.clear();
        out.artic
            .extend(ArticKey::ALL.iter().map(|k| self.artic.x.get(*k) as f32));
        debug_assert_eq!(out.artic.len(), N_ARTIC);

        let c = js_max(1.0, self.rms_count as f64);
        out.pressure_rms.clear();
        out.pressure_rms
            .extend((0..n_t).map(|k| sqrt(self.rms_acc[k] / c) as f32));
        self.rms_acc.fill(0.0);
        self.rms_count = 0;

        let mut start = (self.scope_pos + SCOPE_LEN - 1024) % SCOPE_LEN;
        let mut len = 1024usize;
        if self.cycle_idx >= 3 && self.valve.voicing() > 0.2 {
            let c2 = self.cycle_starts[(self.cycle_idx - 3) & 7];
            let c0 = self.cycle_starts[(self.cycle_idx - 1) & 7];
            let span = (c0 + SCOPE_LEN - c2) % SCOPE_LEN;
            if span > 8 && span < SCOPE_LEN / 2 {
                start = c2;
                len = span;
            }
        }
        out.glottal_area.clear();
        out.glottal_flow.clear();
        for i in 0..SCOPE_POINTS {
            let idx = (start + (i * len) / SCOPE_POINTS) % SCOPE_LEN;
            out.glottal_area.push(self.scope_area[idx]);
            out.glottal_flow.push(self.scope_flow[idx]);
        }

        let mut noise_level = 0.0;
        let mut noise_node = -1i32;
        for s in self.airway.net.noise_sources() {
            if s.level > noise_level {
                noise_level = s.level;
                noise_node = s.src as i32;
            }
        }
        let out_rms = sqrt(self.out_acc / js_max(1.0, self.out_count as f64));
        self.out_acc = 0.0;
        self.out_count = 0;

        out.ph_areas.clear();
        out.ph_areas
            .extend(self.airway.ph_areas.iter().map(|&x| x as f32));
        out.or_areas.clear();
        out.or_areas
            .extend(self.airway.or_areas.iter().map(|&x| x as f32));
        out.ph_len = self.airway.ph_len;
        out.or_len = self.airway.or_len;
        out.flow.clear();
        out.flow.extend(self.flow_sm.iter().map(|&x| x as f32));
        out.velum_port = self.airway.frame.as_ref().map_or(0.0, |f| f.velum_port);
        out.f0 = self.valve.f0_now();
        out.voicing = self.valve.voicing();
        out.p_lung = self.p_lung_now / CM_H2O;
        out.out_rms = out_rms;
        out.noise_level = noise_level;
        out.noise_node = noise_node;
        out.gate = self.gate;
        out.clock = self.clock;
        out.active.clear();
        out.active.reserve(4);
        for g in self.score.active_at(self.clock) {
            if out.active.len() == 4 {
                break;
            }
            if let Some(l) = g.label
                && !l.is_empty()
                && l.as_str() != "breath"
                && l.as_str() != "pitch"
            {
                out.active.push(l);
            }
        }
    }
}
