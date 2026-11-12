//! High-level expressive real-time voice (`VocalBoxVoice`) and 13 physiological emotion presets
//! built atop [`VocalBoxCore`], [`ArticulatorBank`], and [`GestureScore`].
//!
//! Bridges Chimera's articulatory & phonetic planners (`plan_utterance`, `plan_phrases`,
//! `plan_song`, `plan_note_on`, `set_articulation`) directly onto the first-principles
//! [`VocalBoxCore`] synthesis engine with:
//! - **Zero heap allocations** on the audio thread (`render(&mut [f32])` and `telemetry()`),
//! - **13 canonical `EmotionPreset`s** grounded in Fant $R_d$, Sundberg epilarynx singer's ring,
//!   piriform sinus depth, laryngeal VTL posture, subglottal effort, vibrato, whisper, and fry,
//! - **Aerodynamic consonant transitions**: oral stop intraoral pressure buildup + plosive release
//!   bursts + VOT aspiration, place-ordered `/s, ʃ, f, θ/` frication jets, lateral `/l/`, rhotic
//!   `/ɹ, ɝ/`, and zero-leak oral vs `/m, n, ɲ, ŋ/` nasal coupling.

use super::{
    BodyPreset, ConstrictionPlace, VocalBoxCore, VocalFrame, vowel_targets_with_vtl,
};
use crate::anatomy::human::{
    ArticKey, ArticPatch, BODY_CHILD, BODY_FEMALE, BODY_MALE, BodyMix, HumanArtic, HumanBody,
    N_ARTIC,
};
use crate::control::articulators::ArticulatorBank;
use crate::control::score::{Blended, ControlKey, Gesture, GestureScore, N_CONTROL, activation};
use crate::math::{
    Drift, TAU, clamp, exp, js_max, js_min, js_trunc_i32, lerp, pow, sin, sqrt,
};
use crate::phonetics::inventory::{Glottal, Manner, Voicing, phoneme};
use crate::phonetics::planner::{vowel_artic, vowel_table};
use crate::phonetics::vowel_data::VOWELS;
use crate::voice::human_voice::{
    DEFAULT_SOURCE, SCOPE_POINTS, SourcePatch, VoiceSource, VoiceTelemetry,
};
use serde::{Deserialize, Serialize};

const SCOPE_LEN: usize = 4096;
const MAX_VOWEL_ANCHORS: usize = 32;
const N_PH_TELE: usize = 18;
const N_OR_TELE: usize = 22;
const N_TRACT_TELE: usize = N_PH_TELE + N_OR_TELE;

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

    /// Physiological laryngeal, respiratory, supraglottal, and prosodic parameters for this preset.
    #[inline]
    pub const fn profile(self) -> EmotionProfile {
        match self {
            Self::Natural => EmotionProfile {
                rd: 0.92,
                tension: 0.50,
                brightness: 0.25,
                pressure_cm_h2o: 8.0,
                breathiness: 0.12,
                aspiration_gain: 0.015,
                roughness: 0.05,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.14,
                piriform_depth: 0.22,
                formant_scale: 1.000,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.6,
                f0_scale: 1.00,
                rate_scale: 1.00,
                drift_cents: 4.5,
            },
            Self::Warm => EmotionProfile {
                rd: 1.26,
                tension: 0.40,
                brightness: 0.16,
                pressure_cm_h2o: 7.4,
                breathiness: 0.20,
                aspiration_gain: 0.024,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.16,
                piriform_depth: 0.36,
                formant_scale: 0.962,
                vibrato_depth_cents: 18.0,
                vibrato_rate_hz: 5.3,
                f0_scale: 0.95,
                rate_scale: 0.95,
                drift_cents: 4.0,
            },
            Self::Tender => EmotionProfile {
                rd: 1.72,
                tension: 0.29,
                brightness: 0.11,
                pressure_cm_h2o: 5.4,
                breathiness: 0.38,
                aspiration_gain: 0.055,
                roughness: 0.03,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.07,
                piriform_depth: 0.28,
                formant_scale: 0.985,
                vibrato_depth_cents: 12.0,
                vibrato_rate_hz: 5.1,
                f0_scale: 0.98,
                rate_scale: 0.88,
                drift_cents: 4.5,
            },
            Self::Joyful => EmotionProfile {
                rd: 0.72,
                tension: 0.63,
                brightness: 0.60,
                pressure_cm_h2o: 9.8,
                breathiness: 0.08,
                aspiration_gain: 0.014,
                roughness: 0.05,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.40,
                piriform_depth: 0.16,
                formant_scale: 1.048,
                vibrato_depth_cents: 16.0,
                vibrato_rate_hz: 6.1,
                f0_scale: 1.15,
                rate_scale: 1.10,
                drift_cents: 5.0,
            },
            Self::Dramatic => EmotionProfile {
                rd: 0.64,
                tension: 0.68,
                brightness: 0.54,
                pressure_cm_h2o: 11.4,
                breathiness: 0.07,
                aspiration_gain: 0.015,
                roughness: 0.09,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.62,
                piriform_depth: 0.42,
                formant_scale: 0.952,
                vibrato_depth_cents: 44.0,
                vibrato_rate_hz: 5.7,
                f0_scale: 1.06,
                rate_scale: 0.92,
                drift_cents: 6.0,
            },
            Self::Melancholic => EmotionProfile {
                rd: 1.54,
                tension: 0.34,
                brightness: 0.13,
                pressure_cm_h2o: 5.8,
                breathiness: 0.28,
                aspiration_gain: 0.038,
                roughness: 0.08,
                fry: 0.06,
                whisper: false,
                epilarynx_ring: 0.08,
                piriform_depth: 0.30,
                formant_scale: 0.968,
                vibrato_depth_cents: 9.0,
                vibrato_rate_hz: 4.9,
                f0_scale: 0.88,
                rate_scale: 0.84,
                drift_cents: 7.0,
            },
            Self::Angry => EmotionProfile {
                rd: 0.46,
                tension: 0.86,
                brightness: 0.84,
                pressure_cm_h2o: 12.8,
                breathiness: 0.04,
                aspiration_gain: 0.022,
                roughness: 0.34,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.48,
                piriform_depth: 0.14,
                formant_scale: 1.028,
                vibrato_depth_cents: 6.0,
                vibrato_rate_hz: 6.5,
                f0_scale: 1.18,
                rate_scale: 1.12,
                drift_cents: 8.5,
            },
            Self::Pressed => EmotionProfile {
                rd: 0.41,
                tension: 0.93,
                brightness: 0.90,
                pressure_cm_h2o: 10.2,
                breathiness: 0.02,
                aspiration_gain: 0.008,
                roughness: 0.10,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.45,
                piriform_depth: 0.14,
                formant_scale: 1.015,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.8,
                f0_scale: 1.02,
                rate_scale: 1.00,
                drift_cents: 3.5,
            },
            Self::Breathy => EmotionProfile {
                rd: 2.28,
                tension: 0.21,
                brightness: 0.08,
                pressure_cm_h2o: 6.6,
                breathiness: 0.74,
                aspiration_gain: 0.042,
                roughness: 0.06,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.05,
                piriform_depth: 0.24,
                formant_scale: 0.995,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.4,
                f0_scale: 0.97,
                rate_scale: 0.96,
                drift_cents: 5.5,
            },
            Self::Whisper => EmotionProfile {
                rd: 2.50,
                tension: 0.35,
                brightness: 0.30,
                pressure_cm_h2o: 7.5,
                breathiness: 0.95,
                aspiration_gain: 0.40,
                roughness: 0.0,
                fry: 0.0,
                whisper: true,
                epilarynx_ring: 0.10,
                piriform_depth: 0.20,
                formant_scale: 1.010,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.5,
                f0_scale: 1.00,
                rate_scale: 0.95,
                drift_cents: 0.0,
            },
            Self::Fry => EmotionProfile {
                rd: 0.52,
                tension: 0.68,
                brightness: 0.62,
                pressure_cm_h2o: 4.8,
                breathiness: 0.04,
                aspiration_gain: 0.010,
                roughness: 0.28,
                fry: 0.88,
                whisper: false,
                epilarynx_ring: 0.22,
                piriform_depth: 0.20,
                formant_scale: 0.990,
                vibrato_depth_cents: 0.0,
                vibrato_rate_hz: 5.0,
                f0_scale: 0.52,
                rate_scale: 0.92,
                drift_cents: 9.0,
            },
            Self::Operatic => EmotionProfile {
                rd: 0.68,
                tension: 0.64,
                brightness: 0.56,
                pressure_cm_h2o: 11.8,
                breathiness: 0.06,
                aspiration_gain: 0.012,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.85,
                piriform_depth: 0.56,
                formant_scale: 0.945,
                vibrato_depth_cents: 56.0,
                vibrato_rate_hz: 5.8,
                f0_scale: 1.00,
                rate_scale: 0.94,
                drift_cents: 4.0,
            },
            Self::Pop => EmotionProfile {
                rd: 0.58,
                tension: 0.72,
                brightness: 0.74,
                pressure_cm_h2o: 10.6,
                breathiness: 0.09,
                aspiration_gain: 0.016,
                roughness: 0.04,
                fry: 0.0,
                whisper: false,
                epilarynx_ring: 0.55,
                piriform_depth: 0.18,
                formant_scale: 1.038,
                vibrato_depth_cents: 22.0,
                vibrato_rate_hz: 5.9,
                f0_scale: 1.04,
                rate_scale: 1.02,
                drift_cents: 4.0,
            },
        }
    }
}

/// Physical vocal posture parameters defining an [`EmotionPreset`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmotionProfile {
    pub rd: f64,
    pub tension: f64,
    pub brightness: f64,
    pub pressure_cm_h2o: f64,
    pub breathiness: f64,
    pub aspiration_gain: f64,
    pub roughness: f64,
    pub fry: f64,
    pub whisper: bool,
    pub epilarynx_ring: f64,
    pub piriform_depth: f64,
    pub formant_scale: f64,
    pub vibrato_depth_cents: f64,
    pub vibrato_rate_hz: f64,
    pub f0_scale: f64,
    pub rate_scale: f64,
    pub drift_cents: f64,
}

impl EmotionProfile {
    /// Apply this emotion's phonatory and supraglottal posture directly onto a [`VocalFrame`].
    pub fn apply_to_frame(&self, frame: &mut VocalFrame) {
        frame.rd = self.rd;
        frame.tension = self.tension;
        frame.brightness = self.brightness;
        frame.subglottal_pressure_cm_h2o = self.pressure_cm_h2o;
        frame.abduction = if self.whisper {
            0.85
        } else {
            0.35 * self.breathiness * self.breathiness
        };
        frame.aspiration_gain = self.aspiration_gain;
        frame.roughness = self.roughness;
        frame.fry = self.fry;
        frame.whisper = self.whisper;
        frame.epilarynx_ring = self.epilarynx_ring;
        frame.piriform_depth = self.piriform_depth;
        if (self.formant_scale - 1.0).abs() > 1e-6 {
            frame.vtl_scale = clamp(frame.vtl_scale / self.formant_scale, 0.45, 1.65);
            for f in &mut frame.formants_hz {
                *f *= self.formant_scale;
            }
        }
        frame.f0_hz = clamp(frame.f0_hz * self.f0_scale, 24.0, 2400.0);
    }

    /// Convert into a [`SourcePatch`] compatible with [`crate::voice::human_voice::HumanVoice`].
    pub fn to_source_patch(&self, base_f0: f64) -> SourcePatch {
        SourcePatch {
            f0: Some(clamp(base_f0 * self.f0_scale, 32.0, 1600.0)),
            pressure: Some(self.pressure_cm_h2o),
            breathiness: Some(self.breathiness),
            tension: Some(self.tension),
            brightness: Some(self.brightness),
            fry: Some(self.fry),
            roughness: Some(self.roughness),
            vibrato_depth: Some(self.vibrato_depth_cents),
            vibrato_rate: Some(self.vibrato_rate_hz),
            drift: Some(self.drift_cents),
            voiced: Some(!self.whisper),
            glide: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct VowelAnchor {
    coords: [f64; 9],
    formants_hz: [f64; 5],
    bandwidths_hz: [f64; 5],
    rhotic: f64,
}

impl Default for VowelAnchor {
    fn default() -> Self {
        Self {
            coords: [0.0; 9],
            formants_hz: [500.0, 1500.0, 2500.0, 3500.0, 4500.0],
            bandwidths_hz: [70.0, 95.0, 145.0, 210.0, 285.0],
            rhotic: 0.0,
        }
    }
}

#[inline]
fn artic_feature_vec(a: &HumanArtic) -> [f64; 9] {
    [
        a.tb_x * 0.65,
        a.tb_y * 0.65,
        a.jaw * 1.15,
        a.lip_ap * 0.90,
        a.lip_prot * 0.85,
        a.lip_width * 0.95,
        a.root * 0.70,
        a.larynx * 0.60,
        a.tt_curl * 2.20,
    ]
}

#[inline]
fn patch_to_feature_vec(p: &ArticPatch, base: &HumanArtic) -> [f64; 9] {
    let mut a = *base;
    for (k, v) in p.iter() {
        *a.get_mut(k) = v;
    }
    artic_feature_vec(&a)
}

/// Map `PhonemeDef::place` string onto [`ConstrictionPlace`].
#[inline]
fn map_place(place: &str) -> ConstrictionPlace {
    match place {
        "bilabial" => ConstrictionPlace::Bilabial,
        "labiodental" => ConstrictionPlace::Labiodental,
        "dental" => ConstrictionPlace::Dental,
        "alveolar" | "lateral" => ConstrictionPlace::Alveolar,
        "postalveolar" | "alveolopalatal" | "retroflex" => ConstrictionPlace::Postalveolar,
        "palatal" | "labial-velar" => ConstrictionPlace::Palatal,
        "velar" => ConstrictionPlace::Velar,
        "uvular" | "pharyngeal" | "epiglottal" => ConstrictionPlace::Uvular,
        "glottal" => ConstrictionPlace::Glottal,
        _ => ConstrictionPlace::None,
    }
}

/// Approximate consonant formant locus `(F1, F2, F3)` in adult-male Hz (Delattre et al. 1955; Stevens 1998).
#[inline]
fn place_locus_male_hz(place: ConstrictionPlace, current_f2: f64, vtl_scale: f64) -> Option<[f64; 3]> {
    let f2_male = current_f2 * vtl_scale;
    let loc = match place {
        ConstrictionPlace::Bilabial | ConstrictionPlace::Labiodental => [260.0, 880.0, 2180.0],
        ConstrictionPlace::Dental => [280.0, 1620.0, 2560.0],
        ConstrictionPlace::Alveolar => [285.0, 1780.0, 2660.0],
        ConstrictionPlace::Postalveolar => [275.0, 2020.0, 2480.0],
        ConstrictionPlace::Palatal => [255.0, 2260.0, 2920.0],
        ConstrictionPlace::Velar | ConstrictionPlace::VelarBack => {
            // Classic velar pinch: back-velar before back vowels, front-velar before front vowels
            let f2_vel = clamp(0.65 * f2_male + 520.0, 1150.0, 2150.0);
            [260.0, f2_vel, js_max(f2_vel + 220.0, 2240.0)]
        }
        ConstrictionPlace::Uvular => [340.0, 1120.0, 2320.0],
        ConstrictionPlace::None | ConstrictionPlace::Glottal => return None,
    };
    let inv = 1.0 / vtl_scale;
    Some([loc[0] * inv, loc[1] * inv, loc[2] * inv])
}

#[inline]
fn is_known_vowel_label(label: &str) -> bool {
    if label.is_empty() {
        return false;
    }
    let clean = label
        .trim()
        .trim_matches(|c: char| c == 'ˈ' || c == 'ˌ' || c == 'ː' || c == ':');
    matches!(
        clean,
        "aɪ" | "aʊ" | "ɔɪ" | "eɪ" | "oʊ" | "ɚ" | "ʉ" | "ʏ" | "ɘ" | "ɵ" | "ɜ" | "ɞ" | "ɶ"
    ) || VOWELS.iter().any(|v| v.ipa == clean)
}

/// Real-time & offline expressive human voice driven by [`VocalBoxCore`], [`ArticulatorBank`], and [`GestureScore`].
///
/// Allocates nothing on [`VocalBoxVoice::render`] or [`VocalBoxVoice::telemetry`] once constructed.
#[derive(Clone, Debug)]
pub struct VocalBoxVoice {
    pub fs_out: f64,
    pub body: HumanBody,
    pub body_mix: BodyMix,
    pub core: VocalBoxCore,
    pub artic: ArticulatorBank,
    pub source: VoiceSource,
    pub score: GestureScore,
    pub clock: f64,
    pub gate: f64,
    gate_target: f64,
    score_breath_hold: f64,
    f0: f64,
    score_f0: Option<f64>,
    emotion: EmotionPreset,
    two_mass: bool,
    /// When `true` (used for musical melodies `song`/`notes`), `ControlKey::F0` is rendered at
    /// exact musical concert pitch without applying `EmotionProfile::f0_scale` on top of MIDI Hz.
    pub musical_pitch_lock: bool,

    // Control-rate stepping (every 32 output samples = 0.67 ms at 48 kHz)
    ctrl_every: usize,
    ctrl_counter: i64,
    blend: [Blended; N_CONTROL],

    // Vibrato & slow natural pitch drift (allocation-free)
    vib_phase: f64,
    drift: Drift,
    drift_val: f64,

    // Supraglottal stop / burst aerodynamics
    p_oral_cm_h2o: f64,
    was_stop_closed: bool,
    burst_env: f64,
    last_stop_place: ConstrictionPlace,

    // Pre-computed calibrated vowel anchors for continuous ArticulatorBank -> Formant Shepard mapping
    anchors: [VowelAnchor; MAX_VOWEL_ANCHORS],
    n_anchors: usize,

    // Fixed-capacity telemetry & scope buffers (zero heap allocation during render/telemetry)
    scope_area: Vec<f32>,
    scope_flow: Vec<f32>,
    scope_pos: usize,
    cycle_starts: [usize; 8],
    cycle_idx: usize,
    last_area: f64,
    out_acc: f64,
    out_count: u32,
    ph_areas: Vec<f64>,
    or_areas: Vec<f64>,
    rms_acc: Vec<f64>,
    rms_count: u32,
    flow_sm: Vec<f64>,
    last_noise_level: f64,
    last_noise_node: i32,
}

impl VocalBoxVoice {
    /// Construct a warm, zero-allocation [`VocalBoxVoice`] at output sample rate `fs_out` (Hz).
    pub fn new(fs_out: f64, body_mix: impl Into<BodyMix>) -> Self {
        crate::phonetics::planner::init();
        let mix: BodyMix = body_mix.into();
        let mix = BodyMix::new(mix.female, mix.child);
        let human_body = mix.body();
        let preset = BodyPreset::from_mix(mix);
        let mut core = VocalBoxCore::with_body(fs_out, mix);
        let mut rest_frame = super::VocalParams::from_body(mix).build();
        rest_frame.subglottal_pressure_cm_h2o = 0.0;
        core.set_frame_immediate(rest_frame);
        let artic = ArticulatorBank::new();

        // Precompute calibrated vowel anchors for this BodyMix
        let mut anchors = [VowelAnchor::default(); MAX_VOWEL_ANCHORS];
        let mut n_anchors = 0usize;
        let vt = vowel_table();
        if let Some(male_map) = vt.get("male") {
            for ipa in male_map.keys() {
                if n_anchors >= MAX_VOWEL_ANCHORS {
                    break;
                }
                if let Some(patch) = vowel_artic(ipa, mix) {
                    let (formants_hz, bandwidths_hz) =
                        vowel_targets_with_vtl(ipa, mix, preset.vtl_scale);
                    let rhotic = if ipa == "ɝ" || ipa == "ɚ" { 1.0 } else { 0.0 };
                    anchors[n_anchors] = VowelAnchor {
                        coords: patch_to_feature_vec(&patch, &artic.x),
                        formants_hz,
                        bandwidths_hz,
                        rhotic,
                    };
                    n_anchors += 1;
                }
            }
        }

        let ctrl_every = clamp((fs_out / 1500.0).round(), 8.0, 64.0) as usize;
        let mut source = DEFAULT_SOURCE;
        source.f0 = preset.f0_hz;

        Self {
            fs_out,
            body: human_body,
            body_mix: mix,
            core,
            artic,
            source,
            score: GestureScore::new(),
            clock: 0.0,
            gate: 0.0,
            gate_target: 0.0,
            score_breath_hold: 0.0,
            f0: preset.f0_hz,
            score_f0: None,
            emotion: EmotionPreset::Natural,
            two_mass: false,
            musical_pitch_lock: false,
            ctrl_every,
            ctrl_counter: 0,
            blend: [Blended::default(); N_CONTROL],
            vib_phase: 0.0,
            drift: Drift::new(fs_out / ctrl_every as f64, 0.32, 0x4242),
            drift_val: 0.0,
            p_oral_cm_h2o: 0.0,
            was_stop_closed: false,
            burst_env: 0.0,
            last_stop_place: ConstrictionPlace::Bilabial,
            anchors,
            n_anchors,
            scope_area: vec![0.0; SCOPE_LEN],
            scope_flow: vec![0.0; SCOPE_LEN],
            scope_pos: 0,
            cycle_starts: [0; 8],
            cycle_idx: 0,
            last_area: 0.0,
            out_acc: 0.0,
            out_count: 0,
            ph_areas: vec![2.5; N_PH_TELE],
            or_areas: vec![3.0; N_OR_TELE],
            rms_acc: vec![0.0; N_TRACT_TELE],
            rms_count: 0,
            flow_sm: vec![0.0; N_TRACT_TELE],
            last_noise_level: 0.0,
            last_noise_node: -1,
        }
    }

    /// Construct from a [`HumanBody`] (matching `HumanVoice::new(fs_out, body, _)`).
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
        Self::new(fs_out, BodyMix::new(0.0, 0.0))
    }

    #[inline]
    pub fn emotion(&self) -> EmotionPreset {
        self.emotion
    }

    /// Set expressive [`EmotionPreset`] and synchronize `source` phonatory defaults.
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
        self.ctrl_counter = 0;
    }

    #[inline]
    pub fn two_mass(&self) -> bool {
        self.two_mass
    }

    /// Enable or disable coupled self-oscillating Two-Mass vocal folds inside [`VocalBoxCore`].
    pub fn set_two_mass(&mut self, enabled: bool) {
        self.two_mass = enabled;
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
            self.f0 = clamp(f0, 24.0, 2400.0);
        }
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

    /// Reset clock, score, articulators, aerodynamics, and [`VocalBoxCore`] state without allocating.
    pub fn reset(&mut self) {
        self.core.reset();
        let mut init_frame = super::VocalParams::from_body(self.body_mix).build();
        init_frame.subglottal_pressure_cm_h2o = 0.0;
        self.core.set_frame_immediate(init_frame);
        self.artic = ArticulatorBank::new();
        self.score.clear();
        self.clock = 0.0;
        self.gate = 0.0;
        self.gate_target = 0.0;
        self.score_breath_hold = 0.0;
        self.f0 = self.source.f0;
        self.score_f0 = None;
        self.vib_phase = 0.0;
        self.p_oral_cm_h2o = 0.0;
        self.was_stop_closed = false;
        self.burst_env = 0.0;
        self.ctrl_counter = 0;
        self.out_acc = 0.0;
        self.out_count = 0;
        self.rms_acc.fill(0.0);
        self.rms_count = 0;
    }

    /// Render `out.len()` output samples with **zero heap allocations**.
    pub fn render(&mut self, out: &mut [f32]) {
        for o in out.iter_mut() {
            if self.ctrl_counter <= 0 {
                self.control();
            }
            self.ctrl_counter -= 1;

            let y = self.core.step();
            *o = y;

            let y64 = y as f64;
            self.out_acc += y64 * y64;
            self.out_count = self.out_count.wrapping_add(1);

            let tel = self.core.telemetry();
            self.scope_area[self.scope_pos] = tel.glottal_area_cm2 as f32;
            self.scope_flow[self.scope_pos] = tel.glottal_flow_cm3_s as f32;
            if self.last_area <= 0.002 && tel.glottal_area_cm2 > 0.002 {
                self.cycle_starts[self.cycle_idx & 7] = self.scope_pos;
                self.cycle_idx = self.cycle_idx.wrapping_add(1);
            }
            self.last_area = tel.glottal_area_cm2;
            self.scope_pos = (self.scope_pos + 1) % SCOPE_LEN;

            if self.out_count & 3 == 0 {
                let e = tel.raw_tract_out * tel.raw_tract_out;
                for r in &mut self.rms_acc {
                    *r += e;
                }
                self.rms_count = self.rms_count.wrapping_add(1);
            }
        }
    }

    fn map_artic_to_formants(&self) -> ([f64; 5], [f64; 5], f64) {
        if self.n_anchors == 0 {
            let preset = BodyPreset::from_mix(self.body_mix);
            let (f, b) = vowel_targets_with_vtl("ə", self.body_mix, preset.vtl_scale);
            return (f, b, 0.0);
        }
        let q = artic_feature_vec(&self.artic.x);
        let mut f_sum = [0.0; 5];
        let mut b_sum = [0.0; 5];
        let mut r_sum = 0.0;
        let mut w_sum = 0.0;

        for idx in 0..self.n_anchors {
            let anc = &self.anchors[idx];
            let mut d2 = 0.0;
            for (&qv, &av) in q.iter().zip(anc.coords.iter()) {
                let d = qv - av;
                d2 += d * d;
            }
            if d2 < 1e-7 {
                return (anc.formants_hz, anc.bandwidths_hz, anc.rhotic);
            }
            // Power-4 Shepard kernel snaps crisply onto calibrated vowel targets while interpolating smoothly
            let w = 1.0 / (d2 * d2 + 1e-6);
            w_sum += w;
            for k in 0..5 {
                f_sum[k] += w * anc.formants_hz[k];
                b_sum[k] += w * anc.bandwidths_hz[k];
            }
            r_sum += w * anc.rhotic;
        }
        let inv_w = 1.0 / js_max(w_sum, 1e-12);
        for k in 0..5 {
            f_sum[k] *= inv_w;
            b_sum[k] *= inv_w;
        }
        (f_sum, b_sum, clamp(r_sum * inv_w, 0.0, 1.0))
    }

    fn control(&mut self) {
        let dt = self.ctrl_every as f64 / self.fs_out;
        let t = self.clock;
        let preset = BodyPreset::from_mix(self.body_mix);
        let emo = self.emotion.profile();

        let mut breath_act = 0.0;
        let mut breath_val = 0.0;
        let f0_scale_eff = if self.musical_pitch_lock {
            1.0
        } else {
            emo.f0_scale
        };
        let mut f0_target = self.source.f0 * f0_scale_eff;
        let mut score_abduction = 0.0;
        let mut score_closure = 0.0;
        let mut vib_depth = js_max(self.source.vibrato_depth, emo.vibrato_depth_cents);

        if self.score.size() > 0 {
            self.score.blend_all(t, &mut self.blend);
            let b = &self.blend;
            self.artic.step(dt, Some(&b[..]), 0.05);

            let ab = b[ControlKey::ABDUCTION.index()];
            if ab.weight > 0.0 {
                score_abduction = ab.value * js_min(1.0, ab.weight / 2.0);
            }
            let cl = b[ControlKey::CLOSURE.index()];
            if cl.weight > 0.0 {
                score_closure = cl.value * js_min(1.0, cl.weight / 2.0);
            }
            let br = b[ControlKey::BREATH.index()];
            if br.weight > 0.0 {
                breath_act = js_min(1.0, br.weight);
                breath_val = br.value;
            }
            let vb = b[ControlKey::VIBRATO.index()];
            if vb.weight > 0.0 {
                let base_vb = js_max(self.source.vibrato_depth, emo.vibrato_depth_cents);
                let blended = (base_vb * 0.05 + vb.value * vb.weight) / (0.05 + vb.weight);
                vib_depth = if emo.vibrato_depth_cents > 0.0 {
                    js_max(blended, 0.75 * emo.vibrato_depth_cents)
                } else {
                    blended
                };
            }
            let pf = b[ControlKey::F0.index()];
            if pf.weight > 0.0 {
                let raw_f0 = (self.source.f0 * 0.02 + pf.value * pf.weight) / (0.02 + pf.weight);
                f0_target = raw_f0 * f0_scale_eff;
                self.score_f0 = Some(f0_target);
            } else {
                self.score_f0 = None;
            }
            if js_trunc_i32(self.clock * 10.0) != js_trunc_i32((self.clock - dt) * 10.0) {
                self.score.prune(t - 0.2);
            }
        } else {
            self.artic.step(dt, None, 1.0);
            self.score_f0 = None;
        }
        self.clock += dt;

        // Breath envelope (matches HumanVoice phrasing & note-off timing)
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
        let tau_gate = if gate_goal > self.gate {
            0.022
        } else if self.score_breath_hold > 0.0 {
            0.018
        } else {
            0.065
        };
        self.gate += (gate_goal - self.gate) * (1.0 - exp(-dt / tau_gate));

        // Smooth pitch glide + vibrato + physiological drift
        let g_tau = js_max(0.002, if self.score_f0.is_some() { 0.026 } else { self.source.glide });
        self.f0 *= pow(
            clamp(f0_target, 24.0, 2400.0) / js_max(self.f0, 24.0),
            1.0 - exp(-dt / g_tau),
        );
        self.drift_val = self.drift.step();
        let vib_rate = if self.source.vibrato_rate > 0.1 {
            self.source.vibrato_rate
        } else {
            emo.vibrato_rate_hz
        };
        self.vib_phase += (vib_rate * dt) * (1.0 + 0.04 * self.drift_val);
        if self.vib_phase >= 1.0 {
            self.vib_phase -= 1.0;
        }
        let cents_mod = vib_depth * sin(TAU * self.vib_phase) + self.source.drift * self.drift_val;
        let mut f0_inst = self.f0 * pow(2.0, cents_mod / 1200.0);
        let fry_amt = clamp(js_max(self.source.fry, emo.fry), 0.0, 1.0);
        if fry_amt > 0.45 && !self.musical_pitch_lock {
            f0_inst = lerp(f0_inst, js_min(f0_inst, 54.0), 0.70 * fry_amt);
        }

        // Base vowel formants from ArticulatorBank Shepard mapping, refined by active vowel gestures
        let (mut formants, mut bandwidths, mut rhotic) = self.map_artic_to_formants();
        let mut v_wsum = 0.0;
        let mut v_fsum = [0.0; 5];
        let mut v_bsum = [0.0; 5];
        let mut v_rsum = 0.0;

        let mut dom_place = ConstrictionPlace::None;
        let mut dom_place_act = 0.0;
        let mut stop_closed = false;
        let mut stop_voiceless = false;
        let mut stop_act_max = 0.0;
        let mut nasal_seg_act = 0.0;
        let mut fric_gain = 0.0;
        let mut h_asp_boost = 0.0;
        let mut lateral_act = clamp(self.artic.x.lateral, 0.0, 1.0);

        for g in self.score.active_at(t) {
            let act = activation(g, t);
            if act <= 0.0 {
                continue;
            }
            let lbl = g.label_str();
            if is_known_vowel_label(lbl) {
                let w = act * js_max(g.strength, 0.25);
                let (vf, vb) = vowel_targets_with_vtl(lbl, self.body_mix, preset.vtl_scale);
                v_wsum += w;
                for k in 0..5 {
                    v_fsum[k] += w * vf[k];
                    v_bsum[k] += w * vb[k];
                }
                if lbl.contains('ɝ') || lbl.contains('ɚ') {
                    v_rsum += w;
                }
                continue;
            }
            if lbl == "h" || lbl == "ɦ" {
                h_asp_boost = js_max(h_asp_boost, act * if lbl == "h" { 0.24 } else { 0.14 });
                continue;
            }
            if lbl == "aspiration" {
                h_asp_boost = js_max(h_asp_boost, act * 0.16);
                continue;
            }
            if let Some(def) = phoneme(lbl, self.body_mix) {
                let pl = map_place(def.place);
                if act > dom_place_act && pl != ConstrictionPlace::None {
                    dom_place_act = act;
                    dom_place = pl;
                }
                match def.manner {
                    Manner::Stop => {
                        stop_act_max = js_max(stop_act_max, act);
                        if act > 0.42 {
                            stop_closed = true;
                            stop_voiceless |= def.voicing == Voicing::Voiceless;
                            if pl != ConstrictionPlace::None {
                                self.last_stop_place = pl;
                            }
                        }
                    }
                    Manner::Affricate => {
                        let span = js_max(g.t1 - g.t0, 0.02);
                        if t < g.t0 + 0.52 * span && act > 0.42 {
                            stop_closed = true;
                            stop_voiceless |= def.voicing == Voicing::Voiceless;
                            stop_act_max = js_max(stop_act_max, act);
                            if pl != ConstrictionPlace::None {
                                self.last_stop_place = pl;
                            }
                        } else {
                            let sibilant = matches!(
                                pl,
                                ConstrictionPlace::Alveolar | ConstrictionPlace::Postalveolar
                            );
                            fric_gain = js_max(fric_gain, act * if sibilant { 0.44 } else { 0.22 });
                        }
                    }
                    Manner::Nasal => {
                        nasal_seg_act = js_max(nasal_seg_act, act);
                    }
                    Manner::Fricative => {
                        if def.glottal == Some(Glottal::Open) {
                            h_asp_boost = js_max(h_asp_boost, act * 0.14);
                        } else {
                            let sibilant = matches!(
                                pl,
                                ConstrictionPlace::Alveolar | ConstrictionPlace::Postalveolar
                            );
                            let base_fg = if sibilant { 0.48 } else { 0.24 };
                            let v_scale = if def.voicing == Voicing::Voiced {
                                0.55
                            } else {
                                1.0
                            };
                            fric_gain = js_max(fric_gain, act * base_fg * v_scale);
                        }
                    }
                    Manner::Lateral => {
                        lateral_act = js_max(lateral_act, act);
                    }
                    Manner::Approximant => {
                        if def.ipa == "ɹ" || def.ipa == "ɻ" {
                            rhotic = js_max(rhotic, act);
                        }
                    }
                    Manner::Tap | Manner::Trill => {
                        stop_act_max = js_max(stop_act_max, 0.65 * act);
                        if act > 0.65 {
                            stop_closed = true;
                        }
                    }
                    Manner::Glottal | Manner::Vowel => {}
                }
            }
        }

        if v_wsum > 0.0 {
            let base_w = 0.04;
            let tot = v_wsum + base_w;
            for k in 0..5 {
                formants[k] = (v_fsum[k] + base_w * formants[k]) / tot;
                bandwidths[k] = (v_bsum[k] + base_w * bandwidths[k]) / tot;
            }
            rhotic = js_max(rhotic, v_rsum / tot);
        }
        rhotic = js_max(rhotic, clamp((self.artic.x.tt_curl - 0.08) * 1.15, 0.0, 1.0));

        // Place-of-articulation locus transition into/out of adjacent consonants
        if dom_place_act > 0.05
            && let Some(loc) = place_locus_male_hz(dom_place, formants[1], preset.vtl_scale)
        {
            let pull = clamp(0.38 * dom_place_act, 0.0, 0.48);
            for k in 0..3 {
                formants[k] = lerp(formants[k], loc[k], pull);
            }
        }

        // Apply emotion laryngeal VTL / smile-darken formant scaling
        let vtl_eff = clamp(preset.vtl_scale / emo.formant_scale, 0.45, 1.65);
        if (emo.formant_scale - 1.0).abs() > 1e-5 {
            for f in &mut formants {
                *f *= emo.formant_scale;
            }
        }

        // Zero-leak velopharyngeal port: strictly 0.0 cm^2 on oral vowels without active nasal gesture
        let raw_velum = self.artic.x.velum;
        let velum_port_cm2 = if raw_velum <= 0.018 && nasal_seg_act <= 0.01 {
            0.0
        } else {
            clamp(js_max(raw_velum - 0.012, 0.44 * nasal_seg_act), 0.0, 1.8)
        };

        // Subglottal pressure & stop/fricative supraglottal back-pressure aerodynamics (Stevens 1998)
        let p_sub = self.source.pressure * self.gate;
        if stop_closed && velum_port_cm2 <= 0.02 {
            // Voiceless stops equalize intraoral pressure fast (devoicing cleanly); voiced stops
            // expand passive pharyngeal walls so a low-frequency voice bar persists before release.
            let (p_goal, tau_charge) = if stop_voiceless || score_abduction > 0.12 {
                (p_sub, 0.012)
            } else {
                (0.76 * p_sub, 0.062)
            };
            self.p_oral_cm_h2o += (p_goal - self.p_oral_cm_h2o) * (1.0 - exp(-dt / tau_charge));
        } else if fric_gain > 0.08 && velum_port_cm2 <= 0.02 {
            // Narrow oral fricative constriction divides lung pressure across glottis & obstacle:
            // voiceless fricatives drop ~85% of lung pressure at the oral constriction (silencing
            // low-frequency glottal /h/ rumble); voiced fricatives retain ~45% transglottal pressure.
            let p_fric_goal = if score_abduction > 0.15 {
                0.86 * p_sub
            } else {
                0.52 * p_sub
            };
            self.p_oral_cm_h2o += (p_fric_goal - self.p_oral_cm_h2o) * (1.0 - exp(-dt / 0.010));
        } else {
            if self.was_stop_closed && self.p_oral_cm_h2o > 1.0 && velum_port_cm2 <= 0.05 {
                self.burst_env = clamp(self.p_oral_cm_h2o / 8.0, 0.25, 1.25);
            }
            self.p_oral_cm_h2o *= exp(-dt / 0.0055);
            if self.p_oral_cm_h2o < 1e-4 {
                self.p_oral_cm_h2o = 0.0;
            }
        }
        self.was_stop_closed = stop_closed && velum_port_cm2 <= 0.02;

        if self.burst_env > 1e-4 {
            fric_gain = js_max(fric_gain, 0.30 * self.burst_env);
            if dom_place == ConstrictionPlace::None {
                dom_place = self.last_stop_place;
            }
            self.burst_env *= exp(-dt / 0.009);
            if self.burst_env < 1e-4 {
                self.burst_env = 0.0;
            }
        }

        // Oral constriction orifice area (cm^2):
        // - Nasal murmur (/m, n, ŋ/) or full stop closure -> 0.0 cm^2
        // - Fricative -> narrow turbulent orifice (0.08 cm^2)
        // - Open vowel -> open lip area (~2.8..4.2 cm^2)
        let constriction_area_cm2 = if (stop_closed && velum_port_cm2 <= 0.02) || nasal_seg_act > 0.55 {
            0.0
        } else if fric_gain > 0.08 {
            0.08
        } else if stop_act_max > 0.05 || nasal_seg_act > 0.05 {
            lerp(3.2, 0.04, clamp(js_max(stop_act_max, nasal_seg_act), 0.0, 1.0))
        } else {
            clamp(1.6 + 1.4 * js_max(0.1, self.artic.x.lip_ap), 0.4, 6.0)
        };

        // Combine source parameters, emotion profile, and score abduction/closure
        let whisper = !self.source.voiced || emo.whisper;
        let base_abd = if whisper {
            1.0
        } else {
            0.35 * self.source.breathiness * self.source.breathiness
        };
        let abduction = if score_closure > 0.5 {
            1.15 // glottal stop /ʔ/ halts periodic fold vibration cleanly
        } else {
            clamp(base_abd + 3.4 * score_abduction, 0.0, 1.35)
        };

        // Derive continuous Rd from tension & emotion Rd so SourcePatch changes work intuitively too
        let tension = clamp(self.source.tension, 0.0, 1.0);
        let tension_delta = tension - emo.tension;
        let rd = clamp(emo.rd * exp(-1.15 * tension_delta), 0.34, 2.85);
        let glottal_asp_gate = 1.0 - clamp(fric_gain * 4.2, 0.0, 0.92);
        let asp_gain = clamp(
            (emo.aspiration_gain
                + 0.05 * self.source.breathiness * self.source.breathiness
                + h_asp_boost
                + 0.14 * score_abduction)
                * glottal_asp_gate,
            0.0,
            1.2,
        );
        let epilarynx_ring = clamp(
            js_max(emo.epilarynx_ring, self.artic.x.epilarynx),
            0.0,
            1.0,
        );

        let target_frame = VocalFrame {
            f0_hz: clamp(f0_inst, 24.0, 2400.0),
            subglottal_pressure_cm_h2o: if score_closure > 0.65 { 0.0 } else { p_sub },
            oral_pressure_cm_h2o: self.p_oral_cm_h2o,
            rd,
            tension,
            brightness: clamp(self.source.brightness, 0.0, 1.0),
            abduction,
            roughness: clamp(js_max(self.source.roughness, emo.roughness), 0.0, 1.0),
            fry: fry_amt,
            whisper,
            two_mass: self.two_mass,
            aspiration_gain: asp_gain,
            vtl_scale: vtl_eff,
            fold_area_scale: preset.fold_area_scale,
            formants_hz: formants,
            bandwidths_hz: bandwidths,
            epilarynx_ring,
            piriform_depth: emo.piriform_depth,
            lateral_mix: lateral_act,
            rhotic_mix: rhotic,
            velum_port_cm2,
            constriction_place: dom_place,
            constriction_area_cm2,
            frication_gain: clamp(fric_gain, 0.0, 1.35),
        };

        if t == 0.0 {
            self.core.set_frame_immediate(target_frame);
        } else {
            self.core.set_target_frame(target_frame, dt);
        }
        self.ctrl_counter = self.ctrl_every as i64;

        // Update synthetic tract display areas & smoothed flow for `--tele` visual compatibility
        let jaw_open = clamp(self.artic.x.jaw, 0.0, 2.0);
        for (i, a) in self.ph_areas.iter_mut().enumerate() {
            let u = i as f64 / N_PH_TELE as f64;
            *a = clamp(1.8 + 1.4 * u - 0.8 * epilarynx_ring * (1.0 - u), 0.2, 6.0);
        }
        for (i, a) in self.or_areas.iter_mut().enumerate() {
            let u = i as f64 / N_OR_TELE as f64;
            *a = clamp(
                lerp(2.2 + 0.8 * jaw_open, constriction_area_cm2, u * u),
                0.0,
                8.0,
            );
        }
        let flow_now = self.core.telemetry().glottal_flow_cm3_s;
        let af = 1.0 - exp(-dt / 0.03);
        for fl in &mut self.flow_sm {
            *fl += af * (flow_now - *fl);
        }
        self.last_noise_level = clamp(fric_gain + 0.5 * h_asp_boost, 0.0, 1.5);
        self.last_noise_node = if fric_gain > 0.05 {
            (N_PH_TELE + N_OR_TELE - 3) as i32
        } else {
            -1
        };
    }

    /// Fill a [`VoiceTelemetry`] snapshot with **zero heap allocations** once `out` is warm.
    pub fn telemetry(&mut self, out: &mut VoiceTelemetry) {
        out.artic.clear();
        out.artic
            .extend(ArticKey::ALL.iter().map(|k| self.artic.x.get(*k) as f32));
        debug_assert_eq!(out.artic.len(), N_ARTIC);

        let c = js_max(1.0, self.rms_count as f64);
        out.pressure_rms.clear();
        out.pressure_rms
            .extend((0..N_TRACT_TELE).map(|k| sqrt(self.rms_acc[k] / c) as f32));
        self.rms_acc.fill(0.0);
        self.rms_count = 0;

        let tel = self.core.telemetry();
        let mut start = (self.scope_pos + SCOPE_LEN - 1024) % SCOPE_LEN;
        let mut len = 1024usize;
        if self.cycle_idx >= 3 && tel.voicing > 0.2 {
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

        let out_rms = sqrt(self.out_acc / js_max(1.0, self.out_count as f64));
        self.out_acc = 0.0;
        self.out_count = 0;

        let cur = self.core.current_frame();
        out.ph_areas.clear();
        out.ph_areas.extend(self.ph_areas.iter().map(|&x| x as f32));
        out.or_areas.clear();
        out.or_areas.extend(self.or_areas.iter().map(|&x| x as f32));
        out.ph_len = 8.4 * cur.vtl_scale;
        out.or_len = 9.1 * cur.vtl_scale;
        out.flow.clear();
        out.flow.extend(self.flow_sm.iter().map(|&x| x as f32));
        out.velum_port = cur.velum_port_cm2;
        out.f0 = tel.f0_now_hz;
        out.voicing = tel.voicing;
        out.p_lung = cur.subglottal_pressure_cm_h2o;
        out.out_rms = out_rms;
        out.noise_level = self.last_noise_level;
        out.noise_node = self.last_noise_node;
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
