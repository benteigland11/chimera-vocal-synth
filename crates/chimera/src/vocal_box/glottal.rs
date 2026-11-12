//! Decoupled laryngeal anatomy (`BodyAnatomy`) and 5-octave CT/TA registration (`Registration`).

use crate::anatomy::human::{BodyMix, BodyName, HumanBody};
use crate::math::{clamp, js_max, log2, smoothstep};

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
            BodyName::Male => 0.0018,
            BodyName::Female => 0.0048,
            BodyName::Child => 0.0030,
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
