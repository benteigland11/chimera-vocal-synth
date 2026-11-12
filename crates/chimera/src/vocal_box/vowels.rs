//! Vocal-box parameter frames, continuous `BodyMix` presets, and calibrated IPA vowel targets.
//!
//! Pure-Rust, allocation-free, and deterministic across native and `wasm32` targets via [`crate::math`].

use crate::anatomy::human::{BodyMix, BodyName};
use crate::math::{clamp, lerp, pow};
use crate::phonetics::vowel_data::VOWELS;

/// Replace non-finite values (`NaN`, `±Inf`) with `fallback`.
#[inline(always)]
pub(crate) fn sanitize(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// Replace non-finite values with `fallback` and clamp to `[lo, hi]`.
#[inline(always)]
pub(crate) fn sanitize_clamped(x: f64, fallback: f64, lo: f64, hi: f64) -> f64 {
    let v = if x.is_finite() { x } else { fallback };
    clamp(v, lo, hi)
}

/// Flush IEEE subnormals and non-finite numbers to exact `+0.0`.
#[inline(always)]
pub(crate) fn flush_subnormal(x: f64) -> f64 {
    if !x.is_finite() || x.abs() < 1e-24 { 0.0 } else { x }
}

/// $C^1$-continuous polynomial smooth maximum $\approx \max(a, b)$ with transition width `k >= 0`.
///
/// Used for soprano *aggiustamento* ($F_1 \ge 1.08 f_0$) without slope discontinuities.
/// Sanitizes non-finite inputs (`NaN`, `±Inf`) safely to `0.0`.
#[inline]
pub fn soft_max_c1(a: f64, b: f64, k: f64) -> f64 {
    let a = sanitize_clamped(a, 0.0, -1e9, 1e9);
    let b = sanitize_clamped(b, 0.0, -1e9, 1e9);
    let k = sanitize_clamped(k, 0.0, 0.0, 1e6);
    let m = if a > b { a } else { b };
    if k <= 1e-12 {
        return m;
    }
    let h = clamp(1.0 - (a - b).abs() / k, 0.0, 1.0);
    m + 0.25 * k * h * h
}

/// Anatomical preset parameters decoupled across vocal-tract length (`vtl_scale`) and
/// membranous vocal-fold area (`fold_area_scale`).
///
/// Sexual dimorphism in human larynges scales membranous fold length (~16 mm male vs ~10 mm female)
/// more strongly than vocal-tract length (~17.5 cm vs ~15.7 cm), so `fold_area_scale` is strictly
/// decoupled from `vtl_scale^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyPreset {
    /// Vocal-tract length ratio relative to adult male (`1.00` male, `0.90` female, `0.74` child).
    pub vtl_scale: f64,
    /// Membranous glottal area scale relative to adult male (`1.00` male, `0.62` female, `0.48` child).
    pub fold_area_scale: f64,
    /// Default speaking fundamental frequency (Hz).
    pub f0_hz: f64,
    /// Default Fant glottal shape parameter $R_d$.
    pub rd: f64,
}

impl BodyPreset {
    pub const MALE: BodyPreset = BodyPreset {
        vtl_scale: 1.00,
        fold_area_scale: 1.00,
        f0_hz: 110.0,
        rd: 0.92,
    };

    pub const FEMALE: BodyPreset = BodyPreset {
        vtl_scale: 0.90,
        fold_area_scale: 0.62,
        f0_hz: 205.0,
        rd: 1.10,
    };

    pub const CHILD: BodyPreset = BodyPreset {
        vtl_scale: 0.74,
        fold_area_scale: 0.48,
        f0_hz: 265.0,
        rd: 1.20,
    };

    pub fn for_name(name: BodyName) -> BodyPreset {
        match name {
            BodyName::Male => Self::MALE,
            BodyName::Female => Self::FEMALE,
            BodyName::Child => Self::CHILD,
        }
    }

    /// Continuous barycentric blend across `Male`, `Female`, and `Child`.
    pub fn from_mix(body: impl Into<BodyMix>) -> BodyPreset {
        let mix: BodyMix = body.into();
        let mix = BodyMix::new(mix.female, mix.child);
        if let Some(p) = mix.preset() {
            return Self::for_name(p);
        }
        BodyPreset {
            vtl_scale: mix.blend(|b| Self::for_name(b).vtl_scale),
            fold_area_scale: mix.blend(|b| Self::for_name(b).fold_area_scale),
            f0_hz: mix.blend(|b| Self::for_name(b).f0_hz),
            rd: mix.blend(|b| Self::for_name(b).rd),
        }
    }
}

/// Blended vocal-tract length scale (`1.0` adult male) for any [`BodyMix`].
#[inline]
pub fn body_vtl_scale(body: impl Into<BodyMix>) -> f64 {
    BodyPreset::from_mix(body).vtl_scale
}

/// Supraglottal oral constriction place (controls nasal oral-shunt anti-formants and frication locus).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ConstrictionPlace {
    #[default]
    None,
    Bilabial,
    Labiodental,
    Dental,
    Alveolar,
    Postalveolar,
    Palatal,
    Velar,
    VelarBack,
    Uvular,
    Glottal,
}

/// Instantaneous physiological & acoustic control frame for [`super::VocalBoxCore`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VocalFrame {
    /// Target fundamental frequency (Hz).
    pub f0_hz: f64,
    /// Subglottal lung pressure ($\text{cmH}_2\text{O}$).
    pub subglottal_pressure_cm_h2o: f64,
    /// Supraglottal intraoral back-pressure ($\text{cmH}_2\text{O}$); $\Delta P = P_{\text{sub}} - P_{\text{oral}}$.
    pub oral_pressure_cm_h2o: f64,
    /// Continuous Fant glottal wave shape parameter $R_d \in [0.30, 2.80]$.
    pub rd: f64,
    /// Adductory/cricothyroid vocal tension (`0.0` lax .. `1.0` pressed, `0.5` neutral).
    pub tension: f64,
    /// Return-phase brightness (`0.0` dark .. `1.0` bright, `0.25` neutral).
    pub brightness: f64,
    /// Static glottal abduction (`0.0` modal .. `1.0` fully abducted, $\text{cm}^2$-scaled).
    pub abduction: f64,
    /// Aperiodic cycle-to-cycle jitter/shimmer (`0.0` .. `1.0`).
    pub roughness: f64,
    /// Subharmonic vocal fry / creak (`0.0` .. `1.0`).
    pub fry: f64,
    /// Pure turbulent whisper mode (suppresses periodic fold oscillation, keeps open chink turbulence).
    pub whisper: bool,
    /// Select self-oscillating coupled Two-Mass vocal folds (`true`) vs transformed LF model (`false`).
    pub two_mass: bool,
    /// Glottal aspiration noise gain (`0.0` .. `1.0`).
    pub aspiration_gain: f64,
    /// Vocal-tract length scale (`1.0` adult male).
    pub vtl_scale: f64,
    /// Membranous vocal-fold area scale (`1.0` adult male).
    pub fold_area_scale: f64,
    /// Five vocal-tract formant frequencies $F_1 \dots F_5$ (Hz).
    pub formants_hz: [f64; 5],
    /// Five vocal-tract closed-glottis formant bandwidths $B_1 \dots B_5$ (Hz).
    pub bandwidths_hz: [f64; 5],
    /// Epilaryngeal narrow-tube clustering (`0.0` .. `1.0`): boosts singer's formant (`2.85–3.35 kHz`).
    pub epilarynx_ring: f64,
    /// Piriform sinus side-branch depth (`0.0` .. `1.0`): carves `~4.6 kHz / vtl_scale` spectral trough.
    pub piriform_depth: f64,
    /// Lateral tongue-side channel coupling (`0.0` .. `1.0` for `/l/` anti-formant at `2.0–3.2 kHz`).
    pub lateral_mix: f64,
    /// Rhotic retroflex/bunched $F_3$ lowering (`0.0` .. `1.0` for `/ɹ, ɝ/`).
    pub rhotic_mix: f64,
    /// Velopharyngeal port area ($\text{cm}^2$, `0.0` = hermetically sealed oral tract).
    pub velum_port_cm2: f64,
    /// Place of oral constriction (determines nasal shunt zero and fricative jet spectrum).
    pub constriction_place: ConstrictionPlace,
    /// Minimum oral constriction cross-sectional area ($\text{cm}^2$, `0.0` = full stop/nasal occlusion).
    pub constriction_area_cm2: f64,
    /// Supraglottal frication jet excitation level (`0.0` .. `1.0`).
    pub frication_gain: f64,
}

impl Default for VocalFrame {
    fn default() -> Self {
        VocalFrame {
            f0_hz: 110.0,
            subglottal_pressure_cm_h2o: 8.0,
            oral_pressure_cm_h2o: 0.0,
            rd: 0.92,
            tension: 0.50,
            brightness: 0.25,
            abduction: 0.0,
            roughness: 0.0,
            fry: 0.0,
            whisper: false,
            two_mass: false,
            aspiration_gain: 0.015,
            vtl_scale: 1.0,
            fold_area_scale: 1.0,
            formants_hz: [500.0, 1500.0, 2500.0, 3500.0, 4500.0],
            bandwidths_hz: [76.0, 96.0, 148.0, 210.0, 290.0],
            epilarynx_ring: 0.12,
            piriform_depth: 0.22,
            lateral_mix: 0.0,
            rhotic_mix: 0.0,
            velum_port_cm2: 0.0,
            constriction_place: ConstrictionPlace::None,
            constriction_area_cm2: 3.5,
            frication_gain: 0.0,
        }
    }
}

impl VocalFrame {
    /// Return a sanitized copy with all fields guaranteed finite and within safe physical bounds.
    pub fn sanitized(&self) -> VocalFrame {
        let def = VocalFrame::default();
        let vtl = sanitize_clamped(self.vtl_scale, def.vtl_scale, 0.45, 1.65);
        let mut f = [0.0; 5];
        let mut b = [0.0; 5];
        let f_lo = [150.0, 450.0, 1100.0, 2200.0, 3000.0];
        let f_hi = [1600.0, 4200.0, 5600.0, 7200.0, 9000.0];
        for i in 0..5 {
            let fi = sanitize(self.formants_hz[i], def.formants_hz[i]);
            f[i] = clamp(fi, f_lo[i], f_hi[i]);
            if i > 0 && f[i] < f[i - 1] + 120.0 {
                f[i] = (f[i - 1] + 120.0).min(f_hi[i]);
            }
            b[i] = sanitize_clamped(self.bandwidths_hz[i], def.bandwidths_hz[i], 32.0, 900.0);
        }
        let velum_raw = sanitize(self.velum_port_cm2, 0.0);
        let velum = if velum_raw <= 1e-7 { 0.0 } else { clamp(velum_raw, 0.0, 2.5) };
        VocalFrame {
            f0_hz: sanitize_clamped(self.f0_hz, def.f0_hz, 24.0, 2400.0),
            subglottal_pressure_cm_h2o: sanitize_clamped(self.subglottal_pressure_cm_h2o, def.subglottal_pressure_cm_h2o, -5.0, 45.0),
            oral_pressure_cm_h2o: sanitize_clamped(self.oral_pressure_cm_h2o, 0.0, -5.0, 45.0),
            rd: sanitize_clamped(self.rd, def.rd, 0.30, 3.20),
            tension: sanitize_clamped(self.tension, def.tension, 0.0, 1.0),
            brightness: sanitize_clamped(self.brightness, def.brightness, 0.0, 1.0),
            abduction: sanitize_clamped(self.abduction, 0.0, 0.0, 1.5),
            roughness: sanitize_clamped(self.roughness, 0.0, 0.0, 1.0),
            fry: sanitize_clamped(self.fry, 0.0, 0.0, 1.0),
            whisper: self.whisper,
            two_mass: self.two_mass,
            aspiration_gain: sanitize_clamped(self.aspiration_gain, def.aspiration_gain, 0.0, 1.5),
            vtl_scale: vtl,
            fold_area_scale: sanitize_clamped(self.fold_area_scale, def.fold_area_scale, 0.20, 2.50),
            formants_hz: f,
            bandwidths_hz: b,
            epilarynx_ring: sanitize_clamped(self.epilarynx_ring, def.epilarynx_ring, 0.0, 1.0),
            piriform_depth: sanitize_clamped(self.piriform_depth, def.piriform_depth, 0.0, 1.0),
            lateral_mix: sanitize_clamped(self.lateral_mix, 0.0, 0.0, 1.0),
            rhotic_mix: sanitize_clamped(self.rhotic_mix, 0.0, 0.0, 1.0),
            velum_port_cm2: velum,
            constriction_place: self.constriction_place,
            constriction_area_cm2: sanitize_clamped(self.constriction_area_cm2, def.constriction_area_cm2, 0.0, 10.0),
            frication_gain: sanitize_clamped(self.frication_gain, 0.0, 0.0, 1.5),
        }
    }
}

/// Builder over [`VocalFrame`] with safe VTL scaling and physiological sanitization.
#[derive(Clone, Copy, Debug, Default)]
pub struct VocalParams {
    frame: VocalFrame,
}

impl VocalParams {
    pub fn new(frame: VocalFrame) -> Self {
        Self { frame: frame.sanitized() }
    }

    pub fn from_body(body: impl Into<BodyMix>) -> Self {
        let p = BodyPreset::from_mix(body);
        let mut frame = VocalFrame {
            f0_hz: p.f0_hz,
            rd: p.rd,
            vtl_scale: p.vtl_scale,
            fold_area_scale: p.fold_area_scale,
            ..VocalFrame::default()
        };
        let inv = 1.0 / p.vtl_scale;
        for f in &mut frame.formants_hz {
            *f *= inv;
        }
        Self { frame: frame.sanitized() }
    }

    pub fn from_vowel(ipa: &str, body: impl Into<BodyMix>) -> Self {
        Self { frame: vowel_frame(ipa, body) }
    }

    pub fn with_f0(mut self, f0_hz: f64) -> Self {
        self.frame.f0_hz = sanitize_clamped(f0_hz, self.frame.f0_hz, 24.0, 2400.0);
        self
    }

    pub fn with_rd(mut self, rd: f64) -> Self {
        self.frame.rd = sanitize_clamped(rd, self.frame.rd, 0.30, 3.20);
        self
    }

    pub fn with_pressure(mut self, p_sub_cm_h2o: f64, p_oral_cm_h2o: f64) -> Self {
        self.frame.subglottal_pressure_cm_h2o = sanitize_clamped(p_sub_cm_h2o, 8.0, -5.0, 45.0);
        self.frame.oral_pressure_cm_h2o = sanitize_clamped(p_oral_cm_h2o, 0.0, -5.0, 45.0);
        self
    }

    pub fn with_vtl_scale(mut self, new_vtl: f64) -> Self {
        let old_vtl = sanitize_clamped(self.frame.vtl_scale, 1.0, 0.45, 1.65);
        let target_vtl = sanitize_clamped(new_vtl, old_vtl, 0.45, 1.65);
        let ratio = old_vtl / target_vtl;
        for f in &mut self.frame.formants_hz {
            let base = sanitize(*f, 500.0);
            *f = base * ratio;
        }
        self.frame.vtl_scale = target_vtl;
        self
    }

    pub fn with_fold_area_scale(mut self, scale: f64) -> Self {
        self.frame.fold_area_scale = sanitize_clamped(scale, self.frame.fold_area_scale, 0.20, 2.50);
        self
    }

    pub fn with_two_mass(mut self, enabled: bool) -> Self {
        self.frame.two_mass = enabled;
        self
    }

    pub fn with_nasal(mut self, velum_port_cm2: f64, place: ConstrictionPlace) -> Self {
        let v = sanitize(velum_port_cm2, 0.0);
        self.frame.velum_port_cm2 = if v <= 1e-7 { 0.0 } else { clamp(v, 0.0, 2.5) };
        self.frame.constriction_place = place;
        if place != ConstrictionPlace::None && self.frame.velum_port_cm2 > 0.0 {
            self.frame.constriction_area_cm2 = 0.0;
        }
        self
    }

    pub fn build(self) -> VocalFrame {
        self.frame.sanitized()
    }
}

/// Effective glottal closure sharpness (`0.05` very soft .. `0.98` crisp/pressed) derived from
/// continuous Fant $R_d$, vocal tension, and brightness.
#[inline]
pub fn closure_sharpness_from_frame(frame: &VocalFrame) -> f64 {
    let rd = sanitize_clamped(frame.rd, 0.92, 0.30, 3.20);
    let tension = sanitize_clamped(frame.tension, 0.50, 0.0, 1.0);
    let brightness = sanitize_clamped(frame.brightness, 0.25, 0.0, 1.0);
    // Low Rd -> sharp return phase; tension and brightness tilt return corner sharper.
    let from_rd = 1.0 / (1.0 + pow(rd / 0.95, 1.65));
    let raw = from_rd + 0.28 * (tension - 0.50) + 0.36 * (brightness - 0.25);
    clamp(raw, 0.04, 0.98)
}

/// Look up male/female/child `[F1, F2, F3]` (Hz) for any IPA vowel symbol (30 monophthongs +
/// diphthong nuclei + length-stripped tokens).
fn lookup_vowel_ref_f123(ipa: &str) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
    let clean = ipa.trim().trim_matches(|c: char| c == 'ˈ' || c == 'ˌ' || c == 'ː' || c == ':' || c == '\u{303}' || c == '\u{324}');
    // Diphthongs map to their primary nucleus target when queried as a static vowel.
    let key = match clean {
        "eɪ" => "e",
        "oʊ" => "o",
        "aɪ" => "a",
        "aʊ" => "a",
        "ɔɪ" => "ɔ",
        other => other,
    };
    if let Some(v) = VOWELS.iter().find(|v| v.ipa == key) {
        return (v.male, v.female, v.child, v.height);
    }
    // Remaining 8 IPA vowel characters from `phonetics::ipa::VOWEL_CHARS` (`ʉ, ʏ, ɘ, ɵ, ɜ, ɞ, ɶ, ɚ`).
    let scale = |m: [f64; 3], k: [f64; 3]| [m[0] * k[0], m[1] * k[1], m[2] * k[2]];
    let fem = [1.17, 1.20, 1.15];
    let chi = [1.30, 1.33, 1.24];
    let (male, height): ([f64; 3], f64) = match key {
        "ɚ" => {
            return (
                [474.0, 1379.0, 1710.0],
                [523.0, 1588.0, 1929.0],
                [586.0, 1719.0, 2143.0],
                0.45,
            );
        }
        "ʉ" => ([320.0, 1650.0, 2250.0], 0.0),
        "ʏ" => ([380.0, 1680.0, 2250.0], 0.2),
        "ɘ" => ([430.0, 1500.0, 2450.0], 0.35),
        "ɵ" => ([430.0, 1280.0, 2350.0], 0.35),
        "ɜ" => ([560.0, 1420.0, 2480.0], 0.6),
        "ɞ" => ([570.0, 1240.0, 2400.0], 0.6),
        "ɶ" => ([730.0, 1360.0, 2450.0], 0.95),
        _ => ([500.0, 1450.0, 2450.0], 0.5),
    };
    (male, scale(male, fem), scale(male, chi), height)
}

/// Calibrated 5-formant frequencies and closed-glottis bandwidths `([F1..F5], [B1..B5])` (Hz)
/// for an IPA vowel and continuous [`BodyMix`].
pub fn vowel_targets(ipa: &str, body: impl Into<BodyMix>) -> ([f64; 5], [f64; 5]) {
    let mix: BodyMix = body.into();
    let mix = BodyMix::new(mix.female, mix.child);
    let preset = BodyPreset::from_mix(mix);
    vowel_targets_with_vtl(ipa, mix, preset.vtl_scale)
}

/// Calibrated 5-formant frequencies and bandwidths with an explicit `vtl_scale` override
/// (decoupled from the body's laryngeal `fold_area_scale`).
pub fn vowel_targets_with_vtl(
    ipa: &str,
    body: impl Into<BodyMix>,
    vtl_scale: f64,
) -> ([f64; 5], [f64; 5]) {
    let mix: BodyMix = body.into();
    let mix = BodyMix::new(mix.female, mix.child);
    let preset = BodyPreset::from_mix(mix);
    let vtl_eff = sanitize_clamped(vtl_scale, preset.vtl_scale, 0.45, 1.65);
    let vtl_ratio = preset.vtl_scale / vtl_eff;

    let (m, f, c, height) = lookup_vowel_ref_f123(ipa);
    let blend_i = |i: usize| {
        mix.blend(|b| match b {
            BodyName::Male => m[i],
            BodyName::Female => f[i],
            BodyName::Child => c[i],
        })
    };

    let mut f1 = blend_i(0) * vtl_ratio;
    let mut f2 = blend_i(1) * vtl_ratio;
    let mut f3 = blend_i(2) * vtl_ratio;
    let mut f4 = (3460.0 / preset.vtl_scale) * vtl_ratio;
    let mut f5 = (4460.0 / preset.vtl_scale) * vtl_ratio;

    f1 = clamp(f1, 180.0, 1500.0);
    f2 = clamp(f2.max(f1 + 180.0), 500.0, 4000.0);
    f3 = clamp(f3.max(f2 + 220.0), 1200.0, 5400.0);
    f4 = clamp(f4.max(f3 + 300.0), 2400.0, 6800.0);
    f5 = clamp(f5.max(f4 + 400.0), 3200.0, 8600.0);

    // Closed-glottis physiological wall + radiation + viscous bandwidths (Hawks & Miller 1995; Klatt 1980).
    // Open-phase glottal coupling in ResonantTract adds ~12–25 Hz dynamically during phonation so
    // effective LPC B1 lands around ~75–92 Hz for male and scales cleanly with frequency.
    let b1 = lerp(56.0, 68.0, clamp(height, 0.0, 1.0)) + 0.012 * (f1 - 400.0).max(0.0);
    let b2 = 76.0 + 0.011 * f2;
    let b3 = 108.0 + 0.015 * f3;
    let b4 = 175.0 + 0.014 * f4;
    let b5 = 240.0 + 0.015 * f5;

    ([f1, f2, f3, f4, f5], [b1, b2, b3, b4, b5])
}

/// Complete [`VocalFrame`] for a sustained IPA vowel on any continuous [`BodyMix`].
pub fn vowel_frame(ipa: &str, body: impl Into<BodyMix>) -> VocalFrame {
    let mix: BodyMix = body.into();
    let mix = BodyMix::new(mix.female, mix.child);
    let preset = BodyPreset::from_mix(mix);
    let (formants_hz, bandwidths_hz) = vowel_targets(ipa, mix);
    VocalFrame {
        f0_hz: preset.f0_hz,
        rd: preset.rd,
        vtl_scale: preset.vtl_scale,
        fold_area_scale: preset.fold_area_scale,
        formants_hz,
        bandwidths_hz,
        ..VocalFrame::default()
    }
    .sanitized()
}

/// Interpolate all continuous fields between two sanitized [`VocalFrame`]s (`t` in `0.0..=1.0`).
#[inline]
pub fn lerp_frame(a: &VocalFrame, b: &VocalFrame, t: f64) -> VocalFrame {
    let t = clamp(sanitize(t, 1.0), 0.0, 1.0);
    let mut formants_hz = [0.0; 5];
    let mut bandwidths_hz = [0.0; 5];
    for i in 0..5 {
        formants_hz[i] = lerp(a.formants_hz[i], b.formants_hz[i], t);
        bandwidths_hz[i] = lerp(a.bandwidths_hz[i], b.bandwidths_hz[i], t);
    }
    let switch = t >= 0.5;
    VocalFrame {
        f0_hz: lerp(a.f0_hz, b.f0_hz, t),
        subglottal_pressure_cm_h2o: lerp(a.subglottal_pressure_cm_h2o, b.subglottal_pressure_cm_h2o, t),
        oral_pressure_cm_h2o: lerp(a.oral_pressure_cm_h2o, b.oral_pressure_cm_h2o, t),
        rd: lerp(a.rd, b.rd, t),
        tension: lerp(a.tension, b.tension, t),
        brightness: lerp(a.brightness, b.brightness, t),
        abduction: lerp(a.abduction, b.abduction, t),
        roughness: lerp(a.roughness, b.roughness, t),
        fry: lerp(a.fry, b.fry, t),
        whisper: if switch { b.whisper } else { a.whisper },
        two_mass: if switch { b.two_mass } else { a.two_mass },
        aspiration_gain: lerp(a.aspiration_gain, b.aspiration_gain, t),
        vtl_scale: lerp(a.vtl_scale, b.vtl_scale, t),
        fold_area_scale: lerp(a.fold_area_scale, b.fold_area_scale, t),
        formants_hz,
        bandwidths_hz,
        epilarynx_ring: lerp(a.epilarynx_ring, b.epilarynx_ring, t),
        piriform_depth: lerp(a.piriform_depth, b.piriform_depth, t),
        lateral_mix: lerp(a.lateral_mix, b.lateral_mix, t),
        rhotic_mix: lerp(a.rhotic_mix, b.rhotic_mix, t),
        velum_port_cm2: lerp(a.velum_port_cm2, b.velum_port_cm2, t),
        constriction_place: if switch { b.constriction_place } else { a.constriction_place },
        constriction_area_cm2: lerp(a.constriction_area_cm2, b.constriction_area_cm2, t),
        frication_gain: lerp(a.frication_gain, b.frication_gain, t),
    }
}
