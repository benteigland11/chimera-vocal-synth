//! Unit & integration tests for the simplified 4-organ physical Vocal Box building blocks:
//! - [`zipper_area_c1`] & [`ribbon_pulse_c1`] $C^1$ value & derivative continuity (zero slope-corner buzz),
//! - [`BodyAnatomy`] decoupled vocal-tract vs laryngeal sexual dimorphism,
//! - [`register_state`] 5-octave CT/TA pitch registration (`E1` 41 Hz to `C6` 1046 Hz),
//! - [`GlottalValve`] (`RibbonGlottis` & `TwoMassGlottis`) multi-rate stability (`16–96 kHz`),
//! - `Send + Sync`, deterministic `reset()`, and `< 15 ms` post-consonant VOT entrainment.

use chimera::anatomy::human::BodyMix;
use chimera::physics::CM_H2O;
use chimera::vocal_box::{
    BodyAnatomy, EmotionPreset, GlottalValve, VocalBoxVoice, register_state, ribbon_pulse_c1,
    zipper_area_c1,
};
use chimera::voice::human_voice::SourcePatch;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn vocal_box_types_are_send_and_sync() {
    assert_send_sync::<BodyAnatomy>();
    assert_send_sync::<GlottalValve>();
    assert_send_sync::<VocalBoxVoice>();
}

#[test]
fn zipper_and_ribbon_pulses_are_c1_continuous_without_slope_corners() {
    // 1. Parabolic mucosal-zipper contact law: exact C^1 across [-delta, +delta]
    let d = 0.004;
    assert_eq!(zipper_area_c1(-0.01, d), 0.0);
    assert_eq!(zipper_area_c1(-d, d), 0.0);
    assert!((zipper_area_c1(0.0, d) - 0.25 * d).abs() < 1e-12);
    assert!((zipper_area_c1(d, d) - d).abs() < 1e-12);
    assert!((zipper_area_c1(0.02, d) - 0.02).abs() < 1e-12);

    let h = 1e-6;
    for i in -80..=80 {
        let a = (i as f64) * 0.0001;
        let d_left = (zipper_area_c1(a, d) - zipper_area_c1(a - h, d)) / h;
        let d_right = (zipper_area_c1(a + h, d) - zipper_area_c1(a, d)) / h;
        assert!(
            (d_right - d_left).abs() < 0.06,
            "slope discontinuity at a={a}: left={d_left}, right={d_right}"
        );
        assert!((-1e-9..=1.0 + 1e-6).contains(&d_right));
    }

    // 2. Progressive-return RibbonGlottis pulse: C^1 across opening, peak, zipper return, and closure
    for sharp in [0.0, 0.35, 0.70, 1.0] {
        let oq = 0.62;
        let skew = 0.64;
        assert_eq!(ribbon_pulse_c1(0.0, oq, skew, sharp), 0.0);
        assert_eq!(ribbon_pulse_c1(oq, oq, skew, sharp), 0.0);
        assert_eq!(ribbon_pulse_c1(0.85, oq, skew, sharp), 0.0);

        // Derivative vanishes smoothly as ph -> oq^- (progressive return phase)
        let eps = 1e-7;
        let end_slope = (ribbon_pulse_c1(oq, oq, skew, sharp)
            - ribbon_pulse_c1(oq - eps, oq, skew, sharp))
            / eps;
        assert!(
            end_slope.abs() < 1e-4,
            "expected smooth zero-slope return at closure for sharp={sharp}, got slope={end_slope}"
        );
    }
}

#[test]
fn decoupled_body_anatomy_and_five_octave_registration_scale_monotonically() {
    let male = BodyAnatomy::from_mix(BodyMix::MALE);
    let female = BodyAnatomy::from_mix(BodyMix::FEMALE);
    let child = BodyAnatomy::from_mix(BodyMix::CHILD);

    // Laryngeal fold cross-section shrinks significantly faster than vocal-tract length
    assert!(male.vtl_scale > female.vtl_scale && female.vtl_scale > child.vtl_scale);
    assert!(
        female.fold_area_scale < female.vtl_scale * female.vtl_scale,
        "female fold_area_scale ({}) must reflect laryngeal sexual dimorphism (< VTL^2 = {})",
        female.fold_area_scale,
        female.vtl_scale * female.vtl_scale
    );
    assert!(male.fold_area_scale > female.fold_area_scale);
    assert!(female.fold_area_scale > child.fold_area_scale);
    assert!(male.speaking_f0_hz < female.speaking_f0_hz);
    assert!(female.speaking_f0_hz < child.speaking_f0_hz);

    // Registration transitions smoothly from chest (E2 = 82 Hz) to high soprano (C6 = 1046 Hz)
    let chest = register_state(95.0, male.speaking_f0_hz, 0.5);
    let mix_reg = register_state(260.0, male.speaking_f0_hz, 0.5);
    let head = register_state(660.0, male.speaking_f0_hz, 0.5);
    assert_eq!(chest.head_mix, 0.0);
    assert!(mix_reg.head_mix > 0.15 && mix_reg.head_mix < 0.85);
    assert!(head.head_mix > 0.95);
    assert!(head.excursion_scale < chest.excursion_scale);
}

#[test]
fn five_octave_multi_rate_stability_reset_identity_and_fast_vot_recovery() {
    chimera::init();

    // 1. 5-octave stability from E1 (41.2 Hz) to C6 (1046.5 Hz) across 16 kHz, 44.1 kHz, 48 kHz, 96 kHz
    for &fs in &[16_000.0, 44_100.0, 48_000.0, 96_000.0] {
        for &(body, f0) in &[
            (BodyMix::MALE, 41.2),
            (BodyMix::MALE, 110.0),
            (BodyMix::FEMALE, 220.0),
            (BodyMix::FEMALE, 523.25),
            (BodyMix::CHILD, 1046.5),
        ] {
            for two_mass in [false, true] {
                let mut v = VocalBoxVoice::new(fs, body);
                v.set_two_mass(two_mass);
                v.set_source(&SourcePatch::f0(f0));
                v.set_gate(1.0);
                let mut buf = vec![0.0f32; (0.18 * fs) as usize];
                v.render(&mut buf);
                assert!(
                    buf.iter().all(|x| x.is_finite() && x.abs() < 1.0),
                    "non-finite or clipped output at fs={fs}, f0={f0}, two_mass={two_mass}"
                );
                let tail = &buf[(0.08 * fs) as usize..];
                let rms = (tail.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
                    / tail.len() as f64)
                    .sqrt();
                assert!(
                    rms > 0.005,
                    "silent output at fs={fs}, f0={f0}, two_mass={two_mass}: rms={rms}"
                );
            }
        }
    }

    // 2. Deterministic reset() bit-identity
    let mut v = VocalBoxVoice::new(48_000.0, BodyMix::FEMALE);
    v.set_emotion(EmotionPreset::Dramatic);
    v.set_two_mass(true);
    v.set_gate(1.0);
    let mut run_a = vec![0.0f32; 4800];
    v.render(&mut run_a);

    v.reset();
    v.set_gate(1.0);
    let mut run_b = vec![0.0f32; 4800];
    v.render(&mut run_b);
    assert_eq!(run_a, run_b, "VocalBoxVoice::reset() must be bit-identical");

    // 3. Fast (< 15 ms) TwoMassGlottis post-consonant VOT re-onset after transglottal pressure collapse
    let fs_int = 96_000.0;
    let mut valve = GlottalValve::new(fs_int);
    valve.set_two_mass(true);
    {
        let (g1, g2) = valve.params_mut();
        for g in [g1, g2] {
            g.f0 = 130.0;
            g.amplitude = 0.16;
            g.effort = 0.55;
        }
    }
    // Initial vowel phonation (8 cmH2O)
    for _ in 0..(0.08 * fs_int) as usize {
        valve.step(8.0 * CM_H2O);
    }
    assert!(valve.voicing() > 0.85);

    // Oral stop + abduction (0 cmH2O transglottal pressure, abducted folds for 60 ms)
    {
        let (g1, g2) = valve.params_mut();
        for g in [g1, g2] {
            g.abduction = 0.24;
        }
    }
    for _ in 0..(0.06 * fs_int) as usize {
        valve.step(0.0);
    }
    assert!(
        valve.voicing() < 0.08,
        "expected clean stop devoicing, got voicing={}",
        valve.voicing()
    );

    // Stop release & re-adduction: must recover strong oscillation within 15 ms!
    {
        let (g1, g2) = valve.params_mut();
        for g in [g1, g2] {
            g.abduction = 0.0;
        }
    }
    let mut peak_area_15ms = 0.0f64;
    for _ in 0..(0.015 * fs_int) as usize {
        peak_area_15ms = peak_area_15ms.max(valve.step(8.0 * CM_H2O));
    }
    assert!(
        peak_area_15ms > 0.06 && valve.voicing() > 0.50,
        "expected <15 ms post-consonant VOT entrainment, got peak_area={peak_area_15ms}, voicing={}",
        valve.voicing()
    );
}
