//! Comprehensive acoustic, physiological, numerical, and real-time safety test suite
//! for the clean-sheet `chimera::vocal_box` physical vocal synthesis core.

use chimera::analysis::fft::Fft;
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::vocal_box::{
    BodyMix, BodyPreset, Cheby2Decimator, ConstrictionPlace, FrameInterpolator, GlottalEngine,
    NasalBranch, ResonantTract, VocalBoxCore, VocalFrame, VocalParams, lerp_frame,
    oversample_factor_for, soft_max_c1, vowel_frame, vowel_targets_with_vtl,
};

const VOWEL_CHARS: &str = "iyɨʉɯuɪʏʊeøɘɵɤoəɛœɜɞʌɔæɐaɶɑɒɝɚ";
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAlloc;

thread_local! {
    static TRACK_ALLOCS: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACK_ALLOCS.with(|tr| {
            if tr.get() {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TRACK_ALLOCS.with(|tr| {
            if tr.get() {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAlloc = CountingAlloc;

fn count_allocations<F: FnOnce()>(f: F) -> usize {
    ALLOC_COUNT.with(|c| c.set(0));
    TRACK_ALLOCS.with(|tr| tr.set(true));
    f();
    TRACK_ALLOCS.with(|tr| tr.set(false));
    ALLOC_COUNT.with(|c| c.get())
}

fn assert_send_sync<T: Send + Sync>() {}

fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let s: f64 = x.iter().map(|v| v * v).sum();
    (s / x.len() as f64).sqrt()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn band_energy_db(sig: &[f64], fs: f64, f_lo: f64, f_hi: f64) -> f64 {
    let n = 4096usize;
    assert!(sig.len() >= n);
    let fft = Fft::new(n);
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    let start = sig.len() - n;
    for i in 0..n {
        let w = 0.5 - 0.5 * (2.0 * core::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
        re[i] = sig[start + i] * w;
    }
    fft.transform(&mut re, &mut im);
    let mut p = 0.0;
    for k in 1..n / 2 {
        let f = k as f64 * fs / n as f64;
        if f >= f_lo && f <= f_hi {
            p += re[k] * re[k] + im[k] * im[k];
        }
    }
    10.0 * (p + 1e-30).log10()
}

fn harmonic_amp_db(sig: &[f64], fs: f64, f_harm: f64) -> f64 {
    let n = sig.len();
    let mut re = 0.0;
    let mut im = 0.0;
    for (i, &x) in sig.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * core::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
        let ang = 2.0 * core::f64::consts::PI * f_harm * (i as f64) / fs;
        re += x * w * ang.cos();
        im -= x * w * ang.sin();
    }
    10.0 * (re * re + im * im + 1e-30).log10()
}

#[test]
fn zero_heap_allocations_and_send_sync() {
    assert_send_sync::<VocalBoxCore>();
    assert_send_sync::<GlottalEngine>();
    assert_send_sync::<ResonantTract>();
    assert_send_sync::<NasalBranch>();
    assert_send_sync::<FrameInterpolator>();
    assert_send_sync::<Cheby2Decimator>();

    let mut buf = [0.0f32; 512];
    let rates = [8_000.0, 19_877.0, 37_123.0, 44_100.0, 48_000.0, 81_234.0, 96_000.0];
    let f_a = vowel_frame("i", BodyMix::MALE);
    let mut f_b = vowel_frame("ɑ", BodyMix::FEMALE);
    f_b.two_mass = true;
    f_b.velum_port_cm2 = 0.45;
    f_b.constriction_place = ConstrictionPlace::Alveolar;
    f_b.constriction_area_cm2 = 0.12;
    f_b.frication_gain = 0.35;

    let allocs = count_allocations(|| {
        let mut core = VocalBoxCore::with_body(48_000.0, BodyMix::MALE);
        for &fs in &rates {
            core.set_sample_rate(fs);
            core.set_target_frame(f_a, 0.005);
            core.render(&mut buf[..128]);
            core.set_target_frame(f_b, 0.005);
            core.render(&mut buf[..257]);
            let _ = core.telemetry();
        }
        core.reset();
        for idx in 0..64 {
            let f = if idx < 32 { &f_a } else { &f_b };
            buf[idx] = core.step_frame(f);
        }
    });

    assert_eq!(allocs, 0, "VocalBoxCore allocated {allocs} times on the audio path");
}

#[test]
fn determinism_reset_replay_and_block_size_bit_identity() {
    let f_i = vowel_frame("i", BodyMix::MALE);
    let mut f_u = vowel_frame("u", BodyMix::new(0.4, 0.2));
    f_u.velum_port_cm2 = 0.3;
    f_u.aspiration_gain = 0.15;
    f_u.frication_gain = 0.10;
    f_u.constriction_place = ConstrictionPlace::Postalveolar;

    let total = 4096usize;

    // 1. Reset bit-exact replay
    let mut core = VocalBoxCore::new(48_000.0);
    core.set_frame_immediate(f_i);
    core.set_target_frame(f_u, total as f64 / 48_000.0);
    let mut run1 = vec![0.0f32; total];
    core.render(&mut run1);

    core.reset();
    core.set_frame_immediate(f_i);
    core.set_target_frame(f_u, total as f64 / 48_000.0);
    let mut run2 = vec![0.0f32; total];
    core.render(&mut run2);

    for i in 0..total {
        assert_eq!(
            run1[i].to_bits(),
            run2[i].to_bits(),
            "reset() replay diverged at sample {i}: {} vs {}",
            run1[i],
            run2[i]
        );
    }

    // 2. Arbitrary host block sizes bit-identical
    for &blk in &[1usize, 13, 64, 128, 4096] {
        let mut c = VocalBoxCore::new(48_000.0);
        c.set_frame_immediate(f_i);
        c.set_target_frame(f_u, total as f64 / 48_000.0);
        let mut out = vec![0.0f32; total];
        let mut pos = 0;
        while pos < total {
            let end = (pos + blk).min(total);
            c.render(&mut out[pos..end]);
            pos = end;
        }
        for i in 0..total {
            assert_eq!(
                run1[i].to_bits(),
                out[i].to_bits(),
                "block size {blk} diverged at sample {i}"
            );
        }
    }
}

#[test]
fn zero_control_rate_64_sample_block_buzz() {
    let fs = 48_000.0;
    let blocks = 150usize;
    let blk = 64usize;
    let total = blocks * blk;

    // Compare 64-sample FrameInterpolator (`core_ramped`) vs 64-sample zero-order hold (`core_stair`)
    // against a sample-accurate control reference (`core_exact`) during a pitch + formant glide!
    let mut core_exact = VocalBoxCore::new(fs);
    let mut core_ramped = VocalBoxCore::new(fs);
    let mut core_stair = VocalBoxCore::new(fs);

    let traj = |sample_idx: usize| -> VocalFrame {
        let t = sample_idx as f64 / total as f64;
        let mut f = vowel_frame("ɑ", BodyMix::MALE);
        f.aspiration_gain = 0.0;
        f.f0_hz = 130.0 + 95.0 * t;
        f.formants_hz[0] = 730.0 - 390.0 * t;
        f.formants_hz[1] = 1090.0 + 1120.0 * t;
        f
    };

    let f0_init = traj(0);
    core_exact.set_frame_immediate(f0_init);
    core_ramped.set_frame_immediate(f0_init);
    core_stair.set_frame_immediate(f0_init);
    core_exact.render(&mut [0.0f32; 2048]);
    core_ramped.render(&mut [0.0f32; 2048]);
    core_stair.render(&mut [0.0f32; 2048]);

    let mut err_ramped = vec![0.0f64; total];
    let mut err_stair = vec![0.0f64; total];
    let mut tmp_r = [0.0f32; 64];
    let mut tmp_s = [0.0f32; 64];

    for b in 0..blocks {
        let end_frame = traj((b + 1) * blk);
        let mid_frame = traj(b * blk + blk / 2);

        core_ramped.set_target_frame(end_frame, blk as f64 / fs);
        core_ramped.render(&mut tmp_r);

        core_stair.set_frame_immediate(mid_frame);
        core_stair.render(&mut tmp_s);

        for i in 0..blk {
            let idx = b * blk + i;
            let y_ex = core_exact.step_frame(&traj(idx + 1)) as f64;
            err_ramped[idx] = tmp_r[i] as f64 - y_ex;
            err_stair[idx] = tmp_s[i] as f64 - y_ex;
        }
    }

    let rms_r = rms(&err_ramped);
    let rms_s = rms(&err_stair);
    let suppression_db = 20.0 * (rms_s / (rms_r + 1e-30)).log10();
    assert!(
        suppression_db > 10.0,
        "FrameInterpolator did not suppress 64-sample staircase modulation error: suppression={suppression_db:.1} dB (rms_r={rms_r:.5}, rms_s={rms_s:.5})"
    );
}

#[test]
fn body_mix_decoupled_vtl_vs_fold_area_and_continuous_barycentric_blending() {
    assert!((BodyPreset::FEMALE.vtl_scale - 0.90).abs() < 1e-6);
    assert!((BodyPreset::FEMALE.fold_area_scale - 0.62).abs() < 1e-6);
    assert!(
        (BodyPreset::FEMALE.fold_area_scale - BodyPreset::FEMALE.vtl_scale.powi(2)).abs() > 0.15,
        "fold_area_scale must be physiologically decoupled from vtl_scale^2"
    );

    let m1 = BodyPreset::from_mix(BodyMix::new(0.50, 0.20));
    assert!((m1.vtl_scale - 0.9080).abs() < 1e-4);
    assert!((m1.fold_area_scale - 0.7440).abs() < 1e-4);
    assert!((m1.f0_hz - 179.00).abs() < 1e-2);
    assert!((m1.rd - 1.0480).abs() < 1e-4);

    let m2 = BodyPreset::from_mix(BodyMix::new(0.37, 0.19));
    assert!((m2.vtl_scale - 0.9206).abs() < 1e-4);
    assert!((m2.fold_area_scale - 0.7873).abs() < 1e-4);

    let m3 = BodyPreset::from_mix(BodyMix::new(0.73, 0.41));
    assert!((m3.vtl_scale - 0.8503).abs() < 1e-4);
    assert!((m3.fold_area_scale - 0.6231).abs() < 1e-4);

    // Glottal flow scales linearly with fold_area_scale (Challenger 1 regression guard)
    let measure_peak_flow = |fold_scale: f64| -> f64 {
        let mut g = GlottalEngine::new(96_000.0);
        let mut f = vowel_frame("ɑ", BodyMix::MALE);
        f.fold_area_scale = fold_scale;
        let mut peak_flow = 0.0f64;
        for i in 0..9_600 {
            let st = g.step(&f);
            if i > 4_800 {
                peak_flow = peak_flow.max(st.flow_cm3_s);
            }
        }
        peak_flow
    };

    let flow_child = measure_peak_flow(0.48);
    let flow_male = measure_peak_flow(1.00);
    let ratio = flow_child / flow_male;
    assert!(
        (ratio - 0.48).abs() < 0.03,
        "peak glottal_flow_cm3_s ratio {ratio:.4} did not scale linearly with fold_area_scale=0.48"
    );
}

#[test]
fn sample_rate_and_pitch_sweep_lf_and_two_mass_alias_free_and_zero_dc() {
    assert_eq!(oversample_factor_for(8_000.0), 12);
    assert_eq!(oversample_factor_for(16_000.0), 6);
    assert_eq!(oversample_factor_for(44_100.0), 3);
    assert_eq!(oversample_factor_for(48_000.0), 2);
    assert_eq!(oversample_factor_for(96_000.0), 2);

    let mut dec = Cheby2Decimator::new(16_000.0);
    let m = dec.oversample();
    let fs_int = dec.fs_int();
    let mut dc_out = 0.0;
    for _ in 0..2000 {
        for _ in 0..m {
            dc_out = dec.push(1.0);
        }
    }
    assert!((dc_out - 1.0).abs() < 1e-6, "Cheby2 DC gain drifted: {dc_out}");

    // Stopband sine at 0.55 * fs_out (= 8800 Hz, inside alias band above Nyquist 8000 Hz)
    dec.reset();
    let f_alias = 0.55 * 16_000.0;
    let mut out_alias = Vec::with_capacity(2000);
    let mut n_int = 0usize;
    for k in 0..2000 {
        let mut y = 0.0;
        for _ in 0..m {
            let x = (2.0 * core::f64::consts::PI * f_alias * (n_int as f64) / fs_int).sin();
            n_int += 1;
            y = dec.push(x);
        }
        if k >= 500 {
            out_alias.push(y);
        }
    }
    let atten_db = -20.0 * (rms(&out_alias) * core::f64::consts::SQRT_2 + 1e-30).log10();
    assert!(
        atten_db >= 74.0,
        "Cheby2 stopband attenuation at 0.55*fs_out was only {atten_db:.1} dB (expected >= 74 dB)"
    );

    // Full sample rate x pitch sweep (E1=41.2 Hz .. C6=1046.5 Hz), LF & Two-Mass
    for &fs in &[8_000.0, 16_000.0, 22_050.0, 44_100.0, 48_000.0, 96_000.0] {
        for &two_mass in &[false, true] {
            for &(f0, v_ipa) in &[
                (41.2034, "ɑ"),
                (110.0, "e"),
                (220.0, "i"),
                (587.33, "i"),
                (1046.502, "i"),
                (1046.502, "u"),
            ] {
                if f0 > 0.42 * fs {
                    continue;
                }
                let mut core = VocalBoxCore::new(fs);
                let mut f = vowel_frame(v_ipa, BodyMix::FEMALE);
                f.f0_hz = f0;
                f.two_mass = two_mass;
                core.set_frame_immediate(f);
                let n = (0.50 * fs) as usize;
                let mut buf = vec![0.0f32; n];
                core.render(&mut buf);

                // Measure true DC over exact fractional-period span after 250 ms warmup
                let start = (0.25 * fs) as usize;
                let exact_period = fs / f0;
                let n_periods = (((n - start) as f64) / exact_period).floor().max(1.0);
                let span = ((n_periods * exact_period).round() as usize).min(n - start);
                let dc_slice = &buf[start..start + span];
                let mean: f64 =
                    dc_slice.iter().map(|&x| x as f64).sum::<f64>() / dc_slice.len() as f64;
                let peak = dc_slice.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                assert!(
                    mean.abs() < 1e-3,
                    "DC offset {mean:.5} too high at fs={fs}, f0={f0}, two_mass={two_mass}"
                );
                assert!(
                    peak > 0.01 && peak < 0.92,
                    "peak={peak:.4} out of safe non-clipping range at fs={fs}, f0={f0}, vowel={v_ipa}, two_mass={two_mass}"
                );
            }
        }
    }

    // Two-Mass pitch lock within < 15 cents down to E1 (41.2034 Hz)
    for &f0 in &[41.2034, 82.4069, 110.0, 220.0, 440.0] {
        let mut core = VocalBoxCore::new(48_000.0);
        let mut f = vowel_frame("ɑ", BodyMix::MALE);
        f.f0_hz = f0;
        f.two_mass = true;
        core.set_frame_immediate(f);
        let mut buf = vec![0.0f32; 28_800];
        core.render(&mut buf);
        let meas = core.telemetry().f0_now_hz;
        let cents = 1200.0 * (meas / f0).log2().abs();
        assert!(
            cents < 15.0,
            "Two-Mass pitch drifted {cents:.2} cents at target f0={f0} Hz (measured {meas:.2} Hz)"
        );
    }

    // Live LF <-> Two-Mass toggle smoothness (Challenger 2 regression guard)
    let mut core = VocalBoxCore::new(48_000.0);
    let mut f_lf = vowel_frame("ɑ", BodyMix::MALE);
    f_lf.two_mass = false;
    let mut f_tm = f_lf;
    f_tm.two_mass = true;
    core.set_frame_immediate(f_lf);
    let mut switch_buf = vec![0.0f32; 9_600];
    core.render(&mut switch_buf[..4_800]);
    core.set_target_frame(f_tm, 0.004);
    core.render(&mut switch_buf[4_800..7_200]);
    core.set_target_frame(f_lf, 0.004);
    core.render(&mut switch_buf[7_200..]);
    let max_dy = switch_buf
        .windows(2)
        .skip(4_700)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_dy < 0.25,
        "LF <-> Two-Mass live switch produced discontinuity click max |dy| = {max_dy:.4}"
    );
}

#[test]
fn fant_rd_monotonic_h1_h2_spectral_tilt_tension_and_brightness() {
    let fs = 96_000.0;
    let f0 = 120.0;
    let rds = [0.35, 0.70, 1.15, 1.75, 2.55];
    let mut h1_h2_list = Vec::new();
    let mut h1_h8_list = Vec::new();

    for &rd in &rds {
        let mut g = GlottalEngine::new(fs);
        let mut f = VocalFrame::default();
        f.f0_hz = f0;
        f.rd = rd;
        f.aspiration_gain = 0.0;
        let n = 24_000usize;
        let mut sig = Vec::with_capacity(n);
        for i in 0..n + 8_000 {
            let st = g.step(&f);
            if i >= 8_000 {
                sig.push(st.excitation);
            }
        }
        let h1 = harmonic_amp_db(&sig, fs, f0);
        let h2 = harmonic_amp_db(&sig, fs, 2.0 * f0);
        let h8 = harmonic_amp_db(&sig, fs, 8.0 * f0);
        h1_h2_list.push(h1 - h2);
        h1_h8_list.push(h1 - h8);
    }

    for i in 1..rds.len() {
        assert!(
            h1_h2_list[i] > h1_h2_list[i - 1] + 0.4,
            "H1-H2 not strictly monotonic in Rd: {h1_h2_list:?}"
        );
        assert!(
            h1_h8_list[i] > h1_h8_list[i - 1] + 0.8,
            "H1-H8 not strictly monotonic in Rd: {h1_h8_list:?}"
        );
    }

    // Brightness HF boost > 3.0 dB
    let hf_share = |bright: f64| -> f64 {
        let mut g = GlottalEngine::new(fs);
        let mut f = VocalFrame::default();
        f.f0_hz = 120.0;
        f.rd = 1.05;
        f.brightness = bright;
        f.aspiration_gain = 0.0;
        let mut sig = Vec::with_capacity(8192);
        for i in 0..12_000 {
            let st = g.step(&f);
            if i >= 12_000 - 8192 {
                sig.push(st.excitation);
            }
        }
        band_energy_db(&sig, fs, 2_200.0, 8_000.0) - band_energy_db(&sig, fs, 100.0, 800.0)
    };
    let hf_diff = hf_share(1.0) - hf_share(0.25);
    assert!(
        hf_diff > 3.0,
        "brightness=1.0 only boosted HF share by {hf_diff:.2} dB over brightness=0.25 (expected > 3.0 dB)"
    );

    // Continuous rapid Rd sweep 0.30 <-> 2.80 without mid-cycle click spikes
    let mut core = VocalBoxCore::new(48_000.0);
    let base = vowel_frame("ɑ", BodyMix::MALE);
    core.set_frame_immediate(base);
    let mut buf = vec![0.0f32; 12_000];
    for (i, out_s) in buf.iter_mut().enumerate() {
        let mut f = base;
        let s = (2.0 * core::f64::consts::PI * 4.0 * (i as f64) / 48_000.0).sin();
        f.rd = 1.55 + 1.25 * s;
        *out_s = core.step_frame(&f);
    }
    let max_dy = buf
        .windows(2)
        .skip(1000)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_dy < 0.22,
        "continuous Rd sweep produced mid-cycle discontinuity max |dy| = {max_dy:.4}"
    );
}

#[test]
fn aerodynamic_devoicing_open_phase_aspiration_whisper_and_vocal_fry() {
    // 1. Transglottal dP = P_sub - P_oral monotonic decrease & cutoff at dP <= 0.8 cmH2O
    for &two_mass in &[false, true] {
        let mut prev_rms = f64::INFINITY;
        for &dp in &[8.0, 4.5, 2.2, 1.3, 0.7, 0.0] {
            let mut core = VocalBoxCore::new(48_000.0);
            let mut f = vowel_frame("ɑ", BodyMix::MALE);
            f.two_mass = two_mass;
            f.subglottal_pressure_cm_h2o = 8.0;
            f.oral_pressure_cm_h2o = 8.0 - dp;
            f.aspiration_gain = 0.0;
            core.set_frame_immediate(f);
            let mut buf = vec![0.0f32; 9_600];
            core.render(&mut buf);
            let r = rms(&buf[4_800..].iter().map(|&x| x as f64).collect::<Vec<_>>());
            if dp <= 0.8 {
                assert!(
                    r < 1e-5,
                    "failed aerodynamic devoicing at dP={dp} (two_mass={two_mass}): rms={r}"
                );
            } else {
                assert!(
                    r < prev_rms,
                    "non-monotonic RMS vs dP at dP={dp} (two_mass={two_mass}): {r} >= {prev_rms}"
                );
                prev_rms = r;
            }
        }
    }

    // 2. Distinct open & closed glottal phases + open-phase modulated aspiration
    let mut g = GlottalEngine::new(96_000.0);
    let mut f = VocalFrame::default();
    f.f0_hz = 110.0;
    f.rd = 0.95;
    f.aspiration_gain = 0.45;

    let mut g_clean = GlottalEngine::new(96_000.0);
    let mut f_clean = f;
    f_clean.aspiration_gain = 0.0;

    let mut open_n = 0usize;
    let mut closed_n = 0usize;
    let mut open_asp_sq = 0.0;
    let mut closed_asp_sq = 0.0;

    for i in 0..16_000 {
        let st_a = g.step(&f);
        let st_c = g_clean.step(&f_clean);
        if i >= 4_000 {
            let asp_only = st_a.excitation - st_c.excitation;
            if st_c.area_cm2 > 0.01 {
                open_n += 1;
                open_asp_sq += asp_only * asp_only;
            } else {
                closed_n += 1;
                closed_asp_sq += asp_only * asp_only;
            }
        }
    }
    assert!(
        open_n > 2_000 && closed_n > 2_000,
        "expected distinct open ({open_n}) and closed ({closed_n}) glottal samples"
    );
    let open_asp_rms = (open_asp_sq / open_n as f64).sqrt();
    let closed_asp_rms = (closed_asp_sq / closed_n as f64).sqrt();
    assert!(
        open_asp_rms > 1.8 * closed_asp_rms,
        "aspiration was not modulated by open glottal phase: open={open_asp_rms:.4}, closed={closed_asp_rms:.4}"
    );

    // 3. Whisper mode preserves vowel formant centroids (/i/ > /ɑ/ > /u/)
    let whisper_centroid = |ipa: &str| -> f64 {
        let mut core = VocalBoxCore::new(48_000.0);
        let mut wf = vowel_frame(ipa, BodyMix::MALE);
        wf.whisper = true;
        wf.abduction = 0.35;
        core.set_frame_immediate(wf);
        let mut buf = vec![0.0f32; 12_000];
        core.render(&mut buf);
        let sig: Vec<f64> = buf[3_800..].iter().map(|&x| x as f64).collect();
        let n = 4096usize;
        let fft = Fft::new(n);
        let mut re = sig[..n].to_vec();
        let mut im = vec![0.0; n];
        fft.transform(&mut re, &mut im);
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 4..n / 4 {
            let fr = k as f64 * 48_000.0 / n as f64;
            let p = re[k] * re[k] + im[k] * im[k];
            num += fr * p;
            den += p;
        }
        num / (den + 1e-30)
    };
    let c_i = whisper_centroid("i");
    let c_a = whisper_centroid("ɑ");
    let c_u = whisper_centroid("u");
    assert!(
        c_i > c_a + 200.0 && c_a > c_u + 100.0,
        "whisper formant centroids not ordered (/i/={c_i:.0}, /ɑ/={c_a:.0}, /u/={c_u:.0})"
    );

    // 4. Vocal fry f0/2 subharmonic creak
    let subharm_rel_db = |fry: f64| -> f64 {
        let mut core = VocalBoxCore::new(48_000.0);
        let mut vf = vowel_frame("ɑ", BodyMix::MALE);
        vf.f0_hz = 100.0;
        vf.fry = fry;
        vf.aspiration_gain = 0.0;
        core.set_frame_immediate(vf);
        let mut buf = vec![0.0f32; 16_000];
        core.render(&mut buf);
        let sig: Vec<f64> = buf[4_000..].iter().map(|&x| x as f64).collect();
        harmonic_amp_db(&sig, 48_000.0, 50.0) - harmonic_amp_db(&sig, 48_000.0, 100.0)
    };
    let fry_boost = subharm_rel_db(0.85) - subharm_rel_db(0.0);
    assert!(
        fry_boost > 8.0,
        "vocal fry f0/2 subharmonic boost was only {fry_boost:.2} dB (expected > 8 dB)"
    );
}

#[test]
fn hillenbrand_vowel_lpc_accuracy_schwa_reach_and_bandwidths_lf_and_two_mass() {
    let hillenbrand = ["i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u", "ʌ", "ɝ"];
    let all_vowels: Vec<String> = VOWEL_CHARS.chars().map(|c| c.to_string()).collect();
    assert_eq!(all_vowels.len(), 30);

    let bodies = [
        ("Male", BodyMix::MALE),
        ("Female", BodyMix::FEMALE),
        ("Child", BodyMix::CHILD),
        ("Mix(0.37,0.19)", BodyMix::new(0.37, 0.19)),
        ("Mix(0.73,0.41)", BodyMix::new(0.73, 0.41)),
    ];

    for &(label, body) in &bodies {
        let preset = BodyPreset::from_mix(body);
        let (schwa_f, _) = vowel_targets_with_vtl("ə", body, preset.vtl_scale);

        for &two_mass in &[false, true] {
            for &speaking_pitch in &[false, true] {
                let mut f1_errs = Vec::new();
                let mut f2_errs = Vec::new();
                let mut reaches = Vec::new();
                let mut b1s = Vec::new();
                let mut b2s = Vec::new();
                let mut b3s = Vec::new();

                // VTL-matched LPC rate so 5 formants sit cleanly below Nyquist for order=12 LPC
                let fs_lpc = (10_000.0 / preset.vtl_scale).clamp(10_000.0, 16_000.0);

                for ipa in &all_vowels {
                    let base = vowel_frame(ipa, body);
                    let probe_f0s: &[f64] = if speaking_pitch {
                        &[preset.f0_hz.min(0.42 * base.formants_hz[0]).max(95.0)]
                    } else {
                        &[(96.0 / preset.vtl_scale)
                            .min(0.36 * base.formants_hz[0])
                            .max(88.0)]
                    };

                    let mut m_f1_sum = 0.0;
                    let mut m_f2_sum = 0.0;
                    let mut b1_sum = 0.0;
                    let mut b2_sum = 0.0;
                    let mut b3_sum = 0.0;
                    let mut b3_cnt = 0usize;

                    for &f0_probe in probe_f0s {
                        let mut core = VocalBoxCore::new(fs_lpc);
                        let mut f = base;
                        f.two_mass = two_mass;
                        f.aspiration_gain = 0.0;
                        f.f0_hz = f0_probe;
                        core.set_frame_immediate(f);

                        let n = (0.28 * fs_lpc) as usize;
                        let mut buf = vec![0.0f32; n];
                        core.render(&mut buf);

                        let win_len = (0.060 * fs_lpc) as usize;
                        let slice: Vec<f64> =
                            buf[n - win_len..].iter().map(|&x| x as f64).collect();
                        let fm = estimate_formants(
                            &slice,
                            fs_lpc,
                            FormantOpts {
                                order: Some(12),
                                pre_emph: Some(0.96),
                                max_bw: Some(650.0 / preset.vtl_scale),
                                min_freq: Some(185.0 / preset.vtl_scale),
                            },
                        );
                        assert!(
                            fm.len() >= 2,
                            "LPC failed to find 2 formants for /{ipa}/ ({label}, two_mass={two_mass}): {fm:?}"
                        );
                        m_f1_sum += fm[0].freq;
                        m_f2_sum += fm[1].freq;
                        b1_sum += fm[0].bw;
                        b2_sum += fm[1].bw;
                        if fm.len() >= 3 {
                            b3_sum += fm[2].bw;
                            b3_cnt += 1;
                        }
                    }

                    let k_inv = 1.0 / (probe_f0s.len() as f64);
                    let m_f1 = m_f1_sum * k_inv;
                    let m_f2 = m_f2_sum * k_inv;
                    let e1 = (m_f1 - base.formants_hz[0]).abs() / base.formants_hz[0] * 100.0;
                    let e2 = (m_f2 - base.formants_hz[1]).abs() / base.formants_hz[1] * 100.0;
                    f1_errs.push(e1);
                    f2_errs.push(e2);

                    if hillenbrand.contains(&&ipa[..]) {
                        let target_dist = ((base.formants_hz[0] - schwa_f[0]).powi(2)
                            + (base.formants_hz[1] - schwa_f[1]).powi(2))
                        .sqrt();
                        if target_dist > 220.0 / preset.vtl_scale {
                            let meas_dist =
                                ((m_f1 - schwa_f[0]).powi(2) + (m_f2 - schwa_f[1]).powi(2)).sqrt();
                            reaches.push(meas_dist / target_dist * 100.0);
                        }
                        b1s.push(b1_sum * k_inv);
                        b2s.push(b2_sum * k_inv);
                        if b3_cnt > 0 {
                            b3s.push(b3_sum / (b3_cnt as f64));
                        }
                    }
                }

                let med_f1 = median(f1_errs);
                let med_f2 = median(f2_errs);
                let med_reach = median(reaches);
                let med_b1 = median(b1s);
                let med_b2 = median(b2s);
                let med_b3 = median(b3s);

                let mode_tag = if speaking_pitch { "speak_f0" } else { "dense_f0" };
                println!(
                    "[{label} two_mass={two_mass} {mode_tag}] F1_err={med_f1:.2}%, F2_err={med_f2:.2}%, reach={med_reach:.1}%, B1={med_b1:.1} Hz, B2={med_b2:.1} Hz, B3={med_b3:.1} Hz"
                );

                let (f1_bound, f2_bound) = if speaking_pitch {
                    let df0 = (preset.f0_hz - 110.0).max(0.0) / 100.0;
                    (3.0 + 2.3 * df0, 3.0 + 0.9 * df0)
                } else {
                    (3.0, 3.0)
                };
                assert!(
                    med_f1 < f1_bound && med_f2 < f2_bound,
                    "[{label} two_mass={two_mass} {mode_tag}] median LPC error exceeded ({f1_bound:.2}%, {f2_bound:.2}%): F1={med_f1:.2}%, F2={med_f2:.2}%"
                );
                let reach_lo = if speaking_pitch { 90.0 } else { 95.0 };
                let reach_hi = if speaking_pitch { 110.0 } else { 105.0 };
                assert!(
                    (reach_lo..=reach_hi).contains(&med_reach),
                    "[{label} two_mass={two_mass} {mode_tag}] schwa reach {med_reach:.1}% outside [{reach_lo}%, {reach_hi}%]"
                );
                if body == BodyMix::MALE {
                    if !two_mass {
                        assert!(
                            (65.0..=95.0).contains(&med_b1),
                            "Male LF B1={med_b1:.1} Hz outside [65, 95] Hz"
                        );
                    } else {
                        assert!(
                            (55.0..=120.0).contains(&med_b1),
                            "Male Two-Mass B1={med_b1:.1} Hz outside [55, 120] Hz"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn coupled_tract_aggiustamento_singer_ring_piriform_lateral_rhotic_and_ltas() {
    let fs = 48_000.0;

    // 1. Soprano aggiustamento F1 >= 1.08 * f0 & soft_max_c1 C1 continuity
    assert!((soft_max_c1(459.0, 475.2, 45.0) - 479.808).abs() < 1e-3);
    assert!((soft_max_c1(437.0, 475.2, 45.0) - 475.457).abs() < 1e-3);

    let mut core_sop = VocalBoxCore::new(fs);
    let mut f_sop = vowel_frame("i", BodyMix::FEMALE);
    f_sop.f0_hz = 783.99; // G5
    core_sop.set_frame_immediate(f_sop);
    let mut sop_buf = vec![0.0f32; 12_000];
    core_sop.render(&mut sop_buf);
    let tel = core_sop.telemetry();
    assert!(
        tel.effective_f1_hz >= 1.08 * f_sop.f0_hz,
        "aggiustamento failed: eff_f1={} < 1.08*f0={}",
        tel.effective_f1_hz,
        1.08 * f_sop.f0_hz
    );

    // 2. Open-phase subglottal coupling causally shifts F1 up & widens B1
    let mut tract = ResonantTract::new(96_000.0);
    let f_ref = vowel_frame("ɑ", BodyMix::MALE);
    let _ = tract.step(&f_ref, 0.0, 0.0);
    let f1_closed = tract.last_effective_f1_hz();
    let b1_closed = tract.last_effective_b1_hz();
    let _ = tract.step(&f_ref, 0.0, 0.25);
    let f1_open = tract.last_effective_f1_hz();
    let b1_open = tract.last_effective_b1_hz();
    assert!(
        f1_open > f1_closed + 5.0 && b1_open > b1_closed + 20.0,
        "subglottal open-phase coupling inactive: closed=({f1_closed:.1},{b1_closed:.1}), open=({f1_open:.1},{b1_open:.1})"
    );

    // 3. Epilaryngeal singer's ring (2.85-3.35 kHz boost)
    let render_tail = |f: &VocalFrame| -> Vec<f64> {
        let mut c = VocalBoxCore::new(fs);
        c.set_frame_immediate(*f);
        let mut b = vec![0.0f32; 12_000];
        c.render(&mut b);
        b[4_000..].iter().map(|&x| x as f64).collect()
    };
    let f_modal = vowel_frame("ɑ", BodyMix::MALE);
    let mut f_ring = f_modal;
    f_ring.epilarynx_ring = 1.0;
    let ring_gain = band_energy_db(&render_tail(&f_ring), fs, 2_850.0, 3_350.0)
        - band_energy_db(&render_tail(&f_modal), fs, 2_850.0, 3_350.0);
    assert!(
        ring_gain > 6.0,
        "epilarynx_ring=1.0 only boosted 3 kHz singer's band by {ring_gain:.2} dB"
    );

    // 4. Piriform sinus notch (~4.6 kHz / vtl_scale)
    let mut f_no_piri = f_modal;
    f_no_piri.piriform_depth = 0.0;
    let mut f_deep_piri = f_modal;
    f_deep_piri.piriform_depth = 1.0;
    let piri_drop = band_energy_db(&render_tail(&f_no_piri), fs, 4_350.0, 4_850.0)
        - band_energy_db(&render_tail(&f_deep_piri), fs, 4_350.0, 4_850.0);
    assert!(
        piri_drop > 6.0,
        "piriform_depth=1.0 only deepened 4.6 kHz trough by {piri_drop:.2} dB"
    );

    // 5. Lateral /l/ side-branch notch (~2.55 kHz) & rhotic /ɹ/ F3 drop
    let mut f_lat = vowel_frame("ə", BodyMix::MALE);
    f_lat.lateral_mix = 1.0;
    let lat_drop = band_energy_db(
        &render_tail(&vowel_frame("ə", BodyMix::MALE)),
        fs,
        2_350.0,
        2_750.0,
    ) - band_energy_db(&render_tail(&f_lat), fs, 2_350.0, 2_750.0);
    assert!(
        lat_drop > 5.0,
        "lateral_mix=1.0 did not carve /l/ side-branch notch ({lat_drop:.2} dB)"
    );

    let mut f_rho = vowel_frame("ə", BodyMix::MALE);
    f_rho.rhotic_mix = 1.0;
    let mut c_rho = VocalBoxCore::new(12_000.0);
    c_rho.set_frame_immediate(f_rho);
    let mut b_rho = vec![0.0f32; 4_000];
    c_rho.render(&mut b_rho);
    let fm_rho = estimate_formants(
        &b_rho[1_600..].iter().map(|&x| x as f64).collect::<Vec<_>>(),
        12_000.0,
        FormantOpts {
            order: Some(14),
            ..Default::default()
        },
    );
    assert!(
        fm_rho.len() >= 3 && fm_rho[2].freq < 1_950.0,
        "rhotic_mix=1.0 failed to lower F3 below 1950 Hz: {fm_rho:?}"
    );

    // 6. Multi-vowel LTAS octave bands re 500 Hz vs Byrne et al. (1994)
    let mut core_ltas = VocalBoxCore::new(fs);
    let passage_vowels = ["i", "ɪ", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "ʊ", "u", "ʌ", "ə", "ɝ"];
    let mut passage = Vec::new();
    for (idx, v) in passage_vowels.iter().enumerate() {
        let mut fv = vowel_frame(v, BodyMix::MALE);
        fv.f0_hz = 102.0 + 34.0 * ((idx as f64 * 1.3).sin() * 0.5 + 0.5);
        if idx == 0 {
            core_ltas.set_frame_immediate(fv);
        } else {
            core_ltas.set_target_frame(fv, 0.015);
        }
        let mut seg = vec![0.0f32; 7_200];
        core_ltas.render(&mut seg);
        passage.extend_from_slice(&seg[1_200..]);
    }

    let bands = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
    let byrne = [-2.0, 0.0, -6.0, -11.0, -16.0, -20.0];
    let fft = Fft::new(2048);
    let mut band_e = [0.0f64; 6];
    let mut pos = 0usize;
    while pos + 2048 <= passage.len() {
        let mut re = vec![0.0; 2048];
        let mut im = vec![0.0; 2048];
        for i in 0..2048 {
            let w = 0.5 - 0.5 * (2.0 * core::f64::consts::PI * i as f64 / 2047.0).cos();
            re[i] = passage[pos + i] as f64 * w;
        }
        fft.transform(&mut re, &mut im);
        for k in 1..1024 {
            let fr = k as f64 * fs / 2048.0;
            let p = re[k] * re[k] + im[k] * im[k];
            for (bi, &bc) in bands.iter().enumerate() {
                if fr >= bc / core::f64::consts::SQRT_2 && fr < bc * core::f64::consts::SQRT_2 {
                    band_e[bi] += p;
                }
            }
        }
        pos += 1024;
    }
    let ref_500 = 10.0 * (band_e[1] + 1e-30).log10();
    let ltas_db: Vec<f64> = band_e
        .iter()
        .map(|&e| 10.0 * (e + 1e-30).log10() - ref_500)
        .collect();
    println!("Multi-vowel LTAS re 500 Hz: {ltas_db:.1?} (Byrne norms: {byrne:.1?})");

    for bi in 0..5 {
        let diff = (ltas_db[bi] - byrne[bi]).abs();
        assert!(
            diff <= 4.5,
            "LTAS band {} Hz = {:.2} dB drifted > 4.5 dB from Byrne norm {:.1} dB",
            bands[bi],
            ltas_db[bi],
            byrne[bi]
        );
    }
    assert!(
        (ltas_db[5] - byrne[5]).abs() <= 8.5,
        "LTAS 8 kHz band = {:.2} dB drifted too far from Byrne norm -20.0 dB",
        ltas_db[5]
    );
}

#[test]
fn nasal_branch_zero_leak_murmur_and_place_ordered_antiformants() {
    // 1. Exact zero nasal leakage when velum_port_cm2 == 0.0
    let mut nasal = NasalBranch::new(96_000.0);
    let f_oral = vowel_frame("ɑ", BodyMix::MALE);
    for i in 0..4_000 {
        let exc = (2.0 * core::f64::consts::PI * 110.0 * i as f64 / 96_000.0).sin();
        let out = nasal.step(&f_oral, exc, exc);
        assert_eq!(out.nasal_out.to_bits(), 0.0f64.to_bits());
        assert!((out.combined - exc).abs() < 1e-15);
    }

    // 2. Smooth click-free closing 0.65 -> 0.0 snapping cleanly to 0.0
    let mut core = VocalBoxCore::new(48_000.0);
    let mut f_open = f_oral;
    f_open.velum_port_cm2 = 0.65;
    core.set_frame_immediate(f_open);
    let mut open_buf = vec![0.0f32; 4_800];
    let mut close_buf = vec![0.0f32; 9_600];
    core.render(&mut open_buf);
    core.set_target_frame(f_oral, 0.008);
    core.render(&mut close_buf);
    let max_dy = close_buf
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_dy < 0.16,
        "velum closing produced discontinuity spike max |dy| = {max_dy:.4}"
    );
    assert_eq!(
        core.telemetry().nasal_out,
        0.0,
        "nasal_out did not snap to exact 0.0 after velum closure"
    );

    // 3. Place-ordered oral-shunt anti-formant notches (/m/ < /n/ < /ɲ/ < /ŋ/)
    // Measure shunt notch relative to an un-shunted reference (ConstrictionPlace::Glottal -> no oral cul-de-sac)
    let measure_shunt_notch_hz = |place: ConstrictionPlace| -> f64 {
        let fs = 48_000.0;
        let run_ir = |pl: ConstrictionPlace| -> (Vec<f64>, Vec<f64>) {
            let mut nb = NasalBranch::new(fs);
            let mut f = vowel_frame("ə", BodyMix::MALE);
            f.velum_port_cm2 = 0.70;
            f.constriction_place = pl;
            f.constriction_area_cm2 = 0.0;
            for _ in 0..2_000 {
                let _ = nb.step(&f, 0.0, 0.0);
            }
            let n = 4096usize;
            let mut re = vec![0.0f64; n];
            let mut im = vec![0.0f64; n];
            for i in 0..n {
                let x = if i == 0 { 1.0 } else { 0.0 };
                re[i] = nb.step(&f, x, 0.0).nasal_out;
            }
            Fft::new(n).transform(&mut re, &mut im);
            (re, im)
        };

        let (re_p, im_p) = run_ir(place);
        let (re_ref, im_ref) = run_ir(ConstrictionPlace::Glottal);
        let n = 4096usize;
        let mut min_ratio = f64::INFINITY;
        let mut min_hz = 0.0;
        for k in 35..340 {
            let hz = k as f64 * fs / n as f64;
            let mp = re_p[k] * re_p[k] + im_p[k] * im_p[k];
            let mr = re_ref[k] * re_ref[k] + im_ref[k] * im_ref[k] + 1e-24;
            let ratio = mp / mr;
            if ratio < min_ratio {
                min_ratio = ratio;
                min_hz = hz;
            }
        }
        min_hz
    };

    let z_m = measure_shunt_notch_hz(ConstrictionPlace::Bilabial);
    let z_n = measure_shunt_notch_hz(ConstrictionPlace::Alveolar);
    let z_ny = measure_shunt_notch_hz(ConstrictionPlace::Palatal);
    let z_ng = measure_shunt_notch_hz(ConstrictionPlace::Velar);
    println!("Nasal anti-formant notches: /m/={z_m:.0} Hz, /n/={z_n:.0} Hz, /ɲ/={z_ny:.0} Hz, /ŋ/={z_ng:.0} Hz");
    assert!(
        z_m < z_n - 350.0 && z_n < z_ny - 350.0 && z_ny < z_ng - 350.0,
        "nasal anti-formant notches not strictly place-ordered: /m/={z_m:.0}, /n/={z_n:.0}, /ɲ/={z_ny:.0}, /ŋ/={z_ng:.0}"
    );
}

#[test]
fn pathological_nan_inf_and_out_of_range_sanitization() {
    // 1. Standalone soft_max_c1 NaN / Inf safety (Challenger 2 1F)
    for &bad in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-315, 1e300, -1e300] {
        let v1 = soft_max_c1(bad, 500.0, 45.0);
        let v2 = soft_max_c1(500.0, bad, 45.0);
        let v3 = soft_max_c1(500.0, 520.0, bad);
        assert!(v1.is_finite() && !v1.is_subnormal());
        assert!(v2.is_finite() && !v2.is_subnormal());
        assert!(v3.is_finite() && !v3.is_subnormal());
    }

    // 2. Poisoned VocalFrame with NaN, ±Inf, subnormals, and extreme out-of-range values
    let poison = VocalFrame {
        f0_hz: f64::NAN,
        subglottal_pressure_cm_h2o: -1e300,
        oral_pressure_cm_h2o: f64::NAN,
        rd: f64::INFINITY,
        tension: f64::NEG_INFINITY,
        brightness: 1e300,
        abduction: f64::INFINITY,
        roughness: f64::NAN,
        fry: f64::NEG_INFINITY,
        whisper: false,
        two_mass: true,
        aspiration_gain: f64::NAN,
        vtl_scale: 0.0,
        fold_area_scale: -50.0,
        formants_hz: [f64::NAN, f64::INFINITY, -9999.0, 1e9, f64::NAN],
        bandwidths_hz: [-10.0, 0.0, f64::NAN, f64::INFINITY, 1e-320],
        epilarynx_ring: f64::NAN,
        piriform_depth: f64::INFINITY,
        lateral_mix: -99.0,
        rhotic_mix: 99.0,
        velum_port_cm2: f64::NAN,
        constriction_place: ConstrictionPlace::Alveolar,
        constriction_area_cm2: f64::NEG_INFINITY,
        frication_gain: f64::INFINITY,
    };

    // Standalone submodules directly (Challenger 2 1F)
    let mut g = GlottalEngine::new(f64::NAN);
    let mut tr = ResonantTract::new(f64::INFINITY);
    let mut nb = NasalBranch::new(-100.0);
    let mut dec = Cheby2Decimator::new(f64::NAN);
    let mut interp = FrameInterpolator::new(poison);
    interp.set_target(poison, 0);

    for _ in 0..512 {
        let f_step = interp.step();
        let gs = g.step(&poison);
        assert!(gs.excitation.is_finite() && !gs.excitation.is_subnormal());
        assert!(gs.flow_cm3_s.is_finite() && !gs.flow_cm3_s.is_subnormal());

        let ty = tr.step(&poison, f64::NAN, f64::INFINITY);
        assert!(ty.is_finite() && !ty.is_subnormal() && ty.abs() <= 4.0);

        let ny = nb.step(&poison, f64::NAN, f64::INFINITY);
        assert!(ny.nasal_out.is_finite() && !ny.nasal_out.is_subnormal());
        assert!(ny.oral_out.is_finite() && !ny.oral_out.is_subnormal());
        assert!(ny.combined.is_finite() && !ny.combined.is_subnormal());

        let dy = dec.push(f64::NAN);
        let dy2 = dec.push(f64::INFINITY);
        assert!(dy.is_finite() && !dy.is_subnormal());
        assert!(dy2.is_finite() && !dy2.is_subnormal());
        let _ = lerp_frame(&f_step, &poison, f64::NAN);
    }

    // Full VocalBoxCore with poisoned rate + poisoned frame + subsequent recovery to normal speech
    let mut core = VocalBoxCore::new(f64::NAN);
    core.set_sample_rate(f64::INFINITY);
    core.set_frame_immediate(poison);
    let mut out = vec![0.0f32; 2048];
    core.render(&mut out);
    for &y in &out {
        assert!(
            y.is_finite() && !y.is_subnormal() && y.abs() <= 2.0,
            "invalid sample from poisoned VocalBoxCore: {y}"
        );
    }

    // Must recover immediately to clean finite speech without needing reset()
    let clean = vowel_frame("ɑ", BodyMix::MALE);
    core.set_frame_immediate(clean);
    core.render(&mut out);
    let clean_rms = rms(&out[1024..].iter().map(|&x| x as f64).collect::<Vec<_>>());
    assert!(
        clean_rms > 0.01 && clean_rms < 1.0,
        "VocalBoxCore failed to self-heal after NaN/Inf poisoning: rms={clean_rms}"
    );
    let _ = VocalParams::new(poison).with_vtl_scale(f64::NAN).build();
}
