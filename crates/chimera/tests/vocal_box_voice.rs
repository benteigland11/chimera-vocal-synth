//! Integration, phonetic aerodynamics, 13-emotion differentiation, singing intonation,
//! block-size invariance, and zero-allocation audio-thread tests for [`VocalBoxVoice`]
//! and [`TwoMassGlottis`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAlloc;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

use chimera::analysis::fft::{Fft, hann};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::BodyMix;
use chimera::physics::CM_H2O;
use chimera::phonetics::{Segment, Syllable};
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance, vowel_table};
use chimera::phonetics::sing::{SongOptions, SungNote, midi_to_hz, plan_song};
use chimera::sources::two_mass::TwoMassGlottis;
use chimera::vocal_box::{EmotionPreset, VocalBoxVoice};
use chimera::voice::human_voice::{SourcePatch, VoiceTelemetry};

const FS: f64 = 48_000.0;

fn rms(xs: &[f32]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let s: f64 = xs.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (s / xs.len() as f64).sqrt()
}

fn spectral_centroid_hz(xs: &[f32], fs: f64) -> f64 {
    let n = 1024usize;
    if xs.len() < n {
        return 0.0;
    }
    let fft = Fft::new(n);
    let win = hann(n);
    let mut num = 0.0;
    let mut den = 0.0;
    let mut off = 0;
    while off + n <= xs.len() {
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        for i in 0..n {
            re[i] = xs[off + i] as f64 * win[i];
        }
        fft.transform(&mut re, &mut im);
        for k in 1..n / 2 {
            let f = k as f64 * fs / n as f64;
            let p = re[k] * re[k] + im[k] * im[k];
            num += f * p;
            den += p;
        }
        off += n / 2;
    }
    if den > 1e-18 { num / den } else { 0.0 }
}

fn band_energy_ratio_db(xs: &[f32], fs: f64, lo_hz: f64, hi_hz: f64) -> f64 {
    let n = 2048usize;
    let fft = Fft::new(n);
    let win = hann(n);
    let mut band = 0.0;
    let mut total = 0.0;
    let mut off = 0;
    while off + n <= xs.len() {
        let mut re = vec![0.0; n];
        let mut im = vec![0.0; n];
        for i in 0..n {
            re[i] = xs[off + i] as f64 * win[i];
        }
        fft.transform(&mut re, &mut im);
        for k in 1..n / 2 {
            let f = k as f64 * fs / n as f64;
            let p = re[k] * re[k] + im[k] * im[k];
            if (80.0..=8000.0).contains(&f) {
                total += p;
            }
            if (lo_hz..=hi_hz).contains(&f) {
                band += p;
            }
        }
        off += n / 2;
    }
    10.0 * (band.max(1e-24) / total.max(1e-24)).log10()
}

#[test]
fn audio_path_and_telemetry_do_not_allocate() {
    chimera::init();
    let segs = tokenize_ipa("ˈmɑpɑ ˈsɑʃɑ");
    let plan = plan_utterance(
        &segs,
        &PlanOptions {
            body: BodyMix::new(0.4, 0.2),
            t0: 0.05,
            f0: Some(150.0),
            ..Default::default()
        },
    );

    let mut voice = VocalBoxVoice::new(FS, BodyMix::new(0.4, 0.2));
    voice.schedule(&plan.gestures);
    let mut tele = VoiceTelemetry::default();
    let mut block = [0.0f32; 128];

    // Warm up VoiceTelemetry's internal Vec capacities once before entering the RT section
    voice.render(&mut block);
    voice.telemetry(&mut tele);

    let mut tm = TwoMassGlottis::new(96_000.0);
    let _ = tm.step(8.0 * CM_H2O);

    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    for step in 0..40 {
        if step == 10 {
            voice.set_emotion(EmotionPreset::Operatic);
        } else if step == 25 {
            voice.set_emotion(EmotionPreset::Whisper);
        }
        voice.render(&mut block);
        voice.telemetry(&mut tele);
        for _ in 0..64 {
            let _ = tm.step(8.0 * CM_H2O);
        }
    }
    let after = ALLOC_COUNT.load(Ordering::SeqCst);
    assert_eq!(
        after - before,
        0,
        "expected 0 heap allocations on audio & telemetry path, got {}",
        after - before
    );
}

#[test]
fn all_thirteen_emotion_presets_parse_and_differentiate_acoustically() {
    chimera::init();
    assert_eq!(EmotionPreset::ALL.len(), 13);
    for p in EmotionPreset::ALL {
        assert_eq!(EmotionPreset::from_name(p.name()), Ok(p));
        assert_eq!(EmotionPreset::from_name(&p.name().to_uppercase()), Ok(p));
    }
    // Common aliases
    assert_eq!(EmotionPreset::from_name("neutral"), Ok(EmotionPreset::Natural));
    assert_eq!(EmotionPreset::from_name("soft"), Ok(EmotionPreset::Tender));
    assert_eq!(EmotionPreset::from_name("happy"), Ok(EmotionPreset::Joyful));
    assert_eq!(EmotionPreset::from_name("sad"), Ok(EmotionPreset::Melancholic));
    assert_eq!(EmotionPreset::from_name("fierce"), Ok(EmotionPreset::Angry));
    assert_eq!(EmotionPreset::from_name("tense"), Ok(EmotionPreset::Pressed));
    assert_eq!(EmotionPreset::from_name("airy"), Ok(EmotionPreset::Breathy));
    assert_eq!(EmotionPreset::from_name("whispered"), Ok(EmotionPreset::Whisper));
    assert_eq!(EmotionPreset::from_name("creaky"), Ok(EmotionPreset::Fry));
    assert_eq!(EmotionPreset::from_name("classical"), Ok(EmotionPreset::Operatic));
    assert_eq!(EmotionPreset::from_name("belt"), Ok(EmotionPreset::Pop));
    assert!(EmotionPreset::from_name("nonexistent_preset").is_err());

    let segs = tokenize_ipa("ˈɑlɑmɑ");
    struct EmStat {
        rms: f64,
        mid_harm_db: f64,
        ring_db: f64,
        mean_f0: f64,
        max_voicing: f64,
    }
    let mut stats = std::collections::HashMap::new();

    for preset in EmotionPreset::ALL {
        let plan = plan_utterance(
            &segs,
            &PlanOptions {
                body: BodyMix::MALE,
                t0: 0.04,
                f0: Some(125.0),
                rate: Some(preset.profile().rate_scale),
                ..Default::default()
            },
        );
        let mut v = VocalBoxVoice::new(FS, BodyMix::MALE);
        v.set_source(&SourcePatch::f0(125.0));
        v.set_emotion(preset);
        v.schedule(&plan.gestures);

        let n = ((plan.end + 0.20) * FS).round() as usize;
        let mut out = vec![0.0f32; n];
        let mut f0_acc = 0.0;
        let mut f0_cnt = 0usize;
        let mut max_voic = 0.0f64;
        for chunk in out.chunks_mut(128) {
            v.render(chunk);
            let tel = v.instant_telemetry();
            max_voic = max_voic.max(tel.voicing);
            if tel.voicing > 0.5 {
                f0_acc += tel.f0_now_hz;
                f0_cnt += 1;
            }
        }

        assert!(
            out.iter().all(|x| x.is_finite() && x.abs() <= 0.999),
            "{preset:?} produced non-finite or clipped samples"
        );
        let r = rms(&out);
        assert!(r > 0.003, "{preset:?} too quiet: rms={r}");

        let s0 = (0.12 * FS) as usize;
        let s1 = ((plan.end - 0.04) * FS) as usize;
        let mid = &out[s0..s1];
        let f0_band = band_energy_ratio_db(mid, FS, 90.0, 210.0);
        let f1f2_band = band_energy_ratio_db(mid, FS, 600.0, 2000.0);
        stats.insert(
            preset,
            EmStat {
                rms: rms(mid),
                mid_harm_db: f1f2_band - f0_band,
                ring_db: band_energy_ratio_db(mid, FS, 2650.0, 3450.0),
                mean_f0: if f0_cnt > 0 {
                    f0_acc / f0_cnt as f64
                } else {
                    0.0
                },
                max_voicing: max_voic,
            },
        );
    }

    // Physical emotion contrast checks:
    assert!(
        stats[&EmotionPreset::Angry].rms > stats[&EmotionPreset::Tender].rms * 1.25,
        "Angry ({:.4}) should have higher RMS than Tender ({:.4})",
        stats[&EmotionPreset::Angry].rms,
        stats[&EmotionPreset::Tender].rms
    );
    assert!(
        stats[&EmotionPreset::Pressed].mid_harm_db > stats[&EmotionPreset::Breathy].mid_harm_db + 4.0,
        "Pressed F1/F2-to-H1 harmonic tilt ({:.1} dB) should exceed Breathy ({:.1} dB) by >4 dB",
        stats[&EmotionPreset::Pressed].mid_harm_db,
        stats[&EmotionPreset::Breathy].mid_harm_db
    );
    assert!(
        stats[&EmotionPreset::Whisper].max_voicing < 0.01,
        "Whisper must have zero periodic glottal voicing, got {}",
        stats[&EmotionPreset::Whisper].max_voicing
    );
    assert!(
        stats[&EmotionPreset::Joyful].mean_f0 > stats[&EmotionPreset::Melancholic].mean_f0 * 1.18,
        "Joyful f0 ({:.1} Hz) should exceed Melancholic ({:.1} Hz)",
        stats[&EmotionPreset::Joyful].mean_f0,
        stats[&EmotionPreset::Melancholic].mean_f0
    );
    assert!(
        stats[&EmotionPreset::Fry].mean_f0 < stats[&EmotionPreset::Natural].mean_f0 * 0.72,
        "Fry f0 ({:.1} Hz) should drop into pulse register below Natural ({:.1} Hz)",
        stats[&EmotionPreset::Fry].mean_f0,
        stats[&EmotionPreset::Natural].mean_f0
    );
    assert!(
        stats[&EmotionPreset::Operatic].ring_db > stats[&EmotionPreset::Natural].ring_db + 3.5,
        "Operatic singer's ring ({:.1} dB) should exceed Natural ({:.1} dB) by >3.5 dB",
        stats[&EmotionPreset::Operatic].ring_db,
        stats[&EmotionPreset::Natural].ring_db
    );
}

#[test]
fn oral_nasal_isolation_stop_devoicing_and_sibilant_ordering() {
    chimera::init();

    // 1. Zero oral-vowel nasal leak on "ˈɑpɑ" + clean voiceless stop closure devoicing
    let plan_apa = plan_utterance(
        &tokenize_ipa("ˈɑpɑ"),
        &PlanOptions {
            body: BodyMix::MALE,
            t0: 0.05,
            f0: Some(115.0),
            ..Default::default()
        },
    );
    let mut v = VocalBoxVoice::new(FS, BodyMix::MALE);
    v.schedule(&plan_apa.gestures);
    let n_apa = ((plan_apa.end + 0.20) * FS).round() as usize;
    let mut out_apa = vec![0.0f32; n_apa];
    let mut max_oral_velum = 0.0f64;
    let mut max_oral_nasal_flow = 0.0f64;
    for chunk in out_apa.chunks_mut(64) {
        v.render(chunk);
        let tel = v.instant_telemetry();
        max_oral_velum = max_oral_velum.max(tel.velum_port_cm2);
        max_oral_nasal_flow = max_oral_nasal_flow.max(tel.nasal_flow_cm3_s.abs());
    }
    assert_eq!(
        max_oral_velum, 0.0,
        "expected exact 0.0 cm^2 velum port on oral utterance ˈɑpɑ"
    );
    assert_eq!(
        max_oral_nasal_flow, 0.0,
        "expected exact 0.0 cm^3/s nostril flow on oral utterance ˈɑpɑ"
    );

    let v1_seg = &plan_apa.segments[0];
    let p_seg = &plan_apa.segments[1];
    let v2_seg = &plan_apa.segments[2];
    fn slice_at(buf: &[f32], a: f64, b: f64) -> &[f32] {
        let ia = (a * FS).round() as usize;
        let ib = ((b * FS).round() as usize).min(buf.len());
        &buf[ia..ib]
    }
    let v1_rms = rms(slice_at(&out_apa, v1_seg.t0 + 0.03, v1_seg.t1 - 0.02));
    let p_mid = 0.5 * (p_seg.t0 + p_seg.t1);
    let p_rms = rms(slice_at(&out_apa, p_mid - 0.010, p_mid + 0.015));
    let v2_rms = rms(slice_at(&out_apa, v2_seg.t0 + 0.04, v2_seg.t1 - 0.02));
    assert!(
        v1_rms > 0.03 && v2_rms > 0.03 && p_rms < v1_rms * 0.15,
        "expected deep voiceless stop closure attenuation in ˈɑpɑ: v1={v1_rms:.4}, p={p_rms:.4}, v2={v2_rms:.4}"
    );

    // 2. Nasal murmur active on "ˈmɑnɑ"
    let plan_mana = plan_utterance(
        &tokenize_ipa("ˈmɑnɑ"),
        &PlanOptions {
            body: BodyMix::MALE,
            t0: 0.05,
            f0: Some(115.0),
            ..Default::default()
        },
    );
    let mut vm = VocalBoxVoice::new(FS, BodyMix::MALE);
    vm.schedule(&plan_mana.gestures);
    let mut out_mana = vec![0.0f32; ((plan_mana.end + 0.2) * FS).round() as usize];
    let mut max_nasal_velum = 0.0f64;
    let mut max_nasal_flow = 0.0f64;
    for chunk in out_mana.chunks_mut(64) {
        vm.render(chunk);
        let tel = vm.instant_telemetry();
        max_nasal_velum = max_nasal_velum.max(tel.velum_port_cm2);
        max_nasal_flow = max_nasal_flow.max(tel.nasal_flow_cm3_s.abs());
    }
    assert!(
        max_nasal_velum > 0.30 && max_nasal_flow > 10.0,
        "expected strong nasal coupling on ˈmɑnɑ: velum={max_nasal_velum}, flow={max_nasal_flow} cm^3/s"
    );

    // 3. Place-ordered sibilant frication: alveolar /s/ centroid strictly above postalveolar /ʃ/
    let sibilant_centroid = |ipa_utt: &str| -> f64 {
        let pl = plan_utterance(
            &tokenize_ipa(ipa_utt),
            &PlanOptions {
                body: BodyMix::MALE,
                t0: 0.04,
                f0: Some(115.0),
                ..Default::default()
            },
        );
        let mut vs = VocalBoxVoice::new(FS, BodyMix::MALE);
        vs.schedule(&pl.gestures);
        let mut buf = vec![0.0f32; ((pl.end + 0.15) * FS).round() as usize];
        vs.render(&mut buf);
        let c_seg = &pl.segments[1];
        let mid = 0.5 * (c_seg.t0 + c_seg.t1);
        spectral_centroid_hz(slice_at(&buf, mid - 0.025, mid + 0.025), FS)
    };
    let c_s = sibilant_centroid("ˈɑsɑ");
    let c_sh = sibilant_centroid("ˈɑʃɑ");
    assert!(
        c_s > c_sh + 900.0 && c_s > 5200.0,
        "expected /s/ centroid ({c_s:.0} Hz) >> /ʃ/ centroid ({c_sh:.0} Hz)"
    );
}

#[test]
fn singing_phrase_locks_onto_midi_pitches_and_is_block_invariant() {
    chimera::init();
    let syl = |txt: &str, onset: &[&str], nuc: &str| Syllable {
        text: txt.into(),
        onset: onset.iter().map(|s| Segment::new(s)).collect(),
        nucleus: Segment::new(nuc),
        word_end: true,
        ..Default::default()
    };
    let notes = vec![
        SungNote {
            midi: 60.0, // C4 = 261.63 Hz
            beats: 1.5,
            syl: syl("la", &["l"], "ɑ"),
        },
        SungNote {
            midi: 64.0, // E4 = 329.63 Hz
            beats: 1.5,
            syl: syl("mi", &["m"], "i"),
        },
        SungNote {
            midi: 67.0, // G4 = 392.00 Hz
            beats: 2.0,
            syl: syl("so", &["s"], "o"),
        },
    ];
    let plan = plan_song(
        &notes,
        &SongOptions {
            body: BodyMix::FEMALE,
            bpm: 100.0,
            t0: 0.30,
            vibrato: 20.0,
        },
    );

    let total = ((plan.end + 0.25) * FS).round() as usize;
    let render_with_block = |bsz: usize| -> Vec<f32> {
        let mut v = VocalBoxVoice::new(FS, BodyMix::FEMALE);
        v.musical_pitch_lock = true;
        v.set_emotion(EmotionPreset::Operatic);
        v.set_source(&SourcePatch::f0(midi_to_hz(60.0)));
        v.schedule(&plan.gestures);
        let mut out = vec![0.0f32; total];
        for ch in out.chunks_mut(bsz) {
            v.render(ch);
        }
        out
    };

    let out_1 = render_with_block(1);
    let out_37 = render_with_block(37);
    let out_256 = render_with_block(256);
    assert_eq!(out_1, out_37, "block size 1 vs 37 must be bit-identical");
    assert_eq!(out_1, out_256, "block size 1 vs 256 must be bit-identical");

    // Verify YIN pitch lock at each sung vowel's steady-state center (33 ms window avoids half-cycle vibrato smearing)
    let beat = 60.0 / 100.0;
    let centers = [
        (0.30 + 0.70 * beat, midi_to_hz(60.0)),
        (0.30 + 2.25 * beat, midi_to_hz(64.0)),
        (0.30 + 3.85 * beat, midi_to_hz(67.0)),
    ];
    for (t_c, want_hz) in centers {
        let idx = (t_c * FS).round() as usize;
        let frame: Vec<f64> = out_256[idx..idx + 1600].iter().map(|&x| x as f64).collect();
        let est = yin_pitch(
            &frame,
            FS,
            YinOpts {
                min_f0: Some(150.0),
                max_f0: Some(700.0),
                threshold: Some(0.25),
            },
        );
        let cents = (1200.0 * (est.f0 / want_hz).log2()).abs();
        assert!(
            cents < 65.0,
            "sung note at t={t_c:.2}s expected {want_hz:.1} Hz, got {:.1} Hz ({cents:.1} cents, vibrato active)",
            est.f0
        );
    }

    // Also verify direct set_articulation hits calibrated vowel postures cleanly
    let mut va = VocalBoxVoice::new(FS, BodyMix::MALE);
    let tbl = &vowel_table()["male"];
    va.set_articulation(&tbl["i"].artic, Some(true));
    va.set_gate(1.0);
    let mut v_buf = vec![0.0f32; (0.35 * FS) as usize];
    va.render(&mut v_buf);
    assert!(rms(&v_buf[(0.15 * FS) as usize..]) > 0.03);
}

#[test]
fn connected_speech_multi_syllable_has_zero_dropouts() {
    chimera::init();
    // "We built this new voice from first principles" — short function words + voiceless clusters
    let segs = tokenize_ipa("wi ˈbɪlt ðɪs ˈnu ˈvɔɪs frʌm ˈfɝst ˈprɪnsəpəlz");
    let plan = plan_utterance(
        &segs,
        &PlanOptions {
            body: BodyMix::MALE,
            t0: 0.05,
            f0: Some(115.0),
            ..Default::default()
        },
    );

    let mut v = VocalBoxVoice::new(FS, BodyMix::MALE);
    v.schedule(&plan.gestures);
    let mut out = vec![0.0f32; ((plan.end + 0.25) * FS).round() as usize];
    v.render(&mut out);

    for seg in &plan.segments {
        if matches!(seg.ipa.as_str(), "i" | "ɪ" | "u" | "ɔɪ" | "ʌ" | "ɝ" | "ə") {
            let ia = (seg.t0 * FS).round() as usize;
            let ib = ((seg.t1 * FS).round() as usize).min(out.len());
            let seg_rms = rms(&out[ia..ib]);
            assert!(
                seg_rms > 0.012,
                "syllable dropout on vowel {:?} [{:.3}..{:.3}s]: rms={seg_rms:.4}",
                seg.ipa,
                seg.t0,
                seg.t1
            );
        }
    }
}

#[test]
fn speech_and_singing_are_dynamic_and_free_of_clicks_or_flat_topping() {
    chimera::init();

    // 1. Sibilant-to-vowel VOT transitions ("ˈʃɑpɑ ˈsɑmɑ") must be C1-smooth
    //    without single-sample state-teleport pops (bounded |Δ²x| / rms) or brick-wall flat-topping.
    let vot_plan = plan_utterance(
        &tokenize_ipa("ˈʃɑpɑ ˈsɑmɑ"),
        &PlanOptions {
            body: BodyMix::FEMALE,
            t0: 0.05,
            f0: Some(210.0),
            ..Default::default()
        },
    );
    for preset in [
        EmotionPreset::Natural,
        EmotionPreset::Warm,
        EmotionPreset::Dramatic,
        EmotionPreset::Angry,
    ] {
        let mut v = VocalBoxVoice::new(FS, BodyMix::FEMALE);
        v.set_emotion(preset);
        v.schedule(&vot_plan.gestures);
        let mut out = vec![0.0f32; ((vot_plan.end + 0.20) * FS).round() as usize];
        v.render(&mut out);

        let r = rms(&out);
        assert!(
            r > 0.015,
            "expected audible speech for {preset:?}, got rms={r}"
        );

        // Check zero consecutive brick-wall flat-topped peak plateaus
        let mut max_flat_run = 0usize;
        let mut cur_run = 0usize;
        for w in out.windows(2) {
            if w[0].abs() > 0.65 && (w[1] - w[0]).abs() < 1.0e-6 {
                cur_run += 1;
                max_flat_run = max_flat_run.max(cur_run);
            } else {
                cur_run = 0;
            }
        }
        assert!(
            max_flat_run <= 1,
            "{preset:?} exhibited brick-wall flat-topped peak run of {max_flat_run} steps"
        );

        // Check 2nd-derivative smoothness in voiced windows across VOT onset
        for &seg_idx in &[1usize, 5usize] {
            let seg = &vot_plan.segments[seg_idx];
            let ia = ((seg.t0 + 0.008) * FS).round() as usize;
            let ib = ((seg.t0 + 0.075) * FS).round() as usize;
            let win = &out[ia..ib.min(out.len())];
            let win_rms = rms(win).max(0.01);
            let mut max_d2 = 0.0f64;
            for w in win.windows(3) {
                let d2 = ((w[2] as f64) - 2.0 * (w[1] as f64) + (w[0] as f64)).abs();
                max_d2 = max_d2.max(d2);
            }
            assert!(
                max_d2 / win_rms < 2.2,
                "VOT pop/click detected in {preset:?} at seg {seg_idx}: max_d2/rms={:.2}",
                max_d2 / win_rms
            );
        }
    }

    // 2. Loud operatic high singing stays strictly below the 0.86 soft-clip knee with 0 flat plateaus
    let syl = |txt: &str, onset: &[&str], nuc: &str| Syllable {
        text: txt.into(),
        onset: onset.iter().map(|s| Segment::new(s)).collect(),
        nucleus: Segment::new(nuc),
        word_end: true,
        ..Default::default()
    };
    let notes = vec![
        SungNote {
            midi: 69.0, // A4 = 440 Hz
            beats: 2.0,
            syl: syl("ah", &[], "ɑ"),
        },
        SungNote {
            midi: 72.0, // C5 = 523.25 Hz
            beats: 2.0,
            syl: syl("la", &["l"], "ɑ"),
        },
    ];
    let song = plan_song(
        &notes,
        &SongOptions {
            body: BodyMix::FEMALE,
            bpm: 96.0,
            t0: 0.20,
            vibrato: 24.0,
        },
    );
    let mut vs = VocalBoxVoice::new(FS, BodyMix::FEMALE);
    vs.musical_pitch_lock = true;
    vs.set_emotion(EmotionPreset::Operatic);
    vs.set_source(&SourcePatch::f0(midi_to_hz(69.0)));
    vs.schedule(&song.gestures);
    let mut song_out = vec![0.0f32; ((song.end + 0.25) * FS).round() as usize];
    vs.render(&mut song_out);

    let peak = song_out.iter().fold(0.0f64, |m, &x| m.max((x as f64).abs()));
    let song_rms = rms(&song_out);
    let crest_db = 20.0 * (peak / song_rms).log10();
    assert!(
        crest_db > 6.5 && peak < 0.86,
        "Operatic singing hit nonlinear soft-clip or lost crest factor: crest={crest_db:.1} dB, peak={peak:.3}"
    );
    let flat_plateaus = song_out
        .windows(3)
        .filter(|w| {
            w[0].abs() > 0.65 && (w[1] - w[0]).abs() < 1.0e-6 && (w[2] - w[1]).abs() < 1.0e-6
        })
        .count();
    assert_eq!(flat_plateaus, 0, "Operatic singing had {flat_plateaus} flat-topped peak plateaus");
}

#[test]
fn modal_speech_glottal_closure_and_cross_body_intelligibility() {
    chimera::init();

    for (label, body, f0) in [
        ("male", BodyMix::MALE, 112.0),
        ("female", BodyMix::FEMALE, 205.0),
        ("child", BodyMix::CHILD, 250.0),
    ] {
        // Connected phrase with sibilants, stops, clusters, and diphthongs: "She shapes sharp waves"
        let segs = tokenize_ipa("ʃi ˈʃeɪps ˈʃɑrp ˈweɪvz");
        let plan = plan_utterance(
            &segs,
            &PlanOptions {
                body,
                t0: 0.05,
                f0: Some(f0),
                ..Default::default()
            },
        );
        let mut v = VocalBoxVoice::new(FS, body);
        v.set_source(&SourcePatch::f0(f0));
        v.schedule(&plan.gestures);

        let total = ((plan.end + 0.25) * FS).round() as usize;
        let mut out = vec![0.0f32; total];
        let mut min_voiced_glottal_area = f64::INFINITY;
        for chunk in out.chunks_mut(4) {
            v.render(chunk);
            let tel = v.instant_telemetry();
            if tel.voicing > 0.80 {
                min_voiced_glottal_area = min_voiced_glottal_area.min(tel.glottal_area_cm2);
            }
        }

        // 1. Modal speech must achieve true glottal closure (< 0.010 cm^2, vs ~0.12 cm^2 peak open)
        assert!(
            min_voiced_glottal_area < 0.010,
            "{label} modal speech failed to close glottis (min area = {min_voiced_glottal_area:.5} cm^2)"
        );

        // 2. Every vowel must voice clearly and every voiceless sibilant must carry audible frication
        let mut vowel_rms_sum = 0.0;
        let mut vowel_n = 0usize;
        let mut sib_rms_sum = 0.0;
        let mut sib_n = 0usize;
        for seg in &plan.segments {
            let ia = (seg.t0 * FS).round() as usize;
            let ib = ((seg.t1 * FS).round() as usize).min(out.len());
            let r = rms(&out[ia..ib]);
            if matches!(seg.ipa.as_str(), "i" | "eɪ" | "ɑ") {
                assert!(
                    r > 0.015,
                    "{label} vowel {:?} too quiet: rms={r:.4}",
                    seg.ipa
                );
                vowel_rms_sum += r;
                vowel_n += 1;
            } else if matches!(seg.ipa.as_str(), "ʃ" | "s") {
                sib_rms_sum += r;
                sib_n += 1;
            }
        }
        let mean_vowel_rms = vowel_rms_sum / vowel_n as f64;
        let mean_sib_rms = sib_rms_sum / sib_n as f64;
        assert!(
            mean_sib_rms > 0.14 * mean_vowel_rms,
            "{label} sibilants choked relative to vowels: sib={mean_sib_rms:.4}, vowel={mean_vowel_rms:.4}"
        );
    }
}


