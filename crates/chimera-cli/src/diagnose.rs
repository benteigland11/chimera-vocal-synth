//! `chimera diagnose [outdir]`: port of `tools/speech-diagnostics.ts`, the running-speech battery
//! ("marbles in the mouth"). Renders sentences through the full pipeline and measures, against
//! published norms:
//!  - nasal leakage: velopharyngeal port during vowels with no nasal neighbour (heard as nasal
//!    from ~0.01 cm² for high vowels, ~0.15 cm² for low; Bunton & Story 2011)
//!  - vowel undershoot: LPC F1/F2 at vowel midpoints vs the calibrated (Hillenbrand) targets
//!  - formant bandwidths at vowel midpoints (Klatt defaults B1 60, B2 90, B3 150 Hz)
//!  - spectral balance: octave-band long-term spectrum relative to 500 Hz
//!  - duration contrasts: stressed vs unstressed, phrase-final vs not
//!  - pitch: share of voiced time stuck at the baseline (snapping back between vowels)
//!
//! `DIAG_VERBOSE=1` also lists every measured vowel.

use crate::js::{fixed, num, pad_end, pad_start};
use crate::render::g2p;
use crate::{Ctx, FS};
use chimera::analysis::fft::{Fft, hann};
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::math::{hypot, js_max, js_min, js_round, ln, log2, log10};
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::planner::{PhraseInput, PlanOptions, diphthong_parts, plan_phrases};
use chimera::phonetics::vowel_data::VOWELS;
use chimera::vocal_box::VocalBoxVoice;
use chimera::voice::human_voice::HumanVoice;
use std::collections::HashMap;

const SENTENCES: [&str; 5] = [
    "The north wind and the sun were disputing which was the stronger.",
    "We were away a year ago.",
    "Should we chase those cowboys?",
    "Can you hear me now?",
    "She sells sea shells by the sea shore.",
];
const NASALS: [&str; 3] = ["m", "n", "ŋ"];

fn diphthong_of(a: &str, b: &str) -> bool {
    ["aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ"]
        .iter()
        .any(|d| diphthong_parts(d).is_some_and(|(p0, p1)| p0 == a && p1 == b))
}

struct VowelStat {
    ipa: String,
    stressed: bool,
    is_final: bool,
    dur: f64,
    f1: f64,
    f2: f64,
    t1: f64,
    t2: f64,
    bw: Vec<f64>,
}

fn sorted(xs: &[f64]) -> Vec<f64> {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    s
}

fn median(xs: &[f64]) -> f64 {
    let s = sorted(xs);
    s.get(s.len() / 2).copied().unwrap_or(f64::NAN)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / js_max(1.0, xs.len() as f64)
}

/// LPC fails on short vowels that came out voiceless (aspiration): drop frames with no F1 below 1.2 kHz
fn ok<'a>(vs: &[&'a VowelStat]) -> Vec<&'a VowelStat> {
    vs.iter()
        .copied()
        .filter(|x| x.f1 < 1200.0 && x.f2 < 3200.0)
        .collect()
}

fn pct(a: usize, b: usize) -> String {
    format!("{} %", fixed((100.0 * a as f64) / js_max(1.0, b as f64), 1))
}

pub fn run(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let out_dir = a.first().map(String::as_str);
    let g = g2p()?;
    let reference: HashMap<&str, [f64; 3]> = VOWELS.iter().map(|v| (v.ipa, v.male)).collect();

    let mut vowels: Vec<VowelStat> = Vec::new();
    let (mut oral_frames, mut leak1, mut leak5, mut leak15, mut max_oral_port) =
        (0usize, 0usize, 0usize, 0usize, 0.0f64);
    let (mut voiced_frames, mut at_base) = (0usize, 0usize);
    let mut f0s: Vec<f64> = Vec::new();
    let bands = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
    let mut band_e = [0.0f64; 6];
    let fft = Fft::new(2048);
    let win = hann(2048);

    for (si, text) in SENTENCES.iter().enumerate() {
        let phrases = g.to_phrases(text);
        let inputs: Vec<PhraseInput> = phrases
            .iter()
            .map(|p| PhraseInput {
                segments: p.segments.clone(),
                intonation: Some(p.intonation),
                pause: Some(p.pause),
            })
            .collect();
        let rate = ctx.emotion.map(|em| em.profile().rate_scale);
        let plan = plan_phrases(
            &inputs,
            &PlanOptions {
                body: "male".into(),
                t0: 0.1,
                f0: Some(110.0),
                rate,
                ..Default::default()
            },
        );
        let n = js_round(FS * (plan.end + 0.3)) as usize;
        let mut out = vec![0f32; n];
        let (mut port, mut f0t, mut voic) = (Vec::new(), Vec::new(), Vec::new());

        if ctx.use_vocal_box() {
            let mut v = VocalBoxVoice::new(FS, BodyMix::MALE);
            v.set_two_mass(ctx.two_mass);
            if let Some(em) = ctx.emotion {
                v.set_emotion(em);
            }
            v.schedule(&plan.gestures);
            for block in out.chunks_mut(128) {
                v.render(block);
                let tel = v.instant_telemetry();
                port.push(tel.velum_port_cm2);
                f0t.push(tel.f0_now_hz);
                voic.push(tel.voicing);
            }
        } else {
            let mut v = HumanVoice::new(FS, chimera::anatomy::human::BODY_MALE, 2);
            if let Some(em) = ctx.emotion {
                v.set_source(&em.profile().to_source_patch(110.0));
            }
            v.schedule(&plan.gestures);
            for block in out.chunks_mut(128) {
                v.render(block);
                port.push(v.airway.frame.as_ref().map_or(0.0, |f| f.velum_port));
                f0t.push(v.glottis.f0_now);
                voic.push(v.glottis.voicing);
            }
        }
        let frame_at = |t: f64| {
            js_min(
                (port.len() - 1) as f64,
                js_max(0.0, js_round((t * FS) / 128.0)),
            ) as usize
        };
        let segs = &plan.segments;
        let sample = |i: i64| {
            if i >= 0 && (i as usize) < out.len() {
                out[i as usize] as f64
            } else {
                0.0
            }
        };
        for (k, s) in segs.iter().enumerate() {
            if s.kind != "vowel" {
                continue;
            }
            // the second half of a diphthong is a glide, not a vowel that should reach its target
            let pv = if k > 0 { segs.get(k - 1) } else { None };
            if let Some(pv) = pv
                && pv.kind == "vowel"
                && (pv.t1 - s.t0).abs() < 1e-6
                && diphthong_of(&pv.ipa, &s.ipa)
            {
                continue;
            }
            let nasal_nb = pv.is_some_and(|p| NASALS.contains(&p.ipa.as_str()))
                || segs
                    .get(k + 1)
                    .is_some_and(|p| NASALS.contains(&p.ipa.as_str()));
            if !nasal_nb {
                for f in frame_at(s.t0)..=frame_at(s.t1) {
                    oral_frames += 1;
                    if port[f] > 0.01 {
                        leak1 += 1;
                    }
                    if port[f] > 0.05 {
                        leak5 += 1;
                    }
                    if port[f] > 0.15 {
                        leak15 += 1;
                    }
                    max_oral_port = js_max(max_oral_port, port[f]);
                }
            }
            // formants at the midpoint (first target of a diphthong at 30 %)
            let parts = diphthong_parts(&s.ipa);
            let tgt = parts.map_or(s.ipa.as_str(), |p| p.0);
            let r = reference.get(tgt);
            let tm = if parts.is_some() {
                s.t0 + 0.3 * (s.t1 - s.t0)
            } else {
                (s.t0 + s.t1) / 2.0
            };
            let (dec, len) = (4i64, js_round(0.025 * FS) as i64);
            let c = js_round(tm * FS) as i64;
            let mut fr: Vec<f64> = Vec::new();
            let mut i = c - len / 2;
            while i < c + len / 2 {
                let mut acc = 0.0;
                for j in 0..dec {
                    acc += sample(i + j);
                }
                fr.push(acc / dec as f64);
                i += dec;
            }
            let fm = estimate_formants(
                &fr,
                FS / dec as f64,
                FormantOpts {
                    order: Some(14),
                    max_bw: Some(700.0),
                    ..Default::default()
                },
            );
            if let Some(r) = r
                && fm.len() >= 2
            {
                let next = segs[k + 1..].iter().find(|x| x.kind == "vowel");
                vowels.push(VowelStat {
                    ipa: s.ipa.clone(),
                    stressed: s.stress == Some(1),
                    is_final: next.is_none_or(|nx| nx.t0 - s.t1 > 0.2),
                    dur: s.t1 - s.t0,
                    f1: fm[0].freq,
                    f2: fm[1].freq,
                    t1: r[0],
                    t2: r[1],
                    bw: fm.iter().take(3).map(|x| x.bw).collect(),
                });
            }
        }
        // pitch: voiced frames within 3 % of the 110 Hz baseline
        for f in 0..f0t.len() {
            if voic[f] > 0.5 {
                voiced_frames += 1;
                f0s.push(f0t[f]);
                if (f0t[f] / 110.0 - 1.0).abs() < 0.03 {
                    at_base += 1;
                }
            }
        }
        // long-term spectrum over frames with signal
        let pk = crate::peak(&out);
        let mut i = 0;
        while i + 2048 < n {
            let mut e = 0.0;
            for j in 0..2048 {
                e += out[i + j] as f64 * out[i + j] as f64;
            }
            if !((e / 2048.0).sqrt() < pk * 0.01) {
                let mut re = vec![0.0; 2048];
                let mut im = vec![0.0; 2048];
                for j in 0..2048 {
                    re[j] = out[i + j] as f64 * win[j];
                }
                fft.transform(&mut re, &mut im);
                for k in 1..1024 {
                    let f = (k as f64 * FS) / 2048.0;
                    let p = re[k] * re[k] + im[k] * im[k];
                    for (bi, b) in bands.iter().enumerate() {
                        if f >= b / core::f64::consts::SQRT_2 && f < b * core::f64::consts::SQRT_2 {
                            band_e[bi] += p;
                        }
                    }
                }
            }
            i += 1024;
        }
        if let Some(dir) = out_dir {
            ctx.write_wav(&format!("{dir}/s{}.wav", si + 1), &out)?;
        }
    }

    println!("== nasal leakage during vowels with no nasal neighbour");
    println!(
        "   frames with port > 0.01 cm²: {}   > 0.05: {}   > 0.15: {}   max {} cm²   (norm: ~0 %)",
        pct(leak1, oral_frames),
        pct(leak5, oral_frames),
        pct(leak15, oral_frames),
        fixed(max_oral_port, 3)
    );
    println!("== vowel midpoints vs calibrated targets");
    let err = |vs0: &[&VowelStat]| -> String {
        let vs = ok(vs0);
        let e1: Vec<f64> = vs.iter().map(|x| (x.f1 / x.t1 - 1.0).abs()).collect();
        let e2: Vec<f64> = vs.iter().map(|x| (x.f2 / x.t2 - 1.0).abs()).collect();
        format!(
            "F1 {} %  F2 {} % median error (n={}/{} measurable)",
            fixed(100.0 * median(&e1), 0),
            fixed(100.0 * median(&e2), 0),
            vs.len(),
            vs0.len()
        )
    };
    // distance from schwa in (F1, log F2) relative to the target's
    let sch = [500.0, 1500.0];
    let reach = |vs: &[&VowelStat]| -> f64 {
        let d: Vec<f64> = ok(vs)
            .iter()
            .filter(|x| hypot(x.t1 - sch[0], 1000.0 * ln(x.t2 / sch[1])) > 80.0)
            .map(|x| {
                hypot(x.f1 - sch[0], 1000.0 * ln(x.f2 / sch[1]))
                    / hypot(x.t1 - sch[0], 1000.0 * ln(x.t2 / sch[1]))
            })
            .collect();
        median(&d)
    };
    let all: Vec<&VowelStat> = vowels.iter().collect();
    let s_v: Vec<&VowelStat> = vowels.iter().filter(|x| x.stressed).collect();
    let u_v: Vec<&VowelStat> = vowels.iter().filter(|x| !x.stressed).collect();
    println!(
        "   stressed:   {}; distance from schwa reached {} % (norm ≥ 90 %)",
        err(&s_v),
        fixed(100.0 * reach(&s_v), 0)
    );
    println!(
        "   unstressed: {}; distance from schwa reached {} % (norm ≥ 75 %)",
        err(&u_v),
        fixed(100.0 * reach(&u_v), 0)
    );
    if std::env::var_os("DIAG_VERBOSE").is_some_and(|v| !v.is_empty()) {
        for x in &vowels {
            println!(
                "     {} {} {} ms  F1 {}/{}  F2 {}/{}  B1 {}",
                if x.stressed { "S" } else { "u" },
                pad_end(&x.ipa, 3),
                pad_start(&fixed(1000.0 * x.dur, 0), 4),
                pad_start(&fixed(x.f1, 0), 4),
                num(x.t1),
                pad_start(&fixed(x.f2, 0), 5),
                num(x.t2),
                fixed(x.bw[0], 0)
            );
        }
    }
    println!("== formant bandwidths at vowel midpoints (median)");
    let bw = |i: usize| -> Vec<f64> {
        all.iter()
            .map(|x| x.bw.get(i).copied().unwrap_or(f64::NAN))
            .collect()
    };
    let b3: Vec<f64> = bw(2).into_iter().filter(|x| x.is_finite()).collect();
    println!(
        "   B1 {} Hz  B2 {} Hz  B3 {} Hz   (Klatt defaults 60 / 90 / 150)",
        fixed(median(&bw(0)), 0),
        fixed(median(&bw(1)), 0),
        fixed(median(&b3), 0)
    );
    println!("== long-term spectrum, octave bands re 500 Hz");
    let ref500 = band_e[1];
    let ltass = [-2, 0, -6, -11, -16, -20]; // approximate natural male speech (Byrne et al. 1994 LTASS, octave-averaged)
    let cells: Vec<String> = bands
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let name = if b >= 1000.0 {
                format!("{}k", num(b / 1000.0))
            } else {
                num(b)
            };
            format!(
                "{name}: {} (≈{})",
                fixed(10.0 * log10(band_e[i] / ref500), 1),
                ltass[i]
            )
        })
        .collect();
    println!("   {}", cells.join("   "));
    println!("== durations");
    let durs = |vs: &[&VowelStat]| -> Vec<f64> { vs.iter().map(|x| x.dur).collect() };
    let fin: Vec<&VowelStat> = vowels.iter().filter(|x| x.is_final).collect();
    let non: Vec<&VowelStat> = vowels.iter().filter(|x| !x.is_final).collect();
    println!(
        "   stressed {} ms vs unstressed {} ms (norm ≥ 2 ×); phrase-final {} ms vs non-final {} ms (norm: final ≈ 1.3–1.4 ×)",
        fixed(1000.0 * mean(&durs(&s_v)), 0),
        fixed(1000.0 * mean(&durs(&u_v)), 0),
        fixed(1000.0 * mean(&durs(&fin)), 0),
        fixed(1000.0 * mean(&durs(&non)), 0)
    );
    println!("== pitch");
    let st = |a: f64, b: f64| 12.0 * log2(a / b);
    let s = sorted(&f0s);
    let q = |p: f64| {
        s.get((s.len() as f64 * p).floor() as usize)
            .copied()
            .unwrap_or(f64::NAN)
    };
    println!(
        "   voiced time at the 110 Hz baseline (±3 %): {}   range (5–95 %) {} semitones",
        pct(at_base, voiced_frames),
        fixed(st(q(0.95), q(0.05)), 1)
    );
    Ok(())
}
