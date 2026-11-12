//! `chimera render …`: ports of `tools/render-{speech,text,song,beatbox,creature,vowels}.ts`.

use crate::js::{fixed, num, pad_end, pad_start};
use crate::{Ctx, FS, arg, body_mix, peak, repo_path, wav_name};
use chimera::analysis::lpc::{FormantOpts, estimate_formants};
use chimera::analysis::pitch::{YinOpts, yin_pitch};
use chimera::anatomy::human::{ArticKey, BodyMix, N_ARTIC};
use chimera::control::beat_player::BeatPlayer;
use chimera::control::score::Gesture;
use chimera::creatures::types::ParamRole;
use chimera::creatures::voice::CreatureVoice;
use chimera::creatures::{CREATURES, creature_by_id};
use chimera::math::{js_max, js_min, js_round, log10};
use chimera::phonetics::beatbox::{BEAT_PRESETS, BEAT_SOUNDS, BeatStep, beat_ready, parse_beat};
use chimera::phonetics::english::EnglishG2P;
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{
    PhraseInput, PlanOptions, plan_phrases, plan_utterance, vowel_table,
};
use chimera::phonetics::sing::{
    LegatoFrom, NoteOnOptions, SongOptions, SungNote, midi_to_hz, plan_note_off, plan_note_on,
    plan_song,
};
use chimera::phonetics::songs::DEMO_SONGS;
use chimera::phonetics::syllables::lyrics_to_syllables;
use chimera::phonetics::{Intonation, Plan, Syllable};
use chimera::vocal_box::VocalBoxVoice;
use chimera::voice::human_voice::{SCOPE_POINTS, SourcePatch, VoiceTelemetry};
use std::io::Read;
use std::time::Instant;

fn new_voice(ctx: &Ctx, body_name: &str, musical: bool) -> Result<VocalBoxVoice, String> {
    let mut v = VocalBoxVoice::new(FS, body_mix(body_name)?);
    v.musical_pitch_lock = musical;
    if let Some(em) = ctx.emotion {
        v.set_emotion(em);
    }
    Ok(v)
}

fn apply_source(v: &mut VocalBoxVoice, ctx: &Ctx, patch: &SourcePatch) {
    v.set_source(patch);
    if let Some(em) = ctx.emotion {
        v.set_emotion(em);
        if let Some(f0) = patch.f0 {
            v.set_source(&SourcePatch::f0(f0));
        }
    }
}

/// Render `out.len()` samples in 128-sample blocks, as the tools did.
fn render_blocks(v: &mut VocalBoxVoice, out: &mut [f32]) {
    for block in out.chunks_mut(128) {
        v.render(block);
    }
}

/// The CMU dictionary (`public/data/cmudict.txt.gz`) loaded into a G2P.
pub fn g2p() -> Result<EnglishG2P, String> {
    let path = repo_path("public/data/cmudict.txt.gz");
    let gz = std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_string(&mut text)
        .map_err(|e| format!("{path}: {e}"))?;
    let mut g = EnglishG2P::new();
    g.load_dict(&text);
    Ok(g)
}

/// `name.toLowerCase().replace(/[^a-z]+/g, "-")`
fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut run = false;
    for c in name.to_lowercase().chars() {
        if c.is_ascii_lowercase() {
            out.push(c);
            run = false;
        } else if !run {
            out.push('-');
            run = true;
        }
    }
    out
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Mark list `ipa@t` as render-speech printed it.
fn mark_line(marks: &[(f64, String)]) -> String {
    marks
        .iter()
        .map(|(t, l)| format!("{l}@{}", fixed(*t, 3)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// IPA utterances through the planner. `render speech <out> ["ˈɑpɑ / ˈɑbɑ"] [body]`
pub fn speech(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let name = arg(a, 0, "out/speech");
    let text = arg(a, 1, "ˈɑpɑ");
    let body_name = arg(a, 2, "male");
    let f0 = match body_name {
        "female" => 200.0,
        "child" => 250.0,
        _ => 110.0,
    };
    let mut voice = new_voice(ctx, body_name, false)?;
    apply_source(&mut voice, ctx, &SourcePatch::f0(f0));
    let rate = ctx.emotion.map(|em| em.profile().rate_scale);
    let mut marks: Vec<(f64, String)> = Vec::new();
    let mut t0 = 0.05;
    for u in text.split('/').map(str::trim).filter(|u| !u.is_empty()) {
        let segs = tokenize_ipa(u);
        let plan = plan_utterance(
            &segs,
            &PlanOptions {
                body: body_name.into(),
                t0,
                f0: Some(f0),
                rate,
                ..Default::default()
            },
        );
        voice.schedule(&plan.gestures);
        for s in &plan.segments {
            marks.push((s.t0, s.ipa.clone()));
        }
        t0 = plan.end + 0.35;
    }
    let total = js_round(FS * (t0 + 0.2)) as usize;
    let mut out = vec![0f32; total];
    render_blocks(&mut voice, &mut out);
    let file = wav_name(name);
    ctx.write_wav(&file, &out)?;
    println!(
        "wrote {file}  {} s  peak {}",
        fixed(total as f64 / FS, 2),
        fixed(peak(&out), 3)
    );
    println!("{}", mark_line(&marks));
    Ok(())
}

/// A sentence said by the text path: the samples and the plan.
pub struct Spoken {
    pub out: Vec<f32>,
    pub plan: Plan,
    /// the IPA `render text` prints, phrases joined by " | "
    pub shown: String,
}

/// The f0 `render text` speaks at in each body.
pub fn text_f0(body_name: &str) -> f64 {
    match body_name {
        "female" => 205.0,
        "child" => 250.0,
        _ => 112.0,
    }
}

/// Say `text` the way `render text` does: CMUdict G2P, the phrase planner from t0 = 50 ms, the
/// body's speaking f0, 300 ms of tail. `chimera compare tsp` measures exactly these renders.
pub fn say(g: &EnglishG2P, text: &str, body_name: &str) -> Result<Spoken, String> {
    say_with_ctx(&Ctx::default(), g, text, body_name)
}

fn say_with_ctx(ctx: &Ctx, g: &EnglishG2P, text: &str, body_name: &str) -> Result<Spoken, String> {
    let f0 = text_f0(body_name);
    let phrases = g.to_phrases(text);
    let shown: Vec<String> = phrases
        .iter()
        .map(|p| {
            let s: String = p
                .segments
                .iter()
                .map(|s| format!("{}{}", if s.stress == Some(1) { "ˈ" } else { "" }, s.ipa))
                .collect();
            s + if p.intonation == Intonation::Question {
                "?"
            } else {
                ""
            }
        })
        .collect();
    let inputs: Vec<PhraseInput> = phrases
        .iter()
        .map(|p| PhraseInput {
            segments: p.segments.clone(),
            intonation: Some(p.intonation),
            pause: Some(p.pause),
        })
        .collect();

    let mut voice = new_voice(ctx, body_name, false)?;
    apply_source(&mut voice, ctx, &SourcePatch::f0(f0));
    let rate = ctx.emotion.map(|em| em.profile().rate_scale);
    let plan = plan_phrases(
        &inputs,
        &PlanOptions {
            body: body_name.into(),
            t0: 0.05,
            f0: Some(f0),
            rate,
            ..Default::default()
        },
    );
    voice.schedule(&plan.gestures);
    let total = js_round(FS * (plan.end + 0.3)) as usize;
    let mut out = vec![0f32; total];
    render_blocks(&mut voice, &mut out);
    Ok(Spoken {
        out,
        plan,
        shown: shown.join(" | "),
    })
}

/// English text through G2P and the phrase planner. `render text <out> ["Text."] [body]`
pub fn text(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let name = arg(a, 0, "out/text");
    let text = arg(a, 1, "Hello world.");
    let body_name = arg(a, 2, "male");
    let g = g2p()?;
    let text_ctx = Ctx {
        float: ctx.float,
        emotion: Some(ctx.emotion.unwrap_or_default()),
    };
    let t = Instant::now();
    let s = say_with_ctx(&text_ctx, &g, text, body_name)?;
    let ms = ms_since(t);
    println!("{}", s.shown);
    let file = wav_name(name);
    text_ctx.write_wav(&file, &s.out)?;
    println!(
        "wrote {file}  {} s, rendered in {} ms",
        fixed(s.out.len() as f64 / FS, 2),
        fixed(ms, 0)
    );
    Ok(())
}

/// The demo songs, and a live-style key sequence. `render song [outdir] [body]`
pub fn song(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let out_dir = arg(a, 0, "out/song");
    let body_name = arg(a, 1, "male");
    let _ = body_mix(body_name)?;
    let g = g2p()?;
    let syl_text = |syl: &[Syllable]| {
        syl.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    };

    for song in DEMO_SONGS {
        let syl = lyrics_to_syllables(song.lyrics, &g);
        let octave = match body_name {
            "female" => song.octave.female,
            "child" => song.octave.child,
            _ => song.octave.male,
        };
        let shift = octave * 12.0;
        let notes: Vec<SungNote> = song
            .notes
            .iter()
            .enumerate()
            .map(|(i, n)| SungNote {
                midi: n[0] + shift,
                beats: n[1],
                syl: syl[i % syl.len()].clone(),
            })
            .collect();
        let mut v = new_voice(ctx, body_name, true)?;
        let f0_init = midi_to_hz(notes[0].midi);
        apply_source(
            &mut v,
            ctx,
            &SourcePatch {
                f0: Some(f0_init),
                voiced: Some(true),
                pressure: Some(9.0),
                tension: Some(0.55),
                ..Default::default()
            },
        );
        let vibrato = ctx
            .emotion
            .map(|em| {
                let vd = em.profile().vibrato_depth_cents;
                if vd > 0.0 { vd } else { 35.0 }
            })
            .unwrap_or(35.0);
        let plan = plan_song(
            &notes,
            &SongOptions {
                body: body_name.into(),
                bpm: song.bpm,
                t0: 0.3,
                vibrato,
            },
        );
        v.schedule(&plan.gestures);
        let n = js_round(FS * (plan.end + 0.4)) as usize;
        let mut out = vec![0f32; n];
        render_blocks(&mut v, &mut out);
        ctx.write_wav(
            &format!("{out_dir}/{}-{body_name}.wav", slug(song.name)),
            &out,
        )?;
        println!(
            "{}: {} s, peak {}, syllables {}",
            song.name,
            fixed(n as f64 / FS, 1),
            fixed(peak(&out), 3),
            syl_text(&syl)
        );
    }

    // live-style: key presses with gaps and one legato pair, as the keyboard would send them
    let syl = lyrics_to_syllables("hello there my friend", &g);
    let mut v = new_voice(ctx, body_name, true)?;
    apply_source(
        &mut v,
        ctx,
        &SourcePatch {
            f0: Some(130.0),
            voiced: Some(true),
            pressure: Some(9.0),
            ..Default::default()
        },
    );
    let keys: [(f64, f64, f64); 4] = [
        (0.2, 0.55, 57.0),
        (0.6, 1.0, 59.0),
        (1.1, 1.8, 62.0),
        (1.75, 2.6, 60.0),
    ]; // (down, up, midi)
    let n = js_round(FS * 3.0) as usize;
    let mut out = vec![0f32; n];
    let mut tag = 2.0;
    struct Held {
        syl: Syllable,
        midi: f64,
        tag: f64,
    }
    let mut held: Option<Held> = None;
    let mut events: Vec<(f64, bool, f64, usize)> = keys
        .iter()
        .enumerate()
        .flat_map(|(i, &(d, u, m))| [(d, true, m, i), (u, false, m, i)])
        .collect();
    events.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut e = 0;
    for block in out.chunks_mut(128) {
        let now = v.clock;
        while e < events.len() && events[e].0 <= now {
            let (_, on, m, i) = events[e];
            e += 1;
            let shift = |gs: Vec<Gesture>| -> Vec<Gesture> {
                gs.into_iter()
                    .map(|g| Gesture {
                        t0: g.t0 + now + 0.006,
                        t1: g.t1 + now + 0.006,
                        ..g
                    })
                    .collect()
            };
            if on {
                let s = &syl[i];
                if let Some(h) = &held {
                    v.release(h.tag);
                    v.release(h.tag + 1.0);
                }
                let legato_from = held.as_ref().map(|h| LegatoFrom {
                    syl: h.syl.clone(),
                    midi: h.midi,
                });
                let gs = shift(plan_note_on(
                    s,
                    m,
                    &NoteOnOptions {
                        body: body_name.into(),
                        vibrato: 35.0,
                        tag,
                        legato_from,
                    },
                ));
                v.schedule(&gs);
                held = Some(Held {
                    syl: s.clone(),
                    midi: m,
                    tag,
                });
                tag += 2.0;
            } else if let Some(h) = held.take_if(|h| h.midi == m) {
                v.release(h.tag);
                v.release(h.tag + 1.0);
                let gs = shift(plan_note_off(&h.syl, h.midi, body_name.into()));
                v.schedule(&gs);
            }
        }
        v.render(block);
    }
    ctx.write_wav(&format!("{out_dir}/live-{body_name}.wav"), &out)?;
    println!("live: syllables {}", syl_text(&syl));
    Ok(())
}

/// Articulator tracks are sampled at this rate (Hz): 480 samples at 48 kHz.
const ARTIC_RATE: usize = 100;

/// Telemetry frames (`--tele`) are taken every `TELE_HOP` samples: the live app's cadence (the
/// worklet posts one every 6 render quanta of 128), so each frame's RMS levels cover as long.
pub const TELE_HOP: usize = 768;
/// Points kept of the 256-point glottal scope in a telemetry frame (every 8th).
pub const TELE_SCOPE: usize = 32;
/// `--tele` file version (see `TeleRecorder`).
pub const TELE_VERSION: u32 = 1;

struct TeleRecorder {
    t: VoiceTelemetry,
    data: Vec<f32>,
    frames: u32,
    n_ph: usize,
    n_or: usize,
}

impl TeleRecorder {
    fn new(v: &mut VocalBoxVoice) -> Self {
        let mut r = TeleRecorder {
            t: VoiceTelemetry::default(),
            data: Vec::new(),
            frames: 0,
            n_ph: 0,
            n_or: 0,
        };
        v.telemetry(&mut r.t);
        r.n_ph = r.t.ph_areas.len();
        r.n_or = r.t.or_areas.len();
        r.push_current();
        r
    }

    fn frame_len(&self) -> usize {
        11 + N_ARTIC + 3 * (self.n_ph + self.n_or) + 2 * TELE_SCOPE
    }

    fn snap(&mut self, v: &mut VocalBoxVoice) -> Result<(), String> {
        v.telemetry(&mut self.t);
        if self.t.ph_areas.len() != self.n_ph || self.t.or_areas.len() != self.n_or {
            return Err("the airway's section count changed during the render".into());
        }
        self.push_current();
        Ok(())
    }

    fn push_current(&mut self) {
        let t = &self.t;
        let d = &mut self.data;
        d.extend(
            [
                t.ph_len,
                t.or_len,
                t.velum_port,
                t.f0,
                t.voicing,
                t.p_lung,
                t.out_rms,
                t.noise_level,
                t.noise_node as f64,
                t.gate,
                t.clock,
            ]
            .map(|x| x as f32),
        );
        for arr in [&t.artic, &t.ph_areas, &t.or_areas, &t.pressure_rms, &t.flow] {
            d.extend_from_slice(arr);
        }
        let step = SCOPE_POINTS / TELE_SCOPE;
        for arr in [&t.glottal_area, &t.glottal_flow] {
            d.extend((0..TELE_SCOPE).map(|i| arr[i * step]));
        }
        self.frames += 1;
        debug_assert_eq!(self.data.len(), self.frames as usize * self.frame_len());
    }

    fn bytes(&self, body: BodyMix) -> Vec<u8> {
        let mut b = Vec::with_capacity(48 + 4 * self.data.len());
        b.extend_from_slice(b"CHTL");
        for x in [
            TELE_VERSION,
            FS as u32,
            TELE_HOP as u32,
            self.frames,
            self.frame_len() as u32,
            N_ARTIC as u32,
            self.n_ph as u32,
            self.n_or as u32,
            TELE_SCOPE as u32,
        ] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in [body.female as f32, body.child as f32] {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.data {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b
    }
}

fn render_tracked(
    v: &mut VocalBoxVoice,
    out: &mut [f32],
    artic: bool,
    tele: bool,
) -> Result<(Vec<Vec<f64>>, Option<TeleRecorder>), String> {
    let a_hop = FS as usize / ARTIC_RATE;
    let frame = |v: &VocalBoxVoice| {
        let ax = &v.artic.x;
        ArticKey::ALL
            .iter()
            .map(|k| js_round(ax.get(*k) * 1e4) / 1e4)
            .collect()
    };
    let mut frames = if artic { vec![frame(v)] } else { Vec::new() };
    let mut rec = if tele {
        Some(TeleRecorder::new(v))
    } else {
        None
    };
    let n = out.len();
    let (mut pos, mut next_a, mut next_t) = (0usize, a_hop, TELE_HOP);
    while pos < n {
        let mut end = n;
        if artic {
            end = end.min(next_a);
        }
        if tele {
            end = end.min(next_t);
        }
        render_blocks(v, &mut out[pos..end]);
        pos = end;
        if artic && (pos == next_a || pos == n) {
            frames.push(frame(v));
            next_a += a_hop;
        }
        if let Some(r) = rec.as_mut()
            && pos == next_t
        {
            r.snap(v)?;
            next_t += TELE_HOP;
        }
    }
    Ok((frames, rec))
}

/// A melody from a JSON file: `{"bpm", "lyrics", "notes": [[midi, beats], ...], "vibrato"?}`.
pub fn notes(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let mut artic_out: Option<String> = None;
    let mut tele_out: Option<String> = None;
    let mut rest = Vec::new();
    let mut it = a.iter();
    while let Some(x) = it.next() {
        if x == "--artic" {
            artic_out = Some(it.next().ok_or("--artic needs a file")?.clone());
        } else if x == "--tele" {
            tele_out = Some(it.next().ok_or("--tele needs a file")?.clone());
        } else {
            rest.push(x.clone());
        }
    }
    let a = &rest[..];
    let input = a
        .first()
        .ok_or("render notes <in.json> <out> [male|female|child] [--artic tracks.json] [--tele voice.tele]")?;
    let out_name = wav_name(arg(a, 1, "out/notes"));
    let body_name = arg(a, 2, "male");
    let bm = body_mix(body_name)?;
    let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
    let j: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{input}: {e}"))?;
    let bpm = j["bpm"].as_f64().ok_or("bpm missing")?;
    let vibrato = j["vibrato"].as_f64().unwrap_or_else(|| {
        ctx.emotion
            .map(|em| {
                let vd = em.profile().vibrato_depth_cents;
                if vd > 0.0 { vd } else { 35.0 }
            })
            .unwrap_or(35.0)
    });
    let raw: Vec<[f64; 2]> =
        serde_json::from_value(j["notes"].clone()).map_err(|e| format!("notes: {e}"))?;
    let syl: Vec<Syllable> = if j.get("syllables").is_some() {
        serde_json::from_value(j["syllables"].clone()).map_err(|e| format!("syllables: {e}"))?
    } else {
        let g = g2p()?;
        lyrics_to_syllables(j["lyrics"].as_str().ok_or("lyrics missing")?, &g)
    };
    let sounding = raw.iter().filter(|n| n[0] > 0.0).count();
    if syl.len() != sounding {
        return Err(format!(
            "{} syllables for {sounding} notes: {}",
            syl.len(),
            syl.iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    let t0 = j["t0"].as_f64().unwrap_or(0.3);
    let beat = 60.0 / bpm;
    let mut v = new_voice(ctx, body_name, true)?;
    let first = raw.iter().find(|n| n[0] > 0.0).ok_or("no notes")?;
    let f0_init = midi_to_hz(first[0]);
    apply_source(
        &mut v,
        ctx,
        &SourcePatch {
            f0: Some(f0_init),
            voiced: Some(true),
            pressure: Some(9.0),
            tension: Some(0.55),
            ..Default::default()
        },
    );
    let (mut t, mut k, mut end) = (t0, 0usize, 0.0f64);
    let mut phrase: Vec<SungNote> = Vec::new();
    let mut start = t;
    let flush = |phrase: &mut Vec<SungNote>, start: f64, v: &mut VocalBoxVoice, end: &mut f64| {
        if phrase.is_empty() {
            return;
        }
        let plan = plan_song(
            phrase,
            &SongOptions {
                body: body_name.into(),
                bpm,
                t0: start,
                vibrato,
            },
        );
        v.schedule(&plan.gestures);
        *end = js_max(*end, plan.end);
        phrase.clear();
    };
    for n in &raw {
        if n[0] > 0.0 {
            if phrase.is_empty() {
                start = t;
            }
            phrase.push(SungNote {
                midi: n[0],
                beats: n[1],
                syl: syl[k].clone(),
            });
            k += 1;
        } else {
            flush(&mut phrase, start, &mut v, &mut end);
        }
        t += n[1] * beat;
    }
    flush(&mut phrase, start, &mut v, &mut end);
    let n = js_round(FS * (end + 0.4)) as usize;
    let mut out = vec![0f32; n];
    let (frames, rec) = if artic_out.is_some() || tele_out.is_some() {
        render_tracked(&mut v, &mut out, artic_out.is_some(), tele_out.is_some())?
    } else {
        render_blocks(&mut v, &mut out);
        (Vec::new(), None)
    };
    ctx.write_wav(&out_name, &out)?;
    println!(
        "{out_name}: {} s, peak {}, {} notes",
        fixed(n as f64 / FS, 2),
        fixed(peak(&out), 3),
        sounding
    );
    if let Some(path) = &artic_out {
        let j = serde_json::json!({"rate": ARTIC_RATE, "body": body_name, "keys": ArticKey::NAMES, "frames": frames});
        std::fs::write(path, j.to_string()).map_err(|e| format!("{path}: {e}"))?;
        println!("{path}: {} frames at {ARTIC_RATE} Hz", frames.len());
    }
    if let (Some(path), Some(r)) = (&tele_out, rec) {
        std::fs::write(path, r.bytes(bm)).map_err(|e| format!("{path}: {e}"))?;
        println!(
            "{path}: {} telemetry frames every {TELE_HOP} samples",
            r.frames
        );
    }
    Ok(())
}

/// The beatbox kit, each sound alone, and a preset pattern through the sequencer.
/// `render beatbox [outdir] [preset name | pattern] [body] [bpm] [bars]`
pub fn beatbox(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let out_dir = arg(a, 0, "out/beatbox");
    let preset_name = arg(a, 1, "Boots and cats");
    let body_name = arg(a, 2, "male");
    let bm = body_mix(body_name)?;
    let voice = || {
        let mut v = VocalBoxVoice::new(FS, bm);
        v.set_articulation(&beat_ready(body_name.into()), None);
        v.set_source(&SourcePatch {
            f0: Some(100.0),
            voiced: Some(true),
            pressure: Some(8.0),
            ..Default::default()
        });
        v.set_gate(0.0);
        v
    };
    let at = |t: f64| js_round(t * FS) as usize;

    // ---- each sound alone
    let mut player = BeatPlayer::new();
    player.body = body_name.into();
    let mut rows: Vec<String> = Vec::new();
    let n = js_round(FS * 0.5) as usize;
    let mut strip = vec![0f32; js_round(FS * 0.5 * BEAT_SOUNDS.len() as f64) as usize];
    for (idx, s) in BEAT_SOUNDS.iter().enumerate() {
        let mut v = voice();
        let hit = 0.25;
        let mut gs = Vec::new();
        player.gestures(&BeatStep::new(s.id), hit, 0.15, &mut gs);
        v.schedule(&gs);
        let mut out = vec![0f32; n];
        render_blocks(&mut v, &mut out);
        let abs = |i: usize| (out[i] as f64).abs();
        let mut pk = 0.0;
        for i in at(hit - 0.03)..at(hit + 0.08) {
            pk = js_max(pk, abs(i));
        }
        let mut on = at(hit);
        for i in at(hit - 0.04)..at(hit + 0.08) {
            if abs(i) > 0.25 * pk {
                on = i;
                break;
            }
        }
        let mut pre = 0.0;
        for i in at(hit - 0.15)..at(hit - 0.04) {
            pre = js_max(pre, abs(i));
        }
        let mut tail = 0.0;
        for i in at(hit + 0.06)..at(hit + 0.15) {
            tail = js_max(tail, abs(i));
        }
        rows.push(format!(
            "{} {} onset {} ms  peak {}  set-up {} dB  tail {} dB  lead-in {} ms",
            pad_end(s.id, 4),
            pad_end(s.name, 13),
            pad_start(&fixed((on as f64 / FS - hit) * 1000.0, 1), 6),
            fixed(pk, 3),
            fixed(20.0 * log10(js_max(1e-6, pre) / pk), 1),
            fixed(20.0 * log10(js_max(1e-6, tail) / pk), 1),
            fixed(BeatPlayer::lead_in(s.id, body_name.into()) * 1000.0, 0),
        ));
        strip[idx * n..(idx + 1) * n].copy_from_slice(&out);
        let file: String = s
            .id
            .chars()
            .map(|c| if c.is_ascii_alphabetic() { c } else { '_' })
            .collect();
        ctx.write_wav(&format!("{out_dir}/{file}.wav"), &out)?;
    }
    println!("{}", rows.join("\n"));
    ctx.write_wav(&format!("{out_dir}/kit.wav"), &strip)?;

    let preset = BEAT_PRESETS
        .iter()
        .find(|p| p.name.to_lowercase() == preset_name.to_lowercase());
    let (name, text, mut bpm, swing) = match preset {
        Some(p) => (p.name, p.text, p.bpm, p.swing),
        None if parse_beat(preset_name).iter().any(Option::is_some) => {
            ("pattern", preset_name, 100.0, 0.0)
        }
        None => (
            BEAT_PRESETS[0].name,
            BEAT_PRESETS[0].text,
            BEAT_PRESETS[0].bpm,
            BEAT_PRESETS[0].swing,
        ),
    };
    if let Some(b) = a.get(3) {
        bpm = b.parse().map_err(|_| format!("bad bpm {b:?}"))?;
    }
    let bars: f64 = match a.get(4) {
        Some(n) => n.parse().map_err(|_| format!("bad bar count {n:?}"))?,
        None => 2.0,
    };
    let mut v = voice();
    let mut seq = BeatPlayer::new();
    seq.body = body_name.into();
    seq.set_steps(&parse_beat(text));
    seq.set_tempo(bpm, swing);
    seq.start(v.clock);
    let total = js_round(
        FS * (BeatPlayer::HORIZON + bars * seq.steps().len() as f64 * seq.step_dur() + 0.3),
    ) as usize;
    let mut out = vec![0f32; total];
    for block in out.chunks_mut(128) {
        let clock = v.clock;
        seq.tick(clock, &mut |gs: &[Gesture]| v.schedule(gs));
        v.render(block);
    }
    let file = format!("{out_dir}/{}.wav", slug(name));
    ctx.write_wav(&file, &out)?;
    println!(
        "pattern \"{}\": {} s, peak {} -> {file}",
        name,
        fixed(total as f64 / FS, 2),
        fixed(peak(&out), 3)
    );
    Ok(())
}

/// A creature's sustained note and calls; level, stability and pitch.
/// `render creature [creatureId] [outdir]`
pub fn creature(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let id = arg(a, 0, "songbird");
    let out_dir = arg(a, 1, "out/creatures");
    let def = creature_by_id(id).ok_or_else(|| {
        format!(
            "unknown creature {id}; have {}",
            CREATURES
                .iter()
                .map(|c| c.id)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
    let mut v = CreatureVoice::new(def, FS, &[]);
    let mut out: Vec<f32> = Vec::new();
    let t = Instant::now();
    let render_for =
        |v: &mut CreatureVoice, out: &mut Vec<f32>, sec: f64| -> std::ops::Range<usize> {
            let n = js_round(sec * FS) as usize;
            let start = out.len();
            out.resize(start + n, 0.0);
            for block in out[start..].chunks_mut(128) {
                v.render(block);
            }
            start..start + n
        };
    let pitch_param = def.params.iter().find(|p| p.role == Some(ParamRole::Pitch));
    let voices = def.param_index("voices");
    if let Some(k) = voices {
        v.set_params(&[(k, 0.0)]);
    }
    v.set_gate(1.0);
    let sus = render_for(&mut v, &mut out, 1.2);
    v.set_gate(0.0);
    render_for(&mut v, &mut out, 0.4);
    if let Some(k) = voices {
        v.set_params(&[(k, 0.5)]);
    }
    let mut pitch_report = String::new();
    if let Some(p) = pitch_param {
        let s = sus.start + js_round(0.6 * FS) as usize;
        let frame: Vec<f64> = out[s..s + 4096].iter().map(|&x| x as f64).collect();
        let est = yin_pitch(
            &frame,
            FS,
            YinOpts {
                min_f0: Some(js_max(20.0, p.min * 0.5)),
                max_f0: Some(js_min(12000.0, p.max * 2.0)),
                ..Default::default()
            },
        );
        pitch_report = format!(
            "target {} Hz, measured {} Hz (aperiodicity {})",
            num(p.default),
            fixed(est.f0, 1),
            fixed(est.aperiodicity, 2)
        );
    }
    for call in &def.calls {
        v.play_call_id(call.id, None, 1.0);
        let dur = call.keys[call.keys.len() - 1].t + 0.5;
        render_for(&mut v, &mut out, dur);
    }
    let ms = ms_since(t);
    let total = out.len() as f64;
    let bad = out.iter().filter(|x| !x.is_finite()).count();
    let file = format!("{out_dir}/{id}.wav");
    ctx.write_wav(&file, &out)?;
    println!(
        "{}: {} s in {} ms ({}% RT, internal {} Hz)  peak {}  non-finite {bad}",
        def.name,
        fixed(total / FS, 1),
        fixed(ms, 0),
        fixed(ms / 1000.0 / (total / FS) * 100.0, 1),
        num(v.fs),
        fixed(peak(&out), 3)
    );
    println!("  {pitch_report}");
    Ok(())
}

/// The calibrated vowels through the real-time voice; LPC formants and YIN f0 from the audio.
/// `render vowels [body] [out]`
pub fn vowels(ctx: &Ctx, a: &[String]) -> Result<(), String> {
    let body_name = arg(a, 0, "male");
    let default_out = format!("out/vowels-{body_name}");
    let name = arg(a, 1, &default_out);
    let f0 = match body_name {
        "female" => 205.0,
        "child" => 260.0,
        _ => 110.0,
    };
    let mut voice = new_voice(ctx, body_name, false)?;
    apply_source(&mut voice, ctx, &SourcePatch::f0(f0));
    let list = ["i", "e", "ɛ", "æ", "ɑ", "ɔ", "o", "u", "ʌ", "ɝ"];
    let seg = 0.6;
    let mut out = vec![0f32; js_round(FS * seg * list.len() as f64) as usize];
    let table = &vowel_table()[body_name];
    let t = Instant::now();
    let block = 128;
    let mut pos = 0;
    'vowels: for ipa in list {
        voice.set_articulation(&table[ipa].artic, Some(true));
        let n = js_round(FS * seg) as usize;
        let mut i = 0;
        while i < n {
            let local = i as f64 / FS;
            voice.set_gate(if local < seg - 0.12 { 1.0 } else { 0.0 });
            let len = block.min(out.len() - pos);
            voice.render(&mut out[pos..pos + len]);
            pos += block;
            if pos >= out.len() {
                break 'vowels;
            }
            i += block;
        }
    }
    let ms = ms_since(t);
    let secs = out.len() as f64 / FS;
    println!(
        "{body_name}: rendered {} s in {} ms → {}% of real time; peak {}",
        fixed(secs, 1),
        fixed(ms, 0),
        fixed(ms / 1000.0 / secs * 100.0, 1),
        fixed(peak(&out), 3)
    );
    for (k, ipa) in list.iter().enumerate() {
        let c = js_round(FS * (k as f64 * seg + 0.3)) as usize;
        let frame = &out[c..c + 1536];
        let ds: Vec<f64> = (0..384)
            .map(|i| {
                (frame[4 * i] as f64
                    + frame[4 * i + 1] as f64
                    + frame[4 * i + 2] as f64
                    + frame[4 * i + 3] as f64)
                    / 4.0
            })
            .collect();
        let fm = estimate_formants(
            &ds,
            FS / 4.0,
            FormantOpts {
                order: Some(12),
                ..Default::default()
            },
        );
        let y: Vec<f64> = out[c..c + 2400].iter().map(|&x| x as f64).collect();
        let est = yin_pitch(
            &y,
            FS,
            YinOpts {
                min_f0: Some(60.0),
                ..Default::default()
            },
        );
        let reference = &table[*ipa].formants;
        let lpc: String = fm
            .iter()
            .take(3)
            .map(|f| pad_start(&fixed(f.freq, 0), 5))
            .collect();
        let model: String = reference
            .iter()
            .take(3)
            .map(|f| pad_start(&num(*f), 5))
            .collect();
        println!(
            "  {} f0={}  LPC {lpc}   model {model}",
            pad_end(ipa, 2),
            fixed(est.f0, 1)
        );
    }
    ctx.write_wav(&wav_name(name), &out)?;
    Ok(())
}
