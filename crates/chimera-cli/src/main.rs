//! `chimera`: offline renders to WAV, speech diagnostics and table calibration over the CHIMERA
//! engine. Ports of the TypeScript tools (`tools/render-*.ts`, `speech-diagnostics.ts`,
//! `calibrate-*.ts`) with their arguments, defaults and printed summaries. Spectrograms of the
//! WAVs: `bun tools/spectrogram.ts in.wav out.png [title] [maxFreq] [windowMs]`.

mod calibrate;
mod compare;
mod diagnose;
mod js;
mod render;
mod wav;

use chimera::anatomy::human::BodyMix;
use chimera::vocal_box::EmotionPreset;
use std::process::ExitCode;

const USAGE: &str = "\
usage: chimera <command> [args] [--float] [--emotion <preset>]

  render speech  <out> [\"ipa / ipa\"] [male|female|child]
  render text    <out> [\"English text.\"] [male|female|child]
  render song    [outdir] [male|female|child]
  render notes   <in.json> <out> [male|female|child] [--artic tracks.json] [--tele voice.tele]
                 (melody file; midi <= 0 is a rest; --artic also writes the articulators at 100 Hz)
  render beatbox [outdir] [preset name | pattern] [male|female|child] [bpm] [bars]
  render creature [creatureId] [outdir]
  render vowels  [male|female|child] [out]
  syllables      \"lyrics\"    (what the language front ends read: one JSON syllable per sung note)
  diagnose       [outdir]
  compare csd    [csd-dir] [outdir] [--songs a|b|all|en001a,…] [--body female] [--jobs n]
                 [--against earlier/score.json]  (score.json: distance to the singer, and the verdict)
  compare tsp    [tsp-dir] [outdir] [--speakers FA,MA,…] [--jobs n] [--dump]
                 [--against earlier/score.json]  (distance to 23 real speakers, phone by phone)
  calibrate vowels     [male|female|child|all] [--out file]
  calibrate consonants [--out file]

Expressive drive:
  --emotion <preset>  natural, warm, tender, joyful, dramatic, melancholic,
                      angry, pressed, breathy, whisper, fry, operatic, pop

Output names are stems (<out>.wav). --float writes 32-bit float WAVs instead of 16-bit PCM.
Calibration writes the tables the engine embeds, in crates/chimera/src/phonetics/.";

/// The output rate of every render.
pub const FS: f64 = 48000.0;

/// The repository root (data files and the generated tables are found from here).
pub fn repo_path(rel: &str) -> String {
    format!("{}/../../{rel}", env!("CARGO_MANIFEST_DIR"))
}

/// Options shared by every command.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx {
    /// write 32-bit float WAVs
    pub float: bool,
    /// optional expressive emotion preset
    pub emotion: Option<EmotionPreset>,
}

impl Ctx {
    pub fn write_wav(&self, path: &str, x: &[f32]) -> Result<(), String> {
        wav::write(path, x, FS as u32, self.float).map_err(|e| format!("{path}: {e}"))
    }
}

/// The app's body mix for a body preset (female 0..1, child 0..1), as `src/ui/body.ts` PRESETS.
pub fn body_mix(name: &str) -> Result<BodyMix, String> {
    match name {
        "male" => Ok(BodyMix::new(0.0, 0.0)),
        "female" => Ok(BodyMix::new(1.0, 0.0)),
        "child" => Ok(BodyMix::new(0.0, 1.0)),
        _ => Err(format!("unknown body {name:?}; have male, female, child")),
    }
}

/// `<out>.wav` for an output stem (a trailing `.wav` is accepted too).
pub fn wav_name(stem: &str) -> String {
    format!("{}.wav", stem.strip_suffix(".wav").unwrap_or(stem))
}

pub fn peak(x: &[f32]) -> f64 {
    x.iter()
        .fold(0.0f64, |p, &v| chimera::math::js_max(p, (v as f64).abs()))
}

/// Positional argument `i`, or a default.
pub fn arg<'a>(a: &'a [String], i: usize, default: &'a str) -> &'a str {
    a.get(i).map(String::as_str).unwrap_or(default)
}

fn parse_global_flags(raw_args: Vec<String>) -> Result<(Ctx, Vec<String>), String> {
    let mut ctx = Ctx::default();
    let mut args = Vec::with_capacity(raw_args.len());
    let mut it = raw_args.into_iter();
    while let Some(a) = it.next() {
        if a == "--float" {
            ctx.float = true;
        } else if a == "--emotion" {
            let val = it.next().ok_or_else(|| {
                format!(
                    "--emotion requires a preset name ({})",
                    EmotionPreset::NAMES.join(", ")
                )
            })?;
            ctx.emotion = Some(EmotionPreset::from_name(&val)?);
        } else if let Some(val) = a.strip_prefix("--emotion=") {
            if val.is_empty() {
                return Err(format!(
                    "--emotion requires a preset name ({})",
                    EmotionPreset::NAMES.join(", ")
                ));
            }
            ctx.emotion = Some(EmotionPreset::from_name(val)?);
        } else {
            args.push(a);
        }
    }
    Ok((ctx, args))
}

fn run(raw_args: Vec<String>) -> Result<(), String> {
    let (ctx, args) = parse_global_flags(raw_args)?;
    let cmd: Vec<&str> = args.iter().take(2).map(String::as_str).collect();
    match cmd.as_slice() {
        ["render", kind, ..] => {
            let rest = &args[2..];
            match *kind {
                "speech" => render::speech(&ctx, rest),
                "text" => render::text(&ctx, rest),
                "song" => render::song(&ctx, rest),
                "notes" => render::notes(&ctx, rest),
                "beatbox" => render::beatbox(&ctx, rest),
                "creature" => render::creature(&ctx, rest),
                "vowels" => render::vowels(&ctx, rest),
                _ => Err(format!("unknown render kind {kind:?}\n\n{USAGE}")),
            }
        }
        ["compare", "csd", ..] => compare::csd(&ctx, &args[2..]),
        ["compare", "tsp", ..] => compare::tsp::run(&ctx, &args[2..]),
        ["diagnose", ..] => diagnose::run(&ctx, &args[1..]),
        ["syllables", text] => render::g2p()
            .map(|g| chimera::phonetics::syllables::lyrics_to_syllables(text, &g))
            .and_then(|s| serde_json::to_string(&s).map_err(|e| e.to_string()))
            .map(|j| println!("{j}")),
        ["calibrate", "vowels", ..] => calibrate::vowels(&args[2..]),
        ["calibrate", "consonants", ..] => calibrate::consonants(&args[2..]),
        ["help" | "--help" | "-h", ..] => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.to_string()),
    }
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
