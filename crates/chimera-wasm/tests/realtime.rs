//! The audio-thread ABI allocates nothing once an engine exists: from idle into the first labelled
//! speech gestures, telemetry snapshots, the sequencer and pad hits, releases and parameter
//! changes; and, after a creature switch (which may allocate), the creature's rendering,
//! telemetry, params and calls. Driven through the same C ABI the worklet uses.
use chimera::phonetics::ipa::tokenize_ipa;
use chimera::phonetics::planner::{PlanOptions, plan_utterance};
use chimera_wasm::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;
thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|n| n.set(n.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static A: Counting = Counting;
fn allocs() -> usize {
    ALLOCS.with(|n| n.get())
}

fn write_step(rec: &mut [u8], id: &str) {
    rec[0] = id.len() as u8;
    rec[1..1 + id.len()].copy_from_slice(id.as_bytes());
    rec[32..40].copy_from_slice(&f64::NAN.to_le_bytes());
    rec[40..48].copy_from_slice(&f64::NAN.to_le_bytes());
}

#[test]
fn the_audio_thread_abi_does_not_allocate() {
    let h = engine_new(48000.0);
    // planned off the audio thread
    let plan = plan_utterance(&tokenize_ipa("ˈhɛloʊ ˈwɝld ʃi sɛlz"), &PlanOptions { body: "male".into(), t0: 0.0, f0: Some(110.0), ..Default::default() });
    unsafe {
        let render = |blocks: usize| {
            for b in 0..blocks {
                engine_render(h, 128);
                if b % 6 == 0 {
                    engine_telemetry(h);
                    engine_beat_step(h);
                }
            }
        };
        render(40); // idle for a while: nothing labelled has been seen yet
        let before = allocs();
        let rec = std::slice::from_raw_parts_mut(engine_records(h), plan.gestures.len() * GESTURE_RECORD);
        for (i, g) in plan.gestures.iter().enumerate() {
            write_gesture(g, &mut rec[i * GESTURE_RECORD..(i + 1) * GESTURE_RECORD]);
        }
        engine_schedule(h, plan.gestures.len(), 0.03, 1);
        render(400); // speech: labelled gestures in every telemetry snapshot
        let vals = std::slice::from_raw_parts_mut(engine_values(h), 32);
        vals.fill(f64::NAN);
        vals[0] = 0.8; // jaw
        engine_set_artic(h, 2);
        vals.fill(f64::NAN);
        vals[0] = 140.0; // f0
        engine_set_source(h);
        engine_set_gate(h, 1.0);
        let steps = std::slice::from_raw_parts_mut(engine_records(h), 4 * STEP_RECORD);
        for (i, id) in ["B", "t", "Pf", "t"].iter().enumerate() {
            write_step(&mut steps[i * STEP_RECORD..(i + 1) * STEP_RECORD], id);
        }
        engine_set_beat(h, 1, 4, 120.0, 0.1, 1);
        render(300);
        write_step(&mut steps[..STEP_RECORD], "K");
        engine_beat_hit(h);
        engine_release(h, 7.0);
        render(100);
        engine_set_beat(h, 0, 0, f64::NAN, f64::NAN, 0);
        engine_stop_speech(h);
        render(50);
        assert_eq!(allocs() - before, 0, "the human voice's audio path allocated");

        // a creature switch allocates; what follows must not (every creature)
        for def in chimera::creatures::CREATURES.iter() {
            let req = format!(r#"{{"id":"{}","params":{{}}}}"#, def.id).into_bytes();
            assert_eq!(engine_set_creature(h, req.as_ptr(), req.len()), 0);
            let before = allocs();
            for b in 0..400 {
                engine_render(h, 128);
                if b % 6 == 0 {
                    assert!(!engine_creature_telemetry(h).is_null());
                }
                match b {
                    30 => engine_set_gate(h, 1.0),
                    50 => engine_play_call(h, 0, f64::NAN, f64::NAN),
                    150 => engine_creature_param(h, 0, def.params[0].default * 1.1),
                    200 => engine_play_call(h, def.calls.len() - 1, 1.2 * def.params[0].default, 0.8),
                    300 => engine_stop_call(h),
                    _ => {}
                }
            }
            assert_eq!(allocs() - before, 0, "{}: the creature's audio path allocated", def.id);
        }
        engine_free(h);
    }
}

#[test]
fn gesture_wire_v2_roundtrips_expanded_source_keys() {
    use chimera::control::score::{ControlKey, Gesture, Targets};
    assert_eq!(LAYOUT_VERSION, 2);
    assert_eq!(GESTURE_RECORD, 304);
    let mut rec = [0u8; GESTURE_RECORD];
    let g = Gesture {
        t0: 0.125,
        t1: 0.625,
        ramp_in: 0.04,
        ramp_out: 0.05,
        strength: 2.5,
        speed: Some(1.1),
        release_hold: Some(0.02),
        tag: Some(7.0),
        targets: Targets::default()
            .with(ControlKey::TENSION, 0.72)
            .with(ControlKey::FRY, 0.35)
            .with(ControlKey::BRIGHTNESS, 0.18)
            .with(ControlKey::CHEST, 0.94),
        label: Some("chest_tone".into()),
    };
    write_gesture(&g, &mut rec);
    let decoded = read_gesture(&rec);
    assert_eq!(decoded.targets.get(ControlKey::TENSION), Some(0.72));
    assert_eq!(decoded.targets.get(ControlKey::FRY), Some(0.35));
    assert_eq!(decoded.targets.get(ControlKey::BRIGHTNESS), Some(0.18));
    assert_eq!(decoded.targets.get(ControlKey::CHEST), Some(0.94));
    assert_eq!(decoded.label.as_ref().map(|l| l.as_str()), Some("chest_tone"));
}

