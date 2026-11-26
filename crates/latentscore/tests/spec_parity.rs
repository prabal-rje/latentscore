use latentscore::{
    apply_adsr, apply_humanize, apply_master_hall_reverb, apply_update, assemble,
    encode_wav_bytes, fade_into_nothingness, generate_sine, mood_from_text, render, to_internal,
    FieldOrStep, ModeName, MusicConfig, MusicConfigUpdate, RootNote, Score, SeededRng, SpaceLabel,
    Step, TempoLabel, TextureStyle, WavOptions, SAMPLE_RATE,
};
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

fn crate_spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spec")
}

fn repo_spec_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("spec")
}

#[test]
fn embedded_specs_match_root_spec_bytes_when_in_monorepo() {
    let root = repo_spec_dir();
    if !root.exists() {
        return;
    }
    for file in ["mood_lexicon.json", "golden_cases.json"] {
        let root_bytes = fs::read(root.join(file)).unwrap();
        let crate_bytes = fs::read(crate_spec_dir().join(file)).unwrap();
        assert_eq!(crate_bytes, root_bytes, "spec drift in {file}");
    }
}

#[test]
fn mood_from_text_matches_spec_golden_texts() {
    let text = fs::read_to_string(crate_spec_dir().join("mood_lexicon.json")).unwrap();
    let spec: Value = serde_json::from_str(&text).unwrap();
    for case in spec["golden_texts"].as_array().unwrap() {
        let prompt = case["text"].as_str().unwrap();
        let cfg = mood_from_text(prompt);
        let serialized = serde_json::to_value(&cfg).unwrap();
        for (k, expected) in case["expect"].as_object().unwrap() {
            assert_eq!(&serialized[k], expected, "prompt={prompt:?} field={k}");
        }
    }
}

#[test]
fn config_update_steps_saturate_and_deny_unknown_fields() {
    let base = MusicConfig::default();
    let upd = MusicConfigUpdate {
        tempo: Some(FieldOrStep::Delta(Step(2))),
        space: Some(FieldOrStep::Delta(Step(-1))),
        density: Some(FieldOrStep::Delta(Step(i32::MAX))),
        ..Default::default()
    };
    let next = apply_update(&base, &upd);
    assert_eq!(next.tempo, TempoLabel::VeryFast);
    assert_eq!(next.space, SpaceLabel::Small);
    assert_eq!(next.density, 6);

    let min_upd = MusicConfigUpdate {
        tempo: Some(FieldOrStep::Delta(Step(i32::MIN))),
        density: Some(FieldOrStep::Delta(Step(i32::MIN))),
        ..Default::default()
    };
    let low = apply_update(&base, &min_upd);
    assert_eq!(low.tempo, TempoLabel::VerySlow);
    assert_eq!(low.density, 2);

    let bad_json = r#"{"unknown_field": 123}"#;
    assert!(serde_json::from_str::<MusicConfigUpdate>(bad_json).is_err());
}

#[test]
fn adsr_cases_always_start_and_end_at_zero() {
    let text = fs::read_to_string(crate_spec_dir().join("golden_cases.json")).unwrap();
    let spec: Value = serde_json::from_str(&text).unwrap();
    for c in spec["adsr_cases"].as_array().unwrap() {
        let n = c["samples"].as_u64().unwrap() as usize;
        let ones = vec![1.0_f64; n];
        let out = apply_adsr(
            &ones,
            c["attack"].as_f64().unwrap(),
            c["decay"].as_f64().unwrap(),
            c["sustain"].as_f64().unwrap(),
            c["release"].as_f64().unwrap(),
            SAMPLE_RATE,
        );
        assert_eq!(out.len(), n);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[n - 1], 0.0);
    }
}

#[test]
fn humanize_introduces_zero_hf_hiss() {
    let text = fs::read_to_string(crate_spec_dir().join("golden_cases.json")).unwrap();
    let spec: Value = serde_json::from_str(&text).unwrap();
    let tol = spec["tolerances"]["humanize_hf_ratio_above_4khz"]
        .as_f64()
        .unwrap();

    let tone = generate_sine(220.0, 1.0, SAMPLE_RATE, 0.5, 0.0);
    let mut rng = SeededRng::new(7);
    let out = apply_humanize(&tone, 0.8, SAMPLE_RATE, &mut rng);

    let lap_energy = |x: &[f64]| -> (f64, f64) {
        let mut tot = 0.0;
        let mut diff = 0.0;
        for i in 1..(x.len() - 1) {
            tot += x[i] * x[i];
            let l = x[i + 1] - 2.0 * x[i] + x[i - 1];
            diff += l * l;
        }
        (tot, diff)
    };
    let (_, orig_diff) = lap_energy(&tone);
    let (tot, new_diff) = lap_energy(&out);
    let ratio = (new_diff - orig_diff).abs() / (tot + 1e-12);
    assert!(ratio < tol * 10.0, "ratio={ratio}");
}

#[test]
fn calibrated_hall_reverb_keeps_pure_tone_below_clipping() {
    let text = fs::read_to_string(crate_spec_dir().join("golden_cases.json")).unwrap();
    let spec: Value = serde_json::from_str(&text).unwrap();
    let max_peak = spec["tolerances"]["hall_reverb_max_tone_peak"]
        .as_f64()
        .unwrap();
    let tone = generate_sine(330.0, 3.0, SAMPLE_RATE, 0.35, 0.0);
    let wet = apply_master_hall_reverb(&tone, 0.95, SAMPLE_RATE);
    let peak = wet.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    assert!(peak <= max_peak, "peak={peak}");
}

#[test]
fn golden_cases_audio_invariants_pass() {
    let text = fs::read_to_string(crate_spec_dir().join("golden_cases.json")).unwrap();
    let spec: Value = serde_json::from_str(&text).unwrap();
    let tol = &spec["tolerances"];
    let min_peak = tol["min_audible_peak"].as_f64().unwrap();
    let max_peak = tol["max_peak"].as_f64().unwrap();
    let max_dc = tol["max_abs_dc"].as_f64().unwrap();
    let max_jump = tol["max_sample_jump"].as_f64().unwrap();
    let seam_tol = tol["seam_max_diff"].as_f64().unwrap();

    for item in spec["render_cases"].as_array().unwrap() {
        let name = item["name"].as_str().unwrap();
        let seed = item["seed"].as_u64().unwrap() as u32;
        let duration = item["duration"].as_f64().unwrap();
        let cfg: MusicConfig = serde_json::from_value(item["config"].clone()).unwrap();
        let internal = to_internal(&cfg);

        let audio = assemble(&internal, duration, true, seed, 0.0);
        let mut peak = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut jump = 0.0_f64;
        for (i, &v) in audio.iter().enumerate() {
            assert!(v.is_finite(), "{name} non-finite sample");
            peak = peak.max(v.abs());
            sum += v;
            if i > 0 {
                jump = jump.max((v - audio[i - 1]).abs());
            }
        }
        assert!(peak >= min_peak && peak <= max_peak, "{name} peak={peak}");
        assert!((sum / (audio.len() as f64)).abs() < max_dc, "{name} dc");
        assert!(jump < max_jump, "{name} jump={jump}");

        if let Some(t2) = item["t_offset_second"].as_f64() {
            let audio2 = assemble(&internal, duration, true, seed, t2);
            assert!((audio2[0] - audio[audio.len() - 1]).abs() < seam_tol);
        }
    }
}

#[test]
fn score_glide_converges_discrete_fields_over_short_blocks() {
    let mut score = Score::from_text("quiet morning desk")
        .with_seed(11)
        .with_glide_seconds(3.0)
        .with_block_seconds(0.5);
    score.steer("rainy midnight coffee shop");

    // 7 blocks x 0.5s = 3.5s > 3.0s glide: discrete fields must converge completely.
    for _ in 0..7 {
        let blk = score.next_block(None);
        assert_eq!(blk.len(), (0.5 * (SAMPLE_RATE as f64)).round() as usize);
    }
    let cur = score.current_config();
    assert_eq!(cur.root, RootNote::D);
    assert_eq!(cur.mode, ModeName::Dorian);
    assert_eq!(cur.texture, TextureStyle::VinylCrackle);
}

#[test]
fn score_cross_block_continuity_and_nudge_preserve_continuous_state() {
    let mut score = Score::from_text("quiet morning desk")
        .with_seed(19)
        .with_glide_seconds(2.0)
        .with_block_seconds(1.0);

    let b1 = score.next_block(None);
    score.steer("rainy midnight coffee shop");
    let b2 = score.next_block(None);
    let mut b3 = vec![0.0_f32; 22_050];
    score.fill(&mut b3);

    let mut stitched = Vec::with_capacity(b1.len() + b2.len() + b3.len());
    stitched.extend_from_slice(&b1);
    stitched.extend_from_slice(&b2);
    stitched.extend_from_slice(&b3);
    let mut max_jump = 0.0_f32;
    for pair in stitched.windows(2) {
        max_jump = max_jump.max((pair[1] - pair[0]).abs());
    }
    assert!(max_jump < 0.25, "cross-block click detected: max_jump={max_jump}");

    // Drift produces a continuous float tempo; nudging space must not quantize tempo.
    score.drift_next();
    let _ = score.next_block(Some(2.5));
    let before_tempo = score.internal_config().tempo;
    score.nudge(&MusicConfigUpdate {
        space: Some(FieldOrStep::Delta(Step(1))),
        ..Default::default()
    });
    let _ = score.next_block(Some(2.5));
    assert!((score.internal_config().tempo - before_tempo).abs() < 1e-6);
}

#[test]
fn stereo_wav_ends_at_exact_zero_on_both_channels() {
    let mut score = Score::from_text("warm cedar zen meditation")
        .with_seed(42)
        .with_block_seconds(1.0);
    let mut audio = vec![0.0_f32; SAMPLE_RATE as usize];
    score.fill(&mut audio);

    let wav = encode_wav_bytes(
        &audio,
        WavOptions {
            channels: 2,
            stereo_width: 0.85,
            fade_in_seconds: 0.02,
            fade_out_seconds: 0.25,
            ..WavOptions::default()
        },
    );
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(wav.len(), 44 + audio.len() * 4);

    let n = wav.len();
    let last_l = i16::from_le_bytes([wav[n - 4], wav[n - 3]]);
    let last_r = i16::from_le_bytes([wav[n - 2], wav[n - 1]]);
    assert_eq!(last_l, 0, "stereo WAV left channel must end at 0");
    assert_eq!(last_r, 0, "stereo WAV right channel must end at 0");

    let ones = vec![1.0_f64; 4410];
    let faded = fade_into_nothingness(&ones, 0.05, SAMPLE_RATE);
    assert_eq!(faded[faded.len() - 1], 0.0);

    let one_shot = render("warm cedar zen meditation", 1.0);
    assert_eq!(one_shot.len(), SAMPLE_RATE as usize);
}
