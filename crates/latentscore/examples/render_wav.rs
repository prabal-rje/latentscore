use latentscore::{Score, WavOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut score = Score::from_text("quiet morning desk")
        .with_seed(42)
        .with_glide_seconds(4.0);

    let mut part_a = vec![0.0_f32; 44_100 * 4];
    score.fill(&mut part_a);

    score.steer("rainy midnight coffee shop");
    let mut part_b = vec![0.0_f32; 44_100 * 4];
    score.fill(&mut part_b);

    let mut combined = Vec::with_capacity(part_a.len() + part_b.len());
    combined.extend_from_slice(&part_a);
    combined.extend_from_slice(&part_b);

    latentscore::save_wav(
        "morning-to-rain.wav",
        &combined,
        WavOptions {
            channels: 2,
            stereo_width: 0.65,
            fade_out_seconds: 0.5,
            ..WavOptions::default()
        },
    )?;
    println!("Wrote morning-to-rain.wav ({} frames)", combined.len());
    Ok(())
}
