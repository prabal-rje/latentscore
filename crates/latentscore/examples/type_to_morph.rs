use latentscore::Score;
use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting LatentScore live playback. Type a mood and press Enter (or 'quit'):");
    let handle = Score::from_text("quiet morning desk")
        .with_glide_seconds(3.5)
        .play()?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let text = line?;
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("quit") || trimmed.eq_ignore_ascii_case("exit") {
            break;
        }
        if trimmed.eq_ignore_ascii_case("drift") {
            handle.drift_next();
            println!("  -> drifting smoothly to a neighboring mood");
        } else if !trimmed.is_empty() {
            handle.steer(trimmed);
            println!("  -> steering toward \"{trimmed}\"");
        }
        stdout.flush()?;
    }

    handle.stop(0.5);
    println!("Stopped cleanly.");
    Ok(())
}
