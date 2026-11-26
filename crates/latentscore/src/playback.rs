//! Real-time audio-safe `cpal` playback with a lock-free SPSC ring buffer and
//! automatic 44.1 kHz -> output-device sample-rate resampling.
//!
//! Safety guarantees on the OS audio callback thread:
//! - Zero mutex locks.
//! - Zero heap allocations.
//! - Zero blocking syscalls (underruns cleanly zero-fill).

use crate::config::{MusicConfig, MusicConfigUpdate, SAMPLE_RATE};
use crate::dsp::fade_into_nothingness;
use crate::score::Score;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
use std::cell::UnsafeCell;
use std::f64::consts::PI;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const RING_CAPACITY: usize = 1 << 17; // 131,072 samples (~2.7 s @ 48 kHz)

/// Error returned when starting or controlling live `cpal` playback.
#[derive(Debug)]
pub struct PlaybackError(pub String);

impl fmt::Display for PlaybackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PlaybackError {}

/// Backing storage shared exclusively between one [`SpscProducer`] and one [`SpscConsumer`].
struct SpscInner {
    buf: Box<[UnsafeCell<f32>]>,
    mask: usize,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
}

// SAFETY: `SpscInner` is never exposed directly; it is split once by `spsc_channel`
// into one non-Clone `SpscProducer` (`&mut self` writes) and one non-Clone
// `SpscConsumer` (`&mut self` reads). Producer only writes `[write_pos, read_pos + cap)`
// and consumer only reads `[read_pos, write_pos)`, synchronized with Acquire/Release.
unsafe impl Sync for SpscInner {}
unsafe impl Send for SpscInner {}

pub(crate) struct SpscProducer {
    inner: Arc<SpscInner>,
}

pub(crate) struct SpscConsumer {
    inner: Arc<SpscInner>,
}

pub(crate) fn spsc_channel(capacity_pow2: usize) -> (SpscProducer, SpscConsumer) {
    let cap = capacity_pow2.next_power_of_two().max(1024);
    let mut vec = Vec::with_capacity(cap);
    for _ in 0..cap {
        vec.push(UnsafeCell::new(0.0_f32));
    }
    let inner = Arc::new(SpscInner {
        buf: vec.into_boxed_slice(),
        mask: cap - 1,
        write_pos: AtomicUsize::new(0),
        read_pos: AtomicUsize::new(0),
    });
    (
        SpscProducer {
            inner: Arc::clone(&inner),
        },
        SpscConsumer { inner },
    )
}

impl SpscProducer {
    pub(crate) fn len(&self) -> usize {
        let write = self.inner.write_pos.load(Ordering::Relaxed);
        let read = self.inner.read_pos.load(Ordering::Acquire);
        write.wrapping_sub(read)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Push up to `src.len()` samples, returning the count pushed.
    pub(crate) fn push_slice(&mut self, src: &[f32]) -> usize {
        let write = self.inner.write_pos.load(Ordering::Relaxed);
        let read = self.inner.read_pos.load(Ordering::Acquire);
        let used = write.wrapping_sub(read);
        let free = self.inner.buf.len().saturating_sub(used);
        let n = src.len().min(free);
        for (i, &sample) in src.iter().take(n).enumerate() {
            let idx = write.wrapping_add(i) & self.inner.mask;
            // SAFETY: Single `SpscProducer` has exclusive write access to `[write, read + cap)`.
            unsafe {
                *self.inner.buf[idx].get() = sample;
            }
        }
        self.inner
            .write_pos
            .store(write.wrapping_add(n), Ordering::Release);
        n
    }
}

impl SpscConsumer {
    /// Pop one sample on the OS audio callback thread without locking or allocating.
    #[inline]
    pub(crate) fn pop_one(&mut self) -> Option<f32> {
        let read = self.inner.read_pos.load(Ordering::Relaxed);
        let write = self.inner.write_pos.load(Ordering::Acquire);
        if read == write {
            return None;
        }
        let idx = read & self.inner.mask;
        // SAFETY: Single `SpscConsumer` has exclusive read access to `[read, write)`.
        let val = unsafe { *self.inner.buf[idx].get() };
        self.inner
            .read_pos
            .store(read.wrapping_add(1), Ordering::Release);
        Some(val)
    }
}

/// Linear streaming resampler converting 44.1 kHz synthesis to arbitrary device rates
/// (e.g., 48 kHz or 22.05 kHz), with a gentle 1-pole pre-filter when downsampling.
struct LinearResampler {
    ratio: f64, // src_rate / dst_rate
    phase: f64,
    prev: f32,
    has_prev: bool,
    downsample_alpha: Option<f32>,
    lp_state: f32,
    filtered_scratch: Vec<f32>,
}

impl LinearResampler {
    fn new(src_rate: u32, dst_rate: u32) -> Self {
        let safe_dst = dst_rate.max(8_000) as f64;
        let ratio = (src_rate as f64) / safe_dst;
        let downsample_alpha = if ratio > 1.0001 {
            let cutoff_norm = (0.42 / ratio).clamp(0.02, 0.48);
            let alpha = (1.0 - (-2.0 * PI * cutoff_norm).exp()).clamp(0.05, 1.0);
            Some(alpha as f32)
        } else {
            None
        };
        Self {
            ratio,
            phase: 0.0,
            prev: 0.0,
            has_prev: false,
            downsample_alpha,
            lp_state: 0.0,
            filtered_scratch: Vec::new(),
        }
    }

    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        out.clear();
        if input.is_empty() {
            return;
        }
        if (self.ratio - 1.0).abs() < 1e-9 {
            out.extend_from_slice(input);
            self.prev = input[input.len() - 1];
            self.has_prev = true;
            return;
        }

        let src: &[f32] = if let Some(alpha) = self.downsample_alpha {
            self.filtered_scratch.clear();
            self.filtered_scratch.reserve(input.len());
            for &x in input {
                self.lp_state += alpha * (x - self.lp_state);
                self.filtered_scratch.push(self.lp_state);
            }
            &self.filtered_scratch
        } else {
            input
        };

        if !self.has_prev {
            self.prev = src[0];
            self.has_prev = true;
        }
        let len_f = src.len() as f64;
        while self.phase < len_f {
            let idx = self.phase.floor() as isize;
            let frac = (self.phase - (idx as f64)) as f32;
            let s0 = if idx <= 0 {
                self.prev
            } else {
                src[(idx - 1) as usize]
            };
            let s1 = src[(idx as usize).min(src.len() - 1)];
            out.push(s0 + (s1 - s0) * frac);
            self.phase += self.ratio;
        }
        self.phase -= len_f;
        self.prev = src[src.len() - 1];
    }
}

enum Command {
    SteerText(String),
    SteerConfig(Box<MusicConfig>),
    Nudge(Box<MusicConfigUpdate>),
    DriftNext,
    Stop { fade_seconds: f64 },
}

fn handle_command(cmd: Command, score: &mut Score) -> Option<f64> {
    match cmd {
        Command::SteerText(text) => {
            let _ = score.steer(&text);
            None
        }
        Command::SteerConfig(cfg) => {
            let _ = score.steer_config(*cfg);
            None
        }
        Command::Nudge(upd) => {
            let _ = score.nudge(&upd);
            None
        }
        Command::DriftNext => {
            let _ = score.drift_next();
            None
        }
        Command::Stop { fade_seconds } => Some(fade_seconds),
    }
}

struct RunningGuard(Arc<AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Thread-safe (`Send + Sync`) handle controlling a live background `cpal` stream.
pub struct PlaybackHandle {
    tx: mpsc::Sender<Command>,
    running: Arc<AtomicBool>,
    // Stream is kept alive on the producer thread so PlaybackHandle is Send + Sync on every OS.
    producer: Option<JoinHandle<()>>,
}

impl PlaybackHandle {
    /// Steer the live music smoothly toward a text prompt.
    pub fn steer(&self, prompt: impl Into<String>) {
        let _ = self.tx.send(Command::SteerText(prompt.into()));
    }

    /// Steer the live music smoothly toward an explicit [`MusicConfig`].
    pub fn steer_config(&self, cfg: MusicConfig) {
        let _ = self.tx.send(Command::SteerConfig(Box::new(cfg)));
    }

    /// Apply absolute fields or relative [`crate::Step`] adjustments live.
    pub fn nudge(&self, update: MusicConfigUpdate) {
        let _ = self.tx.send(Command::Nudge(Box::new(update)));
    }

    /// Glide one step along the similarity / harmonic gradient.
    pub fn drift_next(&self) {
        let _ = self.tx.send(Command::DriftNext);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Smoothly fade playback into silence and stop the audio stream.
    pub fn stop(mut self, fade_out_seconds: f64) {
        self.stop_inner(fade_out_seconds);
    }

    fn stop_inner(&mut self, fade_out_seconds: f64) {
        let safe_fade = if fade_out_seconds.is_finite() {
            fade_out_seconds.clamp(0.04, 10.0)
        } else {
            0.25
        };
        let _ = self.tx.send(Command::Stop {
            fade_seconds: safe_fade,
        });
        if let Some(handle) = self.producer.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PlaybackHandle {
    fn drop(&mut self) {
        self.stop_inner(0.25);
    }
}

fn build_typed_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    mut consumer: SpscConsumer,
    stream_failed: Arc<AtomicBool>,
) -> Result<Stream, PlaybackError>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = (config.channels as usize).max(1);
    let err_fn = move |err| {
        stream_failed.store(true, Ordering::Release);
        eprintln!("latentscore cpal stream error: {err}");
    };
    device
        .build_output_stream(
            config,
            move |data: &mut [T], _| {
                for frame in data.chunks_mut(channels) {
                    let raw = consumer.pop_one().unwrap_or(0.0).clamp(-1.0, 1.0);
                    let converted = T::from_sample(raw);
                    for sample in frame.iter_mut() {
                        *sample = converted;
                    }
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| PlaybackError(format!("Failed to build output stream: {e}")))
}

fn build_output_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    format: SampleFormat,
    consumer: SpscConsumer,
    stream_failed: Arc<AtomicBool>,
) -> Result<Stream, PlaybackError> {
    match format {
        SampleFormat::F32 => build_typed_stream::<f32>(device, config, consumer, stream_failed),
        SampleFormat::F64 => build_typed_stream::<f64>(device, config, consumer, stream_failed),
        SampleFormat::I8 => build_typed_stream::<i8>(device, config, consumer, stream_failed),
        SampleFormat::I16 => build_typed_stream::<i16>(device, config, consumer, stream_failed),
        SampleFormat::I32 => build_typed_stream::<i32>(device, config, consumer, stream_failed),
        SampleFormat::U8 => build_typed_stream::<u8>(device, config, consumer, stream_failed),
        SampleFormat::U16 => build_typed_stream::<u16>(device, config, consumer, stream_failed),
        SampleFormat::U32 => build_typed_stream::<u32>(device, config, consumer, stream_failed),
        other => Err(PlaybackError(format!(
            "Unsupported cpal output sample format: {other:?}"
        ))),
    }
}

fn push_with_deadline(
    producer: &mut SpscProducer,
    samples: &[f32],
    stream_failed: &AtomicBool,
    deadline: Instant,
) {
    let mut offset = 0_usize;
    while offset < samples.len() {
        if stream_failed.load(Ordering::Acquire) || Instant::now() >= deadline {
            break;
        }
        let pushed = producer.push_slice(&samples[offset..]);
        offset += pushed;
        if offset < samples.len() {
            thread::sleep(Duration::from_millis(5));
        }
    }
}

pub(crate) fn start_playback(mut score: Score) -> Result<PlaybackHandle, PlaybackError> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), PlaybackError>>(1);
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);

    let producer_thread = thread::Builder::new()
        .name("latentscore-audio".to_string())
        .spawn(move || {
            let _running_guard = RunningGuard(running_clone);
            let host = cpal::default_host();
            let Some(device) = host.default_output_device() else {
                let _ = ready_tx.send(Err(PlaybackError("No default audio output device".into())));
                return;
            };
            let supported = match device.default_output_config() {
                Ok(c) => c,
                Err(e) => {
                    let _ = ready_tx.send(Err(PlaybackError(format!(
                        "No default output config: {e}"
                    ))));
                    return;
                }
            };
            let sample_format = supported.sample_format();
            let stream_config: StreamConfig = supported.into();
            let dst_rate = stream_config.sample_rate.0.max(8_000);

            let (mut producer, consumer) = spsc_channel(RING_CAPACITY);
            let stream_failed = Arc::new(AtomicBool::new(false));
            let stream = match build_output_stream(
                &device,
                &stream_config,
                sample_format,
                consumer,
                Arc::clone(&stream_failed),
            ) {
                Ok(s) => s,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };

            let mut resampler = LinearResampler::new(SAMPLE_RATE, dst_rate);
            let mut src_chunk = vec![0.0_f32; 1024];
            let mut dst_chunk = Vec::with_capacity(2048);
            // Keep ~200 ms buffered ahead so steering and stop fades respond quickly with zero underruns.
            let target_fill = ((dst_rate as usize) / 5).clamp(4096, RING_CAPACITY / 2);

            while producer.len() < target_fill {
                score.fill(&mut src_chunk);
                resampler.process(&src_chunk, &mut dst_chunk);
                let pushed = producer.push_slice(&dst_chunk);
                if pushed < dst_chunk.len() {
                    break;
                }
            }

            if let Err(e) = stream.play() {
                let _ = ready_tx.send(Err(PlaybackError(format!("Failed to start stream: {e}"))));
                return;
            }
            let _ = ready_tx.send(Ok(()));

            let mut stop_fade: Option<f64> = None;
            let mut unpushed_tail: Vec<f32> = Vec::new();

            while stop_fade.is_none() && !stream_failed.load(Ordering::Acquire) {
                while let Ok(cmd) = cmd_rx.try_recv() {
                    if let Some(fade) = handle_command(cmd, &mut score) {
                        stop_fade = Some(fade);
                        break;
                    }
                }
                if stop_fade.is_some() {
                    break;
                }

                // Wait for buffer headroom while handling every incoming command immediately.
                while producer.len() >= target_fill
                    && stop_fade.is_none()
                    && !stream_failed.load(Ordering::Acquire)
                {
                    match cmd_rx.recv_timeout(Duration::from_millis(5)) {
                        Ok(cmd) => {
                            if let Some(fade) = handle_command(cmd, &mut score) {
                                stop_fade = Some(fade);
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            stop_fade = Some(0.25);
                            break;
                        }
                    }
                }
                if stop_fade.is_some() || stream_failed.load(Ordering::Acquire) {
                    break;
                }

                score.fill(&mut src_chunk);
                resampler.process(&src_chunk, &mut dst_chunk);
                let mut offset = 0_usize;
                while offset < dst_chunk.len() && !stream_failed.load(Ordering::Acquire) {
                    let pushed = producer.push_slice(&dst_chunk[offset..]);
                    offset += pushed;
                    if offset < dst_chunk.len() {
                        match cmd_rx.recv_timeout(Duration::from_millis(5)) {
                            Ok(cmd) => {
                                if let Some(fade) = handle_command(cmd, &mut score) {
                                    stop_fade = Some(fade);
                                    unpushed_tail.extend_from_slice(&dst_chunk[offset..]);
                                    break;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => {
                                stop_fade = Some(0.25);
                                unpushed_tail.extend_from_slice(&dst_chunk[offset..]);
                                break;
                            }
                        }
                    }
                }
            }

            if let Some(fade_sec) = stop_fade {
                if !stream_failed.load(Ordering::Acquire) {
                    let safe_fade = fade_sec.clamp(0.04, 10.0);
                    let deadline = Instant::now() + Duration::from_secs_f64(safe_fade + 1.5);
                    if !unpushed_tail.is_empty() {
                        push_with_deadline(&mut producer, &unpushed_tail, &stream_failed, deadline);
                    }
                    let fade_frames =
                        ((safe_fade * (SAMPLE_RATE as f64)).round() as usize).max(256);
                    let mut tail_f32 = vec![0.0_f32; fade_frames];
                    score.fill(&mut tail_f32);
                    let tail_f64: Vec<f64> = tail_f32.iter().map(|&s| s as f64).collect();
                    let faded_f64 = fade_into_nothingness(&tail_f64, safe_fade, SAMPLE_RATE);
                    let faded_f32: Vec<f32> = faded_f64.into_iter().map(|s| s as f32).collect();
                    resampler.process(&faded_f32, &mut dst_chunk);
                    push_with_deadline(&mut producer, &dst_chunk, &stream_failed, deadline);

                    // Push ~180 ms of trailing digital silence so higher-latency OS mixers
                    // (PulseAudio, PipeWire, Bluetooth) finish playing the fade before drop.
                    let trailing_silence = vec![0.0_f32; ((dst_rate as usize) * 9 / 50).max(2048)];
                    push_with_deadline(&mut producer, &trailing_silence, &stream_failed, deadline);

                    while !producer.is_empty()
                        && !stream_failed.load(Ordering::Acquire)
                        && Instant::now() < deadline
                    {
                        thread::sleep(Duration::from_millis(5));
                    }
                    if !stream_failed.load(Ordering::Acquire) && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(40));
                    }
                }
            }

            drop(stream);
        })
        .map_err(|e| PlaybackError(format!("Failed to spawn audio thread: {e}")))?;

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(PlaybackHandle {
            tx: cmd_tx,
            running,
            producer: Some(producer_thread),
        }),
        Ok(Err(e)) => {
            let _ = producer_thread.join();
            Err(e)
        }
        Err(_) => {
            let _ = producer_thread.join();
            Err(PlaybackError(
                "Audio thread exited before initialization".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{spsc_channel, LinearResampler};
    use std::thread;

    #[test]
    fn spsc_channel_concurrent_wraparound_preserves_exact_order() {
        let (mut tx, mut rx) = spsc_channel(1024);
        let total = 25_000_usize;

        let prod = thread::spawn(move || {
            let mut next = 0_usize;
            let mut batch = [0.0_f32; 137];
            while next < total {
                let count = (total - next).min(batch.len());
                for (i, slot) in batch[..count].iter_mut().enumerate() {
                    *slot = (next + i) as f32;
                }
                let mut sent = 0;
                while sent < count {
                    let n = tx.push_slice(&batch[sent..count]);
                    sent += n;
                    if n == 0 {
                        thread::yield_now();
                    }
                }
                next += count;
            }
        });

        let mut received = 0_usize;
        while received < total {
            if let Some(sample) = rx.pop_one() {
                assert_eq!(sample, received as f32);
                received += 1;
            } else {
                thread::yield_now();
            }
        }
        prod.join().unwrap();
        assert_eq!(rx.pop_one(), None);
    }

    #[test]
    fn resampler_handles_upsampling_and_downsampling_cleanly() {
        let src: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut up = LinearResampler::new(44_100, 48_000);
        let mut up_out = Vec::new();
        up.process(&src, &mut up_out);
        assert!(up_out.len() > src.len());
        assert!(up_out.iter().all(|s| s.is_finite()));

        let mut down = LinearResampler::new(44_100, 22_050);
        let mut down_out = Vec::new();
        down.process(&src, &mut down_out);
        assert!(down_out.len() < src.len());
        assert!(down_out.iter().all(|s| s.is_finite()));
    }
}
