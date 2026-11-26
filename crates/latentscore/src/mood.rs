//! Deterministic text-to-mood lexicon mapping (embedded from `spec/mood_lexicon.json`),
//! optional external int8 embedding-index loader, and momentum-guided [`AutoDrift`].

use crate::config::{
    interpolate_configs, to_internal, to_public, ModeName, MusicConfig, RootNote, SynthConfig,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

const EMBEDDED_LEXICON_JSON: &str = include_str!("../spec/mood_lexicon.json");

const ROOT_CIRCLE: &[RootNote] = &[
    RootNote::C,
    RootNote::G,
    RootNote::D,
    RootNote::A,
    RootNote::E,
    RootNote::B,
    RootNote::FSharp,
    RootNote::CSharp,
    RootNote::GSharp,
    RootNote::DSharp,
    RootNote::ASharp,
    RootNote::F,
];

const DEFAULT_DRIFT_MOODS: &[&str] = &[
    "quiet morning desk",
    "warm cedar zen meditation",
    "rainy midnight coffee shop",
    "vast deep space nebula with stars",
    "soft ocean mist and slow drift",
    "gentle golden sunrise",
];

#[derive(Debug, Deserialize)]
struct LexiconFile {
    rules: Vec<LexiconGroup>,
}

#[derive(Debug, Deserialize)]
struct LexiconGroup {
    first_match: bool,
    entries: Vec<LexiconEntry>,
}

#[derive(Debug, Deserialize)]
struct LexiconEntry {
    keywords: Vec<String>,
    set: Map<String, Value>,
}

fn lexicon_rules() -> &'static [LexiconGroup] {
    static RULES: OnceLock<Vec<LexiconGroup>> = OnceLock::new();
    RULES.get_or_init(|| {
        let parsed: LexiconFile =
            serde_json::from_str(EMBEDDED_LEXICON_JSON).expect("embedded mood_lexicon.json valid");
        parsed.rules
    })
}

fn is_inflection_suffix(suffix: &str) -> bool {
    matches!(suffix, "s" | "es" | "y" | "ed" | "ing" | "ly" | "er" | "ness")
}

fn matches_keywords(tokens: &[&str], keywords: &[String]) -> bool {
    for kw in keywords {
        let allow_inflection = kw.len() >= 4;
        for &tok in tokens {
            if tok == kw.as_str() {
                return true;
            }
            if allow_inflection {
                if let Some(rest) = tok.strip_prefix(kw.as_str()) {
                    if is_inflection_suffix(rest) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Map free-form text deterministically to a [`MusicConfig`] with zero downloads.
///
/// Shares the exact `spec/mood_lexicon.json` rules and word-boundary + stem
/// matching with Python and TypeScript.
pub fn mood_from_text(text: &str) -> MusicConfig {
    mood_from_text_with_base(text, &MusicConfig::default())
}

/// Map free-form text onto an explicit base [`MusicConfig`].
pub fn mood_from_text_with_base(text: &str, base: &MusicConfig) -> MusicConfig {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|tok| !tok.is_empty())
        .collect();
    let mut obj = match serde_json::to_value(base).expect("MusicConfig serializable") {
        Value::Object(map) => map,
        _ => unreachable!(),
    };

    for group in lexicon_rules() {
        for entry in &group.entries {
            if matches_keywords(&tokens, &entry.keywords) {
                for (k, v) in &entry.set {
                    obj.insert(k.clone(), v.clone());
                }
                if group.first_match {
                    break;
                }
            }
        }
    }

    serde_json::from_value(Value::Object(obj)).unwrap_or_else(|_| base.clone())
}

#[derive(Debug, Deserialize)]
struct ManifestJson {
    rows: usize,
    dims: usize,
    embeddings: String,
    scales: String,
    metadata: String,
}

#[derive(Debug, Deserialize)]
struct MetadataJson {
    fields: Vec<String>,
    vocabs: Vec<Vec<Value>>,
    rows: Vec<(String, Vec<String>, Vec<usize>)>,
}

/// One decoded mood entry from an [`EmbeddingIndex`].
#[derive(Debug, Clone)]
pub struct IndexedMood {
    pub index: usize,
    pub title: String,
    pub palette: Vec<String>,
    pub config: MusicConfig,
    pub score: f32,
}

/// Optional loader for the quantized 14,035-mood int8 embedding map.
#[derive(Debug, Clone)]
pub struct EmbeddingIndex {
    rows: usize,
    dims: usize,
    quantized: Vec<i8>,
    scales: Vec<f32>,
    fields: Vec<String>,
    vocabs: Vec<Vec<Value>>,
    meta_rows: Vec<(String, Vec<String>, Vec<usize>)>,
}

impl EmbeddingIndex {
    /// Load `manifest.json`, `embeddings.int8.bin`, `scales.f32.bin`, and `metadata.json`
    /// from a local directory.
    pub fn load_from_dir(dir: impl AsRef<Path>) -> io::Result<Self> {
        let root = dir.as_ref();
        let manifest_str = fs::read_to_string(root.join("manifest.json"))?;
        let manifest: ManifestJson = serde_json::from_str(&manifest_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let emb_bytes = fs::read(root.join(&manifest.embeddings))?;
        let scale_bytes = fs::read(root.join(&manifest.scales))?;
        let meta_str = fs::read_to_string(root.join(&manifest.metadata))?;
        let metadata: MetadataJson = serde_json::from_str(&meta_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        if manifest.rows == 0 || manifest.dims == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest rows and dims must be positive",
            ));
        }
        if metadata.rows.len() != manifest.rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "metadata.json row count does not match manifest.rows",
            ));
        }
        let expected_emb = manifest
            .rows
            .checked_mul(manifest.dims)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "embedding size overflow"))?;
        let expected_scales = manifest
            .rows
            .checked_mul(4)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "scale size overflow"))?;

        let quantized: Vec<i8> = emb_bytes.into_iter().map(|b| b as i8).collect();
        if quantized.len() != expected_emb {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "embeddings.int8.bin length mismatch",
            ));
        }
        if scale_bytes.len() != expected_scales {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "scales.f32.bin length mismatch",
            ));
        }
        let scales: Vec<f32> = scale_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            rows: manifest.rows,
            dims: manifest.dims,
            quantized,
            scales,
            fields: metadata.fields,
            vocabs: metadata.vocabs,
            meta_rows: metadata.rows,
        })
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Decode a row's title, palette, and [`MusicConfig`].
    pub fn get(&self, row_index: usize, score: f32) -> IndexedMood {
        let idx = row_index % self.rows;
        let (title, palette, indices) = &self.meta_rows[idx];
        let mut map = match serde_json::to_value(MusicConfig::default()).unwrap() {
            Value::Object(m) => m,
            _ => unreachable!(),
        };
        for (i, field) in self.fields.iter().enumerate() {
            if let (Some(vocab), Some(&v_idx)) = (self.vocabs.get(i), indices.get(i)) {
                if let Some(val) = vocab.get(v_idx) {
                    map.insert(field.clone(), val.clone());
                }
            }
        }
        let config = serde_json::from_value(Value::Object(map)).unwrap_or_default();
        IndexedMood {
            index: idx,
            title: title.clone(),
            palette: palette.clone(),
            config,
            score,
        }
    }

    /// Dequantize a row vector into `Vec<f32>`.
    pub fn vector_at(&self, row_index: usize) -> Vec<f32> {
        let idx = row_index % self.rows;
        let scale = self.scales[idx];
        let off = idx * self.dims;
        self.quantized[off..off + self.dims]
            .iter()
            .map(|&q| (q as f32) * scale)
            .collect()
    }

    /// Pick a smooth similarity-gradient neighbor with momentum.
    pub fn pick_smooth_neighbor(
        &self,
        current_index: usize,
        prev_vector: Option<&[f32]>,
        visited: &HashSet<usize>,
        rand_unit: f64,
    ) -> IndexedMood {
        let cur = self.vector_at(current_index);
        let mut probe = vec![0.0_f32; self.dims];
        let mut norm_sq = 0.0_f32;
        for d in 0..self.dims {
            let mom = prev_vector.map_or(0.0, |p| 0.35 * (cur[d] - p[d]));
            let v = cur[d] + mom;
            probe[d] = v;
            norm_sq += v * v;
        }
        let inv = if norm_sq > 1e-12 {
            1.0 / norm_sq.sqrt()
        } else {
            1.0
        };
        for v in &mut probe {
            *v *= inv;
        }

        let mut top: Vec<(usize, f32)> = Vec::with_capacity(41);
        for r in 0..self.rows {
            if r == current_index || visited.contains(&r) {
                continue;
            }
            let off = r * self.dims;
            let mut dot = 0.0_f32;
            for d in 0..self.dims {
                dot += (self.quantized[off + d] as f32) * probe[d];
            }
            let sim = dot * self.scales[r];
            if sim > 0.96 {
                continue;
            }
            if top.len() < 40 || sim > top.last().map_or(-2.0, |x| x.1) {
                top.push((r, sim));
                top.sort_by(|a, b| b.1.total_cmp(&a.1));
                if top.len() > 40 {
                    top.pop();
                }
            }
        }

        if top.is_empty() {
            let fallback = ((rand_unit * (self.rows as f64)) as usize) % self.rows;
            return self.get(fallback, 0.5);
        }
        let span = top.len().min(18);
        let pick_idx = ((rand_unit * (span as f64)) as usize).min(span - 1);
        let (chosen, sim) = top[pick_idx];
        self.get(chosen, sim)
    }
}

/// Result of one [`AutoDrift`] step containing both the public [`MusicConfig`]
/// and the unquantized continuous [`SynthConfig`].
#[derive(Debug, Clone)]
pub struct DriftStep {
    pub config: MusicConfig,
    pub synth_config: SynthConfig,
}

/// Endless smooth harmonic/similarity mood drifter.
#[derive(Debug, Clone)]
pub struct AutoDrift {
    rng_state: u64,
    step_count: u64,
    current: SynthConfig,
    index: Option<EmbeddingIndex>,
    current_row: usize,
    prev_vec: Option<Vec<f32>>,
    visited: HashSet<usize>,
}

impl AutoDrift {
    pub fn new(initial: &SynthConfig, seed: u64) -> Self {
        Self {
            rng_state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
            step_count: 0,
            current: initial.clone(),
            index: None,
            current_row: 0,
            prev_vec: None,
            visited: HashSet::new(),
        }
    }

    /// Re-seed deterministic drift while preserving any attached [`EmbeddingIndex`].
    pub fn reseed(&mut self, seed: u64) {
        self.rng_state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        self.step_count = 0;
        self.prev_vec = None;
        self.visited.clear();
        let index_len = self.index.as_ref().map_or(0, EmbeddingIndex::len);
        if index_len > 0 {
            let start = (self.next_u64() as usize) % index_len;
            self.current_row = start;
            self.visited.insert(start);
        }
    }

    pub fn with_index(mut self, index: EmbeddingIndex) -> Self {
        if !index.is_empty() {
            let start = (self.next_u64() as usize) % index.len();
            self.current_row = start;
            self.visited.insert(start);
            self.index = Some(index);
        }
        self
    }

    /// Re-anchor drift state (including an attached [`EmbeddingIndex`]) to `cfg`
    /// and reset momentum after a manual steer or nudge.
    pub fn anchor_to_config(&mut self, cfg: &SynthConfig) {
        self.current = cfg.clone();
        self.prev_vec = None;
        if let Some(idx) = &self.index {
            if idx.is_empty() {
                return;
            }
            let mut best_row = self.current_row % idx.len();
            let mut best_dist = f64::INFINITY;
            // Sample evenly across large indices so manual steer stays O(1) fast.
            let stride = (idx.len() / 512).max(1);
            let mut r = 0_usize;
            while r < idx.len() {
                let cand = to_internal(&idx.get(r, 0.0).config);
                let mut dist = (cand.tempo - cfg.tempo).abs()
                    + (cand.brightness - cfg.brightness).abs()
                    + (cand.space - cfg.space).abs()
                    + (cand.motion - cfg.motion).abs();
                if cand.root != cfg.root {
                    dist += 0.35;
                }
                if cand.mode != cfg.mode {
                    dist += 0.45;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_row = r;
                }
                r += stride;
            }
            self.current_row = best_row;
            self.visited.clear();
            self.visited.insert(best_row);
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Advance one smooth step along the harmonic or embedding gradient, returning
    /// both the public [`MusicConfig`] and continuous [`SynthConfig`].
    pub fn next_step(&mut self, from: Option<&SynthConfig>) -> DriftStep {
        if let Some(cfg) = from {
            self.current = cfg.clone();
        }
        self.step_count += 1;

        let r = self.next_f64();
        if let Some(idx) = &self.index {
            if self.visited.len() > 160 {
                self.visited.clear();
                self.visited.insert(self.current_row);
            }
            let hop = idx.pick_smooth_neighbor(
                self.current_row,
                self.prev_vec.as_deref(),
                &self.visited,
                r,
            );
            self.prev_vec = Some(idx.vector_at(self.current_row));
            self.current_row = hop.index;
            self.visited.insert(hop.index);
            let synth_config = to_internal(&hop.config);
            self.current = synth_config.clone();
            return DriftStep {
                config: hop.config,
                synth_config,
            };
        }

        let cur = self.current.clone();
        let next = if self.step_count % 4 == 0 {
            let anchor_idx = ((self.step_count / 4) as usize) % DEFAULT_DRIFT_MOODS.len();
            let anchor = to_internal(&mood_from_text(DEFAULT_DRIFT_MOODS[anchor_idx]));
            interpolate_configs(&cur, &anchor, 0.45)
        } else {
            let r_pos = ROOT_CIRCLE
                .iter()
                .position(|&r| r == cur.root)
                .unwrap_or(0) as i32;
            let dir = if self.next_f64() >= 0.5 { 1 } else { -1 };
            let next_root = if self.next_f64() < 0.6 {
                let idx = (r_pos + dir).rem_euclid(ROOT_CIRCLE.len() as i32) as usize;
                ROOT_CIRCLE[idx]
            } else {
                cur.root
            };
            let modes = [
                ModeName::Minor,
                ModeName::Dorian,
                ModeName::Major,
                ModeName::Mixolydian,
            ];
            let next_mode = if self.next_f64() < 0.35 {
                modes[(self.next_u64() as usize) % modes.len()]
            } else {
                cur.mode
            };
            let mut n = cur;
            n.root = next_root;
            n.mode = next_mode;
            n.tempo = (n.tempo + self.next_f64() * 0.16 - 0.08).clamp(0.15, 0.85);
            n.brightness = (n.brightness + self.next_f64() * 0.28 - 0.14).clamp(0.12, 0.88);
            n.space = (n.space + self.next_f64() * 0.24 - 0.12).clamp(0.20, 0.95);
            n.motion = (n.motion + self.next_f64() * 0.24 - 0.12).clamp(0.12, 0.85);
            n.echo = (n.echo + self.next_f64() * 0.24 - 0.12).clamp(0.15, 0.88);
            n
        };

        self.current = next.clone();
        DriftStep {
            config: to_public(&next),
            synth_config: next,
        }
    }

    /// Advance one smooth step along the harmonic or embedding gradient.
    pub fn next_mood(&mut self, from: Option<&SynthConfig>) -> MusicConfig {
        self.next_step(from).config
    }
}
