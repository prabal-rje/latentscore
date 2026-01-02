#!/usr/bin/env python3
"""
RAVE Model Downloader & Infinite Streaming Setup (v2)

Working model sources:
1. Intelligent Instruments Lab (IIL) - organ, voice, birds, water, guitar
2. Tangible Music Lab - trained on 9,455 musical loops! (Best for variety)
"""

import os
import sys
import subprocess
from pathlib import Path

def install_deps():
    """Install required packages."""
    deps = [
        "torch",
        "torchaudio", 
        "numpy",
        "sounddevice",
        "huggingface_hub",
    ]
    
    print("Installing dependencies...")
    for dep in deps:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], check=True)
    print("✅ Dependencies installed\n")


# =============================================================================
# WORKING MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY = {
    # === TANGIBLE MUSIC LAB - Trained on 9,455 musical loops! ===
    # This is the best for "variety" - trained on Freesound Loop Dataset
    "loops": {
        "repo": "Tangible-Music-Lab/RAVE_models",
        "filename": "freesoundloop10k_default_b2048_r48000_z16.ts",  # Main model
        "description": "🎵 BEST FOR VARIETY - Trained on 9,455 musical loops from Freesound",
        "vibe": ["loops", "variety", "musical", "electronic", "ambient"],
        "latent_dim": 16,
        "sample_rate": 48000,
    },
    
    # # === INTELLIGENT INSTRUMENTS LAB ===
    # "organ": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "organ_archive_b2048_r48000_z16.ts",
    #     "description": "Church organ - rich harmonics, cathedral ambiance",
    #     "vibe": ["ambient", "sacred", "drone", "classical"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "organ_bach": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "organ_bach_b2048_r48000_z16.ts",
    #     "description": "J.S. Bach organ music - baroque, structured",
    #     "vibe": ["classical", "baroque", "structured"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "voice": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "voice_vocalset_b2048_r48000_z16.ts",
    #     "description": "Singing voice - vocal textures, human warmth",
    #     "vibe": ["vocal", "human", "expressive"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "speech": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "voice_hifitts_b2048_r48000_z16.ts",
    #     "description": "Speech/audiobook - spoken word textures",
    #     "vibe": ["speech", "human", "intimate"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "guitar": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "guitar_iil_b2048_r48000_z16.ts",
    #     "description": "Electric guitar - plucks, strums, scrapes",
    #     "vibe": ["guitar", "rock", "textural"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "birds": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "birds_dawnchorus_b2048_r48000_z8.ts",
    #     "description": "Dawn chorus bird recordings - nature, organic",
    #     "vibe": ["nature", "ambient", "organic", "peaceful"],
    #     "latent_dim": 8,
    #     "sample_rate": 48000,
    # },
    # "water": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "water_pondbrain_b2048_r48000_z16.ts",
    #     "description": "Water recordings - liquid, flowing",
    #     "vibe": ["water", "ambient", "organic", "meditative"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "whales": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "whales_pondbrain_b2048_r48000_z16.ts",
    #     "description": "Whale songs - deep, mysterious, oceanic",
    #     "vibe": ["ocean", "ambient", "mysterious", "deep"],
    #     "latent_dim": 16,
    #     "sample_rate": 48000,
    # },
    # "magnets": {
    #     "repo": "Intelligent-Instruments-Lab/rave-models",
    #     "filename": "magnets_thales_b2048_r48000_z8.ts",
    #     "description": "Magnets on surfaces - percussive, textural",
    #     "vibe": ["percussive", "textural", "experimental"],
    #     "latent_dim": 8,
    #     "sample_rate": 48000,
    # },
}

# Recommended for your use case
RECOMMENDED = ["loops", "organ", "birds", "water"]


def download_models(model_dir: Path, models: list[str] | None = None):
    """Download models from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if models is None:
        models = RECOMMENDED
    
    downloaded = {}
    
    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            print(f"⚠️  Unknown model: {model_name}")
            print(f"   Available: {list(MODEL_REGISTRY.keys())}")
            continue
        
        info = MODEL_REGISTRY[model_name]
        target_path = model_dir / f"{model_name}.ts"
        
        if target_path.exists():
            print(f"✅ {model_name}: Already downloaded ({target_path})")
            downloaded[model_name] = target_path
            continue
        
        print(f"⬇️  {model_name}: {info['description']}")
        print(f"   From: {info['repo']}/{info['filename']}")
        
        try:
            path = hf_hub_download(
                repo_id=info["repo"],
                filename=info["filename"],
            )
            
            # Copy to our standard location
            import shutil
            shutil.copy(path, target_path)
            
            downloaded[model_name] = target_path
            print(f"   ✅ Saved to {target_path}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    return downloaded


def benchmark_model(model_path: Path, device: str = "mps") -> dict:
    """Benchmark a model for real-time capability."""
    import torch
    import time
    
    print(f"\n🔬 Benchmarking: {model_path.stem}")
    
    results = {"path": str(model_path), "device": device}
    
    try:
        model = torch.jit.load(str(model_path))
        model = model.to(device)
        model.eval()
        print(f"   ✅ Model loaded on {device}")
    except Exception as e:
        results["error"] = f"Load failed: {e}"
        print(f"   ❌ {results['error']}")
        return results
    
    try:
        with torch.no_grad():
            # Try to find latent dim by encoding test audio
            test_audio = torch.randn(1, 1, 48000).to(device)  # 1 second
            
            if hasattr(model, 'encode'):
                z = model.encode(test_audio)
                latent_dim = z.shape[1]
                latent_length = z.shape[2]
                compression = 48000 // latent_length
                results["latent_dim"] = latent_dim
                results["compression"] = compression
                print(f"   Latent: {latent_dim} dims, {compression}x compression")
            else:
                # Fallback
                latent_dim = 16
                latent_length = 128
                results["latent_dim"] = latent_dim
                results["compression"] = "unknown"
            
            # Benchmark decode speed
            z = torch.randn(1, latent_dim, latent_length * 2).to(device)
            
            # Warmup
            for _ in range(3):
                _ = model.decode(z)
                if device == "mps":
                    torch.mps.synchronize()
            
            # Timed runs  
            times = []
            for _ in range(10):
                start = time.perf_counter()
                audio = model.decode(z)
                if device == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - start)
            
            audio_samples = audio.shape[-1]
            audio_duration = audio_samples / 48000
            avg_time = sum(times) / len(times)
            
            results["audio_duration"] = f"{audio_duration:.3f}s"
            results["generation_time"] = f"{avg_time:.3f}s"
            results["realtime_factor"] = audio_duration / avg_time
            results["realtime_ok"] = audio_duration > avg_time
            
            status = "✅" if results["realtime_ok"] else "❌"
            print(f"   {status} {audio_duration:.3f}s audio in {avg_time:.3f}s = {results['realtime_factor']:.1f}x realtime")
            
            if device == "mps":
                mem_mb = torch.mps.current_allocated_memory() / 1024 / 1024
                results["memory_mb"] = f"{mem_mb:.1f}"
                print(f"   Memory: {mem_mb:.1f} MB")
                
    except Exception as e:
        results["error"] = str(e)
        print(f"   ❌ Benchmark failed: {e}")
    
    return results


def benchmark_all(model_dir: Path):
    """Benchmark all downloaded models."""
    import torch
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"BENCHMARKING (device: {device})")
    print('='*60)
    
    results = {}
    for model_path in sorted(model_dir.glob("*.ts")):
        results[model_path.stem] = benchmark_model(model_path, device)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    for name, res in sorted(results.items()):
        if "error" in res:
            print(f"❌ {name}: {res['error']}")
        elif res.get("realtime_ok"):
            print(f"✅ {name}: {res['realtime_factor']:.1f}x realtime")
        else:
            print(f"⚠️  {name}: {res['realtime_factor']:.1f}x (may be slow)")
    
    return results


# =============================================================================
# INFINITE STREAMER MODULE
# =============================================================================

# Now writes the full, updated streamer code from rave_models/rave_streamer.py into STREAMER_CODE.
STREAMER_CODE = '''#!/usr/bin/env python3
"""
RAVE Streamer v2 - Fixed for smooth playback

Changes from v1:
- Larger buffers (0.5s instead of 0.1s)
- More queue depth (16 instead of 4)
- Longer pre-fill (1s instead of 0.2s)
- Generate in bigger chunks, less often
"""

import torch
import numpy as np
import sounddevice as sd
import threading
import queue
import time
import sys
from pathlib import Path


class RAVEStreamer:
    def __init__(
        self, 
        model_path: str, 
        device: str = "mps",
        buffer_sec: float = 0.5,  # BIGGER buffer
        queue_depth: int = 16,     # MORE queue depth
    ):
        self.device = device
        self.sample_rate = 48000
        
        print(f"Loading: {model_path}")
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Probe dimensions
        with torch.no_grad():
            test = torch.randn(1, 1, self.sample_rate).to(device)
            z = self.model.encode(test)
            self.latent_dim = z.shape[1]
            self.compression = self.sample_rate // z.shape[2]
        
        print(f"Latent: {self.latent_dim} dims, {self.compression}x compression")
        
        # Audio settings - BIGGER chunks
        self.buffer_samples = int(buffer_sec * self.sample_rate)
        self.latent_frames = max(1, self.buffer_samples // self.compression)
        
        print(f"Buffer: {buffer_sec}s = {self.buffer_samples} samples, {self.latent_frames} latent frames")
        
        # Mood state
        self.current_z = torch.randn(1, self.latent_dim, 1).to(device)
        self.target_z = self.current_z.clone()
        self.lerp_speed = 0.02
        
        # Threading with BIGGER queue
        self.audio_queue = queue.Queue(maxsize=queue_depth)
        self.running = False
        self.gen_thread = None
        self.stream = None
        
        # Playback state
        self.play_buffer = np.zeros(0)
        self.play_pos = 0
        
        # Presets
        self.presets = {}
        self._init_presets()
    
    def _init_presets(self):
        torch.manual_seed(42)
        names = ["calm", "active", "dark", "bright", "chaos"]
        for i, name in enumerate(names):
            z = torch.randn(1, self.latent_dim, 1).to(self.device)
            z = z * (0.3 + i * 0.2)
            self.presets[name] = z
    
    def _generate_chunk(self) -> np.ndarray:
        with torch.no_grad():
            # Interpolate
            self.current_z = self.current_z * (1 - self.lerp_speed) + self.target_z * self.lerp_speed
            
            # Expand with smooth walk
            z_exp = self.current_z.expand(-1, -1, self.latent_frames).clone()
            
            # Gentle random walk for organic variation
            walk = torch.cumsum(torch.randn_like(z_exp) * 0.003, dim=2)
            z_exp = z_exp + walk
            
            # Decode
            audio = self.model.decode(z_exp)
            
            if self.device == "mps":
                torch.mps.synchronize()
            
            return audio.cpu().numpy().squeeze()
    
    def _gen_loop(self):
        """Background generation - runs ahead of playback."""
        while self.running:
            try:
                # Only generate if queue isn't full
                if self.audio_queue.qsize() < self.audio_queue.maxsize:
                    chunk = self._generate_chunk()
                    self.audio_queue.put(chunk, timeout=0.5)
                else:
                    time.sleep(0.01)  # Queue full, wait
            except queue.Full:
                continue
            except Exception as e:
                print(f"Gen error: {e}")
                if not self.running:
                    break
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback - pulls from buffer."""
        if status:
            print(f"Audio status: {status}")
        
        output = np.zeros(frames)
        written = 0
        
        while written < frames:
            # Need more data from play_buffer?
            if self.play_pos >= len(self.play_buffer):
                try:
                    self.play_buffer = self.audio_queue.get_nowait()
                    self.play_pos = 0
                except queue.Empty:
                    # Underrun! Fill with silence
                    print("!", end="", flush=True)
                    break
            
            # Copy what we can
            available = len(self.play_buffer) - self.play_pos
            to_copy = min(available, frames - written)
            output[written:written + to_copy] = self.play_buffer[self.play_pos:self.play_pos + to_copy]
            self.play_pos += to_copy
            written += to_copy
        
        outdata[:, 0] = output
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        self.play_buffer = np.zeros(0)
        self.play_pos = 0
        
        # Start generator
        self.gen_thread = threading.Thread(target=self._gen_loop, daemon=True)
        self.gen_thread.start()
        
        # Pre-fill queue (wait for at least 4 chunks = 2 seconds)
        print("Pre-filling buffer...", end=" ", flush=True)
        prefill_target = min(8, self.audio_queue.maxsize)
        timeout = time.time() + 5.0  # Max 5 second wait
        
        while self.audio_queue.qsize() < prefill_target and time.time() < timeout:
            time.sleep(0.1)
            print(".", end="", flush=True)
        
        print(f" {self.audio_queue.qsize()} chunks ready")
        
        # Start audio output
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            blocksize=2048,  # Smaller blocksize = lower latency
            latency='high',  # Request high latency for stability
        )
        self.stream.start()
        print("🎵 Streaming!")
    
    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.gen_thread:
            self.gen_thread.join(timeout=2.0)
        print("\\n🛑 Stopped")
    
    def set_mood(self, z: torch.Tensor):
        """Instant mood change."""
        self.target_z = z.to(self.device)
        self.current_z = self.target_z.clone()
    
    def goto_mood(self, z: torch.Tensor, seconds: float = 3.0):
        """Smooth transition."""
        self.target_z = z.to(self.device)
        # Approximate lerp speed
        chunks_per_sec = self.sample_rate / self.buffer_samples
        self.lerp_speed = min(0.5, 1.0 / (seconds * chunks_per_sec))
    
    def goto_preset(self, name: str, seconds: float = 3.0):
        if name in self.presets:
            print(f"→ {name}")
            self.goto_mood(self.presets[name], seconds)
    
    def random_mood(self):
        print("Random!")
        self.set_mood(torch.randn(1, self.latent_dim, 1))


def main():
    if len(sys.argv) < 2:
        print("Usage: python rave_streamer_fixed.py <model.ts>")
        print("\\nAvailable models:")
        for f in Path(".").glob("*.ts"):
            print(f"  {f}")
        return
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    streamer = RAVEStreamer(
        sys.argv[1], 
        device=device,
        buffer_sec=0.5,   # 500ms chunks
        queue_depth=16,   # 8 seconds of buffer
    )
    streamer.start()
    
    print("\\n🎹 Controls:")
    print("  r     = random mood")  
    print("  1-5   = presets (calm→chaos)")
    print("  +/-   = faster/slower transitions")
    print("  q     = quit")
    
    transition_time = 3.0
    
    try:
        while True:
            cmd = input().strip().lower()
            if cmd == "q":
                break
            elif cmd == "r":
                streamer.random_mood()
            elif cmd in "12345":
                names = ["calm", "active", "dark", "bright", "chaos"]
                streamer.goto_preset(names[int(cmd)-1], transition_time)
            elif cmd == "+":
                transition_time = max(0.5, transition_time - 1)
                print(f"Transition: {transition_time}s")
            elif cmd == "-":
                transition_time = min(10, transition_time + 1)
                print(f"Transition: {transition_time}s")
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
'''


def write_streamer(path: Path):
    """Write the streamer module."""
    path.write_text(STREAMER_CODE)
    print(f"✅ Wrote: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAVE Setup v2")
    parser.add_argument("--dir", type=Path, default=Path("./rave_models"))
    parser.add_argument("--download", nargs="*", help="Models to download (default: recommended)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark models")
    parser.add_argument("--skip-deps", action="store_true")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available RAVE models:")
        print("="*60)
        for name, info in MODEL_REGISTRY.items():
            rec = "⭐" if name in RECOMMENDED else "  "
            print(f"{rec} {name:12} - {info['description']}")
        print(f"\n⭐ = Recommended for mood-responsive music")
        return
    
    print("="*60)
    print("RAVE SETUP v2")
    print("="*60)
    
    if not args.skip_deps:
        install_deps()
    
    # Determine models to download
    if args.all:
        models = list(MODEL_REGISTRY.keys())
    elif args.download is not None:
        models = args.download if args.download else RECOMMENDED
    else:
        models = RECOMMENDED
    
    print(f"\n📁 Model directory: {args.dir}")
    print(f"📦 Models: {models}")
    
    downloaded = download_models(args.dir, models)
    
    if args.benchmark or downloaded:
        benchmark_all(args.dir)
    
    # Write streamer
    streamer_path = args.dir / "rave_streamer.py"
    write_streamer(streamer_path)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("="*60)
    print(f"""
1. Test streaming:
   cd {args.dir}
   python rave_streamer.py loops.ts
   
2. Try different models for different vibes:
   - loops.ts   → Musical variety (BEST for your use case)
   - organ.ts   → Ambient, sacred
   - birds.ts   → Nature, peaceful
   - water.ts   → Meditative, flowing
   
3. For your activity tracker:
   - Import RAVEStreamer
   - Map activity metrics to latent vectors
   - Call streamer.goto_mood(latent, seconds=2.0)
""")


if __name__ == "__main__":
    main()