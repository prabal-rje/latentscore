#!/usr/bin/env python3
"""
download_soundfonts.py

Downloads free SoundFonts for use with the synth engine.
All downloads are permissively licensed (MIT, CC0, CC-BY, or public domain).

Usage:
    python download_soundfonts.py              # Download all
    python download_soundfonts.py --gm-only    # Just GM soundfont
    python download_soundfonts.py --ethnic     # Just ethnic instruments
    python download_soundfonts.py --list       # Show what would be downloaded
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class SoundFontInfo(NamedTuple):
    name: str
    url: str
    filename: str
    size_mb: float
    license: str
    description: str
    sha256: Optional[str] = None  # Optional hash for verification


# =============================================================================
# SoundFont Registry
# =============================================================================

GM_SOUNDFONTS: List[SoundFontInfo] = [
    SoundFontInfo(
        name="GeneralUser GS",
        url="https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_v1.471.sf2?dl=1",
        filename="GeneralUser_GS.sf2",
        size_mb=30.0,
        license="Custom Permissive (attribution appreciated)",
        description="High-quality GM soundfont, good balance of quality and size",
    ),
    # Backup: FluidR3 from a more reliable source
    # SoundFontInfo(
    #     name="FluidR3 GM",
    #     url="https://keymusician01.s3.amazonaws.com/FluidR3_GM.sf2",
    #     filename="FluidR3_GM.sf2",
    #     size_mb=140.0,
    #     license="MIT",
    #     description="Classic high-quality GM soundfont",
    # ),
]

ETHNIC_SOUNDFONTS: List[SoundFontInfo] = [
    SoundFontInfo(
        name="ArabTurk",
        url="https://musical-artifacts.com/artifacts/258/ArabTurk.sf2",
        filename="ArabTurk.sf2",
        size_mb=51.0,  # Actual size from your download
        license="Free (Musical Artifacts)",
        description="Oud, Tanbur, Kanun, Ney, Duduk, Darbuka, Doumbek, Bendir, Riq",
    ),
    # Note: Archive.org World Instruments link is dead (404)
    # The GM soundfont (GeneralUser) includes ethnic instruments:
    # - Program 104: Sitar
    # - Program 105: Banjo
    # - Program 107: Koto
    # - Program 108: Kalimba
    # - Program 109: Bagpipe
    # - Program 110: Fiddle
    # - Program 111: Shanai
]

PERCUSSION_SOUNDFONTS: List[SoundFontInfo] = [
    SoundFontInfo(
        name="Taiko Drum Collection",
        url="https://musical-artifacts.com/artifacts/2984/Taiko_Drum_Collection.zip",
        filename="Taiko_Drum_Collection.zip",
        size_mb=28.3,
        license="Various (CC BY 3.0, free use)",
        description="Taiko drums - Jason Champion, S. Christian Collins collection",
    ),
    # Note: Archive.org Arabic Kits link is dead (404)
    # The GM soundfont includes drum kits on bank 128 (percussion channel)
]


# =============================================================================
# Download Utilities
# =============================================================================


def get_soundfonts_dir() -> Path:
    """Get or create the soundfonts directory."""
    # Try relative to this script first
    script_dir = Path(__file__).parent
    sf_dir = script_dir / "soundfonts"
    
    if not sf_dir.exists():
        sf_dir.mkdir(parents=True, exist_ok=True)
    
    return sf_dir


def download_file(url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download a file with progress indication."""
    try:
        # Create a request with a user agent (some servers require it)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (SoundFont Downloader)"}
        )
        
        with urllib.request.urlopen(request, timeout=60) as response:
            total_size = int(response.headers.get("content-length", 0))
            
            # Use temp file to avoid partial downloads
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                downloaded = 0
                block_size = 8192
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    
                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  [{bar}] {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                        sys.stdout.flush()
                
                tmp_path = tmp.name
            
            # Move to final destination
            shutil.move(tmp_path, dest)
            
            if show_progress:
                print()  # Newline after progress bar
            
            return True
            
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def verify_file(path: Path, expected_hash: Optional[str]) -> bool:
    """Verify file hash if provided."""
    if expected_hash is None:
        return True
    
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    actual = sha256.hexdigest()
    if actual.lower() != expected_hash.lower():
        print(f"  Hash mismatch! Expected {expected_hash[:16]}..., got {actual[:16]}...")
        return False
    
    return True


def download_soundfont(sf: SoundFontInfo, dest_dir: Path, force: bool = False) -> bool:
    """Download a single soundfont."""
    dest = dest_dir / sf.filename
    
    if dest.exists() and not force:
        print(f"  ✅ Already exists: {sf.filename}")
        return True
    
    print(f"\nDownloading: {sf.name}")
    print(f"  Size: ~{sf.size_mb:.1f} MB")
    print(f"  License: {sf.license}")
    print(f"  URL: {sf.url[:60]}...")
    
    if download_file(sf.url, dest):
        if verify_file(dest, sf.sha256):
            print(f"  ✅ Saved: {dest}")
            
            # Extract zip files
            if sf.filename.endswith('.zip'):
                print(f"  Extracting zip...")
                try:
                    with zipfile.ZipFile(dest, 'r') as zf:
                        zf.extractall(dest_dir)
                    print(f"  ✅ Extracted to: {dest_dir}")
                except Exception as e:
                    print(f"  Warning: Could not extract zip: {e}")
            
            return True
        else:
            dest.unlink()  # Remove corrupt file
            return False
    
    return False


def download_category(
    soundfonts: List[SoundFontInfo],
    dest_dir: Path,
    category_name: str,
    force: bool = False
) -> int:
    """Download all soundfonts in a category."""
    print(f"\n{'=' * 60}")
    print(f"Downloading: {category_name}")
    print(f"{'=' * 60}")
    
    success_count = 0
    for sf in soundfonts:
        if download_soundfont(sf, dest_dir, force):
            success_count += 1
    
    return success_count


def list_soundfonts() -> None:
    """List all available soundfonts without downloading."""
    print("\nAvailable SoundFonts:")
    print("=" * 70)
    
    total_size = 0.0
    
    print("\n[GM SoundFonts - General MIDI instruments]")
    for sf in GM_SOUNDFONTS:
        print(f"  â€¢ {sf.name} ({sf.size_mb:.1f} MB)")
        print(f"    {sf.description}")
        total_size += sf.size_mb
    
    print("\n[Ethnic SoundFonts - World instruments]")
    for sf in ETHNIC_SOUNDFONTS:
        print(f"  â€¢ {sf.name} ({sf.size_mb:.1f} MB)")
        print(f"    {sf.description}")
        total_size += sf.size_mb
    
    print("\n[Percussion SoundFonts - Drums & rhythms]")
    for sf in PERCUSSION_SOUNDFONTS:
        print(f"  â€¢ {sf.name} ({sf.size_mb:.1f} MB)")
        print(f"    {sf.description}")
        total_size += sf.size_mb
    
    print(f"\nTotal size: ~{total_size:.1f} MB")


def create_instrument_map(dest_dir: Path) -> None:
    """Create a JSON file mapping instrument names to soundfont/program."""
    import json
    
    instrument_map: Dict[str, Dict[str, any]] = {
        # GM instruments (from GeneralUser GS)
        "piano": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 0},
        "acoustic_bass": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 32},
        "synth_bass": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 38},
        "strings": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 48},
        "pad_warm": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 89},
        "pad_choir": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 91},
        "flute": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 73},
        "sitar": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 104},
        "koto": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 107},
        "kalimba": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 108},
        "shanai": {"soundfont": "GeneralUser_GS.sf2", "bank": 0, "program": 111},
        
        # Ethnic instruments (from ArabTurk.sf2)
        "oud": {"soundfont": "ArabTurk.sf2", "bank": 0, "program": 0, "note": "Check actual program"},
        "kanun": {"soundfont": "ArabTurk.sf2", "bank": 0, "program": 1, "note": "Check actual program"},
        "ney": {"soundfont": "ArabTurk.sf2", "bank": 0, "program": 2, "note": "Check actual program"},
        "darbuka": {"soundfont": "ArabTurk.sf2", "bank": 0, "program": 3, "note": "Check actual program"},
        "doumbek": {"soundfont": "ArabTurk.sf2", "bank": 0, "program": 4, "note": "Check actual program"},
        "riq": {"soundfont": "ArabTurk.sf2", "bank": 0, "program": 5, "note": "Check actual program"},
        
        # World instruments (from World_Instruments.sf2)
        "tambura": {"soundfont": "World_Instruments.sf2", "bank": 0, "program": 0, "note": "Check actual program"},
        "shakuhachi": {"soundfont": "World_Instruments.sf2", "bank": 0, "program": 1, "note": "Check actual program"},
        "shamisen": {"soundfont": "World_Instruments.sf2", "bank": 0, "program": 2, "note": "Check actual program"},
    }
    
    map_file = dest_dir / "instrument_map.json"
    with open(map_file, "w") as f:
        json.dump(instrument_map, f, indent=2)
    
    print(f"\nCreated instrument map: {map_file}")
    print("Note: You'll need to update program numbers after inspecting each SF2")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download free SoundFonts for the synth engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_soundfonts.py                # Download all (~85 MB)
  python download_soundfonts.py --gm-only      # Just GM soundfont (~30 MB)
  python download_soundfonts.py --ethnic       # Just ethnic instruments (~18 MB)
  python download_soundfonts.py --list         # Show what would be downloaded
  python download_soundfonts.py --force        # Re-download even if exists
        """
    )
    
    parser.add_argument("--list", action="store_true", help="List available soundfonts")
    parser.add_argument("--gm-only", action="store_true", help="Download only GM soundfont")
    parser.add_argument("--ethnic", action="store_true", help="Download only ethnic soundfonts")
    parser.add_argument("--percussion", action="store_true", help="Download only percussion")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_soundfonts()
        return 0
    
    # Get destination directory
    if args.dest:
        dest_dir = Path(args.dest)
    else:
        dest_dir = get_soundfonts_dir()
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"SoundFont directory: {dest_dir}")
    
    total_success = 0
    total_attempted = 0
    
    # Download based on flags
    download_all = not (args.gm_only or args.ethnic or args.percussion)
    
    if download_all or args.gm_only:
        total_attempted += len(GM_SOUNDFONTS)
        total_success += download_category(GM_SOUNDFONTS, dest_dir, "General MIDI", args.force)
    
    if download_all or args.ethnic:
        total_attempted += len(ETHNIC_SOUNDFONTS)
        total_success += download_category(ETHNIC_SOUNDFONTS, dest_dir, "Ethnic Instruments", args.force)
    
    if download_all or args.percussion:
        total_attempted += len(PERCUSSION_SOUNDFONTS)
        total_success += download_category(PERCUSSION_SOUNDFONTS, dest_dir, "Percussion", args.force)
    
    # Create instrument map
    if total_success > 0:
        create_instrument_map(dest_dir)
    
    print(f"\n{'=' * 60}")
    print(f"Download complete: {total_success}/{total_attempted} successful")
    print(f"SoundFonts saved to: {dest_dir}")
    print(f"{'=' * 60}")
    
    return 0 if total_success == total_attempted else 1


if __name__ == "__main__":
    sys.exit(main())