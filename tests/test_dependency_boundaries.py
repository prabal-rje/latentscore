"""Regression guard for the Phase-1 dependency split.

The contract:
  - Optional extras ([external], [heavy], [expressive]) MUST NOT be imported
    at module-load time anywhere in latentscore/*.py. If anyone later adds a
    top-level `from litellm import ...` or `import laion_clap` in the core
    package, this test catches it before the broken wheel ships.

  - With the optional extras poisoned, the core API must still:
      1. import latentscore as ls
      2. ls.render(ls.MusicConfig(), duration=0.1).save(...)
      3. ls.render("text prompt", duration=0.1).save(...)   (retrieval is core)

This test runs in a subprocess so the poisoning is hermetic — the parent
process keeps its real imports intact.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

POISONED_MODULES: tuple[str, ...] = (
    # [external]
    "litellm",
    "dotenv",
    "json_repair",
    # [heavy]
    "laion_clap",
    "torchvision",
    # [expressive]
    "outlines",
    "instructor",
    "mlx",
    "mlx_lm",
    "llama_cpp",
    "bitsandbytes",
)


def test_core_render_paths_work_without_optional_extras(tmp_path: Path) -> None:
    """Poison every optional-extra module; verify core render still works."""
    config_wav = tmp_path / "config.wav"
    text_wav = tmp_path / "text.wav"

    script = textwrap.dedent(f"""
        import sys
        # Poison every optional-extra module: any `import X` from latentscore
        # core code will raise ImportError. find_spec returns None as well.
        for mod in {POISONED_MODULES!r}:
            sys.modules[mod] = None

        import latentscore as ls

        # Pure-synth path: no model needed.
        ls.render(ls.MusicConfig(), duration=0.1).save({str(config_wav)!r})

        # Retrieval path: sentence-transformers is now CORE, so this must work
        # even with all optional extras poisoned.
        ls.render("warm sunset over water", duration=0.1).save({str(text_wav)!r})

        print("PASS")
    """).strip()

    # 300s timeout because a cold first run (no Hugging Face cache) needs to
    # download MiniLM (~90 MB) before the text-prompt render can succeed.
    # Warm-cache runs finish in <10s.
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"core-only render failed with poisoned optional extras:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "PASS" in result.stdout
    assert config_wav.exists()
    assert text_wav.exists()
    assert config_wav.stat().st_size > 44  # WAV header is 44 bytes
    assert text_wav.stat().st_size > 44


def test_core_modules_dont_import_optional_extras_at_module_top() -> None:
    """Static check: grep `latentscore/*.py` for top-level imports of optional extras.

    Lazy imports (`def f(): from X import Y`) are fine — only flag imports
    that fire at module load time.
    """
    import re

    root = Path(__file__).resolve().parents[1] / "latentscore"
    forbidden = "|".join(re.escape(m) for m in POISONED_MODULES)
    # Capture group 1 = indentation. Top-level imports have empty group 1.
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*)(?P<kw>import|from)\s+(?P<mod>{forbidden})(\s|\.|$)",
        re.MULTILINE,
    )

    offenders: list[tuple[Path, int, str]] = []
    for py in root.rglob("*.py"):
        text = py.read_text()
        for m in pattern.finditer(text):
            if m.group("indent"):
                continue  # indented => inside a function or try/except => lazy => OK
            lineno = text[: m.start()].count("\n") + 1
            offenders.append((py.relative_to(root.parent), lineno, m.group(0).strip()))

    assert not offenders, (
        f"Optional-extra modules are imported at module top in {len(offenders)} location(s) — "
        f"this breaks the dep boundary. Make these imports lazy (inside the function "
        f"that uses them):\n"
        + "\n".join(f"  {p}:{ln}  {snippet!r}" for p, ln, snippet in offenders)
    )
