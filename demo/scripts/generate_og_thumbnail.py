"""Generate a clean 1200x630 Alba-style social share thumbnail (og-card.png).

Renders real Fraunces / Plus Jakarta Sans / JetBrains Mono typography, daylight
aurora gradients, stippled risograph grain, and LatentScore's bespoke pixel-art
music charms in a minimalist editorial layout matching the homepage.

File map:
- HTML template: self-contained 1200x630 markup embedding local pixel assets.
- Renderer: headless Chrome Playwright screenshot with strict file checks.
- CLI entry point: writes frontend/public/og-card.png.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

# === Constants ===

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = REPO_ROOT / "frontend" / "public"
ART_DIR = PUBLIC_DIR / "art"
OUT_PATH = PUBLIC_DIR / "og-card.png"
CHROME_PATH = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CHARM_ROW_FILES: tuple[str, ...] = (
    "charm-vinyl.png",
    "charm-note.png",
    "charm-wave.png",
    "charm-synth.png",
    "charm-cassette.png",
    "charm-bell.png",
    "charm-clef.png",
    "charm-prism.png",
)

# === HTML template ===


def _img_data_uri(path: Path) -> str:
    """Encode a local PNG asset as an inline base64 data URI.

    Examples:
        >>> uri = _img_data_uri(ART_DIR / "charm-vinyl.png")
        >>> uri.startswith("data:image/png;base64,")
        True
    """
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_og_html() -> str:
    """Build minimalist 1200x630 HTML for LatentScore's homepage share card.

    Examples:
        >>> html = build_og_html()
        >>> "Latent" in html and "ambient sound" in html
        True
    """
    vinyl = _img_data_uri(ART_DIR / "charm-vinyl.png")
    grain = _img_data_uri(ART_DIR / "grain.png")
    charm_tags = "".join(
        f'<img src="{_img_data_uri(ART_DIR / name)}" alt="{name}" />'
        for name in CHARM_ROW_FILES
    )
    fonts_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;"
        "1,9..144,400;1,9..144,500"
        "&family=JetBrains+Mono:wght@400;500"
        "&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="{fonts_url}" rel="stylesheet" />
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 1200px;
    height: 630px;
    overflow: hidden;
    background: #FFFFFF;
    font-family: "Plus Jakarta Sans", -apple-system, sans-serif;
    color: #111318;
    -webkit-font-smoothing: antialiased;
  }}
  .stage {{
    position: relative;
    width: 1200px;
    height: 630px;
    padding: 54px 88px 50px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #FFFFFF;
    overflow: hidden;
  }}
  .aurora {{
    position: absolute;
    inset: -60px;
    background:
      radial-gradient(56% 62% at 14% 20%,
        rgba(253, 186, 116, 0.56) 0%, rgba(253, 186, 116, 0) 70%),
      radial-gradient(54% 66% at 86% 20%,
        rgba(52, 211, 153, 0.52) 0%, rgba(52, 211, 153, 0) 72%),
      radial-gradient(56% 62% at 54% 46%,
        rgba(56, 189, 248, 0.40) 0%, rgba(56, 189, 248, 0) 72%),
      radial-gradient(46% 52% at 22% 82%,
        rgba(251, 113, 133, 0.28) 0%, rgba(251, 113, 133, 0) 70%),
      radial-gradient(44% 50% at 88% 82%,
        rgba(99, 102, 241, 0.22) 0%, rgba(99, 102, 241, 0) 72%);
    filter: blur(28px) saturate(1.18);
    z-index: 0;
  }}
  .grain {{
    position: absolute;
    inset: 0;
    background-image: url("{grain}");
    background-repeat: repeat;
    background-size: 210px 210px;
    opacity: 0.36;
    mix-blend-mode: multiply;
    z-index: 1;
  }}
  .top-row {{
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .brand {{
    display: inline-flex;
    align-items: center;
    gap: 14px;
    font-family: "Fraunces", Georgia, serif;
    font-size: 32px;
    font-weight: 500;
    letter-spacing: -0.025em;
    color: #111318;
  }}
  .brand em {{
    font-style: italic;
    color: #2D333E;
  }}
  .brand img {{
    width: 38px;
    height: 38px;
    image-rendering: pixelated;
  }}
  .kicker {{
    display: inline-flex;
    align-items: center;
    gap: 9px;
    font-family: "JetBrains Mono", monospace;
    font-size: 13.5px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2D333E;
  }}
  .kicker-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #059669;
  }}
  .center {{
    position: relative;
    z-index: 2;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    max-width: 960px;
  }}
  .hero-icon {{
    width: 78px;
    height: 78px;
    margin-bottom: 22px;
    image-rendering: pixelated;
    filter: drop-shadow(0 10px 18px rgba(17, 19, 24, 0.14));
  }}
  .headline {{
    font-family: "Fraunces", Georgia, serif;
    font-size: 86px;
    font-weight: 400;
    line-height: 1.04;
    letter-spacing: -0.03em;
    color: #111318;
    margin-bottom: 18px;
  }}
  .headline em {{
    font-style: italic;
    display: inline-block;
    padding-right: 0.06em;
    background-image: linear-gradient(
      90deg, #0284C7 0%, #0D9488 28%, #D97706 58%, #E11D48 84%, #6366F1 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
  }}
  .sub {{
    font-size: 24px;
    line-height: 1.48;
    color: #2D333E;
    max-width: 760px;
  }}
  .charm-strip {{
    position: relative;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 26px;
    padding-top: 12px;
  }}
  .charm-strip img {{
    width: 46px;
    height: 46px;
    image-rendering: pixelated;
    filter: drop-shadow(0 6px 12px rgba(17, 19, 24, 0.12));
  }}
</style>
</head>
<body>
  <div class="stage">
    <div class="aurora"></div>
    <div class="grain"></div>

    <div class="top-row">
      <div class="brand">
        <img src="{vinyl}" alt="LatentScore pixel vinyl" />
        <span>Latent<em>Score</em></span>
      </div>
      <div class="kicker">
        <span class="kicker-dot"></span>
        SIGGRAPH Talks &apos;26 &middot; Open Source
      </div>
    </div>

    <div class="center">
      <img src="{vinyl}" alt="LatentScore" class="hero-icon" />
      <h1 class="headline">Turn vibes into <em>ambient sound</em>.</h1>
      <p class="sub">
        Describe any mood or scene in plain words. Procedural ambient music
        runs locally in your browser with no GPU required.
      </p>
    </div>

    <div class="charm-strip">
      {charm_tags}
    </div>
  </div>
</body>
</html>
"""


# === Renderer ===


def render_og_thumbnail(out_path: Path = OUT_PATH) -> Path:
    """Render the 1200x630 social thumbnail PNG via headless Chrome.

    Args:
        out_path: Destination path for the generated PNG.

    Returns:
        Path to the written PNG file.

    Examples:
        >>> OUT_PATH.name
        'og-card.png'
    """
    html = build_og_html()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        executable = str(CHROME_PATH) if CHROME_PATH.exists() else None
        browser = p.chromium.launch(headless=True, executable_path=executable)
        ctx = browser.new_context(
            viewport={"width": 1200, "height": 630},
            device_scale_factor=1,
        )
        page = ctx.new_page()
        page.set_content(html, wait_until="networkidle")
        page.wait_for_timeout(400)
        page.screenshot(path=str(out_path), type="png")
        browser.close()
    return out_path


# === CLI entry point ===


def main() -> None:
    """Generate frontend/public/og-card.png and compressed og-card.jpg."""
    path = render_og_thumbnail(OUT_PATH)
    size_kb = path.stat().st_size / 1024.0
    print(f"Generated {path} ({size_kb:.1f} KB)")
    jpg_path = path.with_suffix(".jpg")
    if shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "88", str(path), "--out", str(jpg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"Generated {jpg_path} ({jpg_path.stat().st_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    main()
