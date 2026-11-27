import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import AuroraCanvas from "./AuroraCanvas";
import BrandWordmark from "./BrandWordmark";
import PixelCharm from "./PixelCharm";
import ShareButton from "./ShareButton";
import { usePageSeo } from "../seo";

const BIBTEX = `@inproceedings{gupta2026latentscore,
  author    = {Gupta, Prabal},
  title     = {LatentScore: Sketching Soundscapes with
               LLM-Distilled Retrieval for
               Procedural Synthesis},
  booktitle = {SIGGRAPH Talks '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3799818.3812120}
}`;

const FEATURES = [
  {
    title: "Text to Music",
    desc: "Describe a mood or scene in plain words and hear an ambient soundscape right away.",
  },
  {
    title: "No GPU Required",
    desc: "Pure CPU synthesis with NumPy renders 2 seconds of audio in 23 ms on ordinary hardware.",
  },
  {
    title: "Runs in Your Browser",
    desc: "The live studio runs quantized embeddings and audio synthesis locally on your device.",
  },
  {
    title: "Live Parameter Tuning",
    desc: "Adjust 18 musical parameters including tempo, mode, texture, reverb, and stereo spread.",
  },
  {
    title: "Reactive Daylight Color",
    desc: "Each vibe generates a cohesive color palette that tints the aurora canvas softly.",
  },
  {
    title: "Python, JS/TS, and Rust SDKs",
    desc: "Install via pip, npm (zero dependencies), or Cargo to build reactive local music apps.",
  },
];

const PLATFORM_LINKS = [
  {
    label: "GitHub",
    desc: "Source code, library docs, and issues",
    href: "https://github.com/prabal-rje/latentscore",
  },
  {
    label: "Google Colab",
    desc: "Run the interactive quickstart notebook",
    href: "https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb",
  },
  {
    label: "PyPI",
    desc: "pip install latentscore",
    href: "https://pypi.org/project/latentscore/",
  },
  {
    label: "npm",
    desc: "npm install latentscore (pure TypeScript)",
    href: "https://www.npmjs.com/package/latentscore",
  },
  {
    label: "crates.io",
    desc: "cargo add latentscore (pure Rust)",
    href: "https://crates.io/crates/latentscore",
  },
  {
    label: "LinkedIn",
    desc: "Connect with Prabal Gupta",
    href: "https://www.linkedin.com/in/prabal1997",
  },
  {
    label: "X",
    desc: "Follow @prabal_",
    href: "https://x.com/prabal_",
  },
];

export default function LandingPage() {
  usePageSeo({
    title: "LatentScore: Local Procedural Ambient Music from Text (Python, JS, Rust)",
    description:
      "Generate calm, reactive ambient music from text prompts locally with no GPU. Try the free browser studio or use the open-source Python, TypeScript, and Rust SDKs.",
    canonicalPath: "/",
  });
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
    };
  }, []);

  const handleCopyBibtex = () => {
    if (!navigator.clipboard) return;
    void navigator.clipboard
      .writeText(BIBTEX)
      .then(() => {
        if (!mountedRef.current) return;
        setCopied(true);
        if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
        copyTimerRef.current = setTimeout(() => {
          setCopied(false);
          copyTimerRef.current = null;
        }, 1800);
      })
      .catch(() => {
        // Ignore clipboard denial on non-secure origins.
      });
  };

  return (
    <>
      <AuroraCanvas reactive={false} />

      <div className="alba-page">
        <header className="masthead">
          <div className="measure masthead-inner">
            <BrandWordmark />

            <nav className="masthead-right" aria-label="Primary links">
              <Link to="/demo" className="quiet-link">
                Live Demo
              </Link>
              <a
                href="https://github.com/prabal-rje/latentscore"
                target="_blank"
                rel="noopener noreferrer"
                className="quiet-link"
              >
                GitHub
              </a>
              <a
                href="https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb"
                target="_blank"
                rel="noopener noreferrer"
                className="quiet-link"
              >
                Colab
              </a>
            </nav>
          </div>
        </header>

        <main className="landing-main">
          <div className="measure">
            <section aria-label="Introduction">
              <PixelCharm initialIndex={0} size={60} />

              <div className="hero-kicker">
                <span className="hero-kicker-dot" />
                SIGGRAPH Talks &apos;26 &middot; Open Source
              </div>

              <h1 className="hero-title">
                Turn vibes into{" "}
                <span className="no-orphan">
                  <em>ambient sound</em>.
                </span>
              </h1>

              <p className="hero-desc">
                LatentScore is an open-source procedural ambient music studio and reactive SDK for
                browsers, Python, TypeScript, and Rust. No GPU required.
              </p>

              <div className="hero-actions">
                <Link to="/demo" className="btn-solid">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                  Open Live Demo
                </Link>
                <a
                  href="https://github.com/prabal-rje/latentscore"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-outline"
                >
                  View on GitHub
                </a>
                <a
                  href="https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-outline"
                >
                  Google Colab
                </a>
                <ShareButton variant="button" />
              </div>
            </section>

            <section className="landing-section" aria-label="Capabilities">
              <div className="section-kicker">What it does</div>
              <div className="feature-rows">
                {FEATURES.map((item) => (
                  <div key={item.title}>
                    <h2 className="feature-row-title">{item.title}</h2>
                    <p className="feature-row-desc">{item.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            <section className="landing-section" aria-label="Explore platforms">
              <div className="section-kicker">Explore LatentScore</div>
              <div className="link-sparks">
                <Link to="/demo" className="link-spark">
                  <span className="link-spark-left">
                    <span className="link-spark-label">Live Demo</span>
                    <span className="link-spark-desc">Try synthesis directly in your browser</span>
                  </span>
                  <span className="link-spark-arrow" aria-hidden="true">&rarr;</span>
                </Link>
                {PLATFORM_LINKS.map((item) => (
                  <a
                    key={item.label}
                    href={item.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="link-spark"
                  >
                    <span className="link-spark-left">
                      <span className="link-spark-label">{item.label}</span>
                      <span className="link-spark-desc">{item.desc}</span>
                    </span>
                    <span className="link-spark-arrow" aria-hidden="true">&rarr;</span>
                  </a>
                ))}
              </div>
            </section>

            <section className="landing-section" aria-label="Citation">
              <div className="section-kicker">Citing LatentScore</div>
              <div className="citation-box">
                <div className="citation-header">
                  <span>BibTeX</span>
                  <button
                    type="button"
                    className="citation-copy-btn"
                    onClick={handleCopyBibtex}
                  >
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
                <pre>{BIBTEX}</pre>
              </div>
            </section>

            <footer className="landing-footer">
              LatentScore &middot; Apache-2.0 License &middot; Crafted by Prabal Gupta
            </footer>
          </div>
        </main>
      </div>
    </>
  );
}
