import { useRef, useEffect } from "react";
import { Link } from "react-router-dom";

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [100, 90, 75];
}

function LandingParticles() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const mouse = { x: -1000, y: -1000, active: false };
    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.active = true;
    };
    const handleMouseLeave = () => { mouse.active = false; };
    window.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", handleMouseLeave);

    const colors = ["#d4a574", "#c49670", "#e8c9a0", "#b08860", "#dbb88a"];
    interface P { x: number; y: number; vx: number; vy: number; s: number; c: string; a: number; }
    const particles: P[] = [];
    // Density scales with screen area — fewer on mobile, more on desktop
    const area = canvas.width * canvas.height;
    const count = Math.min(100, Math.max(18, Math.floor(area / 18000)));
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        s: 0.8 + Math.random() * 2.5,
        c: colors[Math.floor(Math.random() * colors.length)],
        a: 0.06 + Math.random() * 0.2,
      });
    }

    let t = 0;
    let raf: number;
    const animate = () => {
      t++;
      ctx.fillStyle = "#141210";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Subtle ambient glow
      const grad = ctx.createRadialGradient(
        canvas.width * 0.25, canvas.height * 0.35, 0,
        canvas.width * 0.25, canvas.height * 0.35, canvas.width * 0.6,
      );
      grad.addColorStop(0, "rgba(160, 140, 120, 0.015)");
      grad.addColorStop(1, "transparent");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Mouse glow
      if (mouse.active) {
        const mg = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 180);
        mg.addColorStop(0, "rgba(160, 140, 120, 0.04)");
        mg.addColorStop(1, "transparent");
        ctx.fillStyle = mg;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      for (const p of particles) {
        // Mouse attraction
        if (mouse.active) {
          const dx = mouse.x - p.x;
          const dy = mouse.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 250 && dist > 1) {
            const force = (1 - dist / 250) * 1.5;
            p.vx += (dx / dist) * force * 0.3;
            p.vy += (dy / dist) * force * 0.3;
            p.vx *= 0.92;
            p.vy *= 0.92;
          }
        }

        p.x += p.vx + Math.sin(t * 0.004 + p.y * 0.002) * 0.15;
        p.y += p.vy + Math.cos(t * 0.003 + p.x * 0.002) * 0.15;
        if (p.x < -10) p.x = canvas.width + 10;
        if (p.x > canvas.width + 10) p.x = -10;
        if (p.y < -10) p.y = canvas.height + 10;
        if (p.y > canvas.height + 10) p.y = -10;

        const [r, g, b] = hexToRgb(p.c);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.s, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${p.a})`;
        ctx.fill();

        // Glow halo for larger particles
        if (p.s > 1.5) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.s * 3, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},${p.a * 0.06})`;
          ctx.fill();
        }
      }

      // Mouse connection lines
      if (mouse.active) {
        const nearby = particles
          .map((p) => ({ p, d: Math.sqrt((p.x - mouse.x) ** 2 + (p.y - mouse.y) ** 2) }))
          .filter((e) => e.d < 150)
          .sort((a, b) => a.d - b.d)
          .slice(0, 8);
        for (const { p, d } of nearby) {
          const alpha = (1 - d / 150) * 0.1;
          const [r, g, b] = hexToRgb(p.c);
          ctx.beginPath();
          ctx.moveTo(mouse.x, mouse.y);
          ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      // Inter-particle connections (nearest neighbors)
      const subset = particles.slice(0, 50);
      for (let i = 0; i < subset.length; i++) {
        for (let j = i + 1; j < subset.length; j++) {
          const dx = subset[i].x - subset[j].x;
          const dy = subset[i].y - subset[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 60) {
            const alpha = (1 - dist / 60) * 0.03;
            const [r, g, b] = hexToRgb(subset[i].c);
            ctx.beginPath();
            ctx.moveTo(subset[i].x, subset[i].y);
            ctx.lineTo(subset[j].x, subset[j].y);
            ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
            ctx.lineWidth = 0.3;
            ctx.stroke();
          }
        }
      }

      raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "fixed", top: 0, left: 0, width: "100%", height: "100%", zIndex: 0 }}
    />
  );
}

export default function LandingPage() {
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --ivory: #f0ece4;
          --ivory-dim: rgba(240, 236, 228, 0.45);
          --ivory-faint: rgba(240, 236, 228, 0.12);
          --ivory-ghost: rgba(240, 236, 228, 0.06);
          --warm-black: #141210;
          --accent: #d4a574;
          --font-display: 'DM Serif Display', Georgia, serif;
          --font-body: 'IBM Plex Mono', 'Menlo', monospace;
        }

        body {
          font-family: var(--font-body);
          background: var(--warm-black);
          color: var(--ivory);
          overflow-x: hidden;
          margin: 0;
        }

        /* Grain */
        .grain {
          position: fixed;
          top: 0; left: 0; right: 0; bottom: 0;
          z-index: 1;
          pointer-events: none;
          opacity: 0.035;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
          background-repeat: repeat;
          background-size: 256px 256px;
        }

        .landing {
          position: relative;
          z-index: 2;
          min-height: 100vh;
        }

        /* Staggered reveal */
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .reveal { opacity: 0; animation: fadeUp 0.7s ease forwards; }
        .r1 { animation-delay: 0.1s; }
        .r2 { animation-delay: 0.25s; }
        .r3 { animation-delay: 0.45s; }
        .r4 { animation-delay: 0.65s; }
        .r5 { animation-delay: 0.85s; }

        /* ─── Hero ─── */
        .hero {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          justify-content: center;
          padding: 80px 56px;
          max-width: 900px;
          margin: 0 auto;
        }

        .hero-kicker {
          font-size: 13px;
          font-weight: 400;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: var(--ivory-dim);
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .hero-kicker-dot {
          width: 5px; height: 5px;
          border-radius: 50%;
          background: var(--accent);
          animation: pulse-dot 2.5s ease-in-out infinite;
        }

        @keyframes pulse-dot {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }

        .hero-title {
          font-family: var(--font-display);
          font-size: clamp(56px, 9vw, 110px);
          font-weight: 400;
          line-height: 0.95;
          color: var(--ivory);
          margin-bottom: 24px;
        }

        .hero-title em {
          font-style: italic;
          color: var(--accent);
        }

        .hero-desc {
          font-size: 19px;
          font-weight: 300;
          color: var(--ivory-dim);
          line-height: 1.7;
          max-width: 560px;
          margin-bottom: 48px;
        }

        .hero-actions {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }

        /* Shared glass style */
        .glass {
          background: rgba(26, 24, 22, 0.5);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(240, 236, 228, 0.1);
        }

        .btn {
          display: inline-flex;
          align-items: center;
          gap: 10px;
          padding: 14px 28px;
          border-radius: 6px;
          font-family: var(--font-body);
          font-size: 16px;
          font-weight: 400;
          text-decoration: none;
          cursor: pointer;
          transition: all 0.2s ease;
          letter-spacing: 0.5px;
        }

        .btn-primary {
          background: var(--ivory);
          color: var(--warm-black);
          border: 1px solid var(--ivory);
        }

        .btn-primary:hover {
          background: var(--accent);
          border-color: var(--accent);
        }

        .btn-ghost {
          background: rgba(26, 24, 22, 0.5);
          color: var(--ivory-dim);
          border: 1px solid rgba(240, 236, 228, 0.1);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
        }

        .btn-ghost:hover {
          color: var(--ivory);
          border-color: var(--ivory-dim);
          background: rgba(26, 24, 22, 0.65);
        }

        /* ─── Features ─── */
        .features {
          padding: 0 56px 120px;
          max-width: 900px;
          margin: 0 auto;
        }

        .section-label {
          font-size: 12px;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: var(--ivory-dim);
          margin-bottom: 32px;
          opacity: 0.6;
        }

        .feature-list {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1px;
          background: var(--ivory-ghost);
          border: 1px solid rgba(240, 236, 228, 0.1);
          border-radius: 8px;
          overflow: hidden;
        }

        .feature-item {
          padding: 32px;
          background: rgba(26, 24, 22, 0.55);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
        }

        .feature-item-title {
          font-family: var(--font-display);
          font-size: 28px;
          font-weight: 400;
          margin-bottom: 10px;
          color: var(--ivory);
        }

        .feature-item-desc {
          font-size: 16px;
          font-weight: 300;
          color: var(--ivory-dim);
          line-height: 1.65;
        }

        /* ─── Links ─── */
        .links-section {
          padding: 80px 56px;
          max-width: 900px;
          margin: 0 auto;
          border-top: 1px solid var(--ivory-ghost);
        }

        .links-heading {
          font-family: var(--font-display);
          font-size: 36px;
          font-weight: 400;
          margin-bottom: 8px;
        }

        .links-sub {
          font-size: 16px;
          font-weight: 300;
          color: var(--ivory-dim);
          margin-bottom: 32px;
        }

        .links-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }

        .link-card {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px 18px;
          border: 1px solid rgba(240, 236, 228, 0.1);
          border-radius: 6px;
          text-decoration: none;
          color: var(--ivory);
          transition: all 0.2s ease;
          background: rgba(26, 24, 22, 0.5);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
        }

        .link-card:hover {
          border-color: var(--ivory-faint);
          background: rgba(26, 24, 22, 0.65);
        }

        .link-card-icon {
          width: 32px; height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
        }

        .link-card-text {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .link-card-label {
          font-size: 15px;
          font-weight: 400;
        }

        .link-card-desc {
          font-size: 13px;
          font-weight: 300;
          color: var(--ivory-dim);
        }

        /* ─── Citation ─── */
        .citation {
          padding: 60px 56px 80px;
          max-width: 900px;
          margin: 0 auto;
          border-top: 1px solid var(--ivory-ghost);
        }

        .citation-heading {
          font-family: var(--font-display);
          font-size: 28px;
          font-weight: 400;
          margin-bottom: 20px;
        }

        .citation-block {
          padding: 24px 28px;
          border: 1px solid rgba(240, 236, 228, 0.1);
          border-radius: 6px;
          background: rgba(26, 24, 22, 0.5);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          position: relative;
        }

        .citation-block pre {
          margin: 0;
          white-space: pre-wrap;
          font-family: var(--font-body);
          font-size: 14px;
          font-weight: 300;
          line-height: 1.7;
          color: var(--ivory-dim);
        }

        .citation-copy {
          position: absolute;
          top: 12px; right: 12px;
          padding: 6px 14px;
          border-radius: 4px;
          border: 1px solid var(--ivory-faint);
          background: rgba(26, 24, 22, 0.6);
          color: var(--ivory-dim);
          font-size: 12px;
          font-family: var(--font-body);
          cursor: pointer;
          transition: all 0.2s ease;
          letter-spacing: 0.5px;
        }

        .citation-copy:hover {
          color: var(--ivory);
          border-color: var(--ivory-dim);
        }

        /* ─── Footer ─── */
        .landing-footer {
          padding: 32px 56px;
          max-width: 900px;
          margin: 0 auto;
          border-top: 1px solid var(--ivory-ghost);
          color: var(--ivory-dim);
          font-size: 14px;
          font-weight: 300;
          opacity: 0.5;
        }

        /* ─── Responsive ─── */
        @media (max-width: 768px) {
          .hero { padding: 60px 24px; }
          .features { padding: 0 24px 80px; }
          .feature-list { grid-template-columns: 1fr; }
          .links-section { padding: 60px 24px; }
          .links-grid { grid-template-columns: 1fr 1fr; }
          .citation { padding: 40px 24px 60px; }
          .landing-footer { padding: 24px; }
        }

        @media (max-width: 480px) {
          .links-grid { grid-template-columns: 1fr; }
        }
      `}</style>

      <LandingParticles />
      <div className="grain" />

      <div className="landing">
        {/* Hero */}
        <section className="hero">
          <div className="hero-kicker reveal r1">
            <span className="hero-kicker-dot" />
            open source
          </div>

          <h1 className="hero-title reveal r2">
            Latent<em>Score</em>
          </h1>

          <p className="hero-desc reveal r3">
            A Python library that generates ambient music from text descriptions.
            No GPU required. Turn vibes into sound with a single line of code.
          </p>

          <div className="hero-actions reveal r4">
            <Link to="/demo" className="btn btn-primary">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z" />
              </svg>
              Live Demo
            </Link>
            <a
              href="https://github.com/prabal-rje/latentscore"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              GitHub
            </a>
            <a
              href="http://localhost:8889/lab/tree/quickstart.ipynb"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              </svg>
              JupyterLab
            </a>
          </div>
        </section>

        {/* Features */}
        <section className="features">
          <div className="section-label">What it does</div>
          <div className="feature-list">
            <div className="feature-item">
              <div className="feature-item-title">Text to Music</div>
              <div className="feature-item-desc">
                Describe a vibe in natural language and get ambient music instantly.
                No musical training required.
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-item-title">No GPU Required</div>
              <div className="feature-item-desc">
                Pure CPU synthesis using NumPy. Renders 2 seconds of audio in 23ms.
                Runs on any machine.
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-item-title">Live Tuning</div>
              <div className="feature-item-desc">
                Adjust 18+ musical parameters in real-time during playback.
                Tempo, mode, texture, and more.
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-item-title">Visual Feedback</div>
              <div className="feature-item-desc">
                Particle effects driven by the generated color palette.
                Every piece looks as unique as it sounds.
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-item-title">Playlists</div>
              <div className="feature-item-desc">
                Build playlists of different vibes. Cross-fade between tracks
                for seamless live performance sets.
              </div>
            </div>
            <div className="feature-item">
              <div className="feature-item-title">pip install</div>
              <div className="feature-item-desc">
                One command to install. Use as a Python library, CLI tool,
                or through this web interface.
              </div>
            </div>
          </div>
        </section>

        {/* Links */}
        <section className="links-section">
          <h2 className="links-heading">Get Started</h2>
          <p className="links-sub">Explore LatentScore across platforms</p>

          <div className="links-grid">
            <Link to="/demo" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="var(--ivory)">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">Live Demo</span>
                <span className="link-card-desc">Try in your browser</span>
              </div>
            </Link>

            <a href="https://github.com/prabal-rje/latentscore" target="_blank" rel="noopener noreferrer" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="var(--ivory)">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">GitHub</span>
                <span className="link-card-desc">Source code & docs</span>
              </div>
            </a>

            {/* YouTube link — uncomment when demo video is ready
            <a href="https://www.youtube.com/@prabal-rje" target="_blank" rel="noopener noreferrer" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="#ef4444">
                  <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">YouTube</span>
                <span className="link-card-desc">Watch the demo</span>
              </div>
            </a>
            */}

            <a href="https://colab.research.google.com/github/prabal-rje/latentscore/blob/main/notebooks/quickstart-colab.ipynb" target="_blank" rel="noopener noreferrer" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <polygon points="10,8 16,12 10,16" fill="#f59e0b" stroke="none" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">Google Colab</span>
                <span className="link-card-desc">Run in notebook</span>
              </div>
            </a>

            <a
              href="http://localhost:8889/lab/tree/quickstart.ipynb"
              target="_blank"
              rel="noopener noreferrer"
              className="link-card"
            >
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--ivory)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">Local Jupyter</span>
                <span className="link-card-desc">SDK playground on this machine</span>
              </div>
            </a>

            <a href="https://www.linkedin.com/in/prabal1997" target="_blank" rel="noopener noreferrer" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="#0a66c2">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">LinkedIn</span>
                <span className="link-card-desc">Connect with the author</span>
              </div>
            </a>

            <a href="https://pypi.org/project/latentscore/" target="_blank" rel="noopener noreferrer" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="#3776ab">
                  <path d="M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.68H3.545l-.67-.04-.55-.13-.42-.22-.31-.3-.22-.36-.14-.4-.07-.43-.02-.45V7.5l.06-.57.17-.52.27-.44.35-.37.43-.3.49-.24.53-.18.56-.12.57-.08.56-.04.53-.01.49.02.44.04.38.07.33.08.28.09.22.1.17.1.11.1.07.1.04.1.01.1-.01.1L14.25.18z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">PyPI</span>
                <span className="link-card-desc">pip install latentscore</span>
              </div>
            </a>

            <a href="https://x.com/prabal_" target="_blank" rel="noopener noreferrer" className="link-card">
              <div className="link-card-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="var(--ivory)">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                </svg>
              </div>
              <div className="link-card-text">
                <span className="link-card-label">X</span>
                <span className="link-card-desc">Follow @prabal_</span>
              </div>
            </a>
          </div>
        </section>

        {/* Citation */}
        <section className="citation">
          <h2 className="citation-heading">Citing LatentScore</h2>
          <div className="citation-block">
            <button
              className="citation-copy"
              onClick={() => {
                navigator.clipboard.writeText(
                  `@inproceedings{gupta2026latentscore,\n  author    = {Gupta, Prabal},\n  title     = {LatentScore: Sketching Soundscapes with\n               LLM-Distilled Retrieval for\n               Procedural Synthesis},\n  booktitle = {SIGGRAPH Talks '26},\n  year      = {2026},\n  publisher = {ACM},\n  doi       = {10.1145/3799818.3812120}\n}`
                );
              }}
            >
              Copy BibTeX
            </button>
            <pre>
{`@inproceedings{gupta2026latentscore,
  author    = {Gupta, Prabal},
  title     = {LatentScore: Sketching Soundscapes with
               LLM-Distilled Retrieval for
               Procedural Synthesis},
  booktitle = {SIGGRAPH Talks '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3799818.3812120}
}`}
            </pre>
          </div>
        </section>

        <footer className="landing-footer">
          LatentScore &middot; Open Source &middot; Apache-2.0 License
        </footer>
      </div>
    </>
  );
}
