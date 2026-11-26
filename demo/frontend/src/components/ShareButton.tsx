import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface ShareButtonProps {
  /** Compact pill style for the /demo masthead, or outline button for the landing hero. */
  variant?: "pill" | "button";
  /** Optional active track title to include in share copy. */
  trackTitle?: string;
  /** Optional active vibe string to encode into a shareable /demo?vibe= deep link. */
  trackVibe?: string;
}

export default function ShareButton({
  variant = "pill",
  trackTitle,
  trackVibe,
}: ShareButtonProps) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const buildShareUrl = useCallback(() => {
    if (typeof window === "undefined") return "";
    const origin = window.location.origin;
    const trimmedVibe = trackVibe?.trim();
    if (trimmedVibe) {
      return `${origin}/demo?vibe=${encodeURIComponent(trimmedVibe)}`;
    }
    return `${origin}${window.location.pathname === "/demo" ? "/demo" : "/"}`;
  }, [trackVibe]);

  const shareUrl = buildShareUrl();
  const shareText = trackTitle
    ? `Listen to "${trackTitle}" sketched on LatentScore: turn vibes into ambient sound.`
    : "LatentScore: turn vibes into ambient sound right in your browser (no GPU required).";

  const handleCopy = () => {
    if (!navigator.clipboard) return;
    void navigator.clipboard
      .writeText(shareUrl)
      .then(() => {
        if (!mountedRef.current) return;
        setCopied(true);
        if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
        copyTimerRef.current = setTimeout(() => {
          setCopied(false);
          copyTimerRef.current = null;
        }, 2000);
      })
      .catch(() => {
        // Ignore clipboard errors on non-secure origins.
      });
  };

  const canNativeShare =
    typeof navigator !== "undefined" && typeof navigator.share === "function";

  const handleNativeShare = () => {
    if (!canNativeShare) return;
    void navigator
      .share({
        title: trackTitle ? `${trackTitle} | LatentScore` : "LatentScore",
        text: shareText,
        url: shareUrl,
      })
      .catch(() => {
        // User dismissed native share sheet.
      });
  };

  const xShareUrl = `https://x.com/intent/tweet?text=${encodeURIComponent(shareText)}&url=${encodeURIComponent(shareUrl)}`;
  const linkedInUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;

  const handleClose = () => {
    setOpen(false);
    triggerRef.current?.focus();
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={variant === "pill" ? "share-trigger-pill" : "btn-outline share-trigger-btn"}
        onClick={() => setOpen(true)}
        aria-label={trackTitle ? `Share ${trackTitle}` : "Share LatentScore"}
      >
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8" />
          <polyline points="16 6 12 2 8 6" />
          <line x1="12" y1="2" x2="12" y2="15" />
        </svg>
        <span>Share</span>
      </button>

      {open &&
        typeof document !== "undefined" &&
        createPortal(
          <div
            className="modal-overlay"
            onClick={handleClose}
            role="dialog"
            aria-modal="true"
            aria-labelledby="share-dialog-title"
          >
            <div className="modal-card share-modal-card" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <span id="share-dialog-title" className="modal-header-title">
                  {trackTitle ? "Share This Soundscape" : "Share LatentScore"}
                </span>
                <button
                  type="button"
                  className="modal-close"
                  onClick={handleClose}
                  aria-label="Close"
                  autoFocus
                >
                  &times;
                </button>
              </div>

              <div className="modal-body share-modal-body">
                <div className="share-preview-frame">
                  <img
                    src="/og-card.png"
                    alt="LatentScore share preview card"
                    className="share-preview-img"
                  />
                </div>

                <div className="share-link-row">
                  <input
                    type="text"
                    readOnly
                    value={shareUrl}
                    className="share-link-input"
                    aria-label="Shareable URL"
                    onFocus={(e) => e.currentTarget.select()}
                  />
                  <button
                    type="button"
                    className="btn-solid share-copy-btn"
                    onClick={handleCopy}
                  >
                    {copied ? (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.4"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    ) : (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                      </svg>
                    )}
                    <span>{copied ? "Copied" : "Copy Link"}</span>
                  </button>
                </div>

                <div className="share-actions-row">
                  <a
                    href={xShareUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-outline share-action-chip"
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                    </svg>
                    <span>Post on X</span>
                  </a>
                  <a
                    href={linkedInUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-outline share-action-chip"
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                    </svg>
                    <span>LinkedIn</span>
                  </a>
                  {canNativeShare && (
                    <button
                      type="button"
                      className="btn-outline share-action-chip"
                      onClick={handleNativeShare}
                    >
                      <svg
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <circle cx="18" cy="5" r="3" />
                        <circle cx="6" cy="12" r="3" />
                        <circle cx="18" cy="19" r="3" />
                        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
                        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
                      </svg>
                      <span>More</span>
                    </button>
                  )}
                  <a
                    href="/og-card.png"
                    download="latentscore-card.png"
                    className="btn-outline share-action-chip"
                  >
                    <svg
                      width="13"
                      height="13"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    <span>Save Card</span>
                  </a>
                </div>
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
