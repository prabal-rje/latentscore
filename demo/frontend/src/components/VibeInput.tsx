import { useMutation } from "@tanstack/react-query";
import { generateFromModel, detectProvider } from "../api";
import { useStore } from "../store";

export default function VibeInput() {
  const vibeText = useStore((s) => s.vibeText);
  const setVibeText = useStore((s) => s.setVibeText);
  const setActiveSession = useStore((s) => s.setActiveSession);
  const addPendingTrack = useStore((s) => s.addPendingTrack);
  const resolvePlaylistTrack = useStore((s) => s.resolvePlaylistTrack);
  const failPlaylistTrack = useStore((s) => s.failPlaylistTrack);
  const selectedModel = useStore((s) => s.selectedModel);
  const customModelId = useStore((s) => s.customModelId);
  const providerApiKeys = useStore((s) => s.providerApiKeys);
  const setError = useStore((s) => s.setError);
  const setWarning = useStore((s) => s.setWarning);
  const setPhase = useStore((s) => s.setGeneratingPhase);

  const apiKey = providerApiKeys[detectProvider(customModelId).provider] || "";

  // Play now — blocks UI, shows phase indicator
  const playMutation = useMutation({
    mutationFn: (v: string) =>
      generateFromModel(v, selectedModel, customModelId, apiKey, setPhase),
    onSuccess: (data, v) => {
      if (data.warning) setWarning(data.warning);
      setActiveSession({
        sessionId: data.session_id,
        title: data.title,
        justification: data.justification,
        palette: data.palette,
        config: data.config,
        vibe: v,
      });
      setVibeText("");
      setPhase(null);
    },
    onError: (err: Error) => { setPhase(null); setError(err.message || "Generation failed"); },
  });

  // Only play blocks the UI
  const isPlayPending = playMutation.isPending;
  const hasText = vibeText.trim().length > 0;

  const handlePlay = () => {
    if (hasText && !isPlayPending) playMutation.mutate(vibeText.trim());
  };

  const handleEnqueue = () => {
    if (!hasText) return;
    const vibe = vibeText.trim();
    setVibeText("");
    const trackId = addPendingTrack(vibe);
    generateFromModel(vibe, selectedModel, customModelId, apiKey)
      .then((data) => {
        resolvePlaylistTrack(trackId, {
          sessionId: data.session_id,
          title: data.title,
          palette: data.palette,
          justification: data.justification,
          config: data.config,
        });
      })
      .catch((err: Error) => {
        failPlaylistTrack(trackId, err.message || "Generation failed");
      });
  };

  return (
    <div className="vibe-input-container">
      <input
        type="text"
        className="vibe-input"
        placeholder="A mood, scene, or feeling..."
        value={vibeText}
        onChange={(e) => setVibeText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && hasText && !isPlayPending) handlePlay();
        }}
        disabled={isPlayPending}
      />
      {/* Play Now */}
      <button
        className="vibe-submit vibe-play"
        onClick={handlePlay}
        disabled={isPlayPending || !hasText}
        title="Play now"
      >
        {isPlayPending ? (
          <span className="spinner" />
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>
      {/* Enqueue */}
      <button
        className="vibe-submit vibe-enqueue"
        onClick={handleEnqueue}
        disabled={isPlayPending || !hasText}
        title="Add to playlist"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </button>
    </div>
  );
}
