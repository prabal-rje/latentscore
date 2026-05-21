import { useRef, useCallback, useEffect } from "react";
import { useStore } from "../store";

export default function AudioPlayer() {
  const {
    activeSessionId,
    isPlaying,
    setPlaying,
    activeTitle,
    activePalette,
    toggleParamPanel,
    togglePlaylistPanel,
    activeVibe,
    addToPlaylist,
    activeConfig,
    activeJustification,
  } = useStore();

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef(0);
  const queueRef = useRef<AudioBuffer[]>([]);
  const isSchedulingRef = useRef(false);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);

  const stopAllAudio = useCallback(() => {
    // Stop all scheduled/playing source nodes
    for (const src of activeSourcesRef.current) {
      try { src.stop(); } catch { /* already stopped */ }
    }
    activeSourcesRef.current = [];
    queueRef.current = [];
    nextPlayTimeRef.current = 0;
  }, []);

  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === "closed") {
      const ctx = new AudioContext({ sampleRate: 44100 });
      // iOS silent-mode unlock: play a tiny silent buffer within the user gesture
      const buf = ctx.createBuffer(1, 1, 22050);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start();
      audioContextRef.current = ctx;
    }
    return audioContextRef.current;
  }, []);

  const scheduleBuffers = useCallback(() => {
    if (isSchedulingRef.current) return;
    isSchedulingRef.current = true;

    const ctx = getAudioContext();
    while (queueRef.current.length > 0) {
      const buffer = queueRef.current.shift()!;
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);

      const now = ctx.currentTime;
      const when = Math.max(now + 0.01, nextPlayTimeRef.current);
      source.start(when);
      nextPlayTimeRef.current = when + buffer.duration;

      // Track this source so we can stop it later
      activeSourcesRef.current.push(source);
      source.onended = () => {
        const idx = activeSourcesRef.current.indexOf(source);
        if (idx !== -1) activeSourcesRef.current.splice(idx, 1);
      };
    }

    isSchedulingRef.current = false;
  }, [getAudioContext]);

  const startStream = useCallback(() => {
    if (!activeSessionId) return;

    // Kill everything from any previous stream
    if (wsRef.current) {
      wsRef.current.onclose = null; // prevent setPlaying(false) from old ws
      wsRef.current.close();
      wsRef.current = null;
    }
    stopAllAudio();

    const ctx = getAudioContext();
    if (ctx.state === "suspended") ctx.resume();
    nextPlayTimeRef.current = ctx.currentTime + 0.1;

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host;
    const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/stream/${activeSessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setPlaying(true);
    };

    ws.onmessage = async (event) => {
      const msg = JSON.parse(event.data);

      if (msg.type === "audio_chunk") {
        const bytes = Uint8Array.from(atob(msg.data), (c) => c.charCodeAt(0));
        try {
          const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
          queueRef.current.push(audioBuffer);
          scheduleBuffers();
        } catch (e) {
          console.warn("Failed to decode audio chunk", e);
        }
      } else if (msg.type === "config_updated") {
        useStore.getState().updateConfig(msg.config);
        useStore.getState().updatePalette(msg.palette);
        useStore.getState().updateTitle(msg.title);
      }
    };

    ws.onclose = () => {
      setPlaying(false);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error", err);
      setPlaying(false);
    };
  }, [activeSessionId, getAudioContext, scheduleBuffers, setPlaying, stopAllAudio]);

  const stopStream = useCallback(() => {
    // Close WebSocket (prevent onclose from firing setPlaying again)
    if (wsRef.current) {
      wsRef.current.onclose = null;
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "stop" }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }

    // Actually stop all audio that's playing or scheduled
    stopAllAudio();
    setPlaying(false);
  }, [setPlaying, stopAllAudio]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      stopAllAudio();
      audioContextRef.current?.close();
    };
  }, [stopAllAudio]);

  const handleAddToPlaylist = () => {
    if (!activeSessionId) return;
    addToPlaylist({
      id: `track-${Date.now()}`,
      sessionId: activeSessionId,
      vibe: activeVibe,
      title: activeTitle,
      palette: activePalette,
      justification: activeJustification,
      config: activeConfig,
      duration: 30,
      status: "ready",
    });
  };

  return (
    <div className="player-bar">
      {/* Left: now playing info */}
      <div className="player-left">
        {activeSessionId && (
          <div className="now-playing">
            <div className="now-playing-colors">
              {activePalette.slice(0, 3).map((color, i) => (
                <span
                  key={i}
                  className="color-dot"
                  style={{ backgroundColor: color, color }}
                />
              ))}
            </div>
            <div className="now-playing-info">
              <span className="now-playing-title">{activeTitle}</span>
              <span className="now-playing-vibe">{activeVibe}</span>
            </div>
          </div>
        )}
      </div>

      {/* Center: play button - always centered */}
      <div className="player-center">
        <button
          className={`play-btn ${isPlaying ? "playing" : ""}`}
          onClick={isPlaying ? stopStream : startStream}
          disabled={!activeSessionId}
          title={!activeSessionId ? "Select a vibe first" : isPlaying ? "Stop" : "Play"}
        >
          {isPlaying ? (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>
      </div>

      {/* Right: action buttons */}
      <div className="player-right">
        <button
          className="action-btn"
          onClick={toggleParamPanel}
          disabled={!activeSessionId}
          title="Tune Parameters"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="4" y1="21" x2="4" y2="14" />
            <line x1="4" y1="10" x2="4" y2="3" />
            <line x1="12" y1="21" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12" y2="3" />
            <line x1="20" y1="21" x2="20" y2="16" />
            <line x1="20" y1="12" x2="20" y2="3" />
            <line x1="1" y1="14" x2="7" y2="14" />
            <line x1="9" y1="8" x2="15" y2="8" />
            <line x1="17" y1="16" x2="23" y2="16" />
          </svg>
        </button>
        <button
          className="action-btn"
          onClick={handleAddToPlaylist}
          disabled={!activeSessionId}
          title="Add to Playlist"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>
        <button
          className="action-btn"
          onClick={togglePlaylistPanel}
          title="Playlist"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="8" y1="6" x2="21" y2="6" />
            <line x1="8" y1="12" x2="21" y2="12" />
            <line x1="8" y1="18" x2="21" y2="18" />
            <line x1="3" y1="6" x2="3.01" y2="6" />
            <line x1="3" y1="12" x2="3.01" y2="12" />
            <line x1="3" y1="18" x2="3.01" y2="18" />
          </svg>
        </button>
      </div>
    </div>
  );
}
