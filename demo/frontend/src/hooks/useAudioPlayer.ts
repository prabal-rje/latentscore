import { useRef, useCallback, useEffect } from "react";
import { useStore } from "../store";
import { renderAudio } from "../api";

function encodeWav(buffers: AudioBuffer[]): Blob {
  if (buffers.length === 0) return new Blob();
  const sampleRate = buffers[0].sampleRate;
  const numChannels = buffers[0].numberOfChannels;
  let totalLength = 0;
  for (const b of buffers) totalLength += b.length;

  const result = new Float32Array(totalLength);
  let offset = 0;
  for (const b of buffers) {
    const ch0 = b.getChannelData(0);
    if (numChannels > 1) {
      const ch1 = b.getChannelData(1);
      for (let i = 0; i < ch0.length; i++) {
        result[offset + i] = (ch0[i] + ch1[i]) / 2;
      }
    } else {
      result.set(ch0, offset);
    }
    offset += b.length;
  }

  const dataLength = result.length * 2;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  const writeStr = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataLength, true);

  let p = 44;
  for (let i = 0; i < result.length; i++) {
    const s = Math.max(-1, Math.min(1, result[i]));
    view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    p += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export function useAudioPlayer(onStreamEnd?: () => void) {
  const activeSessionId = useStore((s) => s.activeSessionId);
  const streamMode = useStore((s) => s.streamMode);
  const setPlaying = useStore((s) => s.setPlaying);

  const onStreamEndRef = useRef(onStreamEnd);
  onStreamEndRef.current = onStreamEnd;

  // Shared refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const recordedBuffersRef = useRef<AudioBuffer[]>([]);

  // Stream mode refs
  const wsRef = useRef<WebSocket | null>(null);
  const nextPlayTimeRef = useRef(0);
  const queueRef = useRef<AudioBuffer[]>([]);
  const isSchedulingRef = useRef(false);

  // Fetch mode refs
  const fetchAbortRef = useRef<AbortController | null>(null);

  // Used by WS onclose to restart stream when looping
  const startWsStreamRef = useRef<() => void>(() => {});

  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === "closed") {
      const ctx = new AudioContext({ sampleRate: 44100 });
      // iOS silent-mode unlock: play a tiny silent buffer within the user gesture
      // call stack so the context is treated as user-activated.
      const buf = ctx.createBuffer(1, 1, 22050);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start();
      audioContextRef.current = ctx;
    }
    return audioContextRef.current;
  }, []);

  const stopAllAudio = useCallback(() => {
    for (const src of activeSourcesRef.current) {
      try { src.stop(); } catch { /* already stopped */ }
    }
    activeSourcesRef.current = [];
    queueRef.current = [];
    nextPlayTimeRef.current = 0;
  }, []);

  // ─── Stream mode: WebSocket with phase-continuous chunks ──────────

  const scheduleBuffers = useCallback(() => {
    if (isSchedulingRef.current) return;
    isSchedulingRef.current = true;
    const ctx = getAudioContext();
    while (queueRef.current.length > 0) {
      const buffer = queueRef.current.shift()!;
      recordedBuffersRef.current.push(buffer);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const now = ctx.currentTime;
      const when = Math.max(now + 0.01, nextPlayTimeRef.current);
      source.start(when);
      nextPlayTimeRef.current = when + buffer.duration;
      activeSourcesRef.current.push(source);
      source.onended = () => {
        const idx = activeSourcesRef.current.indexOf(source);
        if (idx !== -1) activeSourcesRef.current.splice(idx, 1);
      };
    }
    isSchedulingRef.current = false;
  }, [getAudioContext]);

  const startWsStream = useCallback(() => {
    if (!activeSessionId) return;
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    stopAllAudio();
    recordedBuffersRef.current = [];
    const ctx = getAudioContext();
    if (ctx.state === "suspended") ctx.resume();
    nextPlayTimeRef.current = ctx.currentTime + 0.1;
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsHost = window.location.host;
    const ws = new WebSocket(`${wsProtocol}//${wsHost}/ws/stream/${activeSessionId}`);
    wsRef.current = ws;
    ws.onopen = () => { setPlaying(true); };
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
      }
    };
    ws.onclose = () => {
      if (useStore.getState().loopMode && wsRef.current === ws) {
        // Loop: reconnect and keep playing
        startWsStreamRef.current();
      } else {
        setPlaying(false);
        onStreamEndRef.current?.();
      }
    };
    ws.onerror = (err) => {
      console.error("WebSocket error", err);
      setPlaying(false);
    };
  }, [activeSessionId, getAudioContext, scheduleBuffers, setPlaying, stopAllAudio]);

  startWsStreamRef.current = startWsStream;

  const stopWsStream = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null; // clear before close so onclose loop check fails
    if (ws) {
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
      }
      ws.close();
    }
    stopAllAudio();
    setPlaying(false);
  }, [setPlaying, stopAllAudio]);

  // ─── Fetch mode: render full audio and play ───────────────────────

  const startFetchPlay = useCallback(() => {
    if (!activeSessionId) return;
    // Abort any in-flight fetch
    fetchAbortRef.current?.abort();
    stopAllAudio();
    recordedBuffersRef.current = [];

    const ctx = getAudioContext();
    if (ctx.state === "suspended") ctx.resume();

    const abort = new AbortController();
    fetchAbortRef.current = abort;
    setPlaying(true);

    renderAudio(activeSessionId, 60)
      .then((wavBytes) => {
        if (abort.signal.aborted) return;
        return ctx.decodeAudioData(wavBytes);
      })
      .then((audioBuffer) => {
        if (!audioBuffer || abort.signal.aborted) return;
        recordedBuffersRef.current.push(audioBuffer);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.loop = useStore.getState().loopMode;
        source.connect(ctx.destination);
        source.start(0);
        activeSourcesRef.current.push(source);
        source.onended = () => {
          const idx = activeSourcesRef.current.indexOf(source);
          if (idx !== -1) activeSourcesRef.current.splice(idx, 1);
          if (!abort.signal.aborted) {
            setPlaying(false);
            onStreamEndRef.current?.();
          }
        };
      })
      .catch((err) => {
        if (abort.signal.aborted) return;
        console.error("Fetch play error", err);
        setPlaying(false);
      });
  }, [activeSessionId, getAudioContext, setPlaying, stopAllAudio]);

  const stopFetchPlay = useCallback(() => {
    fetchAbortRef.current?.abort();
    fetchAbortRef.current = null;
    stopAllAudio();
    setPlaying(false);
  }, [setPlaying, stopAllAudio]);

  // ─── Unified interface ────────────────────────────────────────────

  const startStream = useCallback(() => {
    if (streamMode) {
      startWsStream();
    } else {
      startFetchPlay();
    }
  }, [streamMode, startWsStream, startFetchPlay]);

  const stopStream = useCallback(() => {
    if (streamMode) {
      stopWsStream();
    } else {
      stopFetchPlay();
    }
  }, [streamMode, stopWsStream, stopFetchPlay]);

  const downloadAudio = useCallback((title: string) => {
    if (recordedBuffersRef.current.length === 0) return;
    const blob = encodeWav(recordedBuffersRef.current);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/[^a-zA-Z0-9 ]/g, "").replace(/\s+/g, "_")}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  const hasRecordedAudio = useCallback(() => {
    return recordedBuffersRef.current.length > 0;
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
      fetchAbortRef.current?.abort();
      stopAllAudio();
      audioContextRef.current?.close();
    };
  }, [stopAllAudio]);

  return { startStream, stopStream, downloadAudio, hasRecordedAudio };
}
