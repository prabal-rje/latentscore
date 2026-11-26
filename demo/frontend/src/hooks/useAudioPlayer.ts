import { useRef, useCallback, useEffect } from "react";
import { useStore } from "../store";
import { DEMO_MODE, getWsStreamUrl, renderAudio } from "../api";
import { browserRenderPcm } from "../browser/engine";
import {
  SAMPLE_RATE,
  computeBarAlignedDuration,
  toInternalSynthConfig,
} from "../browser/musicLogic";

const MAX_RECORDED_SECONDS = 120;
const MAX_WS_RETRIES = 3;
const LOOP_XFADE_SEC = 1.6;
const LIVE_XFADE_SEC = 1.6;
const TRACK_XFADE_OUT_SEC = 2.0;
const TRACK_XFADE_IN_SEC = 1.3;
const AUTO_XFADE_OUT_SEC = 3.6;
const AUTO_XFADE_IN_SEC = 2.6;
const AUTO_PREADVANCE_SEC = 4.4;
const AUTO_LIVE_DRIFT_SEC = 30;
const DUCK_GAIN = 0.62;
const PAUSE_FADE_SEC = 0.28;
const END_FADE_OUT_SEC = 5.2;
const OPEN_FILTER_HZ = 18_000;
const CLOSED_FILTER_HZ = 320;
const PARAM_GUARD_SEC = 0.004;

interface ScheduledCurve {
  startAt: number;
  duration: number;
  curve: Float32Array;
}

interface ParamTracker {
  baseValue: number;
  activeCurve: ScheduledCurve | null;
  expRamp: { startAt: number; endAt: number; from: number; to: number } | null;
}

/** Analytically evaluate the exact parameter value at ``t`` without relying on stale param.value. */
function evaluateTrackedParam(tracker: ParamTracker, t: number): number {
  if (tracker.activeCurve) {
    const { startAt, duration, curve } = tracker.activeCurve;
    if (t <= startAt) return tracker.baseValue;
    if (t >= startAt + duration || duration <= 0) {
      return curve[curve.length - 1];
    }
    const frac = (t - startAt) / duration;
    const pos = frac * (curve.length - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(curve.length - 1, i0 + 1);
    const alpha = pos - i0;
    return curve[i0] * (1 - alpha) + curve[i1] * alpha;
  }
  if (tracker.expRamp) {
    const { startAt, endAt, from, to } = tracker.expRamp;
    if (t <= startAt) return from;
    if (t >= endAt || endAt <= startAt) return to;
    const frac = (t - startAt) / (endAt - startAt);
    return from * Math.pow(to / Math.max(1e-6, from), frac);
  }
  return tracker.baseValue;
}

/**
 * Safely hold and cancel an AudioParam at ``now``, returning the analytic held
 * value and scheduling an anchor so subsequent curves at ``now + PARAM_GUARD_SEC``
 * never throw NotSupportedError in Chrome, Firefox, or Safari.
 */
function holdParamAt(param: AudioParam, tracker: ParamTracker, now: number): number {
  const held = evaluateTrackedParam(tracker, now);
  const withHold = param as AudioParam & { cancelAndHoldAtTime?: (t: number) => AudioParam };
  if (typeof withHold.cancelAndHoldAtTime === "function") {
    try {
      withHold.cancelAndHoldAtTime(now);
    } catch {
      param.cancelScheduledValues(now);
      param.setValueAtTime(held, now);
    }
  } else {
    param.cancelScheduledValues(now);
    param.setValueAtTime(held, now);
  }
  tracker.baseValue = held;
  tracker.activeCurve = null;
  tracker.expRamp = null;
  return held;
}

/** Schedule a curve safely after holding previous automation at ``now``. */
function scheduleTrackedCurve(
  param: AudioParam,
  tracker: ParamTracker,
  curve: Float32Array<ArrayBuffer>,
  startAt: number,
  duration: number,
  now: number,
  cancelExisting = true
): void {
  const safeDur = Math.max(0.015, duration);
  const safeStart = Math.max(startAt, now + PARAM_GUARD_SEC);
  if (cancelExisting) {
    holdParamAt(param, tracker, now);
    param.setValueAtTime(curve[0], Math.max(now + 0.002, safeStart - 0.001));
  }
  try {
    param.setValueCurveAtTime(curve, safeStart, safeDur);
    tracker.baseValue = curve[0];
    tracker.activeCurve = { startAt: safeStart, duration: safeDur, curve };
    tracker.expRamp = null;
  } catch {
    const endVal = Math.max(0.0001, curve[curve.length - 1]);
    param.setValueAtTime(curve[0], safeStart);
    param.linearRampToValueAtTime(endVal, safeStart + safeDur);
    tracker.baseValue = curve[curve.length - 1];
    tracker.activeCurve = null;
    tracker.expRamp = null;
  }
}

/** Schedule an exponential frequency sweep safely after holding previous automation. */
function scheduleTrackedFilterSweep(
  param: AudioParam,
  tracker: ParamTracker,
  fromHz: number,
  toHz: number,
  startAt: number,
  duration: number,
  now: number,
  cancelExisting = true
): void {
  const safeFrom = Math.max(20, fromHz);
  const safeTo = Math.max(20, toHz);
  const safeDur = Math.max(0.015, duration);
  const safeStart = Math.max(startAt, now + PARAM_GUARD_SEC);
  if (cancelExisting) {
    holdParamAt(param, tracker, now);
  }
  try {
    param.setValueAtTime(safeFrom, safeStart);
    param.exponentialRampToValueAtTime(safeTo, safeStart + safeDur);
    tracker.baseValue = safeFrom;
    tracker.activeCurve = null;
    tracker.expRamp = {
      startAt: safeStart,
      endAt: safeStart + safeDur,
      from: safeFrom,
      to: safeTo,
    };
  } catch {
    param.setValueAtTime(safeTo, safeStart);
    tracker.baseValue = safeTo;
    tracker.activeCurve = null;
    tracker.expRamp = null;
  }
}

/**
 * Correlation-aware crossfade curve (Fable Max formula):
 * rho = 0 gives equal-power (unrelated tracks), rho = 1 gives linear (correlated pads).
 */
function makeCrossfadeCurve(
  direction: "in" | "out",
  fromLevel: number,
  toLevel: number,
  rho = 0.25,
  steps = 96
): Float32Array<ArrayBuffer> {
  const curve = new Float32Array(new ArrayBuffer(steps * 4));
  for (let i = 0; i < steps; i++) {
    const x = i / (steps - 1);
    const denom = Math.sqrt(1 + rho * Math.sin(Math.PI * x));
    const norm =
      direction === "in"
        ? Math.sin((Math.PI * x) / 2) / denom
        : Math.cos((Math.PI * x) / 2) / denom;
    curve[i] =
      direction === "in"
        ? fromLevel + (toLevel - fromLevel) * norm
        : toLevel + (fromLevel - toLevel) * norm;
  }
  curve[0] = fromLevel;
  curve[steps - 1] = toLevel;
  return curve;
}

/** Cubic fade-into-nothingness curve: g(x) = from * (1 - x)^3 (zero end slope, -60 dB at x=0.9). */
function makeNothingnessCurve(fromGain: number, steps = 128): Float32Array<ArrayBuffer> {
  const curve = new Float32Array(new ArrayBuffer(steps * 4));
  for (let i = 0; i < steps; i++) {
    const u = 1 - i / (steps - 1);
    curve[i] = fromGain * u * u * u;
  }
  curve[0] = fromGain;
  curve[steps - 1] = 0;
  return curve;
}

function makeSoftClipCurve(ceiling = 0.99, knee = 0.92, size = 2048): Float32Array<ArrayBuffer> {
  const curve = new Float32Array(new ArrayBuffer(size * 4));
  const span = Math.max(1e-4, ceiling - knee);
  for (let i = 0; i < size; i++) {
    const x = (i / (size - 1)) * 2 - 1;
    const a = Math.abs(x);
    curve[i] = a <= knee ? x : Math.sign(x) * (knee + span * Math.tanh((a - knee) / span));
  }
  return curve;
}

function softLimitInPlace(buf: Float32Array, ceiling = 0.94): Float32Array {
  const knee = ceiling * 0.8;
  const span = ceiling - knee;
  for (let i = 0; i < buf.length; i++) {
    const v = buf[i];
    const a = Math.abs(v);
    if (a > knee) {
      buf[i] = Math.sign(v) * (knee + span * Math.tanh((a - knee) / span));
    }
  }
  return buf;
}

/**
 * Encode recorded AudioBuffers into a studio-ready WAV with optional
 * mono-compatible M-S stereo widening and clean fade-in / cubic fade-out edges.
 */
function encodeWav(buffers: AudioBuffer[], stereoWidth = 0): Blob {
  if (buffers.length === 0) return new Blob();
  const sampleRate = buffers[0].sampleRate;
  let totalLength = 0;
  for (const b of buffers) totalLength += b.length;
  if (totalLength === 0) return new Blob();

  const mono = new Float32Array(totalLength);
  let offset = 0;
  for (const b of buffers) {
    const ch0 = b.getChannelData(0);
    if (b.numberOfChannels > 1) {
      const ch1 = b.getChannelData(1);
      for (let i = 0; i < ch0.length; i++) {
        mono[offset + i] = (ch0[i] + ch1[i]) * 0.5;
      }
    } else {
      mono.set(ch0, offset);
    }
    offset += b.length;
  }

  // Apply gentle 45 ms fade-in and lush cubic fade-into-nothingness at the end.
  const fadeInN = Math.min(Math.floor(totalLength * 0.1), Math.round(sampleRate * 0.045));
  for (let i = 0; i < fadeInN; i++) {
    const t = i / Math.max(1, fadeInN - 1);
    mono[i] *= 0.5 * (1 - Math.cos(Math.PI * t));
  }
  const fadeOutN = Math.min(Math.floor(totalLength * 0.25), Math.round(sampleRate * 3.2));
  const fadeOutStart = totalLength - fadeOutN;
  for (let i = 0; i < fadeOutN; i++) {
    const u = 1 - i / Math.max(1, fadeOutN - 1);
    mono[fadeOutStart + i] *= u * u * u;
  }
  if (totalLength > 0) {
    mono[0] = 0;
    mono[totalLength - 1] = 0;
  }

  const safeWidth = Math.max(0, Math.min(1, stereoWidth));
  const useStereo = safeWidth > 0.05;
  const numChannels = useStereo ? 2 : 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataLength = totalLength * blockAlign;

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
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataLength, true);

  let p = 44;
  if (!useStereo) {
    for (let i = 0; i < totalLength; i++) {
      const s = Math.max(-1, Math.min(1, mono[i]));
      view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      p += 2;
    }
  } else {
    // Mono-compatible M-S stereo widening matching playback graph (11 ms delay + 250 Hz 1-pole HPF).
    const delaySamples = Math.round(sampleRate * 0.011);
    const sideAmt = safeWidth * 0.28;
    const midScale = 1 / (1 + sideAmt);
    const hpAlpha = Math.exp((-2 * Math.PI * 250) / sampleRate);
    let prevIn = 0;
    let prevOut = 0;
    for (let i = 0; i < totalLength; i++) {
      const delayed = i >= delaySamples ? mono[i - delaySamples] : 0;
      const hp = hpAlpha * (prevOut + delayed - prevIn);
      prevIn = delayed;
      prevOut = hp;
      const m = mono[i] * midScale;
      const s = hp * sideAmt * midScale;
      const l = Math.max(-1, Math.min(1, m + s));
      const r = Math.max(-1, Math.min(1, m - s));
      view.setInt16(p, l < 0 ? l * 0x8000 : l * 0x7fff, true);
      view.setInt16(p + 2, r < 0 ? r * 0x8000 : r * 0x7fff, true);
      p += 4;
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function pcmToAudioBuffer(ctx: AudioContext, pcm: Float32Array): AudioBuffer {
  const buf = ctx.createBuffer(1, pcm.length, SAMPLE_RATE);
  buf.getChannelData(0).set(pcm);
  return buf;
}

/**
 * Fold an extra tail ``[cycleLen .. raw.length)`` onto the head ``[0 .. fadeLen)``
 * with cosine-linear blend (zero +3 dB bump on correlated pads, exact bar seam).
 */
function buildSeamlessLoopPcm(raw: Float32Array, fadeSeconds: number): Float32Array {
  const fadeSamples = Math.min(Math.floor(fadeSeconds * SAMPLE_RATE), Math.floor(raw.length / 4));
  if (fadeSamples <= 1) return raw;
  const cycleLen = raw.length - fadeSamples;
  const out = new Float32Array(cycleLen);
  out.set(raw.subarray(0, cycleLen));
  for (let i = 0; i < fadeSamples; i++) {
    const t = i / (fadeSamples - 1);
    const headGain = 0.5 * (1 - Math.cos(Math.PI * t));
    const tailGain = 1 - headGain;
    out[i] = raw[i] * headGain + raw[cycleLen + i] * tailGain;
  }
  return softLimitInPlace(out);
}

/** Blend block A's retained unplayed tail smoothly into block B's head. */
function blendLiveTransition(prevTail: Float32Array, nextRaw: Float32Array): Float32Array {
  const fadeSamples = Math.min(prevTail.length, Math.floor(nextRaw.length / 3));
  if (fadeSamples <= 1) return nextRaw;
  const out = new Float32Array(nextRaw);
  for (let i = 0; i < fadeSamples; i++) {
    const t = i / (fadeSamples - 1);
    const inGain = 0.5 * (1 - Math.cos(Math.PI * t));
    const outGain = 1 - inGain;
    out[i] = prevTail[i] * outGain + nextRaw[i] * inGain;
  }
  return softLimitInPlace(out);
}

function computePeak(buf: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < buf.length; i++) {
    const a = Math.abs(buf[i]);
    if (a > peak) peak = a;
  }
  return peak;
}

function scalePcmInPlace(buf: Float32Array, gain: number): Float32Array {
  if (Math.abs(gain - 1) < 1e-4) return buf;
  for (let i = 0; i < buf.length; i++) {
    buf[i] *= gain;
  }
  return softLimitInPlace(buf, 0.94);
}

interface ActiveVoiceGroup {
  gain: GainNode;
  duckGain: GainNode;
  filter: BiquadFilterNode;
  gainTracker: ParamTracker;
  duckTracker: ParamTracker;
  filterTracker: ParamTracker;
  sources: AudioBufferSourceNode[];
  stereoNodes: AudioNode[];
  kind: "render" | "live" | "ws";
  sessionId: string;
  stereoWidth: number;
  startedAt: number;
  cycleDuration: number;
  fadeOutCleanupTimer?: ReturnType<typeof setTimeout>;
  autoAdvanceTimer?: ReturnType<typeof setTimeout>;
  autoAdvanceFired?: boolean;
  liveResumedCallback?: () => void;
}

export function useAudioPlayer(onStreamEnd?: () => void, onAutoAdvance?: () => void) {
  const activeSessionId = useStore((s) => s.activeSessionId);
  const streamMode = useStore((s) => s.streamMode);
  const setPlaying = useStore((s) => s.setPlaying);

  const onStreamEndRef = useRef(onStreamEnd);
  useEffect(() => {
    onStreamEndRef.current = onStreamEnd;
  }, [onStreamEnd]);

  const onAutoAdvanceRef = useRef(onAutoAdvance);
  useEffect(() => {
    onAutoAdvanceRef.current = onAutoAdvance;
  }, [onAutoAdvance]);

  const audioContextRef = useRef<AudioContext | null>(null);
  const masterLimiterRef = useRef<AudioNode | null>(null);
  const activeGroupRef = useRef<ActiveVoiceGroup | null>(null);
  const fadingGroupsRef = useRef<Set<ActiveVoiceGroup>>(new Set());

  const recordedBuffersRef = useRef<AudioBuffer[]>([]);
  const recordedDurationRef = useRef(0);
  const recordedStereoRef = useRef(0.5);
  const playbackGenRef = useRef(0);

  const wsRef = useRef<WebSocket | null>(null);
  const wsRetryCountRef = useRef(0);
  const wsRetryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsDecodeChainRef = useRef<Promise<void>>(Promise.resolve());
  const nextPlayTimeRef = useRef(0);
  const fetchAbortRef = useRef<AbortController | null>(null);
  const startWsStreamRef = useRef<() => void>(() => {});

  const getAudioGraph = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === "closed") {
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      const limiter = ctx.createDynamicsCompressor();
      limiter.threshold.setValueAtTime(-2.5, ctx.currentTime);
      limiter.knee.setValueAtTime(6, ctx.currentTime);
      limiter.ratio.setValueAtTime(12, ctx.currentTime);
      limiter.attack.setValueAtTime(0.004, ctx.currentTime);
      limiter.release.setValueAtTime(0.18, ctx.currentTime);

      // Soft true-peak ceiling guarantees zero digital hard clipping even mid-crossfade.
      const softClipper = ctx.createWaveShaper();
      softClipper.curve = makeSoftClipCurve(0.99, 0.92);
      softClipper.oversample = "2x";

      limiter.connect(softClipper);
      softClipper.connect(ctx.destination);

      // iOS silent-mode unlock within user gesture call stack.
      const buf = ctx.createBuffer(1, 1, 22050);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(limiter);
      src.start();

      audioContextRef.current = ctx;
      masterLimiterRef.current = limiter;
    }
    return {
      ctx: audioContextRef.current,
      master: masterLimiterRef.current!,
    };
  }, []);

  /**
   * Build a per-track voice graph:
   * sources -> envGain -> duckGain -> lowpassFilter -> headroom-safe M-S stereo -> master.
   * Mid/Side are normalized by 1/(1 + sideAmt) so peak never exceeds single-track headroom.
   */
  const createVoiceGroup = useCallback(
    (
      kind: "render" | "live" | "ws",
      sessionId: string,
      stereoAmount: number,
      initialGain = 0
    ): ActiveVoiceGroup => {
      const { ctx, master } = getAudioGraph();
      const now = ctx.currentTime;

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(initialGain, now);

      const duckGain = ctx.createGain();
      duckGain.gain.setValueAtTime(1.0, now);

      const filter = ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.frequency.setValueAtTime(OPEN_FILTER_HZ, now);
      filter.Q.setValueAtTime(0.707, now);

      gain.connect(duckGain);
      duckGain.connect(filter);

      const stereoNodes: AudioNode[] = [];
      const safeWidth = Math.max(0, Math.min(1, stereoAmount));

      if (safeWidth > 0.05) {
        const sideAmt = safeWidth * 0.28;
        const midScale = 1 / (1 + sideAmt);

        const midGain = ctx.createGain();
        midGain.gain.setValueAtTime(midScale, now);

        const sideDelay = ctx.createDelay(0.03);
        sideDelay.delayTime.setValueAtTime(0.011, now);
        const sideHpf = ctx.createBiquadFilter();
        sideHpf.type = "highpass";
        sideHpf.frequency.setValueAtTime(250, now);
        sideHpf.Q.setValueAtTime(0.707, now);

        const sidePlus = ctx.createGain();
        sidePlus.gain.setValueAtTime(sideAmt * midScale, now);
        const sideMinus = ctx.createGain();
        sideMinus.gain.setValueAtTime(-sideAmt * midScale, now);

        const merger = ctx.createChannelMerger(2);
        filter.connect(midGain);
        midGain.connect(merger, 0, 0);
        midGain.connect(merger, 0, 1);

        filter.connect(sideDelay);
        sideDelay.connect(sideHpf);
        sideHpf.connect(sidePlus);
        sideHpf.connect(sideMinus);
        sidePlus.connect(merger, 0, 0);
        sideMinus.connect(merger, 0, 1);
        merger.connect(master);

        stereoNodes.push(midGain, sideDelay, sideHpf, sidePlus, sideMinus, merger);
      } else {
        filter.connect(master);
      }

      return {
        gain,
        duckGain,
        filter,
        gainTracker: { baseValue: initialGain, activeCurve: null, expRamp: null },
        duckTracker: { baseValue: 1.0, activeCurve: null, expRamp: null },
        filterTracker: { baseValue: OPEN_FILTER_HZ, activeCurve: null, expRamp: null },
        sources: [],
        stereoNodes,
        kind,
        sessionId,
        stereoWidth: safeWidth,
        startedAt: now,
        cycleDuration: 0,
      };
    },
    [getAudioGraph]
  );

  const clearAutoAdvanceTimer = useCallback((group: ActiveVoiceGroup | null) => {
    if (group?.autoAdvanceTimer) {
      clearTimeout(group.autoAdvanceTimer);
      group.autoAdvanceTimer = undefined;
    }
  }, []);

  const destroyVoiceGroupNow = useCallback((group: ActiveVoiceGroup) => {
    if (group.fadeOutCleanupTimer) {
      clearTimeout(group.fadeOutCleanupTimer);
      group.fadeOutCleanupTimer = undefined;
    }
    if (group.autoAdvanceTimer) {
      clearTimeout(group.autoAdvanceTimer);
      group.autoAdvanceTimer = undefined;
    }
    group.liveResumedCallback = undefined;
    for (const src of group.sources) {
      src.onended = null;
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
      try {
        src.disconnect();
      } catch {
        /* ignore */
      }
    }
    group.sources = [];
    for (const node of [group.gain, group.duckGain, group.filter, ...group.stereoNodes]) {
      try {
        node.disconnect();
      } catch {
        /* ignore */
      }
    }
    group.stereoNodes = [];
    fadingGroupsRef.current.delete(group);
  }, []);

  /**
   * Gently duck the outgoing track on its dedicated ``duckGain`` node so any
   * in-progress fade-in on ``envGain`` finishes cleanly without freezing near 0.
   */
  const softDuckActive = useCallback(() => {
    const group = activeGroupRef.current;
    const ctx = audioContextRef.current;
    if (!group || !ctx || ctx.state === "closed") return;
    clearAutoAdvanceTimer(group);
    const now = ctx.currentTime;
    const cur = evaluateTrackedParam(group.duckTracker, now);
    if (cur > DUCK_GAIN + 0.01) {
      scheduleTrackedCurve(
        group.duckGain.gain,
        group.duckTracker,
        makeCrossfadeCurve("out", cur, DUCK_GAIN, 0.5, 48),
        now,
        0.35,
        now,
        true
      );
    }
  }, [clearAutoAdvanceTimer]);

  /** Smoothly fade out a voice group over ``fadeSec`` with optional warm lowpass close. */
  const fadeOutAndRelease = useCallback(
    (
      group: ActiveVoiceGroup | null,
      fadeSec: number,
      rho = 0.2,
      closeFilter = false
    ) => {
      if (!group) return;
      clearAutoAdvanceTimer(group);
      if (group.fadeOutCleanupTimer) {
        clearTimeout(group.fadeOutCleanupTimer);
        group.fadeOutCleanupTimer = undefined;
      }
      if (activeGroupRef.current === group) {
        activeGroupRef.current = null;
      }
      const ctx = audioContextRef.current;
      if (!ctx || ctx.state === "closed" || fadeSec <= 0.01) {
        destroyVoiceGroupNow(group);
        return;
      }
      fadingGroupsRef.current.add(group);
      const now = ctx.currentTime;
      const curG = Math.max(0, Math.min(1, evaluateTrackedParam(group.gainTracker, now)));
      const curve = closeFilter
        ? makeNothingnessCurve(curG)
        : makeCrossfadeCurve("out", curG, 0, rho);
      scheduleTrackedCurve(group.gain.gain, group.gainTracker, curve, now, fadeSec, now, true);
      if (closeFilter) {
        const curF = Math.max(
          CLOSED_FILTER_HZ,
          evaluateTrackedParam(group.filterTracker, now)
        );
        scheduleTrackedFilterSweep(
          group.filter.frequency,
          group.filterTracker,
          curF,
          CLOSED_FILTER_HZ,
          now,
          fadeSec,
          now,
          true
        );
      }
      group.fadeOutCleanupTimer = setTimeout(() => {
        destroyVoiceGroupNow(group);
      }, Math.ceil((fadeSec + 0.14) * 1000));
    },
    [clearAutoAdvanceTimer, destroyVoiceGroupNow]
  );

  /** Fade out BOTH active and already-fading voice groups cleanly (zero audible orphans). */
  const fadeOutAllActive = useCallback(
    (fadeSec = PAUSE_FADE_SEC) => {
      playbackGenRef.current += 1;
      const targets = new Set<ActiveVoiceGroup>(fadingGroupsRef.current);
      if (activeGroupRef.current) {
        targets.add(activeGroupRef.current);
        activeGroupRef.current = null;
      }
      for (const grp of targets) {
        fadeOutAndRelease(grp, fadeSec, 0.35, true);
      }
      nextPlayTimeRef.current = 0;
    },
    [fadeOutAndRelease]
  );

  const closeActiveWs = useCallback(() => {
    if (wsRetryTimerRef.current !== null) {
      clearTimeout(wsRetryTimerRef.current);
      wsRetryTimerRef.current = null;
    }
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      ws.onclose = null;
      ws.onmessage = null;
      ws.onerror = null;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop" }));
      }
      ws.close();
    }
  }, []);

  const recordBufferBounded = useCallback((buffer: AudioBuffer, stereoWidth: number) => {
    recordedStereoRef.current = stereoWidth;
    recordedBuffersRef.current.push(buffer);
    recordedDurationRef.current += buffer.duration;
    while (
      recordedDurationRef.current > MAX_RECORDED_SECONDS &&
      recordedBuffersRef.current.length > 1
    ) {
      const dropped = recordedBuffersRef.current.shift();
      if (dropped) recordedDurationRef.current -= dropped.duration;
    }
  }, []);

  const clearRecordedBuffers = useCallback(() => {
    recordedBuffersRef.current = [];
    recordedDurationRef.current = 0;
  }, []);

  /**
   * Schedule a lush fade-into-nothingness: cubic (1-x)^3 gain decay paired with
   * a warm 18 kHz -> 320 Hz lowpass dusk sweep over the final ~5.2 seconds.
   * Always evaluates held values analytically so mid-fade toggles never throw.
   */
  const scheduleNothingnessFade = useCallback((group: ActiveVoiceGroup, endAt: number) => {
    const ctx = audioContextRef.current;
    if (!ctx || ctx.state === "closed") return;
    const now = ctx.currentTime;
    const remaining = endAt - now;
    if (remaining <= 0.08) return;
    const fadeSec = Math.min(END_FADE_OUT_SEC, Math.max(1.2, remaining * 0.28));
    const fadeStart = Math.max(now + PARAM_GUARD_SEC, endAt - fadeSec);
    const actualFade = Math.max(0.08, endAt - fadeStart);

    const immediate = fadeStart <= now + 0.06;
    const startGain = immediate
      ? Math.max(0.01, Math.min(1, evaluateTrackedParam(group.gainTracker, now)))
      : 1.0;
    const startFreq = immediate
      ? Math.max(CLOSED_FILTER_HZ, evaluateTrackedParam(group.filterTracker, now))
      : OPEN_FILTER_HZ;

    scheduleTrackedCurve(
      group.gain.gain,
      group.gainTracker,
      makeNothingnessCurve(startGain),
      fadeStart,
      actualFade,
      now,
      immediate
    );
    scheduleTrackedFilterSweep(
      group.filter.frequency,
      group.filterTracker,
      startFreq,
      CLOSED_FILTER_HZ,
      fadeStart,
      actualFade,
      now,
      immediate
    );
  }, []);

  /** Rescue gain and filter smoothly back to open full volume (e.g. Loop or Auto toggled ON mid-fade). */
  const restoreFullVolume = useCallback((group: ActiveVoiceGroup) => {
    const ctx = audioContextRef.current;
    if (!ctx || ctx.state === "closed") return;
    const now = ctx.currentTime;
    const curG = Math.max(0.01, Math.min(1, evaluateTrackedParam(group.gainTracker, now)));
    if (curG < 0.99 || group.gainTracker.activeCurve) {
      scheduleTrackedCurve(
        group.gain.gain,
        group.gainTracker,
        makeCrossfadeCurve("in", curG, 1.0, 0.5, 48),
        now,
        0.45,
        now,
        true
      );
    }
    const curF = Math.max(CLOSED_FILTER_HZ, evaluateTrackedParam(group.filterTracker, now));
    if (curF < OPEN_FILTER_HZ * 0.95 || group.filterTracker.expRamp) {
      scheduleTrackedFilterSweep(
        group.filter.frequency,
        group.filterTracker,
        curF,
        OPEN_FILTER_HZ,
        now,
        0.45,
        now,
        true
      );
    }
  }, []);

  /** Arm a seamless Auto Drift pre-advance timer before the current Render track ends. */
  const armAutoAdvance = useCallback(
    (group: ActiveVoiceGroup, endAt: number, gen: number) => {
      clearAutoAdvanceTimer(group);
      const ctx = audioContextRef.current;
      if (!ctx || ctx.state === "closed") return;
      const fireAt = Math.max(ctx.currentTime + 1.5, endAt - AUTO_PREADVANCE_SEC);
      const delayMs = Math.max(800, Math.round((fireAt - ctx.currentTime) * 1000));
      group.autoAdvanceTimer = setTimeout(() => {
        if (
          gen !== playbackGenRef.current ||
          activeGroupRef.current !== group ||
          !useStore.getState().autoMode ||
          group.autoAdvanceFired
        ) {
          return;
        }
        group.autoAdvanceFired = true;
        onAutoAdvanceRef.current?.();
      }, delayMs);
    },
    [clearAutoAdvanceTimer]
  );

  // Dynamically honor Loop AND Auto mode toggles mid-playback without clicks or throws.
  useEffect(() => {
    return useStore.subscribe((next, prev) => {
      if (next.loopMode === prev.loopMode && next.autoMode === prev.autoMode) return;
      const group = activeGroupRef.current;
      const ctx = audioContextRef.current;
      if (!group || !ctx || ctx.state === "closed") return;

      const now = ctx.currentTime;

      if (group.kind === "live") {
        if (next.loopMode) {
          clearAutoAdvanceTimer(group);
          restoreFullVolume(group);
          group.liveResumedCallback?.();
        } else if (next.autoMode && !prev.autoMode) {
          restoreFullVolume(group);
          armAutoAdvance(group, now + AUTO_LIVE_DRIFT_SEC, playbackGenRef.current);
        } else if (!next.autoMode && prev.autoMode) {
          clearAutoAdvanceTimer(group);
        }
        return;
      }

      if (group.kind !== "render") return;
      const primary = group.sources[0];
      if (!primary) return;
      const dur = group.cycleDuration || primary.buffer?.duration || 0;
      const computeCycleEnd = () => {
        if (dur <= 0) return now + 5;
        const elapsed = Math.max(0, now - group.startedAt);
        const cycleNum = Math.floor(elapsed / dur) + 1;
        return group.startedAt + cycleNum * dur;
      };

      if (next.loopMode) {
        clearAutoAdvanceTimer(group);
        primary.loop = true;
        restoreFullVolume(group);
      } else if (next.autoMode) {
        primary.loop = false;
        restoreFullVolume(group);
        const endAt = computeCycleEnd();
        if (endAt - now <= AUTO_PREADVANCE_SEC + 0.5) {
          if (!group.autoAdvanceFired) {
            group.autoAdvanceFired = true;
            onAutoAdvanceRef.current?.();
          }
        } else {
          armAutoAdvance(group, endAt, playbackGenRef.current);
        }
      } else {
        clearAutoAdvanceTimer(group);
        primary.loop = false;
        if (dur > 0) {
          scheduleNothingnessFade(group, computeCycleEnd());
        }
      }
    });
  }, [armAutoAdvance, clearAutoAdvanceTimer, restoreFullVolume, scheduleNothingnessFade]);

  // ─── Browser-native cohesive Live Stream (exact integer sample clock, 1-block prefetch) ───

  const startBrowserLiveStream = useCallback(() => {
    if (!activeSessionId) return;
    fetchAbortRef.current?.abort();
    closeActiveWs();
    clearRecordedBuffers();

    softDuckActive();
    playbackGenRef.current += 1;
    const gen = playbackGenRef.current;
    const { ctx } = getAudioGraph();
    if (ctx.state === "suspended") void ctx.resume();

    const abort = new AbortController();
    fetchAbortRef.current = abort;
    setPlaying(true);

    const config = useStore.getState().activeConfig;
    const stereoWidth = toInternalSynthConfig(config).stereo;
    const blockDuration = computeBarAlignedDuration(config, 16, 10, 22);
    const bodySamples = Math.max(1, Math.round(blockDuration * SAMPLE_RATE));
    const exactBlockSec = bodySamples / SAMPLE_RATE;
    const maxSeconds = 600;
    const sessionSeed = Math.floor(Math.random() * 0x7fffffff) || 1337;

    void (async () => {
      let elapsedSamples = 0;
      let prevTail: Float32Array | null = null;
      let sessionGain = 1.0;
      let group: ActiveVoiceGroup | null = null;

      try {
        while (!abort.signal.aborted && gen === playbackGenRef.current) {
          // Bound prefetch to 1 prepared successor ahead of the currently playing block.
          if (group !== null) {
            while (
              !abort.signal.aborted &&
              gen === playbackGenRef.current &&
              nextPlayTimeRef.current - ctx.currentTime > exactBlockSec * 0.85
            ) {
              await new Promise((r) => setTimeout(r, 150));
            }
            if (abort.signal.aborted || gen !== playbackGenRef.current) return;
          }

          const tOffsetSec = elapsedSamples / SAMPLE_RATE;
          const isFirstBlock = elapsedSamples === 0;
          const rawBlock = await browserRenderPcm(
            config,
            exactBlockSec + LIVE_XFADE_SEC,
            tOffsetSec,
            false,
            sessionSeed,
            abort.signal
          );
          if (abort.signal.aborted || gen !== playbackGenRef.current) return;

          if (isFirstBlock) {
            const peak = computePeak(rawBlock);
            sessionGain = peak > 1e-4 ? Math.max(0.55, Math.min(2.2, 0.78 / peak)) : 1.0;
          }
          scalePcmInPlace(rawBlock, sessionGain);

          const sliceLen = Math.min(rawBlock.length, bodySamples);
          const nextTail = rawBlock.length > sliceLen ? rawBlock.slice(sliceLen) : null;
          let playable = rawBlock.subarray(0, sliceLen);
          if (prevTail !== null) {
            playable = blendLiveTransition(prevTail, playable);
          }
          prevTail = nextTail;

          const audioBuffer = pcmToAudioBuffer(ctx, playable);
          recordBufferBounded(audioBuffer, stereoWidth);

          if (group === null) {
            const outgoing = activeGroupRef.current;
            const sameTrack = outgoing?.sessionId === activeSessionId;
            const isAuto = useStore.getState().autoMode;
            const rho = sameTrack ? 0.65 : isAuto ? 0.25 : 0.0;
            const fadeInSec = outgoing
              ? isAuto
                ? AUTO_XFADE_IN_SEC
                : TRACK_XFADE_IN_SEC
              : 0.16;
            const fadeOutSec = isAuto ? AUTO_XFADE_OUT_SEC : TRACK_XFADE_OUT_SEC;

            group = createVoiceGroup("live", activeSessionId, stereoWidth, 0);
            activeGroupRef.current = group;

            const now = ctx.currentTime;
            const startAt = now + 0.04;
            group.startedAt = startAt;
            nextPlayTimeRef.current = startAt;

            scheduleTrackedCurve(
              group.gain.gain,
              group.gainTracker,
              makeCrossfadeCurve("in", 0, 1.0, rho),
              startAt,
              fadeInSec,
              now,
              false
            );
            if (outgoing) {
              fadeOutAndRelease(outgoing, fadeOutSec, rho, false);
            }
            if (isAuto) {
              armAutoAdvance(group, startAt + AUTO_LIVE_DRIFT_SEC, gen);
            }
          }

          const src = ctx.createBufferSource();
          src.buffer = audioBuffer;
          src.connect(group.gain);
          const startWhen = Math.max(ctx.currentTime + 0.015, nextPlayTimeRef.current);
          src.start(startWhen);
          nextPlayTimeRef.current = startWhen + audioBuffer.duration;
          group.sources.push(src);
          const capturedGroup = group;
          src.onended = () => {
            const idx = capturedGroup.sources.indexOf(src);
            if (idx !== -1) capturedGroup.sources.splice(idx, 1);
          };

          elapsedSamples += sliceLen;
          const elapsedSec = elapsedSamples / SAMPLE_RATE;

          if (
            elapsedSec >= maxSeconds &&
            !useStore.getState().loopMode &&
            !useStore.getState().autoMode
          ) {
            scheduleNothingnessFade(group, nextPlayTimeRef.current);
            // Remain resumable if the user toggles Loop ON before the fade finishes.
            let resumedByLoop = false;
            const resumePromise = new Promise<void>((resolve) => {
              capturedGroup.liveResumedCallback = () => {
                resumedByLoop = true;
                capturedGroup.liveResumedCallback = undefined;
                resolve();
              };
            });
            const waitMs = Math.max(0, (nextPlayTimeRef.current - ctx.currentTime) * 1000) + 120;
            await Promise.race([
              resumePromise,
              new Promise<void>((resolve) => setTimeout(resolve, waitMs)),
            ]);
            capturedGroup.liveResumedCallback = undefined;
            if (abort.signal.aborted || gen !== playbackGenRef.current) return;
            if (resumedByLoop || useStore.getState().loopMode) {
              continue;
            }
            break;
          }
        }

        if (!abort.signal.aborted && gen === playbackGenRef.current && group !== null) {
          if (activeGroupRef.current === group) activeGroupRef.current = null;
          destroyVoiceGroupNow(group);
          setPlaying(false);
          onStreamEndRef.current?.();
        }
      } catch (err) {
        if (abort.signal.aborted || gen !== playbackGenRef.current) return;
        console.error("Browser live stream error", err);
        fadeOutAllActive(PAUSE_FADE_SEC);
        useStore
          .getState()
          .setError(err instanceof Error ? err.message : "Audio synthesis failed");
        setPlaying(false);
      }
    })();
  }, [
    activeSessionId,
    armAutoAdvance,
    clearRecordedBuffers,
    closeActiveWs,
    createVoiceGroup,
    destroyVoiceGroupNow,
    fadeOutAllActive,
    fadeOutAndRelease,
    getAudioGraph,
    recordBufferBounded,
    scheduleNothingnessFade,
    setPlaying,
    softDuckActive,
  ]);

  // ─── Server Stream mode: WebSocket with deferred handoff until first decoded audio ───

  const startWsStream = useCallback(() => {
    if (DEMO_MODE === "browser") {
      startBrowserLiveStream();
      return;
    }
    if (!activeSessionId) return;
    fetchAbortRef.current?.abort();
    closeActiveWs();
    clearRecordedBuffers();

    softDuckActive();
    playbackGenRef.current += 1;
    const gen = playbackGenRef.current;
    wsDecodeChainRef.current = Promise.resolve();
    const { ctx } = getAudioGraph();
    if (ctx.state === "suspended") void ctx.resume();

    const stereoWidth = toInternalSynthConfig(useStore.getState().activeConfig).stereo;
    let group: ActiveVoiceGroup | null = null;

    const ws = new WebSocket(getWsStreamUrl(activeSessionId));
    wsRef.current = ws;
    let receivedError = false;
    let receivedChunk = false;

    ws.onopen = () => {
      if (gen !== playbackGenRef.current) {
        ws.close();
        return;
      }
      setPlaying(true);
    };

    ws.onmessage = (event) => {
      if (gen !== playbackGenRef.current) return;
      const msg = JSON.parse(event.data) as { type?: string; data?: string; message?: string };
      if (msg.type === "error") {
        receivedError = true;
        fadeOutAllActive(PAUSE_FADE_SEC);
        useStore.getState().setError(msg.message || "Stream session unavailable");
        ws.close();
        setPlaying(false);
        return;
      }
      if (msg.type === "audio_chunk" && typeof msg.data === "string") {
        receivedChunk = true;
        wsRetryCountRef.current = 0;
        const b64 = msg.data;
        wsDecodeChainRef.current = wsDecodeChainRef.current.then(async () => {
          if (gen !== playbackGenRef.current) return;
          const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
          try {
            const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
            if (gen !== playbackGenRef.current) return;
            recordBufferBounded(audioBuffer, stereoWidth);

            // Handoff from outgoing track ONLY once the first decoded replacement chunk is ready.
            if (group === null) {
              const outgoing = activeGroupRef.current;
              const isAuto = useStore.getState().autoMode;
              const rho = outgoing?.sessionId === activeSessionId ? 0.65 : isAuto ? 0.25 : 0.0;
              group = createVoiceGroup("ws", activeSessionId, stereoWidth, 0);
              activeGroupRef.current = group;

              const now = ctx.currentTime;
              const startAt = now + 0.04;
              nextPlayTimeRef.current = startAt;
              scheduleTrackedCurve(
                group.gain.gain,
                group.gainTracker,
                makeCrossfadeCurve("in", 0, 1.0, rho),
                startAt,
                outgoing ? (isAuto ? AUTO_XFADE_IN_SEC : TRACK_XFADE_IN_SEC) : 0.16,
                now,
                false
              );
              if (outgoing) {
                fadeOutAndRelease(
                  outgoing,
                  isAuto ? AUTO_XFADE_OUT_SEC : TRACK_XFADE_OUT_SEC,
                  rho,
                  false
                );
              }
              if (isAuto) {
                armAutoAdvance(group, startAt + AUTO_LIVE_DRIFT_SEC, gen);
              }
            }

            const src = ctx.createBufferSource();
            src.buffer = audioBuffer;
            src.connect(group.gain);
            const when = Math.max(ctx.currentTime + 0.015, nextPlayTimeRef.current);
            src.start(when);
            nextPlayTimeRef.current = when + audioBuffer.duration;
            group.sources.push(src);
            const captured = group;
            src.onended = () => {
              const idx = captured.sources.indexOf(src);
              if (idx !== -1) captured.sources.splice(idx, 1);
            };
          } catch (e) {
            console.warn("Failed to decode audio chunk", e);
          }
        });
      }
    };

    ws.onclose = (ev) => {
      if (wsRef.current !== ws || gen !== playbackGenRef.current) return;
      void wsDecodeChainRef.current.then(() => {
        if (gen !== playbackGenRef.current) return;
        const canRetry =
          !receivedError &&
          ev.code !== 1008 &&
          ev.code !== 1011 &&
          (receivedChunk || wsRetryCountRef.current < MAX_WS_RETRIES);
        const remainingMs = Math.max(0, (nextPlayTimeRef.current - ctx.currentTime) * 1000);
        if (group && !useStore.getState().loopMode && !useStore.getState().autoMode && remainingMs > 250) {
          scheduleNothingnessFade(group, nextPlayTimeRef.current);
        }
        wsRetryTimerRef.current = setTimeout(() => {
          wsRetryTimerRef.current = null;
          if (gen !== playbackGenRef.current) return;
          if (useStore.getState().loopMode && canRetry) {
            if (!receivedChunk) wsRetryCountRef.current += 1;
            startWsStreamRef.current();
          } else if (useStore.getState().autoMode && !receivedError) {
            onAutoAdvanceRef.current?.();
          } else {
            setPlaying(false);
            if (!receivedError) onStreamEndRef.current?.();
          }
        }, remainingMs + (receivedChunk ? 50 : 500));
      });
    };

    ws.onerror = (err) => {
      receivedError = true;
      console.error("WebSocket error", err);
      fadeOutAllActive(PAUSE_FADE_SEC);
      setPlaying(false);
    };
  }, [
    activeSessionId,
    armAutoAdvance,
    clearRecordedBuffers,
    closeActiveWs,
    createVoiceGroup,
    fadeOutAllActive,
    fadeOutAndRelease,
    getAudioGraph,
    recordBufferBounded,
    scheduleNothingnessFade,
    setPlaying,
    softDuckActive,
    startBrowserLiveStream,
  ]);

  useEffect(() => {
    startWsStreamRef.current = startWsStream;
  }, [startWsStream]);

  // ─── Render mode: bar-aligned seamless loop + soft crossfade + dusk fade / Auto Drift ───

  const startFetchPlay = useCallback(() => {
    if (!activeSessionId) return;
    fetchAbortRef.current?.abort();
    closeActiveWs();
    clearRecordedBuffers();

    softDuckActive();
    playbackGenRef.current += 1;
    const gen = playbackGenRef.current;
    const { ctx } = getAudioGraph();
    if (ctx.state === "suspended") void ctx.resume();

    const abort = new AbortController();
    fetchAbortRef.current = abort;
    setPlaying(true);

    const config = useStore.getState().activeConfig;
    const stereoWidth = toInternalSynthConfig(config).stereo;
    const cycleDuration = computeBarAlignedDuration(config, 28, 12, 34);

    const bufferPromise =
      DEMO_MODE === "browser"
        ? browserRenderPcm(
            config,
            cycleDuration + LOOP_XFADE_SEC,
            0,
            true,
            null,
            abort.signal
          ).then((rawPcm) => {
            const seamlessPcm = buildSeamlessLoopPcm(rawPcm, LOOP_XFADE_SEC);
            return pcmToAudioBuffer(ctx, seamlessPcm);
          })
        : renderAudio(
            activeSessionId,
            Math.max(12, Math.round(cycleDuration + LOOP_XFADE_SEC)),
            config,
            abort.signal
          )
            .then((wavBytes) => ctx.decodeAudioData(wavBytes))
            .then((decoded) => {
              const rawMono = decoded.getChannelData(0);
              const folded = buildSeamlessLoopPcm(rawMono, LOOP_XFADE_SEC);
              return pcmToAudioBuffer(ctx, folded);
            });

    bufferPromise
      .then((audioBuffer) => {
        if (!audioBuffer || abort.signal.aborted || gen !== playbackGenRef.current) return;
        recordBufferBounded(audioBuffer, stereoWidth);

        const outgoing = activeGroupRef.current;
        const { loopMode, autoMode } = useStore.getState();
        const rho = outgoing?.sessionId === activeSessionId ? 0.65 : autoMode ? 0.25 : 0.0;
        const fadeInSec = outgoing
          ? autoMode
            ? AUTO_XFADE_IN_SEC
            : TRACK_XFADE_IN_SEC
          : 0.16;
        const fadeOutSec = autoMode ? AUTO_XFADE_OUT_SEC : TRACK_XFADE_OUT_SEC;

        const group = createVoiceGroup("render", activeSessionId, stereoWidth, 0);
        activeGroupRef.current = group;

        const now = ctx.currentTime;
        const startAt = now + 0.03;
        group.startedAt = startAt;
        group.cycleDuration = audioBuffer.duration;

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.loop = loopMode;
        source.connect(group.gain);

        scheduleTrackedCurve(
          group.gain.gain,
          group.gainTracker,
          makeCrossfadeCurve("in", 0, 1.0, rho),
          startAt,
          fadeInSec,
          now,
          false
        );

        if (autoMode && !loopMode) {
          armAutoAdvance(group, startAt + audioBuffer.duration, gen);
        } else if (!source.loop) {
          scheduleNothingnessFade(group, startAt + audioBuffer.duration);
        }

        if (outgoing) {
          fadeOutAndRelease(outgoing, fadeOutSec, rho, false);
        }

        source.start(startAt);
        group.sources.push(source);

        source.onended = () => {
          const idx = group.sources.indexOf(source);
          if (idx !== -1) group.sources.splice(idx, 1);
          if (!abort.signal.aborted && gen === playbackGenRef.current) {
            if (activeGroupRef.current === group) activeGroupRef.current = null;
            destroyVoiceGroupNow(group);
            if (useStore.getState().autoMode) {
              if (!group.autoAdvanceFired) {
                group.autoAdvanceFired = true;
                onAutoAdvanceRef.current?.();
              }
            } else {
              setPlaying(false);
              onStreamEndRef.current?.();
            }
          }
        };
      })
      .catch((err) => {
        if (abort.signal.aborted || gen !== playbackGenRef.current) return;
        console.error("Render play error", err);
        fadeOutAllActive(PAUSE_FADE_SEC);
        useStore
          .getState()
          .setError(err instanceof Error ? err.message : "Audio render failed");
        setPlaying(false);
      });
  }, [
    activeSessionId,
    armAutoAdvance,
    clearRecordedBuffers,
    closeActiveWs,
    createVoiceGroup,
    destroyVoiceGroupNow,
    fadeOutAllActive,
    fadeOutAndRelease,
    getAudioGraph,
    recordBufferBounded,
    scheduleNothingnessFade,
    setPlaying,
    softDuckActive,
  ]);

  // ─── Unified interface (mode-independent soft fade-out stop) ───

  const startStream = useCallback(() => {
    if (streamMode) {
      startWsStream();
    } else {
      startFetchPlay();
    }
  }, [streamMode, startWsStream, startFetchPlay]);

  const stopStream = useCallback(() => {
    fetchAbortRef.current?.abort();
    fetchAbortRef.current = null;
    wsRetryCountRef.current = 0;
    closeActiveWs();
    fadeOutAllActive(PAUSE_FADE_SEC);
    setPlaying(false);
  }, [closeActiveWs, fadeOutAllActive, setPlaying]);

  const downloadAudio = useCallback((title: string) => {
    if (recordedBuffersRef.current.length === 0) return;
    const blob = encodeWav(recordedBuffersRef.current, recordedStereoRef.current);
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
    const fadingSet = fadingGroupsRef.current;
    return () => {
      // Bump generation FIRST so no pending decode or WS retry can ever restart audio after unmount.
      playbackGenRef.current += 1;
      closeActiveWs();
      fetchAbortRef.current?.abort();
      if (activeGroupRef.current) {
        destroyVoiceGroupNow(activeGroupRef.current);
        activeGroupRef.current = null;
      }
      for (const g of Array.from(fadingSet)) {
        destroyVoiceGroupNow(g);
      }
      void audioContextRef.current?.close();
    };
  }, [closeActiveWs, destroyVoiceGroupNow]);

  return { startStream, stopStream, downloadAudio, hasRecordedAudio };
}
