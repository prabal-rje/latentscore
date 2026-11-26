import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchParamOptions, updateParams } from "../api";
import { useStore } from "../store";

const PARAM_CATEGORIES: {
  label: string;
  icon: string;
  params: string[];
}[] = [
  {
    label: "Core",
    icon: "\u2B50",
    params: ["tempo", "density", "motion", "brightness"],
  },
  {
    label: "Key",
    icon: "\u{1F3B9}",
    params: ["root", "mode"],
  },
  {
    label: "Tone",
    icon: "\u{1F3B5}",
    params: ["bass", "pad", "melody", "rhythm", "texture", "accent"],
  },
  {
    label: "Space",
    icon: "\u{1F30C}",
    params: ["space", "echo", "stereo", "grain"],
  },
  {
    label: "Feel",
    icon: "\u{1F91A}",
    params: ["human", "attack"],
  },
];

const PARAM_LABELS: Record<string, string> = {
  tempo: "Tempo",
  brightness: "Brightness",
  space: "Reverb",
  density: "Density",
  motion: "Motion",
  echo: "Echo",
  stereo: "Stereo",
  human: "Human",
  grain: "Grain",
  attack: "Attack",
  bass: "Bass",
  pad: "Pad",
  melody: "Melody",
  rhythm: "Rhythm",
  texture: "Texture",
  accent: "Accent",
  root: "Root",
  mode: "Scale",
};

export default function ParamPanel() {
  const {
    activeSessionId,
    activeConfig,
    activeVibe,
    showParamPanel,
  } = useStore();
  const [openCategory, setOpenCategory] = useState<string>("Core");
  const [pendingConfig, setPendingConfig] = useState<Record<string, unknown>>({});
  const [applying, setApplying] = useState(false);

  const { data: paramOptions } = useQuery({
    queryKey: ["param-options"],
    queryFn: fetchParamOptions,
  });

  // Sync pending config when active config changes externally (new song, etc.)
  useEffect(() => {
    setPendingConfig(activeConfig);
  }, [activeConfig]);

  const handleChange = useCallback(
    (param: string, value: string) => {
      setPendingConfig((prev) => {
        const next: Record<string, unknown> = {
          ...prev,
          [param]: param === "density" ? Number(value) : value,
        };
        if (param === "melody") {
          next.melody_engine = value === "procedural" ? "procedural" : "pattern";
        }
        return next;
      });
    },
    []
  );

  // Compute which params have been changed
  const changedParams = useMemo(() => {
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(pendingConfig)) {
      if (String(pendingConfig[key]) !== String(activeConfig[key])) {
        out[key] = pendingConfig[key];
      }
    }
    return out;
  }, [pendingConfig, activeConfig]);
  const hasChanges = Object.keys(changedParams).length > 0;

  const handleApply = useCallback(async () => {
    if (!activeSessionId || !hasChanges) return;
    const targetSessionId = activeSessionId;
    setApplying(true);
    try {
      const result = await updateParams(targetSessionId, changedParams, activeConfig, activeVibe);
      // Guard against stale async completion after switching to another track.
      if (useStore.getState().activeSessionId !== targetSessionId) return;
      // Atomically commit to active session and its matching playlist entry.
      useStore.getState().commitSessionParams(targetSessionId, {
        config: result.config,
        palette: result.palette,
        title: result.title,
      });
      // Crossfade cleanly to the updated mix only if playback is currently active.
      if (useStore.getState().isPlaying) {
        useStore.getState().triggerRestream();
      }
    } catch {
      if (useStore.getState().activeSessionId === targetSessionId) {
        setPendingConfig(activeConfig);
      }
    } finally {
      setApplying(false);
    }
  }, [activeSessionId, activeVibe, hasChanges, changedParams, activeConfig]);

  const handleReset = useCallback(() => {
    setPendingConfig(activeConfig);
  }, [activeConfig]);

  if (!showParamPanel || !activeSessionId) return null;

  return (
    <div className="param-panel">
      <div className="param-panel-header">
        <h3>Parameters</h3>
        <button
          className="param-close"
          onClick={useStore.getState().toggleParamPanel}
        >
          &times;
        </button>
      </div>

      <div className="param-categories">
        {PARAM_CATEGORIES.map((cat) => {
          // Show dot indicator if any param in this category has pending changes
          const hasCatChanges = cat.params.some(
            (p) => String(pendingConfig[p] ?? "") !== String(activeConfig[p] ?? "")
          );
          return (
            <button
              key={cat.label}
              className={`param-cat-btn ${openCategory === cat.label ? "active" : ""}`}
              onClick={() => setOpenCategory(openCategory === cat.label ? "" : cat.label)}
            >
              <span>{cat.icon}</span>
              <span>{cat.label}{hasCatChanges ? " \u2022" : ""}</span>
            </button>
          );
        })}
      </div>

      <div className="param-list">
        {PARAM_CATEGORIES.filter((cat) => cat.label === openCategory).map((cat) =>
          cat.params.map((param) => {
            const options = paramOptions?.[param];
            const currentValue = String(pendingConfig[param] ?? "");
            const isModified = String(pendingConfig[param] ?? "") !== String(activeConfig[param] ?? "");
            if (!options) return null;

            return (
              <div key={param} className="param-row">
                <label className="param-label">
                  {PARAM_LABELS[param] ?? param}
                  {isModified && <span className="param-modified-dot" />}
                </label>
                <div className="param-options">
                  {options.map((opt) => (
                    <button
                      key={opt}
                      className={`param-option ${currentValue === opt ? "active" : ""}`}
                      onClick={() => handleChange(param, opt)}
                    >
                      {opt.replace(/_/g, " ")}
                    </button>
                  ))}
                </div>
              </div>
            );
          })
        )}
      </div>

      {hasChanges && (
        <div className="param-apply-footer">
          <button
            className="param-apply-btn"
            onClick={handleApply}
            disabled={applying}
          >
            {applying ? "Applying..." : "Apply Changes"}
          </button>
          <button
            className="param-reset-btn"
            onClick={handleReset}
            disabled={applying}
          >
            Reset
          </button>
        </div>
      )}
    </div>
  );
}
