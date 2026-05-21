import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useStore } from "../store";
import { generateFromVibe } from "../api";

export default function PlaylistPanel() {
  const {
    playlist,
    showPlaylistPanel,
    removeFromPlaylist,
    togglePlaylistPanel,
    setActiveSession,
  } = useStore();

  const [newVibe, setNewVibe] = useState("");

  const addMutation = useMutation({
    mutationFn: (vibe: string) => generateFromVibe(vibe),
    onSuccess: (data, vibe) => {
      useStore.getState().addToPlaylist({
        id: `track-${Date.now()}`,
        sessionId: data.session_id,
        vibe,
        title: data.title,
        palette: data.palette,
        justification: data.justification,
        config: data.config,
        duration: 30,
        status: "ready",
      });
      setNewVibe("");
    },
  });

  const handlePlayTrack = (track: (typeof playlist)[0]) => {
    setActiveSession({
      sessionId: track.sessionId,
      title: track.title,
      justification: track.justification,
      palette: track.palette,
      config: track.config,
      vibe: track.vibe,
    });
  };

  if (!showPlaylistPanel) return null;

  return (
    <div className="playlist-panel">
      <div className="playlist-header">
        <h3>Playlist</h3>
        <button className="param-close" onClick={togglePlaylistPanel}>
          &times;
        </button>
      </div>

      <div className="playlist-tracks">
        {playlist.length === 0 && (
          <div className="playlist-empty">
            No tracks yet. Select a preset or type a vibe to add tracks.
          </div>
        )}
        {playlist.map((track, index) => (
          <div
            key={track.id}
            className="playlist-track"
            onClick={() => track.status === "ready" && handlePlayTrack(track)}
            style={track.status !== "ready" ? { opacity: 0.5, cursor: "default" } : undefined}
          >
            <div className="playlist-track-colors">
              {track.status === "pending" ? (
                <span className="spinner-sm" style={{ width: 10, height: 10, border: "2px solid #555", borderTopColor: "#aaa" }} />
              ) : (
                track.palette.slice(0, 3).map((color, i) => (
                  <span
                    key={i}
                    className="color-dot-sm"
                    style={{ backgroundColor: color }}
                  />
                ))
              )}
            </div>
            <div className="playlist-track-info">
              <span className="playlist-track-title">
                {track.status === "pending" ? "Generating..." : track.status === "error" ? "Failed" : track.title}
              </span>
              <span className="playlist-track-vibe">{track.vibe}</span>
            </div>
            <button
              className="playlist-track-remove"
              onClick={(e) => {
                e.stopPropagation();
                removeFromPlaylist(index);
              }}
            >
              &times;
            </button>
          </div>
        ))}
      </div>

      <div className="playlist-add">
        <input
          type="text"
          placeholder="Type a vibe to add..."
          value={newVibe}
          onChange={(e) => setNewVibe(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && newVibe.trim()) {
              addMutation.mutate(newVibe.trim());
            }
          }}
        />
        <button
          className="playlist-add-btn"
          onClick={() => newVibe.trim() && addMutation.mutate(newVibe.trim())}
          disabled={addMutation.isPending || !newVibe.trim()}
        >
          {addMutation.isPending ? "..." : "+"}
        </button>
      </div>
    </div>
  );
}
