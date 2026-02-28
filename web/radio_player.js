/**
 * ComfyUI Radio Player – frontend extension
 */

import { app } from "../../scripts/app.js";
import Lyricer from "./lyricer.js";

// Load CSS
const link = document.createElement("link");
link.rel = "stylesheet";
link.href = new URL("./radio_player.css", import.meta.url).href;
document.head.appendChild(link);

console.log("[RadioPlayer] JS module loaded OK");

function fmtTime(secs) {
  if (!isFinite(secs)) return "--:--";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

function trackLabel(filename) {
  return filename.replace(/\.[^.]+$/, "");
}

function buildRadioWidget(node) {
  //console.log("[RadioPlayer] buildRadioWidget called for node", node.id);

  const root = document.createElement("div");
  root.style.cssText = "background:#1a1a2e;border:1px solid #a78bfa;border-radius:6px;padding:8px;color:#e0e0e0;width:100%;box-sizing:border-box;font-family:monospace;font-size:12px;";

  const BTN = "background:#2d2d4e;border:1px solid #555;border-radius:4px;color:#e0e0e0;cursor:pointer;padding:3px 9px;font-size:13px;";

  root.innerHTML = `
    <div style="font-size:13px;font-weight:bold;color:#a78bfa;margin-bottom:6px;">📻 Radio Player</div>
    <div class="rp-now" style="color:#34d399;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">No track loaded</div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
      <input type="range" class="rp-progress" min="0" max="100" value="0" step="0.1" style="flex:1;accent-color:#a78bfa;">
      <span class="rp-time" style="color:#9ca3af;font-size:11px;min-width:80px;text-align:right;">--:-- / --:--</span>
    </div>
    <div style="display:flex;gap:6px;margin-bottom:6px;">
      <button data-id="prev" style="${BTN}">⏮</button>
      <button data-id="pp"   style="${BTN}">▶</button>
      <button data-id="next" style="${BTN}">⏭</button>
      <button data-id="stop" style="${BTN}">⏹</button>
    </div>
    <div style="color:#9ca3af;font-size:11px;margin-bottom:2px;">Lyrics</div>
    <div id="lyricer-${node.id}" class="rp-lyricer" style="width:100%;"></div>
    <div style="color:#9ca3af;font-size:11px;margin-bottom:2px;margin-top:8px;">Queue (oldest first)</div>
    <div class="rp-queue" style="max-height:110px;overflow-y:auto;border:1px solid #333;border-radius:4px;background:#111;"></div>
    <div class="rp-status" style="color:#6b7280;font-size:10px;margin-top:4px;">Idle – set folder path above</div>
  `;

  let allTracks = [];
  let currentIdx = -1;
  let knownPaths = new Set();
  let polling = null;
  // Set true when user explicitly hits ⏹ — blocks autoplay until ▶ is pressed
  let userStopped = false;

  const audio = new Audio();
  audio.preload = "auto";

  // Initialize Lyricer
  const lyricerElId = `lyricer-${node.id}`;
  const lrc = new Lyricer({ "divID": lyricerElId, "showLines": 2 });

  const nowEl = root.querySelector(".rp-now");
  const progress = root.querySelector(".rp-progress");
  const timeEl = root.querySelector(".rp-time");
  const queueEl = root.querySelector(".rp-queue");
  const statusEl = root.querySelector(".rp-status");
  const btnPP = root.querySelector("[data-id=pp]");

  async function fetchLrc(url) {
    const container = document.getElementById(lyricerElId);
    if (!container) return;

    if (!url) {
      container.innerHTML = "<div class='rp-lyricer-nodata'>No lyrics available</div>";
      return;
    }
    try {
      const res = await fetch(url);
      if (res.ok) {
        const text = await res.text();
        lrc.setLrc(text);
      } else {
        container.innerHTML = "<div class='rp-lyricer-nodata'>Lyrics not found</div>";
      }
    } catch (e) {
      console.error("[RadioPlayer] Error fetching lyrics:", e);
      container.innerHTML = "<div class='rp-lyricer-nodata'>Error loading lyrics</div>";
    }
  }

  function playIdx(idx) {
    if (idx < 0 || idx >= allTracks.length) return;
    userStopped = false;
    currentIdx = idx;
    const track = allTracks[idx];
    audio.src = track.url;
    audio.play().catch(() => { statusEl.textContent = "Autoplay blocked – press ▶"; });
    renderQueue();
    fetchLrc(track.lrc_url);
  }

  function playNext() {
    if (currentIdx + 1 < allTracks.length) {
      playIdx(currentIdx + 1);
    } else {
      // Stay at end — don't reset currentIdx, just wait for new tracks via poll
      btnPP.textContent = "▶";
      nowEl.textContent = "Waiting for next track…";
    }
  }

  audio.addEventListener("play", () => { btnPP.textContent = "⏸"; });
  audio.addEventListener("pause", () => { btnPP.textContent = "▶"; });
  audio.addEventListener("ended", () => playNext());
  audio.addEventListener("timeupdate", () => {
    if (!audio.duration) return;
    progress.value = (audio.currentTime / audio.duration) * 100;
    timeEl.textContent = `${fmtTime(audio.currentTime)} / ${fmtTime(audio.duration)}`;
    lrc.move(audio.currentTime);
  });
  audio.addEventListener("error", () => {
    statusEl.textContent = "Audio error – skipping";
    setTimeout(playNext, 500);
  });

  // Handle click-to-seek from lyrics
  window.addEventListener('lyricerclick', function (e) {
    // Only handle if this event came from our div (bubbles up)
    const target = e.target.closest(`#${lyricerElId}`);
    if (target && e.detail.time >= 0) {
      audio.currentTime = e.detail.time;
    }
  });

  root.querySelector("[data-id=prev]").addEventListener("click", () => {
    if (currentIdx > 0) playIdx(currentIdx - 1);
  });

  btnPP.addEventListener("click", () => {
    if (audio.paused) {
      userStopped = false;
      if (currentIdx >= 0 && currentIdx < allTracks.length) {
        // Resume current track (works after ⏹ since we kept audio.src)
        audio.play().catch(() => { });
        // Re-fetch lyrics if div is empty
        if (!document.getElementById(lyricerElId).hasChildNodes()) {
          fetchLrc(allTracks[currentIdx].lrc_url);
        }
      } else if (allTracks.length > 0) {
        playIdx(0);
      }
    } else {
      audio.pause();
    }
  });

  root.querySelector("[data-id=next]").addEventListener("click", () => {
    if (currentIdx < allTracks.length - 1) playIdx(currentIdx + 1);
  });

  root.querySelector("[data-id=stop]").addEventListener("click", () => {
    userStopped = true;
    audio.pause();
    audio.currentTime = 0;
    nowEl.textContent = "Stopped – press ▶ to resume";
    progress.value = 0;
    timeEl.textContent = "--:-- / --:--";
    // Intentionally keep currentIdx and audio.src intact so ▶ resumes correctly
    renderQueue();
  });

  progress.addEventListener("input", () => {
    if (audio.duration) audio.currentTime = (progress.value / 100) * audio.duration;
  });

  function renderQueue() {
    if (currentIdx >= 0 && currentIdx < allTracks.length && !userStopped)
      nowEl.textContent = `♪ ${trackLabel(allTracks[currentIdx].filename)}`;

    queueEl.innerHTML = "";
    allTracks.forEach((t, i) => {
      const div = document.createElement("div");
      div.style.cssText = "padding:3px 6px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11px;";
      div.style.color = i < currentIdx ? "#555" : i === currentIdx ? "#34d399" : "#e0e0e0";
      if (i === currentIdx) div.style.background = "#1a3a2e";
      const pos = i < currentIdx ? "✓" : i === currentIdx ? "▶" : String(i - currentIdx);
      div.textContent = `${pos}  ${trackLabel(t.filename)}`;
      div.title = t.path;
      div.addEventListener("click", () => playIdx(i));
      queueEl.appendChild(div);
    });

    // Scroll current track into view
    const items = queueEl.children;
    if (currentIdx >= 0 && currentIdx < items.length)
      items[currentIdx].scrollIntoView({ block: "nearest" });
  }

  async function scanFolder(folder) {
    if (!folder || !folder.trim()) return;
    try {
      const res = await fetch(`/radio_player/scan?folder=${encodeURIComponent(folder.trim())}`);
      const data = await res.json();
      if (data.error) { statusEl.textContent = `Error: ${data.error}`; return; }

      let added = 0;
      for (const t of data.tracks) {
        if (!knownPaths.has(t.path)) { knownPaths.add(t.path); allTracks.push(t); added++; }
        else {
          // Update lrc_url in case it was added later
          const existing = allTracks.find(x => x.path === t.path);
          if (existing) existing.lrc_url = t.lrc_url;
        }
      }
      allTracks.sort((a, b) => a.mtime - b.mtime);

      if (added > 0) {
        statusEl.textContent = `+${added} new – ${allTracks.length} total`;
        renderQueue();
        // Autoplay only if: not user-stopped, and player is idle (never started or finished queue)
        const idleAtEnd = currentIdx < 0 || currentIdx >= (allTracks.length - added - 1);
        if (!userStopped && audio.paused && idleAtEnd) {
          playIdx(Math.max(0, allTracks.length - added));
        }
      } else {
        statusEl.textContent = `Polling… ${allTracks.length} track(s)`;
      }
    } catch (e) {
      statusEl.textContent = `Scan error: ${e.message}`;
    }
  }

  root._startPolling = function (folder, intervalMs) {
    if (polling) clearInterval(polling);
    polling = null;
    if (!folder || !folder.trim()) return;
    scanFolder(folder);
    polling = setInterval(() => scanFolder(folder), intervalMs || 5000);
  };

  root._stop = function () {
    if (polling) clearInterval(polling);
    polling = null;
    audio.pause();
    audio.src = "";
  };

  return root;
}

// ─── Extension ────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "Comfy.RadioPlayer",

  nodeCreated(node) {
    //console.log("[RadioPlayer] nodeCreated fired, comfyClass =", node.comfyClass, "type =", node.type);
    if (node.comfyClass !== "RadioPlayer" && node.type !== "RadioPlayer") return;
    console.log("[RadioPlayer] Matched RadioPlayer node – building widget");

    const root = buildRadioWidget(node);

    let domWidget;
    try {
      domWidget = node.addDOMWidget("radio_ui", "RADIO_PLAYER_UI", root, {
        getValue() { return ""; },
        setValue() { },
      });
      domWidget.serialize = false;
      //console.log("[RadioPlayer] addDOMWidget succeeded", domWidget);
    } catch (e) {
      //console.error("[RadioPlayer] addDOMWidget failed:", e);
      return;
    }

    setTimeout(() => {
      const folderW = node.widgets?.find(w => w.name === "folder");
      const intervalW = node.widgets?.find(w => w.name === "poll_interval_seconds");
      //console.log("[RadioPlayer] widgets:", node.widgets?.map(w => w.name));

      if (!folderW) { console.warn("[RadioPlayer] No folder widget found"); return; }

      function restartPolling() {
        root._startPolling(folderW.value ?? "", (intervalW?.value ?? 5) * 1000);
      }

      const origFolderCB = folderW.callback;
      folderW.callback = function (v) { origFolderCB?.call(this, v); restartPolling(); };

      if (intervalW) {
        const origIntervalCB = intervalW.callback;
        intervalW.callback = function (v) { origIntervalCB?.call(this, v); restartPolling(); };
      }

      restartPolling();
    }, 100);

    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
      origRemoved?.call(this);
      root._stop();
    };
  },
});
