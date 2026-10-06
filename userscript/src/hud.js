    // ==========================================
    // 9. FLOATING HUD OVERLAY (TOP FRAME ONLY)
    // ==========================================
    // RL Engine dropdown, oldest first; offline-game scores are means over 6 games (docs/RL_ALGORITHMS.md)
    const ENGINE_OPTIONS = [
        { id: 'dqn_v1', label: '🤖 V1 DQN (4 features)',
          about: 'Neural network over 4 board features. Finishes the Marathon, mostly with doubles and triples (~760k points).' },
        { id: 'dqn_v2', label: '🤖 V2 DQN (6 features)',
          about: 'Neural network over 6 features. Not trained for 20G: tops out around Level 17–18.' },
        { id: 'cem_v3', label: '🧠 V3 CEM-RL (10 features)',
          about: 'Linear policy trained for Marathon score. About half of its lines are Tetrises (~830k points).' },
        { id: 'cem_v4', label: '🧠 V4 CEM-RL (11 features)',
          about: 'Adds a Tetris feature: ~70% of its lines are Tetrises, but the well can be in any column (~970k points).' },
        { id: 'cem_v5', label: '🧠 V5 CEM-RL (13 features)',
          about: 'Default. Keeps the well at the left or right wall and saves I pieces in hold: ~85% Tetrises (~1.12M points). Tops out a little more often than V4.' },
    ];

    function showEngineInHud(engine) {
        const select = document.getElementById("bot-engine-select");
        const about = document.getElementById("bot-engine-about");
        const option = ENGINE_OPTIONS.find((o) => o.id === engine);
        if (select) select.value = engine;
        if (about && option) about.textContent = option.about;
    }

    function createHud() {
        if (!isTopFrame && window.top && window.top !== window) return;
        if (document.getElementById("tetris-rl-hud")) return;

        const hud = document.createElement("div");
        hud.id = "tetris-rl-hud";
        hud.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 2147483647;
            background: rgba(15, 23, 42, 0.95);
            color: #f8fafc;
            border: 1px solid rgba(56, 189, 248, 0.5);
            border-radius: 12px;
            padding: 14px 18px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            font-size: 13px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7), 0 0 15px rgba(56, 189, 248, 0.25);
            backdrop-filter: blur(10px);
            min-width: 280px;
            user-select: none;
            cursor: move;
        `;

        hud.innerHTML = `
            <div id="hud-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.12); padding-bottom:6px; cursor:move;">
                <span style="font-weight:bold; color:#38bdf8; font-size:14px; pointer-events:none;">🤖 Tetris RL <span style="font-weight:normal; color:#64748b; font-size:11px;">script v__VERSION__</span></span>
                <span id="bot-status-tag" style="background:#065f46; color:#34d399; font-size:11px; padding:2px 8px; border-radius:10px; font-weight:600;">STANDBY</span>
            </div>
            <div style="font-size:12px; color:#94a3b8; line-height:1.6; pointer-events:none;">
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>Score: <b id="bot-score" style="color:#f8fafc;">0</b></span>
                    <span>Level: <b id="bot-level" style="color:#38bdf8;">1</b></span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                    <span>Lines: <b id="bot-lines" style="color:#34d399;">0</b></span>
                    <span>Pieces: <b id="bot-pieces" style="color:#facc15;">0</b></span>
                </div>
                <div>Action: <span id="bot-action" style="color:#a78bfa;">None</span></div>
                <div>Fitness/Q: <span id="bot-qval" style="color:#38bdf8;">--</span></div>
            </div>

            <div style="margin-top:10px; padding:7px 10px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
                <span style="font-size:11px; color:#94a3b8; font-weight:600;">RL Engine:</span>
                <select id="bot-engine-select" style="background:#1e293b; color:#38bdf8; border:1px solid #334155; border-radius:6px; padding:3px 6px; font-size:11px; cursor:pointer;">
                    ${ENGINE_OPTIONS.map((o) => `<option value="${o.id}">${o.label}</option>`).join('')}
                </select>
            </div>

            <div style="margin-top:8px; display:flex; gap:8px;">
                <button id="bot-toggle-btn" style="flex:1; background:#0284c7; border:none; color:#fff; font-weight:bold; padding:6px 12px; border-radius:6px; cursor:pointer; font-size:12px; transition:0.2s;">Pause Bot</button>
                <button id="bot-start-btn" style="flex:1; background:#10b981; border:none; color:#fff; font-weight:bold; padding:6px 12px; border-radius:6px; cursor:pointer; font-size:12px; transition:0.2s;">Press Start</button>
            </div>

            <div style="margin-top:8px; padding:7px 10px; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:8px; display:flex; align-items:center; justify-content:space-between;">
                <div>
                    <div style="font-weight:600; color:#38bdf8; font-size:12px; display:flex; align-items:center; gap:4px;">
                        <span>⚡ Direct Snapping</span>
                    </div>
                    <div style="font-size:10px; color:#94a3b8;">Instant placement (Level 20+ 20G safe)</div>
                </div>
                <label style="position:relative; display:inline-block; width:40px; height:22px; margin:0; cursor:pointer;">
                    <input id="bot-snap-toggle" type="checkbox" checked style="opacity:0; width:0; height:0; margin:0;">
                    <span id="bot-snap-slider" style="position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0; background-color:#10b981; transition:.25s; border-radius:22px;"></span>
                    <span id="bot-snap-knob" style="position:absolute; content:''; height:16px; width:16px; left:21px; bottom:3px; background-color:white; transition:.25s; border-radius:50%; box-shadow:0 1px 3px rgba(0,0,0,0.4);"></span>
                </label>
            </div>

            <div id="bot-speed-container" style="margin-top:8px; display:flex; align-items:center; justify-content:space-between; font-size:11px; color:#94a3b8; transition:opacity 0.2s;">
                <span>Key Delay:</span>
                <input id="bot-speed-range" type="range" min="10" max="60" value="15" style="width:120px; cursor:pointer;">
                <span id="bot-speed-label">15ms</span>
            </div>

            <div id="bot-engine-about" style="margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.12); font-size:10px; line-height:1.45; color:#94a3b8; max-width:280px; pointer-events:none;"></div>
        `;

        document.body.appendChild(hud);

        // Draggable
        let isDragging = false;
        let startX, startY, origLeft, origTop;
        const header = document.getElementById("hud-header");

        header.addEventListener("mousedown", (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            const rect = hud.getBoundingClientRect();
            origLeft = rect.left;
            origTop = rect.top;
            hud.style.right = 'auto';
            hud.style.left = `${origLeft}px`;
            hud.style.top = `${origTop}px`;
        });

        window.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            hud.style.left = `${origLeft + (e.clientX - startX)}px`;
            hud.style.top = `${origTop + (e.clientY - startY)}px`;
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
        });

        // Controls
        document.getElementById("bot-toggle-btn").addEventListener("click", () => {
            botEnabled = !botEnabled;
            document.getElementById("bot-toggle-btn").textContent = botEnabled ? "Pause Bot" : "Resume Bot";
            document.getElementById("bot-toggle-btn").style.background = botEnabled ? "#0284c7" : "#64748b";
            updateHudStatus(botEnabled ? "Active" : "Paused");
            broadcastCommand('toggle', botEnabled);
        });

        document.getElementById("bot-start-btn").addEventListener("click", () => {
            broadcastCommand('start', true);
        });

        const engineSelect = document.getElementById("bot-engine-select");
        if (engineSelect) {
            showEngineInHud(activeEngine);
            engineSelect.addEventListener("change", (e) => {
                broadcastCommand('engine', e.target.value);
                showEngineInHud(activeEngine);
            });
        }

        const snapToggle = document.getElementById("bot-snap-toggle");
        const snapSlider = document.getElementById("bot-snap-slider");
        const snapKnob = document.getElementById("bot-snap-knob");
        const speedContainer = document.getElementById("bot-speed-container");

        function updateSnapUi(enabled) {
            if (snapSlider) snapSlider.style.backgroundColor = enabled ? '#10b981' : '#475569';
            if (snapKnob) snapKnob.style.left = enabled ? '21px' : '3px';
            if (speedContainer) {
                speedContainer.style.opacity = enabled ? '0.4' : '1';
                speedContainer.style.pointerEvents = enabled ? 'none' : 'auto';
            }
        }

        snapToggle.addEventListener("change", (e) => {
            snapMode = e.target.checked;
            updateSnapUi(snapMode);
            broadcastCommand('snap', snapMode);
        });
        updateSnapUi(snapMode);

        const speedRange = document.getElementById("bot-speed-range");
        const speedLabel = document.getElementById("bot-speed-label");
        speedRange.addEventListener("input", (e) => {
            keyDelay = parseInt(e.target.value);
            speedLabel.textContent = `${keyDelay}ms`;
            broadcastCommand('speed', keyDelay);
        });
    }

    function updateHudStatus(status) {
        const tag = document.getElementById("bot-status-tag");
        if (!tag) return;
        tag.textContent = status.toUpperCase();
        if (status === "Playing" || status === "Active") {
            tag.style.background = "#065f46";
            tag.style.color = "#34d399";
        } else {
            tag.style.background = "#7f1d1d";
            tag.style.color = "#f87171";
        }
    }

    function updateHudInfo(piece, placement, qVal, total, level, lines, score) {
        const scEl = document.getElementById("bot-score");
        const lvEl = document.getElementById("bot-level");
        const lnEl = document.getElementById("bot-lines");
        const pcEl = document.getElementById("bot-pieces");
        const aEl = document.getElementById("bot-action");
        const qEl = document.getElementById("bot-qval");

        if (scEl && score !== undefined) scEl.textContent = score.toLocaleString();
        if (lvEl && level !== undefined) lvEl.textContent = level;
        if (lnEl && lines !== undefined) lnEl.textContent = lines;
        if (pcEl) pcEl.textContent = total !== undefined ? total : totalPiecesPlaced;
        if (aEl && placement) {
            const pName = placement.pieceName || piece;
            aEl.textContent = `${pName} -> Rot ${placement.rotIdx}, Col ${placement.x}`;
        }
        if (qEl && qVal !== undefined) qEl.textContent = typeof qVal === 'number' ? qVal.toFixed(2) : qVal;
    }

    window.addEventListener("DOMContentLoaded", () => {
        createHud();
    });
    if (document.readyState === "complete" || document.readyState === "interactive") {
        createHud();
    }

})();
