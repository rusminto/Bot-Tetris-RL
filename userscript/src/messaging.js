    // ==========================================
    // 8. CROSS-FRAME MESSAGING (SINGLE HUD BRIDGE)
    // ==========================================
    function notifyState(status) {
        updateHudStatus(status);
        if (!isTopFrame && window.top) {
            try {
                window.top.postMessage({ type: 'TETRIS_BOT_STATE', status: status }, '*');
            } catch(e) {}
        }
    }

    function notifyAction(piece, placement, qVal, total, level, lines, score) {
        updateHudInfo(piece, placement, qVal, total, level, lines, score);
        if (!isTopFrame && window.top) {
            try {
                window.top.postMessage({
                    type: 'TETRIS_BOT_ACTION',
                    piece: piece,
                    placement: placement,
                    qVal: qVal,
                    total: total,
                    level: level,
                    lines: lines,
                    score: score
                }, '*');
            } catch(e) {}
        }
    }

    window.addEventListener("message", (e) => {
        if (!e.data || typeof e.data !== 'object') return;

        if (e.data.type === 'TETRIS_BOT_STATE') {
            updateHudStatus(e.data.status);
        }
        if (e.data.type === 'TETRIS_BOT_ACTION') {
            updateHudInfo(e.data.piece, e.data.placement, e.data.qVal, e.data.total, e.data.level, e.data.lines, e.data.score);
        }

        if (e.data.type === 'TETRIS_BOT_CMD') {
            if (e.data.cmd === 'toggle') botEnabled = e.data.val;
            if (e.data.cmd === 'engine') {
                activeEngine = e.data.val;
                const sel = document.getElementById("bot-engine-select");
                if (sel) sel.value = activeEngine;
            }
            if (e.data.cmd === 'start') {
                sendGameKey(13, 'Enter');
                setTimeout(() => sendGameKey(32, 'Space'), 500);
            }
            if (e.data.cmd === 'speed') keyDelay = e.data.val;
            if (e.data.cmd === 'snap') snapMode = e.data.val;
        }
    });

    function broadcastCommand(cmd, val) {
        if (cmd === 'toggle') botEnabled = val;
        if (cmd === 'engine') activeEngine = val;
        if (cmd === 'speed') keyDelay = val;
        if (cmd === 'snap') snapMode = val;
        if (cmd === 'start') {
            sendGameKey(13, 'Enter');
            setTimeout(() => sendGameKey(32, 'Space'), 500);
        }
        for (let i = 0; i < window.frames.length; i++) {
            try {
                window.frames[i].postMessage({ type: 'TETRIS_BOT_CMD', cmd: cmd, val: val }, '*');
            } catch(e) {}
        }
    }

