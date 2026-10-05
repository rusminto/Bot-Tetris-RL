    // ==========================================
    // 5. GAME HOOKING & LIVE STATS EXTRACTION
    // ==========================================
    let capturedPlayer = null;
    let botEnabled = true;
    let snapMode = true; // Direct snapping mode
    let keyDelay = 15; // ms per keystroke
    let isBusyPlaying = false;
    let lastHandledPiece = null;
    let totalPiecesPlaced = 0;

    function getGameStats(player) {
        let level = 1;
        let lines = 0;
        let score = 0;

        if (!player) return { level, lines, score };

        try {
            const checkComp = (comp) => {
                if (!comp) return;
                if (typeof comp.getCurrentLevelIndex === 'function') {
                    const lvl = comp.getCurrentLevelIndex();
                    if (typeof lvl === 'number' && !isNaN(lvl) && lvl >= 0) {
                        level = Math.max(level, lvl + 1);
                    }
                }
                if (typeof comp.getStatValue === 'function') {
                    const l = comp.getStatValue(2059679406); // kTrackedValueId_Lines
                    if (typeof l === 'number' && !isNaN(l) && l >= 0) {
                        lines = Math.max(lines, l);
                    }
                }
                if (typeof comp.getScore === 'function') {
                    const s = comp.getScore();
                    if (typeof s === 'number' && !isNaN(s) && s >= 0) {
                        score = Math.max(score, s);
                    }
                }
            };

            if (typeof player.getNumComponents === 'function' && typeof player.getComponentAtIndex === 'function') {
                const count = player.getNumComponents();
                for (let i = 0; i < count; i++) {
                    checkComp(player.getComponentAtIndex(i));
                }
            }

            if (player.mComponents) {
                const list = player.mComponents;
                const num = (typeof list.getNumObjects === 'function') ? list.getNumObjects() : (list.length || 0);
                for (let i = 0; i < num; i++) {
                    const c = (typeof list.getAtIndex === 'function') ? list.getAtIndex(i) : list[i];
                    checkComp(c);
                }
            }

            if (player._components && Array.isArray(player._components)) {
                for (const c of player._components) {
                    checkComp(c);
                }
            }

            if (lines > 0 && level === 1) {
                level = Math.min(30, Math.floor(lines / 10) + 1);
            }
        } catch(e) {}

        return { level, lines, score };
    }

    function hookPlayerPrototype(proto) {
        if (!proto || proto._tetrisBotHooked) return;
        proto._tetrisBotHooked = true;

        const origStart = proto.startGame;
        proto.startGame = function() {
            capturedPlayer = this;
            window.__tetrisPlayer = this;
            notifyState("Playing");
            return origStart ? origStart.apply(this, arguments) : undefined;
        };

        const origConstruct = proto.constructMatrix;
        proto.constructMatrix = function() {
            capturedPlayer = this;
            window.__tetrisPlayer = this;
            return origConstruct.apply(this, arguments);
        };

        const origLock = proto.lockLivePieceIntoMatrix;
        if (typeof origLock === 'function') {
            proto.lockLivePieceIntoMatrix = function() {
                capturedPlayer = this;
                window.__tetrisPlayer = this;
                return origLock.apply(this, arguments);
            };
        }
    }

    function hookInputAdapterPrototype(proto) {
        if (!proto || proto._tetrisInputHooked) return;
        proto._tetrisInputHooked = true;

        const origAdd = proto.addKeyEventListeners;
        proto.addKeyEventListeners = function() {
            window.__bpsInputAdapter = this;
            window.__bpsApp = this.mBPSApp;
            return origAdd ? origAdd.apply(this, arguments) : undefined;
        };
    }

    function hookModelPrototype(proto) {
        if (!proto || proto._tetrisModelHooked) return;
        proto._tetrisModelHooked = true;

        const origActivate = proto.handleLivePieceDidActivate;
        proto.handleLivePieceDidActivate = function() {
            window.__tetrisModel = this;
            const res = origActivate ? origActivate.apply(this, arguments) : undefined;
            if (botEnabled && snapMode) {
                try {
                    executeInstantPlacement(this);
                } catch(e) {
                    console.error("[TetrisRL] Instant placement hook error:", e);
                }
            }
            return res;
        };

        const origCapture = proto.captureStateForNewLivePiece;
        proto.captureStateForNewLivePiece = function() {
            window.__tetrisModel = this;
            return origCapture ? origCapture.apply(this, arguments) : undefined;
        };

        const origPerform = proto.performControlAction;
        proto.performControlAction = function() {
            window.__tetrisModel = this;
            return origPerform ? origPerform.apply(this, arguments) : undefined;
        };
    }

    function setupSystemHook(sys) {
        if (!sys || sys._tetrisBotHooked) return;
        sys._tetrisBotHooked = true;
        const origRegister = sys.register;

        sys.register = function() {
            try {
                const declareIdx = arguments.length - 1;
                const origDeclare = arguments[declareIdx];

                if (typeof origDeclare === 'function') {
                    arguments[declareIdx] = function(_export, _context) {
                        const hookedExport = function(key, val) {
                            try {
                                if (val && val.prototype) {
                                    if (typeof val.prototype.constructMatrix === 'function') hookPlayerPrototype(val.prototype);
                                    if (typeof val.prototype.addKeyEventListeners === 'function') hookInputAdapterPrototype(val.prototype);
                                    if (typeof val.prototype.forceHardDropControlAction === 'function') hookModelPrototype(val.prototype);
                                }
                                if (val && typeof val.x1936873422386899430x === 'function') {
                                    window.__bpsKeyConverter = val.x1936873422386899430x;
                                    window.__bpsDeviceType = val.x4119954412175409567x;
                                }
                            } catch (err) {}
                            return _export.apply(this, arguments);
                        };
                        return origDeclare.call(this, hookedExport, _context);
                    };
                }
            } catch (err) {}
            return origRegister.apply(this, arguments);
        };
    }

    let hookAttached = false;
    function tryAttachSystemHook() {
        if (hookAttached) return;
        if (window.System && typeof window.System.register === 'function') {
            hookAttached = true;
            setupSystemHook(window.System);
        }
    }
    tryAttachSystemHook();
    const sysPoll = setInterval(() => {
        tryAttachSystemHook();
        if (hookAttached) clearInterval(sysPoll);
    }, 2);

    function searchSceneForPlayer() {
        if (capturedPlayer && capturedPlayer.isGameActive && capturedPlayer.isGameActive()) return;
        try {
            const cc = window.cc;
            if (cc && cc.director && cc.director.getScene) {
                const scene = cc.director.getScene();
                if (scene) {
                    function checkNode(node) {
                        if (!node) return null;
                        if (node._components) {
                            for (const c of node._components) {
                                if (c.mPlayer && c.mPlayer.getMatrix) return c.mPlayer;
                                if (c.mBPSApp) window.__bpsApp = c.mBPSApp;
                                if (c.getPlayer && typeof c.getPlayer === 'function') {
                                    const p = c.getPlayer();
                                    if (p && p.getMatrix) return p;
                                }
                            }
                        }
                        if (node.children) {
                            for (const ch of node.children) {
                                const found = checkNode(ch);
                                if (found) return found;
                            }
                        }
                        return null;
                    }
                    const p = checkNode(scene);
                    if (p) {
                        capturedPlayer = p;
                        window.__tetrisPlayer = p;
                    }
                }
            }
        } catch(e) {}
    }

    function ensureModelHooked(m) {
        if (!m) return;
        window.__tetrisModel = m;
        const proto = Object.getPrototypeOf(m);
        if (proto && !proto._tetrisModelHooked) {
            hookModelPrototype(proto);
        }
    }

    function getPlayerModel(player) {
        if (window.__tetrisModel && typeof window.__tetrisModel.performControlAction === 'function') {
            return window.__tetrisModel;
        }
        if (!player) return null;
        try {
            const cc = player.mControlComponent || (player.getControlComponent && player.getControlComponent());
            if (cc) {
                if (typeof cc.x449964748170772282x === 'function') {
                    const iface = cc.x449964748170772282x();
                    if (iface && typeof iface.getModel === 'function') {
                        const m = iface.getModel();
                        if (m) { ensureModelHooked(m); return m; }
                    }
                }
                if (cc.x4394682035953266388x && typeof cc.x4394682035953266388x.getModel === 'function') {
                    const m = cc.x4394682035953266388x.getModel();
                    if (m) { ensureModelHooked(m); return m; }
                }
                for (const k in cc) {
                    if (cc[k] && typeof cc[k].getModel === 'function') {
                        const m = cc[k].getModel();
                        if (m) { ensureModelHooked(m); return m; }
                    }
                }
            }
            if (typeof player.getModel === 'function') {
                const m = player.getModel();
                if (m) { ensureModelHooked(m); return m; }
            }
        } catch(e) {}
        return null;
    }

