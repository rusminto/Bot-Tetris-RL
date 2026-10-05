    // ==========================================
    // 6. KEYSTROKE INJECTION (HYBRID BPS + DOM)
    // ==========================================
    function getCanvas() {
        return document.getElementById("GameCanvas") || (window.cc && cc.game && cc.game.canvas) || document.querySelector("canvas");
    }

    function sendGameKey(keyCode, codeStr) {
        const canvas = getCanvas();
        if (canvas && document.activeElement !== canvas) {
            try { canvas.focus(); } catch(e) {}
        }

        if (window.__bpsApp && window.__bpsKeyConverter && window.__bpsDeviceType) {
            try {
                const controlId = window.__bpsKeyConverter(keyCode);
                if (controlId) {
                    window.__bpsApp.handleDeviceControlOn(window.__bpsDeviceType, controlId, 0, 0);
                    setTimeout(() => {
                        if (window.__bpsApp) {
                            window.__bpsApp.handleDeviceControlOff(window.__bpsDeviceType, controlId, 0, 0);
                        }
                    }, 10);
                    return;
                }
            } catch(e) {}
        }

        if (canvas) {
            const downEvt = new KeyboardEvent('keydown', { code: codeStr, key: codeStr, bubbles: true, cancelable: true });
            Object.defineProperty(downEvt, 'keyCode', { get: () => keyCode });
            Object.defineProperty(downEvt, 'which', { get: () => keyCode });
            canvas.dispatchEvent(downEvt);

            setTimeout(() => {
                const upEvt = new KeyboardEvent('keyup', { code: codeStr, key: codeStr, bubbles: true, cancelable: true });
                Object.defineProperty(upEvt, 'keyCode', { get: () => keyCode });
                Object.defineProperty(upEvt, 'which', { get: () => keyCode });
                canvas.dispatchEvent(upEvt);
            }, 10);
        }
    }

    function getPieceLeftCol(livePiece) {
        if (!livePiece) return 3;
        let minX = 10;
        try {
            const count = livePiece.getNumMinos ? livePiece.getNumMinos() : 4;
            for (let i = 0; i < count; i++) {
                const mino = livePiece.getMinoAtIndex ? livePiece.getMinoAtIndex(i) : null;
                if (mino) {
                    const mx = (typeof mino.getX === 'function') ? mino.getX() : mino.mX;
                    if (typeof mx === 'number' && !isNaN(mx) && mx < minX) minX = mx;
                }
            }
        } catch(e) {}
        if (minX < 10) return minX;
        if (typeof livePiece.getMinMinoX === 'function') {
            try {
                const mx = livePiece.getMinMinoX();
                if (typeof mx === 'number' && !isNaN(mx)) return mx;
            } catch(e) {}
        }
        return 3;
    }

    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

