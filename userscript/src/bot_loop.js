    async function executeBotMove() {
        if (!botEnabled || isBusyPlaying) return;

        searchSceneForPlayer();

        const player = capturedPlayer || window.__tetrisPlayer;
        if (!player || !player.isGameActive || !player.isGameActive()) {
            notifyState(botEnabled ? "Standby" : "Paused");
            return;
        }

        const livePiece = player.getLivePiece();
        if (!livePiece || livePiece === lastHandledPiece) return;

        const matrix = player.getMatrix();
        if (!matrix) return;

        // In Direct Snapping mode, attempt instant placement via model first
        if (snapMode) {
            const model = getPlayerModel(player);
            if (model && typeof model.performControlAction === 'function') {
                if (executeInstantPlacement(model)) return;
            }
        }

        // Keystroke mode execution path
        lastHandledPiece = livePiece;
        isBusyPlaying = true;
        notifyState("Playing");

        try {
            // 1. Get Live Game Stats
            const stats = getGameStats(player);

            // 2. Extract 20x10 board from game matrix
            const board = [];
            for (let r = 0; r < 20; r++) {
                const row = new Uint8Array(10);
                const gameY = 19 - r;
                for (let c = 0; c < 10; c++) {
                    row[c] = (matrix.getMinoAt(c, gameY) !== null) ? 1 : 0;
                }
                board.push(row);
            }

            // 3. Identify Piece Type & Hold Candidate
            let pieceName = null;
            if (livePiece.getPieceDefinition) {
                pieceName = livePiece.getPieceDefinition().getTypeName();
            }
            if (!pieceName || !PIECES[pieceName]) {
                isBusyPlaying = false;
                return;
            }

            const holdModel = getPlayerModel(player);
            const canHold = holdModel && typeof holdModel.canHoldLivePiece === 'function' ? holdModel.canHoldLivePiece() : false;
            let holdName = null;
            if (canHold) {
                const holdPiece = player.getHoldPiece ? player.getHoldPiece() : null;
                if (holdPiece && holdPiece.getPieceDefinition) {
                    holdName = holdPiece.getPieceDefinition().getTypeName();
                } else if (player.getPieceQueue && player.getPieceQueue().getPieceAtIndex(0)) {
                    const nextP = player.getPieceQueue().getPieceAtIndex(0);
                    if (nextP && nextP.getPieceDefinition) holdName = nextP.getPieceDefinition().getTypeName();
                }
            }

            let useHold = false;
            let chosenPlacement = null;
            let chosenScore = 0;

            // 4. Decision Making: CEM-RL vs DQN V2 vs DQN V1
            if (activeEngine === 'cem') {
                const result = selectBestCemPlacement(board, pieceName, holdName, canHold);
                if (result) {
                    useHold = result.isHold;
                    chosenPlacement = result;
                    chosenScore = result.score;
                }
            } else {
                const isV1 = (activeEngine === 'dqn_v1');
                const currentEval = selectBestDqnPlacement(board, pieceName, stats.level, isV1);
                if (currentEval) {
                    chosenPlacement = currentEval.placement;
                    chosenScore = currentEval.score;
                }

                if (canHold && holdName && PIECES[holdName]) {
                    const holdEval = selectBestDqnPlacement(board, holdName, stats.level, isV1);
                    if (holdEval && holdEval.score > chosenScore) {
                        useHold = true;
                    }
                }
            }

            // 5. Handle Hold Action
            if (useHold) {
                sendGameKey(67, 'KeyC');
                lastHandledPiece = null;
                await sleep(keyDelay * 2);
                return;
            }

            if (!chosenPlacement) {
                return;
            }

            totalPiecesPlaced++;
            notifyAction(pieceName, chosenPlacement, chosenScore, totalPiecesPlaced, stats.level, stats.lines, stats.score);

            // 6. Execute Placement via Keystrokes
            const targetRot = chosenPlacement.rotIdx;
            const currentFacing = livePiece.getFacing ? livePiece.getFacing() : 0;
            const numRotates = (targetRot - currentFacing + 4) % 4;

            for (let r = 0; r < numRotates; r++) {
                sendGameKey(38, 'ArrowUp');
                await sleep(keyDelay);
            }
            if (numRotates > 0) await sleep(keyDelay);

            const curX = getPieceLeftCol(livePiece);
            const dx = chosenPlacement.x - curX;
            const moveKey = dx < 0 ? 'ArrowLeft' : 'ArrowRight';
            const moveKeyCode = dx < 0 ? 37 : 39;
            const count = Math.abs(dx);

            for (let m = 0; m < count; m++) {
                sendGameKey(moveKeyCode, moveKey);
                await sleep(keyDelay);
            }

            await sleep(keyDelay);
            sendGameKey(32, 'Space');
            await sleep(keyDelay * 2);

        } catch(err) {
            console.error("[TetrisRL] Error executing bot move:", err);
        } finally {
            isBusyPlaying = false;
        }
    }

    if (!isTopFrame || document.getElementById("GameCanvas")) {
        setInterval(executeBotMove, 20);
    }

