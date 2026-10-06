    // ==========================================
    // 7. BOT DECISION & EXECUTION LOOP
    // ==========================================
    let isExecutingInstantPlacement = false;
    const placementStats = { placed: 0, held: 0, mismatches: 0, unreadable: 0, engine: null };
    window.__tetrisBotStats = placementStats;

    function sameState(a, b) {
        return !!(a && b && a[0] === b[0] && a[1] === b[1] && a[2] === b[2]);
    }

    function getPieceName(piece) {
        return piece && piece.getPieceDefinition ? piece.getPieceDefinition().getTypeName() : null;
    }

    // Performs plan.path one action at a time, checking the engine's piece after every step.
    // Returns 'ok', 'mismatch' (the piece ended up somewhere else) or 'lost' (it locked or vanished).
    function executePlannedPath(model, player, livePiece, pieceName, plan) {
        for (let i = 0; i < plan.path.length; i++) {
            model.performControlAction(plan.path[i], null, null);
            if (player.getLivePiece() !== livePiece) return 'lost';
            const actual = readLivePieceState(livePiece, pieceName);
            if (!sameState(actual, plan.states[i + 1])) {
                placementStats.mismatches++;
                console.warn("[TetrisRL] Engine disagreed with the SRS model:", {
                    piece: pieceName, action: plan.path[i], expected: plan.states[i + 1], actual
                });
                return 'mismatch';
            }
        }
        return 'ok';
    }

    // Plans and places the live piece synchronously. Returns 'placed', 'held' or 'skip'.
    function placeLivePiece(model, player, livePiece, matrix) {
        const pieceName = getPieceName(livePiece);
        if (!pieceName || !SRS_CELLS[pieceName]) return 'skip';
        const start = readLivePieceState(livePiece, pieceName);
        if (!start) {
            placementStats.unreadable++;
            return 'skip';
        }

        const stats = getGameStats(player);
        const rows = readMatrixRows(matrix);
        const gravity20G = isGravity20G(model);
        const engine = activeEngine;
        const heldName = getPieceName(player.getHoldPiece ? player.getHoldPiece() : null);
        let plan = planBestPlacement(engine, rows, pieceName, start, gravity20G, stats.level, heldName);

        // Hold if the piece that would be swapped in has a better reachable placement.
        // (Player has no canHoldLivePiece; the model's flag goes false once this piece was held.)
        const canHold = typeof model.canHoldLivePiece === 'function' ? model.canHoldLivePiece() : false;
        if (canHold) {
            const queue = player.getPieceQueue ? player.getPieceQueue() : null;
            const holdName = heldName || getPieceName(queue ? queue.getPieceAtIndex(0) : null);
            if (holdName && SRS_CELLS[holdName]) {
                const spawn = srsSpawnState(holdName);
                if (srsFits(rows, holdName, spawn[0], spawn[1], spawn[2])) {
                    const holdPlan = planBestPlacement(engine, rows, holdName, spawn, gravity20G, stats.level, pieceName);
                    if (holdPlan && (!plan || holdPlan.score > plan.score)) {
                        lastHandledPiece = livePiece;
                        placementStats.held++;
                        model.performControlAction(ACTION_HOLD, null, null);
                        return 'held';
                    }
                }
            }
        }
        if (!plan) return 'skip';

        lastHandledPiece = livePiece;
        totalPiecesPlaced++;
        const leftCol = Math.min(...plan.cells.map(([x]) => x));
        notifyAction(pieceName, { pieceName, rotIdx: plan.rot, x: leftCol }, plan.score, totalPiecesPlaced, stats.level, stats.lines, stats.score);

        // Follow the path; if the engine puts the piece somewhere unexpected, re-plan from there
        for (let attempt = 0; attempt < 3; attempt++) {
            const outcome = executePlannedPath(model, player, livePiece, pieceName, plan);
            if (outcome === 'lost') return 'placed';
            if (outcome === 'ok') break;
            const actual = readLivePieceState(livePiece, pieceName);
            const replan = actual && planBestPlacement(engine, rows, pieceName, actual, gravity20G, stats.level, heldName);
            if (!replan) break;
            plan = replan;
        }

        placementStats.placed++;
        placementStats.engine = engine;  // the engine that placed the last piece
        if (player.getLivePiece() === livePiece) {
            if (typeof player.forceLivePieceHardDrop === 'function') {
                player.forceLivePieceHardDrop();
            } else {
                model.performControlAction(ACTION_HARD_DROP, null, null);
            }
        }
        return 'placed';
    }

    function executeInstantPlacement(model) {
        if (!botEnabled || !snapMode || isExecutingInstantPlacement) return false;

        searchSceneForPlayer();
        const player = capturedPlayer || window.__tetrisPlayer || (model && model.mInterface && model.mInterface.mPlayer);
        if (!player || !player.isGameActive || !player.isGameActive()) {
            notifyState(botEnabled ? "Standby" : "Paused");
            return false;
        }

        const livePiece = player.getLivePiece();
        const matrix = player.getMatrix();
        if (!livePiece || livePiece === lastHandledPiece || !matrix) return false;

        isExecutingInstantPlacement = true;
        try {
            // After a hold the swapped-in piece activates a few frames later and comes back through
            // the activation hook, so there is nothing more to do here in that case
            return placeLivePiece(model, player, livePiece, matrix) !== 'skip';
        } catch (err) {
            console.error("[TetrisRL] Error in executeInstantPlacement:", err);
            return false;
        } finally {
            isExecutingInstantPlacement = false;
        }
    }

