    // ==========================================
    // 3b. REACHABLE PLACEMENT SEARCH (SRS + 20G MODEL OF THE BPS ENGINE)
    // ==========================================
    // When the fall speed is 0 ms (Level 20+), the BPS engine drops the piece onto the stack inside
    // handleLivePieceDidActivate, before our hook runs, and again after every successful move or
    // rotation. So instead of assuming every column can be reached by a straight drop, search the
    // placements the piece can actually reach from where it is, and the action path to each.
    // Mirrored in reinforcement-learning/src/tetris_sim.py, which the CEM policy is trained in.
    const MATRIX_W = 10;
    const MATRIX_H = 24;   // 20 visible rows + 4 hidden rows above them
    const VISIBLE_H = 20;
    const HIDDEN_ROW_PENALTY = 1000;

    const ACTION_MOVE_LEFT = 1;
    const ACTION_MOVE_RIGHT = 2;
    const ACTION_ROTATE_CW = 5;   // superRotateCW: SRS rotation with wall kicks
    const ACTION_ROTATE_CCW = 6;  // superRotateCCW
    const ACTION_HARD_DROP = 9;
    const ACTION_HOLD = 10;

    // SRS spawn orientations, rows listed top -> bottom inside the rotation box
    const SRS_BOXES = {
        I: ['....', 'XXXX', '....', '....'],
        J: ['X..', 'XXX', '...'],
        L: ['..X', 'XXX', '...'],
        O: ['XX', 'XX'],
        S: ['.XX', 'XX.', '...'],
        T: ['.X.', 'XXX', '...'],
        Z: ['XX.', '.XX', '...']
    };

    // SRS_CELLS[piece][rot] = [[r, c], ...]: r = rows below the box top, c = columns right of the box left
    const SRS_CELLS = (() => {
        const out = {};
        for (const p in SRS_BOXES) {
            const box = SRS_BOXES[p];
            const n = box.length;
            let cells = [];
            for (let r = 0; r < n; r++) {
                for (let c = 0; c < n; c++) {
                    if (box[r][c] === 'X') cells.push([r, c]);
                }
            }
            out[p] = [cells];
            for (let k = 1; k < 4; k++) {
                if (p !== 'O') cells = cells.map(([r, c]) => [c, n - 1 - r]); // clockwise
                out[p].push(cells);
            }
        }
        return out;
    })();

    // Kick offsets [dx, dy] with +y = up, keyed by "from>to"
    const SRS_KICKS_JLSTZ = {
        '0>1': [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
        '1>0': [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
        '1>2': [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
        '2>1': [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
        '2>3': [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
        '3>2': [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
        '3>0': [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
        '0>3': [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]]
    };
    const SRS_KICKS_I = {
        '0>1': [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
        '1>0': [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
        '1>2': [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
        '2>1': [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
        '2>3': [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
        '3>2': [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
        '3>0': [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
        '0>3': [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]]
    };

    // A piece state is [rot, bx, by]: rotation, box left column, box top row (y = 0 is the bottom row)
    function srsCellsAt(piece, rot, bx, by) {
        return SRS_CELLS[piece][rot].map(([r, c]) => [bx + c, by - r]);
    }

    function srsFits(rows, piece, rot, bx, by) {
        const cells = SRS_CELLS[piece][rot];
        for (let i = 0; i < cells.length; i++) {
            const x = bx + cells[i][1];
            const y = by - cells[i][0];
            if (x < 0 || x >= MATRIX_W || y < 0 || y >= MATRIX_H) return false;
            if (rows[y] & (1 << x)) return false;
        }
        return true;
    }

    function srsDropY(rows, piece, rot, bx, by) {
        while (srsFits(rows, piece, rot, bx, by - 1)) by--;
        return by;
    }

    function srsRotate(rows, piece, rot, bx, by, dir) {
        const to = (rot + dir + 4) % 4;
        if (piece === 'O') return [to, bx, by];
        const kicks = (piece === 'I' ? SRS_KICKS_I : SRS_KICKS_JLSTZ)[rot + '>' + to];
        for (const [dx, dy] of kicks) {
            if (srsFits(rows, piece, to, bx + dx, by + dy)) return [to, bx + dx, by + dy];
        }
        return null;
    }

    function srsSpawnState(piece) {
        return [0, piece === 'O' ? 4 : 3, VISIBLE_H - 1];
    }

    function readMatrixRows(matrix) {
        const rows = new Int32Array(MATRIX_H);
        for (let y = 0; y < MATRIX_H; y++) {
            let mask = 0;
            for (let x = 0; x < MATRIX_W; x++) {
                if (matrix.getMinoAt(x, y) !== null) mask |= (1 << x);
            }
            rows[y] = mask;
        }
        return rows;
    }

    // The live piece's [rot, bx, by], or null if its minos don't match the SRS model
    function readLivePieceState(livePiece, piece) {
        const minos = [];
        const count = livePiece.getNumMinos ? livePiece.getNumMinos() : 4;
        for (let i = 0; i < count; i++) {
            const m = livePiece.getMinoAtIndex(i);
            if (!m) return null;
            minos.push([m.getX(), m.getY()]);
        }
        const have = minos.map(([x, y]) => x + ',' + y).sort().join(';');
        const matchRot = (rot) => {
            let minC = 99, minR = 99, minX = 99, maxY = -99;
            for (const [r, c] of SRS_CELLS[piece][rot]) {
                minC = Math.min(minC, c);
                minR = Math.min(minR, r);
            }
            for (const [x, y] of minos) {
                minX = Math.min(minX, x);
                maxY = Math.max(maxY, y);
            }
            const state = [rot, minX - minC, maxY + minR];
            const want = srsCellsAt(piece, rot, state[1], state[2]).map(([x, y]) => x + ',' + y).sort().join(';');
            return want === have ? state : null;
        };
        const facing = livePiece.getFacing ? (((livePiece.getFacing() % 4) + 4) % 4) : 0;
        const exact = matchRot(facing);
        if (exact) return exact;
        for (let rot = 0; rot < 4; rot++) {
            const s = matchRot(rot);
            if (s) return s;
        }
        return null;
    }

    function isGravity20G(model) {
        try {
            if (typeof model.getCurrentFallSpeedMSEC === 'function') return model.getCurrentFallSpeedMSEC() <= 0;
            if (typeof model.mNormalFallSpeedMSEC === 'number') return model.mNormalFallSpeedMSEC <= 0;
        } catch (e) {}
        return false;
    }

    // BFS over states reachable with left/right/SRS rotations (settling after each action at 20G).
    // Returns one entry per distinct final placement, with the shortest action path to it and the
    // expected piece state after each action (states[0] is the start).
    function searchReachablePlacements(rows, piece, start, gravity20G) {
        const keyOf = (s) => (s[0] * 64 + (s[1] + 8)) * 64 + (s[2] + 8);
        const s0 = start.slice();
        if (gravity20G) s0[2] = srsDropY(rows, piece, s0[0], s0[1], s0[2]);
        const nodes = new Map([[keyOf(s0), { state: s0, parent: -1, action: 0 }]]);
        const order = [s0];
        const actions = piece === 'O'
            ? [ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT]
            : [ACTION_MOVE_LEFT, ACTION_MOVE_RIGHT, ACTION_ROTATE_CW, ACTION_ROTATE_CCW];

        for (let qi = 0; qi < order.length; qi++) {
            const cur = order[qi];
            const curKey = keyOf(cur);
            for (const action of actions) {
                let next = null;
                if (action === ACTION_MOVE_LEFT || action === ACTION_MOVE_RIGHT) {
                    const dx = action === ACTION_MOVE_LEFT ? -1 : 1;
                    if (srsFits(rows, piece, cur[0], cur[1] + dx, cur[2])) next = [cur[0], cur[1] + dx, cur[2]];
                } else {
                    next = srsRotate(rows, piece, cur[0], cur[1], cur[2], action === ACTION_ROTATE_CW ? 1 : -1);
                }
                if (!next) continue;
                if (gravity20G) next[2] = srsDropY(rows, piece, next[0], next[1], next[2]);
                const k = keyOf(next);
                if (nodes.has(k)) continue;
                nodes.set(k, { state: next, parent: curKey, action });
                order.push(next);
            }
        }

        const placements = new Map();
        for (const s of order) {  // BFS order, so the first path to a placement is the shortest
            const finalY = gravity20G ? s[2] : srsDropY(rows, piece, s[0], s[1], s[2]);
            const cells = srsCellsAt(piece, s[0], s[1], finalY);
            const cellKey = cells.map(([x, y]) => y * 16 + x).sort((a, b) => a - b).join(',');
            if (placements.has(cellKey)) continue;
            const path = [];
            const states = [];
            for (let node = nodes.get(keyOf(s)); node; node = nodes.get(node.parent)) {
                states.push(node.state);
                if (node.parent !== -1) path.push(node.action);
            }
            path.reverse();
            states.reverse();
            placements.set(cellKey, { rot: s[0], bx: s[1], by: finalY, cells, path, states });
        }
        return Array.from(placements.values());
    }

    function lockPlacement(rows, cells) {
        const next = Int32Array.from(rows);
        for (const [x, y] of cells) next[y] |= (1 << x);
        const full = (1 << MATRIX_W) - 1;
        const out = new Int32Array(MATRIX_H);
        let kept = 0;
        let cleared = 0;
        let pieceMinosCleared = 0;
        for (let y = 0; y < MATRIX_H; y++) {
            if (next[y] === full) {
                cleared++;
                for (const [, cy] of cells) if (cy === y) pieceMinosCleared++;
            } else {
                out[kept++] = next[y];
            }
        }
        return { rows: out, cleared, pieceMinosCleared };
    }

    // Visible rows as the top-down 20x10 board the feature extractors expect
    function rowsToBoard(rows) {
        const board = [];
        for (let r = 0; r < VISIBLE_H; r++) {
            const y = VISIBLE_H - 1 - r;
            const row = new Uint8Array(MATRIX_W);
            for (let x = 0; x < MATRIX_W; x++) row[x] = (rows[y] >> x) & 1;
            board.push(row);
        }
        return board;
    }

    // Value of the board after locking `placement` under the active engine; -Infinity for a lock out
    function scorePlacement(engine, rows, placement, level) {
        if (placement.cells.every(([, y]) => y >= VISIBLE_H)) return -Infinity;
        const res = lockPlacement(rows, placement.cells);
        let landing = 0;
        for (const [, y] of placement.cells) landing += y + 1;
        landing /= placement.cells.length;
        const board = rowsToBoard(res.rows);
        let score;
        if (engine === 'cem') {
            score = evaluateCemBoard(extractCemFeatures(board, landing, res.cleared, res.pieceMinosCleared));
        } else if (engine === 'dqn_v1') {
            score = mlpForwardDqnV1(calculateDqnV1Features(board, res.cleared));
        } else {
            score = mlpForwardDqnV2(calculateDqnV2Features(board, res.cleared, level));
        }
        // minos left in the hidden rows are one step away from topping out
        for (let y = VISIBLE_H; y < MATRIX_H; y++) if (res.rows[y]) score -= HIDDEN_ROW_PENALTY;
        return score;
    }

    function planBestPlacement(engine, rows, piece, start, gravity20G, level) {
        let best = null;
        for (const placement of searchReachablePlacements(rows, piece, start, gravity20G)) {
            const score = scorePlacement(engine, rows, placement, level);
            if (best === null || score > best.score) best = Object.assign({ score }, placement);
        }
        return best;
    }

