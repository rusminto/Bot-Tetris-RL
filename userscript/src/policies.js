    // ==========================================
    // 1. AI POLICY WEIGHTS (CEM-RL & DQN V1/V2)
    // ==========================================
    const CEM_WEIGHTS = __CEM_WEIGHTS__;
    const DQN_V1_WEIGHTS = __DQN_V1_WEIGHTS__;
    const DQN_V2_WEIGHTS = __DQN_V2_WEIGHTS__;

    let activeEngine = 'cem'; // 'cem' (Policy Search), 'dqn_v2' (DQN v2 6-feat), or 'dqn_v1' (DQN v1 4-feat)

    // ==========================================
    // 2. TETRIS PIECE DEFINITIONS
    // ==========================================
    const PIECES = {
        'I': [
            [[1, 1, 1, 1]],
            [[1], [1], [1], [1]]
        ],
        'O': [
            [[1, 1], [1, 1]]
        ],
        'T': [
            [[0, 1, 0], [1, 1, 1]],
            [[1, 0], [1, 1], [1, 0]],
            [[1, 1, 1], [0, 1, 0]],
            [[0, 1], [1, 1], [0, 1]]
        ],
        'S': [
            [[0, 1, 1], [1, 1, 0]],
            [[1, 0], [1, 1], [0, 1]]
        ],
        'Z': [
            [[1, 1, 0], [0, 1, 1]],
            [[0, 1], [1, 1], [1, 0]]
        ],
        'J': [
            [[1, 0, 0], [1, 1, 1]],
            [[1, 1], [1, 0], [1, 0]],
            [[1, 1, 1], [0, 0, 1]],
            [[0, 1], [0, 1], [1, 1]]
        ],
        'L': [
            [[0, 0, 1], [1, 1, 1]],
            [[1, 0], [1, 0], [1, 1]],
            [[1, 1, 1], [1, 0, 0]],
            [[1, 1], [0, 1], [0, 1]]
        ]
    };

    function isValidPosition(shape, x, y, board) {
        const shapeH = shape.length;
        const shapeW = shape[0].length;
        for (let r = 0; r < shapeH; r++) {
            for (let c = 0; c < shapeW; c++) {
                if (shape[r][c]) {
                    const by = y + r;
                    const bx = x + c;
                    if (bx < 0 || bx >= 10) return false;
                    if (by >= 20) return false;
                    if (by >= 0 && board[by][bx] > 0) return false;
                }
            }
        }
        return true;
    }

    // ==========================================
    // 3. CEM-RL EVALUATION ENGINE (8 FEATURES)
    // ==========================================
    function extractCemFeatures(board, landingHeight, numCleared, pieceMinosCleared) {
        const eroded = numCleared * pieceMinosCleared;

        // 1. Row transitions
        let rowTrans = 0;
        for (let r = 0; r < 20; r++) {
            let prev = 1; // Left boundary is solid
            for (let c = 0; c < 10; c++) {
                const cur = board[r][c] > 0 ? 1 : 0;
                if (cur !== prev) rowTrans++;
                prev = cur;
            }
            if (prev !== 1) rowTrans++; // Right boundary is solid
        }

        // 2. Col transitions
        let colTrans = 0;
        for (let c = 0; c < 10; c++) {
            let prev = 0; // Ceiling is empty
            for (let r = 0; r < 20; r++) {
                const cur = board[r][c] > 0 ? 1 : 0;
                if (cur !== prev) colTrans++;
                prev = cur;
            }
            if (prev !== 1) colTrans++; // Floor is solid
        }

        // 3. Holes, hole depth, rows with holes
        let holes = 0;
        let holeDepth = 0;
        const rowsWithHole = new Uint8Array(20);

        for (let c = 0; c < 10; c++) {
            let topRow = -1;
            for (let r = 0; r < 20; r++) {
                if (board[r][c] > 0) {
                    topRow = r;
                    break;
                }
            }
            if (topRow !== -1) {
                for (let r = topRow + 1; r < 20; r++) {
                    if (board[r][c] === 0) {
                        holes++;
                        let occAbove = 0;
                        for (let ar = 0; ar < r; ar++) {
                            if (board[ar][c] > 0) occAbove++;
                        }
                        holeDepth += occAbove;
                        rowsWithHole[r] = 1;
                    }
                }
            }
        }

        let rowsWithHolesCount = 0;
        for (let r = 0; r < 20; r++) {
            if (rowsWithHole[r]) rowsWithHolesCount++;
        }

        // 4. Cumulative wells
        let cumulativeWells = 0;
        for (let c = 0; c < 10; c++) {
            let depth = 0;
            for (let r = 19; r >= 0; r--) {
                if (board[r][c] === 0) {
                    const leftFull = (c === 0) || (board[r][c - 1] > 0);
                    const rightFull = (c === 9) || (board[r][c + 1] > 0);
                    if (leftFull && rightFull) {
                        depth++;
                        cumulativeWells += depth;
                    } else {
                        depth = 0;
                    }
                } else {
                    depth = 0;
                }
            }
        }

        return [
            landingHeight,
            eroded,
            rowTrans,
            colTrans,
            holes,
            cumulativeWells,
            holeDepth,
            rowsWithHolesCount
        ];
    }

    function selectBestCemPlacement(board, pieceName, holdPieceName, canHold) {
        const candidates = [{ piece: pieceName, isHold: false }];
        if (canHold && holdPieceName && PIECES[holdPieceName]) {
            candidates.push({ piece: holdPieceName, isHold: true });
        }

        let bestVal = -Infinity;
        let bestResult = null;

        for (const cand of candidates) {
            const orientations = PIECES[cand.piece];
            if (!orientations) continue;

            for (let rotIdx = 0; rotIdx < orientations.length; rotIdx++) {
                const shape = orientations[rotIdx];
                const shapeH = shape.length;
                const shapeW = shape[0].length;

                for (let x = 0; x <= 10 - shapeW; x++) {
                    let y = 0;
                    while (isValidPosition(shape, x, y + 1, board)) {
                        y++;
                    }
                    if (!isValidPosition(shape, x, y, board)) continue;

                    // Landing height (average row index measured from bottom)
                    let sumRow = 0;
                    let countMinos = 0;
                    for (let r = 0; r < shapeH; r++) {
                        for (let c = 0; c < shapeW; c++) {
                            if (shape[r][c]) {
                                sumRow += (20 - (y + r));
                                countMinos++;
                            }
                        }
                    }
                    const landingHeight = countMinos > 0 ? (sumRow / countMinos) : 0;

                    // Next board
                    const nextBoard = board.map(row => new Uint8Array(row));
                    for (let r = 0; r < shapeH; r++) {
                        for (let c = 0; c < shapeW; c++) {
                            if (shape[r][c]) {
                                nextBoard[y + r][x + c] = 1;
                            }
                        }
                    }

                    // Check which rows are cleared
                    let numCleared = 0;
                    let pieceMinosCleared = 0;
                    const nonFull = [];
                    for (let r = 0; r < 20; r++) {
                        let full = true;
                        for (let c = 0; c < 10; c++) {
                            if (nextBoard[r][c] === 0) {
                                full = false;
                                break;
                            }
                        }
                        if (full) {
                            numCleared++;
                            if (r >= y && r < y + shapeH) {
                                for (let c = 0; c < shapeW; c++) {
                                    if (shape[r - y][c]) pieceMinosCleared++;
                                }
                            }
                        } else {
                            nonFull.push(nextBoard[r]);
                        }
                    }

                    let clearedBoard = nextBoard;
                    if (numCleared > 0) {
                        clearedBoard = [];
                        for (let i = 0; i < numCleared; i++) {
                            clearedBoard.push(new Uint8Array(10));
                        }
                        for (let i = 0; i < nonFull.length; i++) {
                            clearedBoard.push(new Uint8Array(nonFull[i]));
                        }
                    }

                    const feat = extractCemFeatures(clearedBoard, landingHeight, numCleared, pieceMinosCleared);
                    const score = evaluateCemBoard(feat);

                    if (score > bestVal) {
                        bestVal = score;
                        bestResult = {
                            isHold: cand.isHold,
                            pieceName: cand.piece,
                            rotIdx: rotIdx,
                            x: x,
                            y: y,
                            score: score
                        };
                    }
                }
            }
        }

        return bestResult;
    }

    function evaluateCemBoard(feat) {
        let score = 0;
        for (let k = 0; k < 8; k++) {
            score += CEM_WEIGHTS[k] * feat[k];
        }
        return score;
    }

    // ==========================================
    // 4. DQN EVALUATION ENGINE (V1 & V2 MLP)
    // ==========================================
    function mlpForwardDqnV1(features) {
        // DQN V1: 4 input features [lines, holes, bumpiness, totalHeight]
        const w0 = DQN_V1_WEIGHTS["net.0.weight"];
        const b0 = DQN_V1_WEIGHTS["net.0.bias"];
        const h1 = new Float32Array(64);
        for (let i = 0; i < 64; i++) {
            let s = b0[i];
            const row = w0[i];
            s += row[0] * features[0] + row[1] * features[1] + row[2] * features[2] + row[3] * features[3];
            h1[i] = s > 0 ? s : 0;
        }

        const w2 = DQN_V1_WEIGHTS["net.2.weight"];
        const b2 = DQN_V1_WEIGHTS["net.2.bias"];
        const h2 = new Float32Array(64);
        for (let i = 0; i < 64; i++) {
            let s = b2[i];
            const row = w2[i];
            for (let j = 0; j < 64; j++) {
                s += row[j] * h1[j];
            }
            h2[i] = s > 0 ? s : 0;
        }

        const w4 = DQN_V1_WEIGHTS["net.4.weight"][0];
        let out = DQN_V1_WEIGHTS["net.4.bias"][0];
        for (let j = 0; j < 64; j++) {
            out += w4[j] * h2[j];
        }
        return out;
    }

    function mlpForwardDqnV2(features) {
        // DQN V2: 6 input features [lines, holes, bumpiness, totalHeight, maxHeight, normalizedLevel]
        const w0 = DQN_V2_WEIGHTS["net.0.weight"];
        const b0 = DQN_V2_WEIGHTS["net.0.bias"];
        const h1 = new Float32Array(64);
        for (let i = 0; i < 64; i++) {
            let s = b0[i];
            const row = w0[i];
            s += row[0] * features[0] + row[1] * features[1] + row[2] * features[2] +
                 row[3] * features[3] + row[4] * features[4] + row[5] * features[5];
            h1[i] = s > 0 ? s : 0;
        }

        const w2 = DQN_V2_WEIGHTS["net.2.weight"];
        const b2 = DQN_V2_WEIGHTS["net.2.bias"];
        const h2 = new Float32Array(64);
        for (let i = 0; i < 64; i++) {
            let s = b2[i];
            const row = w2[i];
            for (let j = 0; j < 64; j++) {
                s += row[j] * h1[j];
            }
            h2[i] = s > 0 ? s : 0;
        }

        const w4 = DQN_V2_WEIGHTS["net.4.weight"][0];
        let out = DQN_V2_WEIGHTS["net.4.bias"][0];
        for (let j = 0; j < 64; j++) {
            out += w4[j] * h2[j];
        }
        return out;
    }

    function calculateDqnV1Features(board, linesCleared) {
        const heights = new Int32Array(10);
        let holes = 0;

        for (let col = 0; col < 10; col++) {
            let topRow = -1;
            for (let r = 0; r < 20; r++) {
                if (board[r][col] > 0) {
                    topRow = r;
                    break;
                }
            }
            if (topRow !== -1) {
                heights[col] = 20 - topRow;
                for (let r = topRow + 1; r < 20; r++) {
                    if (board[r][col] === 0) holes++;
                }
            } else {
                heights[col] = 0;
            }
        }

        let bumpiness = 0;
        for (let col = 0; col < 9; col++) {
            bumpiness += Math.abs(heights[col] - heights[col + 1]);
        }

        let totalHeight = 0;
        for (let col = 0; col < 10; col++) {
            totalHeight += heights[col];
        }

        return [linesCleared, holes, bumpiness, totalHeight];
    }

    function calculateDqnV2Features(board, linesCleared, level) {
        const heights = new Int32Array(10);
        let holes = 0;

        for (let col = 0; col < 10; col++) {
            let topRow = -1;
            for (let r = 0; r < 20; r++) {
                if (board[r][col] > 0) {
                    topRow = r;
                    break;
                }
            }
            if (topRow !== -1) {
                heights[col] = 20 - topRow;
                for (let r = topRow + 1; r < 20; r++) {
                    if (board[r][col] === 0) holes++;
                }
            } else {
                heights[col] = 0;
            }
        }

        let bumpiness = 0;
        for (let col = 0; col < 9; col++) {
            bumpiness += Math.abs(heights[col] - heights[col + 1]);
        }

        let totalHeight = 0;
        let maxHeight = 0;
        for (let col = 0; col < 10; col++) {
            totalHeight += heights[col];
            if (heights[col] > maxHeight) maxHeight = heights[col];
        }

        const curLevel = (level !== undefined && level !== null) ? level : 1;
        const normalizedLevel = Math.max(0, Math.min(30, curLevel) - 1) / 29.0;

        return [linesCleared, holes, bumpiness, totalHeight, maxHeight, normalizedLevel];
    }

    function selectBestDqnPlacement(board, pieceName, level, isV1 = false) {
        const orientations = PIECES[pieceName];
        if (!orientations) return null;

        let bestScore = -Infinity;
        let bestPlacement = null;

        for (let rotIdx = 0; rotIdx < orientations.length; rotIdx++) {
            const shape = orientations[rotIdx];
            const shapeH = shape.length;
            const shapeW = shape[0].length;

            for (let x = 0; x <= 10 - shapeW; x++) {
                let y = 0;
                while (isValidPosition(shape, x, y + 1, board)) {
                    y++;
                }
                if (!isValidPosition(shape, x, y, board)) continue;

                const nextBoard = board.map(row => new Uint8Array(row));
                for (let r = 0; r < shapeH; r++) {
                    for (let c = 0; c < shapeW; c++) {
                        if (shape[r][c]) {
                            nextBoard[y + r][x + c] = 1;
                        }
                    }
                }

                let numCleared = 0;
                const nonFull = [];
                for (let r = 0; r < 20; r++) {
                    let full = true;
                    for (let c = 0; c < 10; c++) {
                        if (nextBoard[r][c] === 0) { full = false; break; }
                    }
                    if (full) numCleared++;
                    else nonFull.push(nextBoard[r]);
                }

                let clearedBoard = nextBoard;
                if (numCleared > 0) {
                    clearedBoard = [];
                    for (let i = 0; i < numCleared; i++) clearedBoard.push(new Uint8Array(10));
                    for (let i = 0; i < nonFull.length; i++) clearedBoard.push(new Uint8Array(nonFull[i]));
                }

                let score;
                if (isV1) {
                    const feat = calculateDqnV1Features(clearedBoard, numCleared);
                    score = mlpForwardDqnV1(feat);
                } else {
                    const feat = calculateDqnV2Features(clearedBoard, numCleared, level);
                    score = mlpForwardDqnV2(feat);
                }

                if (score > bestScore) {
                    bestScore = score;
                    bestPlacement = { rotIdx, x, y, score };
                }
            }
        }

        return bestPlacement ? { placement: bestPlacement, score: bestScore } : null;
    }

