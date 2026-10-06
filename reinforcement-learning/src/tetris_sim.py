"""
Tetris simulator matching the play.tetris.com (BPS engine) rules the browser bot actually faces.

Mirrors the placement search in userscript/src/placement_search.js:
- 10 x (visible + 4 hidden) matrix, y = 0 is the bottom row
- SRS rotation with wall kicks; pieces spawn in the top two visible rows (box top row = visible - 1)
- At 20G (fall speed 0 ms, level 20+) the piece drops to rest on spawn and after every successful
  move/rotation, so only placements reachable by sliding along the stack (plus kicks) are legal
- Below 20G the bot moves the piece at spawn height and hard drops, so any placement reachable
  at spawn height is legal
- Hold once per piece, block out when the spawn position is occupied, lock out when a piece locks
  entirely above the visible rows
- T-spins as the engine detects them (getTSpinTypeForLivePieceCurrentTransform): the last action was a
  rotation the piece didn't fall after (a hard drop keeps it), and 3 of the 4 corners around the T's
  center are filled

Boards are lists of row bitmasks (bit x = column x) for speed.
"""

import random

WIDTH = 10
HIDDEN_ROWS = 4
FULL_ROW = (1 << WIDTH) - 1
LINES_PER_LEVEL = 10
FIRST_20G_LEVEL = 20

PIECE_NAMES = ["I", "J", "L", "O", "S", "T", "Z"]

# SRS spawn orientations, rows listed top -> bottom inside the rotation box
SRS_BOXES = {
    "I": ["....", "XXXX", "....", "...."],
    "J": ["X..", "XXX", "..."],
    "L": ["..X", "XXX", "..."],
    "O": ["XX", "XX"],
    "S": [".XX", "XX.", "..."],
    "T": [".X.", "XXX", "..."],
    "Z": ["XX.", ".XX", "..."],
}


def _build_cells():
    cells = {}
    for p, box in SRS_BOXES.items():
        n = len(box)
        cur = [(r, c) for r in range(n) for c in range(n) if box[r][c] == "X"]
        rots = [cur]
        for _ in range(3):
            if p != "O":
                cur = [(c, n - 1 - r) for (r, c) in cur]  # clockwise
            rots.append(cur)
        cells[p] = rots
    return cells


# SRS_CELLS[piece][rot] = [(r, c), ...]: r = rows below the box top, c = columns right of the box left
SRS_CELLS = _build_cells()

# Kick offsets (dx, dy) with +y = up
KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}

BX_OFFSET = 4  # box left column ranges over [-BX_OFFSET, WIDTH)

# T-spin types, as the engine's score component tells them apart
TSPIN_NONE, TSPIN_MINI, TSPIN_FULL = 0, 1, 2
NO_TSPIN = frozenset((TSPIN_NONE,))


def _build_row_masks():
    """MASKS[p][rot][bx + BX_OFFSET] = [(r, rowmask), ...] or None if a cell leaves the walls."""
    masks = {}
    for p in PIECE_NAMES:
        masks[p] = []
        for rot in range(4):
            per_bx = []
            for bx in range(-BX_OFFSET, WIDTH):
                by_row = {}
                ok = True
                for r, c in SRS_CELLS[p][rot]:
                    x = bx + c
                    if x < 0 or x >= WIDTH:
                        ok = False
                        break
                    by_row[r] = by_row.get(r, 0) | (1 << x)
                per_bx.append(sorted(by_row.items()) if ok else None)
            masks[p].append(per_bx)
    return masks


MASKS = _build_row_masks()


def popcount(x):
    return x.bit_count()


class Matrix:
    """Board geometry; the default matches play.tetris.com (20 visible rows)."""

    def __init__(self, visible_height=20):
        self.visible = visible_height
        self.height = visible_height + HIDDEN_ROWS
        self.spawn_by = visible_height - 1

    def fits(self, rows, p, rot, bx, by):
        if bx < -BX_OFFSET or bx >= WIDTH:
            return False
        m = MASKS[p][rot][bx + BX_OFFSET]
        if m is None:
            return False
        h = self.height
        for r, mask in m:
            y = by - r
            if y < 0 or y >= h or rows[y] & mask:
                return False
        return True

    def drop(self, rows, p, rot, bx, by):
        while self.fits(rows, p, rot, bx, by - 1):
            by -= 1
        return by

    def rotate(self, rows, p, rot, bx, by, d):
        """(rot, bx, by, kick) after an SRS rotation, kick = the kick test used (1-5); None if none fits."""
        to = (rot + d) % 4
        if p == "O":
            return (to, bx, by, 1)
        table = KICKS_I if p == "I" else KICKS_JLSTZ
        for kick, (dx, dy) in enumerate(table[(rot, to)], 1):
            if self.fits(rows, p, to, bx + dx, by + dy):
                return (to, bx + dx, by + dy, kick)
        return None

    def spawn_state(self, p):
        return (0, 4 if p == "O" else 3, self.spawn_by)

    def reachable_placements(self, rows, p, start, gravity_20g):
        """
        Distinct final placements reachable from `start` with left/right/SRS-rotate, then hard drop.
        At 20G the piece settles after every action. Returns {cells: (rot, bx, final_by, hard_drop_rows, tspins)},
        where tspins is the set of T-spin types the placement can be reached with ({TSPIN_NONE} for other pieces).
        """
        if p == "T":
            return self._reachable_t(rows, start, gravity_20g)
        rot, bx, by = start
        if gravity_20g:
            by = self.drop(rows, p, rot, bx, by)
        s0 = (rot, bx, by)
        seen = {s0}
        queue = [s0]
        rotates = p != "O"
        i = 0
        while i < len(queue):
            rot, bx, by = queue[i]
            i += 1
            nexts = []
            for dx in (-1, 1):
                if self.fits(rows, p, rot, bx + dx, by):
                    nexts.append((rot, bx + dx, by))
            if rotates:
                for d in (1, -1):
                    t = self.rotate(rows, p, rot, bx, by, d)
                    if t is not None:
                        nexts.append(t[:3])
            for s in nexts:
                if gravity_20g:
                    s = (s[0], s[1], self.drop(rows, p, s[0], s[1], s[2]))
                if s not in seen:
                    seen.add(s)
                    queue.append(s)
        out = {}
        for rot, bx, by in queue:
            fy = by if gravity_20g else self.drop(rows, p, rot, bx, by)
            cells = tuple(sorted((bx + c, fy - r) for r, c in SRS_CELLS[p][rot]))
            if cells not in out:
                out[cells] = (rot, bx, fy, by - fy, NO_TSPIN)
        return out

    def _reachable_t(self, rows, start, gravity_20g):
        """
        reachable_placements for the T, searching (rot, bx, by, point) states. point is the engine's
        mEndingRotationPointForPiece: the kick test of the last rotation, reset to 0 by a move or a fall
        (not by a hard drop); a no-kick rotation right after a kick-5 rotation keeps 5.
        """
        rot, bx, by = start
        if gravity_20g:
            by = self.drop(rows, "T", rot, bx, by)
        s0 = (rot, bx, by, 0)
        seen = {s0}
        queue = [s0]
        i = 0
        while i < len(queue):
            rot, bx, by, point = queue[i]
            i += 1
            nexts = []
            for dx in (-1, 1):
                if self.fits(rows, "T", rot, bx + dx, by):
                    nexts.append((rot, bx + dx, self.drop(rows, "T", rot, bx + dx, by) if gravity_20g else by, 0))
            for d in (1, -1):
                t = self.rotate(rows, "T", rot, bx, by, d)
                if t is None:
                    continue
                to, nx, ny, kick = t
                np_ = 5 if point == 5 and kick == 1 else kick
                if gravity_20g:
                    fy = self.drop(rows, "T", to, nx, ny)
                    if fy != ny:
                        ny, np_ = fy, 0
                nexts.append((to, nx, ny, np_))
            for s in nexts:
                if s not in seen:
                    seen.add(s)
                    queue.append(s)
        out = {}
        for rot, bx, by, point in queue:
            fy = by if gravity_20g else self.drop(rows, "T", rot, bx, by)
            cells = tuple(sorted((bx + c, fy - r) for r, c in SRS_CELLS["T"][rot]))
            t = self.tspin_type(rows, rot, bx, fy, point)
            entry = out.get(cells)
            if entry is None:
                out[cells] = (rot, bx, fy, by - fy, frozenset((t,)))
            elif t not in entry[4]:
                out[cells] = entry[:4] + (entry[4] | {t},)
        return out

    def tspin_type(self, rows, rot, bx, by, point):
        """
        The engine's T-spin rule for a T locking at (rot, bx, by) with rotation point `point`. Corners
        around the center: both front corners (the side the T points to) and a back one is a T-spin; both
        back corners and a front one is a mini (a T-spin if the last rotation used kick 5, nothing if
        kick 4). Against the floor or a wall only a mini is possible.
        """
        if point == 0:
            return TSPIN_NONE
        x, y = bx + 1, by - 1
        filled = lambda cx, cy: cy < self.height and rows[cy] >> cx & 1
        if y == 0:
            mini = rot == 0 and (filled(x + 1, y + 1) or filled(x - 1, y + 1))
        elif x == 0:
            mini = rot == 1 and (filled(x + 1, y + 1) or filled(x + 1, y - 1))
        elif x == WIDTH - 1:
            mini = rot == 3 and (filled(x - 1, y + 1) or filled(x - 1, y - 1))
        else:
            tl, tr = filled(x - 1, y + 1), filled(x + 1, y + 1)
            bl, br = filled(x - 1, y - 1), filled(x + 1, y - 1)
            # (front corners, back corners) for up, right, down, left
            front, back = (((tl, tr), (bl, br)), ((tr, br), (tl, bl)), ((bl, br), (tl, tr)), ((tl, bl), (tr, br)))[rot]
            if front[0] and front[1] and (back[0] or back[1]):
                return TSPIN_FULL
            mini = back[0] and back[1] and (front[0] or front[1])
        if not mini or point == 4:
            return TSPIN_NONE
        return TSPIN_FULL if point == 5 else TSPIN_MINI

    def lock(self, rows, cells):
        """Returns (new_rows, lines_cleared, piece_minos_cleared)."""
        nr = list(rows)
        for x, y in cells:
            nr[y] |= 1 << x
        full = [y for y in range(self.height) if nr[y] == FULL_ROW]
        if not full:
            return nr, 0, 0
        eroded = sum(1 for _, y in cells if nr[y] == FULL_ROW)
        kept = [v for v in nr if v != FULL_ROW]
        return kept + [0] * len(full), len(full), eroded

    def features(self, rows, landing_height, num_cleared, piece_minos_cleared):
        """
        The 8 Thiery & Scherrer features over the visible rows, identical to extractCemFeatures in the
        userscript: landing height, eroded piece cells, row transitions, column transitions, holes,
        cumulative wells, hole depth, rows with holes.
        """
        row_trans = 0
        col_trans = 0
        holes = 0
        hole_depth = 0
        rows_with_holes = 0
        covered = 0
        prev = 0  # empty ceiling above the top visible row
        # per-column count of filled cells seen so far (from the top), as 5 bit-planes
        c0 = c1 = c2 = c3 = c4 = 0
        for y in range(self.visible - 1, -1, -1):
            row = rows[y]
            padded = (row << 1) | 1 | (1 << (WIDTH + 1))
            row_trans += popcount((padded ^ (padded >> 1)) & ((1 << (WIDTH + 1)) - 1))
            col_trans += popcount(prev ^ row)
            prev = row
            hole = covered & ~row & FULL_ROW
            if hole:
                holes += popcount(hole)
                rows_with_holes += 1
                hole_depth += (popcount(c0 & hole) + 2 * popcount(c1 & hole) + 4 * popcount(c2 & hole)
                               + 8 * popcount(c3 & hole) + 16 * popcount(c4 & hole))
            if row:
                # bit-parallel increment of the per-column counters where this row is filled
                carry = row
                c0, carry = c0 ^ carry, c0 & carry
                c1, carry = c1 ^ carry, c1 & carry
                c2, carry = c2 ^ carry, c2 & carry
                c3, carry = c3 ^ carry, c3 & carry
                c4 ^= carry
            covered |= row
        col_trans += popcount(prev ^ FULL_ROW)  # solid floor below the bottom row

        # cumulative wells: each vertical run of L well cells contributes 1 + 2 + ... + L
        wells = 0
        d0 = d1 = d2 = d3 = d4 = 0  # per-column current run depth, as bit-planes
        for y in range(self.visible):
            row = rows[y]
            well = ~row & ((row << 1) | 1) & ((row >> 1) | (1 << (WIDTH - 1))) & FULL_ROW
            if not well:
                d0 = d1 = d2 = d3 = d4 = 0
                continue
            carry = FULL_ROW
            d0, carry = d0 ^ carry, d0 & carry
            d1, carry = d1 ^ carry, d1 & carry
            d2, carry = d2 ^ carry, d2 & carry
            d3, carry = d3 ^ carry, d3 & carry
            d4 ^= carry
            d0 &= well
            d1 &= well
            d2 &= well
            d3 &= well
            d4 &= well
            wells += popcount(d0) + 2 * popcount(d1) + 4 * popcount(d2) + 8 * popcount(d3) + 16 * popcount(d4)

        return (landing_height, num_cleared * piece_minos_cleared, row_trans, col_trans,
                holes, wells, hole_depth, rows_with_holes)

    def column_heights(self, rows):
        heights = [0] * WIDTH
        covered = 0
        for y in range(self.visible - 1, -1, -1):
            new = rows[y] & ~covered
            if new:
                covered |= new
                for c in range(WIDTH):
                    if new >> c & 1:
                        heights[c] = y + 1
        return heights

    def ready_lines(self, rows):
        """
        Lines a vertical I piece would clear in the lowest column right now (0-4): the rows directly
        above that column's top that are full except for it. Only a strictly lowest column can qualify.
        """
        heights = self.column_heights(rows)
        h = min(heights)
        if heights.count(h) != 1:
            return 0
        mask = FULL_ROW & ~(1 << heights.index(h))
        n = 0
        while n < 4 and h + n < self.visible and rows[h + n] == mask:
            n += 1
        return n

    def dqn_features(self, rows):
        """(holes, bumpiness, total height, max height) over the visible rows, as calculateDqnV*Features."""
        heights = self.column_heights(rows)
        holes = 0
        covered = 0
        for y in range(self.visible - 1, -1, -1):
            holes += popcount(covered & ~rows[y] & FULL_ROW)
            covered |= rows[y]
        bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
        return holes, bumpiness, sum(heights), max(heights)


HIDDEN_ROW_PENALTY = 1000.0

# ---------------------------------------------------------------------------
# Scoring: play.tetris.com's rules, read from the game's score component
# ---------------------------------------------------------------------------
LINE_CLEAR_POINTS = (0, 100, 300, 500, 800)
# Base points by T-spin type and lines; a mini with 2+ lines scores nothing
TSPIN_CLEAR_POINTS = {TSPIN_MINI: (100, 200, 0, 0), TSPIN_FULL: (400, 800, 1200, 1600)}
PERFECT_CLEAR_POINTS = (0, 800, 1200, 1800, 2000)
B2B_TETRIS_PERFECT_CLEAR_POINTS = 3200
COMBO_POINTS = 50
HARD_DROP_POINTS_PER_ROW = 2
MARATHON_LINES = 300


def clear_points(lines, tspin=0):
    """Base points of a clear (before back-to-back, combo and level)."""
    return TSPIN_CLEAR_POINTS[tspin][lines] if tspin else LINE_CLEAR_POINTS[lines]


def is_difficult(lines, tspin=0):
    """Clears that continue (and get x1.5 from) a back-to-back chain: Tetrises, T-spins and mini singles."""
    if tspin == TSPIN_FULL:
        return lines > 0
    if tspin == TSPIN_MINI:
        return lines == 1
    return lines == 4


def best_tspin(tspins, lines):
    """The T-spin type to reach a placement with, given the types it can be reached with."""
    if TSPIN_FULL in tspins:
        return TSPIN_FULL
    if TSPIN_MINI in tspins and (lines <= 1 or TSPIN_NONE not in tspins):
        return TSPIN_MINI
    return TSPIN_NONE


class Scorer:
    """Score, combo and back-to-back state."""

    def __init__(self):
        self.score = 0
        self.combo = 0  # line-clearing pieces in a row before this one
        self.back_to_back = False
        self.clears = [0, 0, 0, 0, 0]  # count by lines cleared
        self.tspins = {TSPIN_MINI: [0] * 4, TSPIN_FULL: [0] * 4}  # count by type and lines

    def add(self, lines, level, hard_drop_rows=0, perfect_clear=False, tspin=TSPIN_NONE):
        self.score += HARD_DROP_POINTS_PER_ROW * hard_drop_rows
        if tspin:
            self.tspins[tspin][lines] += 1
        points = clear_points(lines, tspin)
        difficult = is_difficult(lines, tspin)
        b2b = self.back_to_back and difficult
        if b2b:
            points = points * 3 // 2
        if difficult:
            self.back_to_back = True
        elif lines and tspin == TSPIN_NONE:
            self.back_to_back = False  # a single/double/triple breaks the chain; a mini double doesn't
        if lines == 0:
            self.combo = 0
        else:
            self.clears[lines] += 1
            points += COMBO_POINTS * self.combo
            self.combo += 1
            if perfect_clear:
                points += B2B_TETRIS_PERFECT_CLEAR_POINTS if b2b and lines == 4 else PERFECT_CLEAR_POINTS[lines]
        self.score += points * level


# ---------------------------------------------------------------------------
# Policies: score after-states, like the userscript's engines
# ---------------------------------------------------------------------------
class AfterState:
    __slots__ = ("rows", "cleared", "eroded", "landing", "cells", "is_hold", "hard_drop_rows", "tspin")

    def __init__(self, rows, cleared, eroded, landing, cells, is_hold, hard_drop_rows, tspin=0):
        self.rows = rows
        self.cleared = cleared
        self.eroded = eroded
        self.landing = landing
        self.cells = cells
        self.is_hold = is_hold
        self.hard_drop_rows = hard_drop_rows
        self.tspin = tspin


# Value of the Tetris feature for a placement that clears 4 lines (a Tetris's base points / 100)
TETRIS_FEATURE = LINE_CLEAR_POINTS[4] // 100


def cem_features(matrix, rows, landing_height, num_cleared, piece_minos_cleared):
    """All 11 CEM features of an after-state, as extractCemFeatures in the userscript returns them."""
    return matrix.features(rows, landing_height, num_cleared, piece_minos_cleared) + (
        LINE_CLEAR_POINTS[num_cleared] / 100, matrix.ready_lines(rows), TETRIS_FEATURE if num_cleared == 4 else 0)


class LinearPolicy:
    """
    CEM policy: w . f over the first 8, 10 or 11 of cem_features(): the 8 Thiery & Scherrer features, the
    base points of the clear / 100 (0, 1, 3, 5, 8), ready_lines, and 8 if the placement clears a Tetris.
    """

    def __init__(self, weights):
        self.weights = [float(w) for w in weights]
        if len(self.weights) not in (8, 10, 11):
            raise ValueError("expected 8, 10 or 11 weights")

    def values(self, matrix, afters, level):
        w = self.weights
        n = len(w)
        out = []
        for a in afters:
            f = matrix.features(a.rows, a.landing, a.cleared, a.eroded)
            v = sum(wi * fi for wi, fi in zip(w, f))
            if n > 8:
                v += w[8] * LINE_CLEAR_POINTS[a.cleared] / 100 + w[9] * matrix.ready_lines(a.rows)
                if n == 11 and a.cleared == 4:
                    v += w[10] * TETRIS_FEATURE
            out.append(v)
        return out


class MLPPolicy:
    """DQN after-state value network exported by export_weights.py (version 1: 4 inputs, 2: 6 inputs)."""

    def __init__(self, weights, version):
        import numpy as np
        self.np = np
        self.version = version
        self.layers = [(np.array(weights[f"net.{i}.weight"], dtype=np.float64),
                        np.array(weights[f"net.{i}.bias"], dtype=np.float64)) for i in (0, 2, 4)]

    def values(self, matrix, afters, level):
        np = self.np
        feats = []
        for a in afters:
            holes, bump, total, top = matrix.dqn_features(a.rows)
            f = [a.cleared, holes, bump, total]
            if self.version == 2:
                f += [top, max(0, min(30, level) - 1) / 29.0]
            feats.append(f)
        x = np.array(feats, dtype=np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < 2:
                x = np.maximum(x, 0.0)
        return x[:, 0].tolist()


def as_policy(policy):
    return LinearPolicy(policy) if isinstance(policy, (list, tuple)) else policy


def level_for(start_level, lines):
    return start_level + lines // LINES_PER_LEVEL


class GameResult:
    __slots__ = ("lines", "pieces", "topped_out", "score", "clears", "tspins")

    def __init__(self, lines, pieces, topped_out, score, clears, tspins):
        self.lines = lines
        self.pieces = pieces
        self.topped_out = topped_out
        self.score = score
        self.clears = clears
        self.tspins = tspins  # (minis by lines, T-spins by lines)

    def __repr__(self):
        return (f"GameResult(lines={self.lines}, pieces={self.pieces}, topped_out={self.topped_out}, "
                f"score={self.score}, clears={self.clears}, tspins={self.tspins})")


def play_game(policy, seed, max_pieces, start_level=FIRST_20G_LEVEL, visible_height=20, use_hold=True,
              max_lines=None):
    """
    Play one game like the browser bot does: every reachable placement of the current piece and of the
    piece hold would bring in is scored by `policy` (a LinearPolicy, MLPPolicy or a list of weights).
    With max_lines=MARATHON_LINES and start_level=1 this is play.tetris.com's Marathon.
    """
    policy = as_policy(policy)
    rng = random.Random(seed)
    matrix = Matrix(visible_height)
    vis = matrix.visible
    rows = [0] * matrix.height
    scorer = Scorer()
    bag = []
    queue = []

    def refill():
        while len(queue) < 2:
            if not bag:
                bag.extend(rng.sample(PIECE_NAMES, len(PIECE_NAMES)))
            queue.append(bag.pop())

    def result(pieces, topped_out):
        return GameResult(lines, pieces, topped_out, scorer.score, tuple(scorer.clears),
                          (tuple(scorer.tspins[TSPIN_MINI]), tuple(scorer.tspins[TSPIN_FULL])))

    refill()
    cur = queue.pop(0)
    hold = None
    lines = 0
    for pieces in range(max_pieces):
        refill()
        if not matrix.fits(rows, cur, *matrix.spawn_state(cur)):
            return result(pieces, True)  # block out
        level = level_for(start_level, lines)
        g20 = level >= FIRST_20G_LEVEL

        # (piece to place, uses hold)
        options = [(cur, False)]
        if use_hold:
            swap_in = hold if hold is not None else queue[0]
            if matrix.fits(rows, swap_in, *matrix.spawn_state(swap_in)):
                options.append((swap_in, True))

        afters = []
        for p, is_hold in options:
            for cells, (_, _, _, drop, tspins) in matrix.reachable_placements(rows, p, matrix.spawn_state(p), g20).items():
                if all(y >= vis for _, y in cells):
                    continue  # lock out
                nr, cleared, eroded = matrix.lock(rows, cells)
                landing = sum(y + 1 for _, y in cells) / len(cells)
                afters.append(AfterState(nr, cleared, eroded, landing, cells, is_hold, drop, best_tspin(tspins, cleared)))
        if not afters:
            return result(pieces, True)

        values = policy.values(matrix, afters, level)
        best, best_value = None, None
        for a, v in zip(afters, values):
            if any(a.rows[y] for y in range(vis, matrix.height)):
                v -= HIDDEN_ROW_PENALTY  # minos left in the hidden rows: one step from topping out
            if best is None or v > best_value:
                best, best_value = a, v

        rows = best.rows
        scorer.add(best.cleared, level, best.hard_drop_rows, perfect_clear=not any(rows), tspin=best.tspin)
        lines += best.cleared

        if best.is_hold:
            if hold is None:
                hold = cur
                queue.pop(0)  # the next piece was swapped in and placed
            else:
                hold = cur
        cur = queue.pop(0)
        if max_lines is not None and lines >= max_lines:
            return result(pieces + 1, False)
    return result(max_pieces, False)
