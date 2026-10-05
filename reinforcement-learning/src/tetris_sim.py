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
        to = (rot + d) % 4
        if p == "O":
            return (to, bx, by)
        table = KICKS_I if p == "I" else KICKS_JLSTZ
        for dx, dy in table[(rot, to)]:
            if self.fits(rows, p, to, bx + dx, by + dy):
                return (to, bx + dx, by + dy)
        return None

    def spawn_state(self, p):
        return (0, 4 if p == "O" else 3, self.spawn_by)

    def reachable_placements(self, rows, p, start, gravity_20g):
        """
        Distinct final placements reachable from `start` with left/right/SRS-rotate, then hard drop.
        At 20G the piece settles after every action. Returns {cells_key: (rot, bx, final_by)}.
        """
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
                        nexts.append(t)
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
                out[cells] = (rot, bx, fy)
        return out

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


HIDDEN_ROW_PENALTY = 1000.0


def score_placement(matrix, rows, cells, weights):
    """Policy value of locking `cells`, plus the resulting board. None for a lock out."""
    vis = matrix.visible
    if all(y >= vis for _, y in cells):
        return None
    nr, cleared, eroded = matrix.lock(rows, cells)
    landing = sum(y + 1 for _, y in cells) / len(cells)
    f = matrix.features(nr, landing, cleared, eroded)
    value = sum(w * v for w, v in zip(weights, f))
    for y in range(vis, matrix.height):
        if nr[y]:
            value -= HIDDEN_ROW_PENALTY  # minos left in the hidden rows: one step from topping out
    return value, nr, cleared


def level_for(start_level, lines):
    return start_level + lines // LINES_PER_LEVEL


def play_game(weights, seed, max_pieces, start_level=FIRST_20G_LEVEL, visible_height=20, use_hold=True):
    """
    Play one game with a linear policy over the 8 features, like the browser bot does.
    Returns (lines_cleared, pieces_placed, topped_out).
    """
    rng = random.Random(seed)
    matrix = Matrix(visible_height)
    rows = [0] * matrix.height
    bag = []
    queue = []

    def refill():
        while len(queue) < 2:
            if not bag:
                bag.extend(rng.sample(PIECE_NAMES, len(PIECE_NAMES)))
            queue.append(bag.pop())

    refill()
    cur = queue.pop(0)
    hold = None
    lines = 0
    for pieces in range(max_pieces):
        refill()
        if not matrix.fits(rows, cur, *matrix.spawn_state(cur)):
            return lines, pieces, True  # block out
        g20 = level_for(start_level, lines) >= FIRST_20G_LEVEL

        # (piece to place, uses hold)
        options = [(cur, False)]
        if use_hold:
            swap_in = hold if hold is not None else queue[0]
            if matrix.fits(rows, swap_in, *matrix.spawn_state(swap_in)):
                options.append((swap_in, True))

        best = None
        for p, is_hold in options:
            start = matrix.spawn_state(p)
            for cells in matrix.reachable_placements(rows, p, start, g20):
                res = score_placement(matrix, rows, cells, weights)
                if res is not None and (best is None or res[0] > best[0]):
                    best = (res[0], res[1], res[2], is_hold)
        if best is None:
            return lines, pieces, True  # only lock outs left
        _, rows, cleared, is_hold = best
        lines += cleared

        if is_hold:
            if hold is None:
                hold = cur
                queue.pop(0)  # the next piece was swapped in and placed
            else:
                hold = cur
        cur = queue.pop(0)
    return lines, max_pieces, False
