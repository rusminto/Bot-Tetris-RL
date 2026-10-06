"""
Sanity checks for tetris_sim.py. Run: python3 src/test_tetris_sim.py
"""

import json
import os
import random

import numpy as np

from tetris_sim import (FULL_ROW, MARATHON_LINES, Matrix, MLPPolicy, PIECE_NAMES, SRS_CELLS, Scorer,
                        TSPIN_FULL, TSPIN_MINI, TSPIN_NONE, play_game)

RL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def reference_features(board, landing_height, num_cleared, piece_minos_cleared):
    """The original numpy implementation from cem_train.py (board is 20x10, row 0 = top)."""
    eroded = num_cleared * piece_minos_cleared
    pad_h = np.ones((20, 1), dtype=np.uint8)
    row_b = np.hstack([pad_h, board > 0, pad_h])
    row_trans = int(np.sum(row_b[:, :-1] != row_b[:, 1:]))
    col_b = np.vstack([np.zeros((1, 10), dtype=np.uint8), board > 0, np.ones((1, 10), dtype=np.uint8)])
    col_trans = int(np.sum(col_b[:-1, :] != col_b[1:, :]))
    holes = 0
    hole_depth = 0
    col_occ = board > 0
    row_has_hole = np.zeros(20, dtype=bool)
    for c in range(10):
        col_arr = col_occ[:, c]
        occ_idx = np.where(col_arr)[0]
        if len(occ_idx) > 0:
            top = occ_idx[0]
            under = col_arr[top + 1:]
            num_h = int(np.sum(~under))
            if num_h > 0:
                holes += num_h
                for r in range(top + 1, 20):
                    if not col_arr[r]:
                        hole_depth += np.sum(col_arr[:r])
                        row_has_hole[r] = True
    rows_with_holes = int(np.sum(row_has_hole))
    cumulative_wells = 0
    for c in range(10):
        depth = 0
        for r in range(19, -1, -1):
            if board[r, c] == 0:
                l_full = (c == 0) or (board[r, c - 1] > 0)
                r_full = (c == 9) or (board[r, c + 1] > 0)
                if l_full and r_full:
                    depth += 1
                    cumulative_wells += depth
                else:
                    depth = 0
            else:
                depth = 0
    return (landing_height, eroded, row_trans, col_trans, holes, cumulative_wells, hole_depth, rows_with_holes)


def rows_to_board(rows):
    board = np.zeros((20, 10), dtype=np.uint8)
    for y in range(20):
        for x in range(10):
            if rows[y] >> x & 1:
                board[19 - y, x] = 1
    return board


def test_features_match_reference():
    rng = random.Random(1)
    m = Matrix()
    for _ in range(3000):
        height = rng.randint(0, 20)
        density = rng.random()
        rows = [0] * m.height
        for y in range(height):
            rows[y] = sum(1 << x for x in range(10) if rng.random() < density)
        got = m.features(rows, 3.5, 2, 3)
        want = reference_features(rows_to_board(rows), 3.5, 2, 3)
        assert tuple(got) == tuple(float(v) if i == 0 else int(v) for i, v in enumerate(want)), (rows, got, want)


def test_empty_board_reachability():
    m = Matrix()
    rows = [0] * m.height
    # distinct placements on an empty board: I 7+10, O 9, T/J/L 8+9+8+9, S/Z 8+9
    expected = {"I": 17, "O": 9, "T": 34, "J": 34, "L": 34, "S": 17, "Z": 17}
    for p in PIECE_NAMES:
        for g20 in (False, True):
            got = len(m.reachable_placements(rows, p, m.spawn_state(p), g20))
            assert got == expected[p], (p, g20, got)


def test_20g_blocked_by_tower():
    """A tall column next to the spawn area blocks sliding past it at 20G, but not at spawn height."""
    m = Matrix()
    rows = [0] * m.height
    for y in range(12):
        rows[y] |= 1 << 2  # tower in column 2
    reach_slow = m.reachable_placements(rows, "O", m.spawn_state("O"), False)
    reach_20g = m.reachable_placements(rows, "O", m.spawn_state("O"), True)
    left_of_tower = lambda cells: all(x < 2 for x, _ in cells)
    assert any(left_of_tower(c) for c in reach_slow)
    assert not any(left_of_tower(c) for c in reach_20g)


def test_spawn_matches_engine():
    """Spawn minos observed in the live BPS engine (game coordinates, y = 0 bottom)."""
    observed = {
        "J": {(3, 18), (3, 19), (4, 18), (5, 18)},
        "I": {(3, 18), (4, 18), (5, 18), (6, 18)},
        "O": {(4, 18), (4, 19), (5, 18), (5, 19)},
        "S": {(3, 18), (4, 18), (4, 19), (5, 19)},
        "Z": {(3, 19), (4, 18), (4, 19), (5, 18)},
        "T": {(3, 18), (4, 18), (4, 19), (5, 18)},
        "L": {(3, 18), (4, 18), (5, 18), (5, 19)},
    }
    m = Matrix()
    for p, want in observed.items():
        rot, bx, by = m.spawn_state(p)
        got = {(bx + c, by - r) for r, c in SRS_CELLS[p][rot]}
        assert got == want, (p, got, want)


def test_games_run():
    weights = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]
    r = play_game(weights, seed=3, max_pieces=300)
    assert r.pieces == 300 and not r.topped_out and r.lines > 100, r


def test_scoring_rules():
    """play.tetris.com's rules: base x level, +50 x combo x level, back-to-back Tetris x1.5, 2/row hard drop."""
    s = Scorer()
    s.add(1, level=1)                 # single, starts a combo: 100
    s.add(2, level=1)                 # double + combo 1: 300 + 50
    assert s.score == 450
    s.add(0, level=1, hard_drop_rows=10)  # no clear: combo ends, 20 hard-drop points
    assert s.score == 470 and s.combo == 0
    s.add(4, level=2)                 # Tetris: 1600
    s.add(4, level=2)                 # back-to-back Tetris + combo 1: (1200 + 50) x 2
    assert s.score == 470 + 1600 + 2500
    s.add(0, level=2)
    s.add(1, level=2)                 # a single breaks the back-to-back chain
    s.add(0, level=2)
    s.add(4, level=2)
    assert s.score == 470 + 1600 + 2500 + 200 + 1600 and s.clears == [0, 2, 1, 0, 3]
    s2 = Scorer()
    s2.add(4, level=3, perfect_clear=True)  # Tetris perfect clear: (800 + 2000) x 3
    assert s2.score == 8400


def tsd_board(m):
    """A T-spin double slot: row 0 open at x=4, row 1 open at x=3-5, an overhang at (3, 2)."""
    rows = [0] * m.height
    rows[0] = FULL_ROW & ~(1 << 4)
    rows[1] = FULL_ROW & ~(0b111 << 3)
    rows[2] = 1 << 3
    return rows


TSD_CELLS = ((3, 1), (4, 0), (4, 1), (5, 1))


def test_tspin_double_at_20g():
    """At 20G the T settles into column 4 pointing right, then rotates into the slot: a T-spin."""
    m = Matrix()
    rows = tsd_board(m)
    reach = m.reachable_placements(rows, "T", m.spawn_state("T"), True)
    assert reach[TSD_CELLS][4] == {TSPIN_FULL}, reach[TSD_CELLS]
    _, lines, _ = m.lock(rows, TSD_CELLS)
    assert lines == 2
    # below 20G the bot moves at spawn height and hard drops, which can't get under the overhang
    assert TSD_CELLS not in m.reachable_placements(rows, "T", m.spawn_state("T"), False)


def test_tspin_rule():
    m = Matrix()
    rows = tsd_board(m)
    assert m.tspin_type(rows, 2, 3, 2, 1) == TSPIN_FULL   # both front corners + a back one
    assert m.tspin_type(rows, 2, 3, 2, 0) == TSPIN_NONE   # last action wasn't a rotation
    # pointing right with both back (left) corners and one front corner filled: a mini
    rows = [0] * m.height
    rows[0] = rows[1] = rows[2] = 1 << 3
    rows[0] |= 1 << 5
    assert m.tspin_type(rows, 1, 3, 2, 1) == TSPIN_MINI
    assert m.tspin_type(rows, 1, 3, 2, 4) == TSPIN_NONE   # kick 4 doesn't count
    assert m.tspin_type(rows, 1, 3, 2, 5) == TSPIN_FULL   # kick 5 makes it a T-spin
    # below 20G: rotate at spawn height, then a hard drop (which keeps the rotation) gives the mini;
    # rotating first and moving last doesn't
    reach = m.reachable_placements(rows, "T", m.spawn_state("T"), False)
    assert reach[((4, 0), (4, 1), (4, 2), (5, 1))][4] == {TSPIN_MINI, TSPIN_NONE}
    # against the left wall only a mini is possible, even with both front corners filled
    rows = [0] * m.height
    rows[0] = rows[2] = 1 << 1
    assert m.tspin_type(rows, 1, -1, 2, 1) == TSPIN_MINI


def test_tspin_scoring():
    s = Scorer()
    s.add(2, level=1, tspin=TSPIN_FULL)   # T-spin double: 1200, starts back-to-back
    s.add(2, level=1, tspin=TSPIN_FULL)   # back-to-back x1.5 + combo 1: 1800 + 50
    assert s.score == 3050
    s.add(0, level=1, tspin=TSPIN_FULL)   # T-spin, no lines: 400, ends the combo, keeps back-to-back
    s.add(1, level=1, tspin=TSPIN_MINI)   # mini single continues back-to-back: 200 x1.5
    s.add(2, level=1, tspin=TSPIN_MINI)   # mini double: no points, combo 1, chain untouched
    s.add(4, level=1)                     # back-to-back Tetris + combo 2: 1200 + 100
    assert s.score == 3050 + 400 + 300 + 50 + 1300, s.score
    assert s.tspins == {TSPIN_MINI: [0, 1, 1, 0], TSPIN_FULL: [1, 0, 2, 0]}
    s.add(3, level=1)                     # a triple breaks the chain
    s.add(1, level=1, tspin=TSPIN_FULL)   # T-spin single: 800 + combo 4
    assert s.score == 3050 + 400 + 300 + 50 + 1300 + 650 + 1000


def test_ready_lines():
    m = Matrix()
    rows = [0] * m.height
    for y in range(3):
        rows[y] = FULL_ROW & ~(1 << 9)  # three rows full except the right column
    assert m.ready_lines(rows) == 3
    rows[3] = FULL_ROW & ~(1 << 9)
    rows[4] = FULL_ROW & ~(1 << 9)
    assert m.ready_lines(rows) == 4      # capped at 4
    rows[1] |= 1 << 9                    # covered: the I would land on row 1's cell
    assert m.ready_lines(rows) == 3
    rows2 = [0] * m.height
    rows2[0] = FULL_ROW & ~(1 << 0) & ~(1 << 9)  # two empty columns: nothing is ready
    assert m.ready_lines(rows2) == 0


def test_marathon_and_dqn_policy():
    weights = [-12.63, 6.60, -9.22, -19.77, -13.08, -10.49, -1.61, -24.04]
    r = play_game(weights, seed=5, max_pieces=2000, start_level=1, max_lines=MARATHON_LINES)
    assert not r.topped_out and r.lines >= 300 and r.score > 100000, r
    dqn = MLPPolicy(json.load(open(os.path.join(RL_DIR, "weights_v1.json"))), version=1)
    r = play_game(dqn, seed=5, max_pieces=60, start_level=1)
    assert r.pieces == 60 and r.lines > 0, r


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
