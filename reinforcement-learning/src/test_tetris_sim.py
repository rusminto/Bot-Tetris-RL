"""
Sanity checks for tetris_sim.py. Run: python3 src/test_tetris_sim.py
"""

import random

import numpy as np

from tetris_sim import Matrix, PIECE_NAMES, SRS_CELLS, play_game


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
    lines, pieces, topped = play_game(weights, seed=3, max_pieces=300)
    assert pieces == 300 and not topped and lines > 100, (lines, pieces, topped)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
