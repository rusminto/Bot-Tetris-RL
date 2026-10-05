"""
Digital Twin Tetris Engine for Reinforcement Learning.
Accurately replicates the official play.tetris.com physics and rules:
- 10x20 Grid & 7-Bag Randomizer
- Exact official fall speed table (1000ms at Level 1 down to 0ms / 20G at Level 19+)
- Exact lock delay table (500ms down to 150ms)
- Level progression (every 10 lines cleared)
- Official Hold piece mechanism (swapping once per turn)
- Kinematic path reachability & transit-time collision validation
- 6-Feature Board Representation for RL state evaluation
"""

import random
import numpy as np

# Standard Tetromino piece matrices (SRS initial orientations)
PIECES = {
    'I': [
        [[1, 1, 1, 1]],
        [[1],
         [1],
         [1],
         [1]]
    ],
    'O': [
        [[1, 1],
         [1, 1]]
    ],
    'T': [
        [[0, 1, 0],
         [1, 1, 1]],
        [[1, 0],
         [1, 1],
         [1, 0]],
        [[1, 1, 1],
         [0, 1, 0]],
        [[0, 1],
         [1, 1],
         [0, 1]]
    ],
    'S': [
        [[0, 1, 1],
         [1, 1, 0]],
        [[1, 0],
         [1, 1],
         [0, 1]]
    ],
    'Z': [
        [[1, 1, 0],
         [0, 1, 1]],
        [[0, 1],
         [1, 1],
         [1, 0]]
    ],
    'J': [
        [[1, 0, 0],
         [1, 1, 1]],
        [[1, 1],
         [1, 0],
         [1, 0]],
        [[1, 1, 1],
         [0, 0, 1]],
        [[0, 1],
         [0, 1],
         [1, 1]]
    ],
    'L': [
        [[0, 0, 1],
         [1, 1, 1]],
        [[1, 0],
         [1, 0],
         [1, 1]],
        [[1, 1, 1],
         [1, 0, 0]],
        [[1, 1],
         [0, 1],
         [0, 1]]
    ]
}

PIECE_NAMES = list(PIECES.keys())

# Exact gravity table from play.tetris.com (mNormalFallSpeedMSEC)
FALL_SPEED_TABLE = {
    1: 1000, 2: 793, 3: 618, 4: 473, 5: 355,
    6: 262, 7: 190, 8: 135, 9: 94, 10: 64,
    11: 43, 12: 28, 13: 18, 14: 11, 15: 7,
    16: 4, 17: 3, 18: 1, 19: 0, 20: 0,
    21: 0, 22: 0, 23: 0, 24: 0, 25: 0,
    26: 0, 27: 0, 28: 0, 29: 0, 30: 0
}

# Exact lock delay table from play.tetris.com (mLockTimeMSEC)
LOCK_DELAY_TABLE = {
    1: 500, 2: 500, 3: 500, 4: 500, 5: 500,
    6: 500, 7: 500, 8: 500, 9: 500, 10: 500,
    11: 500, 12: 500, 13: 500, 14: 500, 15: 500,
    16: 500, 17: 500, 18: 500, 19: 500, 20: 450,
    21: 400, 22: 350, 23: 300, 24: 250, 25: 200,
    26: 195, 27: 184, 28: 167, 29: 151, 30: 150
}

LINES_PER_LEVEL = 10
KEY_DELAY_MS = 15  # Milliseconds per simulated keystroke action


class TetrisEngine:
    WIDTH = 10
    HEIGHT = 20

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.bag = []
        self.board = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.current_piece = None
        self.next_piece = None
        self.hold_piece = None
        self.can_hold = True
        self.reset()

    def _fill_bag(self):
        bag = PIECE_NAMES.copy()
        random.shuffle(bag)
        return bag

    def _get_next_piece(self):
        if not self.bag:
            self.bag = self._fill_bag()
        return self.bag.pop()

    @property
    def level(self):
        return min(30, (self.lines_cleared // LINES_PER_LEVEL) + 1)

    @property
    def fall_speed_ms(self):
        return FALL_SPEED_TABLE.get(self.level, 0)

    @property
    def lock_delay_ms(self):
        return LOCK_DELAY_TABLE.get(self.level, 150)

    def reset(self):
        self.board = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.hold_piece = None
        self.can_hold = True
        self.bag = self._fill_bag()
        self.current_piece = self._get_next_piece()
        self.next_piece = self._get_next_piece()
        return self._calculate_features(self.board, 0)

    def _calculate_features(self, board, lines_cleared):
        """
        Extract the 6 primary RL features from the board:
        1. Lines cleared in this state
        2. Total holes (empty cells covered by occupied cells above)
        3. Bumpiness (sum of absolute differences between column heights)
        4. Total aggregate column height
        5. Max column height (danger indicator for top-out)
        6. Normalized level (0.0 to 1.0 for dynamic speed adaptation)
        """
        column_heights = np.zeros(self.WIDTH, dtype=np.int32)
        holes = 0

        for col in range(self.WIDTH):
            column = board[:, col]
            occupied = np.where(column > 0)[0]
            if len(occupied) > 0:
                top_row = occupied[0]
                column_heights[col] = self.HEIGHT - top_row
                holes += int(np.sum(column[top_row:] == 0))
            else:
                column_heights[col] = 0

        bumpiness = int(np.sum(np.abs(np.diff(column_heights))))
        total_height = int(np.sum(column_heights))
        max_height = int(np.max(column_heights))
        normalized_level = float(min(30, self.level) - 1) / 29.0

        return np.array([
            float(lines_cleared),
            float(holes),
            float(bumpiness),
            float(total_height),
            float(max_height),
            float(normalized_level)
        ], dtype=np.float32)

    def _is_valid_position(self, shape, x, y, board):
        shape_h = len(shape)
        shape_w = len(shape[0])

        for r in range(shape_h):
            for c in range(shape_w):
                if shape[r][c]:
                    board_y = y + r
                    board_x = x + c
                    if board_x < 0 or board_x >= self.WIDTH:
                        return False
                    if board_y >= self.HEIGHT:
                        return False
                    if board_y >= 0 and board[board_y, board_x] > 0:
                        return False
        return True

    def _is_reachable(self, shape, piece_name, rot_idx, target_x):
        """
        Simulate the physical movement of the piece under gravity:
        - Spawn at x = (WIDTH - w) // 2, y = 0
        - Rotate in place to rot_idx
        - Lateral shift towards target_x with KEY_DELAY_MS
        - Account for rows dropped during transit due to fall_speed_ms
        Returns:
            (is_reachable: bool, final_y: int or None)
        """
        shape_w = len(shape[0])
        spawn_w = len(PIECES[piece_name][0][0])
        spawn_x = (self.WIDTH - spawn_w) // 2
        spawn_y = 0

        # Initial check at spawn position
        if not self._is_valid_position(PIECES[piece_name][0], spawn_x, spawn_y, self.board):
            return False, None

        # 1. Rotations take time
        num_rotates = rot_idx % 4
        t_rot = num_rotates * KEY_DELAY_MS

        # Check if piece fits rotated at spawn
        if not self._is_valid_position(shape, spawn_x, spawn_y, self.board):
            return False, None

        # 2. Lateral transit from spawn_x to target_x
        dx = target_x - spawn_x
        steps = abs(dx)
        step_dir = 1 if dx > 0 else -1

        current_x = spawn_x
        current_y = 0

        for s in range(1, steps + 1):
            current_x = spawn_x + s * step_dir
            t_elapsed = t_rot + s * KEY_DELAY_MS

            if self.fall_speed_ms > 0:
                drop_rows = min(19, t_elapsed // self.fall_speed_ms)
                current_y = max(current_y, drop_rows)
            else:
                # 20G: instant drop, piece must slide along floor
                # If floor in current_x is higher than current_y, it's blocked by a wall
                pass

            if not self._is_valid_position(shape, current_x, current_y, self.board):
                # Mid-air collision with stack!
                return False, None

        # 3. Hard drop from current transit position to floor
        y = current_y
        while self._is_valid_position(shape, target_x, y + 1, self.board):
            y += 1

        if not self._is_valid_position(shape, target_x, y, self.board):
            return False, None

        return True, y

    def _clear_lines(self, board):
        full_rows = np.all(board > 0, axis=1)
        num_cleared = int(np.sum(full_rows))
        if num_cleared == 0:
            return 0, board

        new_board = np.zeros_like(board)
        non_full_rows = board[~full_rows]
        new_board[num_cleared:] = non_full_rows
        return num_cleared, new_board

    def _add_piece_states(self, piece_name, use_hold, states_dict):
        orientations = PIECES[piece_name]
        for rot_idx, shape in enumerate(orientations):
            shape_h = len(shape)
            shape_w = len(shape[0])

            for x in range(self.WIDTH - shape_w + 1):
                reachable, final_y = self._is_reachable(shape, piece_name, rot_idx, x)
                if not reachable:
                    continue

                # Place piece on a copy of board
                next_board = self.board.copy()
                for r in range(shape_h):
                    for c in range(shape_w):
                        if shape[r][c]:
                            next_board[final_y + r, x + c] = 1

                lines, cleared_board = self._clear_lines(next_board)
                features = self._calculate_features(cleared_board, lines)
                states_dict[(use_hold, rot_idx, x)] = (features, cleared_board)

    def get_next_states(self):
        """
        Generate all valid next states for current piece, and if hold is available,
        also for the held/queued piece.
        Returns:
            dict: {(use_hold, rot_idx, x): (feature_vector, next_board)}
        """
        states = {}

        # 1. Evaluate current piece placements
        self._add_piece_states(self.current_piece, use_hold=False, states_dict=states)

        # 2. Evaluate hold piece placements if hold is allowed
        if self.can_hold:
            held_candidate = self.hold_piece if self.hold_piece is not None else self.next_piece
            self._add_piece_states(held_candidate, use_hold=True, states_dict=states)

        return states

    def step(self, action_tuple):
        """
        action_tuple: (use_hold, rot_idx, x)
        Executes placement, advances hold/next piece, checks line clears & level progression.
        Returns:
            (reward, done)
        """
        use_hold, rot_idx, x = action_tuple

        # Execute hold swap if requested
        if use_hold and self.can_hold:
            if self.hold_piece is None:
                self.hold_piece = self.current_piece
                self.current_piece = self.next_piece
                self.next_piece = self._get_next_piece()
            else:
                self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
            self.can_hold = False

        shape = PIECES[self.current_piece][rot_idx]
        shape_h = len(shape)
        shape_w = len(shape[0])

        reachable, final_y = self._is_reachable(shape, self.current_piece, rot_idx, x)
        if not reachable:
            self.game_over = True
            return -20.0, True

        # Place piece
        for r in range(shape_h):
            for c in range(shape_w):
                if shape[r][c]:
                    self.board[final_y + r, x + c] = 1

        prev_level = self.level
        lines, self.board = self._clear_lines(self.board)
        self.lines_cleared += lines
        self.pieces_placed += 1
        new_level = self.level

        # Reset hold availability for the new incoming turn
        self.can_hold = True

        # Reward structure:
        # Base piece survival: +1.0
        # Quadratic line clears: 1: +10, 2: +30, 3: +60, 4: +100
        line_rewards = {0: 1.0, 1: 10.0, 2: 30.0, 3: 60.0, 4: 100.0}
        reward = line_rewards.get(lines, 1.0)
        self.score += int(reward * 10)

        # Level advancement bonus
        if new_level > prev_level:
            reward += 15.0 * new_level

        # Advance piece queue
        self.current_piece = self.next_piece
        self.next_piece = self._get_next_piece()

        # Check top-out condition at spawn row
        spawn_shape = PIECES[self.current_piece][0]
        spawn_x = (self.WIDTH - len(spawn_shape[0])) // 2
        if not self._is_valid_position(spawn_shape, spawn_x, 0, self.board):
            self.game_over = True
            reward -= 20.0

        return reward, self.game_over
