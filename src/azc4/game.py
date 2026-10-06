"""Connect Four: the rules of the game, and nothing else.

Two players take turns dropping a disc into one of 7 columns; the disc falls
to the lowest empty cell. The first to line up four in a row (across, up, or
diagonally) wins. If the board fills up first, it is a draw.

In reinforcement-learning terms this file is the *environment*:
  state    the board, plus whose turn it is
  action   which column to drop a disc into (0 to 6)
  reward   only at the very end: +1 for a win, -1 for a loss, 0 for a draw

Everything the agent will ever learn comes from playing by these rules.
"""
from __future__ import annotations

import numpy as np

ROWS, COLS = 6, 7
# The four line directions as (row step, column step): across, up, and both diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Connect4:
    """One position in a game. Positions are never changed in place:
    ``play`` returns a new position, so a search can explore many futures
    from the same starting point without them interfering."""

    def __init__(self) -> None:
        # board[row, col]: +1 = first player's disc, -1 = second player's, 0 = empty.
        # Row 0 is the bottom of the board.
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.player = 1           # whose turn it is: +1 or -1
        self.winner: int | None = None  # +1 / -1 once someone wins, 0 for a draw, None while playing
        self.moves = 0
        self.last_move: int | None = None

    # ------------------------------------------------------------------ rules
    @property
    def done(self) -> bool:
        return self.winner is not None

    def legal_moves(self) -> list[int]:
        """Columns that still have room (none once the game is over)."""
        if self.done:
            return []
        return [c for c in range(COLS) if self.board[ROWS - 1, c] == 0]

    def play(self, col: int) -> "Connect4":
        """Drop the current player's disc into ``col`` and return the new position."""
        if self.done:
            raise ValueError("the game is over")
        if not 0 <= col < COLS or self.board[ROWS - 1, col] != 0:
            raise ValueError(f"column {col} is not a legal move")

        nxt = Connect4.__new__(Connect4)
        nxt.board = self.board.copy()
        row = int(np.argmax(nxt.board[:, col] == 0))  # lowest empty cell in the column
        nxt.board[row, col] = self.player
        nxt.moves = self.moves + 1
        nxt.last_move = col
        if _makes_four(nxt.board, row, col):
            nxt.winner = self.player
        elif nxt.moves == ROWS * COLS:
            nxt.winner = 0
        else:
            nxt.winner = None
        nxt.player = -self.player
        return nxt

    # ------------------------------------------------- views for the learner
    def canonical(self) -> np.ndarray:
        """The board as seen by the player about to move: their own discs are
        +1 and the opponent's are -1, whichever colour they actually have.

        This lets one network play both sides: it only ever has to answer
        "what should *I* do here?"."""
        return self.board * self.player

    def result_for(self, player: int) -> int:
        """Final reward for ``player``: +1 win, -1 loss, 0 draw."""
        if not self.done:
            raise ValueError("the game is not over yet")
        return self.winner * player

    # ---------------------------------------------------------------- display
    def render(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = [" ".join(symbols[int(v)] for v in self.board[r]) for r in range(ROWS - 1, -1, -1)]
        return "\n".join(rows + [" ".join(str(c) for c in range(COLS))])


def _makes_four(board: np.ndarray, row: int, col: int) -> bool:
    """Did the disc just placed at (row, col) complete a line of four?

    Only lines through the new disc can be new, so it is enough to count
    outwards from it in each direction."""
    me = board[row, col]
    for dr, dc in DIRECTIONS:
        count = 1
        for sign in (1, -1):  # walk both ways along the line
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == me:
                count += 1
                r, c = r + sign * dr, c + sign * dc
        if count >= 4:
            return True
    return False
