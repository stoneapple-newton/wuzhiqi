"""Gomoku environment and rules.

This module will own board state, legal moves, move application, and terminal
state detection. Keep it dependency-light so rules can be tested quickly.
"""


class Gomoku:
    """Minimal Gomoku game scaffold."""

    def __init__(self, board_size: int = 15) -> None:
        self.board_size = board_size
        self.reset()

    def reset(self) -> list[list[int]]:
        """Reset to an empty board with black to move."""
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = 1
        return self.board

    def legal_moves(self) -> list[int]:
        """Return row-major action indexes for empty intersections."""
        moves: list[int] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 0:
                    moves.append(row * self.board_size + col)
        return moves

    def step(self, action: int) -> tuple[list[list[int]], int, bool, dict[str, object]]:
        """Apply an action.

        Implementation is intentionally deferred until the game-engine story.
        """
        raise NotImplementedError("Gomoku.step is not implemented yet.")
