"""Fast heuristic Gomoku players for lightweight evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from wuzhiqi.game import Board, Move


DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


@dataclass
class HeuristicPlayer:
    """Rule-based player intended as a cheap evaluation baseline."""

    search_radius: int = 2
    player: int = 0

    def set_player_ind(self, p: int) -> None:
        self.player = p

    def reset_player(self) -> None:
        return None

    def get_action(self, board: Board) -> Move:
        if not board.availables:
            raise ValueError("the board is full")

        candidates = self._candidate_moves(board)
        current_player = board.get_current_player()
        opponent = board.players[0] if current_player == board.players[1] else board.players[1]

        winning_move = self._find_finishing_move(board, candidates, current_player)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_finishing_move(board, candidates, opponent)
        if blocking_move is not None:
            return blocking_move

        center = ((board.height - 1) / 2.0, (board.width - 1) / 2.0)
        return max(
            candidates,
            key=lambda move: (
                self._move_score(board, move, current_player),
                self._move_score(board, move, opponent) * 0.8,
                -self._distance_to_center(board, move, center),
            ),
        )

    def _candidate_moves(self, board: Board) -> list[Move]:
        if not board.states:
            return [board.location_to_move([board.height // 2, board.width // 2])]

        candidates: set[Move] = set()
        for move in board.states:
            row, col = board.move_to_location(move)
            for delta_row in range(-self.search_radius, self.search_radius + 1):
                for delta_col in range(-self.search_radius, self.search_radius + 1):
                    candidate = board.location_to_move([row + delta_row, col + delta_col])
                    if candidate in board.availables:
                        candidates.add(candidate)
        return sorted(candidates) or list(board.availables)

    def _find_finishing_move(self, board: Board, candidates: list[Move], player: int) -> Move | None:
        for move in candidates:
            if self._would_win(board, move, player):
                return move
        return None

    def _would_win(self, board: Board, move: Move, player: int) -> bool:
        row, col = board.move_to_location(move)
        return any(
            1 + self._count_direction(board, row, col, row_step, col_step, player)
            + self._count_direction(board, row, col, -row_step, -col_step, player)
            >= board.n_in_row
            for row_step, col_step in DIRECTIONS
        )

    def _move_score(self, board: Board, move: Move, player: int) -> float:
        row, col = board.move_to_location(move)
        score = 0.0
        for row_step, col_step in DIRECTIONS:
            forward = self._count_direction(board, row, col, row_step, col_step, player)
            backward = self._count_direction(board, row, col, -row_step, -col_step, player)
            run_length = 1 + forward + backward
            open_ends = self._is_open(board, row, col, row_step, col_step, forward)
            open_ends += self._is_open(board, row, col, -row_step, -col_step, backward)
            score += (10**run_length) * (1 + open_ends)
        return score

    def _count_direction(
        self,
        board: Board,
        row: int,
        col: int,
        row_step: int,
        col_step: int,
        player: int,
    ) -> int:
        count = 0
        next_row = row + row_step
        next_col = col + col_step
        while 0 <= next_row < board.height and 0 <= next_col < board.width:
            move = board.location_to_move([next_row, next_col])
            if board.states.get(move) != player:
                break
            count += 1
            next_row += row_step
            next_col += col_step
        return count

    def _is_open(
        self,
        board: Board,
        row: int,
        col: int,
        row_step: int,
        col_step: int,
        run_length: int,
    ) -> int:
        next_row = row + row_step * (run_length + 1)
        next_col = col + col_step * (run_length + 1)
        if not (0 <= next_row < board.height and 0 <= next_col < board.width):
            return 0
        return int(board.location_to_move([next_row, next_col]) in board.availables)

    def _distance_to_center(self, board: Board, move: Move, center: tuple[float, float]) -> float:
        row, col = board.move_to_location(move)
        return abs(row - center[0]) + abs(col - center[1])

    def __str__(self) -> str:
        return f"Heuristic {self.player}"
