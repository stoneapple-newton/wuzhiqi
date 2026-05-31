from wuzhiqi.game import Gomoku


def test_reset_starts_empty_with_black_to_move() -> None:
    game = Gomoku(board_size=15)

    assert game.current_player == 1
    assert len(game.board) == 15
    assert all(cell == 0 for row in game.board for cell in row)


def test_legal_moves_are_all_points_on_empty_board() -> None:
    game = Gomoku(board_size=15)

    assert game.legal_moves() == list(range(225))
