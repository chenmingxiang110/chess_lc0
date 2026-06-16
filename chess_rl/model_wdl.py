from pathlib import Path

import chess
import numpy as np
from lczero.backends import Backend, GameState, Weights

from chess_rl.model import DEFAULT_LC0_BACKENDS

DEFAULT_WEIGHTS = Path("ckpts/t1-512x15x8h-distilled-swa-3395000.pb.gz")

class Lc0WdlModel:

    """Estimate WDL probabilities from an Lc0 `.pb.gz` network.

    Parameters
    ----------
    weights_path:
        Path to an Lc0 `.pb.gz` network. Defaults to the small checkpoint in `ckpts`.
    backend:
        Lc0 backend name. If omitted, try cudnn, cuda, metal, blas, and eigen in order.

    Notes
    -----
    The lc0 backend returns `q` and `d` from the side-to-move perspective:
    `q = side_to_move_win - side_to_move_loss`, `d = draw`.
    """

    def __init__(self, weights_path=None, backend=None):
        self.weights_path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS
        self.weights = Weights(str(self.weights_path))
        self.backend_name, self.backend = self._make_backend(backend)

    def evaluate(self, board):
        """Return WDL probabilities for one chess position.

        Parameters
        ----------
        board:
            `chess.Board`, FEN string, or `"startpos"`.

        Returns
        -------
        result:
            Dictionary with `white_win`, `draw`, `black_win`, side-to-move WDL,
            raw `q`, raw `d`, and moves-left estimate `m`.
        """
        board = self._as_board(board)
        if board.is_game_over(claim_draw=True):
            return self._terminal_result(board)

        output = self._evaluate_backend(board)
        stm_win, draw, stm_loss = self._wdl_from_qd(float(output.q()), float(output.d()))
        white_win, black_win = self._white_black_from_stm(board, stm_win, stm_loss)
        return {
            "white_win": white_win,
            "draw": draw,
            "black_win": black_win,
            "side_to_move_win": stm_win,
            "side_to_move_loss": stm_loss,
            "q": float(output.q()),
            "d": float(output.d()),
            "m": float(output.m()),
        }

    def wdl(self, board):
        """Return `(white_win, draw, black_win)` for one chess position."""
        result = self.evaluate(board)
        return result["white_win"], result["draw"], result["black_win"]

    def _evaluate_backend(self, board):
        state = GameState(fen=board.fen())
        return self.backend.evaluate(state.as_input(self.backend))[0]

    def _make_backend(self, backend):
        if backend is not None:
            return backend, Backend(weights=self.weights, backend=backend)

        errors = []
        for candidate in DEFAULT_LC0_BACKENDS:
            try:
                lc0_backend = Backend(weights=self.weights, backend=candidate)
                state = GameState(fen=chess.Board().fen())
                lc0_backend.evaluate(state.as_input(lc0_backend))
                return candidate, lc0_backend
            except Exception as error:
                errors.append(f"{candidate}: {error}")
        message = "No available Lc0 backend. Tried: "+"; ".join(errors)
        raise RuntimeError(message)

    def _as_board(self, board):
        if isinstance(board, chess.Board):
            return board.copy(stack=False)
        if isinstance(board, str):
            return chess.Board() if board=="startpos" else chess.Board(board)
        raise ValueError("board must be chess.Board, FEN string, or 'startpos'")

    def _wdl_from_qd(self, q, draw):
        draw = float(np.clip(draw, 0.0, 1.0))
        q = float(np.clip(q, -(1.0-draw), 1.0-draw))
        win = (1.0-draw+q)/2.0
        loss = (1.0-draw-q)/2.0
        probs = np.asarray([win, draw, loss], dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        probs = probs/probs.sum()
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _white_black_from_stm(self, board, stm_win, stm_loss):
        if board.turn==chess.WHITE:
            return stm_win, stm_loss
        return stm_loss, stm_win

    def _terminal_result(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            white_win, draw, black_win = 0.0, 1.0, 0.0
        elif outcome.winner==chess.WHITE:
            white_win, draw, black_win = 1.0, 0.0, 0.0
        else:
            white_win, draw, black_win = 0.0, 0.0, 1.0

        stm_win = white_win if board.turn==chess.WHITE else black_win
        stm_loss = black_win if board.turn==chess.WHITE else white_win
        return {
            "white_win": white_win,
            "draw": draw,
            "black_win": black_win,
            "side_to_move_win": stm_win,
            "side_to_move_loss": stm_loss,
            "q": stm_win-stm_loss,
            "d": draw,
            "m": 0.0,
        }

def estimate_wdl(board, weights_path=None, backend=None):
    """Convenience function returning `(white_win, draw, black_win)`."""
    model = Lc0WdlModel(weights_path=weights_path, backend=backend)
    return model.wdl(board)
