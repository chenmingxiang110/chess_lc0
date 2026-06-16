import chess
import numpy as np
from lczero.backends import Backend, GameState, Weights

DEFAULT_LC0_BACKENDS = ("cudnn", "cuda", "metal", "blas", "eigen")

class Lc0Net:

    """Small Python wrapper around the official Lc0 backend.

    Parameters
    ----------
    weights_path:
        Path to an Lc0 `.pb.gz` network. Defaults to the small network in `ckpts`.
    backend:
        Lc0 backend name. If omitted, try cudnn, cuda, metal, blas, and eigen in order.

    Returns
    -------
    policy_probs, value:
        Calling the instance returns legal-move probabilities and scalar value.
    """

    def __init__(self, weights_path, backend=None):
        self.weights_path = weights_path
        self.weights = Weights(self.weights_path)
        self.backend_name, self.backend = self._make_backend(backend)

    def __call__(self, board):
        return self.evaluate(board)

    def evaluate(self, board):
        """Evaluate one chess position.

        Parameters
        ----------
        board:
            `chess.Board`, FEN string, `GameState`.

        Returns
        -------
        policy_probs:
            Numpy array with one probability per legal move in `self.legal_moves(board)` order.
        value:
            Float value from the side-to-move perspective.
        """
        state = self._game_state(board)
        result = self.backend.evaluate(state.as_input(self.backend))[0]
        policy_indices = state.policy_indices()
        policy_probs = np.asarray(result.p_softmax(*policy_indices), dtype=np.float32)
        return policy_probs, float(result.q())

    def policy(self, board):
        """Return a dictionary mapping UCI legal moves to policy probabilities."""
        policy_probs, value = self.evaluate(board)
        moves = self.legal_moves(board)
        return {move: float(prob) for move, prob in zip(moves, policy_probs)}, value

    def best_move(self, board):
        """Return the legal move with the largest network policy probability."""
        policy_probs, value = self.evaluate(board)
        moves = self.legal_moves(board)
        move = moves[int(np.argmax(policy_probs))]
        return move, float(policy_probs.max()), value

    def evaluate_wdl(self, board):
        """Return WDL probabilities from White's perspective.

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
        board = self._board(board)
        if board.is_game_over(claim_draw=True):
            return self._terminal_wdl(board)

        state = GameState(fen=board.fen())
        result = self.backend.evaluate(state.as_input(self.backend))[0]
        stm_win, draw, stm_loss = self._wdl_from_qd(float(result.q()), float(result.d()))
        white_win, black_win = self._white_black_from_stm(board, stm_win, stm_loss)
        return {
            "white_win": white_win,
            "draw": draw,
            "black_win": black_win,
            "side_to_move_win": stm_win,
            "side_to_move_loss": stm_loss,
            "q": float(result.q()),
            "d": float(result.d()),
            "m": float(result.m()),
        }

    def wdl(self, board):
        """Return `(white_win, draw, black_win)` for one chess position."""
        result = self.evaluate_wdl(board)
        return result["white_win"], result["draw"], result["black_win"]

    def legal_moves(self, board):
        state = self._game_state(board)
        return list(state.moves())

    def _board(self, board):
        if isinstance(board, chess.Board):
            return board.copy(stack=False)
        if isinstance(board, str):
            return chess.Board() if board=="startpos" else chess.Board(board)
        raise ValueError("board must be chess.Board, FEN string, or 'startpos'")

    def _game_state(self, board):
        if isinstance(board, GameState):
            return board
        if isinstance(board, chess.Board):
            return GameState(fen=board.fen())
        if isinstance(board, str):
            if board=="startpos":
                return GameState(fen=chess.Board().fen())
            return GameState(fen=board)
        raise ValueError("board must be chess.Board, FEN string, GameState")

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

    def _terminal_wdl(self, board):
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
