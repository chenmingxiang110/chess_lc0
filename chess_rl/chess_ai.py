import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

class GreedyWdlAI:

    def choose_move(self, board, net):
        """Return the legal move with the best white_win_rate for the side to move.

        Args:
            board: chess.Board at the current position.
            net: lc0-style model with a wdl(board) method returning white/draw/black probabilities.
        """
        best_move = None
        best_rate = None
        maximize = board.turn==chess.WHITE
        for move in board.legal_moves:
            next_board = board.copy(stack=False)
            next_board.push(move)
            white_win, draw, black_win = net.wdl(next_board)
            rate = float(white_win)+0.5*float(draw)
            if best_rate is None or (maximize and rate>best_rate) or (not maximize and rate<best_rate):
                best_move = move
                best_rate = rate
        return best_move

class MinimaxWdlAI:

    def __init__(self, depth=2):
        self.depth = int(max(1, depth))
        self.cache = {}
        self.eval_cache = {}

    def choose_move(self, board, net):
        """Return the minimax move using white_win_rate as the leaf evaluation.

        Args:
            board: chess.Board at the current position.
            net: lc0-style model with a wdl(board) method returning white/draw/black probabilities.
        """
        self.cache = {}
        self.eval_cache = {}
        best_move = None
        best_score = None
        alpha = -1.0
        beta = 2.0
        maximize = board.turn==chess.WHITE
        for move in self._ordered_moves(board):
            board.push(move)
            score = self._search(board, self.depth-1, alpha, beta, net)
            board.pop()
            if best_score is None or (maximize and score>best_score) or (not maximize and score<best_score):
                best_move = move
                best_score = score
            if maximize:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha>=beta:
                break
        return best_move

    def start_search(self, board, net):
        return MinimaxWdlSearchTask(self, board, net)

    def _search(self, board, depth, alpha, beta, net):
        if depth<=0 or board.is_game_over(claim_draw=True):
            return self._evaluate(board, net)

        key = (board.fen(), depth)
        if key in self.cache:
            return self.cache[key]

        maximize = board.turn==chess.WHITE
        if maximize:
            score, complete = self._maximize(board, depth, alpha, beta, net)
        else:
            score, complete = self._minimize(board, depth, alpha, beta, net)
        if complete:
            self.cache[key] = score
        return score

    def _maximize(self, board, depth, alpha, beta, net):
        best_score = -1.0
        moves = self._ordered_moves(board)
        for move in moves:
            board.push(move)
            score = self._search(board, depth-1, alpha, beta, net)
            board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha>=beta:
                return best_score, False
        return best_score, True

    def _minimize(self, board, depth, alpha, beta, net):
        best_score = 2.0
        moves = self._ordered_moves(board)
        for move in moves:
            board.push(move)
            score = self._search(board, depth-1, alpha, beta, net)
            board.pop()
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if alpha>=beta:
                return best_score, False
        return best_score, True

    def _evaluate(self, board, net):
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner==chess.WHITE else 0.0
        fen = board.fen()
        if fen in self.eval_cache:
            return self.eval_cache[fen]
        white_win, draw, black_win = net.wdl(board)
        score = float(white_win)+0.5*float(draw)
        self.eval_cache[fen] = score
        return score

    def _ordered_moves(self, board):
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self._move_order_score(board, move), reverse=True)
        return moves

    def _move_order_score(self, board, move):
        score = 0
        if board.is_capture(move):
            victim = self._captured_piece(board, move)
            attacker = board.piece_at(move.from_square)
            if victim is not None and attacker is not None:
                score += 10000+PIECE_VALUES[victim.piece_type]-PIECE_VALUES[attacker.piece_type]//10
        if move.promotion is not None:
            score += PIECE_VALUES[move.promotion]
        if board.gives_check(move):
            score += 500
        return score

    def _captured_piece(self, board, move):
        captured = board.piece_at(move.to_square)
        if captured is not None:
            return captured
        if board.is_en_passant(move):
            offset = -8 if board.turn==chess.WHITE else 8
            return board.piece_at(move.to_square+offset)
        return None

class MinimaxWdlSearchTask:

    def __init__(self, ai, board, net):
        self.ai = ai
        self.board = board.copy(stack=False)
        self.net = net
        self.ai.cache = {}
        self.ai.eval_cache = {}
        self.count_cache = {}
        self.done = 0
        self.total = max(1, self._count_leaves(self.board, self.ai.depth))
        self.best_move = None
        self.finished = False
        self._runner = self._run()

    def step(self, budget=1):
        for _ in range(budget):
            if self.finished:
                return
            try:
                next(self._runner)
            except StopIteration:
                self.done = self.total
                self.finished = True
                return

    def _run(self):
        best_score = None
        alpha = -1.0
        beta = 2.0
        maximize = self.board.turn==chess.WHITE
        for move in self.ai._ordered_moves(self.board):
            self.board.push(move)
            score = yield from self._search(self.ai.depth-1, alpha, beta)
            self.board.pop()
            if best_score is None or (maximize and score>best_score) or (not maximize and score<best_score):
                self.best_move = move
                best_score = score
            if maximize:
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha>=beta:
                break
        self.finished = True
        self.done = self.total

    def _search(self, depth, alpha, beta):
        if depth<=0 or self.board.is_game_over(claim_draw=True):
            score = self.ai._evaluate(self.board, self.net)
            self._advance(1)
            yield
            return score

        key = (self.board.fen(), depth)
        if key in self.ai.cache:
            self._advance(self._count_leaves(self.board, depth))
            yield
            return self.ai.cache[key]

        maximize = self.board.turn==chess.WHITE
        if maximize:
            score, complete = yield from self._maximize(depth, alpha, beta)
        else:
            score, complete = yield from self._minimize(depth, alpha, beta)
        if complete:
            self.ai.cache[key] = score
        return score

    def _maximize(self, depth, alpha, beta):
        best_score = -1.0
        moves = self.ai._ordered_moves(self.board)
        for i, move in enumerate(moves):
            self.board.push(move)
            score = yield from self._search(depth-1, alpha, beta)
            self.board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha>=beta:
                self._advance_skipped(moves[i+1:], depth)
                yield
                return best_score, False
        return best_score, True

    def _minimize(self, depth, alpha, beta):
        best_score = 2.0
        moves = self.ai._ordered_moves(self.board)
        for i, move in enumerate(moves):
            self.board.push(move)
            score = yield from self._search(depth-1, alpha, beta)
            self.board.pop()
            best_score = min(best_score, score)
            beta = min(beta, best_score)
            if alpha>=beta:
                self._advance_skipped(moves[i+1:], depth)
                yield
                return best_score, False
        return best_score, True

    def _advance_skipped(self, moves, depth):
        skipped = 0
        for move in moves:
            self.board.push(move)
            skipped += self._count_leaves(self.board, depth-1)
            self.board.pop()
        self._advance(skipped)

    def _count_leaves(self, board, depth):
        if depth<=0 or board.is_game_over(claim_draw=True):
            return 1
        key = (board.fen(), depth)
        if key in self.count_cache:
            return self.count_cache[key]
        total = 0
        for move in self.ai._ordered_moves(board):
            board.push(move)
            total += self._count_leaves(board, depth-1)
            board.pop()
        self.count_cache[key] = max(1, total)
        return self.count_cache[key]

    def _advance(self, amount):
        self.done = min(self.total, self.done+max(0, int(amount)))
