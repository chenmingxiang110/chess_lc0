import chess
import numpy as np

class MCTSNode:

    """One node in a chess MCTS tree.

    Parameters
    ----------
    prior:
        Policy prior for the move that reaches this node.

    Attributes
    ----------
    visit_count:
        Number of simulations through this node.
    value_sum:
        Sum of backed-up values from this node's side-to-move perspective.
    children:
        Dictionary mapping `chess.Move` objects to child nodes.
    """

    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    @property
    def q(self):
        return 0.0 if self.visit_count==0 else self.value_sum/self.visit_count

    @property
    def expanded(self):
        return len(self.children)>0

class MCTS:

    """PUCT Monte Carlo tree search using an Lc0-style policy/value network.

    Parameters
    ----------
    net:
        Evaluator with `policy(board)` returning `{uci_move: prob}, value`.
    c_puct:
        Exploration constant for PUCT.
    dirichlet_alpha:
        Optional root Dirichlet noise alpha.
    exploration_fraction:
        Mixture weight for root Dirichlet noise.
    """

    def __init__(
        self,
        net,
        c_puct=1.5,
        dirichlet_alpha=None,
        exploration_fraction=0.25,
    ):
        self.net = net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.root = None

    def search(self, board, num_simulations=64):
        """Run MCTS from `board` and return the root node."""
        root = MCTSNode()
        if board.is_game_over(claim_draw=True):
            self.root = root
            return root

        self._expand(root, board)
        self._add_root_noise(root)

        for _ in range(num_simulations):
            sim_board = board.copy(stack=False)
            node = root
            path = [node]

            while node.expanded and not sim_board.is_game_over(claim_draw=True):
                move, node = self._select_child(node)
                sim_board.push(move)
                path.append(node)

            value = self._evaluate_leaf(node, sim_board)
            self._backpropagate(path, value)

        self.root = root
        return root

    def best_move(self, board, num_simulations=64, temperature=0.0):
        """Run search and return one move selected from root visit counts."""
        root = self.search(board, num_simulations)
        moves, visits = self._root_visits(root)
        if len(moves)==0:
            return None
        if temperature==0.0:
            return moves[int(np.argmax(visits))]

        probs = self._temperature_probs(visits, temperature)
        index = int(np.random.choice(len(moves), p=probs))
        return moves[index]

    def policy(self, board, num_simulations=64, temperature=1.0):
        """Run search and return `{uci_move: visit_probability}`."""
        root = self.search(board, num_simulations)
        moves, visits = self._root_visits(root)
        probs = self._temperature_probs(visits, temperature)
        return {move.uci(): float(prob) for move, prob in zip(moves, probs)}

    def _evaluate_leaf(self, node, board):
        if board.is_game_over(claim_draw=True):
            return self._terminal_value(board)
        return self._expand(node, board)

    def _expand(self, node, board):
        policy, value = self.net.policy(board)
        if len(policy)==0:
            return value

        priors = self._normalize_policy(policy)
        for move_uci, prior in priors.items():
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                node.children[move] = MCTSNode(prior)
        return value

    def _select_child(self, node):
        parent_visits = max(1, node.visit_count)
        scores = {
            move: self._puct_score(parent_visits, child)
            for move, child in node.children.items()
        }
        move = max(scores, key=scores.get)
        return move, node.children[move]

    def _puct_score(self, parent_visits, child):
        exploration = self.c_puct*child.prior*np.sqrt(parent_visits)/(1+child.visit_count)
        return -child.q+exploration

    def _backpropagate(self, path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _terminal_value(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner==board.turn else -1.0

    def _normalize_policy(self, policy):
        moves = list(policy.keys())
        probs = np.asarray([policy[move] for move in moves], dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total<=0.0:
            probs = np.ones(len(moves), dtype=np.float64)/len(moves)
        else:
            probs = probs/total
        return {move: float(prob) for move, prob in zip(moves, probs)}

    def _add_root_noise(self, root):
        if self.dirichlet_alpha is None or len(root.children)==0:
            return

        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha]*len(moves))
        for move, noise_prob in zip(moves, noise):
            child = root.children[move]
            child.prior = (
                (1-self.exploration_fraction)*child.prior
                + self.exploration_fraction*float(noise_prob)
            )

    def _root_visits(self, root):
        moves = list(root.children.keys())
        visits = np.asarray([root.children[move].visit_count for move in moves], dtype=np.float64)
        return moves, visits

    def _temperature_probs(self, visits, temperature):
        if len(visits)==0:
            return visits
        if temperature==0.0:
            probs = np.zeros(len(visits), dtype=np.float64)
            probs[int(np.argmax(visits))] = 1.0
            return probs

        visits = np.power(visits, 1.0/temperature)
        total = visits.sum()
        if total<=0.0:
            return np.ones(len(visits), dtype=np.float64)/len(visits)
        return visits/total
