"""
Monte Carlo Tree Search with neural-network priors and World Model rollouts.

UCT formula:
    UCT(s,a) = W(s,a)/N(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Uses:
    - DT's action logits  →  prior P(s, a)
    - World Model         →  state transitions during rollout
    - Value head / reward →  leaf evaluation
"""

import math
import numpy as np
import torch


class MCTSNode:
    __slots__ = ('state', 'parent', 'action', 'children', 'N', 'W', 'P',
                 'is_terminal', 'terminal_value')

    def __init__(self, state=None, parent=None, action=None, prior=0.0,
                 is_terminal=False, term_val=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}             # {action_id: MCTSNode}
        self.N = 0                     # visit count
        self.W = 0.0                   # total value accumulated
        self.P = prior                 # policy prior P(s, a)
        self.is_terminal = is_terminal
        self.terminal_value = term_val


class MCTS:
    """
    MCTS planner that uses DT for priors/values and WorldModel for rollouts.

    Usage:
        mcts = MCTS(world_model, dt, state_dim=256, n_simulations=100)
        action = mcts.search(root_state_features)
    """

    def __init__(self, world_model, dt_model,
                 state_dim=256, act_dim=5,
                 c_puct=2.0, n_simulations=100,
                 rollout_depth=10, discount=0.99,
                 device='cpu'):
        self.world_model = world_model
        self.dt = dt_model
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.discount = discount
        self.device = device

    # -----------------------------------------------------------------
    #  Main entry point
    # -----------------------------------------------------------------

    @torch.no_grad()
    def search(self, root_state, legal_actions=None):
        """
        Run MCTS and return the best action.

        Args:
            root_state:  (state_dim,)  numpy array — current state features
            legal_actions: list[int]   allowed actions (default: all)

        Returns:
            best_action: int
            root_node:   MCTSNode  (for debugging / visualization)
        """
        if legal_actions is None:
            legal_actions = list(range(self.act_dim))

        root = MCTSNode(state=root_state)

        # --- Get policy prior from DT for root ---
        # We pass a dummy context: just the root state repeated
        K = self.dt.context_len
        s_tensor = torch.FloatTensor(root_state).unsqueeze(0).unsqueeze(0).repeat(1, K, 1).to(self.device)
        a_tensor = torch.zeros(1, K, self.act_dim).to(self.device)
        r_tensor = torch.zeros(1, K, 1).to(self.device)
        t_tensor = torch.zeros(1, K, dtype=torch.long).to(self.device)

        action_logits, values, _ = self.dt(r_tensor, s_tensor, a_tensor, t_tensor)
        logits = action_logits[0, -1, :].cpu().numpy()  # (act_dim,)

        # Softmax over legal actions only
        exp_logits = np.exp(logits - np.max(logits))
        for a in range(self.act_dim):
            if a not in legal_actions:
                exp_logits[a] = 0.0
        probs = exp_logits / exp_logits.sum()

        # --- MCTS loop ---
        for _ in range(self.n_simulations):
            node = self._select(root)

            if node.is_terminal:
                value = node.terminal_value
                self._backpropagate(node, value)
                continue

            # Expand: create children for this node
            self._expand(node, probs if node is root else None)

            # Evaluate: rollout with WorldModel
            value = self._rollout(node)
            self._backpropagate(node, value)

        # --- Choose best action (by visit count) ---
        best_action = max(legal_actions,
                          key=lambda a: root.children[a].N if a in root.children else 0)
        return best_action, root

    # -----------------------------------------------------------------
    #  Four MCTS steps
    # -----------------------------------------------------------------

    def _select(self, node):
        """Walk down the tree using UCT until a leaf or unexpanded node."""
        while node.children and not node.is_terminal:
            best_uct = -float('inf')
            best_child = None

            for action, child in node.children.items():
                if child.N == 0:
                    return child  # unvisited child → select it

                exploitation = child.W / child.N
                exploration = (self.c_puct * child.P *
                               math.sqrt(node.N) / (1 + child.N))
                uct = exploitation + exploration

                if uct > best_uct:
                    best_uct = uct
                    best_child = child

            if best_child is None:
                break
            node = best_child

        return node

    def _expand(self, node, root_probs=None):
        """Create child nodes for all legal actions. Predict child states via WorldModel."""
        # Get prior probabilities
        if root_probs is not None:
            probs = root_probs
        elif node.state is not None:
            s_tensor = torch.FloatTensor(node.state).unsqueeze(0)
            logits = self._get_priors(s_tensor)
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / exp_l.sum()
        else:
            probs = np.ones(self.act_dim) / self.act_dim  # uniform fallback

        s_tensor = torch.FloatTensor(node.state).unsqueeze(0) if node.state is not None else None

        for a in range(self.act_dim):
            if a not in node.children:
                # Predict child state via WorldModel
                child_state = None
                if s_tensor is not None:
                    a_onehot = torch.zeros(1, self.act_dim).to(self.device)
                    a_onehot[0, a] = 1.0
                    ns, _, _ = self.world_model(s_tensor.to(self.device), a_onehot)
                    child_state = ns.squeeze(0).detach().cpu().numpy()
                node.children[a] = MCTSNode(state=child_state, parent=node,
                                            action=a, prior=probs[a])

    def _rollout(self, node):
        """
        Simulate from one of node's unvisited children using the WorldModel.
        Returns the accumulated discounted reward, bootstrapped by DT value.
        """
        unvisited = [c for c in node.children.values() if c.N == 0]
        if not unvisited:
            return 0.0

        child = unvisited[0]
        state = child.state
        if state is None:
            return 0.0

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        total = 0.0

        for depth in range(self.rollout_depth):
            logits = self._get_priors(s)
            action_id = int(np.argmax(logits))
            a_onehot = torch.zeros(1, self.act_dim).to(self.device)
            a_onehot[0, action_id] = 1.0

            ns, r, d = self.world_model(s, a_onehot)
            total += self.discount ** depth * r.item()

            if d.item() > 0.5:
                child.is_terminal = True
                child.terminal_value = total
                return total

            s = ns

        val = self._get_value(s.squeeze(0).detach().cpu().numpy())
        return total + self.discount ** self.rollout_depth * val

    def _backpropagate(self, node, value):
        """Propagate value up the tree, updating N and W."""
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent

    # -----------------------------------------------------------------
    #  DT helpers
    # -----------------------------------------------------------------

    def _get_priors(self, state_tensor):
        """Get action logits from DT for a state batch (B, D) or (1, D)."""
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        B = state_tensor.shape[0]
        K = self.dt.context_len
        s = state_tensor.unsqueeze(1).repeat(1, K, 1).to(self.device)   # (B, K, D)
        a = torch.zeros(B, K, self.act_dim).to(self.device)
        r = torch.zeros(B, K, 1).to(self.device)
        t = torch.zeros(B, K, dtype=torch.long).to(self.device)
        logits, _, _ = self.dt(r, s, a, t)
        return logits[:, -1, :].detach().cpu().numpy()[-1]  # (act_dim,) for last in batch

    def _get_value(self, state_np):
        """Get V(s) from DT's value head if available, else 0."""
        if self.dt.predict_value is None:
            return 0.0
        K = self.dt.context_len
        s = torch.FloatTensor(state_np).reshape(1, 1, -1).repeat(1, K, 1).to(self.device)
        a = torch.zeros(1, K, self.act_dim).to(self.device)
        r = torch.zeros(1, K, 1).to(self.device)
        t = torch.zeros(1, K, dtype=torch.long).to(self.device)
        _, values, _ = self.dt(r, s, a, t)
        if values is None:
            return 0.0
        return values[0, -1, 0].item()
