"""
PPO training loop for online fine-tuning of Decision Transformer.

collect_rollout → compute GAE → train for n_epochs → repeat
"""

import numpy as np
import torch
from .gae import compute_gae


class PPOTrainer:
    """
    Online PPO trainer that wraps a PPOAdapter (DT + value head) and
    interacts with the skeleton's Pacman environment.

    Usage:
        adapter = PPOAdapter(dt_model)
        optimizer = torch.optim.AdamW(dt_model.parameters(), lr=5e-5)
        trainer = PPOTrainer(adapter, env_config, optimizer, context_len=20)
        trainer.run(total_steps=500_000)
    """

    def __init__(self, adapter, env_config, optimizer,
                 context_len=20, n_steps=2048, n_epochs=4,
                 batch_size=64, gamma=0.99, lam=0.95,
                 max_grad_norm=0.5):
        self.adapter = adapter
        self.context_len = context_len
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer

        # Environment config — passed through to create new episodes
        self.env_config = env_config
        self.state_dim = env_config['state_dim']
        self.act_dim = env_config['act_dim']

    # -----------------------------------------------------------------
    #  Main loop
    # -----------------------------------------------------------------

    def run(self, total_steps):
        """Run PPO training for total_steps environment steps."""
        step = 0
        while step < total_steps:
            rollout = self.collect_rollout()
            step += len(rollout['rewards'])

            # Compute GAE
            rollout['advantages'], rollout['returns'] = compute_gae(
                rollout['rewards'], rollout['values'], rollout['dones'],
                self.gamma, self.lam)

            # Train
            for epoch in range(self.n_epochs):
                stats = self.train_epoch(rollout)

            yield rollout, stats

    # -----------------------------------------------------------------
    #  Rollout collection
    # -----------------------------------------------------------------

    def collect_rollout(self):
        """
        Run one rollout of n_steps, returning stored data.

        Returns dict with keys:
            states:      (n_steps, state_dim)
            actions:     (n_steps,)         int action ids
            rewards:     (n_steps,)
            log_probs:   (n_steps,)
            values:      (n_steps+1,)        includes bootstrap
            dones:       (n_steps,)
            rtg:         (n_steps,)          running return-to-go
            timesteps:   (n_steps,)          step index within episode
        """
        K = self.context_len
        D = self.state_dim
        A_dim = self.act_dim

        # Buffers
        states      = np.zeros((self.n_steps, D), dtype=np.float32)
        actions     = np.zeros(self.n_steps, dtype=np.int64)
        rewards     = np.zeros(self.n_steps, dtype=np.float32)
        log_probs   = np.zeros(self.n_steps, dtype=np.float32)
        values      = np.zeros(self.n_steps + 1, dtype=np.float32)
        dones       = np.zeros(self.n_steps, dtype=np.float32)
        rtg_arr     = np.zeros(self.n_steps, dtype=np.float32)
        timesteps   = np.zeros(self.n_steps, dtype=np.int64)

        # Episode state
        state = self._reset_env()
        rtg = self.env_config.get('target_rtg', 500.0)
        episode_step = 0
        history_states  = []
        history_actions = []
        history_rtgs    = []

        for t in range(self.n_steps):
            # --- Build context window ---
            feat = self._extract_features(state)

            history_states.append(feat)
            history_actions.append(np.zeros(A_dim, dtype=np.float32))  # placeholder
            history_rtgs.append(rtg)

            # Take last K steps
            ctx_s = np.array(history_states[-K:], dtype=np.float32)
            ctx_a = np.array(history_actions[-K:], dtype=np.float32)
            ctx_r = np.array(history_rtgs[-K:], dtype=np.float32)
            ctx_t = np.arange(len(ctx_s), dtype=np.int64)

            # Pad to full context_len
            if len(ctx_s) < K:
                pad = K - len(ctx_s)
                ctx_s = np.pad(ctx_s, ((pad, 0), (0, 0)))
                ctx_a = np.pad(ctx_a, ((pad, 0), (0, 0)))
                ctx_r = np.pad(ctx_r, ((pad, 0),))
                ctx_t = np.pad(ctx_t, ((pad, 0),))

            # --- Act ---
            with torch.no_grad():
                a_id, lp, v = self.adapter.act(
                    torch.FloatTensor(ctx_r).unsqueeze(0).unsqueeze(-1),
                    torch.FloatTensor(ctx_s).unsqueeze(0),
                    torch.FloatTensor(ctx_a).unsqueeze(0),
                    torch.LongTensor(ctx_t).unsqueeze(0),
                )

            # --- Step environment ---
            next_state, reward, done = self._step_env(state, a_id)

            # Fill in the actual action in history (for next step's context)
            one_hot = np.zeros(A_dim, dtype=np.float32)
            one_hot[a_id] = 1.0
            history_actions[-1] = one_hot

            # --- Store ---
            states[t]    = feat
            actions[t]   = a_id
            rewards[t]   = reward
            log_probs[t] = lp
            values[t]    = v
            dones[t]     = float(done)
            rtg_arr[t]   = rtg
            timesteps[t] = episode_step

            rtg -= reward
            episode_step += 1

            if done:
                state = self._reset_env()
                rtg = self.env_config.get('target_rtg', 500.0)
                episode_step = 0
                history_states.clear()
                history_actions.clear()
                history_rtgs.clear()
            else:
                state = next_state

        # Bootstrap value for last step
        with torch.no_grad():
            _, _, last_v = self.adapter.act(
                torch.FloatTensor(ctx_r).unsqueeze(0).unsqueeze(-1),
                torch.FloatTensor(ctx_s).unsqueeze(0),
                torch.FloatTensor(ctx_a).unsqueeze(0),
                torch.LongTensor(ctx_t).unsqueeze(0),
            )
            values[self.n_steps] = last_v if not done else 0.0

        return {
            'states':    states,
            'actions':   actions,
            'rewards':   rewards,
            'log_probs': log_probs,
            'values':    values,
            'dones':     dones,
            'rtg':       rtg_arr,
            'timesteps': timesteps,
            'advantages': None,   # filled in by run()
            'returns':    None,
        }

    # -----------------------------------------------------------------
    #  Training epoch
    # -----------------------------------------------------------------

    def train_epoch(self, rollout):
        """One pass through the rollout data with PPO updates."""
        n = len(rollout['rewards'])
        idxs = np.arange(n)
        np.random.shuffle(idxs)

        stats_sum = {'policy_loss': 0, 'value_loss': 0,
                     'entropy': 0, 'approx_kl': 0}
        n_batches = 0

        for start in range(0, n, self.batch_size):
            batch_idx = idxs[start:start + self.batch_size]
            if len(batch_idx) < 2:
                continue

            # Build context windows (B, K, ...) for all data
            (b_rtg, b_states, b_actions, b_tsteps,
             b_old_lp, b_adv, b_ret) = self._build_batch(rollout, batch_idx)

            loss, stats = self.adapter.evaluate(
                b_rtg, b_states, b_actions, b_tsteps,
                b_old_lp, b_adv, b_ret)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapter.dt.parameters(), self.max_grad_norm)
            self.optimizer.step()

            for k in stats_sum:
                stats_sum[k] += stats[k]
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in stats_sum.items()}

    # -----------------------------------------------------------------
    #  Helpers — override these for your specific environment
    # -----------------------------------------------------------------

    def _reset_env(self):
        """Create a new episode.  Returns initial game state object."""
        raise NotImplementedError

    def _step_env(self, state, action):
        """Execute action.  Returns (next_state, reward, done)."""
        raise NotImplementedError

    def _extract_features(self, state):
        """Game state → flat numpy vector (same as collect_expert_data)."""
        raise NotImplementedError

    def _build_batch(self, rollout, batch_idx):
        """
        For each step index in batch_idx, construct the context window
        of shape (B, K, ...) for all tensors needed by PPOAdapter.evaluate.

        Returns:
            rtg:     (B, K, 1)
            states:  (B, K, D)
            actions: (B, K, A)  one-hot
            tsteps:  (B, K)
            old_lp:  (B, K)     log π_old at each context position
            adv:     (B, K)     GAE advantages
            ret:     (B, K)     target returns
        Padded prefix positions are zeroed (masked by tsteps==0 in evaluate).
        """
        K = self.context_len
        B = len(batch_idx)
        D = self.state_dim
        A = self.act_dim

        b_rtg    = np.zeros((B, K, 1), dtype=np.float32)
        b_states = np.zeros((B, K, D), dtype=np.float32)
        b_acts   = np.zeros((B, K, A), dtype=np.float32)
        b_tsteps = np.zeros((B, K), dtype=np.int64)
        b_old_lp = np.zeros((B, K), dtype=np.float32)
        b_adv    = np.zeros((B, K), dtype=np.float32)
        b_ret    = np.zeros((B, K), dtype=np.float32)

        for i, t in enumerate(batch_idx):
            # Window: last K steps ending at t (inclusive)
            start = max(0, t - K + 1)
            length = min(t - start + 1, K)
            offset = K - length  # left-pad by offset

            sl = slice(offset, K)
            b_rtg[i, sl, 0] = rollout['rtg'][start:t + 1]
            b_states[i, sl]  = rollout['states'][start:t + 1]
            b_tsteps[i, sl]  = rollout['timesteps'][start:t + 1]
            b_old_lp[i, sl]  = rollout['log_probs'][start:t + 1]
            b_adv[i, sl]     = rollout['advantages'][start:t + 1]
            b_ret[i, sl]     = rollout['returns'][start:t + 1]

            for j, a_id in enumerate(rollout['actions'][start:t + 1]):
                b_acts[i, offset + j, a_id] = 1.0

        return (torch.FloatTensor(b_rtg),
                torch.FloatTensor(b_states),
                torch.FloatTensor(b_acts),
                torch.LongTensor(b_tsteps),
                torch.FloatTensor(b_old_lp),
                torch.FloatTensor(b_adv),
                torch.FloatTensor(b_ret))
