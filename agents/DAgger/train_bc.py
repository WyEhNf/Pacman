"""
Behavioral Cloning training script for Decision Transformer.

Loads expert trajectories, trains DT to predict actions via cross-entropy,
and saves checkpoints.

Usage:
    cd e:\Pacman
    python scripts/train_bc.py --episodes 500  # collect data first
    python scripts/train_bc.py --train          # train
"""

import sys, os, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.data.dataset import TrajectoryDataset
from src.model.decision_transformer import DecisionTransformer


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Dataset ----
    ds = TrajectoryDataset(
        args.data,
        context_len=args.context_len,
        state_dim=args.state_dim,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True)

    print(f"State dim: {ds.state_dim}, RtG range: [{ds.rtg_min:.0f}, {ds.rtg_max:.0f}]")

    # ---- Model ----
    model = DecisionTransformer(
        state_dim=ds.state_dim,
        act_dim=5,  # UP/DOWN/LEFT/RIGHT/STOP
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_len=args.context_len,
        dropout=args.dropout,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer & LR schedule ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- Training loop ----
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        losses, accs = [], []

        for rtg, states, actions, mask in loader:
            rtg, states, actions, mask = (
                rtg.to(device), states.to(device),
                actions.to(device), mask.to(device))

            # Pack into DT input format — no timesteps for now, use zeros
            B = rtg.shape[0]
            timesteps = torch.zeros(B, args.context_len, dtype=torch.long, device=device)

            # Forward
            action_logits, _, _ = model(rtg, states, actions, timesteps)
            # action_logits: (B, K, 5), actions: (B, K)

            # Loss — only on real (non-padded) steps
            loss = criterion(
                action_logits[mask.bool()].view(-1, 5),
                actions[mask.bool()].view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy
            with torch.no_grad():
                preds = action_logits[mask.bool()].argmax(dim=-1)
                acc = (preds == actions[mask.bool()]).float().mean()

            losses.append(loss.item())
            accs.append(acc.item())
            global_step += 1

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
        print(f"Epoch {epoch+1:3d}/{args.epochs}  |  loss={avg_loss:.4f}  "
              f"acc={avg_acc:.3f}")

        # Checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(args.checkpoint_dir, f"dt_bc_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'state_dim': ds.state_dim,
                'rtg_min': ds.rtg_min,
                'rtg_max': ds.rtg_max,
            }, ckpt)
            print(f"  Saved: {ckpt}")

    # Final checkpoint
    final = os.path.join(args.checkpoint_dir, "dt_bc_final.pt")
    torch.save(model.state_dict(), final)
    print(f"\nFinal model: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/expert_trajectories.npz')
    parser.add_argument('--state_dim', type=int, default=None,
                        help='Force state dim (auto-detect if None)')
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
