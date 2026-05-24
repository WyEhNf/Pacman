#!/usr/bin/env python3
"""Launch web metrics dashboard for a training run."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gui.dashboard import create_dashboard


def main():
    parser = argparse.ArgumentParser(description="Launch Pac-Man training metrics dashboard")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to run directory containing metrics.db")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port for web dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind to")
    args = parser.parse_args()

    db_path = Path(args.run_dir) / "metrics.db"
    if not db_path.exists():
        print(f"Warning: {db_path} not found yet. Dashboard will wait for data.")

    app = create_dashboard(str(db_path))
    print(f"Dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
