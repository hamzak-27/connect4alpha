# connect4-alphazero

An agent that learns Connect Four only by playing against itself, built step by step.
See [ROADMAP.md](ROADMAP.md) for the plan and progress.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/play.py          # play against a random opponent
.venv/Scripts/python scripts/play.py --demo   # statistics from random games
```
