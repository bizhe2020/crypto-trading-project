# Workflow Wrappers

These wrappers are the stable human-facing entrypoints.

- `live/`: local live bot and Tokyo deployment
- `replay/`: replay and convergence audit
- `research/`: parameter search and candidate generation

The wrappers call the underlying scripts in `scripts/` and keep the current canonical defaults in one place.
