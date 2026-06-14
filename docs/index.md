# NRM-Newton Documentation

This folder is the deep reference for the project. The top-level
[`../README.md`](../README.md) covers install and quickstart; everything that explains *how the pipeline works and how to configure it* lives here.

## Index

| Document | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | Repository layout, the end-to-end pipeline (`main.py`), data flow, the paper↔code map, and output artifacts (CSV schema). |
| [optimization.md](optimization.md) | Deep walkthrough of the morphology optimizer (`optim/nrm_alpha_random_selection.py`): the differentiable preprocessing, batched optimization, early stopping, the selection cascade, and the tiebreak heuristic. |
| [validation.md](validation.md) | IK/FK validation during optimization, final cuRobo motion planning, and the viser visualization. |
| [configuration.md](configuration.md) | Reference for every `config.json` key|
| [troubleshooting.md](troubleshooting.md) | Common failure points and known issues. |
