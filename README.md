# Neural Network Circuit Game
Neuroevolution simulation where a population of AI-controlled cars learns to drive a custom circuit through a genetic algorithm.

[![Download](https://img.shields.io/github/v/release/luangrezende/neural-network-circuit-game?label=Download%20Windows%20EXE&style=for-the-badge&logo=github&color=2ea043)](https://github.com/luangrezende/neural-network-circuit-game/releases/latest)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.4%2B-blue?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Each generation, a population of cars drives the circuit simultaneously. Cars are controlled by a small feedforward neural network whose weights are evolved via selection, crossover, and mutation — no backpropagation involved. The simulation runs in real time with Matplotlib and includes an interactive track editor to design new circuits.

---

## Architecture

The codebase is split into two clear layers:

- **Simulation core** (`sim/track.py`, `sim/simulacao.py`, `sim/neural_network.py`) — pure Python/NumPy with no rendering dependencies. `SimuladorBase` owns the genetic loop.
- **Visualization layer** (`sim/visualizacao.py`) — `SimuladorAprendizado` extends `SimuladorBase` with Matplotlib real-time rendering via blit.

This separation makes it straightforward to run headless training or swap the renderer.

---

## Tech Stack

- Python 3.11+
- NumPy — car physics, SDF collision, batched neural network forward pass
- Matplotlib — real-time simulation rendering, interactive track editor
- PyTorch (optional, listed in requirements for future experiments)
- tkinter — clipboard support in the track editor only

---

## Getting Started

### Option 1 — Download the pre-built executable (Windows)

Grab the latest `.zip` from the [Releases page](https://github.com/luangrezende/neural-network-circuit-game/releases/latest), extract it, and run `neural-network-circuit-game.exe`. No Python required.

> The executable is built automatically by GitHub Actions on every version tag using PyInstaller.

### Option 2 — Run from source

#### Prerequisites

- Python 3.11+
- A CUDA-capable GPU is **not required** (PyTorch is listed for optional use)

#### Installation

```bash
git clone https://github.com/luangrezende/neural-network-circuit-game.git
cd neural-network-circuit-game
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> If you do not need PyTorch, remove the `torch`/`torchvision`/`torchaudio` lines from `requirements.txt` and run `pip install numpy matplotlib`.

#### Run the simulation

```bash
python rede_neural_jogo.py
```

#### Run the track editor

```bash
python editor_pista.py
```

Click to place control points, adjust track width, set the start/finish line, and export to `pista.json`. The simulator picks up the new track on the next run.

---

## Configuration

All simulation hyperparameters live in `config.json`:

| Section | Key parameters |
|---|---|
| `carros` | max speed, turn angle, car size |
| `simulacao` | population size, target laps, top survivors, random newcomers per generation |
| `recompensas` | weights for centerline adherence, speed, lap progress |
| `penalidades` | collision penalty, wrong-way penalty, slow-car penalty |
| `visualizacao` | sensor overlay, display options |

---

## Project Structure

```
.
├── rede_neural_jogo.py   # entry point
├── editor_pista.py       # interactive track editor
├── config.json           # simulation hyperparameters
├── pista.json            # active track (control points + width)
├── requirements.txt
└── sim/
    ├── neural_network.py # feedforward net + genetic operators
    ├── simulacao.py      # genetic algorithm loop (no UI)
    ├── track.py          # track geometry, SDF, car physics
    └── visualizacao.py   # matplotlib rendering layer
```

---

## Neural Network

Each car is controlled by a fully connected network:

```
8 inputs (7 distance sensors + speed)
        ↓
   14 hidden units  (sigmoid)
        ↓
 2 outputs: steer (-1..+1), throttle (0..1)  (sigmoid)
```

Weights are evolved each generation via uniform crossover and Gaussian mutation. No gradient descent.