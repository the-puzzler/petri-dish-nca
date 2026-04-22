# PD-NCA with Autoencoder Competition
**Extension of Petri Dish Neural Cellular Automata (Sakana AI, 2025)**

---

## Overview

This project extends PD-NCA by replacing the cosine similarity competition mechanism with a dataset-grounded autoencoder competition. Agents survive by learning private compression schemes of real data embedded in the substrate. Predation becomes the act of cracking a competitor's encoding.

The original codebase is at https://github.com/SakanaAI/petri-dish-nca. All changes are localized to: (1) the environment tensor, (2) the agent architecture, (3) the competition formula.

---

## Original PD-NCA (brief recap)

The grid state at each cell is $\mathbf{s}_{x,y}^t = [\mathbf{a}, \mathbf{d}, \mathbf{h}] \in \mathbb{R}^C$ where $\mathbf{a}$ is the attack channel, $\mathbf{d}$ is the defense channel, $\mathbf{h}$ is hidden state. Each agent $i$ is a convolutional network $f_{\theta_i}$ producing local state updates. Competition between agents is:

$$\phi_{ij}(x,y) = \langle \mathbf{a}^i, \mathbf{d}^j \rangle - \langle \mathbf{d}^i, \mathbf{a}^j \rangle$$

The background environment $\mathbf{E}_{x,y}$ is fixed random noise, normalized. Agents must beat the environment to hold territory, and beat each other to expand. Softmax over competitive strengths determines territorial contribution weights. Agent loss is negative log-aliveness.

---

## Extension: Three Changes

### 1. Environment Tensor — embed real data

Replace random $\mathbf{E}$ with data-derived embeddings. Given a dataset $\mathcal{D} = \{(\mathbf{X}_k, \mathbf{Y}_k)\}$, assign one sample per cell:

$$\mathbf{e}_{x,y} = \text{normalize}([\phi_x(\mathbf{X}_{x,y}),\ \phi_y(\mathbf{Y}_{x,y})]) \in \mathbb{R}^{C_e}$$

where $\phi_x$, $\phi_y$ are fixed encoders (e.g. frozen CNN for images, linear projection for scalars). $\mathbf{e}_{x,y}$ is read-only — agents observe it via their neighborhood $\mathcal{N}_{x,y}$ but never write to it. Data is tiled or randomly sampled across the grid. $C_e$ should match $C_a + C_d$ for compatibility.

For simple tasks (e.g. sine regression): $\phi_x(x) = \text{normalize}([\cos(\pi x), \sin(\pi x)])$, $\phi_y(y) = \text{normalize}([\cos(\pi y), \sin(\pi y)])$.

For classification (e.g. MNIST): $\phi_x$ = frozen CNN embedding, $\phi_y(c) = \mathbf{e}_c$ (one-hot, $K$-dim).

### 2. Agent Architecture — explicit autoencoder

Replace the single black-box $f_{\theta_i}$ with an encoder-decoder pair sharing parameters $\theta_i$:

$$\mathbf{a}^i_{x,y} = \text{Enc}_{\theta_i}(\mathbf{e}_{x,y},\ \mathbf{h}^i_{x,y}) \in \mathbb{R}^{C_a}$$
$$\mathbf{d}^i_{x,y} = \text{Dec}_{\theta_i}(\mathbf{a}^i_{x,y}) \in \mathbb{R}^{C_e}$$

$\mathbf{a}^i$ is the latent code of the local data. $\mathbf{d}^i$ is the reconstruction. The hidden state $\mathbf{h}^i$ is aggregated from the Moore neighborhood as in the original, providing spatial context to the encoder.

Concretely: $\text{Enc}$ is a small MLP (2-3 layers) taking $[\mathbf{e}_{x,y}, \mathbf{h}^i_{x,y}]$ as input. $\text{Dec}$ is a symmetric MLP taking $\mathbf{a}^i$ back to $\mathbb{R}^{C_e}$. The convolutional neighborhood aggregation from the original is preserved for computing $\mathbf{h}$.

### 3. Competition Formula — cross-decoding reconstruction error

Replace cosine similarity with cross-reconstruction quality. Agent $i$ beats agent $j$ if $i$'s decoder can reconstruct the local data from $j$'s code better than $j$'s decoder can reconstruct it from $i$'s code:

$$\phi_{ij}(x,y) = \underbrace{-\|\text{Dec}_{\theta_i}(\mathbf{a}^j_{x,y}) - \mathbf{e}_{x,y}\|^2}_{\text{i cracks j's code}} + \underbrace{\|\text{Dec}_{\theta_j}(\mathbf{a}^i_{x,y}) - \mathbf{e}_{x,y}\|^2}_{\text{j fails to crack i's code}}$$

Antisymmetry is preserved: $\phi_{ij} = -\phi_{ji}$. ✓

Agent vs environment:

$$\phi_{i,\text{env}}(x,y) = -\|\text{Dec}_{\theta_i}(\mathbf{a}^i_{x,y}) - \mathbf{e}_{x,y}\|^2$$

Beating the environment requires genuine reconstruction of the local data sample.

Everything else — normalization, softmax with temperature $\tau$, aliveness threshold $\alpha$, state update, loss function — is unchanged from the original.

---

## Competitive Dynamics

Three pressures are now in genuine conflict:

- **Beat the environment**: compress $\mathbf{e}$ faithfully — make $\mathbf{a}^i$ informative and $\text{Dec}_{\theta_i}(\mathbf{a}^i) \approx \mathbf{e}$
- **Resist predation**: make $\mathbf{a}^i$ hard for others to decode — idiosyncratic encoding
- **Predate**: learn to invert competitors' encodings — understand their compression scheme

The stable strategy is *private compression*: a scheme good enough to reconstruct the data but opaque enough that competitors cannot invert it. This cannot reduce to the original cosine formula — it requires agents to maintain decoders that generalize across the grid.

---

## Expected Emergent Behavior

Because $\mathbf{e}_{x,y}$ varies across the grid according to the data distribution, different regions of the grid present different compression problems. Agents are expected to spatially specialize — dominating regions whose data structure matches their compression scheme — producing emergent data partitioning without any explicit supervision signal. This is a living mixture-of-experts where territorial boundaries reflect learned data structure.

---

## Implementation Notes

- **Cross-decoding cost**: computing $\text{Dec}_{\theta_i}(\mathbf{a}^j)$ requires running $N$ decoder forward passes per cell during competition. Keep decoders shallow (1-2 layers). Since decoders are small and the operation is embarrassingly parallel across cells, this is tractable on GPU.
- **Normalization**: normalize $\mathbf{a}^i$ after encoding so reconstruction errors are comparable in scale across agents. Clip $\mathbf{d}^i$ to $[-1, 1]$ as in the original.
- **Data grid**: for a $64 \times 64$ grid, tile the dataset by random spatial assignment at initialization. Keep the assignment fixed throughout the simulation.
- **$C_a$ sizing**: for sine regression, $C_a = 4$ is sufficient. For 10-class classification, $C_a = 16$–$32$ with $\phi_y$ as learned (but frozen after init) class embeddings rather than one-hot, to allow partial credit geometry.

---

## Suggested First Experiment

**Dataset**: sine function $y = \sin(x)$, $x \in [-\pi, \pi]$, sampled densely.

**Grid**: $64 \times 64$, data tiled by $x$ value left-to-right.

**Agents**: 6 NCA, same hyperparameters as original paper.

**Hypothesis**: agents will partition the grid by frequency region, with territorial boundaries forming at inflection points of $\sin(x)$ where compression difficulty changes. Cyclic dynamics expected at boundaries.

**Metric**: measure reconstruction MSE per agent per cell over time, alongside the original entropy and open-endedness scores.

---

## Citation for Base System

```
Ivy Zhang and Sebastian Risi and Luke Darlow, "Petri Dish NCA", 2025.
https://pub.sakana.ai/pdnca/
```