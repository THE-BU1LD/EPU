# EPU-Ω∞: A Hopfield-Inspired Engram Processing Unit

## Abstract
EPU-Ω∞ is a memory-driven compute architecture in which a latent state is repeatedly refined by associative retrieval from a memory bank. The design is inspired by Hopfield-style attractor dynamics, but it is framed as a practical digital system: first a Logisim prototype, then a fixed-point HDL implementation, and later an FPGA-ready architecture.

## Core update
\[
s_i = \beta \cdot (z_t \cdot m_i)
\]

\[
w_i = \mathrm{Normalize}(s_1,\dots,s_K)
\]

\[
r_t = \sum_i w_i m_i
\]

\[
z_{t+1} = \alpha z_t + (1-\alpha) r_t + \gamma C(z_t)
\]

## Hardware modules
- STATE_REG
- DOT_UNIT
- SCORE_BANK
- NORM_UNIT
- RECON_UNIT
- RESIDUAL_MIXER
- CONSTRAINT_UNIT
- ENERGY_MONITOR
- CONVERGENCE_CHECK
- CONTROL_FSM

## ISA
- LOAD v
- STEP
- RUN n
- HALT_IF_CONV
- STORE_FAST
- CONSOLIDATE
- SET_BETA b
- SET_ALPHA a
- SET_GAMMA g
- SYNC

## Evaluation
Measure:
- final similarity
- steps to convergence
- weight entropy
- noise tolerance
- capacity interference
- multi-core consensus distance

## Conclusion
The point of the architecture is not to imitate a conventional CPU. The point is to turn memory, convergence, and damping into primitive operations of computation.
