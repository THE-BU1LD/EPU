# Logisim Blueprint

## Subcircuits
- STATE_REG
- DOT4
- SCORE_BANK
- NORM_UNIT
- RECON_UNIT
- RESIDUAL_MIXER
- CONVERGENCE_CHECK
- CONTROL_FSM

## Build order
1. Build STATE_REG.
2. Build DOT4.
3. Copy DOT4 into SCORE_BANK.
4. Build NORM_UNIT.
5. Build RECON_UNIT.
6. Add RESIDUAL_MIXER.
7. Add CONVERGENCE_CHECK.
8. Wrap in CONTROL_FSM.

## Recommended widths
- State coordinate: 8-bit signed
- Dot-product accumulator: 16-bit signed
- Weight scale: 0..1000

## Wiring rule
Keep every block in a separate subcircuit. Do not wire the full machine as a single flat canvas.
