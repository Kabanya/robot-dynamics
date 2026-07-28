# Reproducibility artifact

This artifact supports the deliberately inconclusive ranking result in
`submission/ijhr/main.tex`. It does not claim calibrated risk, safety, hardware
transfer, or morphology transfer.

## Frozen evidence

- Development: two archived 100-row CSVs, seeds `12026` and `22026`.
- Confirmation: two untouched 400-row scrambles, seeds `42026` and `52026`;
  each contains 200 TALOS and 200 iCub rollouts.
- Six footsteps per rollout at 100 Hz.
- B4: gait/disturbance parameters plus binary robot identity.
- B5: the same inputs plus a normalized phase-agnostic whole-body signature.
- Primary endpoint: macro within-robot B5--B4 failure PR-AUC difference with
  10,000 paired resamples stratified by scramble and robot.

The observed difference is `+.034`, conditional 95% CI `[-.009, .073]`.
Both scrambles are positive, but the interval crosses zero, so the frozen gate
fails. Every shortlisted gait still requires the full rollout oracle.

## Inspect and regenerate

Run from the archive root in the locked environment:

```sh
python tests/test_confirmation.py
python tests/test_confirmation_analysis.py
python -m artifact.analyze_confirmation \
  --development-csv data/development_seed12026.csv \
  --development-csv data/development_seed22026.csv \
  --confirmation-dir data/confirmation_seed42026 \
  --confirmation-dir data/confirmation_seed52026 \
  --output-dir . \
  --figures-dir figures
```

The analysis command refits the two fixed HGB models, verifies counts, hashes,
and disjoint rollout seeds, writes `confirmation_analysis.json`, and
regenerates the two PDF/PNG figures without running new simulations.

The expensive confirmations can be repeated from the full repository checkout:

```sh
python -m artifact.run_confirmation --seed 42026 --workers 4
python -m artifact.run_confirmation --seed 52026 --workers 4
```

The runner refuses to overwrite an existing output directory. Per-case JSON
checkpoints allow recovery from interruption without changing the protocol.

## Video

`video/rollout_examples.mp4` contains one successful and one failed closed-loop
rollout for TALOS and iCub. The labels identify the robot and first failure
type. The browser-free renderer draws the archived Pinocchio configurations
with deterministic simplified per-link hulls; this display geometry is not
used by the oracle or analysis. To regenerate it with Matplotlib and a system
FFmpeg executable:

```sh
python -m artifact.generate_rollout_video \
  --confirmation-dir data/confirmation_seed42026
```

## Licenses and assets

Repository code is BSD-3-Clause. Generated tabular data and figures are
CC BY 4.0. TALOS/iCub descriptions and meshes are not included; Pinocchio loads
them from the separately installed `example-robot-data` package under its own
terms. See `third_party_notices.txt`.
