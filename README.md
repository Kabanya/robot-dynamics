# Whole-Body Physics Signatures for Humanoid Gait Failure Ranking

Code, data, manuscript, and visual material for a reproducible simulation study
of offline humanoid gait failure ranking on TALOS and iCub.

## Result

Fixed histogram-gradient-boosting models were trained on 200 development
rollouts and evaluated without refitting on two independent 400-rollout
scrambled-Sobol confirmations.

- Whole-body model (B5): pooled failure PR-AUC **0.725**
- Parameter-plus-robot comparator (B4): pooled failure PR-AUC **0.689**
- Prespecified macro within-robot difference: **+0.034**
- Conditional 95% bootstrap interval: **[-0.009, 0.073]**

Both confirmation scrambles have a positive point estimate, but the interval
crosses zero. The confirmatory gate therefore fails. The repository reports a
modest but inconclusive ranking signal—not calibrated risk control, morphology
transfer, or a safety guarantee. Every shortlisted gait still requires a full
rollout verification.

## Reproduce the reported analysis

The packaged analysis reuses the frozen CSV records and does not run new
simulations. From the repository root:

```sh
conda env create -f environment.yml
conda activate pin-env
unzip submission/ijhr/reproducibility_artifact.zip -d reproduced
cd reproduced
python tests/test_confirmation.py
python tests/test_confirmation_analysis.py
python -m artifact.analyze_confirmation \
  --development-csv data/development_seed12026.csv \
  --development-csv data/development_seed22026.csv \
  --confirmation-dir data/confirmation_seed42026 \
  --confirmation-dir data/confirmation_seed52026 \
  --output-dir regenerated \
  --figures-dir regenerated/figures
```

The regenerated `confirmation_analysis.json` and both manuscript figures are
deterministic.

## Visual walkthrough

- [`gait_feasibility.ipynb`](gait_feasibility.ipynb) explains the
  robot-normalized trajectory, phase physics signature, independent rollout
  oracle, and final confirmation result.
- [`rollout_examples.mp4`](submission/ijhr/rollout_examples.mp4) shows
  representative successful and failed TALOS/iCub rollouts.
- [`main.pdf`](submission/ijhr/main.pdf) is the current manuscript.

The notebook's Meshcat viewers replay stored closed-loop configurations. Keep
the notebook kernel running while viewing them.

## Optional full simulation

Repeating either 400-rollout confirmation is expensive and is not required to
verify the reported analysis:

```sh
python -m artifact.run_confirmation --seed 42026 --workers 4
python -m artifact.run_confirmation --seed 52026 --workers 4
```

The runner refuses to overwrite an existing output directory and stores
per-case checkpoints for safe resumption.

## Repository layout

- `src/` — trajectory, whole-body signature, rollout oracle, and experiment code
- `artifact/` — fixed analysis, packaging, figure, and video tools
- `tests/` — validity and reproducibility checks
- `submission/ijhr/` — manuscript and journal submission files
- `gait_feasibility.ipynb` — reader-facing method and result walkthrough

Public repository:
<https://github.com/Kabanya/robot-dynamics>

## Licenses

Source code is BSD-3-Clause. Generated tabular data and figures are CC BY 4.0.
TALOS and iCub descriptions and meshes are loaded from
`example-robot-data` and are not redistributed.
