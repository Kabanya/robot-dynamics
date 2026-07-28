# International Journal of Humanoid Robotics submission package

This directory contains the current manuscript and the editable files needed
for an initial submission to the *International Journal of Humanoid Robotics*.
The manuscript includes the author's name and affiliation.

## Build

Run from this directory with TeX Live or MacTeX:

```sh
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error cover_letter.tex
```

## Manuscript files

- `main.tex`, `references.bib`, and `figures/*.pdf` — editable manuscript source
- `main.pdf` — generated manuscript
- `cover_letter.tex` and `cover_letter.pdf` — editable and generated cover letter
- `figure_captions.txt` — figure captions for the submission form
- `author_biography.txt` — editable biography if requested by the portal
- `submission_metadata.md` — fields and final private checks

## Supplementary files

- `reproducibility_artifact.zip` — code, frozen data, fixed analysis, and figures
- `rollout_examples.mp4` — representative successful and failed rollouts
- `rollout_video_still.png` — still image for the supplementary video
- `supplementary_captions.txt` — descriptive captions for both files

The manuscript cites the supplementary artifact, video, and public repository.
The repository URL must be live before the journal submission is finalized:
<https://github.com/Kabanya/robot-dynamics>.

Use the standard publication route rather than optional paid open access.
Enter any phone number requested by Editorial Manager privately; do not commit
it to this public repository.
