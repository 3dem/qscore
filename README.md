This is a reimplementation of Pintilie et al., 2020 [1] using pure Python. It is also faster since it does not use Chimera and tries to vectorize as many things as possible. Please check the implementation to make sure the results are correct.
To use this, you can run the `calculate_per_residue_q_scores` function in `q_score.py`

## Install

From the repository root, create and activate a virtual environment, then install
the runtime dependencies and the `qscore` CLI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install (`-e .`) makes the `qscore` command available while still
using the source files in this checkout. For a non-editable install, use
`python -m pip install .` instead.

## Run the CLI

Check that the command is installed:

```bash
qscore --help
qscore --version
```

Calculate per-residue Q-scores from a structure file and cryo-EM map:

```bash
qscore calculate \
  --structure-path path/to/structure.pdb \
  --volume-path path/to/map.mrc \
  --output-path output.csv
```

The structure file can be PDB or mmCIF. The volume file should be an MRC file.
`--output-path` is optional and defaults to `output.csv`.

1 - Pintilie, G., Zhang, K., Su, Z. et al. Measurement of atom resolvability in cryo-EM maps with Q-scores. Nat Methods 17, 328–334 (2020). https://doi.org/10.1038/s41592-020-0731-1
