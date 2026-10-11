# MaSIF tutorials

Choose a tutorial for the application you want to run. The notebooks use the same
commands as the [user guide](../USAGE.md) and expect a source checkout with its
`examples/` directory. Start with [MaSIF installation](../README.md#installation).

| Tutorial | What you will do |
|---|---|
| [MaSIF-site](site.ipynb) | Predict interaction sites from a prepared surface |
| [MaSIF-search](search.ipynb) | Compute surface fingerprints |
| [MaSIF-ligand](ligand.ipynb) | Classify a prepared binding pocket |
| [ScoreNN](score.ipynb) | Score prepared candidate interactions |
| [Alignment](alignment.ipynb) | Score prepared alignment features |
| [MaSIF-seed](seed.ipynb) | Retrieve and dock candidate binders |
| [MaSIF-neosurface](neosurface.ipynb) | Score an aligned protein–ligand target and binder |
| [Protein preparation](prepare_protein.ipynb) | Prepare a protein PDB for MaSIF |
| [Protein–ligand preparation](prepare_ligand.ipynb) | Prepare a ligand-containing target |
| [Rosetta](rosetta.ipynb) | Refine a positioned target–ligand–binder complex |
| [Training](training.ipynb) | Train, fine-tune and continue neural tasks |
| [Public benchmark](benchmark.ipynb) | Configure the optional full dataset workflow |

## Open a tutorial

From the repository root, activate the MaSIF environment and install Jupyter if
it is not already available:

```bash
conda activate masif_torch
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name masif_torch --display-name "Python (masif_torch)"
jupyter lab tutorials/
```

If you use venv, activate it instead of the Conda environment before these commands.
The kernel command registers that environment under the same display name.
The setup follows the official [JupyterLab installation](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
and [IPython kernel instructions](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

Open a notebook and select **Python (masif_torch)**. Run its cells from top to
bottom. The notebook finds the repository from its current directory; start
Jupyter inside this checkout.

## Use your own files

Each notebook has a settings cell for your input paths and output destination.
Model checkpoints are not bundled in the repository or wheel. Use the
[weight catalog](../weights/README.md) and
[weight conversion guide](../USAGE.md#convert-tensorflow-weights) to obtain them.

The default `RUN_TASK = False` lets you inspect the configured command without
starting a calculation. After supplying the required files and tools, change it
to `True` and rerun the settings and subsequent cells. Preparation and Rosetta
also require the external tools listed in their tutorials. The benchmark tutorial
configures the full public dataset workflow; it does not download or run it by default.

For terminal-only instructions, return to the [user guide](../USAGE.md).
