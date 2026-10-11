"""Compatibility exports and model factory; implementations belong to task packages."""

from .shared.layers import GaussianConv, PatchEncoder, KerasBatchNorm1d, linear, initialize_layer
from .tasks.site.model import MaSIF_site
from .tasks.search.model import MaSIF_ppi_search
from .tasks.ligand.model import MaSIF_ligand
from .tasks.score.model import ScoreNN
from .tasks.alignment.model import AlignmentEvaluationNN


def build_model(config):
    """Construct a task model from the primitive configuration saved in checkpoints."""
    config = dict(config)
    task = config.pop("task")
    classes = dict(site=MaSIF_site, ppi_search=MaSIF_ppi_search,
                   ligand=MaSIF_ligand, score=ScoreNN, alignment=AlignmentEvaluationNN)
    if task not in classes:
        raise ValueError(f"Unknown MaSIF task: {task}")
    return classes[task](**config)
