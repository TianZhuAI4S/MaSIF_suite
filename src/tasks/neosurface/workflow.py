"""NeoSurface reuses seed docking and owns already-aligned complex scoring."""

import numpy as np
from ..seed.workflow import SeedWorkflow


class NeoSurfaceWorkflow(SeedWorkflow):
    """Add original neosurf complex diagnostics to the shared seed search application."""

    def actions(self) -> dict:
        """Add already-aligned scoring while retaining retrieval and docking."""
        return dict(super().actions(), score=self.score_pairs)

    def score_pairs(self) -> list[dict]:
        """Score already aligned binders using the original neosurf nearest-center convention."""
        result = []
        for site, center in enumerate(self.centers):
            target = self.target.patch(center, self.shift, flipped=True)
            for surface in self.database:
                distance = np.sqrt(np.sum(np.square(surface.points - self.target.points[center]), axis=1))
                vertex = int(np.argmin(distance))
                source = surface.patch(vertex, self.shift)
                scored = self._score(target, source)
                # Raw descriptor arrays stay at their original dtype for these diagnostics.
                pair_distances = np.sqrt(np.sum(np.square(
                    self.target.points[target["indices"], None] - surface.points[None]), axis=-1))
                nearest = np.argmin(pair_distances, axis=1)
                average = np.mean(np.sqrt(np.sum(np.square(surface.descriptors[nearest] -
                           self.target.flipped_descriptors[target["indices"]]), axis=-1)))
                result.append(dict(key=f"{site}:{surface.name}:{vertex}", site=site, source=surface.name,
                                   target_vertex=center, source_vertex=vertex, accepted=True,
                                   center_desc_dist=float(np.sqrt(np.sum(np.square(surface.descriptors[vertex] -
                                                             self.target.flipped_descriptors[center])))),
                                   average_patch_desc_dist=float(average), transformation=np.eye(4),
                                   aligned_points=source["points"], **scored))
        return result
