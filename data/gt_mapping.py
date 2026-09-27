"""Apply RoboFAC GT mapping onto Episode.ground_truth (idempotent)."""

from __future__ import annotations

from data.schemas.episode import Episode, GroundTruth
from ontology.robofac_type_map import map_robofac_failure_type, map_robofac_success


def map_ground_truth(gt: GroundTruth) -> GroundTruth:
    """Return a new GroundTruth with success/type remapped."""
    return gt.model_copy(
        update={
            "success": map_robofac_success(gt.success),
            "failure_type": map_robofac_failure_type(gt.failure_type),
        }
    )


def apply_gt_mapping(episode: Episode) -> Episode:
    """Return a new Episode with ground_truth success/type remapped."""
    return episode.model_copy(update={"ground_truth": map_ground_truth(episode.ground_truth)})
