from __future__ import annotations

FEATURE_REGISTRY = {
    "dates": {"module": "sbgm.evaluate2.features.dates.feature", "class": "DatesFeature"},
    "distributions": {"module": "sbgm.evaluate2.features.distributions.feature", "class": "DistributionsFeature"},
    "extremes": {"module": "sbgm.evaluate2.features.extremes.feature", "class": "ExtremesFeature"},
    "probabilistic": {"module": "sbgm.evaluate2.features.probabilistic.feature", "class": "ProbabilisticFeature"},
    "scale": {"module": "sbgm.evaluate2.features.scale.feature", "class": "ScaleFeature"},
    "spatial": {"module": "sbgm.evaluate2.features.spatial.feature", "class": "SpatialFeature"},
    "temporal": {"module": "sbgm.evaluate2.features.temporal.feature", "class": "TemporalFeature"},
    "sal": {"module": "sbgm.evaluate2.features.sal.feature", "class": "SALFeature"},
}