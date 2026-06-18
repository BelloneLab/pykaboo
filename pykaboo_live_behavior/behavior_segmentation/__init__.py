"""Supervised temporal behavior segmentation from identity-tracked mouse masks.

This package converts RF-DETR instance masks stored in COCO JSON into temporal
behavior embeddings, trains a supervised framewise segmentation network, runs
inference, exports embeddings for clustering, and ships a desktop GUI for
labeling, inference, evaluation, and correction.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
