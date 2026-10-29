"""Модели одноклассового поиска новизны."""

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.centroid import CentroidDistance

__all__ = ["CentroidDistance", "NoveltyDetector"]
