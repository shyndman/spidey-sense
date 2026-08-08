"""Perceptual duplicate grouping for acquired samples."""

from __future__ import annotations

from collections.abc import Sequence

from .hashing import hamming_distance
from .models import AcceptedSample


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], left: int, right: int) -> None:
    root_left, root_right = _find(parent, left), _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def _mark_near(
    parent: list[int],
    tree: list[tuple[int, dict[int, int]]],
    target: int,
    item_index: int,
) -> None:
    pending = [0]
    while pending:
        current = pending.pop()
        node_hash, children = tree[current]
        distance = hamming_distance(node_hash, target)
        if distance <= 4:
            _union(parent, item_index, current)
        pending.extend(
            child
            for edge, child in children.items()
            if max(0, distance - 4) <= edge <= distance + 4
        )


def ordered_groups(
    items: Sequence[AcceptedSample],
) -> tuple[tuple[AcceptedSample, ...], ...]:
    """Union near hashes and return stable groups in sample-id order."""

    parent = list(range(len(items)))
    tree: list[tuple[int, dict[int, int]]] = []
    for index, item in enumerate(items):
        if tree:
            _mark_near(parent, tree, item.perceptual_hash, index)
        if not tree:
            tree.append((item.perceptual_hash, {}))
            continue
        node_index = 0
        while True:
            node_hash, children = tree[node_index]
            distance = hamming_distance(node_hash, item.perceptual_hash)
            child = children.get(distance)
            if child is None:
                children[distance] = len(tree)
                tree.append((item.perceptual_hash, {}))
                break
            node_index = child
    groups: dict[int, list[AcceptedSample]] = {}
    for index, item in enumerate(items):
        groups.setdefault(_find(parent, index), []).append(item)
    return tuple(
        tuple(group)
        for group in sorted(
            groups.values(), key=lambda group: min(item.sample_id for item in group)
        )
    )


__all__ = ["ordered_groups"]
