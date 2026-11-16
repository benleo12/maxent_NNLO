"""Sparse-cell merging must preserve the physical partition and its probabilities."""

import numpy as np

from examples.zj_refinement import merged_groups


def test_sparse_merging_preserves_connected_physical_cells_and_target_mass():
    b = 5
    counts = np.zeros(b * b, dtype=int)
    counts[[0, 5, 6, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23, 24]] = [
        300,
        90,
        70,
        45,
        35,
        20,
        15,
        10,
        8,
        5,
        1,
        0,
        2,
        0,
        0,
    ]
    mapping, groups = merged_groups(counts, b)
    physical = {i * b + j for i in range(b) for j in range(i + 1)}
    assert sorted(cell for group in groups for cell in group) == sorted(physical)
    assert np.all(mapping[list(set(range(b * b)) - physical)] == -1)
    target = np.zeros(b * b)
    target[sorted(physical)] = np.arange(1, len(physical) + 1)
    target /= target.sum()
    prior = counts / counts.sum()
    corrected = np.zeros_like(prior)
    for group in groups:
        assert counts[group].sum() >= 25
        # The resulting piecewise correction reproduces every merged target,
        # including probability assigned to originally empty physical cells.
        corrected[group] = prior[group] * target[group].sum() / prior[group].sum()
        np.testing.assert_allclose(corrected[group].sum(), target[group].sum())
        reached = {group[0]}
        while True:
            neighbours = {
                cell
                for cell in group
                if any(
                    abs(cell // b - other // b) + abs(cell % b - other % b) == 1
                    for other in reached
                )
            }
            new = reached | neighbours
            if new == reached:
                break
            reached = new
        assert reached == set(group)
    np.testing.assert_allclose(corrected.sum(), 1)
    np.testing.assert_array_equal(merged_groups(counts, b)[0], mapping)


def test_populated_cells_are_left_separate():
    b = 4
    counts = np.zeros(b * b, dtype=int)
    physical = [i * b + j for i in range(b) for j in range(i + 1)]
    counts[physical] = 25
    _, groups = merged_groups(counts, b)
    assert groups == [[cell] for cell in physical]
