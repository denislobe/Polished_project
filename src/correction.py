import numpy as np
import copy
from sklearn.decomposition import PCA
from src.io import *


def fit_line_pca(points):
    centroid = points.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(points)
    direction = pca.components_[0]
    direction = direction / np.linalg.norm(direction)
    return centroid, direction



def point_line_dist(points, point_on_line, direction):
    diff = points - point_on_line
    return np.linalg.norm(np.cross(diff, direction), axis=1)



def ransac_line_fit(axis_data, n_iter=2000, threshold=0.05, min_inliers=None):
    data_xyz = np.array(axis_data.xyz).transpose()[0, :, :]

    n = len(data_xyz)
    best_inliers = None
    best_count = 0

    for _ in range(n_iter):
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = data_xyz[idx]

        direction = p2 - p1
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            continue
        direction = direction / norm

        dist = point_line_dist(data_xyz, p1, direction)
        inliers = dist < threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or (min_inliers is not None and best_count < min_inliers):
        raise ValueError("No good line found. Try increasing threshold or n_iter.")

    inlier_points = data_xyz[best_inliers]
    centroid, direction = fit_line_pca(inlier_points)

    t = np.linspace(-0.8, 0.8, 60)
    line = centroid + t[:, None] * direction

    return centroid, direction, inlier_points, best_inliers, line



def expected_t_from_local_steps(inline_idx, line_points, query_idx):

    inline_idx = np.asarray(inline_idx)
    line_points = np.asarray(line_points)
    query_idx = np.asarray(query_idx)

    if len(inline_idx) < 2:
        raise ValueError("Need at least two inline points.")

    idx_diff = np.diff(inline_idx)
    line_diff = np.diff(line_points)
    steps = line_diff / idx_diff   # local step per index in each interval

    t_expected = np.empty(len(query_idx), dtype=float)

    for n, q in enumerate(query_idx):
        pos = np.searchsorted(inline_idx, q)

        # Case 1: q is before the first inline point
        if pos == 0:
            step = steps[0]
            t_expected[n] = line_points[0] + step * (q - inline_idx[0])

        # Case 2: q is after the last inline point
        elif pos == len(inline_idx):
            step = steps[-1]
            t_expected[n] = line_points[-1] + step * (q - inline_idx[-1])

        # Case 3: q lands exactly on an inline point
        elif inline_idx[pos] == q:
            t_expected[n] = line_points[pos]

        # Case 4: q lies between two inline points
        else:
            left_idx = inline_idx[pos - 1]
            t_left = line_points[pos - 1]
            step = steps[pos - 1]
            t_expected[n] = t_left + step * (q - left_idx)

    return t_expected



def choose_candidates_smooth(
    candidates,
    centroid,
    direction,
    inliers,
    inlier_mask,
):
    data = np.array(candidates.xyz).transpose(1, 0, 2)
    inlier_idx = np.where(inlier_mask)[0]

    if len(inlier_idx) < 2:
        raise ValueError("Need at least two inliers to estimate expected positions.")

    # t-values of trusted inlier points
    line_points = np.dot(inliers - centroid, direction)

    # expected t for every sequence index
    all_idx = np.arange(data.shape[0])
    t_expected = expected_t_from_local_steps(inlier_idx, line_points, all_idx)

    expected_points = centroid + np.outer(t_expected, direction)

    best_match = np.argmin(np.linalg.norm(data[~inlier_mask, :, :] - expected_points[~inlier_mask, :, None], axis=1), axis=1)
    
    return best_match



def correction_algorithm(results_path, sim, n_iter=2000, threshold=0.05, min_inliers=None):
    results = load_results(results_path, sim)
    axis_data = convert_best_match_to_orientations(results).to_axes_angles()

    centroid, direction, inliers, inlier_mask, line = ransac_line_fit(axis_data, n_iter, threshold, min_inliers)
    best_match = choose_candidates_smooth(axis_data, centroid, direction, inliers, inlier_mask)

    corrected_results = load_results(results_path, sim)
    correction_indices = np.where(~inlier_mask)[0]

    for idx, match in enumerate(correction_indices):
        corrected_results[match].results.data[0, 0, 0, :] = corrected_results[match].results.data[0, 0, best_match[idx], :]
        corrected_results[match].orientations[0, 0, 0] = corrected_results[match].orientations[0, 0, best_match[idx]]
    

    return corrected_results, line