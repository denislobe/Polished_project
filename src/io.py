from src.template_matching import *

import pathlib as Path
import numpy as np
import pickle
import orix

def list_sorted_files(path):
    return sorted(path.glob("*.hspy"), key=lambda x: int(x.stem.split("_")[-1]))

def save_hsdata(hsdata, output_path, filename):
    output_path.mkdir(parents=True, exist_ok=True)
    hsdata.save(output_path / f"{filename}.hspy", overwrite=True)

def load_results(file_path, sim):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    tilt_array = np.empty(len(data), dtype=object)
    for i, item in enumerate(data):
        tilt_array[i] = TemplateMatching(data=item, sim=sim)
    return tilt_array

def convert_best_match_to_orientations(tilt_array):
    orientation_array = np.empty(len(tilt_array), dtype=object)

    for i, item in enumerate(tilt_array):
        orientation_array[i] = item.orientations[0, 0, 0]
    
    quaternion_array = np.stack([j.data for j in orientation_array])
    orientation_array = orix.quaternion.Orientation(quaternion_array, symmetry=tilt_array[0].sim.phase.point_group)
    return orientation_array

def line_to_orientations(line, symmetry):
    theta = np.linalg.norm(line, axis=1)
    quaternion_array = np.zeros((len(line), 4))
    small_angles = theta < 1e-6

    quaternion_array[small_angles, 0] = 1.0

    n = line[~small_angles] / theta[~small_angles][:, None]

    quaternion_array[~small_angles, 0] = np.cos(theta[~small_angles] / 2)
    quaternion_array[~small_angles, 1:] = n * np.sin(theta[~small_angles] / 2)[:, None]

    orientations = orix.quaternion.Orientation(quaternion_array, symmetry=symmetry)

    return orientations