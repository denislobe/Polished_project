from src.template_matching import *
from src.io import *

from pathlib import Path

import pickle
import numpy as np

def tilt_template_matching(input_path, output_path, sim):
    input_files = list_sorted_files(input_path)
    
    tilt_array = np.empty(len(input_files), dtype=object)
    for i, file in enumerate(input_files):
        hsdata = hs.load(file)

        result_data = template_matching(hsdata, sim)
        tilt_array[i] = result_data
    
    pickle.dump(tilt_array, open(output_path / f"{input_path.name}_results.pkl", "wb"))

def Al_template_matching():
    input_path = Path("D:\\Datasets\\aluminium_tilt_series\\preprocessed")
    output_path = Path.cwd().parent / "results" / "Al"

    sim = Sim(angular_resolution=0.5, precession_angle=1.0, min_intensity=1e-5, atom_type="Al", lattice_constant=4.05, space_group=225)

    for folder in input_path.iterdir():
        tilt_template_matching(folder, output_path, sim)

def Ag_template_matching():
    input_path = Path("D:\\Datasets\\silver_tilt_series_2\\preprocessed")
    output_path = Path.cwd().parent / "results" / "Ag"

    sim = Sim(angular_resolution=0.5, precession_angle=1.0, min_intensity=1e-5, atom_type="Ag", lattice_constant=4.09, space_group=225)

    for folder in input_path.iterdir():
        tilt_template_matching(folder, output_path, sim)