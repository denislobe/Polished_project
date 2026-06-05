from src.preprocessing import *
from src.io import *
from pathlib import Path

import hyperspy.api as hs
import matplotlib.pyplot as plt

def run_preprocessing(input_path, output_path, tilt_number, sigma, hsw, disk_radius, min_distance, threshold):

    files = list_sorted_files(input_path)
    hsdata = hs.load(files[tilt_number])

    hsdata.calibrate()
    plt.show(block=True)
    scale_f = hsdata.axes_manager[2].scale

    for file in files:
        hsdata = hs.load(file)
        centered_data, area_I, area_II, mean_area_I, mean_area_II, masked_area_I, masked_area_II = preprocess_function(
            hsdata,
            scale_f,
            sigma,
            hsw,
            disk_radius,
            min_distance,
            threshold
            )

        save_hsdata(centered_data, output_path / "centered", f"centered_{file.stem}")
        save_hsdata(area_I, output_path / "area_I", f"area_I_{file.stem}")
        save_hsdata(area_II, output_path / "area_II", f"area_II_{file.stem}")
        save_hsdata(mean_area_I, output_path / "mean_area_I", f"mean_area_I_{file.stem}")
        save_hsdata(mean_area_II, output_path / "mean_area_II", f"mean_area_II_{file.stem}")
        save_hsdata(masked_area_I, output_path / "masked_area_I", f"masked_area_I_{file.stem}")
        save_hsdata(masked_area_II, output_path / "masked_area_II", f"masked_area_II_{file.stem}")


def Al_preprocess():
    input_path = Path("D:\\Datasets\\aluminium_tilt_series\\uncalibrated")
    output_path = Path("D:\\Datasets\\aluminium_tilt_series\\preprocessed")
    tilt_number = 55
    sigma = 5
    hsw = 10
    disk_radius = 7
    min_distance = 15
    threshold = 0.25

    run_preprocessing(input_path, output_path, tilt_number, sigma, hsw, disk_radius, min_distance, threshold)

def Ag_preprocess():
    input_path = Path("D:\\Datasets\\silver_tilt_series_2\\Uncalibrated")
    output_path = Path("D:\\Datasets\\silver_tilt_series_2\\preprocessed")
    tilt_number = 0
    sigma = 5
    hsw = 10
    disk_radius = 7
    min_distance = 15
    threshold = 0.25

    run_preprocessing(input_path, output_path, tilt_number, sigma, hsw, disk_radius, min_distance, threshold)



