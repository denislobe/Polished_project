import hyperspy.api as hs
import matplotlib.pyplot as plt

def define_calibration(hsdata, scale_factor):
    hsdata.axes_manager[2].scale = scale_factor
    hsdata.axes_manager[3].scale = scale_factor
    hsdata.axes_manager[2].units = "1/Å"
    hsdata.axes_manager[3].units = "1/Å"

    ny, nx = hsdata.axes_manager.signal_shape
    center = (ny/2, nx/2)
    hsdata.calibration(center=center)


def center_beam_position(hsdata, sigma=5, hsw=10):
    shifts = hsdata.get_direct_beam_position(method="blur", sigma=sigma, half_square_width=hsw)
    linear_shifts = shifts.get_linear_plane()

    return hsdata.center_direct_beam(shifts=linear_shifts, inplace=False)


def ROI_selection(hsdata):
    area_I_ROI = hs.roi.RectangularROI(left=0, top=0, right=10, bottom=10)
    area_II_ROI = hs.roi.RectangularROI(left=0, top=10, right=10, bottom=20)
    hsdata.plot()

    area_I = area_I_ROI.interactive(hsdata)
    area_II = area_II_ROI.interactive(hsdata, color='C1')

    plt.show(block=True)
    return area_I, area_II


def get_mean_data(hsdata):
    mean = hsdata.mean()

    mean_data = hs.signals.Signal2D(mean.data[None, None, :, :])
    mean_data.set_signal_type("electron_diffraction")
    
    define_calibration(mean_data, scale_factor=hsdata.axes_manager[2].scale)
    
    return mean_data


def masking(hsdata, disk_radius=7, min_distance=15, threshold=0.25):
    match_disk = hsdata.template_match_disk(disk_r=disk_radius, subtract_min=False)
    peak_vectors = match_disk.get_diffraction_vectors(min_distance=min_distance, threshold_abs=threshold)

    mask = peak_vectors.to_mask(disk_r = disk_radius)
    center_mask = ~match_disk.get_direct_beam_mask(10)
    masked_data = hsdata * mask
    masked_data = masked_data * center_mask

    return masked_data


def preprocess_function(hsdata, scale_factor, sigma=5, hsw=10, disk_radius=7, min_distance=15, threshold=0.25):
    centered_data = center_beam_position(hsdata, sigma=sigma, hsw=hsw)

    area_I, area_II = ROI_selection(centered_data)

    mean_area_I = get_mean_data(area_I)
    mean_area_II = get_mean_data(area_II)

    masked_area_I = masking(mean_area_I, disk_radius=disk_radius, min_distance=min_distance, threshold=threshold)
    masked_area_II = masking(mean_area_II, disk_radius=disk_radius, min_distance=min_distance, threshold=threshold)

    define_calibration(centered_data, scale_factor)
    define_calibration(area_I, scale_factor)
    define_calibration(area_II, scale_factor)
    define_calibration(mean_area_I, scale_factor)
    define_calibration(mean_area_II, scale_factor)
    define_calibration(masked_area_I, scale_factor)
    define_calibration(masked_area_II, scale_factor)

    return centered_data, area_I, area_II, mean_area_I, mean_area_II, masked_area_I, masked_area_II