from orix.vector import Vector3d, Miller
from orix.plot import IPFColorKeyTSL
from orix.quaternion import OrientationRegion
from src.io import * 

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import hyperspy.api as hs



def NCC_IPF(result, frame, dp_image_path=None):        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="ipf", symmetry=result[frame].sim.phase.point_group)

        template_index = result[frame].data[0, 0, 0, 0].astype("int16")
        optical_axis = Miller(uvw=[0, 0, 1], phase=result[frame].sim.phase)
        miller = (result[frame].sim.simulations.get_simulation(template_index)[0] * optical_axis).round()

        cors = result[frame].data[0, 0, :, 1] / np.max(result[frame].data[0, 0, :, 1])
        ax.scatter(result[frame].orientations[0, 0], c=cors, cmap='magma')
        ax.scatter(result[frame].orientations[0, 0, 0], c="g", s=100, marker="o",) #label=f"Best match: {miller.uvw[0]}")
        
        if dp_image_path is not None:
            dp_image = hs.load(list_sorted_files(dp_image_path)[frame])

            dp_image.plot(cmap='viridis', norm='symlog', colorbar=False)

            fig = plt.gcf()
            ax = fig.axes[0]
            for text in ax.texts:
                text.remove()
            dp_image.add_marker(result[frame].results.to_markers(annotate=True))
        #ax.legend()
        plt.show()



def plot_orientation_map(results, sim):
    xmap = results.results.to_crystal_map()
    oris = xmap.orientations
    corrs = results.results.data[:,:,0,1].flatten()

    key_x = IPFColorKeyTSL(sim.phase.point_group, Vector3d.xvector())
    key_y = IPFColorKeyTSL(sim.phase.point_group, Vector3d.yvector())
    key_z = IPFColorKeyTSL(sim.phase.point_group, Vector3d.zvector())

    oris_z = key_z.orientation2color(oris)
    xmap.plot(oris_z, overlay=corrs, remove_padding=True)


    
def plot_best_match_ipf(results, sim):
    directions = Vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    titles = ["x", "y", "z"]

    flat_orientations = results.orientations[:,:,0].flatten()

    fig = plt.figure(figsize=(15, 6))
    axes = []
    for i, d in enumerate(directions):
        ax = fig.add_subplot(1, 3, i+1, projection="ipf", direction=d, symmetry=sim.phase.point_group)
        axes.append(ax)

        ax.scatter(flat_orientations)
        ax.set_title({titles[i]})



def tilt_ipf(orientation, sim, line=None):
    c_scalar = np.arange(1, orientation.shape[0] + 1)

    directions = Vector3d([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    titles = ["x", "y", "z"]

    fig = plt.figure(figsize=(15, 6))
    plt.rcParams["axes.grid"] = True

    # Create colorbar (shared)
    norm = mpl.colors.Normalize(vmin=c_scalar.min(), vmax=c_scalar.max())
    cmap = plt.cm.get_cmap('viridis')
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    colors = cmap(norm(c_scalar))

    axes = []
    for i, d in enumerate(directions):
        ax = fig.add_subplot(1, 3, i + 1, projection='ipf', direction=d, symmetry=sim.phase.point_group)
        axes.append(ax)
        if line is not None:
            ax.scatter(line, c='k', linewidth=0.4)

        ax.scatter(orientation, c=colors)
        ax.set_title(titles[i])

    # Add colorbar in its own axis (right side)
    cax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Dataset number")
    
    plt.subplots_adjust(left=0.05, right=0.9, wspace=0.25)



def plot_line_fit(data, line):

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='axangle')
    fundamental_zone = OrientationRegion.from_symmetry(data.symmetry)

    # --- all points ---
    ax.scatter(data[:, 0], alpha=0.3, s=10, label="All points")

    # --- fitted line ---
    ax.scatter(line, linewidth=3, label="Fitted line")
    
    # --- fundamental zone ---
    ax.plot_wireframe(fundamental_zone, color='k', alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    ax.legend()
    plt.tight_layout()
    plt.show()



def plot_misorientation(misorientation):
    plt.figure(figsize=(15, 5))
    frames = np.arange(len(misorientation))

    plt.plot(frames, np.rad2deg(misorientation), marker='o', linestyle='--')
    plt.xlabel("Frame")
    plt.ylabel("Degrees")
    plt.show()