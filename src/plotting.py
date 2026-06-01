from orix.vector import Vector3d, Miller
from orix.plot import IPFColorKeyTSL

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import hyperspy.api as hs



def NCC_IPF(templatematch, dp_image_path=None):        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="ipf", symmetry=templatematch.sim.phase.point_group)

        template_index = templatematch.data[0, 0, 0, 0].astype("int16")
        optical_axis = Miller(uvw=[0, 0, 1], phase=templatematch.sim.phase)
        miller = (templatematch.sim.simulations.get_simulation(template_index)[0] * optical_axis).round()

        cors = templatematch.data[0, 0, :, 1] / np.max(templatematch.data[0, 0, :, 1])
        ax.scatter(templatematch.orientations[0, 0], c=cors, cmap='magma')
        ax.scatter(templatematch.orientations[0, 0, 0], c="g", marker="o", label=f"Best match: {miller.uvw[0]}")
        
        if dp_image_path is not None:
            dp_image = hs.load(dp_image_path)

            dp_image.plot(cmap='viridis', norm='symlog')
            dp_image.add_marker(templatematch.results.to_markers(annotate=True))
        ax.legend()
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


def misorientation