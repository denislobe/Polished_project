from dataclasses import dataclass
from diffpy.structure import Atom, Lattice, Structure
from diffsims.generators.simulation_generator import SimulationGenerator

import pyxem
import hyperspy.api as hs
import orix

@dataclass
class Sim:
    lattice_constant : float
    atom_type : str
    space_group : int
    angular_resolution : float
    precession_angle : float
    min_intensity : float

    phase = object = None
    grid = object = None
    orientations : object = None
    simulation_generator : object = None
    simulations : object = None

    def __post_init__(self):
        atoms = [Atom(self.atom_type, (0, 0, 0)), Atom(self.atom_type, (0.5, 0.5, 0)), Atom(self.atom_type, (0.5, 0, 0.5)), Atom(self.atom_type, (0, 0.5, 0.5))]
        lattice = Lattice(self.lattice_constant, self.lattice_constant, self.lattice_constant, 90, 90, 90)
        self.phase = orix.crystal_map.Phase(name=self.atom_type, space_group=self.space_group, structure=Structure(atoms, lattice))
        self.grid = orix.sampling.get_sample_reduced_fundamental(self.angular_resolution, point_group=self.phase.point_group)
        self.orientations = orix.quaternion.Orientation(self.grid, symmetry=self.phase.point_group)

        self.simulation_generator = SimulationGenerator(precession_angle=self.precession_angle, minimum_intensity=self.min_intensity)
        self.simulations = self.simulation_generator.calculate_diffraction2d(
            phase = self.phase,
            rotation = self.grid,
            reciprocal_radius = 1.5,
            with_direct_beam = False,
            max_excitation_error = 0.01
        )

@dataclass
class TemplateMatching:
    data : object
    sim : object
    orientations : object = None
    results : object = None
    

    def __post_init__(self):
        if self.results is None:
            s = hs.signals.Signal2D(self.data)
            
            s.axes_manager.navigation_axes[0].name = "y"
            s.axes_manager.navigation_axes[1].name = "x"

            s.axes_manager.signal_axes[0].name = "n-best"
            s.axes_manager.signal_axes[1].name = "columns"

            s.set_signal_type("orientation_map")
            s.simulation = self.sim.simulations
            s.column_names = ["index", "correlation", "rotation", "factor"]
            s.units = ["a.u.", "a.u.", "deg", "a.u."]

            self.results = s

            self.orientations = s.to_single_phase_orientations()

def template_matching(data, sim):
    data.change_dtype("float32")
    azi = data.get_azimuthal_integral2d(npt=256)
    result = azi.get_orientation(sim.simulations, n_best=sim.grid.size, frac_keep=1.0)

    result_data = result.data
    return result_data


