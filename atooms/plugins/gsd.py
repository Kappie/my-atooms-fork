from atooms.trajectory.base import TrajectoryBase

import numpy as np
import re
import gsd
import gsd.hoomd
from copy import copy

from atooms.system.particle import Particle, distinct_species
from atooms.system.cell import Cell
from atooms.system import System
from atooms.trajectory.utils import gopen
from atooms.trajectory.base import canonicalize_fields



class TrajectoryGSD(TrajectoryBase):
    """
    Trajectory implementing the Glotzer group's binary GSD format.
    """
    suffix = 'gsd'


    def __init__(self, filename, mode='r', fields=None):
        super(TrajectoryGSD, self).__init__(filename, mode)
        self._fields_default = ['id', 'pos']
        self.fields = copy(self._fields_default) if fields is None else fields
        self.fields = canonicalize_fields(self.fields)

        # self.mode can be 'w' or 'r', but gsd is a binary format, so it only accepts 'wb' or 'rb'.
        file_mode = self.mode + "b"     
        # Trajectory file handle
        self.trajectory = gsd.hoomd.open(name=self.filename, mode=file_mode)
        # When reading, we must define the steps.
        if self.mode == 'r':
            self.steps = [snap.configuration.step for snap in self.trajectory]

    def read_sample(self, frame):
        """ returns System instance. """
        snap = self.trajectory[frame]
        # HOOMD convention in 2D.
        if snap.configuration.box[2] == 1.0:
            ndim = 2
        else:
            ndim = 3

        # Convert typeid from [0, 0, 1, ...] to ['A', 'A', 'B', ...] when snap.particles.types = ['A', 'B']
        distinct_species = snap.particles.types
        distinct_typeids = list( range(len(distinct_species)) )
        typeid_to_species = {}
        for i in distinct_typeids:
            typeid_to_species[i] = distinct_species[i]

        box = snap.configuration.box[:ndim]    # atooms does not handle sheared boxes.
        cell = Cell(side=box)

        N = snap.particles.position.shape[0]
        particles = []
        for i in range(N):
            p = Particle(
                mass     = snap.particles.mass[i],
                species  = typeid_to_species[ snap.particles.typeid[i] ],
                position = snap.particles.position[i, :ndim],
                velocity = snap.particles.velocity[i, :ndim],
                radius   = snap.particles.diameter[i] / 2,
            )
            p.image = snap.particles.image[i, :ndim]
            particles.append(p)

        return System(particle=particles, cell=cell)
         

    def write_sample(self, system, step):
        """ Writes to the file handle self.trajectory."""

        data  = system.dump(['pos', 'vel', 'spe', 'particle.mass', 'particle.radius'])
        box = system.cell.side
        # HOOMD convention for 2D.
        if box.shape[0] == 2 or box[2] == 1.:
            ndim = 2
        else:
            ndim = 3

        N = len(system.particle)
        distinct_species = system.distinct_species()


        # Convert species from ['A', 'A', 'B', ...] to [0, 0, 1, ...] when distinct_species = ['A', 'B']
        species_to_typeid = {}
        typeid = 0
        for species in distinct_species:
            species_to_typeid[species] = typeid
            typeid += 1

        pos = data['particle.position'][:, :ndim]
        vel = data['particle.velocity'][:, :ndim]
        species = data['particle.species']    # This is 'A', 'A', 'B', etc when distinct_species = ['A', 'B']
        mass = data['particle.mass']
        radius = data['particle.radius']
        convert_to_typeid = np.vectorize( lambda species: species_to_typeid[species] )
        typeid = convert_to_typeid(species).astype(int) # This is 0, 0, 1, etc when distinct_species = ['A', 'B']

        # HOOMD conventions in 2D.
        if ndim == 2:
            box = np.array([box[0], box[1], 1.0])
            new_pos = np.zeros((N, 3))
            new_pos[:, :2] = pos
            pos = new_pos
            new_vel = np.zeros((N, 3))
            new_vel[:, :2] = vel
            vel = new_vel

        snap = gsd.hoomd.Snapshot()
        snap.configuration.box = [box[0], box[1], box[2], 0, 0, 0] # Assume all strains 0.
        snap.configuration.step = step
        snap.configuration.dimensions = ndim
        snap.particles.types = distinct_species
        snap.particles.N = N

        snap.particles.position = pos   # atooms.system and gsd both save positions from -L/2 to L/2.
        snap.particles.typeid = typeid
        if 'velocity' in self.fields:
            snap.particles.velocity = vel
        if 'mass' in self.fields:
            snap.particles.mass = mass
        if 'diameter' in self.fields:
            snap.particles.diameter = 2 * radius

        self.trajectory.append(snap)

