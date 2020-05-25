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

import logging

log = logging.getLogger(__name__)



class TrajectoryLerner(TrajectoryBase):
    """
    Trajectory implementing the Lerner group's format.
    """
    suffix = 'dat'
    fields_default = ['id', 'pos']


    def __init__(self, filename, mode='r', fields=None):
        super(TrajectoryLerner, self).__init__(filename, mode)

        # Trajectory file handle
        self.mode = mode
        self.trajectory = gopen(self.filename, self.mode)

        if mode == 'w':
            self.fields = copy(self.fields_default) if fields is None else fields
        if mode == 'r':
            self._read_header()
            self._read_data(filename)
            self._detect_fields()
            # When reading, we must define the steps. There is always only a single snapshot in this trajectory format, with no timestep information.
            self.steps = [0]

        self.fields = canonicalize_fields(self.fields)


    def _read_header(self):
        first_line = self.trajectory.readline()
        self.header = np.array( [float(x) for x in first_line.split()] )


    def _read_data(self, filename):
        self.data = np.loadtxt(filename, skiprows=1)
    

    def _detect_fields(self):
        ncols = self.data.shape[1]
        npart = self.data.shape[0]
        if ncols == 3: # 2D without velocities. 
            self.fields = ['position']
            self.ndim = 2
            self.velocity_present = False
        elif ncols == 4: # 3D without velocities.
            self.fields = ['position']
            self.ndim = 3
            self.velocity_present = False
        elif ncols == 5: # 2D with velocities.
            self.fields = ['position', 'velocity']
            self.ndim = 2
            self.velocity_present = True
        elif ncols == 6: # 3D with velocities.
            self.fields = ['position', 'velocity']
            self.ndim = 3
            self.velocity_present = True
        else:
            RuntimeError("I don't recognize the Lerner group format with %d columns." % ncols)

        # The radius OR species id is in the final column.
        # We say it's a species id when all the entries are ints, and 0 is present.
        final_column = self.data[:, -1]
        if np.all( np.floor(final_column) == final_column ) and np.any( final_column == 0 ):
            self.radius_present = False
            self.species_present = True
            self.fields.append("species")
        else:
            self.radius_present = True
            self.fields.append("radius")
            # Try to map the radii to different particle types (abort if there are more than 5 types, i.e. it's most likely polydisperse.)
            # Assign '0', '1', '2', etc. to the particles with radii in ascending order.
            unique_radii = np.unique(final_column)
            if unique_radii.size <= 5:
                distinct_species = ['0', '1', '2', '3', '4'][:unique_radii.size]
                species = np.zeros((npart)).astype(float)
                for i, r in enumerate(unique_radii):
                    species[ final_column == r ] = distinct_species[i]

                self.data = np.c_[self.data, species]
                self.fields.append("species")
                self.species_present = True
            else:
                self.species_present = False

        log.info("Detected the following fields: ", self.fields)


    def read_sample(self, frame):
        """ returns System instance. """

        L = self.header[0]
        cell = Cell(side=[L, L, L])

        N = self.data.shape[0]
        pos = self.data[:, :self.ndim]
        # Lerner group format has positions from [0, 1).
        pos = pos*L - L/2
        # In 2D, we add a zero z-coordinate to be consistent with other formats.
        if self.ndim == 3:
            # First ndim columns are always the position.
            if self.velocity_present:
                # Next is the velocity, if it exists.
                vel = self.data[:, self.ndim:2*self.ndim]
        elif self.ndim == 2:
            new_pos = np.zeros((N, 3)).astype(float)
            new_pos[:, :self.ndim] = pos
            pos = new_pos
            if self.velocity_present:
                vel = np.zeros((N, 3)).astype(float)
                vel[:, :self.ndim] = self.data[:, self.ndim:2*self.ndim]

        particles = []
        for i in range(N):
            p = Particle()
            p.position = pos[i, :]
            if self.velocity_present:
                p.velocity = vel[i, :]
            if self.species_present and self.radius_present:
                p.species = str( int(self.data[i, -1]) )
                p.radius  = self.data[i, -2]
            elif self.radius_present:
                p.radius = self.data[i, -1]
            elif self.species_present:
                p.species = str( int(self.data[i, -1]) )

            particles.append(p)

        return System(particle=particles, cell=cell)
         

    def write_sample(self, system, step):
        """ Writes to the file handle self.trajectory."""

        data  = system.dump(['pos', 'vel', 'spe', 'particle.radius', 'particle.mass'])
        L = system.cell.side[0]
        distinct_species = system.distinct_species()

        species_to_typeid = {}
        typeid = 0
        for species in distinct_species:
            species_to_typeid[species] = str(typeid)
            typeid += 1

        convert_to_typeid = np.vectorize( lambda species: species_to_typeid[species] )

        pos = data['particle.position']
        species = data['particle.species']    # This is 'A', 'A', 'B', etc when distinct_species = ['A', 'B']
        vel = data['particle.velocity']
        mass = data['particle.mass']
        radius = data['particle.radius']
        typeid = convert_to_typeid(species).astype(np.object) # This is 0, 0, 1, etc when distinct_species = ['A', 'B']

        # If all z coordinates are zero, we assume we are in 2D.
        if np.all( pos[:, 2] == 0 ):
            ndim = 2
            fmt = "%.12g %.12g"
        else:
            ndim = 3
            fmt = "%.12g %.12g %.12g"

        pos = (pos[:, :ndim] + L/2)/L
        vel = vel[:, :ndim]

        columns = (pos,)
        if 'velocity' in self.fields:
            columns += (vel,)
            if ndim == 2:
                fmt += " %.12g %.12g"
            else:
                fmt += " %.12g %.12g %.12g"
        if 'radius' in self.fields:
            columns += (radius,)
            fmt += " %.4g"
        if 'species' in self.fields:
            columns += (typeid,)
            fmt += " %s"

        data = np.column_stack(columns)

        ncols = data.shape[1]
        header = "%.12g" % L
        for i in range(ncols):
            header += " 0"

        np.savetxt(self.trajectory, data, fmt=fmt, header=header, comments='')

