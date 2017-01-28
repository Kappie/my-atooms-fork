# This file is part of atooms
# Copyright 2010-2014, Daniele Coslovich

import numpy
import re
import copy
import logging

from .base import TrajectoryBase
from .utils import gopen
from atooms.utils import tipify
from atooms.system.particle import Particle
from atooms.system.cell import Cell
from atooms.system import System

log = logging.getLogger(__name__)

class TrajectorySimpleXYZ(TrajectoryBase):

    """Trajectory with simple xyz layout.

    It uses a memory light-weight indexed access.
    """

    suffix = 'xyz'

    def __init__(self, filename, mode='r'):
        TrajectoryBase.__init__(self, filename, mode)
        self._cell = None
        self._id_map = [] # list to map numerical ids (indexes) to chemical species (entries)
        self._id_min = 1 # minimum integer for ids, can be modified by subclasses
        self.trajectory = open(self.filename, self.mode)
        if self.mode == 'r':
            # Internal index of lines to seek and tell.
            # We may delay setup, moving to read_init() assuming
            # self.steps becomes a property
            self._setup_index()
            self._setup_steps()

    def _setup_index(self):
        """Sample indexing via tell / seek"""
        self._index_sample = []
        self._index_header = []
        self._index_cell = None
        self.trajectory.seek(0)
        while True:
            line = self.trajectory.tell()
            data = self.trajectory.readline().strip()

            # We break if file is over or we found an empty line
            if not data:
                break

            # The first line contains the number of particles
            # If something goes wrong, this could be the last line
            # with the cell side (Lx,Ly,Lz) and we parse it some other way
            try:
                npart = int(data)
                self._index_header.append(line)
            except ValueError:
                self._index_cell = line
                break

            # Skip npart+1 lines
            _ = self.trajectory.readline()
            self._index_sample.append(self.trajectory.tell())
            for i in range(npart):
                _ = self.trajectory.readline()

    def _read_metadata(self, sample):
        """Internal xyz method to get header metadata from comment line of given *sample*.

        We assume metadata format is a space-separated sequence of
        comma separated entries such as:

        columns:id,x,y,z step:10
        columns=id,x,y,z step=10
        """
        # Go to line and skip Npart info
        self.trajectory.seek(self._index_header[sample])
        npart = int(self.trajectory.readline())
        data = self.trajectory.readline()

        # Fill metadata dictionary
        meta = {}
        meta['npart'] = npart
        for e in data.split():
            s = re.search(r'(\S+)\W*[=:]\W*(\S+)', e)
            if s is not None:
                tag, data = s.group(1), s.group(2)
                # Remove dangling commas
                data = data.strip(',')
                # If there are commas, this is a list, else a scalar.
                # We convert the string to appropriate types
                if ',' in data:
                    meta[tag] = [tipify(_) for _ in data.split(',')]
                else:
                    meta[tag] = tipify(data)
        return meta

    def _setup_steps(self):
        """Find steps list."""
        self.steps = []
        for sample in range(len(self._index_sample)):
            meta = self._read_metadata(sample)
            try:
                self.steps.append(meta['step'])
            except KeyError:
                # If no step info is found, we add steps sequentially
                self.steps.append(sample+1)

    def update_id(self, particle):
        """Update chemical ids of *particle* list and global database id_map."""
        # TODO: use sets instead
        # We keep the id database sorted by name.
        for p in particle:
            if not p.name in self._id_map:
                self._id_map.append(p.name)
                self._id_map.sort()

        # Assign ids to particles according to the updated database
        for p in particle:
            p.id = self._id_map.index(p.name) + self._id_min

    def read_init(self):
        # Grab cell from the end of file if it is there
        try:
            side = self._read_metadata(0)['cell']
            self._cell = Cell(side)
        except KeyError:
            self._cell = self._parse_cell()

    def _parse_cell(self):
        """Internal emergency method to grab the cell."""
        cell = None
        if self._index_cell:
            self.trajectory.seek(self._index_cell)
            side = numpy.fromstring(self.trajectory.readline(), sep=' ')
            cell = Cell(side)
        return cell

    def read_sample(self, sample):
        meta = self._read_metadata(sample)
        self.trajectory.seek(self._index_sample[sample])
        particle = []
        for _ in range(meta['npart']):
            data = self.trajectory.readline().strip().split()
            name = data[0]
            r = numpy.array(data[1:4], dtype=float)
            particle.append(Particle(name=name, position=r))

        self.update_id(particle)
        return System(particle, self._cell)

    def _comment_header(self, step, system):
        fmt = "step:%d columns:id,x,y,z" % step
        if system.cell is not None:
            fmt += " cell:" + ','.join(['%s' % x for x in system.cell.side])
        return fmt

    def write_sample(self, system, step):
        self.trajectory.write("%s\n" % len(system.particle))
        self.trajectory.write(self._comment_header(step, system) + '\n')
        ndim = len(system.particle[0].position)
        fmt = "%s" + ndim*" %14.6f" + "\n"
        for p in system.particle:
            self.trajectory.write(fmt % ((p.name,) + tuple(p.position)))

    def close(self):
        self.trajectory.close()


# Format callbacks

def update_name(particle, data, meta):
    particle.name = data[0]
    return data[1:]

def update_radius(particle, data, meta):
    particle.radius = float(data[0])
    return data[1:]

def update_tag(particle, data, meta):
    particle.tag = data[0:]
    return data[1:]

def update_name(particle, data, meta):
    particle.name = data[0]
    return data[1:]

def update_position(particle, data, meta):
    ndim = meta['ndim']
    # It is crucial to assing position, not to use the slice!
    # Otherwise we get a reference, not a copy.
    particle.position = numpy.array(data[0:ndim], dtype=float)
    return data[ndim:]

def update_velocity(particle, data, meta):
    ndim = meta['ndim']
    particle.velocity = numpy.array(data[0:ndim], dtype=float)
    return data[ndim:]

# def update(particle, data, what):
#     particle.gettatr(what) = tipify(data)

def _optimize_fmt(fmt):
    if 'x' in fmt:
        fmt[fmt.index('x')] = 'pos'
    if 'vx' in fmt:
        fmt[fmt.index('vx')] = 'vel'
    for tag in ['y', 'z', 'vy', 'vz']:
        if tag in fmt:
            fmt.remove(tag)
    return fmt


class TrajectoryXYZ(TrajectoryBase):

    """Trajectory with XYZ layout using memory leightweight indexed access."""

    suffix = 'xyz'
    callback_read = {'name': update_name,
                     'type': update_name, # alias
                     'id': update_name, # alias
                     'tag': update_tag,
                     'radius': update_radius,
                     'pos': update_position,
                     'vel': update_velocity,
    }

    def __init__(self, filename, mode='r', alias=None, fmt=None):
        TrajectoryBase.__init__(self, filename, mode)
        if alias is None:
            alias = {}
        if fmt is None:
            fmt = ['name', 'pos']
        self.fmt = fmt
        self._fmt = None
        self._fmt_float = True
        self._done_format_setup = False
        self.alias = alias
        self.shortcuts = {'pos': 'position',
                          'x': 'position[0]',
                          'y': 'position[1]',
                          'z': 'position[2]',
                          'vel': 'velocity',
                          'vx': 'velocity[0]',
                          'vy': 'velocity[1]',
                          'vz': 'velocity[2]',
                          'id': 'name',
                          'type': 'name'}
        self._id_min = 1 # minimum integer for ids, can be modified by subclasses
        self._cell = None
        self._id_map = [] # list to map numerical ids (indexes) to chemical species (entries)
        self.trajectory = gopen(self.filename, self.mode)
        if self.mode == 'r':
            # Internal index of lines to seek and tell.
            # We may delay setup, moving to read_init() assuming
            # self.steps becomes a property
            self._setup_index()
            # Warning: setting up steps require aliases to be defined in
            # init and not later.
            self._setup_steps()

    def _setup_format(self):
        if not self._done_format_setup:
            self._done_format_setup = True
            # %g allows to format both float and int but it's 2x slower.
            # This switch is for performance
            if self._fmt_float:
                _fmt = '%14.' + str(self.precision) + 'f'
            else:
                _fmt = '%g'
            def array_fmt(arr):
                """Remove commas and [] from numpy array repr."""
                # Passing a scalar will trigger an error (gotcha: even
                # when casting numpy array to list, the elements remain of
                # numpy type and this function gets called! (4% slowdown)
                try:
                    return ' '.join([_fmt % x for x in arr])
                except:
                    return _fmt % arr
            numpy.set_string_function(array_fmt, repr=False)

    def _setup_index(self):
        """Sample indexing via tell / seek"""
        self._index_sample = []
        self._index_header = []
        self._index_cell = None
        self.trajectory.seek(0)
        while True:
            line = self.trajectory.tell()
            data = self.trajectory.readline().strip()

            # We break if file is over or we found an empty line
            if not data:
                break

            # The first line contains the number of particles
            # If something went wrong, this could be the last line
            # with the cell side (Lx,Ly,Lz) and we parse it some other way
            try:
                npart = int(data)
                self._index_header.append(line)
            except ValueError:
                self._index_cell = line
                break

            # Skip npart+1 lines
            _ = self.trajectory.readline()
            self._index_sample.append(self.trajectory.tell())
            for i in range(npart):
                _ = self.trajectory.readline()
            
    def _setup_steps(self):
        """Find steps list."""
        self.steps = []
        for sample in range(len(self._index_sample)):
            meta = self._read_metadata(sample)
            try:
                self.steps.append(meta['step'])
            except KeyError:
                # If no step info is found, we add steps sequentially
                self.steps.append(sample+1)

    def _expand_shortcuts(self):
        _fmt = []
        for field in self.fmt:
            try:
                _fmt.append(self.shortcuts[field])
            except KeyError:
                _fmt.append(field)
        return _fmt

    def _read_metadata(self, sample):
        """Internal xyz method to get header metadata from comment line of given *sample*.

        We assume metadata fmt is a space-separated sequence of comma
        separated entries such as:

        columns:id,x,y,z step:10
        columns=id,x,y,z step=10
        """
        # Go to line and skip Npart info
        self.trajectory.seek(self._index_header[sample])
        npart = int(self.trajectory.readline())
        data = self.trajectory.readline()

        # Remove spaces around : or =
        data = re.sub(r'\W*[=:]\W*', ':', data)
        
        # Fill metadata dictionary
        meta = {}
        meta['npart'] = npart
        for e in data.split():
            s = re.search(r'(\S+)\W*[=:]\W*(\S+)', e)
            if s is not None:
                tag, data = s.group(1), s.group(2)
                # Remove dangling commas
                data = data.strip(',')
                # If there are commas, this is a list, else a scalar.
                # We convert the string to appropriate types
                if ',' in data:
                    meta[tag] = [tipify(_) for _ in data.split(',')]
                else:
                    meta[tag] = tipify(data)

        # Apply an alias dict to tags, e.g. to add step if Cfg was found instead
        for alias, tag in self.alias.items():
            try:
                meta[tag] = meta[alias]
            except KeyError:
                pass

        # Fix dimensions based on side of cell.
        # Fallback to ndim metadata or 3.
        try:
            if not 'ndim' in meta:
                meta['ndim'] = len(meta['cell'])
        except KeyError:
            meta['ndim'] = 3 # default

        return meta

    def update_id(self, particle):
        """Update chemical ids of *particle* list and global database id_map."""
        # We keep the id database sorted by name.
        for p in particle:
            if not p.name in self._id_map:
                self._id_map.append(p.name)
                self._id_map.sort()

        # Assign ids to particles according to the updated database
        for p in particle:
            p.id = self._id_map.index(p.name) + self._id_min

    def update_mass(self, particle, metadata):
        """Fix the masses of *particle* list."""
        # We assume masses read from the header metadata are sorted by name
        try:
            mass_db = {}
            for key, value in zip(self._id_map, metadata['mass']):
                mass_db[key] = value
            for p in particle:
                p.mass = mass_db[p.name]
        except KeyError:
            return
        except TypeError:
            return

    def read_init(self):
        # Grab cell from the end of file if it is there
        try:
            side = self._read_metadata(0)['cell']
            self._cell = Cell(side)
        except KeyError:
            self._cell = self._parse_cell()

    def read_sample(self, sample):
        # Use columns if they are found in the header, or stick to the
        # default format.
        meta = self._read_metadata(sample)
        if 'columns' in meta:
            fmt = meta['columns']
        else:
            fmt = self.fmt
        fmt = _optimize_fmt(fmt)
        # Add null callbacks for missing fmt entries
        for key in [key for key in fmt if key not in self.callback_read]:
            self.callback_read[key] = None

        # Read sample now
        self.trajectory.seek(self._index_sample[sample])
        particle = []
        for i in range(meta['npart']):
            p = Particle()
            data = self.trajectory.readline().split()
            for key in fmt:
                if self.callback_read[key] is not None:
                    data = self.callback_read[key](p, data, meta)
            particle.append(p)
        # Now we fix ids and other metadata
        self.update_id(particle)
        self.update_mass(particle, meta)

        # Check if we also have a cell
        if 'side' in meta:
            cell = Cell(meta['side'])
        else:
            cell = self._cell
        return System(particle, cell)

    def _comment_header(self, step, system):
        # Comment line: concatenate metadata
        line = 'step:%d ' % step
        line += 'columns:' + ','.join(self.fmt)
        if system.cell is not None:
            line += " cell:" + ','.join(['%s' % x for x in system.cell.side])
        return line

    def write_sample(self, system, step):
        self._setup_format()
        if self._fmt is None:
            self._fmt = self._expand_shortcuts()
        self.trajectory.write('%d\n' % len(system.particle))
        self.trajectory.write(self._comment_header(step, system) + '\n')
        fmt = ' '.join(['{0.' + field + '}' for field in self._fmt]) + '\n'
        for p in system.particle:
            self.trajectory.write(fmt.format(p))

    def _parse_cell(self):
        """Internal xyz method to grab the cell. Can be overwritten in subclasses."""
        cell = None
        if self._index_cell:
            self.trajectory.seek(self._index_cell)
            side = numpy.fromstring(self.trajectory.readline(), sep=' ')
            cell = Cell(side)
        return cell

    def close(self):
        self.trajectory.close()


class TrajectoryNeighbors(TrajectoryXYZ):

    """Neighbors trajectory."""

    def __init__(self, filename, mode='r', offset=1):
        super(TrajectoryNeighbors, self).__init__(filename, mode=mode, alias={'time':'step'})
        # TODO: determine minimum value of index automatically
        # TODO: possible regression here if no 'time' tag is found
        self._offset = offset # neighbors produced by voronoi are indexed from 1
        self._netwon3 = False
        self._netwon3_message = False

    def read_sample(self, sample):
        meta = self._read_metadata(sample)
        self.trajectory.seek(self._index_sample[sample])
        s = System()
        s.neighbors = []
        for _ in range(meta['npart']):
            data = self.trajectory.readline().split()
            neigh = numpy.array(data, dtype=int)
            s.neighbors.append(neigh-self._offset)

        # Ensure III law Newton.
        # If this is ok on first sample we skip it for the next ones
        # if not self._netwon3:
        #     self._netwon3 = True
        #     for i, ilist in enumerate(p):
        #         for j in ilist:
        #             if not i in p[j]:
        #                 p[j].append(i)
        #                 self._netwon3 = False
        #     if not self._netwon3 and not self._netwon3_message:
        #         print 'Warning: enforcing 3rd law of Newton...'
        return s
