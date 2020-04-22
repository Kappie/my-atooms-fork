import logging
import numpy

from .xyz import TrajectoryXYZ
from .utils import gopen
from atooms.core.utils import tipify
from atooms.system.particle import Particle
from atooms.system.cell import Cell
from atooms.system import System

log = logging.getLogger(__name__)


class TrajectoryEXYZ(TrajectoryXYZ):
    """
    Trajectory with extended XYZ layout.
    """
    suffix = 'xyz'
    callback_read = {}

    def __init__(self, filename, mode='r'):
        super(TrajectoryXYZ, self).__init__(filename, mode)

        # Trajectory file handle
        self.precision = 12
        self._done_format_setup = False
        self._fields_float = True
        self.trajectory = gopen(self.filename, self.mode)
        self.alias = {'pos': 'position',
                      'vel': 'velocity'}
        # TODO: not necessary anymore
        self.alias_fmt = {'S': 's',
                          'R': '.{{0}}g'.format(self.precision),
                          'I': 'd'}
        self.properties = [['species', 'S', 1],
                           ['pos', 'R', 3]]
        
        # Internal index of lines via seek and tell.
        if self.mode == 'r':
            self._setup_index()
            assert len(self._index_frame) > 0, 'empty file {}'.format(self.trajectory)
            assert len(self._index_header) > 0, 'empty file {}'.format(self.trajectory)
            # Read metadata
            self.metadata = self._read_comment(0)

    def _read_comment(self, frame):
        """
        Internal xyz method to get header metadata from comment line of
        given `frame`.
        """
        # Go to line and skip Npart info
        self.trajectory.seek(self._index_header[frame])
        npart = int(self.trajectory.readline())
        data = self.trajectory.readline()
        meta = {}

        # We first gather all keys
        keys = []
        for entry in data.split("=")[:-1]:
            keys.append(entry.split()[-1])

        # Now we extract the values for each key using a dynamic regexp
        import re
        regexp = ''
        for key in keys:
            regexp += '{}=(.*)'.format(key)
        match = re.match(regexp, data)
        for i, key in enumerate(keys):
            meta[key] = match.groups()[i].strip()

        # The Properties key is special so we deal with it first
        entries = meta['Properties'].split(':')
        assert len(entries) % 3 == 0, len(entries)
        properties = []
        for i in range(0, len(entries), 3):            
            key, fmt, ndims = entries[i: i+3]
            properties.append((key, fmt, ndims))

        meta['Properties'] = properties
            
        # Go through keys, listify and tipify them
        for key in meta:            
            if key != 'Properties':
                if meta[key].startswith('"'):
                    meta[key] = meta[key].strip('"')
                    meta[key] = [tipify(_) for _ in meta[key].split()]
                else:
                    meta[key] = tipify(meta[key])
                    
        return meta

    def read_steps(self):
        """Find steps list."""
        steps = []
        for frame in range(len(self._index_frame)):
            meta = self._read_comment(frame)
            try:
                steps.append(meta['Step'])
            except KeyError:
                # If no step info is found, we add steps sequentially
                steps.append(frame+1)
        return steps

    def read_sample(self, frame):
        # Read metadata of this frame
        meta = self._read_comment(frame)

        # Get number of particles
        self.trajectory.seek(self._index_header[frame])
        npart = int(self.trajectory.readline())

        # Read frame now
        self.trajectory.seek(self._index_frame[frame])
        particle = []
        for i in range(npart):
            p = Particle()
            data = self.trajectory.readline().split()
            i = 0
            for key, fmt, ndims in meta['Properties']:
                ndims = int(ndims)
                if key in self.alias:
                    key = self.alias[key]
                if ndims == 1:
                    if fmt == 'R':
                        setattr(p, key, float(data[i]))
                    elif fmt == 'I':
                        setattr(p, key, int(data[i]))
                    elif fmt == 'S':
                        setattr(p, key, data[i])
                    else:
                        raise ValueError('unknown format key')
                else:
                    if fmt == 'R':
                        setattr(p, key, numpy.array(data[i: i+ndims], dtype=numpy.float64))
                    elif fmt == 'I':
                        setattr(p, key, numpy.array(data[i: i+ndims], dtype=numpy.int64))
                    elif fmt == 'S':
                        setattr(p, key, numpy.array(data[i: i+ndims]))
                    else:
                        raise ValueError('unknown format key')
                i += 1            
            particle.append(p)
                
        side = meta["Lattice"]
        # TODO: remove hard coded
        cell = Cell([side[0], side[4], side[8]])
                
        return System(particle, cell)

    def read_timestep(self):
        for key in ['dt', 'Dt', 'timestep', 'Timestep']:
            if key in self.metadata:                
                return self.metadata[key]
        return 1.0

    def _comment(self, step, system):
        # TODO: improve
        L = []
        for i in range(len(system.cell.side)):
            for j in range(len(system.cell.side)):
                if i == j:
                    L.append(system.cell.side[i])
                else:
                    L.append(0.0)

        # Reformat properties
        # TODO: improve
        _properties = ''
        for i in range(len(self.properties)):
            _properties += ':'.join([str(_) for _ in self.properties[i]]) + ':'
        _properties = _properties.strip(':')

        line = 'Properties={} '.format(_properties)
        line += 'Lattice="{}" '.format(' '.join([str(_) for _ in L]))
        line += 'Timestep={} '.format(self.timestep)
        line += 'Step={} '.format(step)
        return line.strip()
    
    def write_sample(self, system, step):
        #self._setup_format()
        self.trajectory.write('{}\n'.format(len(system.particle)))
        self.trajectory.write(self._comment(step, system) + '\n')

        # Replace aliases
        import copy
        properties = copy.copy(self.properties)
        for i in range(len(properties)):
            key, fmt, ndims = properties[i]
            if key in self.alias:
                properties[i][0] = self.alias[key]
            #properties[i][1] = self.alias_fmt[fmt]

        for p in system.particle:
            line = ''
            for i in range(len(properties)):
                key, fmt, ndims =  properties[i]
                val = getattr(p, key)
                if ndims == 1:
                    line += '{} '.format(val)
                else:
                    line += numpy.array2string(val, precision=self.precision, separator=' ')[1:-1] + ' '
            self.trajectory.write(line.strip() + '\n')
            
