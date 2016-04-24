# This file is part of atooms
# Copyright 2010-2014, Daniele Coslovich

class DryRunBackend(object):

    def __init__(self):
        self.trajectory = Trajectory
        self.output_path = None

    def _get_system(self):
        pass

    def _set_system(self, value): 
        pass

    system = property(_get_system, _set_system, 'System')

    def write_checkpoint(self):
        pass

    def run_pre(self, restart):
        pass

    def run_until(self, n):
        self.steps = n
        pass


class System(object):
    
    def potential_energy(self):
        pass

    def temperature(self):
        pass

    def mean_square_displacement(self, reference):
        pass

    @property
    def thermostat(self):
        pass

class Trajectory(object):

    suffix = 'dry'

    def __init__(self, filename, mode='r'):
        pass

    def write_initial_state(self, system):
        pass

    def write_sample(self, system, step, sample):
        pass

    def close(self):
        pass
    
