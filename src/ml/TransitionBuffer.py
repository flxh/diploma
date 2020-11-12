import numpy as np


class Transition:
    def __init__(self, state=None, action=None, reward=None, next_state=None, done=None, aux_info=None, gae=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.aux_info = aux_info
        self.done = done
        self.gae = gae


class TransitionBuffer(list):
    def __init__(self, vals=[]):
        for v in vals:
            self._check_new_value(v)
        super(TransitionBuffer, self).__init__(vals)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return TransitionBuffer(super(TransitionBuffer, self).__getitem__(key))
        else:
            return super(TransitionBuffer, self).__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise NotImplementedError('Slice insertion not implemented')
        self._check_new_value(value)
        super(TransitionBuffer, self).__setitem__(key, value)

    def __add__(self,other):
        return TransitionBuffer(list.__add__(self,other))

    def __mul__(self,other):
        return TransitionBuffer(list.__mul__(self,other))

    def copy(self):
        return TransitionBuffer(super(TransitionBuffer,self).copy())

    @property
    def states(self):
        return np.array([x.state for x in self])

    @property
    def actions(self):
        return np.array([x.action for x in self])

    @property
    def rewards(self):
        return np.array([x.reward for x in self])

    @property
    def next_states(self):
        return np.array([x.next_state for x in self])

    @property
    def done(self):
        return np.array([x.done for x in self])

    @property
    def aux_info(self):
        return np.array([x.aux_info for x in self])

    @property
    def gae_values(self):
        return np.array([x.gae for x in self])

    def _check_new_value(self, val):
        if not isinstance(val, Transition):
            raise ValueError('Inserted object must be instance of class Transition. ')

    def append(self, val):
        self._check_new_value(val)
        super(TransitionBuffer, self).append(val)

    def extend(self, obj):
        raise NotImplementedError('Method extend is not supported. Use list addition!')

    def add_gea_values(self, gae):
        if len(self) != len(gae):
            raise ValueError('Buffer and list of GAE values must be equal in length!')

        for idx, gae in enumerate(gae):
            self[idx].gae = gae

