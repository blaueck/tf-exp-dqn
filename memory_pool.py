import numpy as np


# See also: https://github.com/google/dopamine
class MemoryPool:

    def __init__(self, size=20000, state_size=(84, 84), n_frame=4, horizon=1, gamma=0.99):

        # memory data
        self.state = np.empty([size, *state_size], dtype='uint8')
        self.action = np.empty(size, dtype='int32')
        self.reward = np.empty(size, dtype='float32')
        self.done = np.empty(size, dtype='bool')
        self.done[-1] = True

        # variable for sample
        self.cursor = 0
        self.full = False
        self.size = size
        self.invalid_range = [0]
        self.n_frame = n_frame

        self.horizon = horizon
        self.cumulative_discount_vector = np.array(
            [np.power(gamma, i) for i in range(horizon)],
            dtype='float32')

    def _add(self, state, action, reward, done):
        self.state[self.cursor, ...] = state
        self.reward[self.cursor] = reward
        self.done[self.cursor] = done
        self.action[self.cursor] = action

        # the range which can not be sampled as state
        self.invalid_range = [
            i % self.size if i >= 0 else i + self.size
            for i in range(self.cursor - self.horizon + 1,
                           self.cursor + self.n_frame)]

        self.cursor += 1
        if self.cursor >= self.size:
            self.full = True
            self.cursor = 0

    def add(self, state, action, reward, done):

        i = self.cursor - 1
        i = i if i >= 0 else i + self.size
        if self.done[i]:
            for _ in range(self.n_frame - 1):
                self._add(0, 0, 0, False)

        self._add(state, action, reward, done)

    def _get_index(self, i):
        idx = [i + k for k in range(-self.n_frame + 1, 1)]
        idx = [i if i >= 0 else i + self.size for i in idx]
        return idx

    def _is_valid(self, idx):
        if idx >= self.size:
            return False

        if (not self.full and not (self.n_frame - 1 <= idx < self.cursor - self.horizon)):
            return False

        if idx in self.invalid_range:
            return False

        if np.any(self.done[self._get_index(idx)]):
            return False

        return True

    def sample_indices(self, batch_size):
        if (not self.full and self.cursor < batch_size + self.horizon
                or self.size < batch_size + self.horizon):
            raise RuntimeError('Wrong batch size.')

        cmax = self.cursor - self.horizon
        if self.full:
            cmin = self.cursor - self.size + self.n_frame - 1
        else:
            cmin = self.n_frame - 1

        # sample indices
        indices = []
        while len(indices) != batch_size:
            i = np.random.randint(cmin, cmax)
            i = i if i >= 0 else i + self.size

            if not self._is_valid(i):
                continue

            indices.append(i)

        return indices

    def sample(self, batch_size, indices=None):

        if indices is None:
            indices = self.sample_indices(batch_size)

        # prepair for sample
        state_idx = [self._get_index(i) for i in indices]
        state_idx_next, done, reward = [], [], []
        for i in indices:
            cur_idx = [(i + k) % self.size for k in range(self.horizon)]

            is_done = self.done[cur_idx]
            if not np.any(is_done):
                horizon_len = self.horizon
            else:
                horizon_len = np.argmax(is_done)

            next_i = (i + horizon_len) % self.size

            r = np.dot(self.reward[cur_idx][:horizon_len],
                       self.cumulative_discount_vector[:horizon_len])

            state_idx_next.append(self._get_index(next_i))
            done.append(self.done[next_i])
            reward.append(r)

        return (self.state[state_idx], self.action[indices],
                np.array(reward, dtype='float32'), np.array(
                    done, dtype='bool'),
                self.state[state_idx_next])


class SumTree:

    def __init__(self, size=20000):
        self.nodes = []

        tree_depth = int(np.ceil(np.log2(size)))
        level_size = 1
        for _ in range(tree_depth + 1):
            self.nodes.append(np.zeros(level_size, dtype='float32'))
            level_size *= 2

        self.max_value = 1.

    def get(self, index):
        return self.nodes[-1][index]

    def set(self, index, value):
        if value < 0.:
            raise ValueError('value should be nonnegative.')

        self.max_value = max(self.max_value, value)
        delta = value - self.nodes[-1][index]

        for node in reversed(self.nodes):
            node[index] += delta
            index //= 2

    def total_sum(self):
        return self.nodes[0][0]

    def sample(self, query_value=None):
        if self.total_sum() == 0.:
            raise Exception('the tree is empty.')

        query_value = np.random.rand() if query_value is None else query_value
        query_value *= self.total_sum()
        index = 0
        for node in self.nodes[1:]:
            index *= 2
            lsum = node[index]
            if query_value < lsum:
                index = index
            else:
                query_value -= lsum
                index = index + 1

        return index

    def stratified_sample(self, batch_size):
        if self.total_sum() == 0.0:
            raise Exception('the tree is empty.')

        bounds = np.linspace(0., 1., batch_size + 1)
        segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
        query_values = [np.random.uniform(x[0], x[1]) for x in segments]
        return [self.sample(query_value=x) for x in query_values]


class PriorityMemoryPool(MemoryPool):

    def __init__(self, size=20000, *args, **argv):
        super(PriorityMemoryPool, self).__init__(size, *args, **argv)
        self.tree = SumTree(size)

    def get_priority(self, indices):
        priorities = np.empty(len(indices), dtype='float32')
        for i, idx in enumerate(indices):
            priorities[i] = self.tree.get(idx)
        return priorities

    def set_priority(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.tree.set(i, p)

    def add(self, state, action, reward, done, priority=None):
        if priority is None:
            priority = self.tree.max_value
        self.tree.set(self.cursor, priority)
        super(PriorityMemoryPool, self).add(state, action, reward, done)

    def sample_indices(self, batch_size):
        indices = self.tree.stratified_sample(batch_size)
        for i in range(len(indices)):
            idx = indices[i]
            indices[i] = -1
            while not self._is_valid(idx):
                idx = self.tree.sample()
            indices[i] = idx
        return indices

    def sample(self, batch_size, indices=None):
        if indices is None:
            indices = self.sample_indices(batch_size)
        ret = super(PriorityMemoryPool, self).sample(batch_size, indices)
        priorities = self.get_priority(indices)
        return ret + (indices, priorities)