class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
 
    def add(self, experience):
        self.buffer.append(experience)
 
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
 
    def len(self):
        return len(self.buffer)

def replay(self, target_core_model):
        inputs = np.zeros((self.batch_size, state_input))
        targets = np.zeros((self.batch_size, action_space))
        mini_batch = self.memory.sample(self.batch_size)###!!!
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b
        if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            retmainQs = self.model.predict(next_state_b)[0]
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + self.gamma * target_core_model.model.predict(next_state_b)[0][next_action]
            
        targets[i] = self.model.predict(state_b)    # Qネットワークの出力
        targets[i][action_b] = target               # 教師信号
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

import numpy as np

class SingleActionMemory:
    def __init__(self, capacity, state_shape, continuous_action=False):
        self.capacity = capacity
        self._next = 0
        self.filled = (capacity == self._next)
        #
        self.s = np.zeros(((capacity,) + state_shape), dtype=np.float32)
        if continuous_action:
            self.a = np.zeros(capacity, dtype=np.float32)
        else:
            self.a = np.zeros(capacity, dtype=np.int32)
        self.ns = np.zeros_like(self.s)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.c = np.zeros(capacity, dtype=np.bool)
        #
        self.last_state = None
        self.last_action = None
    def start_episode(self, state):
        assert self.last_state is None
        assert self.last_action is None
        self.last_state = state
    def step(self, state, reward):
        assert self.last_state is not None
        assert self.last_action is not None
        #
        self._push(state, reward, cont=True)
        self.last_state = state
    def end_episode(self, state, reward):
        if self.last_state is None:
            return
        #
        self._push(state, reward, cont=False)
        #
        self.last_state = None
        self.last_action = None
    def set_action(self, action):
        self.last_action = action
    def _push(self, next_state, reward, cont):
        p = self._next
        self.s[p] = self.last_state
        self.a[p] = self.last_action
        self.ns[p] = next_state
        self.r[p] = reward
        self.c[p] = cont
        #
        self._next += 1
        self._next %= self.capacity
        if self._next == 0:
            self.filled = True
    def _get_current_size(self):
        return self.capacity if self.filled else self._next
    def sample(self, num):
        assert self._get_current_size() >= num
        idx = np.random.choice(self._get_current_size(), size=num, replace=False)
        return self.s[idx], self.a[idx], self.ns[idx], self.r[idx], self.c[idx]
