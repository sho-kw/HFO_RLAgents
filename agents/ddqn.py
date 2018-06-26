import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Lambda, dot

from .core import AbstractAgent
from ..memory import SingleActionMemory
from ..policy import Greedy

class DQNAgent(AbstractAgent):
    def __init__(self,
                 action_space,
                 core_model,
                 optimizer,
                 policy,
                 memory,
                 *args,
                 loss='mean_squared_error',
                 gamma=0.99,
                 target_model_update=1,
                 warmup=100,
                 batch_size=32,
                 eval_policy=Greedy(),
                 **kwargs):
        super().__init__(action_space, *args, **kwargs)
        assert isinstance(action_space, int), action_space
        self.core_model = core_model
        state_input = self.core_model.input
        action_switch = Input(shape=(1,), dtype='int32')
        one_hot = Lambda(lambda x: K.squeeze(K.one_hot(x, action_space), axis=1), output_shape=(action_space,))
        self.target_model = Model([state_input, action_switch], dot([self.core_model(state_input), one_hot(action_switch)], axes=1))
        self.main_model = Model([state_input, action_switch], dot([self.core_model(state_input), one_hot(action_switch)], axes=1))
        #
        self.policy = policy
        if isinstance(memory, int):
            self.memory = SingleActionMemory(int(memory), state_input._keras_shape[1:])
        else:
            self.memory = memory
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.warmup = warmup
        self.batch_size = batch_size
        self.eval_policy = eval_policy
        #
        if target_model_update == 1:
            self.target_core_model = self.core_model
            self.main_core_model = self.core_model
        else:
            self.target_core_model = model_from_json(self.core_model.to_json())
            self.main_core_model = model_from_json(self.core_model.to_json())
        self.episode_count = 0
        self.train_count = 0
        self.train_history = []
        # compile
        self.target_model.compile(optimizer, loss=loss)
        self.main_model.compile(optimizer, loss=loss)
        self._sync_target_model()
    # primary method
    def start_episode(self, state):
        self.memory.start_episode(state)
        action = self._select_action(state)
        self.memory.set_action(action)
        return action
    def step(self, state, reward):
        self.memory.step(state, reward)
        self._train()#更新
        action = self._select_action(state)
        self.memory.set_action(action)
        return action
    def end_episode(self, state, reward):
        self.memory.end_episode(state, reward)
        self._train()#更新
        self.episode_count += 1
    # select action
    def _select_action(self, state):
        scores = self.main_core_model.predict_on_batch(np.asarray([state]))[0]
        action = self.policy(scores)
        return action
    def select_best_action(self, state):
        scores = self.main_core_model.predict_on_batch(np.asarray([state]))[0]
        action = self.eval_policy(scores)
        return action
    # training
    def _gen_training_data(self):
            states, actions, next_states, rewards, cont_flags = self.memory.sample(self.batch_size)
            pred_Q = self.main_core_model.predict(next_states)
            max_Q = np.max(pred_Q, axis=-1)  # 最大の報酬を返す行動を選択する
            inputs = [states, actions]
            targets = rewards + cont_flags * self.gamma * self.target_core_model.predict(next_states)
            return inputs, targets
    def _train(self):
        if self.warmup < self.memory._get_current_size():# Qネットワークの重みを学習・更新する replay
            x, y = self._gen_training_data()
            history = self.main_model.train_on_batch(x, y)
            self.train_history.append(history)
            #
            self.train_count += 1
            if self.target_model_update > 1 and self.train_count % self.target_model_update == 0:
                self._sync_target_model()
    def _sync_target_model(self):
        if self.target_model_update != 1:
            self.main_core_model.set_weights(self.core_model.get_weights())