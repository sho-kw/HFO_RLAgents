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
        self.model = Model([state_input, action_switch], dot([self.core_model(state_input), one_hot(action_switch)], axes=1))
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
        else:
            self.target_core_model = model_from_json(self.core_model.to_json())
        self.episode_count = 0
        self.train_count = 0
        self.train_history = []
        # compile
        self.model.compile(optimizer, loss=loss)
        self._sync_target_model()

        ###experience replay
        if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            retmainQs = self.main_core_model.predict(next_state_b)[0]
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            target = reward_b + self.gamma * self.target_core_model.model.predict(next_state_b)[0][next_action]
            
        targets[i] = self.model.predict(state_b)    # Qネットワークの出力
        targets[i][action_b] = target               # 教師信号
        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
    ###

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

    ### select action

    def _select_action(self, state):
        scores = self.target_core_model.predict_on_batch(np.asarray([state]))[0]
        action = self.policy(scores)
        return action
    def select_best_action(self, state):
        scores = self.target_core_model.predict_on_batch(np.asarray([state]))[0]
        action = self.eval_policy(scores)
        return action

    # training
    def _gen_training_data(self):
            states, actions, next_states, rewards, cont_flags = self.memory.sample(self.batch_size)
            ###
            retmainQs = self.main_core_model.predict(next_states)[0]
            next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
            inputs = [states, actions]
            target = rewards + cont_flags * self.gamma * self.target_core_model.predict(next_state)[0][next_action]
            
            targets[i] = self.main_core_model.predict(states)    # Qネットワークの出力
            targets[i][actions] = target               # 教師信号

            return inputs, targets
    def _train(self):#バッチ学習mainQN.replay(memory, batch_size, gamma, targetQN)がしたい
        if self.warmup < self.memory._get_current_size():# Qネットワークの重みを学習・更新する replay
            x, y = self._gen_training_data()
            history = self.main_core_model.train_on_batch(x, y)
            self.train_history.append(history)
            #
            self.train_count += 1
            if self.target_model_update > 1 and self.train_count % self.target_model_update == 0:
                self._sync_target_model()
    def _sync_target_model(self):
        if self.target_model_update != 1:
            self.target_core_model.set_weights(self.core_model.get_weights())








# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
self.target_core_model = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化


# [5.3]メインルーチン--------------------------------------------------------
for episode in range(num_episodes):  # 試行数分繰り返す
    env.reset()  # cartPoleの環境初期化
    state, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
    state = np.reshape(state, [1, 4])   # list型のstateを、1行4列の行列に変換
    episode_reward = 0
 
    self.target_core_model = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
 
    for t in range(max_number_of_steps + 1):  # 1試行のループ
        if (islearned == 1) and LENDER_MODE:  # 学習終了したらcartPoleを描画する
            env.render()
            time.sleep(0.1)
            print(state[0, 0])  # カートのx位置を出力するならコメントはずす
 
        action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
        next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
        next_state = np.reshape(next_state, [1, 4])     # list型のstateを、1行4列の行列に変換

 
        # 1施行終了時の処理
        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
            break