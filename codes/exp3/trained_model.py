from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

import csv
import time as real_time
import random
import gym
import numpy as np

class Agent:
    ## Constructor
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.discount_rate = 0.9
        self.exp_rate = 0.2  # exploration rate
        self.learning_rate = 0.00005
        self.model = self._build_model()
        self.last_action = 0
        self.act_sequence = 0
        
    ## Network Structure
    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=832, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mape'])
        return model

    ## Save experience to memory
    def save_xp(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    ## Choose an action
    def act(self, state, repeat_action):
    
        # If agent must choose a new action
        if self.act_sequence == 0:
            if np.random.rand() <= self.exp_rate:
                action = random.randint(0, self.action_size - 1)
            else:
                act_values = self.model.predict(state)
                action = np.argmax(act_values[0])
                
        # If agent must repeat the last action
        else:
            action = self.last_action

        self.act_sequence = self.act_sequence + 1
        if self.act_sequence % repeat_action == 0:
            self.act_sequence = 0

        self.last_action = action
        return action
        
    ## Train model on batch
    def batch_train(self, batch_size):

        batch_size = min(batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        train_x = np.zeros((batch_size, self.state_size))
        train_y = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_rate * np.amax(self.model.predict(next_state)[0])

            train_x[i] = state
            train_y[i] = target

        history = self.model.evaluate(train_x, train_y, batch_size=batch_size, verbose=0)
        return history
        
    ## Load weights of trained model
    def load_model(self, name):
        self.model.load_weights(name)

## Main function
if __name__ == "__main__":
    env = gym.make('SuperMarioBros-3-2-Tiles-v0')
    wrapper = gym.wrappers.ToDiscrete()
    env = wrapper(env)

    state_size = 832
    action_size = 14
    agent = Agent(state_size, action_size)
    done = False
    batch_size = 512
    observation = env.reset()
    episode_count = 100
    repeat_action = 2
    record_list = []

    agent.load_model("weights.h5")

    for episode in range(1, episode_count + 1):
        state = env.reset()
        # One hot encoding
        state = to_categorical(state, 4)
        state = np.reshape(state, [1, state_size])

        reward_list = []
        total_reward = 0
        start_time = real_time.time()
        for frame in range(1, 100000):
            action = agent.act(state, repeat_action)      
            next_state, reward, done, info = env.step(action)
            
            # One hot encoding
            next_state = to_categorical(next_state, 4)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Bug fix
            if reward > 20:
                reward = 0

            # Negativity bias
            if reward == 0:
                reward = -0.5

            # If game is finished
            if done and 'life' in info:
                # If time is not zero
                if info['time']!=0:
                    # If mario died 
                    if info['life'] == 0:
                        reward = -50
                    # If mario passed the level 
                    else:
                        reward = 100

            # Save values and proceed
            reward_list.append(reward)
            agent.save_xp(state, action, reward, next_state, done)                
            state = next_state

            # Finish episode
            if done:
                score = info['distance']
                break

        # Evaluate at the end of an episode
        history = agent.batch_train(batch_size)  
        mape = round(history[1], 2)

        # Print values
        time_elapsed = round(real_time.time() - start_time, 2)
        average_reward = round(sum(reward_list) / frame, 2)
        print("episode: {}/{}, score: {}, frames: {}, average reward: {}, time: {}, mape: {}"\
            .format(episode, episode_count, score, frame, average_reward, time_elapsed, mape))
        
        # Record them for csv file
        record_tuple = (episode, score, frame, average_reward, time_elapsed, mape)
        record_list.append(record_tuple)
        
    ## Save results to csv file
    file_name = "results.csv"
    with open(file_name, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        write_data = [record_list]
        for row in record_list:
            writer.writerow(row)
            
    print("The results of training are saved to csv file.")

