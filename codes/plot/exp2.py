## Graph generator from csv files

import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path
from scipy.signal import savgol_filter

class result:
    def __init__(self):
        self.episodes = []
        self.episode_score = []
        self.frames = []
        self.reward_per_frame = []
        self.times = []
        self.loss = []

row_size = 500
online_file_name = "results.csv"
online_result = result()

with open(online_file_name, newline='') as read_file:  
    reader = csv.reader(read_file)
    online_list = list(reader)

for i in range(row_size):
    online_result.episodes.append(int(online_list[i][0]))
    online_result.episode_score.append(int(online_list[i][1]))
    online_result.frames.append(int(online_list[i][2]))
    online_result.reward_per_frame.append(float(online_list[i][3]))
    online_result.times.append(float(online_list[i][4]))
    online_result.loss.append(float(online_list[i][5]))


plt.figure(figsize=(7,5), linewidth=3, edgecolor = "black")
smooth_y = savgol_filter(online_result.episode_score, 51, 1)
plt.plot(online_result.episodes, smooth_y, dashes=[6, 2], color="#ff7f0e")

plt.title("Earned score in an episode")
plt.xlabel("Episodes")
plt.ylabel("Score")

plt.savefig("score.png", edgecolor = "black")
plt.show()


            

