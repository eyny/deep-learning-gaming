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

row_size = 250
a_file_name = "a_results.csv"
b_file_name = "b_results.csv"
c_file_name = "c_results.csv"

a_result = result()
b_result = result()
c_result = result()

with open(a_file_name, newline='') as read_file:  
    reader = csv.reader(read_file)
    a_list = list(reader)

with open(b_file_name, newline='') as read_file:  
    reader = csv.reader(read_file)
    b_list = list(reader)

with open(c_file_name, newline='') as read_file:  
    reader = csv.reader(read_file)
    c_list = list(reader)

for i in range(row_size):
    a_result.episodes.append(int(a_list[i][0]))
    a_result.episode_score.append(int(a_list[i][1]))
    a_result.frames.append(int(a_list[i][2]))
    a_result.reward_per_frame.append(float(a_list[i][3]))
    a_result.times.append(float(a_list[i][4]))
    a_result.loss.append(float(a_list[i][5]))

    b_result.episodes.append(int(b_list[i][0]))
    b_result.episode_score.append(int(b_list[i][1]))
    b_result.frames.append(int(b_list[i][2]))
    b_result.reward_per_frame.append(float(b_list[i][3]))
    b_result.times.append(float(b_list[i][4]))
    b_result.loss.append(float(b_list[i][5]))

    c_result.episodes.append(int(c_list[i][0]))
    c_result.episode_score.append(int(c_list[i][1]))
    c_result.frames.append(int(c_list[i][2]))
    c_result.reward_per_frame.append(float(c_list[i][3]))
    c_result.times.append(float(c_list[i][4]))
    c_result.loss.append(float(c_list[i][5]))

#Loss
plt.figure(figsize=(7,5), linewidth=3, edgecolor = "black")

smooth_y = savgol_filter(a_result.loss, 51, 1)
plt.plot(a_result.episodes, smooth_y)
smooth_y = savgol_filter(b_result.loss, 51, 1)
plt.plot(b_result.episodes, smooth_y, dashes=[6, 2])
smooth_y = savgol_filter(c_result.loss, 51, 1)
plt.plot(c_result.episodes, smooth_y, dashes=[3, 2])

plt.title("Mean absolute percentage error")
plt.xlabel("Episodes")
plt.ylabel("Error")

plt.axis([0, 250, 0, 18])
plt.legend(["Model A", "Model B", "Model C"], loc='upper right')
plt.savefig("loss.png", edgecolor = "black")
plt.show()

#Reward
plt.figure(figsize=(7,5), linewidth=3, edgecolor = "black")

smooth_y = savgol_filter(a_result.reward_per_frame, 31, 3)
plt.plot(a_result.episodes, smooth_y)
smooth_y = savgol_filter(b_result.reward_per_frame, 31, 3)
plt.plot(b_result.episodes, smooth_y, dashes=[6, 2])
smooth_y = savgol_filter(c_result.reward_per_frame, 31, 3)
plt.plot(c_result.episodes, smooth_y, dashes=[3, 2])

plt.title("Average reward per frame")
plt.xlabel("Episodes")
plt.ylabel("Reward")

plt.legend(["Model A", "Model B", "Model C"], loc="upper left")
plt.savefig("reward.png", edgecolor = "black")
plt.show()

#Time
for i in range(row_size-1):
    a_result.times[i+1] = a_result.times[i] + a_result.times[i+1]
    b_result.times[i+1] = b_result.times[i] + b_result.times[i+1]
    c_result.times[i+1] = c_result.times[i] + c_result.times[i+1]


plt.figure(figsize=(7,5), linewidth=3, edgecolor = "black")

smooth_y = savgol_filter(a_result.times, 31, 3)
plt.plot(a_result.episodes, smooth_y)
smooth_y = savgol_filter(b_result.times, 31, 3)
plt.plot(b_result.episodes, smooth_y, dashes=[6, 2])
smooth_y = savgol_filter(c_result.times, 31, 3)
plt.plot(c_result.episodes, smooth_y, dashes=[3, 2])

plt.title("Total time elapsed")
plt.xlabel("Episodes")
plt.ylabel("Time in seconds")

plt.legend(["Model A", "Model B", "Model C"], loc="upper left")
plt.savefig("time.png", edgecolor = "black")
plt.show()


            

