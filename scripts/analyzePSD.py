#!/usr/bin/env python3.8

import rospy
import rosbag
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib.pyplot as plt

##################################
#### Extract data from bagfile ###
##################################

# Path to the bag file
bagfile_path = "../bagfiles/contact_force.bag"

# Topics containing the time data
wrench_applied = "/detection_experiment/wrench_applied"
wrench_external = "/detection_experiment/wrench_external"
time_topics = [wrench_applied, wrench_external]

# Read the bag file
bag = rosbag.Bag(bagfile_path)
total_samples = 200000  # 136005 (operator) 122437 (contact) 93093 (free)

wrench_app_arr = np.zeros([6, total_samples])
# Iterate over messages in the bag file
counter = 0
for topic, msg, t in bag.read_messages(topics=[wrench_applied]):
    # Check if the message is of type WrenchStamped
    if msg._type == 'geometry_msgs/WrenchStamped':
        # Access the wrench data
        wrench = msg.wrench
        # Access the timestamp of the message
        timestamp = msg.header.stamp
        # Access other fields if needed (e.g., wrench.force.x, wrench.torque.y, etc.)
        wrench_app_arr[0, counter] = wrench.force.x
        wrench_app_arr[1, counter] = wrench.force.y
        wrench_app_arr[2, counter] = wrench.force.z
        wrench_app_arr[3, counter] = wrench.torque.x
        wrench_app_arr[4, counter] = wrench.torque.y
        wrench_app_arr[5, counter] = wrench.torque.z
        counter += 1
print(counter)
wrench_ext_arr = np.zeros([6, total_samples])
# Iterate over messages in the bag file
counter = 0
for topic, msg, t in bag.read_messages(topics=[wrench_external]):
    # Check if the message is of type WrenchStamped
    if msg._type == 'geometry_msgs/WrenchStamped':
        # Access the wrench data
        wrench = msg.wrench
        # Access the timestamp of the message
        timestamp = msg.header.stamp
        # Access other fields if needed (e.g., wrench.force.x, wrench.torque.y, etc.)
        wrench_ext_arr[0, counter] = wrench.force.x
        wrench_ext_arr[1, counter] = wrench.force.y
        wrench_ext_arr[2, counter] = wrench.force.z
        wrench_ext_arr[3, counter] = wrench.torque.x
        wrench_ext_arr[4, counter] = wrench.torque.y
        wrench_ext_arr[5, counter] = wrench.torque.z
        counter += 1
print(counter)
wrench_err_arr = wrench_ext_arr - wrench_app_arr
# Close the bag file
bag.close()

##################################
###### Process data with FFT #####
##################################

# Discrete time
time_step = 1/1000.
# Create a Hann window
window_size = 1000
window = np.hanning(window_size)

psd_fe_array = np.zeros((3, total_samples-1, int(window_size/2)))

for seg_index in range(0, total_samples-window_size):
    init_seg = seg_index
    final_seg = init_seg + window_size
    for dof_f in range(0, 3):
        wrench_err_segment = wrench_err_arr[dof_f, init_seg:final_seg]
        fft_values = np.fft.fft(window*wrench_err_segment)[0:500]/window_size
        psd_fe_values = np.sqrt(np.real(fft_values*fft_values.conj()))
        psd_fe_array[dof_f, seg_index, :] = psd_fe_values

frequencies = np.fft.fftfreq(1000, d=time_step)[0:100]
good_instants_z = []
for instant in range(0, total_samples-window_size):
    psd_f = psd_fe_array[2, instant, 0:100]
    excitation_energy = np.sum(psd_f)
    if excitation_energy > 2:
        good_instants_z.append(instant)

num_f_ranges = 9
data_wrench_err_z = np.zeros((total_samples-window_size, num_f_ranges))

for instant in good_instants_z:
    frequency0 = psd_fe_array[2, instant, :][0]
    frequency1_2 = np.sum(psd_fe_array[2, instant, :][1:3])
    frequency3_6 = np.sum(psd_fe_array[2, instant, :][3:7])
    frequency7_15 = np.sum(psd_fe_array[2, instant, :][7:16])
    frequency16_25 = np.sum(psd_fe_array[2, instant, :][16:26])
    frequency26_40 = np.sum(psd_fe_array[2, instant, :][26:41])
    frequency41_60 = np.sum(psd_fe_array[2, instant, :][41:61])
    frequency61_85 = np.sum(psd_fe_array[2, instant, :][61:86])
    frequency86_110 = np.sum(psd_fe_array[2, instant, :][86:111])
    data_wrench_err_z[instant] = np.asarray([frequency0, frequency1_2, frequency3_6, frequency7_15, frequency16_25,
                                             frequency26_40, frequency41_60, frequency61_85, frequency86_110])
"""
for instant in good_instants_z:
    frequency_0 = psd_fe_array[2, instant, :][0]
    frequency_1 = np.sum(psd_fe_array[2, instant, :][1])
    frequency_2 = np.sum(psd_fe_array[2, instant, :][2])
    frequency_34 = np.sum(psd_fe_array[2, instant, :][3:5])
    frequency_56 = np.sum(psd_fe_array[2, instant, :][5:7])
    frequency_78 = np.sum(psd_fe_array[2, instant, :][7:9])
    frequency_910 = np.sum(psd_fe_array[2, instant, :][9:11])
    frequency_1113 = np.sum(psd_fe_array[2, instant, :][11:14])
    frequency_1417 = np.sum(psd_fe_array[2, instant, :][14:17])
    frequency_1720 = np.sum(psd_fe_array[2, instant, :][17:21])

    data_wrench_err_z[instant] = np.asarray([frequency_0, frequency_1, frequency_2, frequency_34, frequency_56,
                                             frequency_78, frequency_910, frequency_1113, frequency_1417, frequency_1720])
"""
print(len(good_instants_z))
for i in range(0, len(good_instants_z), 10000):

    plot_instant = good_instants_z[i]
    psd_f = psd_fe_array[2, plot_instant, 0:100]

    excitation_energy_5 = np.sum(psd_f)
    print("excitation energy in range 0-20: ", excitation_energy_5)

    plt.plot(frequencies, psd_f, color='red', linewidth=0.5)

    plt.xlim(0, 100)
    plt.xlabel('Frequencies')
    plt.ylabel('Magnitude')
    plt.title('Power Spectral Density')
    plt.grid(True)

    # Display the plot
    plt.show()

nonzero_rows = np.any(data_wrench_err_z != 0, axis=1)
filtered_arr = data_wrench_err_z[nonzero_rows]
# Save the array to a file
np.save('../Qp/psd_new/contact_force_array100.npy', filtered_arr)
