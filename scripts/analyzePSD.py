#!/usr/bin/env python3.8

import rospy
import rosbag
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

##################################
#### Extract data from bagfile ###
##################################

# Path to the bag file
bagfile_path = "bagfiles/operator.bag"

# Topics containing the time data
wrench_applied = "/detection_experiment/wrench_applied"
wrench_external = "/detection_experiment/wrench_external"
time_topics = [wrench_applied, wrench_external]

# Read the bag file
bag = rosbag.Bag(bagfile_path)
total_samples = 105172  # 95924 57720 49792 105172

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

wrench_err_arr = wrench_ext_arr - wrench_app_arr
# Close the bag file
bag.close()

##################################
###### Process data with FFT #####
##################################

# Discrete time
time_step = 1/1000.
seg_l = 1000

psd_fe_array = np.zeros((3, total_samples-1, int(seg_l/2+1)))

for seg_index in range(0, total_samples-seg_l):
    init_seg = seg_index
    final_seg = init_seg + seg_l
    for dof_f in range(0, 3):
        wrench_err_segment = wrench_err_arr[dof_f, init_seg:final_seg]
        freqs, psd_fe_values = welch(wrench_err_segment, fs=1/time_step, nperseg=seg_l, noverlap=0)
        psd_fe_array[dof_f, seg_index, :] = psd_fe_values

good_instants_z = []
for instant in range(0, total_samples-seg_l):
    psd_f = psd_fe_array[2, instant, :]
    excitation_energy = np.sum(psd_f)
    if excitation_energy > 1:
        good_instants_z.append(instant)

num_f_ranges = 9
data_wrench_err_z = np.zeros((total_samples-seg_l, num_f_ranges))
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
for plot_instant in good_instants_z:

    psd_f = psd_fe_array[2, plot_instant, :]

    excitation_energy_5 = np.sum(psd_f[0:int(len(psd_f)/100)])
    print("excitation energy in range 0-5: ", excitation_energy_5)

    plt.plot(freqs, psd_f, color='red', linewidth=0.5)

    plt.xlim(0, 100)
    plt.xlabel('Frequencies')
    plt.ylabel('Magnitude')
    plt.title('Power Spectral Density')
    plt.grid(True)

    # Display the plot
    plt.show()
"""
nonzero_rows = np.any(data_wrench_err_z != 0, axis=1)
filtered_arr = data_wrench_err_z[nonzero_rows]
# Save the array to a file
np.save('PSD/operator_array.npy', filtered_arr)
