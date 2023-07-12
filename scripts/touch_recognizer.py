#!/usr/bin/env python3.8

import numpy as np
from joblib import load
import rospy
from geometry_msgs.msg import WrenchStamped
from scipy.signal import welch


class WrenchListenerNode:
    def __init__(self):
        self.wrench_applied = []
        self.wrench_external = []

        self.max_norm = 144.17156909948386
        self.time_step = 1/1000.
        # Load the saved model
        self.gmm_loaded = load('models/gmm_model_10.joblib')

    def wrench_applied_callback(self, msg):
        # Extract the linear force vector
        force = msg.wrench.force

        # Store the force vector in the array
        self.wrench_applied.append([force.x, force.y, force.z])

        # If the array size exceeds 1000, remove the oldest entry
        if len(self.wrench_applied) > 1000:
            self.wrench_applied.pop(0)

    def wrench_external_callback(self, msg):
        # Extract the linear force vector
        force = msg.wrench.force

        # Store the force vector in the array
        self.wrench_external.append([force.x, force.y, force.z])

        # If the array size exceeds 1000, remove the oldest entry
        if len(self.wrench_external) > 1000:
            self.wrench_external.pop(0)

    def compute_power_spectral_density(self):
        if len(self.wrench_external) != 1000 or len(self.wrench_applied) != 1000:
            return None, None

        # Convert wrench_msgs to a numpy array
        wrench_external_array = np.array(self.wrench_external)
        wrench_applied_array = np.array(self.wrench_applied)
        wrench_error_array = wrench_external_array - wrench_applied_array

        # Compute power spectral density using Welch's method
        frequencies_wrench, psd_wrench = welch(wrench_error_array.T, fs=1/self.time_step, nperseg=1000, noverlap=0)
        # psd_wrench: 3x501
        return frequencies_wrench, psd_wrench

    def contact_identification(self, dof, power_spectral_density):
        data = self.extract_data(dof, power_spectral_density)
        if data[-1, -1]*self.max_norm < 1:
            prediction = "No interaction"
        else:
            label = self.gmm_loaded.predict(data)
            if (label == 1) or (label == 11) or (label == 5):
                prediction = "Free motion"
            elif (label == 6) or (label == 7) or (label == 8):
                prediction = "Contact with the environment"
            elif (label == 0) or (label == 2) or (label == 3) or (label == 4):
                prediction = "Interation with the operator"
            else:
                prediction = "Undefined"

        return prediction

    def extract_data(self, dof, power_spectral_density):
        frequency0 = power_spectral_density[dof, 0]
        frequency1_2 = np.sum(power_spectral_density[dof,  1:3])
        frequency3_6 = np.sum(power_spectral_density[dof, 3:7])
        frequency7_15 = np.sum(power_spectral_density[dof, 7:16])
        frequency16_25 = np.sum(power_spectral_density[dof, 16:26])
        frequency26_40 = np.sum(power_spectral_density[dof, 26:41])
        frequency41_60 = np.sum(power_spectral_density[dof, 41:61])
        frequency61_85 = np.sum(power_spectral_density[dof, 61:86])
        frequency86_110 = np.sum(power_spectral_density[dof, 86:111])

        data = np.asarray([frequency0, frequency1_2, frequency3_6, frequency7_15, frequency16_25,
                           frequency26_40, frequency41_60, frequency61_85, frequency86_110])

        row_sums = np.sum(data)
        normalized_data = data/row_sums
        data_with_energy = np.hstack((normalized_data, row_sums/self.max_norm))
        data_2d = data_with_energy.reshape((1, -1))
        return data_2d

    def run(self):

        rospy.init_node('wrench_listener_node', anonymous=True)

        rospy.Subscriber("/detection_experiment/wrench_external", WrenchStamped, self.wrench_external_callback)
        rospy.Subscriber("/detection_experiment/wrench_applied", WrenchStamped, self.wrench_applied_callback)

        rate = rospy.Rate(100)  # Adjust the rate according to your requirements

        while not rospy.is_shutdown():
            frequencies, psd = self.compute_power_spectral_density()
            if frequencies is not None and psd is not None:
                prediction_x = self.contact_identification(0, psd)
                prediction_y = self.contact_identification(1, psd)
                prediction_z = self.contact_identification(2, psd)
                print(prediction_z)

            rate.sleep()


if __name__ == '__main__':
    node = WrenchListenerNode()
    print("Initialization completed")
    node.run()
