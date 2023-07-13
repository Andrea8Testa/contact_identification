#!/usr/bin/env python3.8

import numpy as np
from joblib import load
import rospy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3


class WrenchListenerNode:
    def __init__(self):
        self.wrench_applied = []
        self.wrench_external = []

        self.max_norm = 24.649895669582985
        self.time_step = 1/1000.
        # Create a Hann window
        self.window_size = 1000
        self.window = np.hanning(self.window_size)
        # Load the saved model
        self.gmm_loaded = load('../models/gmm_model_15.joblib')

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
            return None

        # Convert wrench_msgs to a numpy array
        wrench_external_array = np.array(self.wrench_external)
        wrench_applied_array = np.array(self.wrench_applied)
        wrench_error_array = wrench_external_array - wrench_applied_array

        fft_values_x = np.fft.fft(self.window*wrench_error_array[:, 0])[0:500]/self.window_size
        psd_fe_values_x = np.sqrt(np.real(fft_values_x*fft_values_x.conj()))

        fft_values_y = np.fft.fft(self.window*wrench_error_array[:, 1])[0:500]/self.window_size
        psd_fe_values_y = np.sqrt(np.real(fft_values_y*fft_values_y.conj()))

        fft_values_z = np.fft.fft(self.window*wrench_error_array[:, 2])[0:500]/self.window_size
        psd_fe_values_z = np.sqrt(np.real(fft_values_z*fft_values_z.conj()))

        psd_wrench = np.vstack((psd_fe_values_x, psd_fe_values_y, psd_fe_values_z))
        # psd_wrench: 3x500
        return psd_wrench

    def contact_identification(self, dof, power_spectral_density):
        data = self.extract_data(dof, power_spectral_density)
        if data[-1, -1]*self.max_norm < 1:
            prediction = "No interaction"
        else:
            label = self.gmm_loaded.predict(data)
            # if dof == 2:
            #    print(label)
            if (label == 2) or (label == 4) or (label == 6) or (label == 8) or (label == 9) or (label == 12):
                prediction = 0  # Free motion
            elif (label == 1) or (label == 3) or (label == 7) or (label == 10):
                prediction = 1  # Contact with the environment
            elif (label == 0) or (label == 5) or (label == 11) or (label == 13) or (label == 14):
                prediction = 2  # Interation with the operator
            else:
                prediction = 3  # Undefined

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

        energy = np.sum(data)
        normalized_data = data/energy
        data_with_energy = np.hstack((normalized_data, energy/self.max_norm))
        data_2d = data_with_energy.reshape((1, -1))
        return data_2d

    def run(self):

        rospy.init_node('wrench_listener_node', anonymous=True)

        rospy.Subscriber("/detection_experiment/wrench_external", WrenchStamped, self.wrench_external_callback)
        rospy.Subscriber("/detection_experiment/wrench_applied", WrenchStamped, self.wrench_applied_callback)
        pub = rospy.Publisher("/adaptive_qp/touch_detection", Vector3, queue_size=1)

        rate = rospy.Rate(100)  # Adjust the rate according to your requirements
        while not rospy.is_shutdown():
            psd = self.compute_power_spectral_density()
            msg = Vector3()
            msg.x = 0
            msg.y = 0
            msg.z = 0
            if psd is not None:
                prediction_x = self.contact_identification(0, psd)
                prediction_y = self.contact_identification(1, psd)
                prediction_z = self.contact_identification(2, psd)
                print(prediction_z)
                msg.x = prediction_x
                msg.y = prediction_y
                msg.z = prediction_z

            pub.publish(msg)
            rate.sleep()


if __name__ == '__main__':
    node = WrenchListenerNode()
    print("Initialization completed")
    node.run()
