#!/usr/bin/env python3.8

import numpy as np
from joblib import load
import rospy
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import pickle
from sklearn.neighbors import KNeighborsClassifier


class WrenchListenerNode:
    def __init__(self):
        self.wrench_applied = []
        self.wrench_external = []

        self.time_step = 1/1000.
        # Create a Hann window
        self.window_size = 1000
        self.window = np.hanning(self.window_size)
        # Load the saved model
        self.knc_train = KNeighborsClassifier(n_neighbors=3)
        self.stacked_array = np.empty((0, 9))
        self.stacked_vector = np.empty((0, 1))
        self.super_label = 0
        self.knc_loaded = pickle.load(open('../knc/knnpickle_incr2', 'rb'))

    def supervised_label_callback(self, msg):
        self.super_label = msg.data
        print("supervised label changed to: ", self.super_label)

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
        wrench_error_array = wrench_external_array #- wrench_applied_array

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
        if np.sum(data) < 1e-5:
            label = 0
        else:
            label = self.knc_loaded.predict(data.reshape(1, -1))[0]
            self.stacked_array = np.vstack((self.stacked_array, data.reshape(1, -1)))
            self.stacked_vector = np.vstack((self.stacked_vector, np.array(self.super_label)))
            #label = 0

        return label

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

        if np.sum(data[0:100]) < 2:
            normalized_data = np.zeros(9)
        else:
            normalized_data = np.divide(data, np.max(data, axis=0))

        return normalized_data

    def run(self):

        rospy.init_node('wrench_listener_node', anonymous=True)

        rospy.Subscriber("/detection_experiment/wrench_external", WrenchStamped, self.wrench_external_callback)
        rospy.Subscriber("/detection_experiment/wrench_applied", WrenchStamped, self.wrench_applied_callback)
        rospy.Subscriber("/detection_experiment/supervised_label", Float64, self.supervised_label_callback)
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
           
                msg.x = prediction_x
                msg.y = prediction_y
                msg.z = prediction_z

            pub.publish(msg)
            rate.sleep()
        
        
        self.knc_train.fit(self.stacked_array, self.stacked_vector)
        knnPickle = open('../knc/knnpickle_no', 'wb')
        # source, destination
        pickle.dump(self.knc_train, knnPickle)
        # close the file
        knnPickle.close()
        print("model saved")
        


if __name__ == '__main__':
    node = WrenchListenerNode()
    print("Initialization completed")
    node.run()
