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
        self.stacked_array = np.empty((0, 10))
        self.stacked_vector = np.empty((0, 1))
        self.super_label = 0
        self.knc_loaded = pickle.load(open('../knc/knnpickle_last', 'rb'))

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
        if np.sum(data) < 1e-5:
            label = 0
        else:
            label = self.knc_loaded.predict(data.reshape(1, -1))[0]
            # self.stacked_array = np.vstack((self.stacked_array, data.reshape(1, -1)))
            # self.stacked_vector = np.vstack((self.stacked_vector, np.array(self.super_label)))
            #label = 0

        return label

    def extract_data(self, dof, power_spectral_density):

        frequency_0 = power_spectral_density[dof, 0]
        frequency_1 = power_spectral_density[dof, 1]
        frequency_2 = power_spectral_density[dof, 2]
        frequency_34 = np.sum(power_spectral_density[dof, 3:5])
        frequency_56 = np.sum(power_spectral_density[dof, 5:7])
        frequency_78 = np.sum(power_spectral_density[dof, 7:9])
        frequency_910 = np.sum(power_spectral_density[dof, 9:11])
        frequency_1113 = np.sum(power_spectral_density[dof, 11:14])
        frequency_1417 = np.sum(power_spectral_density[dof, 14:17])
        frequency_1720 = np.sum(power_spectral_density[dof, 17:21])

        data = np.asarray([frequency_0, frequency_1, frequency_2, frequency_34, frequency_56,
                           frequency_78, frequency_910, frequency_1113, frequency_1417, frequency_1720])

        if np.sum(data[0:100]) < 2:
            normalized_data = np.zeros(10)
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
                print(prediction_z)
                msg.x = prediction_x
                msg.y = prediction_y
                msg.z = prediction_z

            pub.publish(msg)
            rate.sleep()
        """
        self.knc_train.fit(self.stacked_array, self.stacked_vector)
        knnPickle = open('../knc/knnpickle_last', 'wb')
        # source, destination
        pickle.dump(self.knc_train, knnPickle)
        # close the file
        knnPickle.close()
        print("model saved")
        """


if __name__ == '__main__':
    node = WrenchListenerNode()
    print("Initialization completed")
    node.run()
