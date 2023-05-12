#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two-dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        # TODO: Implement me
        X = sorted(np.random.uniform(0, 1, m))
        Y = np.array([np.random.choice([0, 1], p=self.conditional_probability(x)) for x in X])
        return np.column_stack([X, Y])  # return a 2d array of shape (m,2)

    def experiment_m_range_erm(self, m_start, m_end, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two-dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # TODO: Implement the loop
        array_average_errors = []
        for n in range(m_start, m_end + 1, step):
            empirical_and_true_errors = [self.calculate_empirical_true_error_experiment(n, k) for _ in range(T)]
            average_empirical_and_true_error = [sum(error) / T for error in zip(*empirical_and_true_errors)]
            array_average_errors.append(average_empirical_and_true_error)
        np_array_average_errors = np.asarray(array_average_errors)
        # plot the average empirical and true errors as a function of the sample size
        n_axis = np.arange(m_start, m_end + 1, step)
        plt.title("ERM errors as a function of the sample size")
        plt.xlabel("sample size (n)")
        plt.plot(n_axis, np_array_average_errors[:, 0], label="Empirical Error")
        plt.plot(n_axis, np_array_average_errors[:, 1], label="True Error")
        plt.legend()
        plt.show()

        return np_array_average_errors

    def experiment_k_range_erm(self, m, k_start, k_end, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        sample = self.sample_from_D(m)
        errors = [self.calculate_empirical_true_error_for_sample(sample, m, k) for k in range(k_start, k_end + 1, step)]
        np_array_errors = np.array(errors)
        min_k = np.argmin(np_array_errors[:, 0]) * step + k_start  # find the k with the minimum empirical error
        # plot the empirical and true errors as a function of k
        k_axis = np.arange(k_start, k_end + 1, step)
        plt.title("ERM as a function of number of intervals (k)")
        plt.xlabel("number of intervals (k)")
        plt.plot(k_axis, np_array_errors[:, 0], label="Empirical Error")
        plt.plot(k_axis, np_array_errors[:, 1], label="True Error")
        plt.legend()
        plt.show()

        return min_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
                Input: m - an integer, the size of the data sample.

                Returns: The best k value (an integer) found by the cross validation algorithm.
                """
        # TODO: Implement me
        sample = self.sample_from_D(m)
        training_set, test_set = sample[:int(m * 0.8)], sample[int(m * 0.8):]  # split the sample to training and test
        # sets
        training_set = training_set[training_set[:, 0].argsort()]  # sort the training set by x
        best_hypotheses = [intervals.find_best_interval(training_set[:, 0], training_set[:, 1], k)[0] for k in
                           range(1, 11)]  # find the best hypothesis for each k
        test_errors = [self.calculate_empirical_error(test_set, intervals_list) for intervals_list in best_hypotheses]
        # calculate the test error for each k
        np_array_test_errors = np.array(test_errors)
        res = np.argmin(np_array_test_errors)  # find the k with the minimum test error
        return res + 1, best_hypotheses  # return the best k and the best hypotheses matching this k

    #################################
    # Place for additional methods
    def zero_one_loss(self, intervals_list, x, y):
        """Returns the zero-one loss for a given x and y."""
        x_in_intervals = self.is_in_intervals(intervals_list, x)
        if (y == 1 and x_in_intervals) or (y == 0 and not x_in_intervals):
            return 0
        return 1

    @staticmethod
    def conditional_probability(x):
        """Returns the conditional probability for a given x."""
        if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
            return [0.2, 0.8]
        return [0.9, 0.1]

    @staticmethod
    def length_of_intersection(list1, list2):
        """Returns the length of the intersection between two lists of intervals."""
        length_of_intersection = 0
        ptr_list1 = 0
        ptr_list2 = 0
        # iterate over the two lists and calculate the length of the intersection
        while ptr_list1 < len(list1) and ptr_list2 < len(list2):
            start = max(list1[ptr_list1][0], list2[ptr_list2][0])
            end = min(list1[ptr_list1][1], list2[ptr_list2][1])
            if start < end:
                length_of_intersection += (end - start)
            if list1[ptr_list1][1] == list2[ptr_list2][1]:
                ptr_list1 += 1
                ptr_list2 += 1
            elif list1[ptr_list1][1] < list2[ptr_list2][1]:
                ptr_list1 += 1
            else:
                ptr_list2 += 1
        return length_of_intersection

    def calculate_true_error(self, intervals_list):
        """Returns the true error for a given list of intervals."""
        intervals_1 = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        intervals_0 = [(0.2, 0.4), (0.6, 0.8)]
        intersection_high = self.length_of_intersection(intervals_list,
                                                        intervals_1)
        intersection_low = self.length_of_intersection(intervals_list,
                                                       intervals_0)
        high = 0.6 - intersection_high
        low = 0.4 - intersection_low
        return 0.8 * high + 0.1 * low + 0.2 * intersection_high + 0.9 * intersection_low

    def calculate_empirical_true_error_experiment(self, n, k):
        """Calculate the empirical error and the true error for a given n and k."""
        sample = self.sample_from_D(n)  # Draw a sample from D
        return self.calculate_empirical_true_error_for_sample(sample, n, k)

    def calculate_empirical_true_error_for_sample(self, sample, n, k):
        """Calculate the empirical error and the true error for the given sample."""
        best_intervals, error_count = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)  # apply ERM algorithm
        return error_count / n, self.calculate_true_error(best_intervals)

    @staticmethod
    def is_in_intervals(intervals_list, x):
        """Check if x in the intervals."""
        for interval in intervals_list:
            if interval[0] <= x <= interval[1]:  # interval [start, end] represented by the tuple (start, end)
                return True
        return False

    def calculate_empirical_error(self, sample: np.ndarray, intervals_list: list[tuple]) -> float:
        """Returns the empirical error for a given sample and a given list of intervals."""
        return sum([self.zero_one_loss(intervals_list, x, y) for x, y in sample]) / len(sample)
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    print(ass.cross_validation(1500))
