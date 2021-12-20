import pandas as pd  # Data manipulation
import numpy as np  # Data manipulation
import matplotlib.pyplot as plt  # Visualization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


class InsuranceData:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.test_Ax = None
        self.test_b = None

    def normalize(self):
        # changing sex to 0 and 1
        self.df.loc[self.df['sex'] == "male", 'sex'] = 1
        self.df.loc[self.df['sex'] == "female", 'sex'] = 0

        # changing smoker to 0 and 1
        self.df.loc[self.df['smoker'] == "yes", 'smoker'] = 1
        self.df.loc[self.df['smoker'] == "no", 'smoker'] = 0

        # changing region to 0 - 3
        self.df.loc[self.df['region'] == "southwest", 'region'] = 0
        self.df.loc[self.df['region'] == "southeast", 'region'] = 1
        self.df.loc[self.df['region'] == "northwest", 'region'] = 2
        self.df.loc[self.df['region'] == "northeast", 'region'] = 3

        # reducing the mean
        self.df.charges = self.df.charges / 1000
        self.df.charges = self.df.charges - self.df['charges'].mean()
        self.df.bmi = self.df.bmi - self.df['bmi'].mean()
        self.df.age = self.df.age - self.df['age'].mean()

    def experiments(self, num_of_experiments):
        for i in range(num_of_experiments):
            # define train and test parts of dat frame.
            train, test = train_test_split(self.df, test_size=0.2)

            # define A and b and convert to numpy
            b = train.charges.to_numpy(dtype='float')
            A = train.drop(['charges'], axis=1).to_numpy(dtype='float')

            # calculate (find x)
            At = A.T
            AtA = At @ A
            Atb = At @ b
            AtA_inverse = np.linalg.inv(AtA)
            x = AtA_inverse @ Atb

            # define test data
            test_b = test.charges.to_numpy(dtype='float')
            test_A = test.drop(['charges'], axis=1).to_numpy(dtype='float')

            # compute the mean squared error for the test data
            test_Ax = test_A @ x
            Ax = A @ x

            testMSE = mean_squared_error(test_b, test_Ax)
            trainMSE = mean_squared_error(b, Ax)
            self.test_Ax = test_Ax
            self.test_b = test_b

            print(f"Experiment {i + 1}: The comparison between the MSE's is {testMSE / trainMSE:.2f}")

    def visualize(self):
        plt.hist(np.abs(self.test_Ax - self.test_b))
        plt.title("Distribution of error values")
        plt.xlabel('error values')
        plt.ylabel('frequency')
        plt.legend("error value")
        plt.show()

    def build_model(self, num_of_experiments=5):
        self.normalize()
        self.experiments(num_of_experiments)
        self.visualize()
