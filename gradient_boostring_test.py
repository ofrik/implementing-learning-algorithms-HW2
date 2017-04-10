from __future__ import print_function

__author__ = 'Ofri'

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
# from gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from gradient_boosting import GradientBoostingRegressor as MyGradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing


def main():
    cal_housing = fetch_california_housing()

    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)
    names = cal_housing.feature_names

    print("Training GBRT...")
    params = {'n_estimators':100,'max_depth':4,'learning_rate':0.1,'loss':'huber','random_state':1,'verbose':1}
    clf = GradientBoostingRegressor(**params)

    myclf = MyGradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    myclf.fit(X_train, y_train)
    print(" done.")


    print("Original GradientBoostingRegressor Score: %s"%(clf.score(X_test,y_test)))
    print("My GradientBoostingRegressor Score: %s"%(myclf.score(X_test,y_test)))



# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()
