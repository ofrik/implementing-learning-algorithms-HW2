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
import pandas as pd
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")


def compare_algorithms(datasetName, data, target):
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=1)
    params = {'n_estimators': [10, 20, 30, 40], 'loss': ['ls', 'huber'], 'min_samples_leaf': [6],
              'max_depth': [3, 4, 5, 6]}

    print("Training GBRT on %s..." % datasetName)
    clf = GridSearchCV(GradientBoostingRegressor(), params, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Best params original: %s" % clf.best_params_)
    print("Avg train time original: %s seconds" % clf.cv_results_["mean_fit_time"][clf.best_index_])
    bestOriginal = clf.best_estimator_

    myclf = GridSearchCV(MyGradientBoostingRegressor(), params, cv=5, n_jobs=-1)
    myclf.fit(X_train, y_train)
    print("Best params mine: %s" % myclf.best_params_)
    print("Avg train time mine: %s seconds" % myclf.cv_results_["mean_fit_time"][myclf.best_index_])
    bestMine = myclf.best_estimator_

    originalPredictions = bestOriginal.predict(X_test)
    myPredicttions = bestMine.predict(X_test)
    print("The dataset: %s with %s train instances" % (datasetName, data.shape[0]))
    print("Original GradientBoostingRegressor R2: %s\tMSE: %s\tMAE: %s" % (
        r2_score(y_test, originalPredictions), mean_squared_error(y_test, originalPredictions),
        mean_absolute_error(y_test, originalPredictions)))
    print("My GradientBoostingRegressor R2: %s\tMSE: %s\tMAE: %s" % (
        r2_score(y_test, myPredicttions), mean_squared_error(y_test, myPredicttions),
        mean_absolute_error(y_test, myPredicttions)))


def main():
    # compare_algorithms("California Housing",cal_housing.data,cal_housing.target)

    self_noise_df = pd.read_csv(
        "data\\airfoil_self_noise.dat", sep="\t",
        names=["Frequency", "Angle", "Chord", "Velocity", "Thickness", "Sound_pressure"])
    self_noise_df_X = self_noise_df[self_noise_df.columns[:-1]]
    self_noise_df_Y = self_noise_df[self_noise_df.columns[-1:]]

    # maintenance_df = pd.read_csv("data\\maintenance.txt",
    #                              sep="   ", names=["f" + str(i) for i in range(18)])
    # maintenance_df_X = maintenance_df[maintenance_df.columns[:-2]]
    # maintenance_df_Y1 = maintenance_df[maintenance_df.columns[-2:-1]]
    # maintenance_df_Y2 = maintenance_df[maintenance_df.columns[-1:]]
    # compare_algorithms("Condition Based Maintenance of Naval Propulsion Plants Data Set", maintenance_df_X,
    #                    maintenance_df_Y2)

    parkinsons = pd.read_csv("data\\parkinsons_updrs.csv")
    parkinsons_Y = parkinsons["total_UPDRS"]
    parkinsons_X = parkinsons.drop(["total_UPDRS", "motor_UPDRS"], axis=1)

    facebook_comments_df = pd.read_csv(
        "data\\facebook_comments.csv",
        names=["f" + str(i) for i in range(54)])
    facebook_comments_df_X = facebook_comments_df[facebook_comments_df.columns[:-1]]
    facebook_comments_df_Y = facebook_comments_df[facebook_comments_df.columns[-1:]]

    popularity_df = pd.read_csv("data\\OnlineNewsPopularity.csv")
    popularity_df_X = popularity_df[popularity_df.columns[2:-1]]
    popularity_df_Y = popularity_df[popularity_df.columns[-1:]]

    blogdata_df = pd.read_csv("data\\blogData_train.csv",
                              names=["f" + str(i) for i in range(281)])
    blogdata_df_X = blogdata_df[blogdata_df.columns[:-1]]
    blogdata_df_Y = blogdata_df[blogdata_df.columns[-1:]]

    compare_algorithms("Airfoil Self Noise", self_noise_df_X, self_noise_df_Y)
    compare_algorithms("Parkinsons Telemonitoring Data Set", parkinsons_X, parkinsons_Y)
    compare_algorithms("Facebook Comment Volume Dataset Data Set", facebook_comments_df_X, facebook_comments_df_Y)
    compare_algorithms("Online News Popularity Data Set", popularity_df_X, popularity_df_Y)
    compare_algorithms("Blog Feedback Data Set", blogdata_df_X, blogdata_df_Y)


# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()
