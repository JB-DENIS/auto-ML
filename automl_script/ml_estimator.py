import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

###########################
#  Classifiers and spaces #
###########################

# KNN
KNN = {
    "estimator": KNeighborsClassifier(),
    "space": {'leaf_size': [1, 5, 10],  # , 20, 30, 50],
              'n_neighbors': [5, 10],  # , 15, 20, 30],
              'p': [1, 2]
              }
}

# Randomforest
RFC = {
    "estimator": RandomForestClassifier(n_estimators=100),
    "space": {'max_depth': [10, 20, 50],
              'min_samples_leaf': [5, 10, 20],
              'max_features': ['sqrt', 'auto']  # ==> add X_train.shape[1]
              }
}


# SVC
SVClass = {
    "estimator": SVC(probability=True),
    "space": {'gamma': np.logspace(-2, 0, 3),
              'C': np.logspace(-2, 2, 5)
              }
}

# XGBoost
XGB = {
    "estimator": XGBClassifier(use_label_encoder=False),
    "space": {'learning_rate': [0.03, 0.05, 0.07, 0.1],
              'n_estimators': [100, 500, 1000, 2000]
              }
}

# LightGBM
LGBM = {
    "estimator": LGBMClassifier(),
    "space": {'num_leaves': [10, 30, 50],
              'min_child_samples': [10, 25, 50],
              # [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'min_child_weight': [1e-1, 1, 1e2],
              # [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_alpha': [0, 1, 10, 100],
              'reg_lambda': [0, 1, 10, 100]  # [0, 1e-1, 1, 5, 10, 20, 50, 100]
              }
}


dic_classifier = {}
dic_classifier["KNN"] = KNN
dic_classifier["RFC"] = RFC
dic_classifier["SVClass"] = SVClass
dic_classifier["XGB"] = XGB
dic_classifier["LGBM"] = LGBM

############################
#  Regressors  and spaces  #
############################
# Elasticnet
ELNReg = {
    "estimator": ElasticNet(),
    "space": {"alpha": [0.01, 0.1, 1],
              "l1_ratio": [0.1, 0.5, 0.9]
              }
}

# KNN
KNReg = {
    "estimator": KNeighborsRegressor(),
    "space": {'leaf_size': [1, 5, 10],  # , 20, 30, 50],
              'n_neighbors': [5, 10],  # , 15, 20, 30],
              'p': [1, 2]
              }
}

# Randomforest
RFReg = {
    "estimator": RandomForestRegressor(n_estimators=500),
    "space": {'max_depth': [10, 20, 50],
              'min_samples_leaf': [5, 10, 20],
              'max_features': ['sqrt', 'auto']  # ==> add X_train.shape[1]
              }
}

# SVC
SVReg = {
    "estimator": SVR(),
    "space": {'gamma': np.logspace(-2, 0, 3),
              'C': np.logspace(-2, 2, 5)
              }
}

# XGBoost
XGBReg = {
    "estimator": XGBRegressor(),
    "space": {'learning_rate': [0.03, 0.05, 0.07, 0.1],
              'n_estimators': [100, 500, 1000, 2000]
              }
}

# LightGBM
LGBMReg = {
    "estimator": LGBMRegressor(),
    "space": {'num_leaves': [10, 30, 50],
              'min_child_samples': [10, 25, 50],
              # [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
              'min_child_weight': [1e-1, 1, 1e2],
              # [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
              'reg_alpha': [0, 1, 10, 100],
              'reg_lambda': [0, 1, 10, 100]  # [0, 1e-1, 1, 5, 10, 20, 50, 100]
              }
}


dic_regressor = {}
dic_regressor["ELNREg"] = ELNReg
dic_regressor["KNReg"] = KNReg
dic_regressor["RFReg"] = RFReg
dic_regressor["SVReg"] = SVReg
dic_regressor["XGBReg"] = XGBReg
dic_regressor["LGBMReg"] = LGBMReg
