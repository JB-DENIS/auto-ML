
from typing import DefaultDict, Optional
from sklearn.exceptions import DataConversionWarning
import warnings
from abc import ABCMeta, abstractmethod

from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, \
    confusion_matrix, \
    accuracy_score, \
    mean_squared_error, \
    mean_absolute_error, \
    r2_score, \
    fbeta_score, \
    precision_recall_fscore_support

from automl_script.ml_estimator import dic_regressor, dic_classifier


class SupMLEngine(metaclass=ABCMeta):
    def __init__(self):
        self.status = bool
        self.features_name = []

    # Split data

    def split_data(self, feat: np.ndarray, target: np.ndarray, ratio_split: float = 0.3):
        """Split the value for train / test value

        Args:
            feat (np.ndarray): features
            target (np.ndarray): target
            ratio_split (float, optional): split ratio. Defaults to 0.3.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(feat,
                                                            target,
                                                            test_size=ratio_split,
                                                            random_state=42)
        return X_train, X_test, y_train, y_test

    def sampling_transform(self, X_train, y_train, sampling_strategy):
        """handle imbalance value

        Args:
            X_train (_type_): X
            y_train (_type_): Y
            sampling_strategy (_type_): over or under

        Returns:
            _type_: _description_
        """

        if sampling_strategy == "over":
            sm = SMOTE(random_state=42)
            X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
            print(sorted(Counter(y_resampled).items()))
            print("oversampling done")
            return X_resampled, y_resampled

        if sampling_strategy == "under":
            try:
                rus = RandomUnderSampler(random_state=42)
                X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                print(sorted(Counter(y_resampled).items()))
                print("undersampling done")

                return X_resampled, y_resampled

            except Exception as e:
                print("under error", e)

    def drop_outlier(self, feat, strategy):
        """Drop outlier

        Args:
            feat (_type_): _description_
            strategy (_type_): _description_

        Returns:
            _type_: _description_
        """
        if strategy == 'iqr':

            if isinstance(feat, np.ndarray):
                data = feat
            else:
                data = feat.values
            # calculate interquartile range
            q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
            iqr = q75 - q25
            print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' %
                  (q25, q75, iqr))
            # calculate the outlier cutoff
            cut_off = iqr * 1.5
            lower, upper = q25 - cut_off, q75 + cut_off
            # identify outliers
            outliers = [x for x in data if (
                x < lower).any() or (x > upper).any()]
            print('Identified outliers: %d' % len(outliers))
            # remove outliers
            outliers_removed = [x for x in data if (
                x >= lower).any() and (x <= upper).any()]
            print('Non-outlier observations: %d' % len(outliers_removed))

            return np.array(outliers_removed)

    def eval_metrics_cla(self, supervised_type: str, true_value: object, pred: object, estimator: str):
        """function with evaluation metrics

        Args:
            supervised_type (str): type of ML
            true_value (object): ytest
            pred (object): y from xtest
            estimator (str): estimator name

        Returns:
            dict with all the scores
        """
        result_scor = {}

        if supervised_type == "cla":

            result_scor["accuracy"] = accuracy_score(true_value, pred)

            f_resulte = precision_recall_fscore_support(
                true_value, pred, average="macro")
            result_scor["precision"] = f_resulte[0]
            result_scor["recall"] = f_resulte[1]
            result_scor["f0.5_score"] = fbeta_score(
                true_value, pred, beta=0.5, average="weighted")
            result_scor["f1_score"] = f_resulte[2]
            result_scor["f2_score"] = fbeta_score(
                true_value, pred, beta=2, average="weighted")

            cm = confusion_matrix(true_value, pred)
            try:
                self.showconfusionmatrix(cm, estimator='')
            except Exception as e:
                print(e)

        if supervised_type == "reg":
            result_scor['neg_mean_squared_error'] = np.sqrt(
                mean_squared_error(true_value, pred))  # TO CHECK
            result_scor['mae'] = mean_absolute_error(true_value, pred)
            result_scor['r2'] = r2_score(true_value, pred)

            self.plot_tru_mod(true_value=true_value,
                              pred=pred, estimator='')

        return result_scor

    ##### PLOTS ####

    # Confusion Matrix

    def showconfusionmatrix(self,
                            cf,
                            estimator,
                            group_names=None,
                            categories='auto',
                            count=True,
                            percent=True,
                            cbar=True,
                            xyticks=True,
                            xyplotlabels=True,
                            figsize=None,
                            cmap='Blues',
                            title='Confusion matrix'):
        '''
        Plot a Confusion Matrix 

        Arguments
        ---------
        cf:            confusion matrix to be passed in
        group_names:   List of strings that represent the labels row by row to be shown in each square.
        categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
        count:         If True, show the raw number in the confusion matrix. Default is True.
        normalize:     If True, show the proportions for each category. Default is True.
        cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                    Default is True.
        xyticks:       If True, show x and y ticks. Default is True.
        xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
        figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
        cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
        title:         Title for the heatmap. Default is None.
        '''

        # CODE TO GENERATE TEXT INSIDE EACH SQUARE
        blanks = ['' for i in range(cf.size)]

        if group_names and len(group_names) == cf.size:
            group_labels = ["{}\n".format(value) for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = ["{0:0.0f}\n".format(
                value) for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = ["{0:.2%}".format(
                value) for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(
            group_labels, group_counts, group_percentages)]
        box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

        # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
        if figsize == None:
            # Get default figure size if not set
            figsize = plt.rcParams.get('figure.figsize')

        if xyticks == False:
            # Do not show categories if xyticks is False
            categories = False

        # MAKE THE HEATMAP VISUALIZATION
        fig = plt.figure(figsize=figsize)
        sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap,
                    cbar=cbar, xticklabels=categories, yticklabels=categories)

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        else:
            pass

        if title:
            plt.title(title)

        plt.show()

        # True/False pred

    def plot_tru_mod(self, true_value, pred, estimator):
        """plot y=x and compare the modelisation values with the true values

        Args:
            true_value (_type_): ytest
            pred (_type_): y from xtest
            estimator (_type_): estimator name
        """
        fig = plt.figure()
        plt.plot(true_value, pred, 'o', alpha=0.5)
        min_tru = true_value.min()
        max_tru = true_value.max()
        plt.plot([min_tru, max_tru], [min_tru, max_tru], '--')
        plt.title('Comparaison des valeurs modélisées avec les valeurs réelles')
        plt.xlabel("mod")
        plt.ylabel("True")
        plt.show()

    # feat importance
    def plot_feature_importance(self, importance, names, estimator):
        """plot the features importance

        Args:
            importance (_type_): features importance
            names (_type_): feature names
            estimator (_type_): estimator name
        """
        feature_importance = np.array(importance)
        feature_names = np.array(names)
        data = {'feature_names': feature_names,
                'feature_importance': feature_importance}
        fi_df = pd.DataFrame(data)
        fi_df.sort_values(by=['feature_importance'],
                          ascending=False, inplace=True)

        df_10 = fi_df.head(10)  # TODO add input nb of feat to plot
        fig = plt.figure(figsize=(10, 8))
        sns.barplot(x=df_10['feature_importance'], y=df_10['feature_names'])
        plt.title(f"{estimator} FEATURES IMPORTANCE")
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
        plt.show()

     # AutoML for classification with cross validation

    def ml_crossvalidation(self,
                           supervised_type,
                           features,
                           target,
                           estimator,
                           space,
                           score,
                           ratio_split=0.3,
                           cv=5,
                           **kwargs
                           ):
        """Gather the chain of training operation in a CV 

        Args:
            supervised_type (_type_): type of ML
            features (_type_): X
            target (_type_): Y
            estimator (_type_): estimator object
            space (_type_): HP space
            score (_type_): type of evaluation for training
            ratio_split (float, optional): split ration. Defaults to 0.3.
            cv (int, optional): number of CV. Defaults to 5.

        Returns:
            trained model, best_param, result_scor
        """
       # Split
        try:

            X_train, X_test, y_train, y_test = self.split_data(feat=features,
                                                               target=target,
                                                               ratio_split=ratio_split)

            print("split done")
        except Exception as e:
            print("split error", e)

        # Sampling

        if kwargs.get("sampling_strategy"):
            try:
                X_train, y_train = self.sampling_transform(
                    X_train, y_train, kwargs.get("sampling_strategy"))
            except Exception as e:
                print("Sampling error", e)

        # Train

        model_CV = GridSearchCV(estimator=estimator,
                                param_grid=space,
                                scoring=score,
                                cv=cv,
                                error_score='raise',
                                verbose=2)
        print(model_CV)
        best_param = {}
        try:
            model_CV.fit(X_train, y_train)
            print('train done')

            best_param = model_CV.best_params_
            print(f"best_parameters: {best_param}")
            print(f"Best_score : {model_CV.best_score_}")

        except Exception as e:
            print("error during train CV", e)

        # Validation

        result_scor = {}
        try:
            predic_label = model_CV.predict(X_test)
            # predic_label_proba = model_CV.predict_proba(X_test)
            print("Predictions : ", predic_label)

            result_scor = self.eval_metrics_cla(supervised_type=supervised_type,
                                                true_value=y_test,
                                                pred=predic_label,
                                                estimator=f"Model_{str(estimator).split('(')[0]}"
                                                )
            print(result_scor)

        except Exception as e:
            print('error during validation', e)
        try:
            ft_importance = model_CV.best_estimator_.feature_importances_
            self.plot_feature_importance(importance=ft_importance,
                                         names=self.features_name,
                                         estimator=f"Model_{str(estimator).split('(')[0]}")
        except Exception as e:
            print("error during feat importance plot", e)

        return model_CV.best_estimator_, best_param, result_scor

    def start_ml(self,
                 supervised_type: str,
                 feature_names: list,
                 features: list,
                 target: str,
                 models: list,  # dict,
                 ratio_split: float,
                 nb_cv: int,
                 score: Optional[str] = None,
                 **kwargs):
        """ML tracking with mlflow for training

        Args:
            supervised_type (SupervisedEnum): supervised type
            target (str): Y name
            models (list): estimator names
            ratio_split (float): split ratio
            nb_cv (int): number of CV
            score (Optional[str], optional): type of evaluation. set to accuracy for Classification and to neg_mean_squared_error for regrssion Defaults to None.

        Returns:
            trained models into a dict
        """

        self.features_name = feature_names

        if not score:

            if supervised_type == "cla":
                score = "accuracy"

            if supervised_type == "reg":
                score = "neg_mean_squared_error"

       # start ML

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if supervised_type == "cla":
                dic_models = dic_classifier

            if supervised_type == "reg":

                dic_models = dic_regressor

            dic_estimator = {}
            for mod in models:
                estimator = dic_models[mod]["estimator"]
                print("Cv for model:", estimator)

                space = dic_models[mod]["space"]
                print(space)

                try:
                    model_best, model_param, model_score = self.ml_crossvalidation(features=features,
                                                                                   supervised_type=supervised_type,
                                                                                   target=target,
                                                                                   estimator=estimator,
                                                                                   space=space,
                                                                                   score=score,
                                                                                   ratio_split=ratio_split,
                                                                                   cv=nb_cv,
                                                                                   **kwargs
                                                                                   )

                except Exception as e:
                    print("cv error", e)

                dic_estimator[estimator] = model_best, model_param, model_score

                print('-' * 45)
                print("Done")
                print('-' * 45)

            print('-' * 45)
            print("Resume: ", dic_estimator)
            print('-' * 45)

        return dic_estimator

    def show_ml_result(self, dic_estimator: dict):

        dic_result = DefaultDict()
        dic_result["model"] = []
        dic_result["parametres"] = []

        for model, it in dic_estimator.items():
            for score_name, score_value in it[2].items():
                dic_result[f"{score_name}"] = []

        for model, it in dic_estimator.items():

            dic_result['model'].append(str(model).split('(')[0])

            dic_result['parametres'].append(it[1])

            for score_name, score_value in it[2].items():
                dic_result[f"{score_name}"].append(score_value)

        df_result = pd.DataFrame(dic_result)

        return df_result

    def make_prediction(self, X, col_name, Y, estimator, supervised_type):
        """load new data and make new prediction

        Args:
            estimator (_type_): selected model

        Returns:
            dataframe with the prediction
        """
        print(estimator)

        try:
            pred = estimator.predict(X)
            print(pred)

        except Exception as e:
            print("pred error", e)
        if supervised_type == "cla":
            cm = confusion_matrix(Y, pred)
            self.showconfusionmatrix(cm, estimator='')

        if supervised_type == "reg":
            self.plot_tru_mod(true_value=Y, pred=pred,
                              estimator=str(estimator).split('(')[0])

        print("pred succeed")
        df = pd.DataFrame(X, columns=col_name)
        df["True_value"] = Y
        df["Prediction"] = pred

        return df
