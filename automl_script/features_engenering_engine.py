from sklearn.preprocessing import LabelEncoder,\
    OneHotEncoder, \
    OrdinalEncoder, \
    StandardScaler, \
    RobustScaler, \
    MinMaxScaler, \
    MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


class DropFeatures(BaseEstimator, TransformerMixin):
    """ Features engeneering class for scikit learn pipeline
    
    Drop a list of columns from a pandas dataframe
    
    """

    def __init__(self, feat_to_drop = None, columns=None):
        
        self.columns = columns
        self.feat_to_drop = feat_to_drop
        

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        try:
            X.drop(self.feat_to_drop, axis=1, inplace=True)
            print(X.columns)
            
            print("features droped")
            self.feat_names = X.columns.tolist()
            # settings.features_name = self.feat_names
            return X

        except Exception as e:
            print("error during dropping features", e)

    
    def get_feature_names_out(self):
        return self.feat_names


class DropFeaturesCorr(BaseEstimator, TransformerMixin):
    """ Features engeneering class for scikit learn pipeline
    
    Drop a correlated columns from a pandas dataframe with a given threshold (feat_corr)
    
    """
    def __init__(self, feat_corr, columns=None):
        
        self.columns = columns
        self.feat_corr = feat_corr
        

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        try:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.feat_corr)]
            X = X.drop(to_drop, axis=1)
            print("features corr droped")
            self.feat_names = X.columns.tolist()
            # settings.features_name = self.feat_names
            return X

        except Exception as e:
            print("corr", e)
    
    def get_feature_names_out(self):
        return self.feat_names



strat_encoder = {'le': LabelEncoder,
                 'ohe': OneHotEncoder,
                 'ord': OrdinalEncoder,
                 #  'tgt': TargetEncoder
                 }


class FeaturesEncoder(BaseEstimator, TransformerMixin):
    """ Features engeneering class for scikit learn pipeline
    
    Encode by selected strategy a pandas Dataframe with auto select categorical columns
    
    """
    def __init__(self, strategy='ohe', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.cat_col = []
        self.encoder = strat_encoder[strategy]()

    def get_categorial_col(self, X, y=None):

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'str':
                self.cat_col.append(col)

        print(self.cat_col)

    def fit(self, X, y=None):
        self.get_categorial_col(X=X)

        return self

    def transform(self, X):
        
        if self.cat_col is not None:

            if self.strategy == 'ohe':

                try:
                    self.encoder.fit(X[self.cat_col])
                except Exception as e:
                    print("fit error encode", e)
                try:
                    encoded = self.encoder.transform(
                        X[self.cat_col]).toarray()

                    encoded_output = pd.DataFrame(
                        encoded, columns=self.encoder.get_feature_names_out())
                    X = pd.concat([X, encoded_output], axis=1).drop(
                        [*self.cat_col], axis=1)

                    print("encoded")

                except Exception as e:
                    print("transform", e)

            elif self.strategy == 'ord':

                try:
                    self.encoder.fit(X[self.cat_col])
                except Exception as e:
                    print("fit", e)
                try:
                    X[self.cat_col] = self.encoder.fit_transform(
                        X[self.cat_col])

                    print("encoded")

                except Exception as e:
                    print("transform", e)

            else:
                for col in self.cat_col:
                    X[col] = self.encoder.fit_transform(X[col])
                print("encoded")

        else:
            print("no column to encode")

        print(X)
        self.feat_names = X.columns.tolist()
        # settings.features_name = self.feat_names
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self):
        return self.feat_names


strat_imputer = {'mean': SimpleImputer(strategy="mean"),
                 'median': SimpleImputer(strategy="median"),
                 'most_frequent': SimpleImputer(strategy="most_frequent"),
                 'constant': SimpleImputer(strategy="constant"),
                 'knn': KNNImputer(n_neighbors=5)
                 }


class NanImputation(BaseEstimator, TransformerMixin):
    """ Features engeneering class for scikit learn pipeline
    
    Nan impute values by selected strategy 
    
    """
    def __init__(self, strategy='mean', columns=None):
        self.strategy = strategy
        self.columns = columns
        # If strategy == "constant" ==> fill_value == 0
        self.imputer = strat_imputer[strategy]

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.imputer.transform(X)
        print("imputed")
        return X
    
    # def get_feature_names_out(self):
    #     return settings.features_name


strat_scaler = {
    'standard': StandardScaler,
    'minmax': MinMaxScaler,
    'robust': RobustScaler,
    'maxabs': MaxAbsScaler}


class FeaturesScaler(BaseEstimator, TransformerMixin):
    """ Features engeneering class for scikit learn pipeline
    
    Scale values by selected strategy 
    
    """

    def __init__(self, strategy='standard', columns=None):
        self.strategy = strategy
        self.columns = columns
        self.scaler = strat_scaler[strategy]()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        X = self.scaler.transform(X)
        print("scaled")
        return X

    # def get_feature_names_out(self):
    #     return settings.features_name


class AcpReduction(BaseEstimator, TransformerMixin):
    """ Features engeneering class for scikit learn pipeline
    
    Perform ACP analysis and reduction 
    
    """

    def __init__(self, analyse=False, columns=None):
        self.analyse: bool = analyse
        self.columns = columns
        self.nb_comp = 10

    # fonction affichant le cercle de corrélation

    def display_circles(self, pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):

        for num, (d1, d2) in enumerate(axis_ranks):
            if d2 < n_comp:

                # initialisation de la figure
                fig, ax = plt.subplots(figsize=(7, 6))

                # détermination des limites du graphique

                if lims is not None:
                    xmin, xmax, ymin, ymax = lims
                elif pcs.shape[1] < 30:
                    xmin, xmax, ymin, ymax = -1, 1, -1, 1
                else:
                    xmin, xmax, ymin, ymax = min(pcs[d1, :]), max(pcs[d1, :]), min(pcs[d2, :]), max(pcs[d2,:])

                # affichage des flèches

                if pcs.shape[1] < 30:
                    plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                    pcs[d1, :], pcs[d2, :],
                        angles='xy', scale_units='xy', scale=1, color="grey")

                else:
                    lines = [[[0, 0], [x, y]] for x,y in pcs[[d1,d2]].T]
                    ax.add_collection(LineCollection(
                        lines, axes=ax, alpha=.1, color='black'))

                # affichage des noms des variables

                if labels is not None:
                    for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
                            plt.text(x, y, labels[i], fontsize='14', ha='center',
                                        va='center', rotation=label_rotation, color="blue", alpha=0.5)

                # affichage du cercle
                circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)

                # limites du graphique
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)

                # affichage des lignes horizontales et verticales
                plt.plot([-1, 1], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-1, 1], color='grey', ls='--')

                # nom des axes, avec le pourcentage d'inertie
                plt.xlabel('F{} ({}%)'.format(
                    d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
                plt.ylabel('F{} ({}%)'.format(
                    d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

                plt.title(
                    "Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
                # plt.show(block=False)
                # if settings.mlflow_instance.mlflow_start:
                #     settings.mlflow_instance.mlflow_start.log_figure(fig,f"corr_cicles_{num}.png")
                plt.show()

    # fonction Eboulis des valeurs
    def display_scree_plot(self, pca):
        fig = plt.figure()
        scree = pca.explained_variance_ratio_*100
        plt.bar(np.arange(len(scree))+1, scree)
        plt.plot(np.arange(len(scree))+1, scree.cumsum(), c="red", marker='o')
        plt.xlabel("rang de l'axe d'inertie")
        plt.ylabel("pourcentage d'inertie")
        plt.title("Eboulis des valeurs propres")
        # plt.show(block=False)
        # if settings.mlflow_instance.mlflow_start:
        #             settings.mlflow_instance.mlflow_start.log_figure(fig,"scree_values.png")
        plt.show()

    # fonction plan factoriel
    def display_factorial_planes(self, X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
        for num, (d1, d2)  in enumerate(axis_ranks):
            if d2 < n_comp:

                # initialisation de la figure
                fig = plt.figure(figsize=(7, 6))

                # affichage des points
                if illustrative_var is None:
                    plt.scatter(X_projected[:, d1],
                                X_projected[:, d2], alpha=alpha)
                else:
                    illustrative_var = np.array(illustrative_var)
                    for value in np.unique(illustrative_var):
                        selected = np.where(illustrative_var == value)
                        plt.scatter(
                            X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                    plt.legend()

                # détermination des limites du graphique
                boundary = np.max(np.abs(X_projected[:, [d1, d2]])) * 1.1
                plt.xlim([-boundary, boundary])
                plt.ylim([-boundary, boundary])

                # affichage des lignes horizontales et verticales
                plt.plot([-100, 100], [0, 0], color='grey', ls='--')
                plt.plot([0, 0], [-100, 100], color='grey', ls='--')

                # nom des axes, avec le pourcentage d'inertie expliqué
                plt.xlabel('F{} ({}%)'.format(
                    d1+1, round(100*pca.explained_variance_ratio_[d1], 1)))
                plt.ylabel('F{} ({}%)'.format(
                    d2+1, round(100*pca.explained_variance_ratio_[d2], 1)))

                plt.title(
                    "Projection des individus (sur F{} et F{})".format(d1+1, d2+1))

                # if settings.mlflow_instance.mlflow_start:
                #     settings.mlflow_instance.mlflow_start.log_figure(fig,f"row_projection_{num}.png")
                plt.show()

    def fit(self, X, y=None):
        try:
            # préparation des données pour l'ACP
            n_comp_init = round(0.85*X.shape[1])

            # Calcul du nombre de composantes
            pca = PCA(n_components=n_comp_init)
            reduc = pca.fit(X)
            variance = reduc.explained_variance_ratio_

            def n_comp_var(max_var=0.95):
                for n_comp in range(1,n_comp_init):
                    a = variance.cumsum()[n_comp]
                    
                    if a >= max_var:
                        print(a)
                        print(
                            f"{n_comp} composantes principales expliquent au moins 95% de la variance totale")
                        break

                return n_comp

            self.nb_comp = n_comp_var()
            print("nb composantes principale", self.nb_comp)
        except Exception as e:
            print("acp fit error", e)

        return self

    def transform(self, X, y=None):
        
        try:
            # ACP
            pca = PCA(n_components=self.nb_comp)
            X_project = pca.fit_transform(X)

            if self.analyse:  # TODO CHANGE IN HTML INPUT
                print("ACP analyse started")
                # Eboulis des valeurs propres
                self.display_scree_plot(pca)

                n_comp = 4

                if n_comp >= self.nb_comp:
                    n_comp = self.nb_comp

                axis_ranks = []
                for i in range(0, n_comp, 2):
                    axis_ranks.append((i, i+1))
                

                # Cercle des corrélations
                pcs = pca.components_

                if pcs.shape[1] >= 10:
                    pcs = pcs[:, :10]

                try:
                    self.display_circles(pcs, n_comp, pca, axis_ranks, labels=np.array(
                        self.columns)[:pcs.shape[1]])

                except Exception as e:
                    print("acp circle error", e)
                # Coefficient des composantes principales

                for i in range(n_comp):
                    print("F"+str(i+1))
                    print(pcs[i])

                # Projection des individus

                self.display_factorial_planes(X_project, n_comp, pca, axis_ranks, labels=[j for j in range(np.array(X.shape[0]))])  # les plans sont à modifier en fct du nb de composantes principales

                plt.show()
                
            print("ACP done")

            self.X_columns = ["F"+str(i+1) for i in range(self.nb_comp)]
            # settings.features_name = self.X_columns
            return X_project

        except Exception as e:
            print("acp transform error", e)

    def get_feature_names_out(self):
        
        return self.X_columns
