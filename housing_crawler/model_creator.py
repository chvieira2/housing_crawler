import numpy as np
import pandas as pd
# settings to display all columns
pd.set_option("display.max_columns", None)

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from sklearn import set_config; set_config(display='diagram')
from sklearn.impute import SimpleImputer
# SimpleImputer does not have get_feature_names_out, so we need to add it manually.
SimpleImputer.get_feature_names_out = (lambda self, names = None: self.feature_names_in_)
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sklearn.metrics as metrics

from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import pickle

from housing_crawler.ads_table_processing import get_processed_ads_table
from housing_crawler.utils import report_best_scores, create_dir
from config.config import ROOT_DIR
from housing_crawler.params import dict_city_number_wggesucht
from housing_crawler.string_utils import standardize_characters


class ModelGenerator():
    """Implementation of a class that creates the best model for predicting prices"""

    def __init__(self, market_type_filter = 'WG', # WG, Single-room flat, Apartment
                 target='price_per_sqm_cold', # price_per_sqm_cold, price_euros
                 target_log_transform = False,
                 city='allcities'):

        print('\n\n\n')
        print(f'Starting search for {market_type_filter} in {city}')
        print('\n\n\n')

        # Create directory for saving
        if city == 'allcities':
            self.model_path = f'{ROOT_DIR}/housing_crawler/models'
            self.model_name = f'Pred_pipeline_{market_type_filter}'
        else:
            self.model_path = f'{ROOT_DIR}/housing_crawler/models/{standardize_characters(city)}'
            self.model_name = f'Pred_pipeline_{market_type_filter}_allcities'

        create_dir(path = self.model_path)

        self.target = target
        self.columns_to_remove = []

        ## Main dataframe
        self.df_filtered = get_processed_ads_table()

        # Filter only ads that have been searched for details (search added from august on)
        self.df_filtered = self.df_filtered[self.df_filtered['details_searched']==1]
        # Filter ad maket type
        self.df_filtered = self.df_filtered[self.df_filtered['type_offer_simple']==market_type_filter]
        # Filter ads with address
        self.df_filtered = self.df_filtered[self.df_filtered['km_to_centroid'].notna()]
        self.df_filtered = self.df_filtered.drop(columns=['details_searched','type_offer_simple'])
        self.df_filtered = self.df_filtered.set_index('id')

        if city != 'allcities':
            self.df_filtered = self.df_filtered[self.df_filtered['city']==city]
            self.df_filtered = self.df_filtered.drop(columns=['city'])

        if target_log_transform:
            self.df_filtered[self.target] = np.log2(self.df_filtered[self.target])
            self.model_name = self.model_name + '_targetLogTrans'


        ## Creates the list of features according to the processing pipeline they will be processed by
        features_OSM = [
                'comfort_leisure_spots',
                'comfort_warehouse',
                'activities_education',
                'mobility_public_transport_bus',
                'activities_economic',
                'comfort_industrial', # Has not been blurried?
#                 'activities_goverment', # I can't even plot it. There's something weird
                'social_life_eating',
                'comfort_comfort_spots',
                'social_life_culture',
#                 'activities_supermarket', # I can't even plot it. There's something weird
#                 'activities_public_service', # Data is constant
                'social_life_community',
                'comfort_leisure_mass',
                'activities_educational',
                'mobility_street_secondary', # Has not been blurried?
                'mobility_public_transport_rail',
                'activities_retail', # Has not been blurried?
                'social_life_night_life',
#                 'comfort_green_natural', # I can't even plot it. There's something weird
                'comfort_railway', # Has not been blurried?
                'mobility_bike_infraestructure', # Has not been blurried
#                 'comfort_green_forests', # Data is constant
                'mobility_street_primary', # Has not been blurried?
                'comfort_lakes', # Has not been blurried?
#                 'activities_health_regional', # Data is constant
                'activities_health_local',
                'comfort_green_space', # Has not been blurried
#                 'comfort_rivers', # I can't even plot it. There's something weird
                'activities_post',
                'comfort_green_parks', # Has not been blurried?
#                 'comfort_street_motorway' # Has not been blurried. Empty?
]

        # features_already_OHE
        if market_type_filter == 'WG':
            self.features_already_OHE = ['tv_kabel','tv_satellit',

                        'shower_type_badewanne','shower_type_dusche',

                       'floor_type_dielen','floor_type_parkett','floor_type_laminat','floor_type_teppich',
                       'floor_type_fliesen','floor_type_pvc','floor_type_fußbodenheizung',

                       'extras_waschmaschine','extras_spuelmaschine','extras_terrasse','extras_balkon',
                       'extras_garten','extras_gartenmitbenutzung','extras_keller','extras_aufzug',
                       'extras_haustiere','extras_fahrradkeller','extras_dachboden',

                       'languages_deutsch','languages_englisch',

                       'wg_type_studenten','wg_type_keine_zweck','wg_type_maenner','wg_type_business',
                       'wg_type_wohnheim','wg_type_vegetarisch_vegan','wg_type_alleinerziehende','wg_type_funktionale',
                       'wg_type_berufstaetigen','wg_type_gemischte','wg_type_mit_kindern','wg_type_verbindung',
                       'wg_type_lgbtqia','wg_type_senioren','wg_type_inklusive','wg_type_wg_neugruendung',

                       'internet_dsl','internet_wlan','internet_flatrate']
        elif market_type_filter == 'Apartment':
            self.features_already_OHE = ['tv_kabel','tv_satellit',

                        'shower_type_badewanne','shower_type_dusche',

                       'floor_type_dielen','floor_type_parkett','floor_type_laminat','floor_type_teppich',
                       'floor_type_fliesen','floor_type_pvc','floor_type_fußbodenheizung',

                       'extras_waschmaschine','extras_spuelmaschine','extras_terrasse','extras_balkon',
                       'extras_garten','extras_gartenmitbenutzung','extras_keller','extras_aufzug',
                       'extras_haustiere','extras_fahrradkeller','extras_dachboden',

                       'wg_possible',

                       'internet_dsl','internet_wlan','internet_flatrate']
        else:
            self.features_already_OHE = ['tv_kabel','tv_satellit',

                        'shower_type_badewanne','shower_type_dusche',

                       'floor_type_dielen','floor_type_parkett','floor_type_laminat','floor_type_teppich',
                       'floor_type_fliesen','floor_type_pvc','floor_type_fußbodenheizung',

                       'extras_waschmaschine','extras_spuelmaschine','extras_terrasse','extras_balkon',
                       'extras_garten','extras_gartenmitbenutzung','extras_keller','extras_aufzug',
                       'extras_haustiere','extras_fahrradkeller','extras_dachboden']


        # cols_PowerTrans_SimpImpMean
        if market_type_filter == 'WG':
            self.cols_PowerTrans_SimpImpMean = ['km_to_centroid',#'size_sqm',
                                        'min_age_flatmates', 'max_age_flatmates', 'home_total_size', 'days_available',
                                        'room_size_house_fraction','energy_usage']
        else:
            self.cols_PowerTrans_SimpImpMean = ['km_to_centroid', 'days_available',
                                        'construction_year','energy_usage']

        # cols_PowerTrans_SimpImpMedian_MinMaxScaler
        if market_type_filter == 'WG':
            self.cols_PowerTrans_SimpImpMedian_MinMaxScaler = features_OSM +\
            ['capacity', 'min_age_searched', 'max_age_searched','public_transport_distance','number_languages', 'energy_efficiency_class']
        else:
            self.cols_PowerTrans_SimpImpMedian_MinMaxScaler = features_OSM +\
            ['public_transport_distance', 'energy_efficiency_class']


        # cols_SimpImpMean_MinMaxScaler
        if market_type_filter == 'WG':
            self.cols_SimpImpMean_MinMaxScaler = ['internet_speed','sin_degrees_to_centroid','cos_degrees_to_centroid']
        else:
            self.cols_SimpImpMean_MinMaxScaler = ['sin_degrees_to_centroid','cos_degrees_to_centroid']


        # cols_SimpImpMedian_MinMaxScaler
        if market_type_filter == 'WG':
            self.cols_SimpImpMedian_MinMaxScaler = ['commercial_landlord',
                                        'male_flatmates', 'female_flatmates', 'diverse_flatmates','flat_with_kids',
                                        'schufa_needed','smoking_numerical', 'building_floor',
                                        'furniture_numerical'] + self.features_already_OHE
        elif market_type_filter == 'Apartment':
            self.cols_SimpImpMedian_MinMaxScaler = ['commercial_landlord',
                                        'schufa_needed',
                                        'building_floor', 'toilet',
                                        'furniture_numerical', 'kitchen_numerical', 'available_rooms'] + self.features_already_OHE
        else:
            self.cols_SimpImpMedian_MinMaxScaler = ['commercial_landlord',
                                        'schufa_needed',
                                        'building_floor', 'toilet',
                                        'furniture_numerical', 'kitchen_numerical'] + self.features_already_OHE


        # cols_PowerTrans_SimpImpMedian_MinMaxScaler
        self.cols_SimpImpConst0_PowerTrans_MinMaxScaler = ['transfer_costs_euros']


        # cols_PowerTrans_SimpImpMedian_MinMaxScaler
        if market_type_filter == 'WG':
            self.cols_SimpImpConstNoAns_OHE = ['rental_length_term','gender_searched','building_type','heating', 'parking']
        else:
            self.cols_SimpImpConstNoAns_OHE = ['rental_length_term','building_type','heating', 'parking']

        if city == 'allcities':
            self.cols_SimpImpConstNoAns_OHE += ['city']



        # Final preprocessing pipeline
        self.preprocessor_analysed = None

        # Best model
        self.model = None

    def define_preprocessing_backbone(self):
        """
        Creates the preprocessing backbone. Should match the list created in self.define_columns_preprocessing()
        """

        ## Build the imputter/scaler pairs
        PowerTrans_SimpImpMean = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="mean")),
            ('PowerTransformer', PowerTransformer())
        ])

        PowerTrans_SimpImpMedian_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="median")),
            ('PowerTransformer', PowerTransformer()),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpMean_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="mean")),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpMedian_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="median")),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpConst0_PowerTrans_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="constant", fill_value=0)),
            ('PowerTransformer', PowerTransformer()),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpConstNoAns_OHE = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="constant", fill_value='no_answer')),
            ('OHE', OneHotEncoder(sparse=False, drop='if_binary', categories='auto'))
        ])

        ## Build column transformer pipeline
        self.preprocessor_transformer = ColumnTransformer([
            ('pipeline-1', PowerTrans_SimpImpMean, self.cols_PowerTrans_SimpImpMean),
            ('pipeline-2', PowerTrans_SimpImpMedian_MinMaxScaler, self.cols_PowerTrans_SimpImpMedian_MinMaxScaler),
            ('pipeline-3', SimpImpMean_MinMaxScaler, self.cols_SimpImpMean_MinMaxScaler),
            ('pipeline-4', SimpImpMedian_MinMaxScaler, self.cols_SimpImpMedian_MinMaxScaler),
            ('pipeline-5', SimpImpConst0_PowerTrans_MinMaxScaler, self.cols_SimpImpConst0_PowerTrans_MinMaxScaler),
            ('pipeline-6', SimpImpConstNoAns_OHE, self.cols_SimpImpConstNoAns_OHE)
            ],
            remainder='drop',
            verbose_feature_names_out=False)

        print('Preprocessing_backbone has been created.')
        print('\n')
        return self.preprocessor_transformer

    def minimize_features(self, df_minimal, cols_exclude_minimization = []):
        """
        Exclude all columns (except cities) with >99% of the same value as it contains very little information. This was originally designed to remove OHE-generated columns with very little information, but I later decided to not exclude OHE-generated columns. See https://inmachineswetrust.com/posts/drop-first-columns/
        Columns for exclusion are added to self.columns_to_remove and the function returns the minimized dataframe
        """
        print('=========================')
        print('Minimizing features...')
        print('\n')
        # Define columns to be tested. Don't test the target, commercial_landlord and 'city'
        cols_to_search = [col for col in df_minimal.columns if col not in [self.target]]

        for col in cols_to_search:
            # How many times the most frequent val exists
            most_freq_count = list(df_minimal[col].value_counts())[0]

            if most_freq_count > len(df_minimal)*0.99:
                cols_exclude_minimization.append(col)

        ## Finalize
        print(f'Columns excluded during minimization:')
        print(', '.join(cols_exclude_minimization))
        print('\n')


        self.columns_to_remove = self.columns_to_remove + cols_exclude_minimization
        return df_minimal.drop(columns=cols_exclude_minimization)

    def VIF_colinearity_analysis(self, df_VIF, cols_to_exclude_VIF = [], VIF_threshold=10):
        """
        This function implements a automatized Variance Inflation Factor (VIF) analysis and identifies columns with high colinearity.
        Columns for exclusion are added to self.columns_to_remove and the function returns the reduced dataframe without features with VIF score above VIF_threshold.
        """
        print('=========================')
        print('Colinearity (VIF) analysis...')
        print('\n')
        remove = True
        while remove:

            df = pd.DataFrame()

            # Ignore the targer column
            selected_columns = [self.target]
            selected_columns = [col for col in df_VIF.columns.to_list() if col not in selected_columns]

            df["features"] = selected_columns

            df["vif_index"] = [vif(df_VIF[selected_columns].values, i) for i in range(df_VIF[selected_columns].shape[1])]

            df = round(df.sort_values(by="vif_index", ascending = False),2)

            # Look only at the highest feature VIF value first beacause removing columns must be done one at a time. Each feature removed influences each others VIF results
            df = df.head(1)

            # Remove features with VIF higher than VIF-threshold
            if float(df.vif_index) >= VIF_threshold:
                print(df)
                cols_to_exclude_VIF = cols_to_exclude_VIF + df.features.to_list()
                df_VIF = df_VIF.drop(columns = df.features)
            else:
                remove = False


        ## Prints the Variation Inflation Factor (VIF) analysis for the top 10 most impactful features that were not excluded
        df = pd.DataFrame()
        selected_columns = [col for col in df_VIF.columns.to_list() if col not in [self.target]]

        df["features"] = selected_columns

        df["vif_index"] = [vif(df_VIF[selected_columns].values, i) for i in range(df_VIF[selected_columns].shape[1])]

        print(round(df.sort_values(by="vif_index", ascending = False),2)[:10])


        ## Finalize
        print(f'Columns excluded in VIF analysis:')
        print(', '.join(cols_to_exclude_VIF))
        print('\n')

        self.columns_to_remove = self.columns_to_remove + cols_to_exclude_VIF
        return df_VIF

    def feature_importance_permutation(self, df_permuted, importance_threshold=0.001, scoring='r2', estimator=Ridge()):
        """
        This function performs a permutation importance analysis using a Ridge() model to identify columns with highest impact on variance. Columns with impact lower than importance_threshold are removed to decrease dimentionality of the data.
        Columns for exclusion are added to self.columns_to_remove and the function returns the reduced dataframe without features with low permutation importance.
        """
        print('=========================')
        print('Permutation feature importance analysis...')
        print('\n')

        ## Calculate permutation scores
        X = df_permuted
        y = self.df_filtered[self.target]
        model = estimator.fit(df_permuted, y) # Fit model
         # Perform Permutation
        permutation_score = permutation_importance(model, X, y,
                                                scoring = [scoring],
                                                n_repeats=100, n_jobs=-1)
        # Unstack results
        importance_df = pd.DataFrame(np.vstack((X.columns,
                                        permutation_score[scoring].importances_mean,
                                       permutation_score[scoring].importances_std)).T)
        importance_df.columns=['feature',
                       'r2 score decrease','r2 score decrease std']


        importance_df = importance_df.sort_values(by="r2 score decrease", ascending = False) # Order by importance
        print(importance_df[:20])


        ## Calculating the score increase per feature
        top_features = []
        scores = []

         # Loop over the total number of features, from the feature with highest to lowest importance
        for features in range(1, len(importance_df)):
             # List the name of the features in specific loop
            most_important_features = list(importance_df.head(features).feature)
             # Make feature set with the selected features
            X_reduced = X[most_important_features]
             # cross validate model with selected faetures
            cv_results = cross_val_score(model, X_reduced, y, cv=10)
             # Append scores
            scores.append(cv_results.mean())
            # Append number of features
            top_features.append(features)

        # Obtain features with importance below threshold
        columns_excluded_permutation = importance_df[importance_df['r2 score decrease']< importance_threshold]
        columns_excluded_permutation = columns_excluded_permutation.feature.to_list()

        ## Finalize
        print(f'Columns excluded during importance permutation:')
        print(', '.join(columns_excluded_permutation))
        print('\n')

        self.columns_to_remove = self.columns_to_remove + columns_excluded_permutation
        return df_permuted.drop(columns=columns_excluded_permutation)

    def identify_num_cols_to_remove(self):
        """
        Creates a list of column names to be ignored during modelling.
        """
        ## Get transformed table excluding OHE features for further automated analysis
        # Obtain df_processed
        preprocessor_transformer = self.define_preprocessing_backbone()
        df_processed = pd.DataFrame(preprocessor_transformer.fit_transform(self.df_filtered), columns=preprocessor_transformer.get_feature_names_out())


        # List of features to be ignored during analyis. Generally, do not exclude features created by OHE. See https://inmachineswetrust.com/posts/drop-first-columns/
        features_ignore_analysis = self.cols_SimpImpConstNoAns_OHE + ['tv','shower_type','floor_type','extras','languages','wg_type']
        features_ignore_analysis = [col+'_' for col in features_ignore_analysis]
        features_ignore_analysis = [cols for cols in df_processed.columns.to_list() if any(substring in cols for substring in features_ignore_analysis)]\
        + ['commercial_landlord','flat_with_kids','schufa_needed'] + ['internet_dsl','internet_wlan','internet_flatrate']


        # Remove from analysis OHE features
        df_processed = df_processed[[col for col in df_processed.columns if col not in features_ignore_analysis]]
        print('Ignoring OHE columns for analysis...')

        ## Apply minimization
        df_minimal = self.minimize_features(df_processed)
        print('Finished Minimizing features')
        ## Apply colinearity VIF analyis
        df_VIF = self.VIF_colinearity_analysis(df_minimal)
        print('Finished colinearity analysis')
        ## Apply permutation importance analysis
        df_permuted = self.feature_importance_permutation(df_VIF)
        print('Finished permutation importance analysis')

        # Return the analysed dataframe
        return df_permuted

    def define_preprocessing_after_analysis(self):
        """
        Creates the preprocessing backbone after features have been analysed. This function is ideantical to define_preprocessing_backbone() except that it excludes from the model features identified during analysis.
        """

        # Identify the columns that should be removed
        self.identify_num_cols_to_remove()

        print('Creating final preprocessing backbone. Columns to exclude from modeling:')
        print(', '.join(sorted(self.columns_to_remove)))
        print('\n')

        ## Build the imputter/scaler pairs
        PowerTrans_SimpImpMean = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="mean")),
            ('PowerTransformer', PowerTransformer())
        ])

        PowerTrans_SimpImpMedian_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="median")),
            ('PowerTransformer', PowerTransformer()),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpMean_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="mean")),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpMedian_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="median")),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpConst0_PowerTrans_MinMaxScaler = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="constant", fill_value=0)),
            ('PowerTransformer', PowerTransformer()),
            ('MinMaxScaler', MinMaxScaler())
        ])

        SimpImpConstNoAns_OHE = Pipeline([
            ('SimpleImputer', SimpleImputer(strategy="constant", fill_value='no_answer')),
            ('OHE', OneHotEncoder(sparse=False, drop='if_binary', categories='auto'))
        ])

        ## Build column transformer pipeline
        self.preprocessor_analysed = ColumnTransformer([
            ('pipeline-1', PowerTrans_SimpImpMean, [col for col in self.cols_PowerTrans_SimpImpMean if col not in self.columns_to_remove]),
            ('pipeline-2', PowerTrans_SimpImpMedian_MinMaxScaler, [col for col in self.cols_PowerTrans_SimpImpMedian_MinMaxScaler if col not in self.columns_to_remove]),
            ('pipeline-3', SimpImpMean_MinMaxScaler, [col for col in self.cols_SimpImpMean_MinMaxScaler if col not in self.columns_to_remove]),
            ('pipeline-4', SimpImpMedian_MinMaxScaler, [col for col in self.cols_SimpImpMedian_MinMaxScaler if col not in self.columns_to_remove]),
            ('pipeline-5', SimpImpConst0_PowerTrans_MinMaxScaler, [col for col in self.cols_SimpImpConst0_PowerTrans_MinMaxScaler if col not in self.columns_to_remove]),
            ('pipeline-6', SimpImpConstNoAns_OHE, [col for col in self.cols_SimpImpConstNoAns_OHE if col not in self.columns_to_remove])
            ],
            remainder='drop',
            verbose_feature_names_out=False)


        print('preprocessor_analysed pipeline has been created.')
        return self.preprocessor_analysed

    def hyperparametrization(self,X,y, model,
                             search_space, scoring='neg_root_mean_squared_error'):

        # Instanciate GridSearchCV
        rsearch = GridSearchCV(
            model, search_space,
            n_jobs=-1, scoring=scoring, cv=5, verbose=0)

        rsearch.fit(X,y)
        print(type(model).__name__)
        report_best_scores(rsearch.cv_results_, 1)
        return rsearch

    def create_NN_model(nbr_features,
                        n_neurons_layer1 = 32,
                        n_neurons_layer2 = 32,
                        activation_layer1 = 'relu',
                        activation_layer2 = 'relu',
                        learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999,
                        loss = 'mse',
                        metrics = ['mae','msle'],
                        patience = 20,
                        init_mode = 'uniform',
                        dropout_rate = 0.2,
                        weight_constraint = 3.0):


        #### 1. ARCHITECTURE
        model = Sequential()

        model.add(Dense(n_neurons_layer1, input_shape=(nbr_features,), activation=activation_layer1))
        model.add(Dropout(dropout_rate))   # Dropout helps protect the model from memorizing or "overfitting" the training data
        model.add(Dense(n_neurons_layer2, activation=activation_layer2))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear', kernel_initializer=init_mode))

        #### 2. COMPILATION
        adam_opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        model.compile(loss=loss,
                optimizer=adam_opt,
                metrics=metrics)

        #### 3. FIT
        es = EarlyStopping(patience=patience, restore_best_weights=True)

        # Don't set the fit method so the model can be used in GridSearchCV
    #     history = model.fit(X_train, y_train,
    #                         validation_split=0.3,
    #                         shuffle=True,
    #                         batch_size=batch_size, # When batch size is too small --> no generalization
    #                         epochs=epochs,    # When batch size is too large --> slow computations
    #                         callbacks=[es],
    #                         verbose=0)

    #     plot_history(history)

        return model

    def find_best_model(self, models_to_test = [
                                                # 'NeuralNetwork',
                                                'Ridge',
                                                'Lasso',
                                                'ElasticNet',
                                                'SGDRegressor',
                                                'KNeighborsRegressor',
                                                'SVR',
                                                # 'DecisionTreeRegressor',
                                                'RandomForestRegressor',
                                                # 'GradientBoostingRegressor',
                                                'XGBRegressor'
                                                ]):
        """
        This function will find the best hyper parameter for several possible models.
        """

        if self.preprocessor_analysed is None:
            self.define_preprocessing_after_analysis()

        # Get transformed table with corresponding column names for hyperparametrization
        X = pd.DataFrame(self.preprocessor_analysed.fit_transform(self.df_filtered),
                        columns=self.preprocessor_analysed.get_feature_names_out())
        y = self.df_filtered[self.target]


        ## Hyperparametrazing multiple models
        models_parametrized = []
        # NeuralNetwork
        if 'NeuralNetwork' in models_to_test:
            # Instanciate model
            model = KerasRegressor(model=self.create_NN_model(nbr_features= len(X.columns)), verbose=0)

            # fix random seed for reproducibility
            tf.random.set_seed(42)
            # Hyperparameter search space
            search_space = {
                'epochs': [50],#[50, 100, 150],
                'batch_size': [32],#[16,32,64],
                'model__n_neurons_layer1': [32],#[16,32,64],
                'model__n_neurons_layer2': [32],#[16,32,64],
                'model__activation_layer1': ['relu'],#['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
                'model__activation_layer2': ['relu'],#['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
                'model__dropout_rate': [0.2],#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'model__weight_constraint': [3.0],#[1.0, 2.0, 3.0, 4.0, 5.0],
            #     'optimizer__learning_rate': [0.01],
            #     'optimizer__beta_1': [0.9],
            #     'optimizer__beta_2': [0.999],
            #     'loss': ['mse'],
            #     'validation_split': [0.3],
            #     'shuffle':[True],
                'model__init_mode': ['normal'],#['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
            #     'callbacks__patience': [20]
            }

            print('Finding best parameters for NeuralNetwork model...')
            NeuralNetwork_rsearch = GridSearchCV(
                                        model, search_space,
                                        cv=5, verbose=0)
            models_parametrized.append(NeuralNetwork_rsearch)

        # Ridge
        if 'Ridge' in models_to_test:
            print('Finding best parameters for Ridge model...')
            Ridge_rsearch = self.hyperparametrization(X,y, model = Ridge(),
                                search_space = {
                                                'alpha': [0.001,0.01,0.1,1,10,100],
                                                'tol': [0.001,0.01,0.1,1,10,100],
                                                'solver': ['lsqr','auto']# auto, 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
                                            })
            models_parametrized.append(Ridge_rsearch)

        # Lasso
        if 'Lasso' in models_to_test:
            print('Finding best parameters for Lasso model...')
            Lasso_rsearch = self.hyperparametrization(X,y, model = Lasso(),
                                search_space = {
                                                'alpha': [0.001,0.01,0.1,1,10,100],
                                                'tol': [0.001,0.01,0.1,1,10,100],
                                                'selection': ['cyclic', 'random']
                                                })
            models_parametrized.append(Lasso_rsearch)

        # ElasticNet
        if 'ElasticNet' in models_to_test:
            print('Finding best parameters for ElasticNet model...')
            ElasticNet_rsearch = self.hyperparametrization(X,y, model = ElasticNet(),
                                search_space = {
                                                'alpha': [0.001,0.01,0.1,1,10,100],
                                                'tol': [0.001],#[0.001,0.01,0.1,1,10,100],
                                                'l1_ratio': [0,0.3,0.6,1],
                                                'selection': ['cyclic', 'random']
                                                })
            models_parametrized.append(ElasticNet_rsearch)

        # SGDRegressor
        if 'SGDRegressor' in models_to_test:
            print('Finding best parameters for SGDRegressor model...')
            SGDRegressor_rsearch = self.hyperparametrization(X,y, model = SGDRegressor(),
                                search_space = {
    'loss':['squared_error','epsilon_insensitive', 'squared_epsilon_insensitive'],#, 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'alpha': [0.0001, 0.001,0.01],
    'penalty': ['l1','l2','elasticnet'],
    'tol': [1,10,100],
    'l1_ratio': [0,0.3,0.6,1],
    'epsilon': [0.1,1,10],
    'learning_rate': ['invscaling'],#,'constant','optimal','adaptive'],
    'eta0': [0.001,0.01,0.1],
    'power_t': [0.25],
    'early_stopping': [True]
    })
            models_parametrized.append(SGDRegressor_rsearch)

        # KNeighborsRegressor
        if 'KNeighborsRegressor' in models_to_test:
            print('Finding best parameters for KNeighborsRegressor model...')
            KNeighborsRegressor_rsearch = self.hyperparametrization(X,y, model = KNeighborsRegressor(),
                                search_space = {
    'n_neighbors': range(10,50,2),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': range(2,15,1)
})
            models_parametrized.append(KNeighborsRegressor_rsearch)

        # SVR
        if 'SVR' in models_to_test:
            print('Finding best parameters for SVR model...')
            SVR_rsearch = self.hyperparametrization(X,y, model = SVR(),
                                search_space = {
    'kernel': ['poly','sigmoid', 'rbf'],#['linear','poly','sigmoid', 'rbf'],
    'degree': range(2,5,1),
    'C': [10,100,1000],
    'tol': [0.001,0.01,0.1],
    'gamma': ['auto'],#[0,0.1,1,'scale','auto'],
    'coef0': [0,0.1,1],
    'epsilon': [0.1,1,10]
})
            models_parametrized.append(SVR_rsearch)

        # DecisionTreeRegressor
        if 'DecisionTreeRegressor' in models_to_test:
            print('Finding best parameters for DecisionTreeRegressor model...')
            DecisionTreeRegressor_rsearch = self.hyperparametrization(X,y, model = DecisionTreeRegressor(),
                                search_space = {
    'criterion': ['squared_error'],#['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best','random'],
    'max_depth': range(2,5,1),
    'min_samples_split': range(14,17,1),
    'min_samples_leaf': range(2,5,1),
    'min_weight_fraction_leaf': [0.0],#[0.0,0.1,0.2],
    'max_features': [0.8],#[0.6,0.7,0.8,0.9],
    'max_leaf_nodes': [4],#range(3,5,1), #int, default=None
    'min_impurity_decrease': [0.3],#[0.2,0.3,0.4,0.5],
    'ccp_alpha':[0.0],#[0.2,0.25,0.3,0.35],
})
            models_parametrized.append(DecisionTreeRegressor_rsearch)

        # RandomForestRegressor
        if 'RandomForestRegressor' in models_to_test:
            print('Finding best parameters for RandomForestRegressor model...')
            RandomForestRegressor_rsearch = self.hyperparametrization(X,y, model = RandomForestRegressor(),
                                search_space = {
    'n_jobs':[-1],
    'n_estimators': [100],#[100,200,500,1000],
    'criterion': ['squared_error'],#['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth': range(5,20,5),
    'min_samples_split': range(2,22,5),
    'min_samples_leaf': range(2,10,2),
    'min_weight_fraction_leaf': [0.0,0.1,0.2],
    'max_features': [0.7,0.8,0.9,1.0],
    'max_leaf_nodes': range(3,8,2), #int, default=None
    'min_impurity_decrease': [0.3,0.4,0.5],
    'bootstrap': [True, False],
    'ccp_alpha':[0.0],
})
            models_parametrized.append(RandomForestRegressor_rsearch)

        # GradientBoostingRegressor
        if 'GradientBoostingRegressor' in models_to_test:
            print('Finding best parameters for GradientBoostingRegressor model...')
            GradientBoostingRegressor_rsearch = self.hyperparametrization(X,y, model = GradientBoostingRegressor(),
                                search_space = {
    'learning_rate': [0.1],#[0.001,0.01,0.1],
    'n_estimators': [100],#[100,200,500,1000],
    'loss': ['squared_error'],#['squared_error', 'absolute_error', 'huber', 'quantile'],
    'subsample':[0.66],#[0.33,0.66,1.0],
    'criterion': ['friedman_mse'],#['squared_error', 'friedman_mse'],
    'min_samples_split': [7],#range(6,8,1),
    'min_samples_leaf': [3],#range(2,4,1),
    'min_weight_fraction_leaf': [0.0],#[0.0,0.1,0.2],
    'max_depth': range(2,4,1),
    'min_impurity_decrease': [0.4],#[0.3,0.4,0.5],
    'max_features': [0.7,0.8,0.9,1.0],
    'max_leaf_nodes': [4],#range(3,5,1),
    'ccp_alpha':[0.3],#[0.25,0.3,0.35],
})
            models_parametrized.append(GradientBoostingRegressor_rsearch)

        # XGBRegressor
        if 'XGBRegressor' in models_to_test:
            print('Finding best parameters for XGBRegressor model...')
            XGBRegressor_rsearch = self.hyperparametrization(X,y, model = XGBRegressor(),
                                search_space = {
    "colsample_bytree": [0.6,0.7,0.8],
#     "gamma": [0.3,0.4,0.5],
    "learning_rate": [0.1],#[0.1,0.01,0.001], # default 0.1
    "max_depth": range(2,5,1), # default 3
    "n_estimators": range(100,150,10), # default 100
    "subsample": [0.2],#[0.1,0.2,0.3]
})
            models_parametrized.append(XGBRegressor_rsearch)

        ## Compare models side by side
        # train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = 0.1,
                                                            random_state = 0)
        scores = pd.DataFrame(columns=['model','r2','explained_variance','MSE',#'MSLE',
                                    'MAE','RMSE'])

        print('Finding best model...')
        for mod in [LinearRegression(n_jobs=-1)] + models_parametrized:
            mod.fit(X_train,y_train)
            mod_name = type(mod).__name__
            y_pred = mod.predict(X_test)

            explained_variance=metrics.explained_variance_score(y_test, y_pred)
            mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred)
            mse=metrics.mean_squared_error(y_test, y_pred)
        #     mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
            mod_r2=metrics.r2_score(y_test, y_pred)


            scores = pd.concat([scores, pd.DataFrame.from_dict({'model':[mod_name],
                                        'r2':[round(mod_r2,4)],
                                        'explained_variance':[round(explained_variance,4)],
                                        'MAE':[round(mean_absolute_error,4)],
                                        'MSE':[round(mse,4)],
        #                                  'MSLE':[round(mean_squared_log_error,4)],
                                        'RMSE':[round(np.sqrt(mse),4)]
                                        })
                            ]).reset_index(drop=True)

            if self.model is None:
                self.model = mod
            if mod_r2> self.model.score(X_test, y_test):
                self.model = mod

        print(scores.sort_values(by='r2', ascending=False))

    def save_best_model(self):
        """
        This function will create and save the full, fitted pipeline. The full pipeline receives a dataframe directly from get_processed_ads_table() (that is optionally filtered) and returns the best predictive model.
        This best model is also saved locally.
        """
        if self.preprocessor_analysed is None:
            self.define_preprocessing_after_analysis()

        if self.model is None:
            self.find_best_model()

        # Create the prediction pipeline
        self.pred_pipeline = make_pipeline(self.preprocessor_analysed,self.model)

        # Fit model on the full dataset
        self.pred_pipeline = self.pred_pipeline.fit(self.df_filtered.drop(columns=self.target), self.df_filtered[self.target])

        # Export Pipeline as pickle file
        print(f'Saving predictive pipeline {self.model_path}/{self.model_name}')
        with open(f"{self.model_path}/{self.model_name}.pkl", "wb") as file:
            pickle.dump(self.pred_pipeline, file)

        # Load Pipeline from pickle file
        # pred_pipeline = pickle.load(open("pred_pipeline_allcities.pkl","rb"))

        return self.pred_pipeline



if __name__ == "__main__":
    didnt_work = []
    models_to_create = list(dict_city_number_wggesucht.keys()) + ['allcities']
    for city in models_to_create:
        for offer_type in ['WG', 'Single-room flat', 'Apartment']: #'WG', 'Single-room flat', 'Apartment'
            try:
                ModelGenerator(market_type_filter = offer_type,
                    target='price_per_sqm_cold', target_log_transform = False,
                    city=city).save_best_model()
            except:
                didnt_work.append(city+'+'+offer_type)

    print(didnt_work)
