import os
import itertools

import numpy as np
import scipy as sp
import pandas as pd

from functools import wraps
from typing import Dict
from collections import defaultdict
from copy import deepcopy

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from ..autowoe import AutoWoE
from ..utilities.refit import calc_p_val_on_valid
from .report_generator import ReportGenerator
from .utilities_images.utilities_images import plot_roc_curve_image, plot_model_weights,\
    plot_roc_curve_feature_image, plot_feature_split, plot_ginis, plot_woe_bars, plot_double_roc_curve,\
        plot_backlash_check, plot_binned, plot_binned_stats, plot_corr_heatmap, plot_mean_target


class ReportDeco:
    """
    Класс-декоратор для генерации отчета
    """

    def __init__(self, auto_woe: AutoWoE):
        """
        Parameters
        ----------
        auto_woe
        """
        self.__auto_woe = auto_woe
        self.__stat = dict()

        self.__target_name = None
        self.__train_target = None
        self.__test_target = None
        self.__predict_proba = None
        self.__nan_stat = [[], []]
        self.__train_enc = None
        self.__test_enc = None
        self.__predict_proba_train = None

    @property
    @wraps(AutoWoE.p_vals)
    def p_vals(self):
        return self.__auto_woe.p_vals

    @property
    @wraps(AutoWoE.features_type)
    def features_type(self):
        return self.__auto_woe.features_type

    @property
    @wraps(AutoWoE.private_features_type)
    def private_features_type(self):
        return self.__auto_woe.private_features_type

    @property
    def features_fit(self):
        return self.__auto_woe.features_fit

    @wraps(AutoWoE.get_split)
    def get_split(self, *args, **kwargs):
        return self.__auto_woe.get_split(*args, **kwargs)

    @wraps(AutoWoE.get_woe)
    def get_woe(self, *args, **kwargs):
        return self.__auto_woe.get_woe(*args, **kwargs)

    @wraps(AutoWoE.get_sql_inference_query)
    def get_sql_inference_query(self, *args, **kwargs):
        return self.__auto_woe.get_sql_inference_query(*args, **kwargs)

    @wraps(AutoWoE.fit)
    def fit(self, *args, **kwargs):
        # parse stat
        self.__auto_woe.fit(*args, **kwargs)
        ###################################################################
        ###################################################################
        train = kwargs["train"] if "train" in kwargs else args[0]
        self.__target_name = kwargs["target_name"] if "target_name" in kwargs else args[4]
        self.__train_enc = self.__auto_woe.test_encoding(train)
        self.__train_target = train[self.__target_name]

        self.__stat["count_train"] = int(train.shape[0])
        self.__stat["train_target_cnt"] = int(train[self.__target_name].sum())
        self.__stat["train_nontarget_cnt"] = int(self.__stat["count_train"] - self.__stat["train_target_cnt"])

        self.__stat["count_train"] = str(self.__stat["count_train"])
        self.__stat["train_target_cnt"] = str(self.__stat["train_target_cnt"])
        self.__stat["train_nontarget_cnt"] = str(self.__stat["train_nontarget_cnt"])

        self.__stat["train_target_perc"] = __class__.str_format(train[self.__target_name].mean() * 100)

        self.__predict_proba_train = self.predict_proba(train)
        self.__stat["train_auc_full"] = ReportDeco.roc_auc_str(y_true=train[self.__target_name].values,
                                                               y_score=self.__predict_proba_train)
        self.__stat["train_gini_full"] = ReportDeco.roc_gini_str(y_true=train[self.__target_name].values,
                                                                 y_score=self.__predict_proba_train)

        # 95% (by default) confidence interval for gini index
        self.__stat["train_gini_confint"] = ReportDeco.calc_gini_confint(
            y_true=train[self.__target_name].values,
            y_score=self.__predict_proba_train
        )
        
        features_fit = self.features_fit.sort_values()
        self.__stat["model_coef"] = list(zip(features_fit.index, features_fit))
        self.__stat["model_coef"] = [(str(pair[0]), 0+round(pair[1], 6)) for pair in self.__stat["model_coef"]]
        
        # P-values are present only if regularized_refit == False
        if self.p_vals is not None:
            self.__stat["p_vals"] = [(k, 0+round(v, 6)) for k, v in sorted(zip(self.p_vals.index, self.p_vals), key=lambda item: item[1], reverse=True)]
        else:
            self.__stat["p_vals"] = None

        # VIF calculation (need more than 1 feature)
        self.__stat["train_vif"] = ReportDeco.calc_vif(self.__train_enc)

        self.__stat["scorecard"] = self.__get_scorecard()

        # Copying feature history from the model and enriching it with Gini and IV
        feature_history = deepcopy(self.__auto_woe.feature_history)
        for feature in self.__train_enc.columns:
            feature_gini = round((roc_auc_score(self.__train_target, -self.__train_enc[feature].values) - 0.5) * 2, 2)
            feature_iv = round(self.__auto_woe.woe_dict[feature].iv, 2)
            feature_history[feature] = f'Selected; Gini = {feature_gini}, IV = {feature_iv}'
        
        self.__stat["feature_history"] = sorted(feature_history.items())
        ###################################################################
        self.__nan_stat = [[], []]
        for feature in features_fit.index:
            feature_ = feature.split("__F__")[0]
            not_nan_count = train[feature_].count()
            nan_count = train.shape[0] - not_nan_count
            not_nan_count_per = 100*(nan_count / train.shape[0])
            self.__nan_stat[0].append((feature_, not_nan_count, nan_count, not_nan_count_per))

    @wraps(AutoWoE.predict_proba)
    def predict_proba(self, *args, **kwargs):
        # parse stat
        predict_proba = self.__auto_woe.predict_proba(*args, **kwargs)
        ###################################################################
        ###################################################################
        test = kwargs["test"] if "test" in kwargs else args[0]
        self.__test_enc = self.__auto_woe.test_encoding(test)

        self.__stat["count_test"] = int(test.shape[0])
        self.__stat["test_target_cnt"] = int(test[self.__target_name].sum())
        self.__stat["test_nontarget_cnt"] = int(self.__stat["count_test"] - self.__stat["test_target_cnt"])

        self.__stat["count_test"] = str(self.__stat["count_test"])
        self.__stat["test_target_cnt"] = str(self.__stat["test_target_cnt"])
        self.__stat["test_nontarget_cnt"] = str(self.__stat["test_nontarget_cnt"])

        self.__stat["test_target_perc"] = __class__.str_format(test[self.__target_name].mean() * 100)

        self.__stat["test_auc_full"] = ReportDeco.roc_auc_str(y_true=test[self.__target_name].values,
                                                              y_score=predict_proba)
        self.__stat["test_gini_full"] = ReportDeco.roc_gini_str(y_true=test[self.__target_name].values,
                                                                y_score=predict_proba)

        # 95% (by default) confidence interval for gini index
        self.__stat["test_gini_confint"] = ReportDeco.calc_gini_confint(
            y_true=test[self.__target_name].values,
            y_score=predict_proba
        )

        self.__test_target = test[self.__target_name]
        self.__predict_proba = predict_proba

        # Calculate p-values on test only if regularized_refit == False
        if self.p_vals is not None:
            p_vals_test, _ = calc_p_val_on_valid(self.__test_enc, self.__test_target)
            self.__stat["p_vals_test"] = [(k, 0+round(v, 6)) for k, v in sorted(zip(list(self.__test_enc.columns) + ['Intercept_'], p_vals_test), key=lambda item: item[1], reverse=True)]
        else:
            self.__stat["p_vals_test"] = None
        
        # Don't calculate feature contribution if regularized_refit=True
        if self.__auto_woe._AutoWoE__regularized_refit:
            self.__stat["feature_contribution"] = None
        else:
            self.__stat["feature_contribution"] = self.__refit_leave_one_out()
        ###################################################################
        features_fit = self.features_fit.sort_values()
        self.__nan_stat[1] = []
        for feature in features_fit.index:
            feature_ = feature.split("__F__")[0]
            not_nan_count = test[feature_].count()
            nan_count = test.shape[0] - not_nan_count
            not_nan_count_per = 100*(nan_count / test.shape[0])
            self.__nan_stat[1].append((feature_, not_nan_count, nan_count, not_nan_count_per))

        return predict_proba

    def generate_report(self, report_params: Dict):
        """
        Метод для генерации отчета
        Для корректного отображения необходимо последоватьельно запустить метод fit и predict_proba
        Parameters
        ----------
        report_params

        Returns
        -------

        """
        if not os.path.exists(report_params['output_path']):
            os.mkdir(report_params['output_path'])
        
        rg = ReportGenerator()
        
        plot_double_roc_curve(
            self.__train_target,
            self.__predict_proba_train,
            self.__test_target,
            self.__predict_proba,
            os.path.join(report_params['output_path'], 'AUC_test_plot_full.png')
        )

        plot_model_weights(self.features_fit.sort_values(), os.path.join(report_params['output_path'],
                                                                         'Model_weights.png'))

        final_nan_stat = []
        train_nan, test_nan = self.__nan_stat
        for i in range(len(train_nan)):
            final_nan_stat.append((str(train_nan[i][0]),
                                   str(train_nan[i][1]), str(test_nan[i][1]),
                                   str(train_nan[i][2]), str(test_nan[i][2]),
                                   __class__.str_format(train_nan[i][3]), __class__.str_format(test_nan[i][3]),
                                   __class__.str_format(train_nan[i][3] - test_nan[i][3]),
                                   ))

        self.__stat["final_nan_stat"] = final_nan_stat

        self.__stat["features_roc_auc"] = []
        for feature in self.__auto_woe.features_fit.index:
            name = feature + '_roc_auc.png'
            self.__stat["features_roc_auc"].append(name)

            plot_roc_curve_feature_image(feature, self.__test_target,
                                         -self.__test_enc[feature].values,
                                         os.path.join(report_params['output_path'], name))

        self.__stat["features_woe"] = []
        for feature in self.__auto_woe.features_fit.index:
            name = feature + '_woe.png'
            self.__stat["features_woe"].append(name)

            plot_feature_split(feature, self.get_woe(feature), os.path.join(report_params['output_path'], name))

        # Gini indices on train dataset features
        if self.__train_target is not None:
            plot_ginis(
                self.__train_enc,
                self.__train_target,
                os.path.join(report_params['output_path'], 'train_enc_ginis.png')
            )
        
        # Gini indices on test dataset features
        if self.__test_target is not None:
            plot_ginis(
                self.__test_enc,
                self.__test_target,
                os.path.join(report_params['output_path'], 'test_enc_ginis.png')
            )
        
        # Monotonicity check
        self.__stat["woe_bars"] = []
        if self.__train_target is not None and self.__test_target is not None:
            for feature in self.__auto_woe.features_fit.index:
                name = feature + '_woe_bars.png'
                self.__stat["woe_bars"].append(name)
                plot_woe_bars(
                    self.__train_enc,
                    self.__train_target,
                    self.__test_enc,
                    self.__test_target,
                    self.__target_name,
                    feature,
                    os.path.join(report_params['output_path'], name)
                )

        # Backlash check using only train data
        self.__stat["backlash_plots"] = []
        if self.__train_target is not None:
            for feature in self.__auto_woe.features_fit.index:
                name = feature + '_backlash_plot.png'
                self.__stat["backlash_plots"].append(name)
                plot_backlash_check(
                    self.__predict_proba_train,
                    self.__train_enc,
                    self.__train_target,
                    feature,
                    os.path.join(report_params['output_path'], name)
                )

        # PSI calculation
        if self.__train_target is not None and self.__test_target is not None:
            self.__stat["psi_total"] = ReportDeco.calc_psi(self.__train_enc, self.__test_enc)
            self.__stat["psi_zeros"] = ReportDeco.calc_psi(
                self.__train_enc[self.__train_target == 0],
                self.__test_enc[self.__test_target == 0]
            )
            self.__stat["psi_ones"] = ReportDeco.calc_psi(
                self.__train_enc[self.__train_target == 1],
                self.__test_enc[self.__test_target == 1]
            )
            
            # Split score into 10 bins for train and test
            train_binned, test_binned = self.__get_binned_data(10)
            names = ['binned_train_total.png', 'binned_train_posneg.png']
            plot_binned(
                train_binned,
                os.path.join(report_params['output_path'], names[0]),
                os.path.join(report_params['output_path'], names[1])
            )
            names = ['binned_test_total.png', 'binned_test_posneg.png']
            plot_binned(
                test_binned,
                os.path.join(report_params['output_path'], names[0]),
                os.path.join(report_params['output_path'], names[1])
            )
            
            # Selecting [0][1] because there is only 1 feature in the output
            self.__stat["psi_binned_total"] = ReportDeco.calc_psi(
                train_binned[['ScoreBin']],
                test_binned[['ScoreBin']]
            )[0][1]
            self.__stat["psi_binned_zeros"] = ReportDeco.calc_psi(
                train_binned[['ScoreBin']][(self.__train_target == 0).values],
                test_binned[['ScoreBin']][(self.__test_target == 0).values]
            )[0][1]
            self.__stat["psi_binned_ones"] = ReportDeco.calc_psi(
                train_binned[['ScoreBin']][(self.__train_target == 1).values],
                test_binned[['ScoreBin']][(self.__test_target == 1).values]
            )[0][1]
        else:
            self.__stat["psi_total"] = None
            self.__stat["psi_zeros"] = None
            self.__stat["psi_ones"] = None
            self.__stat["psi_binned_total"] = None
            self.__stat["psi_binned_zeros"] = None
            self.__stat["psi_binned_ones"] = None

        # Bin prediction stats
        self.__stat["binned_p_stats_train"] = None
        self.__stat["binned_p_stats_test"] = None
        if self.__train_target is not None and self.__test_target is not None:
            train_binned, test_binned = self.__get_binned_data(20)
            plot_mean_target(train_binned, test_binned, os.path.join(report_params['output_path'], 'binned_stats_target.png'))
            plot_binned_stats(train_binned, os.path.join(report_params['output_path'], 'binned_stats_train.png'))
            plot_binned_stats(test_binned, os.path.join(report_params['output_path'], 'binned_stats_test.png'))
            self.__stat["binned_p_stats_train"] = ReportDeco.get_binned_p_stats(train_binned)
            self.__stat["binned_p_stats_test"] = ReportDeco.get_binned_p_stats(test_binned)

        # Correlation heatmap
        if self.__train_enc is not None:
            corr_map = self.__train_enc.corr()
            plot_corr_heatmap(corr_map, os.path.join(report_params['output_path'], 'corr_heatmap.png'))
            self.__stat["corr_map_table"] = [
                (x1, x2, 0+round(corr_map[x1][x2], 6))
                for x1, x2 
                in itertools.combinations(self.__train_enc.columns, 2)
            ]
        else:
            self.__stat["corr_map_table"] = None

        report_params_final = dict(**report_params, **self.__stat)
        rg.generate_report(report_params_final)

    def __get_binned_data(self, bin_count):
        train_binned = pd.DataFrame({'P': self.__predict_proba_train, 'Target': self.__train_target.values})
        test_binned = pd.DataFrame({'P': self.__predict_proba, 'Target': self.__test_target.values})

        bins = None

        for df in [train_binned, test_binned]:
            df['Score'] = np.log(df['P'] / (1-df['P']))
            if bins is not None:
                df['ScoreBin'] = pd.cut(df['Score'], bins, retbins=False)
            else:
                df['ScoreBin'], bins = pd.cut(df['Score'], bin_count, retbins=True)

        return train_binned, test_binned

    def __get_scorecard(self):
        # Round value to 2 decimals and transform -0.0 into 0.0
        round_ext = lambda x: 0 + round(x, 2)
        
        # Format of the result is "Variable - Value - WOE - COEF - POINTS"
        result = []
        intercept = round_ext(self.__auto_woe.intercept)
        result.append(('Intercept', None, None, intercept, intercept))
        # Iterate over features that take part in the regression
        for feature, coef in self.__auto_woe.features_fit.items():
            woe = self.__auto_woe.woe_dict[feature]
            # "split" field for continuous and categorical
            # variables is different
            if woe._f_type == 'cat':
                cat_split = defaultdict(list)
                special_values = {key for key in woe.cod_dict if type(key) is str}
                for k, v in woe.split.items():
                    if k not in special_values:
                        cat_split[v].append(k)

            for key, w in woe.cod_dict.items():
                # Case for '__NaN__'s and similar (works for both 'cat' and 'real')
                if type(key) is str:
                    label = str(key)
                # Cat values with the same WOE
                elif woe._f_type == 'cat':
                    label = ', '.join(str(x) for x in cat_split[key])
                # Below are split cases only for continuous variables
                elif key == 0:
                    # It is possible that split list is empty - means that
                    # any value of the feature maps to the same WOE value
                    if len(woe.split) == 0:
                        label = f'-inf < {feature} < +inf'
                    else:
                        label = f'{feature} <= {round_ext(woe.split[int(key)])}'
                elif key == len(woe.split):
                    label = f'{feature} > {round_ext(woe.split[int(key - 1)])}'
                else:
                    label = f'{round_ext(woe.split[int(key - 1)])} < {feature} <= {round_ext(woe.split[int(key)])}'
                
                row = (feature, label, round_ext(w), round_ext(coef), round_ext(w * coef))
                result.append(row)
            
        return result

    def __refit_leave_one_out(self):
        if len(self.features_fit) < 2:
            return []
        
        result = dict()
        initial_score = roc_auc_score(y_true=self.__test_target.values, y_score=self.__predict_proba)
        for feature in self.features_fit.index:
            feature_subset = [x for x in self.features_fit.index if x != feature]
            X, y = self.__train_enc[feature_subset].values, self.__train_target.values
            clf = LogisticRegression(penalty='none', solver='lbfgs', warm_start=False, intercept_scaling=1)
            clf.fit(X, y)
            test_subset = self.__test_enc[feature_subset].values
            prob = 1 / (1 + np.exp(-(np.dot(test_subset, clf.coef_[0]) + clf.intercept_[0])))
            score = roc_auc_score(y_true=self.__test_target.values, y_score=prob)
            result[feature] = round(initial_score - score, 4)

        return sorted(result.items())

    @staticmethod
    def get_binned_p_stats(binned_df):
        return binned_df \
            .groupby('ScoreBin')['P'] \
            .agg('describe') \
            .round(4) \
            .sort_values(by='ScoreBin') \
            .reset_index() \
            .astype(str) \
            .values \
            .tolist()

    @staticmethod
    def roc_auc_str(y_true, y_score):
        """

        Parameters
        ----------
        y_true
        y_score

        Returns
        -------

        """
        auc = 100 * roc_auc_score(y_true=y_true, y_score=y_score)
        return __class__.str_format(auc)

    @staticmethod
    def roc_gini_str(y_true, y_score):
        """

        Parameters
        ----------
        y_true
        y_score

        Returns
        -------

        """
        gini = 100 * (2 * roc_auc_score(y_true=y_true, y_score=y_score) - 1)
        return __class__.str_format(gini)

    @staticmethod
    def str_format(x):
        return '{:.2f}'.format(x)

    @staticmethod
    def calc_gini_confint(y_true, y_score, n=1000, p=0.05):
        idx = np.arange(y_true.shape[0])
        bounds = p / 2, 1 - p / 2
        scores = []
        for _ in range(n):
            idx_ = np.random.choice(idx, size=idx.shape[0], replace=True)
            scores.append((roc_auc_score(y_true[idx_], y_score[idx_]) - .5) * 2)
            
        return np.round(np.quantile(scores, bounds), 3)

    @staticmethod
    def calc_vif(data_enc):
        cc = sp.corrcoef(data_enc.values, rowvar=False)
        if cc.ndim < 2:
            return []
        VIF = np.round(np.linalg.inv(cc).diagonal(), 6)
        return sorted(zip(data_enc.columns, VIF), key=lambda item: item[1], reverse=True)

    @staticmethod
    def calc_psi(train_data, test_data):
        tr_len, val_len = train_data.shape[0], test_data.shape[0]
        
        name = [0] * tr_len + [1] * val_len
        data = pd.concat([train_data, test_data], ignore_index=True)
        data['_sample_'] = name
        
        PSIs = {}
        
        for woe in data.columns.drop('_sample_'):
            agg = data.groupby(woe)['_sample_'].agg(['sum', 'count']) # + .5
            # % from val
            prc_val = (agg['sum'] / val_len) 
            # % from train
            prc_tr = ((agg['count'] - agg['sum']) / tr_len) 
            # psi
            # global psi
            psi = (prc_tr - prc_val) * np.log((prc_tr + 1e-3) / (prc_val + 1e-3))
            
            PSIs[woe] = np.round(psi.sum(), 6)
            
        return sorted(PSIs.items(), key=lambda item: item[1], reverse=True)
