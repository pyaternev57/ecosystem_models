
# import os
import sys
import warnings
from copy import deepcopy
# from collections import OrderedDict
from getpass import getpass
sys.path.append("../")

# import IPython
import numpy as np
# from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from ds_template.dspl.models import classifiers
from ds_template.dspl.feature_extraction import transformers
from ds_template.dspl.feature_extraction.utils import compare_gini_features
from ds_template.dspl.feature_extraction.utils import select_subset_features, drop_features
from ds_template.dspl.validation_report.ValidationReporter import ValidationReporter
from ds_template.dspl.reports.report import ClassificationDevelopmentReport
from ds_template.dspl.utils import choose_psi_sample, get_input, INFOSaver
from ds_template.dspl.optimizer import BayesianOptimizationModel
from ds_template.dspl.utils import get_notebook_path_and_save
from ds_template.dspl.validation import DataSplitter
from ld_utils.utils import dict_from_file
warnings.simplefilter("ignore")

conf = None

# TODO create arguments of script
partner = 'delivery_club'
target = 'target'


with open("../ds_template/conf/never_used_features.txt", "r") as file:
    never_used_features = file.readlines()

logins = dict_from_file("../conf/logins.json")

config = {
    # логин и пароль от TeraData TDSB15
    "login": logins["td_username"],
    "password": logins["td_password"],

    "dev_data_path": f"{partner}_train.csv",
    "oot_data_path": f"{partner}_train.csv",
    "data_path": "../data",
    "runs_path": "../runs",

    # название столбца с целевой меткой
    "target_name": target,
     # название столбцов, которые не участвуют в обучении
    "drop_features": ['report_dt', 'card_number', 'card_dwh_id', 'reissued_card_dwh_id', 'agrmnt_dwh_id', 'card_w4_id', 'agrmnt_w4_id',
                      'epk_id', 'mnth3_nfc_rate', 'mnth3_trans_qty', 'mnth3_nfc_trans_qty', 'actual_client_dk', 'seg_main_card_dwh_id',
                      'actual_agrmnt_dwh_id', 'actual_card_dwh_id'],

    # никогда не используемые признаки в обучении
    "never_used_features": never_used_features,

    # категориальные признаки, добавление опционально
    "categorical_features": None,

    # параметры разбиения данных на train / valid / test
    "split_params": [0.6, 0.2, 0.2],
    "split_column": None,
    "sort": None,

    "absolute_diff": 5,
    "relative_diff": 10,
    "valid_sample": "valid",

    # порог для отбора признаков по PSI
    # и выборка для расчета PSI
    "psi_threshold": 0.01,
    "psi_sample": "OOT",

    # порог для отбора признаков по метрике Джини
    "gini_threshold": 5,

    # порог для отбора признаков на основе перестановок (см. документацию подробнее)
    "permutation_threshold": 0.05,

    # количество итераций оптимизатора гиперпараметров моделей
    "n_iterations": 25,

    # Флаг использования WhiteBox AutoML
    "use_automl": True
}

saver = INFOSaver(config=config, path=config["runs_path"])

train, target = get_input("dev_data_path", config, conf=conf)

np.random.seed(27)

splitter = DataSplitter(
    config["split_params"], config["split_column"]
)
eval_sets = splitter.transform(train, target)

if config.get("oot_data_path"):
    test, test_target = get_input(
        "oot_data_path", config=config
    )
    eval_sets["OOT"] = (test, test_target)
    eval_sets, psi_sample = choose_psi_sample(eval_sets, config)

saver.save_data(data=eval_sets)

eval_sets = drop_features(config, **eval_sets)
eval_sets, transformer = transformers.encode_categorical(config, **eval_sets)

models = {}
models["encoder"] = transformer
config["categorical_features"] = transformer.cat_features

feature_importance, used_features = compare_gini_features(eval_sets,config)
models["gini_importance"] = feature_importance

used_features = sorted(used_features)
print(f"GINI Количество признаков: {len(used_features)}")

if "OOT" in eval_sets:
    train, target = eval_sets["train"]
    psi = transformers.PSI(config["psi_threshold"], transformer.cat_features)
    psi_score = psi.fit_transform(train[used_features], psi_sample[used_features])
    psi_features = sorted(psi.used_features)
    models["psi_importance"] = psi_score

    print(f"PSI Количество признаков: {len(psi_features)}")

rf_params = {
    "n_estimators": 200,
    "max_depth": 5,
    "n_jobs": 16,
    "random_state": 27
}

delta_scores = {
    "absolute_diff": config.get("absolute_diff", 5),
    "relative_diff": config.get("relative_diff", 10)
}

optional_params = {
    "delta_scores": delta_scores,
    "categorical_features": transformer.cat_features
}

rf_model = classifiers.RFClassifierModel(
    rf_params, used_features, **optional_params
)
rf_model.fit(*eval_sets["train"], *eval_sets["valid"])

rf_model.evaluate_model(**eval_sets)
models[f"RF.{len(used_features)}"] = rf_model

if config.get("oot_data_path"):
    rf_psi_model = classifiers.RFClassifierModel(
        rf_params, psi_features, **optional_params
    )
    rf_psi_model.fit(*eval_sets["train"], *eval_sets["valid"])

    rf_psi_model.evaluate_model(**eval_sets)
    models[f"RF.{len(psi_features)}.PSI"] = rf_psi_model

np.random.seed(27)

if config.get("oot_data_path"):
    perm_features = rf_psi_model.feature_importance(*eval_sets["valid"])
else:
    perm_features = rf_model.feature_importance(*eval_sets["valid"])

perm_threshold = config.get("permutation_threshold", 0.05)
perm_features = select_subset_features(perm_features, perm_threshold)
print(f"Количество признаков: {len(perm_features)}")

rf_perm_model = classifiers.RFClassifierModel(
    rf_params, perm_features, **optional_params
)
rf_perm_model.fit(*eval_sets["train"], *eval_sets["valid"])

rf_perm_model.evaluate_model(**eval_sets)
models[f"RF.{len(perm_features)}.Perm"] = rf_perm_model

rf_bounds = {
    "max_depth": (3, 8),
    "min_samples_split": (0.01, 1),
    "min_samples_leaf": (0.01, 0.5),
    "max_features": (0.2, 0.8)
}

rf_opt_model = deepcopy(rf_perm_model)
n_iter = config.get("n_iterations", 25)

optimizer = BayesianOptimizationModel(
    rf_opt_model, roc_auc_score, eval_sets["valid"], rf_bounds, n_iter=n_iter
)
optimizer.fit(*eval_sets["train"])

rf_opt_model.evaluate_model(**eval_sets)
models[f"RF.{len(perm_features)}.Optimized"] = rf_opt_model

xgb_params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "learning_rate": 0.1,
    "n_estimators": 1000,
    "reg_lambda": 10,
    "max_depth": 4,
    "gamma": 10,
    "nthread": 6,
    "seed": 27
}

xgb_model = classifiers.XGBClassifierModel(
    xgb_params, used_features, **optional_params
)
xgb_model.fit(*eval_sets["train"], *eval_sets["valid"])

xgb_model.evaluate_model(**eval_sets)
models[f"XGBoost.{len(used_features)}"] = xgb_model

if config.get("oot_data_path"):
    xgb_psi_model = classifiers.XGBClassifierModel(
        xgb_params, psi_features, **optional_params
    )
    xgb_psi_model.fit(*eval_sets["train"], *eval_sets["valid"])

    xgb_psi_model.evaluate_model(**eval_sets)
    models[f"XGBoost.{len(psi_features)}.PSI"] = xgb_psi_model

np.random.seed(27)

if config.get("oot_data_path"):
    perm_features = xgb_psi_model.feature_importance(*eval_sets["valid"])
else:
    perm_features = xgb_model.feature_importance(*eval_sets["valid"])

perm_threshold = config.get("permutation_threshold", 0.05)
perm_features = select_subset_features(perm_features, perm_threshold)
print(f"Количество признаков: {len(perm_features)}")

xgb_perm_model = classifiers.XGBClassifierModel(
    xgb_params, perm_features, **optional_params
)
xgb_perm_model.fit(*eval_sets["train"], *eval_sets["valid"])

xgb_perm_model.evaluate_model(**eval_sets)
models[f"XGBoost.{len(perm_features)}.Perm"] = xgb_perm_model

xgb_bounds = {
    "max_depth": (3, 6),
    "min_child_weights": (100, 2500),
    "reg_lambda": (5, 100),
    "gamma": (1, 100)
}

xgb_opt_model = deepcopy(xgb_perm_model)
n_iter = config.get("n_iterations", 25)

optimizer = BayesianOptimizationModel(
    xgb_opt_model, roc_auc_score, eval_sets["valid"], xgb_bounds, n_iter=n_iter
)
optimizer.fit(*eval_sets["train"])

xgb_opt_model.evaluate_model(**eval_sets)
models[f"XGBoost.{len(perm_features)}.Optimized"] = xgb_opt_model

wb_params = {
    "interpreted_model": True,
    "monotonic": False,
    "max_bin_count": 5,
    "select_type": None,
    "pearson_th": 0.9,
    "auc_th": .525,
    "vif_th": 10.,
    "imp_th": 0,
    "th_const": 32,
    "force_single_split": True,
    "th_nan": 0.01,
    "th_cat": 0.005,
    "woe_diff_th": 0.01,
    "min_bin_size": 0.01,
    "min_bin_mults": (2, 4),
    "min_gains_to_split": (0.0, 0.5, 1.0),
    "auc_tol": 1e-4,
    "cat_alpha": 100,
    "cat_merge_to": "to_woe_0",
    "nan_merge_to": "to_woe_0",
    "oof_woe": True,
    "n_folds": 6,
    "n_jobs": 4,
    "l1_base_step": 20,
    "l1_exp_step": 6,
    "population_size": None,
    "feature_groups_count": 7,
    "imp_type": "perm_imp",
    "regularized_refit": False,
    "p_val": 0.05,
    "debug": False
}

wb_model = classifiers.WBAutoML(
    params=wb_params,
    used_features=used_features,
    categorical_features=config["categorical_features"],
)

if config["use_automl"]:
    wb_model.fit(*eval_sets["train"])

    wb_model.evaluate_model(**eval_sets)
    models["WhiteBox AutoML"] = wb_model

ipynb_name = get_notebook_path_and_save()
saver.save_artifacts(models=models, ipynb_name=ipynb_name)

report = ClassificationDevelopmentReport(models, saver, config)
report.transform(**eval_sets)
