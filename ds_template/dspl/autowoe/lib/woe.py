from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd


class WoE:
    """
    Class for WoE transformation
    """

    def __init__(self, f_type: str, split: List[float]):
        """
        Parameters
        ----------
        f_type
             "cat" - категориальный, "real" - вещественный
        split
            (в случае вещественного признака). Формат [-27, 1, 4, 5, 12, 100]
            (в случае категориального) {12: 1, 17: 1, 20: 2, 35: 3}
        """
        self._f_type = f_type
        self.split = split

        self.iv = None
        self.cod_dict = None

    def __codding(self, x: pd.Series):
        """
        Кодируем обычные значения

        Parameters
        ----------
        x

        Returns
        -------

        """
        if self._f_type == "cat":
            x_cod = x.map(self.split)
        elif self._f_type == "real":
            x_cod = np.searchsorted(self.split, x.values, side="left")  # check
            x_cod = pd.Series(data=x_cod, index=x.index)
        else:
            raise ValueError("_f_type is cat or real")
        return x_cod

    @staticmethod
    def _bucket_woe(x, total_good: int, total_bad: int):
        """
        Parameters
        ----------
        x
        total_good
        total_bad

        Returns
        -------

        """
        t_bad = x['bad']
        t_good = x['count_nonzero']
        t_bad = 0.5 if t_bad == 0 else t_bad
        t_good = 0.5 if t_good == 0 else t_good
        return np.log((t_bad / total_bad) / (t_good / total_good))

    def __woe(self, df: pd.DataFrame):
        """
        Получение WoE для каждой категории

        Parameters
        ----------
        df

        Returns
        -------

        """
        df.columns = [0, "target"]
        stat = df.groupby(0)["target"].agg([np.mean, np.count_nonzero, np.size])

        stat['bad'] = stat['size'] - stat['count_nonzero']
        t_good = np.maximum(stat['count_nonzero'].sum(), 0.5)  # Если меток вообще нет
        t_bad = np.maximum(stat['bad'].sum(), 0.5)  # Если меток вообще нет

        stat['woe'] = stat.apply(lambda x: self._bucket_woe(x, t_good, t_bad),
                                 axis=1)  # ||P.Correction|-> + np.log(t_good / t_bad)||
        iv_stat = (stat['bad'] / t_bad - stat['count_nonzero'] / t_good) * stat['woe']  # Кульбака-Лейблера
        self.iv = iv_stat.sum()
        stat = stat["woe"].to_dict()
        return stat

    def __df_cod_transform(self, x: pd.Series, spec_values):
        """

        Parameters
        ----------
        x
        spec_values
            Если значаение не None, то кодируем WoE по дефолту, если же нет, то кодируем 0

        Returns
        -------

        """
        x_ = deepcopy(x)
        if isinstance(spec_values, list):
            spec_values_ = spec_values.copy()
        elif isinstance(spec_values, dict):
            spec_values_ = spec_values.keys()
        else:
            spec_values_ = []

        x_.loc[x_.isin(spec_values_)] = -np.inf
        df_cod = self.__codding(x_)
        df_cod.loc[x.isin(spec_values_)] = x.loc[x.isin(spec_values_)]
        return df_cod

    def fit(self, x, y, spec_values):
        """
        Parameters
        ----------
        x
        y
        spec_values

        Returns
        -------

        """
        df_cod = self.__df_cod_transform(x, spec_values)
        df_cod = pd.concat([df_cod, y], axis=1)
        stat = self.__woe(df_cod)

        if isinstance(spec_values, dict):  # Случай особых значений
            for key in spec_values:  # Если задано woe в spec_values то изменяем значение WoE
                if spec_values[key] is not None:
                    stat[key] = spec_values[key]
                elif key not in stat.keys():
                    stat[key] = 0

        self.cod_dict = stat
        return df_cod

    def fit_transform(self, x: pd.Series, y: pd.Series, spec_values):
        """

        Parameters
        ----------
        x
        y
        spec_values
           Если значение не None, то кодируем WoE по дефолту, если же нет, то кодируем 0

        Returns
        -------

        """
        df_cod = self.fit(x, y, spec_values)
        df_cod = df_cod[0].map(self.cod_dict).copy()
        return df_cod

    def transform(self, x: pd.Series):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        df_cod = self.__df_cod_transform(x, self.cod_dict)
        df_cod = df_cod.map(self.cod_dict)
        return df_cod

    def fit_transform_cv(self, x: pd.Series, y: pd.Series, spec_values, cv_index_split: Dict[int, List[int]]):
        """
        WoE кодирование по cv

        Parameters
        ----------
        x
        y
        spec_values:
            Если значаение не None, то кодируем WoE по дефолту, если же нет, то кодируем 0
        cv_index_split

        Returns
        -------

        """
        x_ = deepcopy(x)
        for key in cv_index_split:
            train_index, test_index = cv_index_split[key]
            self.fit(x.iloc[train_index], y.iloc[train_index], spec_values)
            x_.iloc[test_index] = self.transform(x.iloc[test_index])
        return x_.astype(float)

