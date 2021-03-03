import pandas as pd
import numpy as np

from typing import Union, Dict, List, Hashable, Optional, Tuple
from copy import deepcopy

# TODO: Накатывать и на test
# TODO: fit & transform


class TypesConverter:  # TODO: Добавить зависимости колонок.
    """
    Класс для автоматического определения типов признаков.
    Базовая имплементация порядка разработки:

    0.
        0.a) Парсим то, что указал юзер
        0.b) Даты пасим через словарь dependencies. С указанием сезонности ("m", "d", "wd", "h", "min")
        (месяц, день, день недели, час, минута)
    1.
        Если стринга, то категория
    2.
        Если отношение shape[1] к количеству уникальных значений >> 5, то категория
    """
    def __init__(self, max_bin_count: Dict[Hashable, Optional[int]], features_type: Union[List, Dict],  # ???
                 train: pd.DataFrame, dependencies: Dict[Hashable, Tuple[str, Tuple[str]]]):
        """
        TODO В будующем надо будет добавить инфу о зависимостях между признаками вместе/вместо с данными параметрами

        Parameters
        ----------
        max_bin_count
        features_type
        train
        dependencies
        """
        self.__max_bin_count = max_bin_count
        self.__features_type = deepcopy(features_type)
        self.__train = train
        self.__dependencies = dependencies

    @property
    def max_bin_count(self):
        """

        Returns
        -------

        """
        return self.__max_bin_count

    @property
    def features_type(self):
        """
        Read only

        Returns
        -------

        """
        return self.__features_type

    @property
    def train(self):
        """
        Read only

        Returns
        -------

        """
        return self.__train

    @property
    def dependencies(self):
        """
        Read only

        Returns
        -------

        """
        return self.__dependencies

    def __add_feature(self, feature_name, new_feature: List[Hashable]):
        """
        Метод на случай генерирования нового признака на основе старого.
        Старый в этом случае удаляем.


        Parameters
        ----------
        feature_name
        new_feature

        Returns
        -------

        """
        # self.__train[new_feature_name] = new_feature
        # self.__max_bin_count[new_feature_name] = self.__max_bin_count[feature_name]
        # self.__features_type[new_feature_name] = "real"
        pass

    def __del_feature(self, feature_name):
        """

        Parameters
        ----------
        feature_name

        Returns
        -------

        """
        del self.__train[feature_name]
        self.__max_bin_count.pop(feature_name)
        self.__features_type.pop(feature_name)


    def feature_handler(self, feature_name):
       """

       Parameters
       ----------
       feature_name

       Returns
       -------

       """
        # TODO: 1. Пытаемя парсить даты
        # TODO: 2. Проверяем на категориальнось


    def fit_transform(self):
        """
        Returns
        -------

        """
        pass



    def transform(self):
        """


        Returns
        -------

        """
        self.dates_parser()

        for feature in self.__features_type:
            if feature:  # если указали что какого-то определенного типа
                self.feature_handler(feature)
            else:  # если тип не указан
                pass


def dates_handler(feature: pd.Series, dependence: Optional[Tuple] = (None, ("d", "wd", "h", "min"))):
    """
    Даты пасим через словарь dependence.
    С указанием сезонности ("%Y%d%m", ("m", "d", "wd", "h", "min"))

    Parameters
    ----------
    feature:
        Колонка для парсинга
    dependence:
        Формат признака для парсинга
    Returns
    -------

    """
    new_feature = []

    feature = pd.to_datetime(feature, format=dependence[0])
    if feature.max().year == 1970:
        return feature, False

    for seasonality in dependence[1]:
        if seasonality == "m":
            new_feature = feature.map(lambda x: x.month)
        elif seasonality == "d":
            new_feature = feature.map(lambda x: x.day)
        elif seasonality == "wd":
            new_feature = feature.map(lambda x: x.weekday())
        elif seasonality == "h":
            new_feature = feature.map(lambda x: x.hour)
        elif seasonality == "min":
            new_feature = feature.map(lambda x: x.minute)
        else:
            ValueError(f"Seasonality {seasonality} is not supported")

        new_feature_name = str(feature.name) + seasonality
        new_feature.append((new_feature_name, new_feature))

    return new_feature, True


def cat_handler(feature: pd.Series):  # парсер категории
    """

    Parameters
    ----------
    feature

    Returns
    -------

    """
    # feature_ = self.__train[feature_name]
    pass






