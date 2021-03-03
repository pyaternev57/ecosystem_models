import numpy as np

from typing import TypeVar, List
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..woe import WoE

DataFrame = TypeVar("DataFrame")
Series = TypeVar("Series")


class QTransform:
    """
    Качество хуже, чем у деревьев + С генетикой будет лучше, но работать будет намного дольше.
    В генетике нет доказательства сходимости
    """

    def __init__(self, x: Series, y: Series, cv_splits: int):
        """
        Бининг по квантилям

        Parameters
        ----------
        x
        y
        cv_splits
        """
        self.x = x
        self.y = y
        self.cv_splits = cv_splits
        self.tr_coeff = None

    def __call__(self, q_splits, n_iter: int):
        """

        Parameters
        ----------
        q_splits
        n_iter

        Returns
        -------

        """
        split = self._init_split(q_splits)
        score_ = [np.mean(self.cv_transform(split))]

        for _ in range(n_iter):
            # print(_)
            new_split = self.__bins_unite(split)
            new_score = np.mean(self.cv_transform(new_split))
            if new_score > score_[-1]:
                score_.append(new_score)
                split = new_split.copy()

        # woe = WoE.fit(split, self.x, self.y)
        # x_tr = woe.transform(self.x)

        # TODO: Добавить критерий остановки !!!!!

        return score_, split

    def cv_transform(self, split=None):
        """

        Parameters
        ----------
        split

        Returns
        -------

        """
        score_ = []

        if split is None:
            split = self._init_split(q_splits=20)

        cv = StratifiedKFold(n_splits=self.cv_splits, random_state=323, shuffle=True)
        for train_index, test_index in cv.split(self.x, self.y):
            x_train, y_train = self.x.iloc[train_index], self.y.iloc[train_index]
            x_test, y_test = self.x.iloc[test_index], self.y.iloc[test_index]

            woe = self.__woe_transform(split, x_train, y_train)
            x_train, x_test = woe.transform(x_train), woe.transform(x_test)

            score_.append(self.__fp_clf(x_test["woe"], y_test))

        return score_

    @staticmethod
    def __fp_clf(x_test, y_test):
        """

        Parameters
        ----------
        x_test
        y_test

        Returns
        -------

        """
        return roc_auc_score(y_true=y_test, y_score=-x_test.values)

    def _init_split(self, q_splits=20):
        """
        Дробление по бинам с помощью квантилей

        Parameters
        ----------
        q_splits: int

        Returns
        -------

        """
        range_ = np.arange(100 / q_splits, 99.99, 100 / q_splits)
        range_ = np.append(range_, 100)
        split = np.array([np.percentile(self.x, q) for q in range_])
        # split[0] = -np.inf
        return np.unique(split)

    @staticmethod
    def __bins_unite(split: List):
        """
        В дальнейшем можно переписать так, чтобы не вызывать self.WoE каждый раз

        Parameters
        ----------
        split

        Returns
        -------

        """
        ind = np.random.choice(np.arange(1, len(split) - 1), size=1, replace=False)

        # if 0 in ind or len(split-1) in ind:
        #     raise IndexError(f"can not drop left border {0} or right border {len(split-1)}")

        split = np.delete(split, ind)
        return split

    @staticmethod
    def __woe_transform(split, x, y):
        """

        Parameters
        ----------
        split
        x
        y

        Returns
        -------

        """
        woe = WoE(bins=split)
        woe.fit(x, y)
        return woe
