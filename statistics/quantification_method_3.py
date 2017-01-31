# 数量化Ⅲ類
import numpy as np
import pandas as pd


class QM3:

    def __init__(self, D):
        self.df = D  # Pandas.DataFrame
        self.r2, self.x, self.y = self._qm3(np.array(D))  # 固有値、列配置ベクトル、行配置ベクトル

    def _create_x(self, v):
        return np.sqrt(self.g.sum()) * (v @ self.sqrg)

    def _create_y(self, rho2, v):
        return (self.fid @ v[:, np.newaxis] / np.sqrt(rho2)).T[0]

    def _qm3(self, D):
        self.fid = D / D.sum(axis=1)[:, np.newaxis]
        self.g = D.sum(axis=0)
        self.sqrg = np.diag(np.sqrt(1 / self.g))
        rho2, vec = np.linalg.eig(self.sqrg @ D.T @ self.fid @ self.sqrg)
        rho2 = rho2[1:]  # 形式解は除外
        vec = vec.T[1:]  # 形式解は除外
        x = np.array([self._create_x(v) for v in vec])
        y = np.array([self._create_y(r2, v) for r2, v in zip(rho2, x)])
        return rho2, x.argsort(), y.argsort()

    def sort(self, i=0):
        # i番目の固有値で並べ替え
        return self.df.ix[self.y[i], self.x[i]]
