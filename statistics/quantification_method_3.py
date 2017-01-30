class QM3:
    def __init__(self, D):
        self.D = D # 行列
        self.r2, self.x, self.y = self._qm3(np.array(D)) # 固有値、列配置ベクトル、行配置ベクトル
        
    def _qm3(self, D):
        fid = D/D.sum(axis=1)[:, np.newaxis]
        sqrg = np.sqrt(1/data.sum(axis=0))
        r2, vec = np.linalg.eig(np.multiply(np.multiply(sqrg.reshape((sqrg.size, 1)), data.T @ fid), sqrg))
        r2 = r2[1:] # 形式解は除外
        vec = vec.T[1:] # 形式解は除外
        x = np.array(list(map(lambda v: np.sqrt(g.sum()) * np.multiply(v, sqrg), vec)))
        y = np.array(list(map(lambda rho2, v: (fid @ v[:, np.newaxis] / np.sqrt(rho2)).T[0], r2, x)))
        return r2, x.argsort(), y.argsort()

    def sort(self, i=0):
        # i番目の固有値で並べ替え
        return self.D.ix[self.y[i], self.x[i]]
