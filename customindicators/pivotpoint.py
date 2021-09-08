import pandas as pd
import numpy as np
from pandas.core import series
from pandas.core.frame import DataFrame
from scipy.signal import argrelextrema


class PivotPoint_scipy:
    def __pivotpoint(self, v: pd.Series, period: int):
        '''
        ### Calculate the relative extrema of data.
        :param period: number of points to be checked before and after
        '''
        df = DataFrame({'data': v})
        df['pivots'] = pd.Series(np.full(v.size, np.nan))
        minima = df.iloc[argrelextrema(v.values, np.less, order=period)[0]]
        df.loc[minima.index, 'pivots'] = -1
        maxima = df.iloc[argrelextrema(v.values, np.greater, order=period)[0]]
        df.loc[maxima.index, 'pivots'] = 1
        return df['pivots']

    def pivothigh(self, source: pd.Series, period: int):
        pphl = self.pivotpoints(source, period)
        return pd.Series(np.where(pphl == 1, pphl, np.nan))

    def pivotlow(self, source: pd.Series, period: int):
        pphl = self.pivotpoints(source, period)
        return pd.Series(np.where(pphl == -1, pphl, np.nan))

    def pivotpoints(self, source: pd.Series, period: int):
        pphl = self.__pivotpoint(source, period)
        return pd.Series(pphl)
