import pandas as pd
import numpy as np
from pandas.core import series


class PivotPoint:

    def __pivotpoint(self, v: np.ndarray, leftbars: int, rightbars: int, waitForConfirmation: bool):
        middle = leftbars
        s = pd.Series(v - v.item(middle))
        left = s[:middle]
        right = s[middle+1:]
        if right.size == rightbars:
            if right.hasnans:
                right = right.loc[right.notna()]

            if waitForConfirmation:
                if np.all(left < 0) & np.all(right < 0):
                    return 1
                elif np.all(left > 0) & np.all(right > 0):
                    return -1
                else:
                    return np.nan
            else:
                if np.all(left < 0):
                    return 1
                elif np.all(left > 0):
                    return -1
                else:
                    return np.nan
        else:
            return np.nan

    def pivothigh(self, source: pd.Series, leftbars: int, rightbars: int, waitForConfirmation: bool = True):
        pphl = self.pivotpoints(source, leftbars, rightbars, waitForConfirmation)
        return pd.Series(np.where(pphl == 1, pphl, np.nan))

    def pivotlow(self, source: pd.Series, leftbars: int, rightbars: int, waitForConfirmation: bool = True):
        pphl = self.pivotpoints(source, leftbars, rightbars, waitForConfirmation)
        return pd.Series(np.where(pphl == -1, pphl, np.nan))

    def pivotpoints(self, source: pd.Series, leftbars: int, rightbars: int, waitForConfirmation: bool = True):
        series = source.append(pd.Series(np.full(rightbars, np.nan)), ignore_index=True)
        w = leftbars + rightbars + 1
        pphl = series.rolling(w, min_periods=leftbars+2).apply(
            raw=True, func=lambda x: self.__pivotpoint(x, leftbars, rightbars, waitForConfirmation)).shift(-rightbars)
        pphl = pphl.reindex(source.index)
        return pd.Series(pphl)
