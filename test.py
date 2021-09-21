import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import customindicators as ci

l = pd.Series([3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 7, 5, 6, 2, 1, 0, 3, 1, 0])
pphl = ci.PivotPoint().pivotpoints(l, 2)
print(pphl)
a = [*l, 90]
print(a)
