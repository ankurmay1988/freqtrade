import pandas as pd
import numpy as np
import customindicators as ci

l = pd.Series([3, 1, 2, 3, 4, 5, 4, 2, 1, 2, 7, 5, 6, 2, 1, 0, 3])
pphl = ci.PivotPoint().pivotpoints(l, 4, 4, False)
print(pphl)
