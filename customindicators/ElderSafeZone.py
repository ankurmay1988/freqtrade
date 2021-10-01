import pandas as pd
import numpy as np

# // Elder SafeZone Stops
# // Converted from http://chartingwithchris.blogspot.com/2008/10/elder-safezone-stop-system-thinkorswim.html
# //@version=2
# study("Elder SafeZone", overlay=true)

# coeff = input(2.5, "CoEff", type=float)
# lookbackLength = input(15, "LookBackLength", type=integer)

# countShort = high > high[1] ? 1 : 0
# diffShort = high > high[1] ? high - high[1] : 0
# totalCountShort = sum(countShort, lookbackLength)
# totalSumShort = sum(diffShort, lookbackLength)
# penAvgShort = (totalSumShort / totalCountShort)
# safetyShort = high[1] + (penAvgShort[1] * coeff)
# finalSafetyShort = min(min(safetyShort, safetyShort[1]), safetyShort[2])

# count = low < low[1] ? 1 : 0
# diff = low < low[1] ? low[1] - low : 0
# totalCount = sum(count, lookbackLength)
# totalSum = sum(diff, lookbackLength)
# penAvg = (totalSum / totalCount)
# safety = low[1] - (penAvg[1] * coeff)
# finalSafetyLong = max(max(safety, safety[1]), safety[2])

# p1 = plot(finalSafetyShort, "Short Stop", color=#ff00ff)
# p2 = plot(finalSafetyLong, "Long Stop", color=#ff00ff)


def ElderSafeZone(high: pd.Series, low: pd.Series, coeff: float = 2.5, lookbackLength: int = 15):
    highdiff = high.diff()
    higherHigh = highdiff > 0
    countShort = pd.Series(np.where(higherHigh, 1, 0))
    diffShort = pd.Series(np.where(higherHigh, highdiff, 0))
    totalCountShort = countShort.rolling(window=lookbackLength, min_periods=1).sum()
    totalSumShort = diffShort.rolling(window=lookbackLength, min_periods=1).sum()
    penAvgShort = (totalSumShort / totalCountShort).round(3)
    safetyShort = (high.shift(1) + (penAvgShort.shift(1) * coeff)).round(3)
    finalSafetyShort = safetyShort.rolling(3).min()
    
    lowdiff = low.diff()
    lowerLow = lowdiff < 0
    count = pd.Series(np.where(lowerLow, 1, 0))
    diff = pd.Series(np.where(lowerLow, lowdiff, 0))
    totalCount = count.rolling(window=lookbackLength, min_periods=1).sum()
    totalSum = diff.rolling(window=lookbackLength, min_periods=1).sum()
    penAvg = (totalSum / totalCount).round(3)
    safety = (low.shift(1) - (penAvg.shift(1) * coeff)).round(3)
    finalSafetyLong = safety.rolling(3).max()
    
    return (finalSafetyLong, finalSafetyShort)
