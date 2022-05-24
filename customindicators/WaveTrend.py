import pandas as pd
import numpy as np
import pandas_ta as pta

def WaveTrendOscillator(close: pd.Series, channelLength: int = 9, avgLength: int = 12, smoothingLength: int = 3):
    # WaveTrend oscillator
    ap = close
    esa = pd.Series(pta.ema(ap, channelLength))
    d = pd.Series(pta.ema((ap-esa).abs(), channelLength))
    ci = (ap - esa) / (0.015 * d)
    tci = pd.Series(pta.ema(ci, avgLength))
    
    wt1 = tci
    wt2 = pd.Series(pta.sma(tci, smoothingLength))
    
    return (wt1, wt2)