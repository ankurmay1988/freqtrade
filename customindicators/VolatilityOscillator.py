import pandas as pd
import numpy as np
import pandas_ta as pta

def VolatilityOscillator(close: pd.Series, open: pd.Series, length: int = 100):
    # Volatility oscillator
    spike = close - open
    volatility = pd.Series(pta.stdev(spike, length))
    upperLimit = volatility
    lowerLimit = -1 * volatility
    
    return (spike, upperLimit, lowerLimit)