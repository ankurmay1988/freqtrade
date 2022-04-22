import pandas as pd
import numpy as np
import pandas_ta as pta

# //@version=4
# study(title="Directional Movement Index", shorttitle="DMI", format=format.price, precision=4, resolution="")
# lensig = input(14, title="ADX Smoothing", minval=1, maxval=50)
# len = input(14, minval=1, title="DI Length")
# up = change(high)
# down = -change(low)
# plusDM = na(up) ? na : (up > down and up > 0 ? up : 0)
# minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
# trur = rma(tr, len)
# plus = fixnan(100 * rma(plusDM, len) / trur)
# minus = fixnan(100 * rma(minusDM, len) / trur)
# sum = plus + minus
# adx = 100 * rma(abs(plus - minus) / (sum == 0 ? 1 : sum), lensig)
# plot(adx, color=#F50057, title="ADX")
# plot(plus, color=#2962FF, title="+DI")
# plot(minus, color=#FF6D00, title="-DI")


def DMI(high: pd.Series, low: pd.Series, close: pd.Series, adx_smoothing: int = 14, DI_length: int = 14):
    up = high.diff()
    down = -1 * low.diff()
    plusDM = pd.Series(np.where((up > down) & (up > 0), up, 0))
    minusDM = pd.Series(np.where((down > up) & (down > 0), down, 0))
    tr = pta.true_range(high, low, close)
    trur = pta.rma(tr, DI_length)
    plus = (100 * pta.rma(plusDM, DI_length) / trur).ffill()
    minus = (100 * pta.rma(minusDM, DI_length) / trur).ffill()
    sum = plus + minus
    tmp = np.abs(plus - minus) / sum.where(sum == 0, 1)
    adx = 100 * pta.rma(tmp, adx_smoothing)
    return (adx, plus, minus)
