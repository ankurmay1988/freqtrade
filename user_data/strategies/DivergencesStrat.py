# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

# --------- Strategy -----------
# Price < EMA200 (Bearish Trend):
# - (Buy)
#     * +ve divergence count >= 2
#     * Exclude MFI divergences from count
# - (Sell)
#     * -ve divergence count >= 1
#     * include all divergences
# ------------------------------
# Price > EMA200 (Bullish Trend):
# - (Buy)
#     * +ve divergence count >= 1
# - (Sell)
#     * -ve divergence count >= 2

from customindicators.DictObj import DictObj
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series
from pandas_ta.utils import data

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter

# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import customindicators as ci
# This class is a sample. Feel free to customize it.


class DivStrat(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.01

    # Trailing stop:
    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100
    }

    # Hyperoptable parameters
    # Pivot Period
    prd = IntParameter(default=5, low=1, high=50)
    # Source for Pivot Points
    source = CategoricalParameter(default="close", categories=["close", "high/low"])
    # Divergence Type
    searchdiv = CategoricalParameter(default="Regular", categories=[
                                     "Regular", "Hidden", "Regular/Hidden"])
    # Show Indicator Names
    showindis = CategoricalParameter(default="Full", categories=[
                                     "Full", "First Letter", "Don't Show"])
    # Minimum Number of Divergence
    showlimit = IntParameter(default=1, low=1, high=11)
    # Maximum Pivot Points to Check
    maxpp = IntParameter(default=10, low=1, high=20)
    # Maximum Bars to Check
    maxbars = IntParameter(default=100, low=30, high=200)

    osc_flags = {
        'stk': 2 ** 0,
        'mom': 2 ** 1,
        'mfi': 2 ** 2,
        'rsi': 2 ** 3,
        'cci': 2 ** 4,
        'cmf': 2 ** 5,
        'macd': 2 ** 6,
        'uo': 2 ** 7,
        'obv': 2 ** 8
    }

    oscillators = DictObj(osc_flags)
    maxflagnum = (2 ** len(osc_flags)) - 1

    # Buy osc signal flags
    buy_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                            space='buy', load=True, optimize=True)
    # Sell osc signal flags
    sell_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                             space='sell', load=True, optimize=True)

    minsignals = IntParameter(default=0, low=0, high=len(osc_flags),
                              space='buy', load=True, optimize=True)

    # Optimal timeframe for the strategy.
    timeframe = '30m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {

        },
        'subplots': {

        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]

        # add other pairs if needed maybe BTC ?
        informative_pairs += [
            ("BTC/USDT", "30m")
        ]

        return informative_pairs
        # return []

    """
    Adds several different TA indicators to the given DataFrame

    Performance Note: For the best performance be frugal on the number of indicators
    you are using. Let uncomment only the indicator you are using in your strategies
    or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
    :param dataframe: Dataframe with data from the exchange
    :param metadata: Additional information, like the currently traded pair
    :return: a Dataframe with all mandatory indicators for the strategies
    """

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.getOscFlags()

        dataframe['hlc3'] = dataframe.ta.hlc3()
        dataframe['rsi'] = dataframe.ta.rsi(14)
        dataframe['macd'], dataframe['deltamacd'], dataframe['signal'] = \
            zip(*dataframe.ta.macd(12, 26, 9).values)
        dataframe['mom'] = dataframe.ta.mom(10)
        dataframe['cci'] = dataframe.ta.cci(10)
        dataframe['obv'] = dataframe.ta.obv()
        stk, std = zip(*dataframe.ta.stoch().values)
        dataframe["stk"] = pd.Series(stk)
        dataframe['cmf'] = dataframe.ta.cmf()
        dataframe['mfi'] = dataframe.ta.mfi(14)
        dataframe['uo'] = dataframe.ta.uo()

        priceSource = dataframe['close'] if self.source.value == 'close' else dataframe['close']

        pp = ci.PivotPoint().pivotpoints(priceSource, self.prd.value, self.prd.value)
        ph = pd.Series(np.where(pp == 1, True, False))
        pl = pd.Series(np.where(pp == -1, True, False))

        dataframe['ph'] = ph
        dataframe['pl'] = pl

        indicatorList = ['rsi', 'mom', 'macd', 'cci', 'obv', 'stk', 'cmf', 'mfi', 'uo']
        for ind in indicatorList:
            dataframe['{}_rbull'.format(ind)], dataframe['{}_hbull'.format(ind)], dataframe['{}_rbear'.format(ind)], dataframe['{}_hbear'.format(ind)] = \
                self.calculateDivergences(priceSource=priceSource,
                                          osc=dataframe[ind], phFound=ph, plFound=pl)

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = dataframe.ta.cdl_pattern(name='ha')
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # dataframe['ha_green'] = dataframe['ha_open'] < dataframe['ha_close']
        # dataframe['ha_opencandle'] = (dataframe['ha_open'] == dataframe['ha_high']) | (dataframe['ha_open'] == dataframe['ha_low'])

        # dataframe.to_csv('/disk/freqtrade/data.csv')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        self.getOscFlags()
        n_signals = np.where(dataframe['cci_rbull'] & self.calccci, 1, 0) + \
            np.where(dataframe['cmf_rbull'] & self.calccmf, 1, 0) + \
            np.where(dataframe['macd_rbull'] & self.calcmacd, 1, 0) + \
            np.where(dataframe['mfi_rbull'] & self.calcmfi, 1, 0) + \
            np.where(dataframe['mom_rbull'] & self.calcmom, 1, 0) + \
            np.where(dataframe['obv_rbull'] & self.calcobv, 1, 0) + \
            np.where(dataframe['rsi_rbull'] & self.calcrsi, 1, 0) + \
            np.where(dataframe['stk_rbull'] & self.calcstoc, 1, 0) + \
            np.where(dataframe['uo_rbull'] & self.calcuo, 1, 0)

        cond = (dataframe['cci_rbull'] & self.calccci) \
            | (dataframe['cmf_rbull'] & self.calccmf) \
            | (dataframe['macd_rbull'] & self.calcmacd) \
            | (dataframe['mfi_rbull'] & self.calcmfi) \
            | (dataframe['mom_rbull'] & self.calcmom) \
            | (dataframe['obv_rbull'] & self.calcobv) \
            | (dataframe['rsi_rbull'] & self.calcrsi) \
            | (dataframe['stk_rbull'] & self.calcstoc) \
            | (dataframe['uo_rbull'] & self.calcuo)

        dataframe.loc[
            (
                cond & (n_signals > self.minsignals.value)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        self.getOscFlags()

        cond = (dataframe['cci_rbear'] & self.sell_calccci) \
            | (dataframe['cmf_rbear'] & self.sell_calccmf) \
            | (dataframe['macd_rbear'] & self.sell_calcmacd) \
            | (dataframe['mfi_rbear'] & self.sell_calcmfi) \
            | (dataframe['mom_rbear'] & self.sell_calcmom) \
            | (dataframe['obv_rbear'] & self.sell_calcobv) \
            | (dataframe['rsi_rbear'] & self.sell_calcrsi) \
            | (dataframe['stk_rbear'] & self.sell_calcstoc) \
            | (dataframe['uo_rbear'] & self.sell_calcuo)
        dataframe.loc[
            (
                cond
            ),
            'sell'] = 1
        return dataframe

    def getOscFlags(self):
        # MACD
        self.calcmacd = (self.buy_flag.value & self.oscillators.macd) > 0
        # RSI
        self.calcrsi = (self.buy_flag.value & self.oscillators.rsi) > 0
        # Stochastic
        self.calcstoc = (self.buy_flag.value & self.oscillators.stk) > 0
        # CCI
        self.calccci = (self.buy_flag.value & self.oscillators.cci) > 0
        # Momentum
        self.calcmom = (self.buy_flag.value & self.oscillators.mom) > 0
        # OBV
        self.calcobv = (self.buy_flag.value & self.oscillators.obv) > 0
        # Chaikin Money Flow
        self.calccmf = (self.buy_flag.value & self.oscillators.cmf) > 0
        # Money Flow Index
        self.calcmfi = (self.buy_flag.value & self.oscillators.mfi) > 0
        # Ultimate Oscillator
        self.calcuo = (self.buy_flag.value & self.oscillators.uo) > 0

        # MACD
        self.sell_calcmacd = (self.sell_flag.value & self.oscillators.macd) > 0
        # RSI
        self.sell_calcrsi = (self.sell_flag.value & self.oscillators.rsi) > 0
        # Stochastic
        self.sell_calcstoc = (self.sell_flag.value & self.oscillators.stk) > 0
        # CCI
        self.sell_calccci = (self.sell_flag.value & self.oscillators.cci) > 0
        # Momentum
        self.sell_calcmom = (self.sell_flag.value & self.oscillators.mom) > 0
        # OBV
        self.sell_calcobv = (self.sell_flag.value & self.oscillators.obv) > 0
        # Chaikin Money Flow
        self.sell_calccmf = (self.sell_flag.value & self.oscillators.cmf) > 0
        # Money Flow Index
        self.sell_calcmfi = (self.sell_flag.value & self.oscillators.mfi) > 0
        # Ultimate Oscillator
        self.sell_calcuo = (self.sell_flag.value & self.oscillators.uo) > 0

    def valuewhen(self, condition, source, occurrence):
        """
        First drop unwanted values using reindex and a condition as a mask.
        Next, shift the series according to which occurrence is desired.
        Then, use reindex to add in the index values that was dropped from the mask and first reindex.
        These index values will point to np.nan. Finally, use ffill() to forward fill values onto the np.nan values.
        This assumes that occurrences is a properly bounded non-negative number, source is an ordered sequence,
        and the condition is related to the source by their index.
        This can be written in Python like this:
        """
        return source \
            .reindex(condition[condition].index) \
            .shift(-occurrence) \
            .reindex(source.index) \
            .ffill()

    def calculateDivergences(self, priceSource: Series, osc: Series, phFound: Series, plFound: Series):
        oscL = osc.loc[plFound]
        oscH = osc.loc[phFound]
        priceSourceL = priceSource.loc[plFound]
        priceSourceH = priceSource.loc[phFound]

        # ------------------------------------------------------------------------------
        # Regular Bullish
        # Osc: Higher Low
        oscHL = (oscL.diff() > 0) | (oscL.diff(2) > 0)

        # Price: Lower Low
        priceLL = (priceSourceL.diff() < 0) | (priceSourceL.diff(2) > 0)
        bullCond = priceLL & oscHL & plFound

        # ------------------------------------------------------------------------------
        # Hidden Bullish
        # Osc: Lower Low
        oscLL = (oscL.diff() < 0) | (oscL.diff(2) < 0)

        # Price: Higher Low
        priceHL = (priceSourceL.diff() > 0) | (priceSourceL.diff(2) > 0)
        hiddenBullCond = priceHL & oscLL & plFound

        # ------------------------------------------------------------------------------
        # Regular Bearish
        # Osc: Lower High
        oscLH = (oscH.diff() < 0) | (oscH.diff(2) < 0)

        # Price: Higher High
        priceHH = (priceSourceH.diff() > 0) | (priceSourceH.diff(2) > 0)

        bearCond = priceHH & oscLH & phFound

        # ------------------------------------------------------------------------------
        # Hidden Bearish
        # Osc: Higher High
        oscHH = (oscH.diff() > 0) | (oscH.diff(2) > 0)

        # Price: Lower High
        priceLH = (priceSourceH.diff() < 0) | (priceSourceH.diff(2) < 0)

        hiddenBearCond = priceLH & oscHH & phFound

        return (
            bullCond.fillna(value=False),
            hiddenBullCond.fillna(value=False),
            bearCond.fillna(value=False),
            hiddenBearCond.fillna(value=False)
        )

# Results:
# Timeframe: 2021-05-13 - 2021-07-20
# HODL: -50%
# rsi -> 804%
# uo -> 632%
# obv -> 602%
# mom -> 1096%
# macd -> 800%
# cci -> 964%
# stk -> 6976%
# cmf -> 918%
# mfi -> 1075%
# flag = stk,mom,mfi,rsi,cci,cmf,macd,uo,obv
