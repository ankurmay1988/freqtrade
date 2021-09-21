# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

from datetime import datetime, timedelta
from math import pi
from typing import Sequence
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexing import IndexSlice
from pandas.core.series import Series

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter, BooleanParameter

# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import customindicators as ci
# This class is a sample. Feel free to customize it.


class DivStratAngle(IStrategy):
    """
    https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.013

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

    oscillators = ci.DictObj(osc_flags)
    maxflagnum = (2 ** len(osc_flags)) - 1

    # Hyperoptable parameters
    # Pivot Period
    prd = IntParameter(default=3, low=1, high=10,
                       space='buy', load=True, optimize=True)
    # Source for Pivot Points
    source = CategoricalParameter(default="close", categories=["close", "high/low"])
    # Maximum Pivot Points to Check
    maxpp = IntParameter(default=3, low=1, high=10,
                         space='buy', load=True, optimize=True)

    # Buy osc signal flags
    buy_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                            space='buy', load=True, optimize=True)
    
    buy_minsignals = IntParameter(default=0, low=0, high=len(osc_flags),
                                  space='buy', load=True, optimize=True)
    
    buy_down_minsignals = IntParameter(default=3, low=0, high=len(osc_flags),
                                  space='buy', load=True, optimize=True)
    # Sell osc signal flags
    sell_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                             space='sell', load=True, optimize=True)

    sell_minsignals = IntParameter(default=0, low=0, high=len(osc_flags),
                                   space='sell', load=True, optimize=True)

    # Optimal timeframe for the strategy.
    timeframe = '30m'
    timeframe_above = '1h'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

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
        informative_pairs = [(pair, self.timeframe_above) for pair in pairs]

        # add other pairs if needed maybe BTC ?
        # informative_pairs += [("BTC/USDT", self.timeframe)]

        return informative_pairs
        # return []

    def calc_indicators(self, dataframe: DataFrame):
        self.getOscFlags()

        dataframe['hlc3'] = dataframe.ta.hlc3()
        dataframe['rsi'] = dataframe.ta.rsi(14)
        dataframe['macd'], dataframe['deltamacd'], dataframe['signal'] = zip(
            *dataframe.ta.macd(12, 26, 9).values)
        dataframe['mom'] = dataframe.ta.mom(10)
        dataframe['cci'] = dataframe.ta.cci(10)
        dataframe['obv'] = dataframe.ta.obv()
        stk, std = zip(*dataframe.ta.stoch().values)
        dataframe["stk"] = pd.Series(stk)
        dataframe['cmf'] = dataframe.ta.cmf()
        dataframe['mfi'] = dataframe.ta.mfi(14)
        dataframe['uo'] = dataframe.ta.uo()
        dataframe['ema200'] = dataframe.ta.ema(200)
        dataframe['sma50'] = dataframe.ta.sma(50)
        
        priceSource = dataframe['close'] if self.source.value == 'close' else dataframe['close']

        pp = ci.PivotPoint().pivotpoints(priceSource, self.prd.value)
        ph = pd.Series(np.where(pp == 1, True, False))
        pl = pd.Series(np.where(pp == -1, True, False))

        dataframe['ph'] = ph
        dataframe['pl'] = pl

        indicatorList = list(self.osc_flags.keys())
        for ind in indicatorList:
            dataframe = self.calculateDivergenceData(dataframe, ind, 'close')


        self.calculateDivergences(dataframe, 0)

        dataframe['uptrend'] = dataframe['sma50'] > dataframe['ema200']
        dataframe['price_above_sma50'] = (dataframe['hlc3'] > dataframe['sma50'])
        dataframe['price_above_ema200'] = (dataframe['hlc3'] > dataframe['ema200'])
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict):
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        dataframe = self.calc_indicators(dataframe)

        # if not self.dp is None:
        #     df1h = self.dp.get_pair_dataframe(metadata['pair'], self.timeframe_above)
        #     df1h = self.calc_indicators(df1h)
        #     dataframe = merge_informative_pair(
        #         dataframe, df1h, self.timeframe, self.timeframe_above, ffill=True)

        # # Graph object to plot and analyze divergence data
        graphData = DataFrame()
        
        dataframe.to_csv('D:\\data.csv')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        self.getOscFlags()
        indicatorList = list(self.osc_flags.keys())

        uptrend = dataframe['uptrend']
        price_above_sma50 = dataframe['price_above_sma50']
        price_above_ema200 = dataframe['price_above_ema200']
        buycondition = np.full(dataframe['close'].shape[0], False)
        
        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bullStr = '{}_rbull'.format(ind)
            osc_condition = (dataframe[bullStr]).shift(1).fillna(False) \
                & (uptrend & price_above_ema200 & self.oscBuyFlags[ind])
            n_signals += np.where(osc_condition, 1, 0)
            cond = cond | osc_condition

        min_signal = (n_signals > self.buy_minsignals.value)

        signal = cond & min_signal
        dataframe.loc[signal, 'buy_tag'] = 'rbull_30m'
        buycondition = buycondition | signal
        
        # Use regular bullish in downtrend condition
        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bullStr = '{}_rbull'.format(ind)
            osc_condition = (dataframe[bullStr]).shift(1).fillna(False) \
                & (~uptrend & (~price_above_ema200 | ~price_above_sma50) & self.oscBuyFlags[ind])
            n_signals += np.where(osc_condition, 1, 0)
            cond |= osc_condition

        min_signal = (n_signals > self.buy_down_minsignals.value)

        signal = cond & min_signal
        dataframe.loc[signal, 'buy_tag'] = 'rbull_30m_downtrend'
        buycondition |= signal

        # When chart show hidden bullish divergence and price is above the 200 ema & 50 sma
        # that means trend will continue to be bullish so we can buy at this time also. We
        # can ignore minimum signal count because trend is quite strong when this happens.
        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bullStr = '{}_hbull'.format(ind)
            osc_condition = dataframe[bullStr].shift(1).fillna(False) \
                & uptrend & price_above_sma50 & self.oscBuyFlags[ind]
            n_signals += np.where(osc_condition, 1, 0)
            cond |= osc_condition

        signal = cond
        dataframe.loc[signal, 'buy_tag'] = 'hbull_30m'
        buycondition |= signal
        
        # When 1h chart show regular bullish divergence
        # n_signals = np.full(dataframe['close'].shape[0], 0)
        # cond = np.full(dataframe['close'].shape[0], False)
        # for ind in indicatorList:
        #     bullStr = '{}_rbull_{}'.format(ind, self.timeframe_above)
        #     osc_condition = dataframe[bullStr].shift(1).fillna(False) \
        #         & ((uptrend & self.oscBuyFlags[ind]) | (~uptrend & self.oscBuyDowntrendFlags[ind]))
        #     n_signals += np.where(osc_condition, 1, 0)
        #     cond |= osc_condition

        # min_signal = (((uptrend & (n_signals > self.buy_minsignals.value)) |
        #               (~uptrend & (n_signals > self.buy_down_minsignals.value))))

        # signal = cond & min_signal
        # dataframe.loc[signal, 'buy_tag'] = 'rbull_1h'
        # buycondition |= signal

        dataframe.loc[
            (
                buycondition
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        self.getOscFlags()
        indicatorList = list(self.osc_flags.keys())

        uptrend = dataframe['uptrend']
        price_above_sma50 = dataframe['price_above_sma50']
        price_above_ema200 = dataframe['price_above_ema200']
        
        sellcondition = np.full(dataframe['close'].shape[0], False)
        
        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bearStr = '{}_rbear'.format(ind)
            osc_condition = dataframe[bearStr].shift(1).fillna(False) \
                & (self.oscSellFlags[ind])
            n_signals += np.where(osc_condition, 1, 0)
            cond |= osc_condition

        min_signal = (n_signals > self.sell_minsignals.value)

        signal = cond & min_signal
        dataframe.loc[signal, 'sell_tag'] = 'sell_rbear_30m'
        sellcondition |= signal
        
        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bearStr = '{}_hbear'.format(ind)
            osc_condition = dataframe[bearStr].shift(1).fillna(False) \
                & (self.oscSellFlags[ind])
            n_signals += np.where(osc_condition, 1, 0)
            cond |= osc_condition

        min_signal = (n_signals > self.sell_minsignals.value)

        signal = cond & min_signal
        dataframe.loc[signal, 'sell_tag'] = 'sell_hbear_30m'
        sellcondition |= signal

        # # Using 1h timeframe is it supposed to be more accurate
        # n_signals = np.full(dataframe['close'].shape[0], 0)
        # cond = np.full(dataframe['close'].shape[0], False)
        # for ind in indicatorList:
        #     bearStr = '{}_rbear_{}'.format(ind, self.timeframe_above)
        #     osc_condition = dataframe[bearStr].shift(1).fillna(False) \
        #         & ((uptrend & self.oscSellFlags[ind]) | (~uptrend & self.oscSellDowntrendFlags[ind]))
        #     n_signals += np.where(osc_condition, 1, 0)
        #     cond |= osc_condition

        # min_signal = (((uptrend & (n_signals > self.sell_minsignals.value)) |
        #               (~uptrend & (n_signals > self.sell_down_minsignals.value))))

        # signal = cond & min_signal
        # dataframe.loc[signal, 'sell_tag'] = 'sell_rbear_1h'
        # sellcondition |= signal

        # n_signals = np.full(dataframe['close'].shape[0], 0)
        # cond = np.full(dataframe['close'].shape[0], False)
        # for ind in indicatorList:
        #     bearStr = '{}_hbear_{}'.format(ind, self.timeframe_above)
        #     osc_condition = dataframe[bearStr].shift(1).fillna(False) \
        #         & ((uptrend & self.oscSellFlags[ind]) | (~uptrend & self.oscSellDowntrendFlags[ind]))
        #     n_signals += np.where(osc_condition, 1, 0)
        #     cond |= osc_condition

        # min_signal = (((uptrend & (n_signals > self.sell_minsignals.value)) |
        #               (~uptrend & (n_signals > self.sell_down_minsignals.value))))

        # signal = cond & min_signal
        # dataframe.loc[signal, 'sell_tag'] = 'sell_hbear_1h'
        # sellcondition |= signal

        dataframe.loc[
            (
                sellcondition
            ),
            'sell'] = 1
        return dataframe
    
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if trade.buy_tag == 'rbull_30m_downtrend' and last_candle['open'] > last_candle['close']:
            return 'sell_candle_red_downtrend'

        return None

    def getOscFlags(self):
        self.oscBuyFlags = {}
        self.oscSellFlags = {}
        indicatorList = list(self.osc_flags.keys())
        for ind in indicatorList:
            self.oscBuyFlags[ind] = (self.buy_flag.value & self.osc_flags[ind]) > 0
            self.oscSellFlags[ind] = (self.sell_flag.value & self.osc_flags[ind]) > 0

    def calculateDivergenceData(self, dataframe: DataFrame, oscName: str, source: str):
        mpp = self.maxpp.value
        plFound = dataframe['pl']
        phFound = dataframe['ph']
        osc = dataframe[oscName]
        oscL = osc.loc[plFound]
        oscH = osc.loc[phFound]
        priceSource = dataframe[source]
        priceSourceL = priceSource.loc[plFound]
        priceSourceH = priceSource.loc[phFound]

        oscData = DataFrame()

        for ppidx in range(0, mpp):
            dataframe[('price', 'low', ppidx, 'diff')] = priceSourceL.diff(ppidx + 1)
            dataframe[('price', 'high', ppidx, 'diff')] = priceSourceH.diff(ppidx + 1)

        for ppidx in range(0, mpp):
            oscData[(osc.name, 'low', ppidx, 'diff')] = oscLDiff = oscL.diff(ppidx + 1)
            oscData[(osc.name, 'high', ppidx, 'diff')] = oscHDiff = oscH.diff(ppidx + 1)

            # oscVec = DataFrame({'unit': ppidx+1, 'diff': oscLDiff.fillna(0)}, index=oscLDiff.index)
            # norm = pd.Series(np.linalg.norm(oscVec, axis=1), index=oscVec.index)
            # oscUnitVec = oscVec['unit'] / norm
            # oscLangle = (np.arccos(oscUnitVec)/pi) * 180

            # oscVec = DataFrame({'unit': ppidx+1, 'diff': oscHDiff.fillna(0)}, index=oscHDiff.index)
            # norm = pd.Series(np.linalg.norm(oscVec, axis=1), index=oscVec.index)
            # oscUnitVec = oscVec['unit'] / norm
            # oscHangle = (np.arccos(oscUnitVec)/pi) * 180

            # oscData[(osc.name, 'low', ppidx, 'angle')] = oscLangle
            # oscData[(osc.name, 'high', ppidx, 'angle')] = oscHangle

        return pd.concat([dataframe, oscData], axis=1)

    def calculateDivergences(self, dataframe: DataFrame, min_angle: float):
        mpp = self.maxpp.value
        idx = pd.IndexSlice
        indicators = list(self.osc_flags.keys())
        indexes = MultiIndex.from_product(
            [[*indicators, 'price'], ['ll', 'lh', 'hl', 'hh'], range(0, mpp)])
        oscDiv = DataFrame(columns=indexes)
        for ind in indicators:
            for ppidx in range(0, mpp):
                oscLDiff = dataframe[(ind, 'low', ppidx, 'diff')]
                priceLDiff = dataframe[('price', 'low', ppidx, 'diff')]
                oscHDiff = dataframe[(ind, 'high', ppidx, 'diff')]
                priceHDiff = dataframe[('price', 'high', ppidx, 'diff')]
                # oscLangle = dataframe[(ind, 'low', ppidx, 'angle')]
                # oscHangle = dataframe[(ind, 'high', ppidx, 'angle')]

                # ------------------------------------------------------------------------------
                #   Regular Bullish
                # ------------------------------------------------------------------------------
                # Osc: Higher Low
                oscDiv[(ind, 'hl', ppidx)] = (oscLDiff > 0) #& (oscLangle > min_angle)
                # Price: Lower Low
                oscDiv[('price', 'll', ppidx)] = (priceLDiff < 0)

                # ------------------------------------------------------------------------------
                #   Hidden Bullish
                # ------------------------------------------------------------------------------
                # Osc: Lower Low
                oscDiv[(ind, 'll', ppidx)] = (oscLDiff < 0) #& (oscLangle > min_angle)
                # Price: Higher Low
                oscDiv[('price', 'hl', ppidx)] = (priceLDiff > 0)

                # ------------------------------------------------------------------------------
                #   Regular Bearish
                # ------------------------------------------------------------------------------
                # Osc: Lower High
                oscDiv[(ind, 'lh', ppidx)] = (oscHDiff < 0) #& (oscHangle > min_angle)
                # Price: Higher High
                oscDiv[('price', 'hh', ppidx)] = (priceHDiff > 0)

                # ------------------------------------------------------------------------------
                #   Hidden Bearish
                # ------------------------------------------------------------------------------
                # Osc: Higher High
                oscDiv[(ind, 'hh', ppidx)] = (oscHDiff > 0) #& (oscHangle > min_angle)
                # Price: Lower High
                oscDiv[('price', 'lh', ppidx)] = (priceHDiff < 0)

            priceLL = oscDiv.loc[:, idx['price', 'll', :]][('price', 'll')]
            priceLH = oscDiv.loc[:, idx['price', 'lh', :]][('price', 'lh')]
            priceHL = oscDiv.loc[:, idx['price', 'hl', :]][('price', 'hl')]
            priceHH = oscDiv.loc[:, idx['price', 'hh', :]][('price', 'hh')]
            oscLL = oscDiv.loc[:, idx[ind, 'll', :]][(ind, 'll')]
            oscLH = oscDiv.loc[:, idx[ind, 'lh', :]][(ind, 'lh')]
            oscHL = oscDiv.loc[:, idx[ind, 'hl', :]][(ind, 'hl')]
            oscHH = oscDiv.loc[:, idx[ind, 'hh', :]][(ind, 'hh')]
            bullCond = (priceLL & oscHL).apply(np.any, axis=1) & dataframe['pl']
            hiddenBullCond = (priceHL & oscLL).apply(np.any, axis=1) & dataframe['pl']
            bearCond = (priceHH & oscLH).apply(np.any, axis=1) & dataframe['ph']
            hiddenBearCond = (priceLH & oscHH).apply(np.any, axis=1) & dataframe['ph']
            dataframe['{}_rbull'.format(ind)] = bullCond.fillna(value=False)
            dataframe['{}_hbull'.format(ind)] = hiddenBullCond.fillna(value=False)
            dataframe['{}_rbear'.format(ind)] = bearCond.fillna(value=False)
            dataframe['{}_hbear'.format(ind)] = hiddenBearCond.fillna(value=False)
