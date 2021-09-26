# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

from customindicators.ElderSafeZone import ElderSafeZone
from datetime import datetime, timedelta
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter, BooleanParameter
from freqtrade.persistence import Trade

# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import customindicators as ci
# This class is a sample. Feel free to customize it.


class DivStratNew2(IStrategy):
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
    stoploss = -0.1
    
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
                       space='buy', load=True, optimize=False)
    # Source for Pivot Points
    source = CategoricalParameter(default="close", categories=["close", "high/low"])
    # Maximum Pivot Points to Check
    maxpp = IntParameter(default=3, low=1, high=10,
                         space='buy', load=True, optimize=False)

    # Buy osc signal flags
    buy_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                            space='buy', load=True, optimize=True)
    # Sell osc signal flags
    sell_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                             space='sell', load=True, optimize=True)

    buy_minsignals = IntParameter(default=1, low=0, high=len(osc_flags),
                                  space='buy', load=True, optimize=True)

    sell_minsignals = IntParameter(default=1, low=0, high=len(osc_flags),
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
        # pairs = self.dp.current_whitelist()

        # # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.timeframe_above) for pair in pairs]

        # # add other pairs if needed maybe BTC ?
        # # informative_pairs += [("BTC/USDT", self.timeframe)]

        # return informative_pairs
        return []

    def calc_indicators(self, dataframe: DataFrame):
        self.getOscFlags()

        dataframe['hlc3'] = dataframe.ta.hlc3()
        dataframe['rsi'] = dataframe.ta.rsi(14).round(2)
        dataframe['macd'], dataframe['deltamacd'], dataframe['signal'] = zip(
            *dataframe.ta.macd(9, 21, 5).values)
        dataframe['mom'] = dataframe.ta.mom(10).round(2)
        dataframe['cci'] = dataframe.ta.cci(10).round(2)
        dataframe['obv'] = dataframe.ta.obv().round(2)
        stk, std = zip(*dataframe.ta.stoch().values)
        dataframe["stk"] = pd.Series(stk)
        dataframe['cmf'] = dataframe.ta.cmf().round(2)
        dataframe['mfi'] = dataframe.ta.mfi(14).round(2)
        dataframe['uo'] = dataframe.ta.uo().round(2)
        dataframe['ema200'] = dataframe.ta.ema(200).round(2)
        dataframe['sma50'] = dataframe.ta.sma(50).round(2)
        psar_l, psar_s, psar_af, psar_reversal = zip(*dataframe.ta.psar(af0=0.03, af=0.03, af_max=0.3).values)
        dataframe['psar_l'] = pd.Series(psar_l)
        strsik, strsid = zip(*pta.stochrsi(dataframe['close'], 14, 14, 14, 3).values)
        dataframe["strsik"] = pd.Series(strsik).round(2)
        longStop, shortStop = ElderSafeZone(dataframe['high'], dataframe['low'])
        dataframe["elderSafezoneLong"] = longStop
        
        priceSource = dataframe['close'] if self.source.value == 'close' else dataframe['close']

        pp = ci.PivotPoint().pivotpoints(priceSource, self.prd.value)
        ph = pd.Series(np.where(pp == 1, True, False))
        pl = pd.Series(np.where(pp == -1, True, False))

        dataframe['ph'] = ph
        dataframe['pl'] = pl

        indicatorList = list(self.osc_flags.keys())
        for ind in indicatorList:
            dataframe['{}_rbull'.format(ind)], dataframe['{}_hbull'.format(ind)], dataframe['{}_rbear'.format(ind)], dataframe['{}_hbear'.format(ind)] = \
                self.calculateDivergences(priceSource=priceSource,
                                          osc=dataframe[ind], phFound=ph, plFound=pl)

        dataframe['uptrend'] = dataframe['sma50'] > dataframe['ema200']
        dataframe['price_above_sma50'] = (dataframe['hlc3'] > dataframe['sma50'])
        dataframe['price_above_ema200'] = (dataframe['hlc3'] > dataframe['ema200'])
        dataframe['macd_up'] = dataframe['deltamacd'].round(2) > 0
        dataframe['rsi_crosses_50'] = ((dataframe['rsi'].shift(1) < 50) & (dataframe['rsi'] > 50)).ffill().fillna(False)
        dataframe['rsi_crosses_70'] = ((dataframe['rsi'].shift(1) < 70) & (dataframe['rsi'] > 70)).ffill().fillna(False)
        stochrsi_logic = dataframe['strsik'].apply(lambda x: True if x < 20 else False if x > 80 else np.nan)
        dataframe['stochrsi_oversold'] = stochrsi_logic.ffill().fillna(False)
        dataframe['buy_rsi_stochrsi_macd'] = dataframe['rsi_crosses_50'] & dataframe['macd_up'] & dataframe['stochrsi_oversold']
        dataframe['sell_rsi_stochrsi_macd'] = dataframe['rsi_crosses_70']
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
        # graphData = DataFrame()
        # graphData = graphData.append(dataframe.loc[:, ["date", "open", "high", "low", "close"]])
        # indicatorList = list(self.osc_flags.keys())
        # for ind in indicatorList:
        #     rbull = "{}_rbull".format(ind)
        #     rbear = "{}_rbear".format(ind)
        #     hbull = "{}_hbull".format(ind)
        #     hbear = "{}_hbear".format(ind)
        #     # graphData = graphData.append(dataframe.loc[:, [rbull, rbear, hbull, hbear]])
        #     graphData[rbull] = np.where(dataframe[rbull], dataframe['low'], np.nan)
        #     graphData[rbear] = np.where(dataframe[rbear], dataframe['high'], np.nan)
        #     graphData[hbull] = np.where(dataframe[hbull], dataframe['low'], np.nan)
        #     graphData[hbear] = np.where(dataframe[hbear], dataframe['high'], np.nan)
        # graphData.to_csv('D:\\data.csv')
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
        buycondition = pd.Series(np.full(dataframe['close'].size, False))
        uptrend = dataframe['uptrend']
        price_above_sma50 = dataframe['price_above_sma50']
        price_above_ema200 = dataframe['price_above_ema200']
        macd_up = dataframe['macd_up']
        rsi_crosses_50 = dataframe['rsi_crosses_50']
        
        n_signals = np.full(dataframe['close'].size, 0)
        cond = np.full(dataframe['close'].size, False)
        for ind in indicatorList:
            bullStr = '{}_rbull'.format(ind)
            osc_condition = dataframe[bullStr].shift(1).fillna(False) & self.oscBuyFlags[ind]
            n_signals += np.where(osc_condition, 1, 0)
            cond = cond | osc_condition

        if self.buy_minsignals.value > 0:
            signal = cond & (n_signals > self.buy_minsignals.value)
        else:
            signal = cond

        signal = (signal | signal.shift(1)) & (rsi_crosses_50 | rsi_crosses_50.shift(1)) & (macd_up | macd_up.shift(1))
        dataframe.loc[signal, 'buy_tag'] = 'rbull_30m'
        buycondition = buycondition | signal
            
        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bullStr = '{}_hbull'.format(ind)
            osc_condition = dataframe[bullStr].shift(1).fillna(False) & self.oscBuyFlags[ind]
            n_signals += np.where(osc_condition, 1, 0)
            cond = cond | osc_condition
            
        cond = cond & price_above_sma50

        if self.buy_minsignals.value > 0:
            signal = cond & (n_signals > self.buy_minsignals.value)
        else:
            signal = cond

        dataframe.loc[signal, 'buy_tag'] = 'hbull_30m'
        buycondition = buycondition | signal
        
        # n_signals = np.full(dataframe['close'].size, 0)
        # cond = np.full(dataframe['close'].size, False)
        # for ind in indicatorList:
        #     bullStr = '{}_rbull'.format(ind)
        #     osc_condition = dataframe[bullStr].shift(1).fillna(False)
        #     n_signals += np.where(osc_condition, 1, 0)
        #     cond = cond | osc_condition

        # cond = cond & (~uptrend) & (~price_above_ema200)
        
        # signal = cond & (n_signals > 5)
        
        signal = dataframe["buy_rsi_stochrsi_macd"] & (~uptrend) & (~price_above_ema200)
        dataframe.loc[signal, 'buy_tag'] = 'rbull_30m_downtrend'
        buycondition = buycondition | signal

        dataframe.loc[buycondition, 'buy'] = 1

        return dataframe.copy()

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        self.getOscFlags()
        indicatorList = list(self.osc_flags.keys())
        sellcondition = pd.Series(np.full(dataframe['close'].size, False))
        uptrend = dataframe['uptrend']
        price_above_sma50 = dataframe['price_above_sma50']
        price_above_ema200 = dataframe['price_above_ema200']

        n_signals = np.full(dataframe['close'].shape[0], 0)
        cond = np.full(dataframe['close'].shape[0], False)
        for ind in indicatorList:
            bearStr = '{}_rbear'.format(ind)
            osc_condition = dataframe[bearStr].shift(1).fillna(False) & self.oscSellFlags[ind]
            n_signals += np.where(osc_condition, 1, 0)
            cond = cond | osc_condition

        # cond = cond & uptrend & price_above_ema200
        if self.sell_minsignals.value > 0:
            signal = cond & (n_signals > self.sell_minsignals.value)
        else:
            signal = cond
            
        dataframe.loc[signal, 'sell_tag'] = 'rbear_30m'
        sellcondition = sellcondition | signal
        
        # n_signals = np.full(dataframe['close'].shape[0], 0)
        # cond = np.full(dataframe['close'].shape[0], False)
        # for ind in indicatorList:
        #     bearStr = '{}_hbear'.format(ind)
        #     osc_condition = dataframe[bearStr].shift(1).fillna(False) & self.oscSellFlags[ind]
        #     n_signals += np.where(osc_condition, 1, 0)
        #     cond = cond | osc_condition

        # cond = cond & uptrend & price_above_sma50
        
        # if self.sell_minsignals.value > 0:
        #     signal = cond & (n_signals > self.sell_minsignals.value)
        # else:
        #     signal = cond
            
        # dataframe.loc[signal, 'sell_tag'] = 'hbear_30m'
        # sellcondition = sellcondition | signal

        dataframe.loc[sellcondition, 'sell'] = 1
        
        return dataframe.copy()
    
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        profit_pct = current_profit * 100
        if trade.buy_tag == 'rbull_30m_downtrend':
            # if profit_pct > 1.2 or last_candle['sell_rsi_stochrsi_macd']:
            if last_candle['sell_rsi_stochrsi_macd']:
                return 'sell_candle_red_downtrend'

        return None
    
    use_custom_stoploss = True
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        profit_pct = current_profit * 100
        # Use parabolic sar as absolute stoploss price
        # stoploss_price = last_candle['psar_l']
        stoploss_price = last_candle["elderSafezoneLong"]
        # Convert absolute price to percentage relative to current_rate
        if trade.buy_tag != 'rbull_30m_downtrend' and stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1
        
        # if profit_pct > 4:
        #     return 0.01
        
        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1

    def getOscFlags(self):
        self.oscBuyFlags = {}
        self.oscSellFlags = {}
        indicatorList = list(self.osc_flags.keys())
        for ind in indicatorList:
            self.oscBuyFlags[ind] = (self.buy_flag.value & self.osc_flags[ind]) > 0
            self.oscSellFlags[ind] = (self.sell_flag.value & self.osc_flags[ind]) > 0

    def calculateDivergences(self, priceSource: Series, osc: Series, phFound: Series, plFound: Series):
        mpp = self.maxpp.value
        thresholdDiff = 1
        
        oscL = osc.loc[plFound]
        oscH = osc.loc[phFound]
        priceSourceL = priceSource.loc[plFound]
        priceSourceH = priceSource.loc[phFound]

        oscHL = pd.DataFrame()
        priceLL = pd.DataFrame()
        oscLL = pd.DataFrame()
        priceHL = pd.DataFrame()
        oscLH = pd.DataFrame()
        priceHH = pd.DataFrame()
        oscHH = pd.DataFrame()
        priceLH = pd.DataFrame()

        for ppidx in range(0, mpp):
            oscLDiff = oscL.diff(ppidx + 1)
            oscLDiff.where(oscLDiff.abs() < thresholdDiff, 0, inplace=True)
            
            priceLDiff = priceSourceL.diff(ppidx + 1)
            priceLDiff.where(priceLDiff.abs() < thresholdDiff, 0, inplace=True)
            
            oscHDiff = oscH.diff(ppidx + 1).clip(lower = 1)
            oscHDiff.where(oscHDiff.abs() < thresholdDiff, 0, inplace=True)
            
            priceHDiff = priceSourceH.diff(ppidx + 1)
            priceHDiff.where(priceHDiff.abs() < thresholdDiff, 0, inplace=True)
            
            # ------------------------------------------------------------------------------
            #   Regular Bullish
            # ------------------------------------------------------------------------------
            # Osc: Higher Low
            oscHL[ppidx] = (oscLDiff > 0)
            # Price: Lower Low
            priceLL[ppidx] = (priceLDiff < 0)

            # ------------------------------------------------------------------------------
            #   Hidden Bullish
            # ------------------------------------------------------------------------------
            # Osc: Lower Low
            oscLL[ppidx] = (oscLDiff < 0)
            # Price: Higher Low
            priceHL[ppidx] = (priceLDiff > 0)

            # ------------------------------------------------------------------------------
            #   Regular Bearish
            # ------------------------------------------------------------------------------
            # Osc: Lower High
            oscLH[ppidx] = (oscHDiff < 0)
            # Price: Higher High
            priceHH[ppidx] = (priceHDiff > 0)

            # ------------------------------------------------------------------------------
            #   Hidden Bearish
            # ------------------------------------------------------------------------------
            # Osc: Higher High
            oscHH[ppidx] = (oscHDiff > 0)
            # Price: Lower High
            priceLH[ppidx] = (priceHDiff < 0)

        bullCond = (priceLL & oscHL).apply(np.any, axis=1) & plFound
        hiddenBullCond = (priceHL & oscLL).apply(np.any, axis=1) & plFound
        bearCond = (priceHH & oscLH).apply(np.any, axis=1) & phFound
        hiddenBearCond = (priceLH & oscHH).apply(np.any, axis=1) & phFound

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
