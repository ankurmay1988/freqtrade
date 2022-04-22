# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

# Support-Resistance-Trendline Breakout Strategy

from freqtrade.enums.runmode import RunMode
from customindicators import ElderSafeZone, WaveTrendOscillator
from datetime import datetime, timedelta
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series

from freqtrade.strategy import IStrategy, merge_informative_pair, informative
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter, BooleanParameter
from freqtrade.persistence import Trade

# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import customindicators as ci
from findiff import FinDiff
# This class is a sample. Feel free to customize it.


class SRTBreakout(IStrategy):
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

    # Hyperoptable parameters
    # Maximum Pivot Points to Check
    maxpp = IntParameter(default=3, low=1, high=10,
                         space='buy', load=True, optimize=True)

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

        # # add other pairs if needed maybe BTC ?
        # # informative_pairs += [("BTC/USDT", self.timeframe)]

        return informative_pairs
        # return []

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
        self.getOscFlags()
    
        hlc3 = pd.Series(dataframe.ta.hlc3())
        d_dx = FinDiff(0, 1, 1)
        d2_dx2 = FinDiff(0, 1, 2)
        pivots = hlc3.loc[d_dx(hlc3) == 0]
        dataframe = pd.concat([dataframe, 
            hlc3.rename('hlc3')
        ], axis=1)
        
        priceSource = dataframe['close'] if self.source.value == 'close' else dataframe['close']
        
        pp = ci.PivotPoint().pivotpoints(priceSource, 3, 1)
        ph = pd.Series(np.where(pp == 1, True, False))
        pl = pd.Series(np.where(pp == -1, True, False))

        dataframe = pd.concat([dataframe, ph.rename('ph'), pl.rename('pl')], axis=1)
        
        indicatorList = list(self.osc_flags.keys())
        for ind in indicatorList:
            for pp_count in range(self.maxpp.low, self.maxpp.range.stop):
                rbull, hbull, rbear, hbear = self.calculateDivergences(priceSource=priceSource, osc=dataframe[ind], phFound=ph, plFound=pl, pivotback=pp_count)
                rbull_col = f'{ind}_rbull_{pp_count}'
                hbull_col = f'{ind}_hbull_{pp_count}'
                rbear_col = f'{ind}_rbear_{pp_count}'
                hbear_col = f'{ind}_hbear_{pp_count}'
                dataframe = pd.concat([dataframe, rbull.rename(rbull_col), hbull.rename(hbull_col), rbear.rename(rbear_col), hbear.rename(hbear_col)], axis=1)

        uptrend = (sma50 > ema200)
        price_above_sma50 = (dataframe['close'] > sma50)
        price_above_ema200 = (dataframe['close'] > ema200)
        macd_up = deltamacd.round(2) > 0
        rsi_below_30 = (rsi < 30)
        signal_rsi_below_30 = rsi_below_30 | rsi_below_30.shift(1) | rsi_below_30.shift(2)
        rsi_crosses_70 = (rsi.shift(1) < 70) & (rsi > 70)
        stochrsi_oversold = (strsik < 5) | (strsid < 5) 
        signal_stochrsi_oversold = stochrsi_oversold | stochrsi_oversold.shift(1) | stochrsi_oversold.shift(2)
        stochrsi_overbought = (strsik > 80) | (strsid > 80)
        buy_rsi_stochrsi_macd = signal_rsi_below_30 & signal_stochrsi_oversold & (strsik > strsid)
        sell_rsi_stochrsi_macd = (strsik == 100) | (strsid == 100)

        dataframe = pd.concat([dataframe, 
            uptrend.rename('uptrend'),
            price_above_sma50.rename('price_above_sma50'),
            price_above_ema200.rename('price_above_ema200'),
            macd_up.rename('macd_up'),
            rsi_below_30.rename('rsi_below_30'),
            signal_rsi_below_30.rename('signal_rsi_below_30'),
            rsi_crosses_70.rename('rsi_crosses_70'),
            stochrsi_oversold.rename('stochrsi_oversold'),
            signal_stochrsi_oversold.rename('signal_stochrsi_oversold'),
            stochrsi_overbought.rename('stochrsi_overbought'),
            buy_rsi_stochrsi_macd.rename('buy_rsi_stochrsi_macd'),
            sell_rsi_stochrsi_macd.rename('sell_rsi_stochrsi_macd')
        ], axis=1)
        
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
        if self.dp.runmode.value == RunMode.BACKTEST.value:
            dataframe.to_csv('D:\\{}.csv'.format(metadata['pair'].replace('/', "_")))
        return dataframe.copy()
    
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict):
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # self.getOscFlags()

        # hlc3 = dataframe.ta.hlc3()
        # rsi = dataframe.ta.rsi(14).round(2)
        # macd_df = dataframe.ta.macd()
        # macd, deltamacd, signalmacd = macd_df.iloc[:, 0].round(2), macd_df.iloc[:, 1].round(2), macd_df.iloc[:, 2].round(2)
        # mom = dataframe.ta.mom(10).round(2)
        # cci = dataframe.ta.cci(10).round(2)
        # obv = dataframe.ta.obv().round(2)
        # stoch_df = dataframe.ta.stoch()
        # stk, std = stoch_df.iloc[:, 0].round(2), stoch_df.iloc[:, 1].round(2)
        
        # cmf = dataframe.ta.cmf().round(2)
        # mfi = dataframe.ta.mfi(14).round(2)
        # uo = dataframe.ta.uo().round(2)
        # ema200 = dataframe.ta.ema(200).round(2)
        # sma50 = dataframe.ta.sma(50).round(2)
        # stochrsi_df = dataframe.ta.stochrsi()
        # strsik, strsid =  stochrsi_df.iloc[:, 0].round(2), stochrsi_df.iloc[:, 1].round(2)
        # wt1, wt2 = WaveTrendOscillator(hlc3)
        # wt1 = wt1.round(2)
        # wt2 = wt2.round(2)
        
        # dataframe = pd.concat([dataframe, 
        #     hlc3.rename('hlc3'),
        #     rsi.rename('rsi'),
        #     macd.rename('macd'),
        #     deltamacd.rename('deltamacd'),
        #     signalmacd.rename('signalmacd'),
        #     mom.rename('mom'),
        #     cci.rename('cci'),
        #     obv.rename('obv'),
        #     stk.rename('stk'),
        #     std.rename('std'),
        #     cmf.rename('cmf'),
        #     mfi.rename('mfi'),
        #     uo.rename('uo'),
        #     ema200.rename('ema200'),
        #     sma50.rename('sma50'),
        #     strsik.rename('strsik'),
        #     strsid.rename('strsid'),
        #     wt1.rename('wt1'),
        #     wt2.rename('wt2'),
        # ], axis=1)
        
        # priceSource = dataframe['close'] if self.source.value == 'close' else dataframe['close']

        # pp = ci.PivotPoint().pivotpoints(priceSource, 3, 1)
        # ph = pd.Series(np.where(pp == 1, True, False))
        # pl = pd.Series(np.where(pp == -1, True, False))

        # dataframe = pd.concat([dataframe, ph.rename('ph'), pl.rename('pl')], axis=1)
        
        # indicatorList = list(self.osc_flags.keys())
        # for ind in indicatorList:
        #     for pp_count in range(self.maxpp.low, self.maxpp.range.stop):
        #         rbull, hbull, rbear, hbear = self.calculateDivergences(priceSource=priceSource, osc=dataframe[ind], phFound=ph, plFound=pl, pivotback=pp_count)
        #         rbull_col = f'{ind}_rbull_{pp_count}'
        #         hbull_col = f'{ind}_hbull_{pp_count}'
        #         rbear_col = f'{ind}_rbear_{pp_count}'
        #         hbear_col = f'{ind}_hbear_{pp_count}'
        #         dataframe = pd.concat([dataframe, rbull.rename(rbull_col), hbull.rename(hbull_col), rbear.rename(rbear_col), hbear.rename(hbear_col)], axis=1)

        # uptrend = (sma50 > ema200)
        # price_above_sma50 = (dataframe['close'] > sma50)
        # price_above_ema200 = (dataframe['close'] > ema200)
        # macd_up = deltamacd.round(2) > 0
        # rsi_below_30 = (rsi < 30)
        # signal_rsi_below_30 = rsi_below_30 | rsi_below_30.shift(1) | rsi_below_30.shift(2)
        # rsi_crosses_70 = (rsi.shift(1) < 70) & (rsi > 70)
        # stochrsi_oversold = (strsik < 5) | (strsid < 5) 
        # signal_stochrsi_oversold = stochrsi_oversold | stochrsi_oversold.shift(1) | stochrsi_oversold.shift(2)
        # stochrsi_overbought = (strsik > 80) | (strsid > 80)
        # buy_rsi_stochrsi_macd = signal_rsi_below_30 & signal_stochrsi_oversold & (strsik > strsid)
        # sell_rsi_stochrsi_macd = (strsik == 100) | (strsid == 100)

        # dataframe = pd.concat([dataframe, 
        #     uptrend.rename('uptrend'),
        #     price_above_sma50.rename('price_above_sma50'),
        #     price_above_ema200.rename('price_above_ema200'),
        #     macd_up.rename('macd_up'),
        #     rsi_below_30.rename('rsi_below_30'),
        #     signal_rsi_below_30.rename('signal_rsi_below_30'),
        #     rsi_crosses_70.rename('rsi_crosses_70'),
        #     stochrsi_oversold.rename('stochrsi_oversold'),
        #     signal_stochrsi_oversold.rename('signal_stochrsi_oversold'),
        #     stochrsi_overbought.rename('stochrsi_overbought'),
        #     buy_rsi_stochrsi_macd.rename('buy_rsi_stochrsi_macd'),
        #     sell_rsi_stochrsi_macd.rename('sell_rsi_stochrsi_macd')
        # ], axis=1)
        
        return dataframe.copy()

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
        
        n_signals = np.full(dataframe['close'].size, 0)
        cond = np.full(dataframe['close'].size, False)
        # ind_list = pd.Series(np.full(dataframe['close'].size, ''))
        for ind in filter(lambda ind: self.oscBuyFlags[ind], indicatorList):
            osc_condition = np.full(dataframe['close'].size, False)
            
            for mpp in range(1, self.maxpp.value + 1):
                osc_condition |= dataframe[f'{ind}_rbull_{mpp}']
                
            n_signals += np.where(osc_condition, 1, 0)
            # appended = ind_list.apply(lambda x: '{}_{}'.format(x, ind))
            # ind_list = ind_list.where(osc_condition, appended)
            cond = cond | osc_condition

        if self.buy_minsignals.value > 0:
            signal = cond & (n_signals > self.buy_minsignals.value)
        else:
            signal = cond

        dataframe.loc[signal, 'buy_tag'] = 'rbull_30m'
        
        buycondition = buycondition | signal
            
        n_signals = np.full(dataframe['close'].size, 0)
        cond = np.full(dataframe['close'].size, False)
        # ind_list = pd.Series(np.full(dataframe['close'].size, ''))
        for ind in filter(lambda ind: self.oscBuyFlags[ind], indicatorList):
            osc_condition = np.full(dataframe['close'].size, False)
            
            for mpp in range(1, self.maxpp.value + 1):
                osc_condition |= dataframe[f'{ind}_hbull_{mpp}']
                
            n_signals += np.where(osc_condition, 1, 0)
            # appended = ind_list.apply(lambda x: '{}_{}'.format(x, ind))
            # ind_list = ind_list.where(osc_condition, appended)
            cond = cond | osc_condition
            
        cond = cond & price_above_sma50

        if self.buy_minsignals.value > 0:
            signal = cond & (n_signals > self.buy_minsignals.value)
        else:
            signal = cond

        dataframe.loc[signal, 'buy_tag'] = 'hbull_30m'
        buycondition = buycondition | signal

        dataframe.loc[buycondition, 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        # self.getOscFlags()
        # indicatorList = list(self.osc_flags.keys())
        # sellcondition = pd.Series(np.full(dataframe['close'].size, False))
        # uptrend = dataframe['uptrend']
        # price_above_sma50 = dataframe['price_above_sma50']
        # price_above_ema200 = dataframe['price_above_ema200']

        # n_signals = np.full(dataframe['close'].size, 0)
        # cond = np.full(dataframe['close'].size, False)
        # # ind_list = pd.Series(np.full(dataframe['close'].size, ''))
        # for ind in filter(lambda ind: self.oscBuyFlags[ind], indicatorList):
        #     osc_condition = np.full(dataframe['close'].size, False)
            
        #     for mpp in range(1, self.maxpp.value + 1):
        #         osc_condition |= dataframe[f'{ind}_rbear_{mpp}']
                
        #     n_signals += np.where(osc_condition, 1, 0)
        #     # appended = ind_list.apply(lambda x: '{}_{}'.format(x, ind))
        #     # ind_list = ind_list.where(osc_condition, appended)
        #     cond = cond | osc_condition

        # # cond = cond & uptrend & price_above_ema200
        # if self.sell_minsignals.value > 0:
        #     signal = cond & (n_signals > self.sell_minsignals.value)
        # else:
        #     signal = cond
            
        # dataframe.loc[signal, 'sell_tag'] = 'rbear_30m'
        # sellcondition = sellcondition | signal
        
        # n_signals = np.full(dataframe['close'].size, 0)
        # cond = np.full(dataframe['close'].size, False)
        # # ind_list = pd.Series(np.full(dataframe['close'].size, ''))
        # for ind in filter(lambda ind: self.oscBuyFlags[ind], indicatorList):
        #     osc_condition = np.full(dataframe['close'].size, False)
            
        #     for mpp in range(1, self.maxpp.value + 1):
        #         osc_condition |= dataframe[f'{ind}_hbear_{mpp}']
                
        #     n_signals += np.where(osc_condition, 1, 0)
        #     # appended = ind_list.apply(lambda x: '{}_{}'.format(x, ind))
        #     # ind_list = ind_list.where(osc_condition, appended)
        #     cond = cond | osc_condition
        
        # if self.sell_minsignals.value > 0:
        #     signal = cond & (n_signals > self.sell_minsignals.value)
        # else:
        #     signal = cond
            
        # dataframe.loc[signal, 'sell_tag'] = 'hbear_30m'
        # sellcondition = sellcondition | signal

        # dataframe.loc[sellcondition, 'sell'] = 1
        
        return dataframe
    
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        profit_pct = current_profit * 100
        if profit_pct > 2:
            return 'sell_profit_reached'

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
        if stoploss_price < current_rate:
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

    def calculateDivergences(self, priceSource: Series, osc: Series, phFound: Series, plFound: Series, pivotback: int):
        thresholdDiff = 1
        
        oscL = osc.loc[plFound]
        oscH = osc.loc[phFound]
        priceSourceL = priceSource.loc[plFound]
        priceSourceH = priceSource.loc[phFound]

        oscLDiff = oscL.diff(pivotback)
        oscLDiff.where(oscLDiff.abs() < thresholdDiff, 0, inplace=True)
        
        priceLDiff = priceSourceL.diff(pivotback)
        priceLDiff.where(priceLDiff.abs() < thresholdDiff, 0, inplace=True)
        
        oscHDiff = oscH.diff(pivotback).clip(lower = 1)
        oscHDiff.where(oscHDiff.abs() < thresholdDiff, 0, inplace=True)
        
        priceHDiff = priceSourceH.diff(pivotback)
        priceHDiff.where(priceHDiff.abs() < thresholdDiff, 0, inplace=True)
        
        # ------------------------------------------------------------------------------
        #   Regular Bullish
        # ------------------------------------------------------------------------------
        # Osc: Higher Low
        oscHL = (oscLDiff > 0)
        # Price: Lower Low
        priceLL = (priceLDiff < 0)

        # ------------------------------------------------------------------------------
        #   Hidden Bullish
        # ------------------------------------------------------------------------------
        # Osc: Lower Low
        oscLL = (oscLDiff < 0)
        # Price: Higher Low
        priceHL = (priceLDiff > 0)

        # ------------------------------------------------------------------------------
        #   Regular Bearish
        # ------------------------------------------------------------------------------
        # Osc: Lower High
        oscLH = (oscHDiff < 0)
        # Price: Higher High
        priceHH = (priceHDiff > 0)

        # ------------------------------------------------------------------------------
        #   Hidden Bearish
        # ------------------------------------------------------------------------------
        # Osc: Higher High
        oscHH = (oscHDiff > 0)
        # Price: Lower High
        priceLH = (priceHDiff < 0)

        bullCond = (priceLL & oscHL).apply(np.any) & plFound
        hiddenBullCond = (priceHL & oscLL).apply(np.any) & plFound
        bearCond = (priceHH & oscLH).apply(np.any) & phFound
        hiddenBearCond = (priceLH & oscHH).apply(np.any) & phFound

        return (
            bullCond.fillna(value=False),
            hiddenBullCond.fillna(value=False),
            bearCond.fillna(value=False),
            hiddenBearCond.fillna(value=False)
        )
    # class HyperOpt:
    #     def generate_estimator():
    #         return "GBRT"

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
