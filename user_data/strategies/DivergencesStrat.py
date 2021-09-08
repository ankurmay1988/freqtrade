# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

from datetime import timedelta
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter, BooleanParameter

# --------------------------------
# Add your lib to import here
import pandas_ta as pta
import customindicators as ci
# This class is a sample. Feel free to customize it.


class DivStrat(IStrategy):
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
    prd = IntParameter(default=5, low=1, high=50)
    # Source for Pivot Points
    source = CategoricalParameter(default="close", categories=["close", "high/low"])
    # Maximum Pivot Points to Check
    maxpp = IntParameter(default=10, low=1, high=20)

    # Buy osc signal flags
    buy_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                            space='buy', load=True, optimize=True)
    # Sell osc signal flags
    sell_flag = IntParameter(default=maxflagnum, low=0, high=maxflagnum,
                             space='sell', load=True, optimize=True)

    buy_minsignals = IntParameter(default=1, low=0, high=len(osc_flags),
                                  space='buy', load=True, optimize=False)

    sell_minsignals = IntParameter(default=1, low=0, high=len(osc_flags),
                                   space='sell', load=True, optimize=False)

    buy_useBTC = BooleanParameter(default=False, space='buy', load=False, optimize=False)

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
        informative_pairs += [
            ("BTC/USDT", self.timeframe)
        ]

        return informative_pairs
        # return []

    def calc_indicators(self, dataframe: DataFrame) -> DataFrame:
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
        dataframe['ema200'] = dataframe.ta.ema(200)

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

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

        if not self.dp is None:
            df1h = self.dp.get_pair_dataframe("ETH/USDT", self.timeframe_above)
            df1h = self.calc_indicators(df1h)
            dataframe = merge_informative_pair(
                dataframe, df1h, self.timeframe, self.timeframe_above, ffill=True)

        if (not self.dp is None) & (self.buy_useBTC.value):
            dfbtc = self.dp.get_pair_dataframe("BTC/USDT", self.timeframe)
            dfbtc = self.calc_indicators(dfbtc)
            indicatorList = list(self.osc_flags.keys())
            for ind in indicatorList:
                dataframe['{}_rbull_btc'.format(ind)] = dfbtc['{}_rbull'.format(ind)]
                dataframe['{}_hbull_btc'.format(ind)] = dfbtc['{}_hbull'.format(ind)]
                dataframe['{}_rbear_btc'.format(ind)] = dfbtc['{}_rbear'.format(ind)]
                dataframe['{}_hbear_btc'.format(ind)] = dfbtc['{}_hbear'.format(ind)]

        # # Graph object to plot and analyze divergence data
        graphData = DataFrame()
        # if self.dp.runmode.value not in ["live", "dry_run"]:
        #     dataframe.to_csv('/disk/freqtrade/data.csv')
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        self.getOscFlags()

        price_above_200ema = dataframe['close'] > dataframe['ema200']

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

        if self.buy_minsignals.value > 0:
            buycondition = cond & (n_signals > self.buy_minsignals.value)
        else:
            buycondition = cond

        # When 1h chart show regular bullish divergence
        n_signals_1h = np.where(dataframe['cci_rbull_{}'.format(self.timeframe_above)] & self.calccci, 1, 0) + \
            np.where(dataframe['cmf_rbull_{}'.format(self.timeframe_above)] & self.calccmf, 1, 0) + \
            np.where(dataframe['macd_rbull_{}'.format(self.timeframe_above)] & self.calcmacd, 1, 0) + \
            np.where(dataframe['mfi_rbull_{}'.format(self.timeframe_above)] & self.calcmfi, 1, 0) + \
            np.where(dataframe['mom_rbull_{}'.format(self.timeframe_above)] & self.calcmom, 1, 0) + \
            np.where(dataframe['obv_rbull_{}'.format(self.timeframe_above)] & self.calcobv, 1, 0) + \
            np.where(dataframe['rsi_rbull_{}'.format(self.timeframe_above)] & self.calcrsi, 1, 0) + \
            np.where(dataframe['stk_rbull_{}'.format(self.timeframe_above)] & self.calcstoc, 1, 0) + \
            np.where(dataframe['uo_rbull_{}'.format(self.timeframe_above)] & self.calcuo, 1, 0)

        cond_reg_1h = (dataframe['cci_rbull_{}'.format(self.timeframe_above)] & self.calccci) \
            | (dataframe['cmf_rbull_{}'.format(self.timeframe_above)] & self.calccmf) \
            | (dataframe['macd_rbull_{}'.format(self.timeframe_above)] & self.calcmacd) \
            | (dataframe['mfi_rbull_{}'.format(self.timeframe_above)] & self.calcmfi) \
            | (dataframe['mom_rbull_{}'.format(self.timeframe_above)] & self.calcmom) \
            | (dataframe['obv_rbull_{}'.format(self.timeframe_above)] & self.calcobv) \
            | (dataframe['rsi_rbull_{}'.format(self.timeframe_above)] & self.calcrsi) \
            | (dataframe['stk_rbull_{}'.format(self.timeframe_above)] & self.calcstoc) \
            | (dataframe['uo_rbull_{}'.format(self.timeframe_above)] & self.calcuo)

        if self.buy_minsignals.value > 0:
            buycondition = buycondition | (cond_reg_1h & (n_signals_1h > self.buy_minsignals.value))
        else:
            buycondition = buycondition | cond_reg_1h

        # When 1h chart show hidden bullish divergence and price is above the 200 ema
        # that means trend will continue to be bullish so we can buy at this time also.
        n_signals_1h = np.where(dataframe['cci_hbull_{}'.format(self.timeframe_above)] & self.calccci, 1, 0) + \
            np.where(dataframe['cmf_hbull_{}'.format(self.timeframe_above)] & self.calccmf, 1, 0) + \
            np.where(dataframe['macd_hbull_{}'.format(self.timeframe_above)] & self.calcmacd, 1, 0) + \
            np.where(dataframe['mfi_hbull_{}'.format(self.timeframe_above)] & self.calcmfi, 1, 0) + \
            np.where(dataframe['mom_hbull_{}'.format(self.timeframe_above)] & self.calcmom, 1, 0) + \
            np.where(dataframe['obv_hbull_{}'.format(self.timeframe_above)] & self.calcobv, 1, 0) + \
            np.where(dataframe['rsi_hbull_{}'.format(self.timeframe_above)] & self.calcrsi, 1, 0) + \
            np.where(dataframe['stk_hbull_{}'.format(self.timeframe_above)] & self.calcstoc, 1, 0) + \
            np.where(dataframe['uo_hbull_{}'.format(self.timeframe_above)] & self.calcuo, 1, 0)

        cond_1h = (dataframe['cci_hbull_{}'.format(self.timeframe_above)] & self.calccci) \
            | (dataframe['cmf_hbull_{}'.format(self.timeframe_above)] & self.calccmf) \
            | (dataframe['macd_hbull_{}'.format(self.timeframe_above)] & self.calcmacd) \
            | (dataframe['mfi_hbull_{}'.format(self.timeframe_above)] & self.calcmfi) \
            | (dataframe['mom_hbull_{}'.format(self.timeframe_above)] & self.calcmom) \
            | (dataframe['obv_hbull_{}'.format(self.timeframe_above)] & self.calcobv) \
            | (dataframe['rsi_hbull_{}'.format(self.timeframe_above)] & self.calcrsi) \
            | (dataframe['stk_hbull_{}'.format(self.timeframe_above)] & self.calcstoc) \
            | (dataframe['uo_hbull_{}'.format(self.timeframe_above)] & self.calcuo)

        if self.buy_minsignals.value > 0:
            buycondition |= ((cond_1h & price_above_200ema) & (
                n_signals_1h > self.buy_minsignals.value))
        else:
            buycondition |= (cond_1h & price_above_200ema)

        # If Bitcoin also shows bullish divergence then also we can buy
        if self.buy_useBTC.value:
            cond_btc = (dataframe['cci_rbull_btc'] & self.calccci) \
                | (dataframe['cmf_rbull_btc'] & self.calccmf) \
                | (dataframe['macd_rbull_btc'] & self.calcmacd) \
                | (dataframe['mfi_rbull_btc'] & self.calcmfi) \
                | (dataframe['mom_rbull_btc'] & self.calcmom) \
                | (dataframe['obv_rbull_btc'] & self.calcobv) \
                | (dataframe['rsi_rbull_btc'] & self.calcrsi) \
                | (dataframe['stk_rbull_btc'] & self.calcstoc) \
                | (dataframe['uo_rbull_btc'] & self.calcuo)

            buycondition |= cond_btc

        dataframe.loc[
            (
                buycondition.shift(1).fillna(False)
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

        # Using 1h timeframe is it supposed to be more accurate
        n_signals = np.where(dataframe['cci_rbear'] & self.calccci, 1, 0) + \
            np.where(dataframe['cmf_rbear'] & self.calccmf, 1, 0) + \
            np.where(dataframe['macd_rbear'] & self.calcmacd, 1, 0) + \
            np.where(dataframe['mfi_rbear'] & self.calcmfi, 1, 0) + \
            np.where(dataframe['mom_rbear'] & self.calcmom, 1, 0) + \
            np.where(dataframe['obv_rbear'] & self.calcobv, 1, 0) + \
            np.where(dataframe['rsi_rbear'] & self.calcrsi, 1, 0) + \
            np.where(dataframe['stk_rbear'] & self.calcstoc, 1, 0) + \
            np.where(dataframe['uo_rbear'] & self.calcuo, 1, 0)

        cond = (dataframe['cci_rbear'] & self.sell_calccci) \
            | (dataframe['cmf_rbear'] & self.sell_calccmf) \
            | (dataframe['macd_rbear'] & self.sell_calcmacd) \
            | (dataframe['mfi_rbear'] & self.sell_calcmfi) \
            | (dataframe['mom_rbear'] & self.sell_calcmom) \
            | (dataframe['obv_rbear'] & self.sell_calcobv) \
            | (dataframe['rsi_rbear'] & self.sell_calcrsi) \
            | (dataframe['stk_rbear'] & self.sell_calcstoc) \
            | (dataframe['uo_rbear'] & self.sell_calcuo)

        if self.sell_minsignals.value > 0:
            sellcondition = cond & (n_signals > self.sell_minsignals.value)
        else:
            sellcondition = cond

        # Using 1h timeframe is it supposed to be more accurate
        n_signals = np.where(dataframe['cci_rbear_{}'.format(self.timeframe_above)] & self.calccci, 1, 0) + \
            np.where(dataframe['cmf_rbear_{}'.format(self.timeframe_above)] & self.calccmf, 1, 0) + \
            np.where(dataframe['macd_rbear_{}'.format(self.timeframe_above)] & self.calcmacd, 1, 0) + \
            np.where(dataframe['mfi_rbear_{}'.format(self.timeframe_above)] & self.calcmfi, 1, 0) + \
            np.where(dataframe['mom_rbear_{}'.format(self.timeframe_above)] & self.calcmom, 1, 0) + \
            np.where(dataframe['obv_rbear_{}'.format(self.timeframe_above)] & self.calcobv, 1, 0) + \
            np.where(dataframe['rsi_rbear_{}'.format(self.timeframe_above)] & self.calcrsi, 1, 0) + \
            np.where(dataframe['stk_rbear_{}'.format(self.timeframe_above)] & self.calcstoc, 1, 0) + \
            np.where(dataframe['uo_rbear_{}'.format(self.timeframe_above)] & self.calcuo, 1, 0)

        cond = (dataframe['cci_rbear_{}'.format(self.timeframe_above)] & self.sell_calccci) \
            | (dataframe['cmf_rbear_{}'.format(self.timeframe_above)] & self.sell_calccmf) \
            | (dataframe['macd_rbear_{}'.format(self.timeframe_above)] & self.sell_calcmacd) \
            | (dataframe['mfi_rbear_{}'.format(self.timeframe_above)] & self.sell_calcmfi) \
            | (dataframe['mom_rbear_{}'.format(self.timeframe_above)] & self.sell_calcmom) \
            | (dataframe['obv_rbear_{}'.format(self.timeframe_above)] & self.sell_calcobv) \
            | (dataframe['rsi_rbear_{}'.format(self.timeframe_above)] & self.sell_calcrsi) \
            | (dataframe['stk_rbear_{}'.format(self.timeframe_above)] & self.sell_calcstoc) \
            | (dataframe['uo_rbear_{}'.format(self.timeframe_above)] & self.sell_calcuo)

        if self.sell_minsignals.value > 0:
            sellcondition |= cond & (n_signals > self.sell_minsignals.value)
        else:
            sellcondition |= cond

        n_signals = np.where(dataframe['cci_hbear'] & self.calccci, 1, 0) + \
            np.where(dataframe['cmf_hbear'] & self.calccmf, 1, 0) + \
            np.where(dataframe['macd_hbear'] & self.calcmacd, 1, 0) + \
            np.where(dataframe['mfi_hbear'] & self.calcmfi, 1, 0) + \
            np.where(dataframe['mom_hbear'] & self.calcmom, 1, 0) + \
            np.where(dataframe['obv_hbear'] & self.calcobv, 1, 0) + \
            np.where(dataframe['rsi_hbear'] & self.calcrsi, 1, 0) + \
            np.where(dataframe['stk_hbear'] & self.calcstoc, 1, 0) + \
            np.where(dataframe['uo_hbear'] & self.calcuo, 1, 0)

        cond = (dataframe['cci_hbear'] & self.sell_calccci) \
            | (dataframe['cmf_hbear'] & self.sell_calccmf) \
            | (dataframe['macd_hbear'] & self.sell_calcmacd) \
            | (dataframe['mfi_hbear'] & self.sell_calcmfi) \
            | (dataframe['mom_hbear'] & self.sell_calcmom) \
            | (dataframe['obv_hbear'] & self.sell_calcobv) \
            | (dataframe['rsi_hbear'] & self.sell_calcrsi) \
            | (dataframe['stk_hbear'] & self.sell_calcstoc) \
            | (dataframe['uo_hbear'] & self.sell_calcuo)

        if self.sell_minsignals.value > 0:
            sellcondition |= cond & (n_signals > self.sell_minsignals.value)
        else:
            sellcondition |= cond

        dataframe.loc[
            (
                sellcondition.shift(1).fillna(False)
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

    def calculateDivergences(self, priceSource: Series, osc: Series, phFound: Series, plFound: Series):
        mpp = self.maxpp.value

        oscL = osc.loc[plFound]
        oscH = osc.loc[phFound]
        priceSourceL = priceSource.loc[plFound]
        priceSourceH = priceSource.loc[phFound]

        # ------------------------------------------------------------------------------
        # Regular Bullish
        # Osc: Higher Low
        oscHL = pd.DataFrame()
        priceLL = pd.DataFrame()
        for ppidx in range(0, mpp):
            oscHL[ppidx] = (oscL.diff(ppidx + 1) > 0)

        # Price: Lower Low
        for ppidx in range(0, mpp):
            priceLL[ppidx] = (priceSourceL.diff(ppidx + 1) < 0)

        bullCond = (priceLL & oscHL).apply(np.any, axis=1) & plFound

        del oscHL
        del priceLL

        # ------------------------------------------------------------------------------
        # Hidden Bullish
        # Osc: Lower Low
        oscLL = pd.DataFrame()
        priceHL = pd.DataFrame()
        for ppidx in range(0, mpp):
            oscLL[ppidx] = (oscL.diff(ppidx + 1) < 0)

        # Price: Higher Low
        for ppidx in range(0, mpp):
            priceHL[ppidx] = (priceSourceL.diff(ppidx + 1) > 0)

        hiddenBullCond = (priceHL & oscLL).apply(np.any, axis=1) & plFound

        del oscLL
        del priceHL

        # ------------------------------------------------------------------------------
        # Regular Bearish
        # Osc: Lower High
        oscLH = pd.DataFrame()
        priceHH = pd.DataFrame()

        for ppidx in range(0, mpp):
            oscLH[ppidx] = (oscH.diff(ppidx + 1) < 0)

        # Price: Higher High
        for ppidx in range(0, mpp):
            priceHH[ppidx] = (priceSourceH.diff(ppidx + 1) > 0)

        bearCond = (priceHH & oscLH).apply(np.any, axis=1) & phFound

        del oscLH
        del priceHH

        # ------------------------------------------------------------------------------
        # Hidden Bearish
        # Osc: Higher High

        oscHH = pd.DataFrame()
        priceLH = pd.DataFrame()
        for ppidx in range(0, mpp):
            oscHH[ppidx] = (oscH.diff(ppidx + 1) > 0)

        # Price: Lower High
        for ppidx in range(0, mpp):
            priceLH[ppidx] = (priceSourceH.diff(ppidx + 1) < 0)

        hiddenBearCond = (priceLH & oscHH).apply(np.any, axis=1) & phFound

        del oscHH
        del priceLH

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
