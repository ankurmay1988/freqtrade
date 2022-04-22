# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

from os import rename
from customindicators.DMI import DMI
from freqtrade.enums.runmode import RunMode
from customindicators import ElderSafeZone, WaveTrendOscillator
import freqtrade.vendor.qtpylib.indicators as qtpylib
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
# This class is a sample. Feel free to customize it.


class DMIStrat(IStrategy):
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
    max_profit = IntParameter(default=3, low=1, high=20, space='sell', load=True, optimize=True)

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    timeframe_above = '30m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

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
    
        adx, plusDI, minusDI = DMI(dataframe['high'], dataframe['low'], dataframe['close'])
        
        adx_inc = (adx.diff() > 0)
        # plusDI_inc = (plusDI.diff() > 0)
        # minusDI_dec = (minusDI.diff() < 0)
        buy_condition = (adx_inc & adx_inc.shift(1)) & pta.cross(plusDI, minusDI, True, False)
        buy_signal = buy_condition & (minusDI < 20)
        # buy_condition = adx_inc & plusDI_inc & minusDI_dec & (plusDI > minusDI) & ((minusDI - plusDI) < 10)
        # buy_signal = buy_condition & buy_condition.shift(1)
        
        macd_df = dataframe.ta.macd()
        macd, deltamacd, signalmacd = macd_df.iloc[:, 0].round(2), macd_df.iloc[:, 1].round(2), macd_df.iloc[:, 2].round(2)
        sma50 = dataframe.ta.sma(50).round(2)
        price_above_sma50 = dataframe['close'] > sma50
        dataframe = pd.concat([ dataframe,
                                adx.rename('adx'),
                                plusDI.rename('plusDI'),
                                minusDI.rename('minusDI'),
                                macd.rename('macd'),
                                deltamacd.rename('deltamacd'),
                                signalmacd.rename('signalmacd'),
                                sma50.rename('sma50'),
                                price_above_sma50.rename('price_above_sma50'),
                                buy_signal.rename('buy_signal')
                            ], axis=1)
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        
        dataframe.loc[dataframe['buy_signal'], ['buy', 'buy_tag']] = (1, 'dmi')

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        
        # ema_crosses = qtpylib.crossed_below(dataframe[f'ema_short_{self.ema_len_short.value}'], dataframe[f'ema_long_{self.ema_len_long.value}'])
        # sellcondition = ema_crosses
        
        # dataframe.loc[sellcondition, 'sell'] = 1
        
        return dataframe
    
    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        profit_pct = current_profit * 100
        if profit_pct > self.max_profit.value:
            return 'profit_pct'

        return None
    
    # use_custom_stoploss = True
    # def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs):
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    #     last_candle = dataframe.iloc[-1].squeeze()
    #     profit_pct = current_profit * 100
    #     # Use parabolic sar as absolute stoploss price
    #     # stoploss_price = last_candle['psar_l']
    #     stoploss_price = last_candle["elderSafezoneLong"]
    #     # Convert absolute price to percentage relative to current_rate
    #     if stoploss_price < current_rate:
    #         return (stoploss_price / current_rate) - 1
        
    #     # if profit_pct > 4:
    #     #     return 0.01
        
    #     # return maximum stoploss value, keeping current stoploss price unchanged
    #     return 1

    class HyperOpt:
        def generate_estimator():
            return "ET"

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
