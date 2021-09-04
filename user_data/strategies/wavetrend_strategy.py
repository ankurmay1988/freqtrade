# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter

# --------------------------------
# Add your lib to import here
import ta.wrapper as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class WavetrendStrategy(IStrategy):
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

    # Buy hyperspace params:
    buy_params = {
        "wt_oversold_l1": -72.1342,
        "wt_oversold_l2": -51.46166,
    }

    # Sell hyperspace params:
    sell_params = {
        "wt_overbought_l1": 78.78404,
        "wt_overbought_l2": 84.88377,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.124

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
    wt_channelLength = IntParameter(low=9, high=25, default=10, optimize=True, load=True)
    wt_avgLength = IntParameter(low=18, high=25, default=21, optimize=True, load=True)
    wt_overbought_l1 = RealParameter(low=60, high=100, default=60, space='sell', optimize=True, load=True)
    wt_overbought_l2 = RealParameter(low=50, high=100, default=53, space='sell', optimize=True, load=True)
    wt_oversold_l1 = RealParameter(low=-100, high=-60, default=-60, space='buy', optimize=True, load=True)
    wt_oversold_l2 = RealParameter(low=-100, high=-50, default=-53, space='buy', optimize=True, load=True)
    
    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 3

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'ema200': { 'color': 'yellow' }
        },
        'subplots': {
            "WaveTrend": {
                'wt1': {'color': 'green'},
                'wt2': {'color': 'red'},
            }
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
        return []

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

        # MACD
        # macd = ta.MACD(dataframe.close, 26, 12, 9)
        # dataframe['macd'] = macd.macd()
        # dataframe['macdsignal'] = macd.macd_signal()

        # MFI
        mfi = ta.MFIIndicator(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])
        dataframe['mfi'] = mfi.money_flow_index()

        # EMA - Exponential Moving Average
        ema = ta.EMAIndicator(dataframe.close, 200)
        dataframe['ema200'] = ema.ema_indicator()

        # WaveTrend oscillator
        ap = dataframe['close']
        esa = ta.EMAIndicator(ap, self.wt_channelLength.value).ema_indicator()
        d = ta.EMAIndicator((ap-esa).abs(), self.wt_channelLength.value).ema_indicator()
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMAIndicator(ci, self.wt_avgLength.value).ema_indicator()
        
        dataframe['wt1'] = tci
        dataframe['wt2'] = ta.SMAIndicator(tci, 4).sma_indicator()
        
        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        
        crosses = qtpylib.crossed_above(dataframe['wt1'], dataframe['wt2'])
        
        dataframe.loc[
            (
                # Signal: WaveTrend crosses and wt1 > wt2 and is below oversold level
                crosses & (dataframe['wt2'] <= self.wt_oversold_l2.value)
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
        
        crosses = qtpylib.crossed_below(dataframe['wt1'], dataframe['wt2'])
        
        dataframe.loc[
            (
                # Signal: WaveTrend crosses and wt1 < wt2
                crosses & (dataframe['wt2'] >= self.wt_overbought_l2.value)
            ),
            'sell'] = 1
        return dataframe
