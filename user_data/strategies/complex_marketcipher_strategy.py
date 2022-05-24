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
import freqtrade.vendor.qtpylib.indicators as qtpylib


# This class is a sample. Feel free to customize it.
class MC2Strategy(IStrategy):
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
    wt_channelLength = 9
    wt_avgLength = 12
    wt_ma_length = 3
    mfi_period = 60
    mfi_multiplier = 150
    mfi_ypos = 2.5
    
    wt_overbought_l1 = RealParameter(low=50, high=99, default=53, space='sell', optimize=True, load=True)
    wt_overbought_l2 = 60
    wt_overbought_l3 = 100
    wt_oversold_l1 = RealParameter(low=-99, high=-50, default=-53, space='buy', optimize=True, load=True)
    wt_oversold_l2 = -60
    wt_oversold_l3 = -75
    use_mf = CategoricalParameter(categories=[True, False], default=False, space='buy', optimize=True, load=True)
    mfi_level = RealParameter(low=0, high=50, default=0, space='buy', optimize=True, load=True)
    
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
        
        # MFI
        mfi = ta.SMAIndicator(((dataframe['close'] - dataframe['open']) / (dataframe['high'] - dataframe['low'])) * self.mfi_multiplier, self.mfi_period)
        dataframe['mfi'] = mfi.sma_indicator() - self.mfi_ypos
        
        # EMA - Exponential Moving Average
        ema = ta.EMAIndicator(dataframe.close, 200)
        dataframe['ema200'] = ema.ema_indicator()

        # MC WaveTrend oscillator
        ap = dataframe['high'] + dataframe['low'] + dataframe['close'] / 3.0
        esa = ta.EMAIndicator(ap, self.wt_channelLength).ema_indicator()
        d = ta.EMAIndicator((ap-esa).abs(), self.wt_channelLength).ema_indicator()
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMAIndicator(ci, self.wt_avgLength).ema_indicator()
        
        dataframe['wt1'] = tci
        dataframe['wt2'] = ta.SMAIndicator(tci, self.wt_ma_length).sma_indicator()
        
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
        
        crosses =  qtpylib.crossed_above(dataframe['wt1'], dataframe['wt2'])
        check_mfi = np.full(dataframe.shape[0], True)
        if self.use_mf.value is True:
            check_mfi = dataframe['mfi'] > self.mfi_level.value
            
        bearishTrend = dataframe['close'] < dataframe['ema200']
        bullishTrend = dataframe['close'] > dataframe['ema200']
                
        dataframe.loc[
            (
                # Signal: WaveTrend crosses and wt1 > wt2 and is below oversold level
                # crosses & (dataframe['wt2'] <= self.wt_oversold_l1.value) & check_mfi
                crosses & (dataframe['wt2'] <= self.wt_oversold_l1.value) & check_mfi
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
                crosses & (dataframe['wt2'] >= self.wt_overbought_l1.value)
            ),
            'sell'] = 1
        return dataframe
