# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter

# --------------------------------
# Add your lib to import here
import ta.wrapper as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import plotly.graph_objects as go

# This class is a sample. Feel free to customize it.
class MC3Strategy(IStrategy):
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

    wt_overbought_l1 = IntParameter(low=25, high=75, default=75, space='sell', optimize=True, load=True)
    wt_overbought_l2 = 60
    wt_overbought_l3 = 100
    wt_oversold_l1 = IntParameter(low=-100, high=-25, default=-90, space='buy', optimize=True, load=True)
    wt_oversold_l2 = -60
    wt_oversold_l3 = -75
    # mfi_level = IntParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    vwap_level = IntParameter(low=-25, high=25, default=5, space='buy', optimize=True, load=True)

    # Optimal timeframe for the strategy.
    timeframe = '30m'

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
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {

        },
        'subplots': {
            "WaveTrend": {
                'wt1': { 'color': '#90caf9', 'plotly': { 'fill': 'tozeroy', 'opacity': 0.9}},
                'wt2': { 'color': '#0d47a1', 'plotly': { 'fill': 'tozeroy', 'opacity': 0.2}},
                'buy_pos': { 'type': 'scatter' , 'color': 'lime', 'plotly': { 'marker' : { 'size': 5  } }},
                'sell_pos': { 'type': 'scatter' , 'color': 'red', 'plotly': { 'marker' : { 'size': 5  } }},

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
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        # add other pairs if needed maybe BTC ?
        # informative_pairs += [
        #     ("BTC/USDT", "30m")
        # ]
        return informative_pairs

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

        dataframe['hlc3'] = dataframe['high'] + dataframe['low'] + dataframe['close'] / 3.0

        # MFI
        dataframe['mfi'] = self.MCMoneyFlowIndex(dataframe, self.mfi_multiplier, self.mfi_period)

        # MC WaveTrend oscillator
        dataframe['wt1'], dataframe['wt2'], dataframe['vwap'] = self.WaveTrendIndicator(dataframe)
        dataframe['crosses_above'] = qtpylib.crossed_above(dataframe['wt1'], dataframe['wt2'])
        dataframe['crosses_below'] = qtpylib.crossed_below(dataframe['wt1'], dataframe['wt2'])

        cross_signal = np.full(dataframe.shape[0], np.nan)
        cross_signal = np.where(dataframe['crosses_above'], -1, cross_signal)
        cross_signal = np.where(dataframe['crosses_below'], 1, cross_signal)
        dataframe['cross_signal'] = pd.Series(cross_signal).fillna(method='ffill')

        dataframe['crosses_below_obLevel'] = qtpylib.crossed_below(dataframe['wt1'], np.full(dataframe.shape[0], self.wt_overbought_l1.value))

        dataframe['buy_pos'] = dataframe['wt1'].loc[dataframe['crosses_above']]
        dataframe['sell_pos'] = dataframe['wt1'].loc[dataframe['crosses_below']]

        # =============================================
        #     Higher timeframe data calculation
        # =============================================
        if self.dp is not None:
            df1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            df1h['hlc3'] = df1h['high'] + df1h['low'] + df1h['close'] / 3.0

            df1h['wt1'], df1h['wt2'], df1h['vwap'] = self.WaveTrendIndicator(df1h)

            df1h['mfi'] = self.MCMoneyFlowIndex(df1h, self.mfi_multiplier, self.mfi_period)

            dataframe = merge_informative_pair(dataframe, df1h, self.timeframe, '1h', ffill=True)

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        money_flow_in_green = dataframe['mfi_1h'] > 0
        vwap_above0 = dataframe['vwap'] > 0
        vwap1h_above0 = dataframe['vwap_1h'] > 0
        # vwap_above0 = np.full(dataframe.shape[0], True)

        hashi_green_open_candle = (dataframe['ha_open'] < dataframe['ha_close']) & \
                                ((dataframe['ha_open'].shift(1) == dataframe['ha_low'].shift(1)) | (dataframe['ha_open'].shift(2) == dataframe['ha_low'].shift(2)))

        dataframe.loc[
            (
                 (dataframe['wt2'] < self.wt_oversold_l1.value) \
                     & (dataframe['cross_signal'] == -1) \
                         & vwap_above0
                         # & vwap1h_above0
                         # & hashi_green_open_candle
                         # & money_flow_in_green
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

        dataframe.loc[
            (
                # Signal: WaveTrend crosses overbought level 1
                (dataframe['wt1'] > 0) \
                    & (dataframe['vwap'] < self.vwap_level.value)
            ),
            'sell'] = 1
        return dataframe

    def WaveTrendIndicator(self, df: DataFrame):
        ap = df['hlc3']
        esa = ta.EMAIndicator(ap, self.wt_channelLength).ema_indicator()
        d = ta.EMAIndicator((ap-esa).abs(), self.wt_channelLength).ema_indicator()
        ci = (ap - esa) / (0.015 * d)
        tci = ta.EMAIndicator(ci, self.wt_avgLength).ema_indicator()

        wt1 = tci
        wt2 = ta.SMAIndicator(tci, self.wt_ma_length).sma_indicator()
        vwap = wt1 - wt2
        return ( wt1, wt2, vwap )

    def MCMoneyFlowIndex(self, df: DataFrame, multiplier: int, period: int):
        mfi = ta.SMAIndicator(((df['close'] - df['open']) / (df['high'] - df['low'])) * multiplier, period)
        return mfi.sma_indicator() - self.mfi_ypos