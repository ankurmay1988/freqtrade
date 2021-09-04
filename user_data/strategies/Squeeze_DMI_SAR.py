# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
from numpy.core.numeric import cross  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from pandas.core.series import Series
from pandas_ta.utils import data
from ta.volatility import keltner_channel_lband

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, RealParameter

# --------------------------------
# Add your lib to import here
import ta.wrapper as taw
import pandas_ta as pta
from talib import abstract as tal
import freqtrade.vendor.qtpylib.indicators as qtpylib
import plotly.graph_objects as go

# This class is a sample. Feel free to customize it.
class SQM_DMI_SAR(IStrategy):
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

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 20

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
        "main_plot": {
            "psar": {
                "color": "#aa00e5",
                "type": "scatter"
            }
        },
        "subplots": {
            "DMI": {
                "adx": {
                    "color": "yellow",
                    "type": "line"
                },
                "di+": {
                    "color": "green",
                    "type": "line"
                },
                "di-": {
                    "color": "red",
                    "type": "line"
                }
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
        informative_pairs = [(pair, '30m') for pair in pairs]

        # add other pairs if needed maybe BTC ?
        # informative_pairs += [
        #     ("BTC/USDT", "30m")
        # ]

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

        dataframe['hlc3'] = dataframe.ta.hlc3()

        # PSAR
        psar = dataframe.ta.psar()
        dataframe['psar_long'] = psar['PSARl_0.02_0.2'].fillna(0)
        dataframe['psar_short'] = psar['PSARs_0.02_0.2'].fillna(0)
        dataframe['psar'] = np.where(dataframe['psar_long'] > 0, dataframe['psar_long'], dataframe['psar_short'])

        # Squeeze Momentum
        squeeze = dataframe.ta.squeeze(asint=False)
        dataframe['sqm_val'], dataframe['sqm_sqOn'], dataframe['sqm_sqOff'] = squeeze['SQZ_20_2.0_20_1.5'], squeeze['SQZ_ON'], squeeze['SQZ_OFF']

        # DMI
        adx = dataframe.ta.adx(length=14, lensig=14)
        dataframe["adx"], dataframe["di+"], dataframe["di-"] = adx['ADX_14'], adx['DMP_14'], adx['DMN_14']

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
            df1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='30m')
            df1h['hlc3'] = df1h.ta.hlc3()
            df1h['wt1'], df1h['wt2'], df1h['vwap'] = self.WaveTrendIndicator(df1h)
            squeeze = df1h.ta.squeeze(asint=False)
            df1h['sqm_val'], df1h['sqm_sqOn'], df1h['sqm_sqOff'] = squeeze['SQZ_20_2.0_20_1.5'], squeeze['SQZ_ON'], squeeze['SQZ_OFF']

            dataframe = merge_informative_pair(dataframe, df1h, self.timeframe, '30m', ffill=True)

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        dataframe['ha_green'] = dataframe['ha_open'] < dataframe['ha_close']
        dataframe['ha_opencandle'] = (dataframe['ha_open'] == dataframe['ha_high']) | (dataframe['ha_open'] == dataframe['ha_low'])
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        hashi_green_open_candle = dataframe['ha_green'] & (dataframe['ha_opencandle'] | dataframe['ha_opencandle'].shift(1))

        dataframe.loc[
            (
                (dataframe['di+'] > 14 & (dataframe['di+'] > dataframe['di-'])) \
                    & dataframe['psar_long'] > 0 \
                          & hashi_green_open_candle \
                                 & (dataframe['adx'] > 24)
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
        hashi_red_candle = (~dataframe['ha_green']) & (~dataframe['ha_green']).shift(1) # & dataframe['ha_opencandle']

        dataframe.loc[
            (
                hashi_red_candle
            ),
            'sell'] = 1
        return dataframe

    def WaveTrendIndicator(self, df: DataFrame):
        ap = pd.Series(df.ta.hlc3())
        esa = pta.ema(ap, self.wt_channelLength)
        d = pta.ema((ap-esa).abs(), self.wt_channelLength)
        ci = (ap - esa) / (0.015 * d)
        tci = pta.ema(ci, self.wt_avgLength)

        wt1 = tci
        wt2 = pta.sma(tci, self.wt_ma_length)
        vwap = wt1 - wt2
        return ( wt1, wt2, vwap )