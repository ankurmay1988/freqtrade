# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
from collections import namedtuple
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame  # noqa
from datetime import datetime  # noqa
from typing import Optional
from freqtrade.persistence.trade_model import Trade  # noqa

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import customindicators as ci
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TrendMeterStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    # The configuration would therefore mean:
    # Exit whenever 4% profit was reached
    # Exit when 2% profit was reached (in effect after 20 minutes)
    # Exit when 1% profit was reached (in effect after 30 minutes)
    # Exit when trade is non-loosing (in effect after 40 minutes)
    # minimal_roi = {
    #     "60": 0.01,
    #     "30": 0.02,
    #     "0": 0.04
    # }
    minimal_roi = {
        "0": 100
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.1
    use_custom_stoploss = True
    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Strategy parameters
    trendMeter1 = CategoricalParameter([
        "MACD Crossover - 12, 26, 9", 
        "MACD Crossover - Fast - 8, 21, 5", 
        "Mom Dad Cross (Top Dog Trading)", 
        "RSI Signal Line Cross - RSI 13, Sig 21", 
        "RSI 13: > or < 50", "RSI 5: > or < 50", 
        "Trend Candles", 
        "N/A"], default="MACD Crossover - Fast - 8, 21, 5", space="buy", optimize=True, load=True)
    
    trendMeter2 = CategoricalParameter([
        "MACD Crossover - 12, 26, 9", 
        "MACD Crossover - Fast - 8, 21, 5", 
        "Mom Dad Cross (Top Dog Trading)", 
        "RSI Signal Line Cross - RSI 13, Sig 21", 
        "RSI 13: > or < 50", "RSI 5: > or < 50", 
        "Trend Candles", 
        "N/A"], default="RSI 13: > or < 50", space="buy", optimize=True, load=True)
    
    trendMeter3 = CategoricalParameter([
        "MACD Crossover - 12, 26, 9", 
        "MACD Crossover - Fast - 8, 21, 5", 
        "Mom Dad Cross (Top Dog Trading)", 
        "RSI Signal Line Cross - RSI 13, Sig 21", 
        "RSI 13: > or < 50", "RSI 5: > or < 50", 
        "Trend Candles", 
        "N/A"], default="RSI 5: > or < 50", space="buy", optimize=True, load=True)
    
    trendBar1 = CategoricalParameter([
        "MA Crossover", 
        "MA Direction - Fast MA - TB1", 
        "MA Direction - Slow MA - TB1", 
        "N/A"], default="MA Crossover", space="buy", optimize=True, load=True)
    
    trendBar2 = CategoricalParameter([
        "MA Crossover", 
        "MA Direction - Fast MA - TB1", 
        "MA Direction - Slow MA - TB1", 
        "N/A"], default="MA Crossover", space="buy", optimize=True, load=True)

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
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
        
        hlc3 = dataframe.ta.hlc3()
        ohlc4 = dataframe.ta.ohlc4()
        
        # WaveTrend
        wt1, wt2 = ci.WaveTrendOscillator(hlc3)
        wt1 = wt1.round(2)
        wt2 = wt2.round(2)
        obLevel2 = 60 
        obLevel = 50  
        osLevel = -50  
        osLevel2 = -60
        WTCross = pta.cross(wt1, wt2)
        WTCrossUp = wt2 - wt1 <= 0
        WTCrossDown = wt2 - wt1 >= 0
        WTOverSold = wt2 <= osLevel2
        WTOverBought = wt2 >= obLevel2
        
        # Elders Safe Zone
        # shortStop - stoploss when short order (higher price)
        # longStop - stoploss when long order (lower price)
        shortStop, longStop = ci.ElderSafeZone(dataframe['high'], dataframe['low'])
        longStop = longStop.round(2)
        shortStop = shortStop.round(2)
        
        # RSI
        rsi = pd.Series(pta.rsi(dataframe['close'])).round(2)

        # EMA
        trendBar1FastMA = pd.Series(dataframe.ta.ema(5)).round(2)
        trendBar1SlowMA = pd.Series(dataframe.ta.ema(11)).round(2)
        trendBar2FastMA = pd.Series(dataframe.ta.ema(13)).round(2)
        trendBar2SlowMA = pd.Series(dataframe.ta.sma(36)).round(2)
        
        ema15 = pd.Series(dataframe.ta.ema(15)).round(2)
        ema20 = pd.Series(dataframe.ta.ema(20)).round(2)
        ema30 = pd.Series(dataframe.ta.ema(30)).round(2)
        ema40 = pd.Series(dataframe.ta.ema(40)).round(2)
        ema50 = pd.Series(dataframe.ta.ema(50)).round(2)
        ema200 = pd.Series(dataframe.ta.ema(200)).round(2)
        
        # MA Crossover Condition
        MACrossover1 = trendBar1FastMA > trendBar1SlowMA
        MACrossover2 = trendBar2FastMA > trendBar2SlowMA

        # MA Direction Condition
        MA1Direction = trendBar1FastMA > trendBar1FastMA.shift(1)
        MA2Direction = trendBar1SlowMA > trendBar1SlowMA.shift(1)
        MA3Direction = trendBar2FastMA > trendBar2FastMA.shift(1)
        MA4Direction = trendBar2SlowMA > trendBar2SlowMA.shift(1)
        
        # MACD
        macd_df = dataframe.ta.macd()
        macd, deltamacd, signalmacd = macd_df.iloc[:, 0].round(2), macd_df.iloc[:, 1].round(2), macd_df.iloc[:, 2].round(2)
        MACDHistogramCross = pd.Series(deltamacd > 0)
        MACDLineOverZero = pd.Series(macd > 0)

        fast_macd_df = dataframe.ta.macd(8, 21, 5)
        macd_fast, deltamacd_fast, signalmacd_fast = fast_macd_df.iloc[:, 0].round(2), fast_macd_df.iloc[:, 1].round(2), fast_macd_df.iloc[:, 2].round(2)
        FastMACDHistogramCross = pd.Series(deltamacd_fast > 0)
        FastMACDLineOverZero = pd.Series(macd_fast > 0)
        
        # Top Dog Trading - Mom Dad Calculations
        TopDog_Fast_MA = 5
        TopDog_Slow_MA = 20
        TopDog_Sig = 30

        TopDogMom = pd.Series(dataframe.ta.ema(TopDog_Fast_MA)).round(2) - pd.Series(dataframe.ta.ema(TopDog_Slow_MA)).round(2)
        TopDogDad = pd.Series(pta.ema(TopDogMom, TopDog_Sig)).round(2)

        # Top Dog Dad - Background Color Change Condition
        TopDogDadDirection = TopDogDad > TopDogDad.shift(1)
        TopDogMomOverDad = TopDogMom > TopDogDad
        TopDogMomOverZero = TopDogMom > 0
        TopDogDadDirectandMomOverZero = (TopDogDadDirection & TopDogMomOverZero) > 0
        TopDogDadDirectandMomUnderZero = (~TopDogDadDirection) & (~TopDogMomOverZero) > 0

        # UCS_Trend by ucsgears copy Trend Candles
        # Interpretation of TTM Trend bars. It is really close to the actual. 
        heikinashi = qtpylib.heikinashi(dataframe)
        ha_open = heikinashi['open']
        ha_close = heikinashi['close']
        ha_high = heikinashi['high']
        ha_low = heikinashi['low']
        
        ccolor = ha_close - ha_open > 0

        haopen = ha_open.shift(6)
        haclose = ha_close.shift(6)
        inside6 = (ha_open <= np.fmax(haopen, haclose)) & (ha_open >= np.fmin(haopen, haclose)) & (haclose <= np.fmax(haopen, haclose)) & (haclose >= np.fmin(haopen, haclose))

        haopen = ha_open.shift(5)
        haclose = ha_close.shift(5)
        inside5 = (ha_open <= np.fmax(haopen, haclose)) & (ha_open >= np.fmin(haopen, haclose)) & (haclose <= np.fmax(haopen, haclose)) & (haclose >= np.fmin(haopen, haclose))
        
        haopen = ha_open.shift(4)
        haclose = ha_close.shift(4)
        inside4 = (ha_open <= np.fmax(haopen, haclose)) & (ha_open >= np.fmin(haopen, haclose)) & (haclose <= np.fmax(haopen, haclose)) & (haclose >= np.fmin(haopen, haclose))
        
        haopen = ha_open.shift(3)
        haclose = ha_close.shift(3)
        inside3 = (ha_open <= np.fmax(haopen, haclose)) & (ha_open >= np.fmin(haopen, haclose)) & (haclose <= np.fmax(haopen, haclose)) & (haclose >= np.fmin(haopen, haclose))
        
        haopen = ha_open.shift(2)
        haclose = ha_close.shift(2)
        inside2 = (ha_open <= np.fmax(haopen, haclose)) & (ha_open >= np.fmin(haopen, haclose)) & (haclose <= np.fmax(haopen, haclose)) & (haclose >= np.fmin(haopen, haclose))
        
        haopen = ha_open.shift(1)
        haclose = ha_close.shift(1)
        inside1 = (ha_open <= np.fmax(haopen, haclose)) & (ha_open >= np.fmin(haopen, haclose)) & (haclose <= np.fmax(haopen, haclose)) & (haclose >= np.fmin(haopen, haclose))

        colorvalue = np.where(inside6, ccolor.shift(6), np.where(inside5, ccolor.shift(5), np.where(inside4, ccolor.shift(4), np.where(inside3, ccolor.shift(3), np.where(inside2, ccolor.shift(2), np.where(inside1, ccolor.shift(1), ccolor))))))

        TrendBarTrend_Candle = pd.Series(colorvalue)
        
        
        # RSI 5 Trend Barmeter Calculations
        RSI5 = pd.Series(dataframe.ta.rsi(5)).round(2)
        RSI5Above50 = RSI5 > 50

        # RSI 13 Trend Barmeter Calculations
        RSI13 = pd.Series(dataframe.ta.rsi(13)).round(2)
        RSI13Above50 = RSI13 > 50
        
        # Linear Regression Calculation For RSI Signal Line

        SignalLineLength1 = 21
        reg_line = pta.linreg(RSI13, SignalLineLength1)
        LinReg1 = pd.Series(reg_line)

        RSISigDirection = LinReg1 > LinReg1.shift(1)
        RSISigCross = RSI13 > LinReg1

        # Volatility Oscillator
        vo_spike, vo_upper, vo_lower = ci.VolatilityOscillator(dataframe['close'], dataframe['open'])
        vo_spike = vo_spike.round(2)
        vo_upper = vo_upper.round(2)
        vo_lower = vo_lower.round(2)
        
        dataframe = pd.concat([dataframe, 
            hlc3.rename('hlc3'),
            rsi.rename('rsi'),
            macd.rename('macd'),
            MACrossover1.rename('MACrossover1'),
            MACrossover2.rename('MACrossover2'),
            MA1Direction.rename('MA1Direction'),
            MA2Direction.rename('MA2Direction'),
            MA3Direction.rename('MA3Direction'),
            MA4Direction.rename('MA4Direction'),
            MACDHistogramCross.rename('MACDHistogramCross'),
            FastMACDHistogramCross.rename('FastMACDHistogramCross'),
            TopDogMomOverDad.rename('TopDogMomOverDad'),
            TopDogDadDirection.rename('TopDogDadDirection'),
            RSISigCross.rename('RSISigCross'),
            RSI5Above50.rename('RSI5Above50'),
            RSI13Above50.rename('RSI13Above50'),
            TrendBarTrend_Candle.rename('TrendBarTrend_Candle'),
            ema15.rename('ema15'),
            ema20.rename('ema20'),
            ema30.rename('ema30'),
            ema40.rename('ema40'),
            ema50.rename('ema50'),
            ema200.rename('ema200'),
            longStop.rename('elderSafezoneLong'),
            shortStop.rename('elderSafezoneShort'),
            WTCross.rename('WTCross'),
            WTCrossUp.rename('WTCrossUp'),
            WTCrossDown.rename('WTCrossDown'),
            vo_spike.rename('vo_spike'),
            vo_upper.rename('vo_upper'),
            vo_lower.rename('vo_lower'),
        ], axis=1)
        
        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # Trend Meter 1
        TrendBar1Result = dataframe["MACrossover1"] if self.trendMeter1.value == "MA Crossover" else \
        dataframe["MACDHistogramCross"] if self.trendMeter1.value == "MACD Crossover - 12, 26, 9" else \
        dataframe["FastMACDHistogramCross"] if self.trendMeter1.value == "MACD Crossover - Fast - 8, 21, 5" else \
        dataframe["TopDogMomOverDad"] if self.trendMeter1.value == "Mom Dad Cross (Top Dog Trading)" else \
        dataframe["TopDogDadDirection"] if self.trendMeter1.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["RSISigCross"] if self.trendMeter1.value == "RSI Signal Line Cross - RSI 13, Sig 21" else \
        dataframe["RSI5Above50"] if self.trendMeter1.value == "RSI 5: > or < 50" else \
        dataframe["RSI13Above50"] if self.trendMeter1.value == "RSI 13: > or < 50" else \
        dataframe["TrendBarTrend_Candle"] if self.trendMeter1.value == "Trend Candles" else pd.Series(np.full(dataframe["MACrossover1"].shape, False))
        
        # Trend Meter 2
        TrendBar2Result = dataframe["MACrossover1"] if self.trendMeter2.value == "MA Crossover" else \
        dataframe["MACDHistogramCross"] if self.trendMeter2.value == "MACD Crossover - 12, 26, 9" else \
        dataframe["FastMACDHistogramCross"] if self.trendMeter2.value == "MACD Crossover - Fast - 8, 21, 5" else \
        dataframe["TopDogMomOverDad"] if self.trendMeter2.value == "Mom Dad Cross (Top Dog Trading)" else \
        dataframe["TopDogDadDirection"] if self.trendMeter2.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["RSISigCross"] if self.trendMeter2.value == "RSI Signal Line Cross - RSI 13, Sig 21" else \
        dataframe["RSI5Above50"] if self.trendMeter2.value == "RSI 5: > or < 50" else \
        dataframe["RSI13Above50"] if self.trendMeter2.value == "RSI 13: > or < 50" else \
        dataframe["TrendBarTrend_Candle"] if self.trendMeter2.value == "Trend Candles" else pd.Series(np.full(dataframe["MACrossover1"].shape, False))
        
        # Trend Meter 3
        TrendBar3Result = dataframe["MACrossover1"] if self.trendMeter3.value == "MA Crossover" else \
        dataframe["MACDHistogramCross"] if self.trendMeter3.value == "MACD Crossover - 12, 26, 9" else \
        dataframe["FastMACDHistogramCross"] if self.trendMeter3.value == "MACD Crossover - Fast - 8, 21, 5" else \
        dataframe["TopDogMomOverDad"] if self.trendMeter3.value == "Mom Dad Cross (Top Dog Trading)" else \
        dataframe["TopDogDadDirection"] if self.trendMeter3.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["RSISigCross"] if self.trendMeter3.value == "RSI Signal Line Cross - RSI 13, Sig 21" else \
        dataframe["RSI5Above50"] if self.trendMeter3.value == "RSI 5: > or < 50" else \
        dataframe["RSI13Above50"] if self.trendMeter3.value == "RSI 13: > or < 50" else \
        dataframe["TrendBarTrend_Candle"] if self.trendMeter3.value == "Trend Candles" else pd.Series(np.full(dataframe["MACrossover1"].shape, False))

        TrendBars3Positive = TrendBar1Result & TrendBar2Result & TrendBar3Result
        TrendBars3Negative = (~TrendBar1Result) & (~TrendBar2Result) & (~TrendBar3Result)

        TrendFilterPlus = (dataframe["ema15"] > dataframe["ema20"]) & (dataframe["ema20"] > dataframe["ema30"]) & \
                            (dataframe["ema30"] > dataframe["ema40"]) & (dataframe["ema40"] > dataframe["ema50"])

        TrendFilterMinus = (dataframe["ema15"] < dataframe["ema20"]) & (dataframe["ema20"] < dataframe["ema30"]) & \
                            (dataframe["ema30"] < dataframe["ema40"]) & (dataframe["ema40"] < dataframe["ema50"])

        MSBar1PositiveWaveTrendSignal = TrendFilterPlus & dataframe["WTCross"] & dataframe["WTCrossUp"]
        MSBar1NegativeWaveTrendSignal = TrendFilterMinus & dataframe["WTCross"] & dataframe["WTCrossDown"]

        MSBar2PositiveWaveTrendSignal = TrendFilterPlus & dataframe["WTCross"] & dataframe["WTCrossUp"]
        MSBar2NegativeWaveTrendSignal = TrendFilterMinus & dataframe["WTCross"] & dataframe["WTCrossDown"]
        
        BackgroundColorChangePositive = TrendBars3Positive & (~TrendBars3Positive.shift(1, fill_value=False))
        BackgroundColorChangeNegative = TrendBars3Negative & (~TrendBars3Negative.shift(1, fill_value=False))
        
        # Signals 1 - Wave Trend Signals
        MSBar1Color = np.where(MSBar1PositiveWaveTrendSignal, 1, np.where(MSBar1NegativeWaveTrendSignal, -1, 0))
        # Signals 2 - All 3 Trend Meters Now Align
        MSBar2Color = np.where(BackgroundColorChangePositive, 1, np.where(BackgroundColorChangeNegative, -1, 0))

        # Trend Barmeter Color Assignments
        # Trend Bar 1 - Thin Line
        TrendBar4Result = dataframe["TopDogDadDirection"] if self.trendBar1.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["MACDHistogramCross"] if self.trendBar1.value == "MACD Crossover" else \
        dataframe["MA1Direction"] if self.trendBar1.value == "MA Direction - Fast MA - TB1" else \
        dataframe["MA2Direction"] if self.trendBar1.value == "MA Direction - Slow MA - TB1" else dataframe["MACrossover1"]
        
        # Trend Bar 2 - Thin Line
        TrendBar5Result = dataframe["TopDogDadDirection"] if self.trendBar2.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["MACDHistogramCross"] if self.trendBar2.value == "MACD Crossover" else \
        dataframe["MA3Direction"] if self.trendBar2.value == "MA Direction - Fast MA - TB2" else \
        dataframe["MA4Direction"] if self.trendBar2.value == "MA Direction - Slow MA - TB2" else dataframe["MACrossover2"]
        
        # Trend Meter Background Highlight - 3 Trend Meter Conditions Met
        TrendBar3BarsSame = np.where(TrendBars3Positive, 1, np.where(TrendBars3Negative, -1, 0))
        
        dataframe.loc[
            (
                # Uptrend Checks
                (dataframe["ema50"] > dataframe["ema200"]) & (dataframe['hlc3'] > dataframe["ema200"]) & \
                # Trend Meter Checks
                (MSBar2Color == 1) & (TrendBar3BarsSame == 1) & TrendBar4Result & TrendBar5Result & \
                # Volatility Oscillator Checks
                (dataframe["vo_spike"] > (dataframe["vo_upper"] * 1.25)) & \
                # Basic Checks
                # Make sure Volume is not 0
                (dataframe['volume'] > 0)
            ), 'enter_long'] = 1
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
                (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
                (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1
        """

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict):
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # Trend Meter 1
        TrendBar1Result = dataframe["MACrossover1"] if self.trendMeter1.value == "MA Crossover" else \
        dataframe["MACDHistogramCross"] if self.trendMeter1.value == "MACD Crossover - 12, 26, 9" else \
        dataframe["FastMACDHistogramCross"] if self.trendMeter1.value == "MACD Crossover - Fast - 8, 21, 5" else \
        dataframe["TopDogMomOverDad"] if self.trendMeter1.value == "Mom Dad Cross (Top Dog Trading)" else \
        dataframe["TopDogDadDirection"] if self.trendMeter1.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["RSISigCross"] if self.trendMeter1.value == "RSI Signal Line Cross - RSI 13, Sig 21" else \
        dataframe["RSI5Above50"] if self.trendMeter1.value == "RSI 5: > or < 50" else \
        dataframe["RSI13Above50"] if self.trendMeter1.value == "RSI 13: > or < 50" else \
        dataframe["TrendBarTrend_Candle"] if self.trendMeter1.value == "Trend Candles" else pd.Series(np.full(dataframe["MACrossover1"].shape, False))
        
        # Trend Meter 2
        TrendBar2Result = dataframe["MACrossover1"] if self.trendMeter2.value == "MA Crossover" else \
        dataframe["MACDHistogramCross"] if self.trendMeter2.value == "MACD Crossover - 12, 26, 9" else \
        dataframe["FastMACDHistogramCross"] if self.trendMeter2.value == "MACD Crossover - Fast - 8, 21, 5" else \
        dataframe["TopDogMomOverDad"] if self.trendMeter2.value == "Mom Dad Cross (Top Dog Trading)" else \
        dataframe["TopDogDadDirection"] if self.trendMeter2.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["RSISigCross"] if self.trendMeter2.value == "RSI Signal Line Cross - RSI 13, Sig 21" else \
        dataframe["RSI5Above50"] if self.trendMeter2.value == "RSI 5: > or < 50" else \
        dataframe["RSI13Above50"] if self.trendMeter2.value == "RSI 13: > or < 50" else \
        dataframe["TrendBarTrend_Candle"] if self.trendMeter2.value == "Trend Candles" else pd.Series(np.full(dataframe["MACrossover1"].shape, False))
        
        # Trend Meter 3
        TrendBar3Result = dataframe["MACrossover1"] if self.trendMeter3.value == "MA Crossover" else \
        dataframe["MACDHistogramCross"] if self.trendMeter3.value == "MACD Crossover - 12, 26, 9" else \
        dataframe["FastMACDHistogramCross"] if self.trendMeter3.value == "MACD Crossover - Fast - 8, 21, 5" else \
        dataframe["TopDogMomOverDad"] if self.trendMeter3.value == "Mom Dad Cross (Top Dog Trading)" else \
        dataframe["TopDogDadDirection"] if self.trendMeter3.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["RSISigCross"] if self.trendMeter3.value == "RSI Signal Line Cross - RSI 13, Sig 21" else \
        dataframe["RSI5Above50"] if self.trendMeter3.value == "RSI 5: > or < 50" else \
        dataframe["RSI13Above50"] if self.trendMeter3.value == "RSI 13: > or < 50" else \
        dataframe["TrendBarTrend_Candle"] if self.trendMeter3.value == "Trend Candles" else pd.Series(np.full(dataframe["MACrossover1"].shape, False))

        TrendBars3Positive = TrendBar1Result & TrendBar2Result & TrendBar3Result
        TrendBars3Negative = (~TrendBar1Result) & (~TrendBar2Result) & (~TrendBar3Result)

        TrendFilterPlus = (dataframe["ema15"] > dataframe["ema20"]) & (dataframe["ema20"] > dataframe["ema30"]) & \
                            (dataframe["ema30"] > dataframe["ema40"]) & (dataframe["ema40"] > dataframe["ema50"])

        TrendFilterMinus = (dataframe["ema15"] < dataframe["ema20"]) & (dataframe["ema20"] < dataframe["ema30"]) & \
                            (dataframe["ema30"] < dataframe["ema40"]) & (dataframe["ema40"] < dataframe["ema50"])

        MSBar1PositiveWaveTrendSignal = TrendFilterPlus & dataframe["WTCross"] & dataframe["WTCrossUp"]
        MSBar1NegativeWaveTrendSignal = TrendFilterMinus & dataframe["WTCross"] & dataframe["WTCrossDown"]

        MSBar2PositiveWaveTrendSignal = TrendFilterPlus & dataframe["WTCross"] & dataframe["WTCrossUp"]
        MSBar2NegativeWaveTrendSignal = TrendFilterMinus & dataframe["WTCross"] & dataframe["WTCrossDown"]
        
        BackgroundColorChangePositive = TrendBars3Positive & (~TrendBars3Positive.shift(1, fill_value=False))
        BackgroundColorChangeNegative = TrendBars3Negative & (~TrendBars3Negative.shift(1, fill_value=False))
        
        # Signals 1 - Wave Trend Signals
        MSBar1Color = np.where(MSBar1PositiveWaveTrendSignal, 1, np.where(MSBar1NegativeWaveTrendSignal, -1, 0))
        # Signals 2 - All 3 Trend Meters Now Align
        MSBar2Color = np.where(BackgroundColorChangePositive, 1, np.where(BackgroundColorChangeNegative, -1, 0))

        # Trend Barmeter Color Assignments
        # Trend Bar 1 - Thin Line
        TrendBar4Result = dataframe["TopDogDadDirection"] if self.trendBar1.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["MACDHistogramCross"] if self.trendBar1.value == "MACD Crossover" else \
        dataframe["MA1Direction"] if self.trendBar1.value == "MA Direction - Fast MA - TB1" else \
        dataframe["MA2Direction"] if self.trendBar1.value == "MA Direction - Slow MA - TB1" else dataframe["MACrossover1"]
        
        # Trend Bar 2 - Thin Line
        TrendBar5Result = dataframe["TopDogDadDirection"] if self.trendBar2.value == "DAD Direction (Top Dog Trading)" else \
        dataframe["MACDHistogramCross"] if self.trendBar2.value == "MACD Crossover" else \
        dataframe["MA3Direction"] if self.trendBar2.value == "MA Direction - Fast MA - TB2" else \
        dataframe["MA4Direction"] if self.trendBar2.value == "MA Direction - Slow MA - TB2" else dataframe["MACrossover2"]
        
        # Trend Meter Background Highlight - 3 Trend Meter Conditions Met
        TrendBar3BarsSame = np.where(TrendBars3Positive, 1, np.where(TrendBars3Negative, -1, 0))
        
        dataframe.loc[
            (
                ~TrendBar1Result
            ),
            'exit_long'] = 1
        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
                (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
                (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1
        """
        return dataframe
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs):
        """
        Custom stoploss logic, returning the new distance relative to current_rate (as ratio).
        e.g. returning -0.05 would create a stoploss 5% below current_rate.
        The custom stoploss can never be below self.stoploss, which serves as a hard maximum loss.

        When not implemented by a strategy, returns the initial stoploss value
        Only called when use_custom_stoploss is set to True.

        :param pair: Pair that's currently analyzed
        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: New stoploss value, relative to the current rate
        """
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # Use parabolic sar as absolute stoploss price
        stoploss_price = last_candle['ema200']

        # Convert absolute price to percentage relative to current_rate
        if stoploss_price < current_rate:
            return (stoploss_price / current_rate) - 1

        # return maximum stoploss value, keeping current stoploss price unchanged
        return 1
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs):
        """
        Called right before placing a regular sell order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be sold.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in quote currency.
        :param rate: Rate that's going to be used when using limit orders
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the exit-order is placed on the exchange.
            False aborts the process
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if exit_reason == 'exit_signal' and trade.calc_profit_ratio(rate) < 0:
            # Reject exit_signal with negative profit
            return False
        return True