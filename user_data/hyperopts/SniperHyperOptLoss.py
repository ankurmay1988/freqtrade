from datetime import date, datetime
from math import exp
from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


class SniperHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    This is intended to give you some inspiration for your own loss function.

    The Function needs to return a number (float) - which becomes smaller for better backtest
    results.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               config: Dict, processed: Dict[str, DataFrame],
                               *args, **kwargs) -> float:
        """
        Objective function, returns smaller number for better results
        """
        total_profit = results['profit_abs'].sum()
        avg_profit_pct = results.loc[results['profit_ratio'] > 0]['profit_ratio'].mean() * 100
        loss_trades = np.where(results['profit_abs'] < 0, 1, 0).sum()
        total_loss = np.where(results['profit_abs'] < 0, (-1 * results['profit_abs']), 0).sum()
        win_trades = np.where(results['profit_abs'] > 0, 1, 0).sum()
        total_win = np.where(results['profit_abs'] > 0, results['profit_abs'], 0).sum()
        timediff = max_date - min_date
        loss_trades_normalized = 99999 if loss_trades > timediff.days else loss_trades
        avg_trade_duration = results['trade_duration'].mean()
        profitAndLoss = total_profit / (loss_trades_normalized + 1)
        winRatio = win_trades / (loss_trades_normalized + 1)
        result = -1 * winRatio * win_trades * (total_profit/abs(total_profit))
        return result
