from datetime import datetime
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
        sell_loss = np.where(results['profit_abs'] < 0, 1, 0).sum()
        avg_trade_duration = results['trade_duration'].mean()
        
        result = -1 * (total_profit / (sell_loss + 1))
        return result
