# Best Divergence Strategy Configs

## Best Divergence Strategy Config using Sortino Ratio

Using optimizer random state: `58339` <br>
Command Line: `freqtrade hyperopt --strategy DivStrat --hyperoptloss SortinoHyperOptLossDaily --spaces buy sell -e 500`

**Best result:**

```
204 trades
163/0/41 Wins/Draws/Losses
Avg profit 2.31%
Median profit 1.49%
Total profit 96037.13062909 USDT ( 9603.71%).
Avg duration 4:50:00
min. Objective: -137.46508`
```

| Best |  Epoch  | Trades | Win Draw Loss | Avg profit |            Profit            |  Avg duration   |      Max Drawdown      | Objective |
| :--: | :-----: | :----: | :-----------: | :--------: | :--------------------------: | :-------------: | :--------------------: | :-------: |
| Best | 20/500  |  278   | 207 / 0 / 71  |   1.91%    | 171086.902 USDT (17,108.69%) | 0 days 04:59:00 | 3692.006 USDT (5.99%)  | -85.8251  |
| Best | 39/500  |  285   | 215 / 0 / 70  |   1.87%    | 175787.838 USDT (17,578.78%) | 0 days 04:31:00 | 3792.862 USDT (11.27%) | -98.3547  |
| Best | 45/500  |  173   | 137 / 0 / 36  |   2.32%    |  47960.870 USDT (4,796.09%)  | 0 days 06:04:00 |  585.629 USDT (7.49%)  | -99.9144  |
| Best | 98/500  |  209   | 166 / 0 / 43  |   2.21%    |  87581.722 USDT (8,758.17%)  | 0 days 05:23:00 | 1456.988 USDT (5.99%)  | -102.861  |
| Best | 115/500 |  208   | 165 / 0 / 43  |   2.26%    |  93885.584 USDT (9,388.56%)  | 0 days 04:49:00 | 1589.966 USDT (5.99%)  | -104.494  |
| Best | 123/500 |  217   | 171 / 0 / 46  |   2.26%    | 115372.934 USDT (11,537.29%) | 0 days 05:01:00 | 1803.250 USDT (5.99%)  |  -108.51  |
| Best | 147/500 |  255   | 198 / 0 / 57  |   2.08%    | 168765.014 USDT (16,876.50%) | 0 days 04:42:00 | 2442.644 USDT (5.99%)  | -110.234  |
| Best | 205/500 |  200   | 161 / 0 / 39  |   2.43%    | 109921.952 USDT (10,992.20%) | 0 days 04:57:00 | 1590.718 USDT (5.99%)  | -124.661  |
| Best | 217/500 |  185   | 149 / 0 / 36  |   2.28%    |  59068.804 USDT (5,906.88%)  | 0 days 04:50:00 |  722.045 USDT (5.99%)  | -130.359  |
| Best | 244/500 |  204   | 163 / 0 / 41  |   2.31%    |  96037.131 USDT (9,603.71%)  | 0 days 04:50:00 | 1166.415 USDT (5.99%)  | -137.465  |

**Configuration:**

```json
{
  "strategy_name": "DivStrat",
  "params": {
    "roi": {
      "0": 100
    },
    "stoploss": {
      "stoploss": -0.013
    },
    "trailing": {
      "trailing_stop": false,
      "trailing_stop_positive": null,
      "trailing_stop_positive_offset": 0.0,
      "trailing_only_offset_is_reached": false
    },
    "buy": {
      "buy_useBTC": false,
      "buy_flag": 282,
      "buy_minsignals": 1,
      "maxpp": 3,
      "prd": 1
    },
    "sell": {
      "sell_flag": 117,
      "sell_minsignals": 0
    },
    "protection": {}
  },
  "ft_stratparam_v": 1,
  "export_time": "2021-09-10 04:03:35.810299+00:00"
}
```
