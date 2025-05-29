import os
import asyncio
from pydantic import Field
from typing import Any
from proconfig.widgets.base import WIDGETS, BaseWidget
from proconfig.widgets.custom_widgets import run_custom_widgets
import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

@WIDGETS.register_module()
class USStockTradingSignalWidget(BaseWidget):
    CATEGORY = "Custom Widgets/US Stock Trading"
    NAME = "US Stock Trading Signal"
    
    class InputsSchema(BaseWidget.InputsSchema):
        prompt: str = Field("test prompt", description="the prompt")
        date: str = Field(None, description="回测起始日期，格式如20240101")
        init_balance: str = Field("100000", description="回测初始资金")
    
    class OutputsSchema(BaseWidget.OutputsSchema):
        reply: str
        
    @staticmethod
    def get_current_signal():
        tsla = yf.Ticker("TSLA")
        data = tsla.history(period="1mo")
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Long_Entry'] = ((data['SMA_5'] > data['SMA_20']) & (data['SMA_5'].shift(1) <= data['SMA_20'].shift(1))).astype(int)
        data['Long_Exit'] = ((data['SMA_5'] < data['SMA_20']) & (data['SMA_5'].shift(1) >= data['SMA_20'].shift(1))).astype(int)
        data['Position'] = 0
        for i in range(1, len(data)):
            if data['Long_Entry'].iloc[i] == 1 and data['Position'].iloc[i-1] == 0:
                data['Position'].iloc[i] = 1
            elif data['Long_Exit'].iloc[i] == 1 and data['Position'].iloc[i-1] == 1:
                data['Position'].iloc[i] = 0
            else:
                data['Position'].iloc[i] = data['Position'].iloc[i-1]
        if len(data) > 0:
            if data['Long_Entry'].iloc[-1] == 1:
                return "Buy"
            elif data['Long_Exit'].iloc[-1] == 1:
                return "Sell"
            else:
                return "Hold"
        else:
            return "无数据"

    @staticmethod
    def run_backtest(start_date_str, init_balance):
        # Robustly convert init_balance to float, fallback to 100000 if invalid
        try:
            init_balance = float(init_balance)
        except (TypeError, ValueError):
            init_balance = 100000.0
        tsla = yf.Ticker("TSLA")
        data = tsla.history(period="1y")
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data_tz = data['Date'].dt.tz
        start_date = pd.to_datetime(start_date_str, format='%Y%m%d')
        if data_tz is not None and start_date.tzinfo is None:
            start_date = start_date.tz_localize(data_tz)
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['Long_Entry'] = ((data['SMA_5'] > data['SMA_20']) & (data['SMA_5'].shift(1) <= data['SMA_20'].shift(1))).astype(int)
        data['Long_Exit'] = ((data['SMA_5'] < data['SMA_20']) & (data['SMA_5'].shift(1) >= data['SMA_20'].shift(1))).astype(int)
        data['Position'] = 0
        start_idx = data.index[data['Date'] >= start_date]
        if len(start_idx) == 0:
            return "## 起始日期超出数据范围"
        start_idx = start_idx[0]
        data.loc[start_idx, 'Position'] = 0  # always start with no position
        for i in range(start_idx+1, len(data)):
            if data.loc[i, 'Long_Entry'] == 1 and data.loc[i-1, 'Position'] == 0:
                data.loc[i, 'Position'] = 1
            elif data.loc[i, 'Long_Exit'] == 1 and data.loc[i-1, 'Position'] == 1:
                data.loc[i, 'Position'] = 0
            else:
                data.loc[i, 'Position'] = data.loc[i-1, 'Position']
        transactions = data.iloc[start_idx:].copy()
        transactions = transactions[transactions['Position'] != transactions['Position'].shift(1)]
        transactions['Signal'] = np.where(transactions['Position'] == 1, 'Buy', 'Sell')
        # markdown table
        table_md = transactions[['Date', 'Close', 'SMA_5', 'SMA_20', 'Signal']].to_markdown(index=False)
        # 计算图表的起始日期
        plot_start_date = start_date - pd.DateOffset(months=1)
        plot_data = data[data['Date'] >= plot_start_date]
        plt.figure(figsize=(14, 7))
        plt.plot(plot_data['Date'], plot_data['Close'], label='Close Price', color='blue')
        plt.plot(plot_data['Date'], plot_data['SMA_5'], label='SMA 5', color='teal')
        plt.plot(plot_data['Date'], plot_data['SMA_20'], label='SMA 20', color='orange')
        plot_transactions = transactions[transactions['Date'] >= plot_start_date]
        plt.scatter(
            plot_transactions[plot_transactions['Signal'] == 'Buy']['Date'],
            plot_transactions[plot_transactions['Signal'] == 'Buy']['Close'],
            marker='^', color='green', label='Buy Signal',
            s=180, edgecolors='black', linewidths=2, zorder=10
        )
        plt.scatter(
            plot_transactions[plot_transactions['Signal'] == 'Sell']['Date'],
            plot_transactions[plot_transactions['Signal'] == 'Sell']['Close'],
            marker='v', color='red', label='Sell Signal',
            s=180, edgecolors='black', linewidths=2, zorder=10
        )
        plt.legend()
        plt.title('TSLA 5-20 MA Crossover Strategy Backtest')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        img_md = f'![Backtest Plot](data:image/png;base64,{img_base64})'
        # 资金回测参数
        balance = init_balance
        position = 0  # 0: 空仓, 1: 满仓
        shares = 0
        balances = []
        for i in range(start_idx, len(data)):
            price = data.loc[i, 'Close']
            signal = data.loc[i, 'Position']
            if position == 0 and signal == 1:
                shares = balance // price
                balance -= shares * price
                position = 1
            elif position == 1 and signal == 0:
                balance += shares * price
                shares = 0
                position = 0
            total = balance + shares * price
            balances.append(total)
        if position == 1 and shares > 0:
            balance += shares * data.loc[len(data)-1, 'Close']
            shares = 0
            position = 0
        final_balance = balance
        profit_loss_ratio = (final_balance - init_balance) / init_balance
        ror = final_balance / init_balance - 1
        stats_md = f'''
### 回测统计
- 起始资金 (init_balance): {init_balance:.2f}
- 结束资金 (balance): {final_balance:.2f}
- 盈亏额比例 (profit_loss_ratio): {profit_loss_ratio:.2%}
- 收益率 (ror): {ror:.2%}
'''
        md_report = f"""
## TSLA 5-20 MA Crossover 回测报告

**回测起始日期:** {start_date.strftime('%Y-%m-%d')}  
**初始资金:** {init_balance}

{img_md}

{stats_md}

#### 交易信号表

{table_md}
"""
        return md_report

    def execute(self, environ, config):
        prompt = getattr(config, "prompt", "")
        if prompt == "交易信号" or prompt.lower() == "signal":
            signal = self.get_current_signal()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {"reply": f"{now} 的交易信号为: {signal}"}
        elif prompt == "回测报告":
            date_str = getattr(config, "date", None)
            init_balance = getattr(config, "init_balance", "100000")
            if not date_str or not (isinstance(date_str, str) and len(date_str) == 8 and date_str.isdigit()):
                return {"reply": "请提供合法的回测起始日期参数（date），格式如20240101"}
            md_report = self.run_backtest(date_str, init_balance)
            return {"reply": md_report}
        else:
            output = asyncio.run(run_custom_widgets('custom_widget_usstock/USStockTradingSignalWidget', environ, config))
            return output
