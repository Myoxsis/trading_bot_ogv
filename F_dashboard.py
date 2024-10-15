#%%
import streamlit as st 
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime
import os

import plotly.graph_objects as go

pd.options.display.max_columns = 999
pd.options.display.max_rows = 49

st.set_page_config(page_title="Portfolio Evolution", layout="wide")

#%%

def f_memoize_dt(s):
    """
    Memoization technique
    """
    dates = {date:datetime.datetime.strptime(date,'%d/%m/%Y') for date in s.unique()}
    return s.map(dates)

#%%


# utlis function
def keep_first_signal(signal_list):
    result = []
    in_sequence = False  # A flag to track if we are in a sequence of ones
    
    for value in signal_list:
        if value == 1:
            if not in_sequence:
                result.append(1)
                in_sequence = True
            else:
                result.append(0)
        else:
            result.append(0)
            in_sequence = False
            
    return result

class DataLoader:
    def __init__(self, csv_file: str = None, data: pd.DataFrame = None, min_date : str = None, max_date : str = None):
        """
        Initialize the DataLoader object to load and preprocess data.

        Parameters:
        - csv_file: Path to a CSV file (formatted like Yahoo Finance data).
        - data: A pandas DataFrame with historical price data.
        """
        if csv_file:
            self.data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either 'data' DataFrame or 'csv_file' must be provided.")

        if min_date is not None:
            self.data = self.data[self.data.index >= min_date]
        else:
            pass
        
        if max_date is not None:
            self.data = self.data[self.data.index <= max_date]
        else :
            pass


    def get_data(self) -> pd.DataFrame:
        """
        Return the loaded data.
        """
        return self.data


class Strategy:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the Strategy class with data.

        Parameters:
        - data: A pandas DataFrame containing historical price data.
        """
        self.data = data.copy()
        self.indicators = self.data.copy()

    def calculate_moving_averages(self, moving_average_types: list = ['short', 'long1'], 
                                   short_window: int = 20, medium_window: int = 50, 
                                   long_window1: int = 100, long_window2: int = 200):
        """
        Calculate the specified moving averages and add them to the DataFrame.
        """
        if 'short' in moving_average_types:
            self.indicators = self.indicators.assign(short_ma=self.indicators['Adj Close'].rolling(window=short_window, min_periods=1).mean())
        if 'medium' in moving_average_types:
            self.indicators = self.indicators.assign(medium_ma=self.indicators['Adj Close'].rolling(window=medium_window, min_periods=1).mean())
        if 'long1' in moving_average_types:
            self.indicators = self.indicators.assign(long_ma1=self.indicators['Adj Close'].rolling(window=long_window1, min_periods=1).mean())
        if 'long2' in moving_average_types:
            self.indicators = self.indicators.assign(long_ma2=self.indicators['Adj Close'].rolling(window=long_window2, min_periods=1).mean())

    def calculate_rsi(self, period: int = 14):
        """
        Calculate the Relative Strength Index (RSI) and add it to the DataFrame.
        """

        delta = self.indicators['Adj Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        self.indicators = self.indicators.assign(RSI=100 - (100 / (1 + rs)))

    def get_indicators(self) -> pd.DataFrame:
        """
        Return the data with indicators.
        """
        return self.indicators


class TradingStrategy:
    def __init__(self, strategy: Strategy):
        """
        Initialize the TradingStrategy class with indicators.

        Parameters:
        - indicators: A pandas DataFrame containing indicators calculated by the Strategy class.
        """
        self.indicators = strategy.indicators
        self.trades = None
        self.trading_instructions = pd.DataFrame({
            "date": self.indicators.index,
            "instruction" : 0
        })
        self.rsi_high = 80
        self.rsi_low = 20
        self.dca_investments = None

    def simulate_trades(self, rsi_low: int = 20, rsi_high: int = 80) -> pd.DataFrame:
        """
        Simulate trades based on RSI and moving average signals and calculate performance metrics.
        Signals are generated only when the moving average crosses over or under another.
        
        Parameters:
        - rsi_low: RSI threshold for oversold (long entry).
        - rsi_high: RSI threshold for overbought (short entry).
        """
        #self.indicators['long_signal'] = 0
        #self.indicators['short_signal'] = 0

        if rsi_low is None:
            rsi_low = self.rsi_low
        else:
            self.rsi_low = rsi_low

        if rsi_high is None:
            rsi_high = self.rsi_high
        else:
            self.rsi_high = rsi_high
        
        #for i in range(1, len(self.indicators)):
            # Buy signal when RSI < rsi_low and short_ma crosses above medium_ma (if available)
            #if (self.indicators['RSI'].iloc[i] < rsi_low #and 
                #self.indicators['short_ma'].iloc[i] > self.indicators.get('medium_ma', self.indicators['short_ma']).iloc[i] and 
                #self.indicators['short_ma'].iloc[i - 1] <= self.indicators.get('medium_ma', self.indicators['short_ma']).iloc[i - 1]
            #    ):
            #    self.indicators['long_signal'].iloc[i] = 1
                
        self.indicators = self.indicators.assign(
            long_signal=keep_first_signal(np.where(self.indicators['RSI'] < rsi_low, 1, 0))
        )

            

        # Sell signal when RSI > rsi_high and short_ma crosses below medium_ma (if available)
        #if (self.indicators['RSI'].iloc[i] > rsi_high #and 
            #self.indicators['short_ma'].iloc[i] < self.indicators.get('medium_ma', self.indicators['short_ma']).iloc[i] and 
            #self.indicators['short_ma'].iloc[i - 1] >= self.indicators.get('medium_ma', self.indicators['short_ma']).iloc[i - 1]
        #    ):
            
        #    self.indicators['short_signal'].iloc[i] = 1

        self.indicators = self.indicators.assign(
            short_signal=keep_first_signal(np.where(self.indicators['RSI'] > rsi_high, 1, 0))
        )
        
        # Define position based on the signal
        self.indicators = self.indicators.assign(
            position=np.where(self.indicators['long_signal'] == 1, 1,
                            np.where(self.indicators['short_signal'] == 1, -1, 0))
        )
        self.indicators['position'] = self.indicators['position'].shift().fillna(0)

        # Calculate daily returns
        self.indicators = self.indicators.assign(daily_return=self.indicators['Adj Close'].pct_change())
        self.indicators = self.indicators.assign(strategy_return=self.indicators['daily_return'] * self.indicators['position'])

        # Calculate cumulative returns
        self.indicators = self.indicators.assign(cumulative_return=(1 + self.indicators['strategy_return']).cumprod())

        self.trades = self.indicators
        return self.trades
    
    def generate_trading_instructions(self):
        self.trading_instructions = self.trading_instructions.assign(
            date=np.where(self.indicators['short_signal'] == 1, self.indicators.index,
                          np.where(self.indicators['long_signal'] == 1, self.indicators.index, np.datetime64('NaT')))
        )
        self.trading_instructions = self.trading_instructions.assign(
            instruction=np.where(self.indicators['short_signal'] == 1, "Short position advised",
                          np.where(self.indicators['long_signal'] == 1, "Long Position advised", 0))
        )

        return self.trading_instructions

    def get_trades(self) -> pd.DataFrame:
        """
        Return the data with indicators.
        """
        return self.trades

    def plot_strategy(self):
        """
        Plot the trading strategy performance using Plotly. It returns two graphs:
        - Price and signals graph
        - RSI and overbought/oversold levels graph
        """
        if self.trades is None:
            raise ValueError("No trades simulated. Run simulate_trades() first.")

        # First graph: Price, signals, and moving averages
        fig1 = go.Figure()

        # Plot adjusted close price
        fig1.add_trace(go.Scatter(
            x=self.trades.index, 
            y=self.trades['Adj Close'], 
            mode='lines', 
            name='Adjusted Close Price', 
            line=dict(color='blue', width=2)
        ))

        # Plot buy and sell signals
        long_signals = self.trades[self.trades['long_signal'] == 1]
        short_signals = self.trades[self.trades['short_signal'] == 1]

        fig1.add_trace(go.Scatter(
            x=long_signals.index, 
            y=long_signals['Adj Close'], 
            mode='markers', 
            name='Buy Signal', 
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))

        fig1.add_trace(go.Scatter(
            x=short_signals.index, 
            y=short_signals['Adj Close'], 
            mode='markers', 
            name='Sell Signal', 
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))

        # Plot moving averages if available
        for ma_col in ['short_ma', 'medium_ma', 'long_ma1', 'long_ma2']:
            if ma_col in self.trades.columns:
                fig1.add_trace(go.Scatter(
                    x=self.trades.index, 
                    y=self.trades[ma_col], 
                    mode='lines', 
                    name=ma_col, 
                    line=dict(dash='dash')
                ))

        fig1.add_trace(go.Scatter(
            x=self.indicators.index,
            y=self.indicators['cumulative_return'],
            mode='lines',
            name='Cumulative Strategy Return',
            line=dict(color='green', width=2, dash='dash')
        ))

        fig1.update_layout(
            title='Trading Strategy Performance',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white'
        )

        # Second graph: RSI and oversold/overbought levels
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=self.trades.index,
            y=self.trades['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ))

        fig2.add_shape(type='line', x0=self.trades.index.min(), y0=self.rsi_high, x1=self.trades.index.max(), y1=self.rsi_high,
                       line=dict(color='red', dash='dash'))
        fig2.add_shape(type='line', x0=self.trades.index.min(), y0=self.rsi_low, x1=self.trades.index.max(), y1=self.rsi_low,
                       line=dict(color='green', dash='dash'))

        fig2.update_layout(
            title='RSI and Overbought/Oversold Levels',
            xaxis_title='Date',
            yaxis_title='RSI',
            template='plotly_white'
        )

        # Show both graphs
        #fig1.show()
        #fig2.show()
        return fig1, fig2

    def simulate_dca(self, monthly_amount: float = 1000.0) -> pd.DataFrame:
        """
        Simulate a Dollar Cost Averaging (DCA) strategy where a fixed amount is invested every month.

        Parameters:
        - monthly_amount: The fixed amount to invest at the start of each month.
        """
        # Ensure the data is sorted by date
        self.indicators = self.indicators.sort_index()

        # Create columns to store the DCA positions and value over time
        self.indicators['DCA_Units'] = 0.0
        self.indicators['DCA_Total_Investment'] = 0.0
        self.indicators['DCA_Portfolio_Value'] = 0.0

        # Track total units purchased and total investment
        total_units = 0.0
        total_investment = 0.0

        # Iterate over each row (each trading day)
        for i in range(len(self.indicators)):
            date = self.indicators.index[i]
            price = self.indicators['Adj Close'].iloc[i]

            # Check if it's the first trading day of the month
            if i == 0 or date.month != self.indicators.index[i - 1].month:
                # Calculate the units bought with the fixed amount
                units_bought = monthly_amount / price
                total_units += units_bought
                total_investment += monthly_amount

            # Update the columns with the current DCA stats
            self.indicators.at[date, 'DCA_Units'] = total_units
            self.indicators.at[date, 'DCA_Total_Investment'] = total_investment
            self.indicators.at[date, 'DCA_Portfolio_Value'] = total_units * price

        self.dca_investments = self.indicators
        return self.dca_investments

    def plot_dca_strategy(self):
        """
        Plot the DCA strategy performance using Plotly. It shows:
        - Adjusted Close price over time.
        - Portfolio value over time using the DCA strategy.
        - Total investment over time using the DCA strategy.
        """
        if self.dca_investments is None:
            raise ValueError("No DCA investments simulated. Run simulate_dca() first.")

        fig = go.Figure()

        # Plot adjusted close price
        fig.add_trace(go.Scatter(
            x=self.dca_investments.index, 
            y=self.dca_investments['Adj Close'], 
            mode='lines', 
            name='Adjusted Close Price', 
            line=dict(color='blue', width=2)
        ))

        # Plot total investment
        fig.add_trace(go.Scatter(
            x=self.dca_investments.index, 
            y=self.dca_investments['DCA_Total_Investment'], 
            mode='lines', 
            name='Total Investment', 
            line=dict(color='orange', width=2, dash='dash')
        ))

        # Plot DCA portfolio value
        fig.add_trace(go.Scatter(
            x=self.dca_investments.index, 
            y=self.dca_investments['DCA_Portfolio_Value'], 
            mode='lines', 
            name='DCA Portfolio Value', 
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title='DCA Strategy Performance',
            xaxis_title='Date',
            yaxis_title='Value (EUR)',
            template='plotly_white'
        )

        #fig.show()
        return fig
    
    def plot_combined_strategy(self) -> go.Figure:
        """
        Create a combined plot for trading strategy performance, RSI levels, and DCA.

        Returns:
        - A Plotly figure object.
        """
        # Create a subplot grid for the combined figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Trading Strategy Performance", "RSI & Overbought/Oversold Levels", "DCA Strategy Performance"),
            row_heights=[0.4, 0.3, 0.3]
        )

        # Trading strategy performance
        fig.add_trace(go.Scatter(
            x=self.indicators.index,
            y=self.indicators['Adj Close'],
            mode='lines',
            name='Adjusted Close Price',
            line=dict(color='blue', width=2)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.indicators.index,
            y=self.indicators['cumulative_return'],
            mode='lines',
            name='Cumulative Strategy Return',
            line=dict(color='green', width=2, dash='dash')
        ), row=1, col=1)

        # RSI and Overbought/Oversold levels
        fig.add_trace(go.Scatter(
            x=self.indicators.index,
            y=self.indicators['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2)
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=self.indicators.index,
            y=[80] * len(self.indicators),
            mode='lines',
            name='Overbought Level (80)',
            line=dict(color='red', width=1, dash='dot')
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=self.indicators.index,
            y=[20] * len(self.indicators),
            mode='lines',
            name='Oversold Level (20)',
            line=dict(color='green', width=1, dash='dot')
        ), row=2, col=1)

        # DCA Strategy performance
        if self.dca_investments is not None:
            fig.add_trace(go.Scatter(
                x=self.dca_investments.index,
                y=self.dca_investments['DCA_Total_Investment'],
                mode='lines',
                name='Total Investment',
                line=dict(color='orange', width=2, dash='dash')
            ), row=3, col=1)

            fig.add_trace(go.Scatter(
                x=self.dca_investments.index,
                y=self.dca_investments['DCA_Portfolio_Value'],
                mode='lines',
                name='DCA Portfolio Value',
                line=dict(color='green', width=2)
            ), row=3, col=1)

        fig.update_layout(
            height=1000,
            title='Trading Strategy, RSI Levels, and DCA Performance',
            xaxis_title='Date',
            yaxis_title='Value (USD)',
            template='plotly_white'
        )

        return fig

    def create_table(self) -> go.Figure:
        """
        Create a table summarizing key indicators.

        Returns:
        - A Plotly figure object.
        """
        table_data = self.indicators[['Adj Close', 'RSI', 'cumulative_return']].tail(10)  # Last 10 rows for simplicity

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(table_data.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[table_data[col] for col in table_data.columns],
                       fill_color='lavender',
                       align='left'))
        ])

        fig.update_layout(title="Summary Table of Indicators (Last 10 Rows)")
        return fig

# Example usage
#data_loader = DataLoader(csv_file='./ALO.PA.csv', min_date="2023-11-01")
data_loader = DataLoader(csv_file='./data/quote/WPEA.PA.csv', min_date="2024-05-01")
indicators = data_loader.get_data()

strategy = Strategy(indicators)
strategy.calculate_moving_averages()
strategy.calculate_rsi()

trading_strategy = TradingStrategy(strategy)
trading_strategy.simulate_trades(rsi_low=20, rsi_high=80)
ti = trading_strategy.generate_trading_instructions()
trading_strategy.simulate_dca(50)
trades = trading_strategy.get_trades()

trading_fig, rsi_fig = trading_strategy.plot_strategy()
dca_fig = trading_strategy.plot_dca_strategy()
table_pane = trading_strategy.create_table()

def run(ticker=None, rsi_strat=None, dca_amount=50):
    if ticker == None:
        #data_loader = DataLoader(csv_file='./ALO.PA.csv', min_date="2023-11-01")
        data_loader = DataLoader(csv_file='./data/quote/WPEA.PA.csv', min_date="2024-05-01")
        indicators = data_loader.get_data()
    else:
        data_loader = DataLoader(csv_file=f'./data/quote/{ticker}.csv', min_date="2023-05-01")
        indicators = data_loader.get_data()

    strategy = Strategy(indicators)
    strategy.calculate_moving_averages()
    strategy.calculate_rsi()

    trading_strategy = TradingStrategy(strategy)

    if rsi_strat == "80/20":
        trading_strategy.simulate_trades(rsi_low=20, rsi_high=80)
    elif rsi_strat == "70/30":
        trading_strategy.simulate_trades(rsi_low=30, rsi_high=70)
    else:
        trading_strategy.simulate_trades(rsi_low=30, rsi_high=70)

    ti = trading_strategy.generate_trading_instructions()
    trading_strategy.simulate_dca(dca_amount)

    trading_fig, rsi_fig = trading_strategy.plot_strategy()
    dca_fig = trading_strategy.plot_dca_strategy()
    table_pane = trading_strategy.create_table()

    return trading_fig, rsi_fig, dca_fig, table_pane


# Display Streamlit

st.title("Portfolio Management")

with st.sidebar:
    option_ticker = st.selectbox('Ticker', [x.replace('.csv', '') for x in os.listdir('./data/quote')])
    st.write(option_ticker)
    option_rsi = st.selectbox('RSI Strategy', ["80/20", "70/30"])
    st.write(option_rsi)
    dca_amount = st.number_input("Insert a DCA Amount")
    st.write(dca_amount)
    trading_fig, rsi_fig, dca_fig, table_pane = run(ticker=str(option_ticker), rsi_strat=str(option_rsi), dca_amount = dca_amount)
    print(option_ticker, option_rsi)

c1 = st.container()
col1, col2 = c1.columns(2)
with col1 :
    col1_1 = st.container()
    col1_1.plotly_chart(trading_fig)
    col1_2 = st.container()
    st.plotly_chart(dca_fig)
with col2:
    col2_1 = st.container()
    col2_1.plotly_chart(rsi_fig)
    col2_2 = st.container()
    col2_2.plotly_chart(table_pane)
