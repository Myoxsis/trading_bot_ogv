import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import datetime

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

    def simulate_trades(self, rsi_low: int = None, rsi_high: int = None) -> pd.DataFrame:
        """
        Simulate trades based on RSI and moving average signals and calculate performance metrics.

        Parameters:
        - rsi_low: RSI threshold for oversold (long entry).
        - rsi_high: RSI threshold for overbought (short entry).
        """

        if rsi_low is None:
            rsi_low = self.rsi_low
        else:
            self.rsi_low = rsi_low

        if rsi_high is None:
            rsi_high = self.rsi_high
        else:
            self.rsi_high = rsi_high
        

        self.indicators = self.indicators.assign(
            long_signal=np.where(
                (self.indicators['RSI'] < rsi_low) & (self.indicators['short_ma'] > self.indicators.get('long_ma1', self.indicators['short_ma'])), 1, 0
            ),
            short_signal=np.where(
                (self.indicators['RSI'] > rsi_high) & (self.indicators['short_ma'] < self.indicators.get('long_ma1', self.indicators['short_ma'])), 1, 0
            )
        )

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

    def simulate_trades2(self, rsi_low: int = 20, rsi_high: int = 80) -> pd.DataFrame:
        """
        Simulate trades based on RSI and moving average signals and calculate performance metrics.

        Parameters:
        - rsi_low: RSI threshold for oversold (long entry).
        - rsi_high: RSI threshold for overbought (short entry).
        """
        long_signal = []
        short_signal = []
        in_long_position = False
        in_short_position = False

        for i in range(len(self.indicators)):
            rsi = self.indicators['RSI'].iloc[i]
            short_ma = self.indicators['short_ma'].iloc[i]
            medium_ma = self.indicators.get('medium_ma', self.indicators['short_ma']).iloc[i]

            # Long Signal
            if not in_long_position and rsi < rsi_low and short_ma > medium_ma:
                long_signal.append(1)
                short_signal.append(0)
                in_long_position = True
                in_short_position = False
            # Short Signal
            elif not in_short_position and rsi > rsi_high and short_ma < medium_ma:
                short_signal.append(1)
                long_signal.append(0)
                in_short_position = True
                in_long_position = False
            else:
                long_signal.append(0)
                short_signal.append(0)

        self.indicators['long_signal'] = long_signal
        self.indicators['short_signal'] = short_signal

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

    def simulate_trades3(self, rsi_low: int = 20, rsi_high: int = 80) -> pd.DataFrame:
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
        fig1.show()
        fig2.show()
        return fig1, fig2

# Example usage
data_loader = DataLoader(csv_file='./ALO.PA.csv', min_date="2023-01-05")
indicators = data_loader.get_data()

strategy = Strategy(indicators)
strategy.calculate_moving_averages()
strategy.calculate_rsi()
#strategy.get_indicators()

trading_strategy = TradingStrategy(strategy)
trading_strategy.simulate_trades3(rsi_low=20, rsi_high=80)
ti = trading_strategy.generate_trading_instructions()
fig1, fig2 = trading_strategy.plot_strategy()
trades = trading_strategy.get_trades()


with open('Daily Update Report.html', 'w') as f:
    f.write("""
    <html>
    <head>
        <title>Daily Report Trades</title>
    </head>
    <body>
        <div class="banner">
            <h1 style="text-align: center;">Usage / Options report for stock movements</h1>
            <h2 style="text-align: center;font-size:14px;color:grey;">Based on Portfolio and data maintained in DCA Follow Up program</h2>
        </div>
        <div class="last_update">Last Update : """)
    f.write(str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    
    f.write("""</div>
            <div class="wrapper">
	<div class="one">""")
    f.write(fig1.to_html(full_html=False, default_height='50%', default_width='40%'))
    f.write("""</div> <div class="two">""")
    f.write(fig2.to_html(full_html=False, default_height='50%', default_width='40%'))

    f.write("""
    </div>
    </div class="three">
    <div>Trading Instruction</div></br>
    <p>Please find below the latest trading instructions : 
    """)
    ti = ti[ti['date'].apply(lambda x :str(x)) != str('NaT')]
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(ti.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=ti.transpose().values.tolist(),
                fill_color='lavender',
                align='left'))
    ])
    f.write(fig.to_html(full_html=False, default_height='50%', default_width='30%'))

 

    f.write("""
    </p></div>
    <style>
    * {font-family : Arial;}
    body {background-color: white;}
    h1   {color: Black; font-size: 20px;}
    p    {color: red;}
    div.last_update {position : absolute; top:0; right:0;}
	
	.wrapper { display: grid; grid-template-columns: repeat(1, 1fr); grid-gap: 5px; grid-auto-rows: minmax(100px, auto);}	
	.one { grid-column: 1 ; grid-row: 1/3; background-color: none;}
    .three { grid-column: 3 ; grid-row: 1/3; background-color: none;}
	.two { grid-column: 1 ; grid-row: 3/3; background-color: none;}
    .two > div {right: 5;}   
    
             
    </style></body>
    </html>""")