#%%
import csv
import pandas as pd
from datetime import datetime, timedelta

#%%

class Asset:
    def __init__(self, symbol: str, long_name: str, asset_type: str, price: float, quantity: int):
        """
        Initializes an Asset instance.

        :param symbol: The symbol of the asset (e.g., stock ticker).
        :param long_name: The long name of the asset.
        :param asset_type: The type of the asset (e.g., stock, bond, etc.).
        :param price: Current price of the asset.
        :param quantity: Quantity of the asset owned.
        """
        self.symbol = symbol
        self.long_name = long_name
        self.asset_type = asset_type
        self.price = price
        self.quantity = quantity

    def get_value(self) -> float:
        """
        Returns the total value of the asset based on its price and quantity.

        :return: Total value of the asset.
        """
        return self.price * self.quantity

    def update_price(self, new_price: float):
        """
        Updates the current price of the asset.

        :param new_price: The new price to set for the asset.
        """
        self.price = new_price

    def display_info(self) -> str:
        """
        Displays information about the asset.

        :return: A string containing the asset's information.
        """
        return (f"Asset Symbol: {self.symbol}, Long Name: {self.long_name}, "
                f"Type: {self.asset_type}, Price: {self.price:.2f}, "
                f"Quantity: {self.quantity}, Total Value: {self.get_value():.2f}")
    

class Order:
    def __init__(self, asset: Asset):
        """
        Initializes an Order instance with a specified asset.

        :param asset: The Asset object associated with this order.
        """
        self.asset = asset
        self.order_type = None
        self.quantity = 0

    def create_order(self, order_type: str, quantity: int):
        """
        Creates an order by specifying the type and quantity.

        :param order_type: The type of order (e.g., "buy" or "sell").
        :param quantity: The quantity of the asset to buy or sell.
        """
        if order_type.lower() not in ["buy", "sell"]:
            raise ValueError("Order type must be 'buy' or 'sell'.")
        if quantity <= 0:
            raise ValueError("Qauntity must be a positive integer.")
        
        self.order_type = order_type.lower()
        self.quantity = quantity

    def submit_order(self) -> str:
        """
        Executes the order and returns the result.

        :return: A string indicating the order execution result.
        """
        if self.order_type is None:
            return "Order not created. Please create an order before submitting it."

        if self.order_type == "buy":
            # Logic for executing a buy order
            total_cost = self.asset.price * self.quantity
            return f'Bought {self.quantity} shares of {self.asset.long_name} at {self.asset.price} each for a total of {total_cost}.'
        
        elif self.order_type == "sell":
            # Logic for executing a sell order
            total_revenue = self.asset.price * self.quantity
            return f'Sold {self.quantity} shares of {self.asset.long_name} at {self.asset.price} each for a total of {total_revenue}.'

    def display_order_info(self) -> str:
        """
        Displays information about the order.

        :return: A string containing the order's information.
        """
        if self.order_type is None:
            return "Order not created. Please provide order details."

        return (f"Order Type: {self.order_type.capitalize()}, "
                f"Asset: {self.asset.long_name} ({self.asset.symbol}), "
                f"Quantity: {self.quantity}")
    
class Strategy:
    def __init__(self, start_date: datetime, end_date: datetime):
        """
        Initializes a Strategy instance.

        :param start_date: The start date of the strategy.
        :param end_date: The end date of the strategy.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.day_increment = 1  # Default increment for iterating days

    def initialize(self, day_increment: int):
        """
        Initializes the strategy with a specified day increment.

        :param day_increment: The number of days to move between iterations.
        """
        self.day_increment = day_increment

    def run(self):
        """
        Runs the trading strategy from start_date to end_date, 
        iterating over each specified day.
        """
        current_date = self.start_date

        while current_date <= self.end_date:
            self.execute_strategy(current_date)
            current_date += timedelta(days=self.day_increment)

    def execute_strategy(self, current_date: datetime):
        """
        A placeholder for the strategy execution logic on a given day.

        :param current_date: The date to execute the strategy on.
        """
        # Implement your trading logic here. This can log, analyze,
        # or execute trades based on the current_date.
        print(f"Executing strategy for date: {current_date.strftime('%Y-%m-%d')}")


class QuoteSource:
    def __init__(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        """
        Initializes the QuoteSource instance by loading quotes from a CSV file.

        :param csv_file: Path to the CSV file containing quotes.
        :param date_col: The name of the column containing the date.
        :param symbol_col: The name of the column containing the asset symbol.
        :param price_col: The name of the column containing the asset price.
        """
        self.quotes = {}
        self.load_quotes(csv_file, date_col, symbol_col, price_col)

    def load_quotes(self, csv_file: str, date_col: str, symbol_col: None, price_col: str):
        """
        Loads quotes from the specified CSV file into a dictionary.

        :param csv_file: Path to the CSV file containing quotes.
        :param date_col: The name of the column containing the date.
        :param symbol_col: The name of the column containing the asset symbol.
        :param price_col: The name of the column containing the asset price.
        """
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            date = pd.to_datetime(row[date_col]).date()
            symbol = symbol_col
            # symbol = row[symbol_col]
            price = row[price_col]

            if symbol not in self.quotes:
                self.quotes[symbol] = {}

            self.quotes[symbol][date] = float(price)

    def get_quote(self, symbol: str, date: datetime) -> float:
        """
        Retrieves the quote for a given symbol on a specific date.

        :param symbol: The symbol to look up.
        :param date: The date to look up the quote.
        :return: The price of the asset on the given date, or None if not found.
        """
        return self.quotes.get(symbol, {}).get(date, None)
    
class TradingFees:
    def __init__(self, flat_fee: float = 0.0, percent_fee: float = 0.0):
        """
        Initializes the TradingFees instance.

        :param flat_fee: A flat fee for each order.
        :param percent_fee: A percentage fee of the order value.
        """
        self.flat_fee = flat_fee
        self.percent_fee = percent_fee

    def calculate_fee(self, order_value: float) -> float:
        """
        Calculates the total fees for the given order value.

        :param order_value: The value of the order.
        :return: Total fees for the order.
        """
        return self.flat_fee + (self.percent_fee / 100 * order_value)
    

class Trader:
    def __init__(self, quote_source: QuoteSource, trading_fees: TradingFees = None):
        """
        Initializes the Trader instance.

        :param quote_source: An instance of QuoteSource to retrieve quotes.
        :param trading_fees: Optional TradingFees to apply to orders.
        """
        self.quote_source = quote_source
        self.trading_fees = trading_fees
        self.strategies = []

    def add_strategy(self, strategy: Strategy):
        """
        Adds a strategy to the trader's list of strategies.

        :param strategy: An instance of Strategy to execute.
        """
        self.strategies.append(strategy)

    def execute_strategies(self):
        """
        Executes all strategies assigned to the trader.
        """
        for strategy in self.strategies:
            strategy.run()

    def execute_order(self, order: Order) -> str:
        """
        Executes a given order, applying trading fees.

        :param order: The Order object to execute.
        :return: A string indicating the result of the order execution.
        """
        quote = self.quote_source.get_quote(order.asset.symbol, order.asset.price)
        if quote is None:
            return f"No quote available for {order.asset.symbol} on the given date."

        order_value = quote * order.quantity
        fee = self.trading_fees.calculate_fee(order_value) if self.trading_fees else 0
        total_value_after_fee = order_value - fee
        
        if order.order_type == "buy":
            return f'Bought {order.quantity} shares of {order.asset.long_name} at {quote} each (Fee: {fee}). Total value after fees: {total_value_after_fee}.'
        elif order.order_type == "sell":
            return f'Sold {order.quantity} shares of {order.asset.long_name} at {quote} each (Fee: {fee}). Total value after fees: {total_value_after_fee}.'





#%%%

# TEST ================================================================================><===============


import pandas as pd
from datetime import datetime, timedelta

class Asset:
    def __init__(self, symbol: str, long_name: str, asset_type: str, price: float, quantity: int):
        self.symbol = symbol
        self.long_name = long_name
        self.asset_type = asset_type
        self.price = price
        self.quantity = quantity

    def get_value(self) -> float:
        return self.price * self.quantity

    def update_price(self, new_price: float):
        self.price = new_price

    def display_info(self) -> str:
        return (f"Asset Symbol: {self.symbol}, Long Name: {self.long_name}, "
                f"Type: {self.asset_type}, Price: {self.price:.2f}, "
                f"Quantity: {self.quantity}, Total Value: {self.get_value():.2f}")


class Order:
    def __init__(self, asset: Asset):
        self.asset = asset
        self.order_type = None
        self.quantity = 0

    def create_order(self, order_type: str, quantity: int):
        if order_type.lower() not in ["buy", "sell"]:
            raise ValueError("Order type must be 'buy' or 'sell'.")
        if quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        
        self.order_type = order_type.lower()
        self.quantity = quantity

    def submit_order(self) -> str:
        if self.order_type is None:
            return "Order not created. Please create an order before submitting it."

        total_amount = self.asset.price * self.quantity
        if self.order_type == "buy":
            return f'Bought {self.quantity} shares of {self.asset.long_name} at {self.asset.price:.2f} each for a total of {total_amount:.2f}.'
        
        elif self.order_type == "sell":
            return f'Sold {self.quantity} shares of {self.asset.long_name} at {self.asset.price:.2f} each for a total of {total_amount:.2f}.'

    def display_order_info(self) -> str:
        if self.order_type is None:
            return "Order not created. Please provide order details."
        return (f"Order Type: {self.order_type.capitalize()}, "
                f"Asset: {self.asset.long_name} ({self.asset.symbol}), "
                f"Quantity: {self.quantity}")


class Strategy:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.day_increment = 1

    def initialize(self, day_increment: int):
        self.day_increment = day_increment

    def run(self):
        current_date = self.start_date
        while current_date <= self.end_date:
            self.execute_strategy(current_date)
            current_date += timedelta(days=self.day_increment)

    def execute_strategy(self, current_date: datetime):
        print(f"Executing strategy for date: {current_date.strftime('%Y-%m-%d')}")


class QuoteSource:
    def __init__(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        self.quotes = {}
        self.load_quotes(csv_file, date_col, symbol_col, price_col)

    def load_quotes(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            date = pd.to_datetime(row[date_col]).date()
            symbol = symbol_col
            #symbol = row[symbol_col]
            price = row[price_col]

            if symbol not in self.quotes:
                self.quotes[symbol] = {}

            self.quotes[symbol][date] = float(price)

    def get_quote(self, symbol: str, date: datetime) -> float:
        return self.quotes.get(symbol, {}).get(date, None)


class TradingFees:
    def __init__(self, flat_fee: float = 0.0, percent_fee: float = 0.0):
        self.flat_fee = flat_fee
        self.percent_fee = percent_fee

    def calculate_fee(self, order_value: float) -> float:
        return self.flat_fee + (self.percent_fee / 100 * order_value)


class Trader:
    def __init__(self, quote_source: QuoteSource, trading_fees: TradingFees = None):
        self.quote_source = quote_source
        self.trading_fees = trading_fees
        self.strategies = []

    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy)

    def execute_strategies(self):
        for strategy in self.strategies:
            strategy.run()

    def execute_order(self, order: Order, current_date: datetime) -> str:
        quote = self.quote_source.get_quote(order.asset.symbol, current_date)
        if quote is None:
            return f"No quote available for {order.asset.symbol} on {current_date.strftime('%Y-%m-%d')}."

        order_value = quote * order.quantity
        fee = self.trading_fees.calculate_fee(order_value) if self.trading_fees else 0
        total_value_after_fee = order_value - fee

        if order.order_type == "buy":
            return (f'Bought {order.quantity} shares of {order.asset.long_name} at {quote:.2f} each (Fee: {fee:.2f}). '
                    f'Total value after fees: {total_value_after_fee:.2f}.')
        elif order.order_type == "sell":
            return (f'Sold {order.quantity} shares of {order.asset.long_name} at {quote:.2f} each (Fee: {fee:.2f}). '
                    f'Total value after fees: {total_value_after_fee:.2f}.')

    

#%%


# Create an instance of Asset
my_asset = Asset("ALO.PA", "Alstom SA", "Stock", 50.0, 10)

# Create an order instance linked to the asset
order = Order(my_asset)

# Create a buy order
order.create_order("buy", 5)

# Display order information
print(order.display_order_info())

# Submit the buy order
print(order.submit_order())

# Create a sell order
order.create_order("sell", 3)

# Submit the buy order
print(order.submit_order())

# Create a sell order
order.create_order("sell", 3)

# Display order information
print(order.display_order_info())

# Submit the sell order
print(order.submit_order())

#%%

# Define the start and end dates for the strategy
start = datetime(2023, 1, 1)
end = datetime(2023, 1, 10)

# Create an instance of Strategy
trading_strategy = Strategy(start, end)

# Initialize the strategy with a day increment of 2 (e.g., every 2 days)
trading_strategy.initialize(day_increment=1)

# Run the strategy
trading_strategy.run()

#%%

# Initialize QuoteSource with a CSV file containing quotes
quote_source = QuoteSource('ALO.PA.csv', date_col='Date', price_col="Adj Close", symbol_col="ALO.PA")

# Initialize TradingFees
trading_fees = TradingFees(flat_fee=1.0, percent_fee=0.5)

# Create an instance of Trader
trader = Trader(quote_source, trading_fees)

# Create an asset
my_asset = Asset("ALO.PA", "Alstom SA", "Stock", 50.0, 10)

# Create an order for buying the asset
buy_order = Order.create_order(my_asset, "buy", 5)

# Add strategies and execute them
strategy = Strategy(datetime(2023, 1, 1), datetime(2023, 1, 10))
trader.add_strategy(strategy)

# Execute all strategies
trader.execute_strategies()

# Execute a buy order
print(trader.execute_order(buy_order))

# Create an order for selling the asset and execute
sell_order = Order(my_asset, "sell", 3)
print(trader.execute_order(sell_order))

#%%

print(quote_source.quotes)

#%%


print(quote_source.get_quote("ALO.PA", datetime.date(2023, 8, 24)))


#%%


import random
import pandas as pd

class Order:
    def __init__(self, action, stock, quantity, price):
        self.action = action
        self.stock = stock
        self.quantity = quantity
        self.price = price

class QuoteSource:
    def __init__(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        self.quotes = {}
        self.load_quotes(csv_file, date_col, symbol_col, price_col)

    def load_quotes(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            date = pd.to_datetime(row[date_col]).date()
            symbol = row[symbol_col]
            price = row[price_col]

            if symbol not in self.quotes:
                self.quotes[symbol] = {}

            self.quotes[symbol][date] = float(price)

    def get_quote(self, stock, date):
        return self.quotes.get(stock, {}).get(date, None)

class Strategy:
    def __init__(self):
        self.portfolio = {"cash": 10000, "stocks": {}}
        self.transaction_history = []

    def buy_stock(self, stock, quantity, stock_price):
        total_cost = stock_price * quantity
        if self.check_funds_for_purchase(total_cost):
            self.portfolio["cash"] -= total_cost
            self.portfolio["stocks"][stock] = self.portfolio["stocks"].get(stock, 0) + quantity
            self.log_transaction("buy", stock, quantity, stock_price)
            print(f"Successfully purchased {quantity} shares of {stock} at ${stock_price:.2f} per share.")
        else:
            print(f"Insufficient funds to buy {quantity} shares of {stock}.")

    def sell_stock(self, stock, quantity, stock_price):
        if self.portfolio["stocks"].get(stock, 0) >= quantity:
            total_sale_value = stock_price * quantity
            self.portfolio["cash"] += total_sale_value
            self.portfolio["stocks"][stock] -= quantity

            if self.portfolio["stocks"][stock] == 0:
                del self.portfolio["stocks"][stock]

            self.log_transaction("sell", stock, quantity, stock_price)
            print(f"Successfully sold {quantity} shares of {stock} at ${stock_price:.2f} per share.")
        else:
            print(f"Insufficient shares of {stock} to sell.")

    def check_funds_for_purchase(self, total_cost):
        return self.portfolio["cash"] >= total_cost

    def log_transaction(self, action, stock, quantity, price):
        order = Order(action, stock, quantity, price)
        self.transaction_history.append(order)

    def view_portfolio(self):
        print("\nPortfolio:")
        print(f"Cash: ${self.portfolio['cash']:.2f}")
        for stock, quantity in self.portfolio["stocks"].items():
            print(f"{stock}: {quantity} shares")

    def view_transaction_history(self):
        print("\nTransaction History:")
        for transaction in self.transaction_history:
            print(f"{transaction.action.capitalize()} {transaction.quantity} shares of {transaction.stock} at ${transaction.price:.2f}")

class StockSimulation:
    def __init__(self, quote_source: QuoteSource):
        self.strategy = Strategy()
        self.quote_source = quote_source

    def run_simulation(self, days=90, transaction_interval=7):
        for day in range(1, days + 1):
            print(f"\nDay {day}")
            current_date = pd.Timestamp.now().date() + pd.DateOffset(days=day)  # Simulate specific date
            self.update_stock_prices()

            if day % transaction_interval == 0:
                self.strategy.view_portfolio()
                user_input = input(f"Day {day}: Do you want to buy or sell stocks? (yes/no): ").strip().lower()

                if user_input == 'yes':
                    action = input("Do you want to 'buy' or 'sell'?: ").strip().lower()
                    stock = input("Which stock (AAPL, GOOGL, TSLA)?: ").strip().upper()

                    if stock not in self.quote_source.quotes:
                        print("Invalid stock symbol. Please try again.")
                        continue

                    try:
                        quantity = int(input("How many shares?: "))
                        stock_price = self.quote_source.get_quote(stock, current_date)
                        if action == "buy":
                            self.strategy.buy_stock(stock, quantity, stock_price)
                        elif action == "sell":
                            self.strategy.sell_stock(stock, quantity, stock_price)
                    except ValueError:
                        print("Invalid quantity, please enter a valid integer.")

            self.strategy.view_portfolio()

        print("\n--- End of Simulation ---")
        self.strategy.view_portfolio()
        self.strategy.view_transaction_history()

    def update_stock_prices(self):
        # Randomly update stock prices for simulation if needed
        # This method is intentionally left empty for dynamic price handling from CSV
        pass

# Example usage, ensure to specify your CSV file path correctly
quote_source = QuoteSource('ALO.PA.csv', date_col='Date', symbol_col='symbol', price_col='Adj Close')
simulation = StockSimulation(quote_source)

simulation.run_simulation()


#%%



class Asset:
    def __init__(self, symbol: str, long_name: str, asset_type: str, price: float, quantity: int):
        self.symbol = symbol
        self.long_name = long_name
        self.asset_type = asset_type
        self.price = price
        self.quantity = quantity

    def get_value(self) -> float:
        return self.price * self.quantity

    def update_price(self, new_price: float):
        self.price = new_price

    def display_info(self) -> str:
        return (f"Asset Symbol: {self.symbol}, Long Name: {self.long_name}, "
                f"Type: {self.asset_type}, Price: {self.price:.2f}, "
                f"Quantity: {self.quantity}, Total Value: {self.get_value():.2f}")
    
class Order:
    def __init__(self, asset: Asset):
        self.asset = asset
        self.order_type = None
        self.quantity = 0

    def create_order(self, order_type: str, quantity: int):
        if order_type.lower() not in ["buy", "sell"]:
            raise ValueError("Order type must be 'buy' or 'sell'.")
        if quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")
        
        self.order_type = order_type.lower()
        self.quantity = quantity

    def submit_order(self) -> str:
        if self.order_type is None:
            return "Order not created. Please create an order before submitting it."

        total_amount = self.asset.price * self.quantity
        if self.order_type == "buy":
            return f'Bought {self.quantity} shares of {self.asset.long_name} at {self.asset.price:.2f} each for a total of {total_amount:.2f}.'
        
        elif self.order_type == "sell":
            return f'Sold {self.quantity} shares of {self.asset.long_name} at {self.asset.price:.2f} each for a total of {total_amount:.2f}.'

    def display_order_info(self) -> str:
        if self.order_type is None:
            return "Order not created. Please provide order details."
        return (f"Order Type: {self.order_type.capitalize()}, "
                f"Asset: {self.asset.long_name} ({self.asset.symbol}), "
                f"Quantity: {self.quantity}")
    
#%%

alo = Asset("ALO.PA", "Alstom", "stock", 10., 3)

print(alo.display_info())

a = Order(alo)
a.create_order("buy", 2)

a.display_order_info()
a.submit_order()

print(alo.display_info())



#%%

import pandas as pd
from datetime import datetime, timedelta

class Asset:
    def __init__(self, symbol: str, long_name: str, asset_type: str):
        self.symbol = symbol
        self.long_name = long_name
        self.asset_type = asset_type

class Position:
    def __init__(self, asset: Asset, price: float, quantity: int):
        self.asset = asset
        self.price = price
        self.quantity = quantity
        self.total_cost = self.price * self.quantity

    def update_price(self, new_price: float):
        self.price = new_price

    def add_shares(self, additional_quantity: int, price: float):
        if additional_quantity <= 0:
            raise ValueError("Additional quantity must be a positive integer.")
        self.quantity += additional_quantity
        self.total_cost += price * additional_quantity

    def get_value(self) -> float:
        return self.price * self.quantity

    def display_info(self) -> str:
        return (f"Asset Symbol: {self.asset.symbol}, Long Name: {self.asset.long_name}, "
                f"Type: {self.asset.asset_type}, Purchase Price: {self.price:.2f}, "
                f"Quantity: {self.quantity}, Total Cost: {self.total_cost:.2f}, "
                f"Current Value: {self.get_value():.2f}")

class Order:
    def __init__(self, position: Position):
        self.position = position
        self.order_type = None
        self.quantity = 0

    def create_order(self, order_type: str, quantity: int):
        if order_type.lower() not in ["buy", "sell"]:
            raise ValueError("Order type must be 'buy' or 'sell'.")
        if quantity <= 0:
            raise ValueError("Quantity must be a positive integer.")

        self.order_type = order_type.lower()
        self.quantity = quantity

    def submit_order(self) -> str:
        if self.order_type is None:
            return "Order not created. Please create an order before submitting it."
        total_amount = self.position.price * self.quantity
        if self.order_type == "buy":
            self.position.add_shares(self.quantity, self.position.price)
            return f'Bought {self.quantity} shares of {self.position.asset.long_name} at {self.position.price:.2f} each for a total of {total_amount:.2f}.'
        elif self.order_type == "sell":
            if self.quantity > self.position.quantity:
                return "Cannot sell more shares than owned."
            self.position.quantity -= self.quantity
            return f'Sold {self.quantity} shares of {self.position.asset.long_name} at {self.position.price:.2f} each for a total of {total_amount:.2f}.'

    def display_order_info(self) -> str:
        if self.order_type is None:
            return "Order not created. Please provide order details."
        return (f"Order Type: {self.order_type.capitalize()}, "
                f"Asset: {self.position.asset.long_name} ({self.position.asset.symbol}), "
                f"Quantity: {self.quantity}")

# Rest of the classes remain unchanged

class Strategy:
    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date
        self.day_increment = 1
        self.orders = []  # List to hold associated orders

    def initialize(self, day_increment: int):
        self.day_increment = day_increment

    def add_order(self, order: Order):
        self.orders.append(order)

    def run(self):
        current_date = self.start_date
        while current_date <= self.end_date:
            self.execute_strategy(current_date)
            current_date += timedelta(days=self.day_increment)

    def execute_strategy(self, current_date: datetime):
        print(f"Executing strategy for date: {current_date.strftime('%Y-%m-%d')}")
        for order in self.orders:
            result = order.submit_order()  # Execute each order
            print(result)

class QuoteSource:
    def __init__(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        self.quotes = {}
        self.load_quotes(csv_file, date_col, symbol_col, price_col)

    def load_quotes(self, csv_file: str, date_col: str, symbol_col: str, price_col: str):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            date = pd.to_datetime(row[date_col]).date()
            symbol = row[symbol_col]
            price = row[price_col]
            if symbol not in self.quotes:
                self.quotes[symbol] = {}
            self.quotes[symbol][date] = float(price)

    def get_quote(self, symbol: str, date: datetime) -> float:
        return self.quotes.get(symbol, {}).get(date, None)

class TradingFees:
    def __init__(self, flat_fee: float = 0.0, percent_fee: float = 0.0):
        self.flat_fee = flat_fee
        self.percent_fee = percent_fee

    def calculate_fee(self, order_value: float) -> float:
        return self.flat_fee + (self.percent_fee / 100 * order_value)

class Trader:
    def __init__(self, quote_source: QuoteSource, trading_fees: TradingFees = None):
        self.quote_source = quote_source
        self.trading_fees = trading_fees
        self.strategies = []

    def add_strategy(self, strategy: Strategy):
        self.strategies.append(strategy)

    def execute_strategies(self):
        for strategy in self.strategies:
            strategy.run()

    def execute_order(self, order: Order, current_date: datetime) -> str:
        quote = self.quote_source.get_quote(order.position.asset.symbol, current_date)
        if quote is None:
            return f"No quote available for {order.position.asset.symbol} on {current_date.strftime('%Y-%m-%d')}."
        order_value = quote * order.quantity
        fee = self.trading_fees.calculate_fee(order_value) if self.trading_fees else 0
        total_value_after_fee = order_value - fee
        if order.order_type == "buy":
            return (f'Bought {order.quantity} shares of {order.position.asset.long_name} at {quote:.2f} each (Fee: {fee:.2f}). '
                    f'Total value after fees: {total_value_after_fee:.2f}.')
        elif order.order_type == "sell":
            return (f'Sold {order.quantity} shares of {order.position.asset.long_name} at {quote:.2f} each (Fee: {fee:.2f}). '
                    f'Total value after fees: {total_value_after_fee:.2f}.')
        
#%%

# Step 1: Create Assets
apple_asset = Asset(symbol="AAPL", long_name="Apple Inc.", asset_type="Tech")
tesla_asset = Asset(symbol="TSLA", long_name="Tesla Inc.", asset_type="Auto")

# Step 2: Create Positions
apple_position = Position(asset=apple_asset, price=150.00, quantity=10)
tesla_position = Position(asset=tesla_asset, price=700.00, quantity=5)

# Step 3: Create Orders
buy_apple_order = Order(position=apple_position)
buy_apple_order.create_order(order_type="buy", quantity=5)

sell_tesla_order = Order(position=tesla_position)
sell_tesla_order.create_order(order_type="sell", quantity=2)

# Step 4: Create a Strategy and Add Orders
start_date = datetime.now() - timedelta(days=30)  # Start tracking for the last 30 days
end_date = datetime.now()

investment_strategy = Strategy(start_date=start_date, end_date=end_date)
investment_strategy.add_order(buy_apple_order)
investment_strategy.add_order(sell_tesla_order)

# Step 5: Execute the Strategy
investment_strategy.run()
