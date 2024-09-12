class StockSimulation:
    def __init__(self):
        self.portfolio = self.initialize_portfolio()
        self.stock_prices = self.generate_stock_list()

    def initialize_portfolio(self):
        # Initialize with some cash and no stocks
        return {"cash": 10000, "stocks": {}}

    def view_portfolio(self):
        # Display portfolio holdings
        pass

    def update_portfolio(self, stock, quantity, action):
        # Buy or sell and update portfolio
        pass

    def generate_stock_list(self):
        # Create a list of stocks and initial prices
        return {"AAPL": 150, "GOOGL": 2800, "TSLA": 700}

    def update_stock_prices(self):
        # Simulate random walk or fetch new prices
        pass

    def get_current_stock_price(self, stock):
        # Return the current stock price
        return self.stock_prices[stock]

    def buy_stock(self, stock, quantity):
        # Buy the specified quantity of stock
        pass

    def sell_stock(self, stock, quantity):
        # Sell the specified quantity of stock
        pass

    def log_transaction(self, action, stock, quantity, price):
        # Log the transaction details
        pass

    def calculate_portfolio_value(self):
        # Calculate total value of portfolio
        pass

    def calculate_profit_loss(self):
        # Calculate profit/loss for all stocks
        pass

    def run_simulation(self):
        # Main simulation loop
        pass


#%%

class StockSimulation:
    def __init__(self):
        self.portfolio = self.initialize_portfolio()
        self.stock_prices = self.generate_stock_list()
        self.transaction_history = []

    def initialize_portfolio(self):
        # Initialize with some cash and no stocks
        return {"cash": 10000, "stocks": {}}

    def generate_stock_list(self):
        # Create a list of stocks and their current prices
        return {"AAPL": 150, "GOOGL": 2800, "TSLA": 700}

    def get_current_stock_price(self, stock):
        # Return the current price of the specified stock
        return self.stock_prices[stock]

    def check_funds_for_purchase(self, stock, quantity):
        # Check if the user has enough cash to buy the stock
        stock_price = self.get_current_stock_price(stock)
        total_cost = stock_price * quantity
        return self.portfolio["cash"] >= total_cost

    def buy_stock(self, stock, quantity):
        # Buy the specified quantity of stock if there are enough funds
        stock_price = self.get_current_stock_price(stock)
        total_cost = stock_price * quantity

        if self.check_funds_for_purchase(stock, quantity):
            # Deduct the total cost from available cash
            self.portfolio["cash"] -= total_cost

            # Update portfolio with the new stock purchase
            if stock in self.portfolio["stocks"]:
                self.portfolio["stocks"][stock] += quantity
            else:
                self.portfolio["stocks"][stock] = quantity

            # Log the transaction
            self.log_transaction("buy", stock, quantity, stock_price)

            print(f"Successfully purchased {quantity} shares of {stock} at ${stock_price} per share.")
        else:
            print(f"Insufficient funds to buy {quantity} shares of {stock}.")

    def sell_stock(self, stock, quantity):
        # Check if the user owns the stock and has enough shares to sell
        if stock in self.portfolio["stocks"] and self.portfolio["stocks"][stock] >= quantity:
            # Get the current price of the stock
            stock_price = self.get_current_stock_price(stock)
            
            # Calculate the total sale value
            total_sale_value = stock_price * quantity
            
            # Add the sale value to the portfolio's cash
            self.portfolio["cash"] += total_sale_value
            
            # Reduce the stock quantity in the portfolio
            self.portfolio["stocks"][stock] -= quantity
            
            # If the stock quantity becomes zero, remove it from the portfolio
            if self.portfolio["stocks"][stock] == 0:
                del self.portfolio["stocks"][stock]
            
            # Log the transaction
            self.log_transaction("sell", stock, quantity, stock_price)
            
            print(f"Successfully sold {quantity} shares of {stock} at ${stock_price} per share.")
        else:
            print(f"Insufficient shares of {stock} to sell.")


    def log_transaction(self, action, stock, quantity, price):
        # Log the buy or sell transaction
        transaction = {
            "action": action,
            "stock": stock,
            "quantity": quantity,
            "price": price
        }
        self.transaction_history.append(transaction)

    def view_portfolio(self):
        # Display portfolio holdings and cash
        print("\nPortfolio:")
        print(f"Cash: ${self.portfolio['cash']}")
        for stock, quantity in self.portfolio["stocks"].items():
            print(f"{stock}: {quantity} shares")

    def view_transaction_history(self):
        # Display all transactions
        print("\nTransaction History:")
        for transaction in self.transaction_history:
            print(f"{transaction['action'].capitalize()} {transaction['quantity']} shares of {transaction['stock']} at ${transaction['price']}")

# Create an instance of the simulation
simulation = StockSimulation()

# View portfolio before buying
simulation.view_portfolio()

# Buy 2 shares of AAPL
simulation.buy_stock("AAPL", 2)

# View portfolio after buying
simulation.view_portfolio()

# View transaction history
simulation.view_transaction_history()
#%%

# Buy 2 shares of AAPL
simulation.buy_stock("AAPL", 5)
simulation.buy_stock("TSLA", 5)

simulation.view_portfolio()



#%%


import random

class StockSimulation:
    def __init__(self):
        self.portfolio = self.initialize_portfolio()
        self.stock_prices = self.generate_stock_list()
        self.transaction_history = []

    def initialize_portfolio(self):
        # Initialize with some cash and no stocks
        return {"cash": 10000, "stocks": {}}

    def generate_stock_list(self):
        # Create a list of stocks and their current prices
        return {"AAPL": 150, "GOOGL": 2800, "TSLA": 700}

    def get_current_stock_price(self, stock):
        # Return the current price of the specified stock
        return self.stock_prices[stock]

    def check_funds_for_purchase(self, stock, quantity):
        # Check if the user has enough cash to buy the stock
        stock_price = self.get_current_stock_price(stock)
        total_cost = stock_price * quantity
        return self.portfolio["cash"] >= total_cost

    def buy_stock(self, stock, quantity):
        # Buy the specified quantity of stock if there are enough funds
        stock_price = self.get_current_stock_price(stock)
        total_cost = stock_price * quantity

        if self.check_funds_for_purchase(stock, quantity):
            # Deduct the total cost from available cash
            self.portfolio["cash"] -= total_cost

            # Update portfolio with the new stock purchase
            if stock in self.portfolio["stocks"]:
                self.portfolio["stocks"][stock] += quantity
            else:
                self.portfolio["stocks"][stock] = quantity

            # Log the transaction
            self.log_transaction("buy", stock, quantity, stock_price)

            print(f"Successfully purchased {quantity} shares of {stock} at ${stock_price} per share.")
        else:
            print(f"Insufficient funds to buy {quantity} shares of {stock}.")

    def sell_stock(self, stock, quantity):
        # Check if the user owns the stock and has enough shares to sell
        if stock in self.portfolio["stocks"] and self.portfolio["stocks"][stock] >= quantity:
            # Get the current price of the stock
            stock_price = self.get_current_stock_price(stock)
            
            # Calculate the total sale value
            total_sale_value = stock_price * quantity
            
            # Add the sale value to the portfolio's cash
            self.portfolio["cash"] += total_sale_value
            
            # Reduce the stock quantity in the portfolio
            self.portfolio["stocks"][stock] -= quantity
            
            # If the stock quantity becomes zero, remove it from the portfolio
            if self.portfolio["stocks"][stock] == 0:
                del self.portfolio["stocks"][stock]
            
            # Log the transaction
            self.log_transaction("sell", stock, quantity, stock_price)
            
            print(f"Successfully sold {quantity} shares of {stock} at ${stock_price} per share.")
        else:
            print(f"Insufficient shares of {stock} to sell.")

    def log_transaction(self, action, stock, quantity, price):
        # Log the buy or sell transaction
        transaction = {
            "action": action,
            "stock": stock,
            "quantity": quantity,
            "price": price
        }
        self.transaction_history.append(transaction)

    def view_portfolio(self):
        # Display portfolio holdings and cash
        print("\nPortfolio:")
        print(f"Cash: ${self.portfolio['cash']}")
        for stock, quantity in self.portfolio["stocks"].items():
            print(f"{stock}: {quantity} shares")

    def view_transaction_history(self):
        # Display all transactions
        print("\nTransaction History:")
        for transaction in self.transaction_history:
            print(f"{transaction['action'].capitalize()} {transaction['quantity']} shares of {transaction['stock']} at ${transaction['price']}")

    def update_stock_prices(self):
        # Random walk to update stock prices each day
        for stock in self.stock_prices:
            # Simulate daily stock price fluctuation (random change between -5% and +5%)
            fluctuation = random.uniform(-0.05, 0.05)
            self.stock_prices[stock] = round(self.stock_prices[stock] * (1 + fluctuation), 2)

    def calculate_profit_loss(self):
        total_profit_loss = 0  # To track overall profit/loss for the portfolio

        print("\nProfit/Loss Report:")
        for stock, quantity in self.portfolio["stocks"].items():
            current_price = self.get_current_stock_price(stock)
            
            # Calculate the average price at which the stock was purchased
            avg_purchase_price = self.get_average_purchase_price(stock)

            # Calculate the profit or loss per share
            profit_loss_per_share = current_price - avg_purchase_price
            
            # Calculate the total profit or loss for the given stock
            total_profit_loss_for_stock = profit_loss_per_share * quantity

            # Add to overall portfolio profit/loss
            total_profit_loss += total_profit_loss_for_stock

            # Display the result for this stock
            print(f"{stock}:")
            print(f"  Shares: {quantity}")
            print(f"  Current Price: ${current_price}")
            print(f"  Average Purchase Price: ${avg_purchase_price}")
            print(f"  Profit/Loss per Share: ${profit_loss_per_share}")
            print(f"  Total Profit/Loss for {stock}: ${total_profit_loss_for_stock}")

        print(f"\nOverall Profit/Loss for the Portfolio: ${total_profit_loss}\n")
        return total_profit_loss

    def get_average_purchase_price(self, stock):
        # Calculate the average price at which the stock was purchased
        # We assume you are logging the transaction history and the buy prices
        total_shares_bought = 0
        total_cost = 0
        
        for transaction in self.transaction_history:
            if transaction['action'] == 'buy' and transaction['stock'] == stock:
                total_shares_bought += transaction['quantity']
                total_cost += transaction['quantity'] * transaction['price']

        if total_shares_bought > 0:
            return total_cost / total_shares_bought  # Average purchase price
        else:
            return 0  # If no shares were bought, return 0


    def run_simulation(self, days=90, transaction_interval=7):
        # Run the simulation for the specified number of days (e.g., 90 for 3 months)
        for day in range(1, days + 1):
            print(f"\nDay {day}")
            
            # Update stock prices daily
            self.update_stock_prices()
            self.display_stock_prices()

            # Allow the user to make transactions every `transaction_interval` days (e.g., weekly)
            if day % transaction_interval == 0:
                self.view_portfolio()
                user_input = input(f"Day {day}: Do you want to buy or sell stocks? (yes/no): ").strip().lower()
                
                if user_input == 'yes':
                    action = input("Do you want to 'buy' or 'sell'?: ").strip().lower()
                    stock = input("Which stock (AAPL, GOOGL, TSLA)?: ").strip().upper()
                    quantity = int(input("How many shares?: "))

                    if action == "buy":
                        self.buy_stock(stock, quantity)
                    elif action == "sell":
                        self.sell_stock(stock, quantity)  # Assume a sell function exists

            # End of day report
            self.view_portfolio()

        # End of simulation, show the final portfolio and transactions
        print("\n--- End of Simulation ---")
        self.view_portfolio()
        self.view_transaction_history()

    def display_stock_prices(self):
        # Display current stock prices
        print("\nCurrent Stock Prices:")
        for stock, price in self.stock_prices.items():
            print(f"{stock}: ${price}")


# Example usage:

# Create an instance of the simulation
simulation = StockSimulation()

# Run the simulation for 90 days, allowing transactions every 7 days (weekly)
simulation.run_simulation(days=90, transaction_interval=7)


#%%

simulation.calculate_profit_loss()