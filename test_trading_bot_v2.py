#%%

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

#%%

def f_memoize_dt(s):
    """
    Memoization technique
    """
    dates = {date:datetime.strptime(date,'%Y-%m-%d') for date in s.unique()}
    return s.map(dates)

#%%

df = pd.read_csv("../Stock_Portfolio_Management/data/quote/ALO.PA.csv")
df = df.drop(columns=['Unnamed: 0'])

df['Date'] = f_memoize_dt(df['Date'])

#%%

df.head()

# %%

df.info()

#%%

CASH_ON_HAND = 1000

class Asset:
    def __init__(self, symbol : str, long_name : str) -> None:
        self.symbol = symbol
        self.long_name = long_name

class Position:
    def __init__(self, asset : Asset, quantity = 0, unit_price = 0) -> None:
        self.symbol = asset.symbol
        self.long_name = asset.long_name
        self.quantity = quantity
        self.unit_price = unit_price

    def update_quantity_and_unit_price(self, transaction_type, new_quantity, new_price):
        if transaction_type.lower() in ['buy', 'sell']:
            if transaction_type == "buy":
                self.unit_price = (self.quantity * self.unit_price) + (new_quantity * new_price)
                self.quantity += new_quantity
            elif transaction_type == "sell":
                self.unit_price = (self.quantity * self.unit_price) - (new_quantity * new_price)
                self.quantity -= new_quantity
        else:
            raise ValueError('Transaction type should be : Buy or Sell')
        

    def display_position(self):
        print(f"Position : Asset name : {self.long_name}, Quantity : {self.quantity}, PRU : {self.unit_price}")

class Order:
    def __init__(self, asset : Asset) -> None:
        self.asset = asset
        self.symbol = asset.symbol
        self.long_name = asset.long_name

    def create_order(self, order_type, quantity):
        self.quantity = quantity
        self.order_type = order_type
        self.price = np.random.randint(0, 5)

    def submit_order(self):
        pos = Position(self.asset)
        pos.update_quantity_and_unit_price(self.order_type, self.quantity, self.price)
        

class Strategy:
    def __init__(self, start_date, end_date, initial_cash, incrementation_step=1):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash

    def initialize(self):
        pass

    def on_iteration(self):
        pass

    def run(self):
        self.initialize()
        for date in pd.date_range(self.start_date, self.end_date):
            self.on_iteration()


#%%

CASH_ON_HAND = 1000
ORDER_TO_MANAGE = 2
LOW_LIMIT_PERCENT = 1
HIGH_LIMIT_PERCENT = 2
LAST_TRADE_VAL = 0

# initialize

alo = Asset("ALO.PA", "Alstom S.A.")
alo_position = Position(alo)
alo_position.display_position()

last_trade = None

# On iteration

date_l = []
quote_l = []
trade_l = []
cash_on_hand_l = []

quote = 5

for current_date in pd.date_range(datetime(2024, 8, 8), datetime(2024, 9, 5)):
    #print(current_date.strftime("%Y-%m-%d"))
    #quote = df.loc[df['Date'] == current_date.strftime("%Y-%m-%d"), :]['Adj Close'].reset_index(drop=True)[0]#.values()[0]
    #print(quote)
    quote = quote + (1 * (np.random.randint(-4, 4)/100) * (np.random.randint(2, 5)/10) + pow((0.0005/10),6) ) 
    date_l.append(current_date)
    quote_l.append(quote)
    last_trade_val = 0

    if last_trade == None:
        if quote < 5.00:
            alo_ord = Order(alo)
            alo_ord.create_order("buy", ORDER_TO_MANAGE)
            alo_ord.submit_order()
            last_trade = "buy"
            LAST_TRADE_VAL = quote
            trade_l.append(quote)
            CASH_ON_HAND = CASH_ON_HAND - (quote * alo_ord.quantity)
        else:
            trade_l.append(None)
    
    elif last_trade == "buy":
        low_limit_price = LAST_TRADE_VAL * 1-(LOW_LIMIT_PERCENT/100)
        high_limit_price = LAST_TRADE_VAL * 1+(HIGH_LIMIT_PERCENT/100)
        if quote <= low_limit_price:
            alo_ord = Order(alo)
            alo_ord.create_order("sell", ORDER_TO_MANAGE)
            alo_ord.submit_order()
            last_trade = "sell"
            trade_l.append(quote)
            CASH_ON_HAND = CASH_ON_HAND + (quote * alo_ord.quantity)
        elif quote >= high_limit_price:
            alo_ord = Order(alo)
            alo_ord.create_order("sell", ORDER_TO_MANAGE)
            alo_ord.submit_order()
            last_trade = "sell"
            trade_l.append(quote)
            CASH_ON_HAND = CASH_ON_HAND + (quote * alo_ord.quantity)
        else:
            trade_l.append(None)
    else:
        trade_l.append(None)
    cash_on_hand_l.append(CASH_ON_HAND)

plt.plot(date_l, quote_l)
plt.scatter(date_l, trade_l)

#alo_position.display_position()

print(trade_l)
print(cash_on_hand_l)
print(f"TOTAL EARNING : ", cash_on_hand_l[-1] - 1000)