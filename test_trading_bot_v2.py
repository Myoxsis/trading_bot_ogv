#%%

import numpy as np

#%%

CASH_ON_HAND = 1000
global CASH_ON_HAND

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
        if CASH_ON_HAND > 0:
            if self.price <= CASH_ON_HAND:
                pos = Position(asset)
                pos.update_quantity_and_unit_price(self.order_type, self.price)
                CASH_ON_HAND -= self.price
        else:
            raise ValueError('No cash available')


#%%

alo = Asset("ALO.PA", "Alstom S.A.")

alo_position = Position(alo)
alo_position.display_position()

alo_ord = Order(alo)
alo_ord.create_order("buy", 2)
alo_ord.submit_order()
