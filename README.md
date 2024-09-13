

# Process

1. Asset is created
2. Initial Position is created at zero
3. Order is created for the asset
4. Order is taking Quotesource into account to define price.
5. Trading fees are added to the order
6. If conditions are validated the order is submitted
7. When submitted, the order will update the qty and the unit_price_value in the Position
8. Strategy will run over a period of time defined by start and end date with a incrementation step


# Technical steps

- Asset
    - Parameters :
        - Symbol
        - Long Name
- Position
    - Parameters :
        - Symbol
        - quantity
        - PRU
    - Method : 
        - set_quantity
        - get_quantity
        - update_quantity
        - set_pru
        - update_pru
- Order
    - Parameters:
        - Symbol
        - Quantity
    - Method :
        - create_order
        - submit_order
- TradingFees
    - Parameters:
        - Flat_Fees
        - Percent_Fees
- QuoteSource
    - Parameters:
        - quotes
    - Method:
        - Get_quote

- Strategy
    - Parameters:
        - Initial Cash
    - Method :
        - initialize
        - on_iteration
            - Runs day by day with incremented steps being defined in strategy init
