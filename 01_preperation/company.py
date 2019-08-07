import pandas as pd

# --- Company ---------------------------------------------------------------------------------------------------
class Company:
    def __init__(self, name, symbol, country):
        self.symbol     = symbol
        self.name       = name
        self.country    = country
        self.df         = pd.DataFrame()
# ---------------------------------------------------------------------------------------------------------------