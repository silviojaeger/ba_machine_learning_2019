import pandas as pd

# --- Currency --------------------------------------------------------------------------------------------------
class Currency:
    def __init__(self, symbol, relations):
        self.symbol     = symbol
        self.df         = pd.DataFrame()
        self.relations  = relations
        self.unitSize   = pd.Series()
# ---------------------------------------------------------------------------------------------------------------