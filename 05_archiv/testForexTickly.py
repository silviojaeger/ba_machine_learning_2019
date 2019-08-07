import sys
sys.path.append('01_preperation')

import matplotlib.pyplot as plt

from collector import *
from calculator import *
from company import *

# --- Start -----------------------------------------------------------------------------------------------------
def main():
    col = Collector(replace=False, replaceForex=True)
    df = col.getBySymbol('EURUSD').df
    df.plot()
    plt.show()
    
if __name__ == "__main__": main()
# ---------------------------------------------------------------------------------------------------------------