import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib import ticker

# --- Trader ----------------------------------------------------------------------------------------------------
class Trade:
    def __init__(self, stockValue, size, startTime, type):
        self.startValue = stockValue
        self.size = size
        self.startTime = startTime
        self.type = type # long or short
    def update(self, stockValue):
        if self.type == "long":
            return self.size*(stockValue - self.startValue)
        if self.type == "short":
            return self.size*(self.startValue - stockValue)
        return 0

class Trader:
    def __init__(self, balance):
        self.balance = balance
        self.trades = []
    def updateTrades(self, stockValue):
        for trade in self.trades:
           self.balance += trade.update(stockValue=stockValue)
        self.trades = []
        return self.balance
    def goLong(self, stockValue, size, startTime):
        self.trades += [Trade(stockValue=stockValue, size=size, startTime=startTime, type="long")]
    def goShort(self, stockValue, size, startTime):
        self.trades += [Trade(stockValue=stockValue, size=size, startTime=startTime, type="short")]
    def getTrades(self):
        return self.trades
    def getTrades(self, type):
        return [t for t in self.trades if t.type == type]
# ---------------------------------------------------------------------------------------------------------------

# --- Logger ----------------------------------------------------------------------------------------------------
class Logger:
    
    @staticmethod
    def plotTimeseries(df, replaceDates=True):
        df = df.copy()
        df.index = range(len(df.index))
        df.plot()
        plt.show()

    @staticmethod
    def bulk(process, max):
        percent = process / (max + 1e-6)
        percent = int(percent*100/2)
        str =  f'{process}/{max} '
        str += '['
        for i in range(50):
            if i < percent: str += '='
            elif i == percent: str += '>'
            else: str += '.'
        str += ']'
        return str

    @staticmethod
    def plotKerasCategories(pred, truth, predictionLength):
        # evaluate: [list(x).index(1) for x in truth].count(2)
        # convert to list: [list(x) for x in pred]
        truth = [x[0][0] for x in truth]

        # create plot
        fig=plt.figure(figsize=(18, 9))
        ax=fig.add_subplot(111)

        # plot predition line
        for line in range(int(pred.shape[0]/predictionLength)):
            predData = np.empty(line*predictionLength)
            predData[:] = np.nan
            predData = np.append(predData, truth[line*predictionLength])
            predBlock = list(pred[line*predictionLength])
            predValue = predBlock.index(max(predBlock))
            predValue = (predValue-2)*(-0.0001) + truth[line*predictionLength]
            predData = np.append(predData, predValue)
            ax.plot(range(len(predData)), predData, 'b-', scalex=False, scaley=False)
        
        # plot truth line
        ax.plot(range(len(truth)), truth, 'k-')

        # plot grid
        major_ticks = np.arange(0, len(truth), 100)
        minor_ticks = np.arange(0, len(truth), 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        
        # plot labels
        plt.xlabel('Data', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.legend(['truth','prediction'])
        legendPred = mpatches.Patch(color='blue',  label='Prediction')
        legendTrut = mpatches.Patch(color='black', label='Truth')
        plt.legend(handles=[legendPred, legendTrut])
        
        return plt

    @staticmethod
    def plotKeras(pred, truth, predictionLength):
        #pred = pred.reshape(-1)

        # create plot
        fig=plt.figure(figsize=(18, 9))
        ax=fig.add_subplot(111)

        # plot predition line
        for line in range(int(len(pred)/predictionLength)):
            predData = np.empty(line*predictionLength)
            predData[:] = np.nan
            # add exact gradient
            diff = pred[line*predictionLength] - truth[line*predictionLength]
            baseValue = truth[line*predictionLength]
            for i in range(predictionLength+1):
                value = baseValue + diff*(i)/predictionLength
                predData = np.append(predData, value)
            # add end-value
            ax.plot(range(len(predData)), predData, 'b-', scalex=False, scaley=False)
        
        # plot truth line
        ax.plot(range(len(truth)), truth, 'k-')

        # plot grid
        major_ticks = np.arange(0, len(truth), 100)
        minor_ticks = np.arange(0, len(truth), 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        
        # plot labels
        plt.xlabel('Data', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.legend(['truth','prediction'])
        legendPred = mpatches.Patch(color='blue',  label='Prediction')
        legendTrut = mpatches.Patch(color='black', label='Truth')
        plt.legend(handles=[legendPred, legendTrut])
        
        return plt

    @staticmethod
    def plotMoneyMaker(directions, gradients):
        x = range(len(gradients))
        # gradients
        ax1 = plt.subplot(211)
        plt.plot(x, gradients)
        plt.ylabel('Gradient Match [%]', fontsize=14)
        plt.setp(ax1.get_xticklabels(), visible=False)
        # directions
        # share x only
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(x, directions)
        plt.ylabel('Direction Match [%]', fontsize=14)
        plt.xlabel('Epoch', fontsize=14)
        return plt
        
    @staticmethod
    def plotCompareMatrix(matrix, predictionLength):
        fig, ax = plt.subplots()
        im, cbar = heatmap(np.array(matrix), ax=ax,
                        cmap="YlGn", cbarlabel="sum")
        texts = annotate_heatmap(im, valfmt="{x:.1f}")
        fig.tight_layout()
        return fig

    @staticmethod
    def plotSlices(x, y, p, plots=9):
        for i in range(plots):
            ax = plt.subplot(plots,1,i+1)
            for j in range(len(x[i,0])):
                plt.plot(list(range(len(x[i,:,j]))), x[i,:,j])
            yrplt = (len(x[0,:,0])-1)*[None] + [y[i,1]] + [y[i,0]]
            plt.plot(list(range(len(yrplt))), yrplt)
        return plt
# ---------------------------------------------------------------------------------------------------------------


# - https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html ----------------------
def heatmap(data, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
# ---------------------------------------------------------------------------------------------------------------