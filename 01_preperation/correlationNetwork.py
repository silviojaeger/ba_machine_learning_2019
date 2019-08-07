import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pandas as pd
import os
import glob
import re

stocks = 'stocks_de'
workdir = os.path.join(os.getcwd(), 'logs', 'cross_correlation', stocks, 'csv')
output  = os.path.join(os.getcwd(), 'logs', 'cross_correlation', stocks, 'correlationNetwork.svg')
catdir  = os.path.join(os.getcwd(), 'data_temp', 'collectorDuka')
reqMinIndex = 0.7
maxEdges = 4

matplotlib.use('TkAgg')
plt.figure(1,figsize=(200,200)) 
G = nx.Graph()

# --- Colors ----------------------------------------------------------------------------------------------------
if not os.path.exists(workdir): os.makedirs(workdir)
colorlist = {}
path = os.path.join(catdir, stocks, '*_Ask_*.csv')
catdirFiles = glob.glob(path)
for catdirFile in catdirFiles:
    name = re.sub(r"_.*", '', os.path.basename(catdirFile), 0)
    directory = os.path.basename(os.path.dirname(catdirFile))
    if not directory in colorlist: colorlist[directory] = len(colorlist)
    G.add_node(name, color=colorlist[directory])
# ---------------------------------------------------------------------------------------------------------------

symbols = {}
minIndex = 0
path = os.path.join(workdir, '*.csv')
symbolFiles = glob.glob(path)
for file in symbolFiles:
    name = re.sub(r"\..*", '', os.path.basename(file), 0)
    symbols[name] = pd.read_csv(file, index_col=0)
    symbols[name] = symbols[name][symbols[name]['shift'] >= 0]
    symbols[name] = symbols[name][symbols[name]['index'] > reqMinIndex]
    symbols[name] = symbols[name].iloc[:maxEdges]
    symbols[name] = symbols[name].sort_values(ascending=False, by=['index', 'shift'])
    if symbols[name]['index'].min()>minIndex: minIndex = symbols[name]['index'].min()

for symbol, df in symbols.items():
    if df.shape[0] <= 0: continue
    for index, row in df.iterrows():
        if G.has_edge(symbol, index): continue
        G.add_edge(symbol, index, weight=row['index'], color=row['index']/(1-minIndex))

edge_colors = [G.edges[i]['color'] for i in G.edges()]
node_colors = [G.node[i]['color'] for i in G.nodes()]
node_sizes  = [1400 for i in G.nodes()]

pos = nx.spring_layout(G)  # positions for all nodes

nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.Dark2)
edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=2)

nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif', font_color="white")

ax = plt.gca()
ax.set_axis_off()
plt.savefig(output, format="SVG", dpi=200)
plt.show()