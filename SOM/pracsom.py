import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset =  pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,  :-1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)


from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len= 15, sigma= 1.0, learning_rate= 0.5, random_seed= 1)
som.random_weights_init(X)
som.train_random(X, num_iteration = 138000)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map())
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()    


mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,2)], mappings[(6,7)]), axis=0)
frauds = sc.inverse_transform(frauds)