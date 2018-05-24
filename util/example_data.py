import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .frange import frange
from math import pi
import pylab as lab



def circle(samples_per_class=100,numclasses=2,shift_rad=0.5, step_size=0.01, variance=0.1):

    num_steps = shift_rad/step_size
    samples_per_step = int(np.ceil(samples_per_class/num_steps))
    interclass_distance_circle = 2/numclasses
    class_start = [i for i in frange(0, 2-interclass_distance_circle, interclass_distance_circle)]
    data = np.array([]).reshape(0,2)
    labels = np.array([]).reshape(0,1)

    for i, c in enumerate(class_start):
        for j in frange(c, c+shift_rad , step_size):
            x = np.cos(pi*j)
            y = np.sin(pi*j)
            samples = np.random.normal(1, variance, samples_per_step)
            tmp = np.array([np.array([x, y]) * s for s in samples])
            data = np.vstack([data,tmp ])
            tmp = np.full([samples_per_step,1], i)
            labels = np.vstack([labels, tmp])

    return data, labels



def norm(numsamples=100, numclasses=2):
    data = np.array([]).reshape(0,2)
    labels = np.array([]).reshape(0,1)

    #two normal distributions moving towards each other

    return data, labels



def plot(data, label):
    x = data[:, 0]
    y = data[:, 1]
    N = np.size(np.unique(label))
    if N > 1:
        # define the colormap
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = np.linspace(0, N, N + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        plt.scatter(x,y,c=label[:,0],cmap=cmap, norm=norm)
    else:
        plt.scatter(x, y)
    plt.show()



if __name__=='__main__':
    numclasses=2
    shift_rad = 2/numclasses
    data, labels = circle(numclasses=numclasses, shift_rad=shift_rad)

    plot(data,labels)

