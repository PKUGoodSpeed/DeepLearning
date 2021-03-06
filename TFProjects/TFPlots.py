import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rc('font',family='Times New Roman',size = 12)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

class MnistImagePlot:
    __img_height = 28
    __img_width = 28
    __img_shape = (__img_height, __img_width)

    def __init__(self, height = 28, width = 28):
        ''''''
        self.__img_height = height
        self.__img_width = width
        self.__img_shape = (height, width)

    def plotImage(self, r, c, imgs, cls, cls_pred = None, file_name = None):
        '''
        Plot images in an rXc grid
        :param r: number of rows
        :param c: number of columns
        :param imgs: image array
        :param cls: true value
        :param cls_pred: predicted classifications
        :param file_name: to which the figure will be saved
        :return: None
        '''
        assert len(imgs) == len(cls) == r * c
        assert r > 0 and c > 0

        # Create subplots within an r X c grid
        fig, axes = plt.subplots(r, c)
        fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

        for i, ax in enumerate(axes.flat):
            # Plot images
            ax.imshow(imgs[i].reshape(self.__img_shape), cmap = 'binary')

            # Show Labels
            if cls_pred is None:
                xlabel = "True: {0}".format(cls[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls[i], cls_pred[i])
            ax.set_xlabel(xlabel)

            # Remove ticks from the images
            ax.set_xticks([])
            ax.set_yticks([])
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.3)

        return


    def plotWeights(self, wghts, r = 2, c = 5, file_name = None):
        '''
        Plot weight as images in an r X c grid
        Only for linear model
        :param wghts: weights
        :param r: number of rows
        :param c: number of columns
        :param file_name: to which the figure will be saved
        :return: None
        '''

        # There are intotal 10 sub-figures
        assert r * c >= 10

        w_min = np.min(wghts)
        w_max = np.max(wghts)

        # Create image grid with r X c sub-plots
        fig, axes = plt.subplots(r, c)
        fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

        for i, ax in enumerate(axes.flat):
            if(i >= 10):
                break

            # Need to convert the weights
            img = wghts[:, i].reshape(self.__img_shape)

            # Set Label for each subplot
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image corresponding to a particular weight
            ax.imshow(img, vmin = w_min, vmax = w_max, cmap = 'seismic')

            # Remove tick marks
            ax.set_xticks([])
            ax.set_yticks([])

        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.3)

        return

    def plotCost(self, t_list, l_list, xscale = 'linear', yscale = 'linear',
                 xlabel = "Nstep", ylabel = "Cost", file_name = None):
        '''
        Plot Cost/Lost function
        :param t_list: Nstep
        :param l_list: Cost/Lost function
        :param xlabel: xlabel
        :param ylabel: ylabel
        :param file_name: to which the figure will be saved
        :return: None
        '''
        fig, ax = plt.subplots(1,1)
        ax.plot(t_list, l_list, '-o', lw = 1., markersize = 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches = 'tight', pad_inches = 0.3)

        return
