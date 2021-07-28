import matplotlib.pyplot as plt
import numpy as np


def plot_image(data, ax=None, font_size=12, title="before"):
    '''
    vis image
    2D array
    '''
    if ax is not None:
        ax.imshow(data, cmap='gray')
        ax.set_title(title, size=font_size, weight='bold')
        ax.set_axis_off()
        ax.grid(False)
    else:
        plt.imshow(data, cmap='gray')
        plt.title(title, size=font_size, weight='bold')
        plt.axis('off')
    return ax


def plot_general(data, ax=None, font_size=12, title="", cmap='gray'):
    '''
    vis image
    2D array
    '''
    if ax is not None:
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, size=font_size, weight='bold')
        ax.set_axis_off()
        ax.grid(False)
    else:
        plt.imshow(data, cmap=cmap)
        plt.title(title, size=font_size, weight='bold')
        plt.axis('off')
    return ax


def plot_noise(data, ax=None, font_size=12, title="rand noise"):
    '''
    vis noise
    2D array
    '''
    if ax is not None:
        ax.imshow(data, cmap='RdBu_r', interpolation='none',
                  vmin=-np.max(np.array(data)))
        ax.set_title(title, size=font_size, weight='bold')
        ax.set_axis_off()
        ax.grid(False)
    else:
        plt.imshow(data, cmap='RdBu_r', interpolation='none',
                   vmin=-np.max(np.array(data)))
        plt.title(title, size=font_size, weight='bold')
        plt.axis('off')

    return ax


def plot_bias_field(data, ax=None, font_size=12, title="rand bias"):
    '''
    vis bias
    input 2D array
    '''
    if ax is not None:
        ax.imshow(data, cmap='jet')
        ax.set_title(title, size=font_size, weight='bold')
        ax.set_axis_off()
        ax.grid(False)
    else:
        plt.imshow(data, cmap='jet')
        plt.title(title, size=font_size)
        plt.axis('off')


def plot_warped_grid(dvf, ax=None, bg_img=None, interval=3, title="$\mathcal{T}_\phi$", fontsize=20, linewidth=0.5, show=True):
    '''
    dvf: deformation grid (offsets): 3D input, [2HW],  pytorch offsets: dvf=[[dx,dy]]. Using ij coordinates with zero (0,0) as origin. range (-1,1)
    return 
    ax
    '''

    """dvf shape (2, H, W)"""
    if bg_img is not None:
        background = bg_img
    else:
        background = np.zeros(dvf.shape[1:])

    # mesh grid
    h, w = dvf.shape[1], dvf.shape[2]
    yy, xx = np.meshgrid(range(0, h, interval),
                         range(0, w, interval),
                         indexing='ij')
    # pytorch grid position lies in the range of [-1, 1] (scale=2)，need to rescale it so that it matches to the image scales (w,h)
    dvf[0] = (dvf[0])*(background.shape[1]/2)  # in x-direction
    dvf[1] = (dvf[1])*(background.shape[0]/2)  # in y-direction

    new_grid_X = xx + (dvf[0, yy, xx])
    new_grid_Y = yy + (dvf[1, yy, xx])

    kwargs = {"linewidth": linewidth, "color": "r"}
    if show:
        if ax is not None:
            img = ax.imshow(background, cmap='gray')
            # matplot direction (width, height)
            # ax.plot(0,10,'o')
        else:
            img = plt.imshow(background, cmap='gray')

    # plt: (0,0) is on the left upper corner.
    if ax is not None:
        for i in range(xx.shape[0]):
            ax.plot((new_grid_X[i, :]), (new_grid_Y[i, :]),
                    **kwargs)  # each draws a horizontal line
        for i in range(xx.shape[1]):
            ax.plot(new_grid_X[:, i], (new_grid_Y[:, i]),
                    **kwargs)  # each draws a vertical line
        ax.set_title(title, fontsize=fontsize, weight='bold')
        ax.axis('off')
    else:
        for i in range(xx.shape[0]):
            # each draws a horizontal line
            plt.plot((new_grid_X[i, :]), (new_grid_Y[i, :]), **kwargs)
        for i in range(xx.shape[1]):
            plt.plot(new_grid_X[:, i], (new_grid_Y[:, i]),
                     **kwargs)  # each draws a vertical line
        plt.title(title, size=fontsize, weight='bold')

        plt.axis('off')

    return ax
