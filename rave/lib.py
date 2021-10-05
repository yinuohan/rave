## Need to import these
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import time
from scipy.ndimage import gaussian_filter as blur
from scipy.ndimage import rotate
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import convolve

## Matplotlib GUI
import matplotlib
matplotlib.use('QT5Agg')
plt.rcParams['savefig.dpi'] = 300

## Aliases
pi = np.pi
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
read = fits.getdata

## Image processing functions
def rotate180(image):
    '''Rotates image by 180 deg about its center'''
    return np.flipud(np.fliplr(image))

def normalise(image):
    '''Sets brightest pixel in image to 1'''
    return image / np.max(image)

def reshape_image(image, new_ydim, new_xdim):
    '''Takes in an image, retuns an image with new dimensions.
    Pads image with 0's if new dimensions are bigger.
    Cuts out a region centred at the input image's centre if new dimensions are smaller.'''
    ydim, xdim = image.shape
    ycut, xcut = min(new_ydim, ydim), min(new_xdim, xdim)

    image_cut = image[ydim//2 - ycut//2 : ydim//2 + ycut//2, xdim//2 - xcut//2 : xdim//2 + xcut//2]

    new_image = np.zeros([new_ydim, new_xdim])
    new_image[new_ydim//2 - ycut//2 : new_ydim//2 + ycut//2, new_xdim//2 - xcut//2 : new_xdim//2 + xcut//2] = image_cut

    return new_image

def clip(image, lower, upper, rel=True):
    '''Sets everything in image bigger than UPPER to UPPER.
    Sets everything in image smaller than LOWER to LOWER.
    If REL is TRUE, LOWER and UPPER are interpreted as a fraction pf the brightest pixel.'''
    if rel:
        return np.clip(image, lower * np.max(image), upper * np.max(image))
    else:
        return np.clip(image, lower, upper)

## Plotting functions
def plot(image, x=None, title=None, cmap='viridis', n=True, flip=False, extent=None, marker='.-'):
    '''An umbrella plotting function that can plot 1D or 2D arrays.
    For example:
    >>> plot(image)
    >>> plot(y)
    >>> plot(x, y)'''
    newfigure = n
    if type(image) == str:
        if image[-5:] != '.fits':
            image += '.fits'
        image = fits.getdata(image)
    if newfigure:
        fig, ax = plt.subplots()
    image = np.array(image)
    if image.ndim == 2:
        if flip:
            image = np.fliplr(image)
        if extent is not None:
            plt.imshow(image, origin='lower', cmap=cmap, extent=extent)
        else:
            plt.imshow(image, origin='lower', cmap=cmap)
    elif image.ndim == 1:
        if x is not None:
            plt.plot(image, x, marker)
        else:
            plt.plot(image, marker)
    if title:
        plt.title(title)
    plt.show()
    if newfigure:
        return fig, ax

def cliplot(image, lo=None, hi=None):
    '''Plots the image after clipping the image between LO and HI.
    If only one bound is given, it is inperpreted as HI, and LO is set to 0.
    LO and HI are relative to the brightest pixel.'''
    if type(image) is str:
        if image[-5:] != '.fits':
            image += '.fits'
        image = fits.getdata(image)
    if hi is None:
        if lo is None:
            lo, hi = 0, 1
        else:
            lo, hi = 0, lo
    plot(clip(image, lo, hi))

def c():
    '''Close all MATPLOTLIB windows.'''
    plt.close('all')

