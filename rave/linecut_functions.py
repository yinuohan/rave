from .lib import *
from .short_functions import *


## Linecut
def make_linecut(image, cut_height='full'):
    '''Takes in an image and sums all the flux within CUT_HEIGHT from the x axis onto the x axis.
    Returns the dictinary with keys LEFT and RIGHT that correspond to the left and right halves of the image.
    If CUTHEIGHT is 'FULL': squish whole image onto the x axis.
    All units in pixels.'''

    if type(image) is dict:
        return {
            'left': make_linecut(image['left'], cut_height)['left'],
            'right': make_linecut(image['right'], cut_height)['right']
            }

    ydim, xdim = image.shape # pixels
    cx, cy = xdim//2, ydim//2 # pixels

    if cut_height == 'full':
        right_segment = image[:, cx: ]
        if xdim % 2 == 0:
            left_segment = image[:, cx-1: :-1]
        else:
            # Include central point corresponding to star
            left_segment = image[:, cx: :-1]
    else:
        cut_height = int(round(cut_height))
        if xdim % 2 == 0:
            assert ydim % 2 == 0
            left_segment = image[cy-cut_height:cy+cut_height, cx-1: :-1]
            right_segment = image[cy-cut_height:cy+cut_height, cx: ]
        else:
            assert ydim % 2 == 1
            # Include central point corresponding to star
            left_segment = image[cy-cut_height:cy+cut_height+1, cx: :-1]
            right_segment = image[cy-cut_height:cy+cut_height+1, cx: ]

    linecut = {
        'left': np.sum(left_segment, 0),
        'right': np.sum(right_segment, 0)
        }

    return linecut


def bin_linecut(linecut, r_bounds):
    '''Discretises an array LINECUT by averaging the values within each bin defined by R_BOUNDS. R_BOUNDS defines the index boundaries of each bin.'''

    if type(linecut) is dict:
        return {
            'left': bin_linecut(linecut['left'], r_bounds),
            'right': bin_linecut(linecut['right'], r_bounds)
            }

    r_bounds = np.array(r_bounds)
    linecut = np.array(linecut)
    coords = np.arange(0, len(linecut), 1) + 0.5

    assert r_bounds[-1] <= coords[-1] + 0.5

    nbins = len(r_bounds) - 1
    binned_linecut = np.zeros(nbins)

    for i in range(nbins):
        ks = np.where((coords >= r_bounds[i]) & (coords < r_bounds[i+1]))
        binned_linecut[i] = np.mean(linecut[ks])

    return binned_linecut


def cut_rings(rings, cut_height='full'):
    '''Apply MAKE_LINECUT to each element of the list RINGS'''
    return np.array([make_linecut(rings[i], cut_height) for i in range(len(rings))])


def bin_rings(rings, r_bounds, cut_height='full'):
    '''Apply MAKE_LINECUT then BIN_LINECUT to each element of the array RINGS.
    Returns a dictionary with keys LEFT and RIGHT. '''
    x = np.array([bin_linecut(make_linecut(rings[i], cut_height), r_bounds) for i in range(len(rings))])
    nrings = len(r_bounds) - 1
    abrl = handlelr(np.zeros)([nrings, nrings])
    for i in range(nrings):
        abrl['left'][i] = x[i]['left']
        abrl['right'][i] = x[i]['right']
    return abrl
