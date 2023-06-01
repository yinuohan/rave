#from .lib import *
#from .short_functions import *


## Linecut
def make_linecut(image, cut_height='full'):
    '''Takes in an image and sums all the flux within CUT_HEIGHT from the x axis onto the x axis.
    Returns the dictionary with keys LEFT and RIGHT that correspond to the left and right halves of the image.
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


## Azimuthal profiles
def get_azimuthal_profile(image, inclination=0, increment=10):
    '''Takes in an image and returns the azimuthally averaged radial profile assuming the given inclination.
    All units in pixels.'''

    distortion_factor = 1/abs(np.cos(inclination/180*pi))
    distorted_image = zoom(image, [distortion_factor, 1])
    distorted_image = reshape_image(distorted_image, *image.shape) / distortion_factor
    #plot(distorted_image)

    y, x = image.shape
    cy, cx = y//2, x//2

    PROFILES = []
    for theta in range(0, 360, increment):
        rotated_image = rotate(distorted_image, theta, reshape=False)
        half_width = int(increment/2)
        PROFILES.append(rotated_image[cx-half_width:cx+half_width].mean(0))
    PROFILES = np.array(PROFILES)
    profile = PROFILES.mean(0)[cx:]

    return profile

def get_azimuthal_profile2(im, inclination=0, deltar=1):
    distortion_factor = 1/abs(np.cos(inclination/180*pi))
    distorted_image = zoom(im, [distortion_factor, 1])
    distorted_image = reshape_image(distorted_image, *im.shape) / distortion_factor
    #plot(distorted_image)

    y, x = im.shape
    assert y % 2 == 0 and x % 2 == 0
    cy, cx = y//2, x//2
    ycord, xcord = np.arange(y) - cy + 0.5, np.arange(x) - cx + 0.5
    xx, yy = np.meshgrid(xcord, ycord)
    rr = (xx**2 + yy**2) ** (1/2)

    rcoord = np.arange(deltar/2, cy, deltar)
    profile = np.zeros(rcoord.shape)
    count = np.zeros(rcoord.shape)

    for j in range(y):
        for i in range(x):
            "A distance of 1 corresponds to the 0th entry in profile"
            r = int(np.floor(rr[j, i] / deltar))
            if r < len(rcoord):
                profile[r] += distorted_image[j, i]
                count[r] += 1

    profile = profile / count
    #plot(rcoord, profile)

    return profile

def get_binned_azimuthal_profile(im, r_bounds, inclination=0):
    '''Equivalent to get_azimuthal_profile2 but slower.'''

    distortion_factor = 1/abs(np.cos(inclination/180*pi))
    distorted_image = zoom(im, [distortion_factor, 1])
    distorted_image = reshape_image(distorted_image, *im.shape) / distortion_factor
    #plot(distorted_image)

    y, x = im.shape
    assert y % 2 == 0 and x % 2 == 0
    cy, cx = y//2, x//2
    ycord, xcord = np.arange(y) - cy + 0.5, np.arange(x) - cx + 0.5
    xx, yy = np.meshgrid(xcord, ycord)
    rr = (xx**2 + yy**2) ** (1/2)

    #r_bounds = np.arange(0, cy + 0.1, 1)
    r_centre = (r_bounds[1:] + r_bounds[:-1]) / 2
    N = len(r_bounds) - 1

    binned_profile = np.zeros(N)
    for i in range(N):
        r_lower = r_bounds[i]
        r_upper = r_bounds[i + 1]
        mask = (rr > r_lower) & (rr < r_upper)
        #plot(mask)
        values = distorted_image[mask]
        binned_profile[i] = values.mean()

    #plot(r_centre, binned_profile)

    return binned_profile

def get_binned_azimuthal_profile2(im, r_bounds, inclination=0):
    '''Equivalent to get_binned_azimuthal_profile but faster'''

    distortion_factor = 1/abs(np.cos(inclination/180*pi))

    y, x = im.shape
    assert y % 2 == 0 and x % 2 == 0
    cy, cx = y//2, x//2
    ycord, xcord = np.arange(y) - cy + 0.5, np.arange(x) - cx + 0.5
    xx, yy = np.meshgrid(xcord, ycord)
    rr = (xx**2 + (yy * distortion_factor)**2) ** (1/2)

    #r_bounds = np.arange(0, cy + 0.1, 10)
    r_centre = (r_bounds[1:] + r_bounds[:-1]) / 2
    N = len(r_bounds) - 1

    binned_profile = np.zeros(N)
    for i in range(N):
        r_lower = r_bounds[i]
        r_upper = r_bounds[i + 1]
        mask = (rr > r_lower) & (rr < r_upper)
        #plot(mask)
        values = im[mask]
        if len(values) == 0:
            print('Warning: R_bounds too closely spaced! Trying slower method')
            "When inclination is large and r_bounds is closely spaced, there may be no pixels that fall within an annulus. In this case, use the slower method which first transforms the image, effectively interpolating between the pixels."
            return get_binned_azimuthal_profile(im, r_bounds, inclination=0)
        else:
            binned_profile[i] = values.mean() / distortion_factor
        "Adjusted by distortion_factor to get value per pixel if viewed face-on"

    return binned_profile

def azimuthal_rings(rings, inclination=0):
    '''Apply MAKE_LINECUT to each element of the list RINGS'''
    return np.array([get_azimuthal_profile2(rings[i], inclination) for i in range(len(rings))])


def azimuthal_bin_rings(rings, r_bounds, inclination=0):
    '''Apply MAKE_LINECUT then BIN_LINECUT to each element of the array RINGS.'''
    return np.array([get_binned_azimuthal_profile2(rings[i], r_bounds, inclination) for i in range(len(rings))])