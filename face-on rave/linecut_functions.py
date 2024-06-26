#from .lib import *
#from .short_functions import *
from skimage.transform import warp_polar

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


def bin_linecut(linecut, r_bounds, annulus_weighted=True):
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
        if annulus_weighted:
            binned_linecut[i] = np.sum(linecut[ks] * coords[ks]) / np.sum(coords[ks])
        else:
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
    '''Don't use.
    Takes in an image and returns the azimuthally averaged radial profile assuming the given inclination.
    Stretches image then rotates image around to take vertical stripes.
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

def get_azimuthal_profile2(im, inclination=0, deltar=1, return_count=False, phi_range=None):
    '''Stretches image then counts flux in circle.'''
    if np.any(phi_range):
        return get_binned_azimuthal_profile2(im, r_bounds=np.arange(0, im.shape[1]//2+1, deltar), inclination=inclination, phi_range=phi_range)

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

    if return_count:
        return profile, count
    return profile

def get_binned_azimuthal_profile(im, r_bounds, inclination=0, phi_range=None):
    '''Stretches image then counts flux in circle. Equivalent to get_binned_azimuthal_profile2 but slower.'''

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
    phiphi = np.arctan2(yy, xx)
    phiphi[phiphi < 0] = phiphi[phiphi < 0] + 2 * np.pi
    phiphi = phiphi / np.pi * 180
    if np.any(phi_range):
        phi_mask = True
        for [phi_lo, phi_hi] in phi_range:
            phi_mask &= ~ ( (phiphi > phi_lo) & (phiphi < phi_hi) )
        if 0:
            phiphi_plot = phiphi.copy()
            phiphi_plot[~phi_mask] = -10
            plot(phiphi_plot)

    #r_bounds = np.arange(0, cy + 0.1, 1)
    r_centre = (r_bounds[1:] + r_bounds[:-1]) / 2
    N = len(r_bounds) - 1

    binned_profile = np.zeros(N)
    for i in range(N):
        r_lower = r_bounds[i]
        r_upper = r_bounds[i + 1]
        mask = (rr > r_lower) & (rr < r_upper)
        if np.any(phi_range):
            mask &= phi_mask
            #"Sometimes there are only a few central pixels in central radial bin and all of them are excluded by phi_range to give np.nan in the radial profile. In that case, assume those central pixels are fine to avoid numerical issues damaging the radial profile."
            #test_mask = mask &= phi_mask
            #if np.sum(mask &= phi_mask) > 0:
            #    mask = test_mask
        #plot(mask)
        values = distorted_image[mask]
        binned_profile[i] = values.mean()

    #plot(r_centre, binned_profile)

    return binned_profile

def get_binned_azimuthal_profile2(im, r_bounds, inclination=0, phi_range=None, interpolate_subpixel=False):
    '''Counts flux in ellipses. Equivalent to get_binned_azimuthal_profile but faster.
    Uniform sampling (e.g., sampling pixels) in projected image plane, as is done here <=> uniform sampling in de-projected disk plane <=> uniform averaging over phi in disk plane. All this is different from uniformly sampling phi' in projected image plane.
    phi_range shows the range of phi NOT to include when averaging over. Can take as input either None or a list of 2-element lists, with the elements corresponding to the lower and upper limit of each interval to exclude.'''

    if interpolate_subpixel:
        dr = 1
        r_bounds_1 = np.arange(0, im.shape[1]//2+dr, dr)
        r_centre_1 = bin_to_centre(r_bounds_1)

        dr1_profile = get_binned_azimuthal_profile2(im, r_bounds=r_bounds_1, inclination=inclination, phi_range=phi_range, interpolate_subpixel=False)

        flux_in_ring = dr1_profile * pi * (r_bounds_1[1:]**2 - r_bounds_1[:-1]**2)
        area_in_ring = pi * (r_bounds_1[1:]**2 - r_bounds_1[:-1]**2)

        int_flux_in_ring_0_to_r = np.r_[0, [np.sum(flux_in_ring[0:i]) for i in range(1, len(r_centre_1)+1)]]

        interpolated_int = np.interp(r_bounds, r_bounds_1, int_flux_in_ring_0_to_r)

        binned_profile = np.zeros(len(r_bounds)-1)
        for i in range(len(r_bounds)-1):
            lower_lim_flux, upper_lim_flux = interpolated_int[i], interpolated_int[i+1]
            area = np.pi * (r_bounds[i+1]**2 - r_bounds[i]**2)
            binned_profile[i] = (upper_lim_flux - lower_lim_flux) / area

        return binned_profile

    else:
        distortion_factor = 1/abs(np.cos(inclination/180*pi))

        y, x = im.shape
        assert y % 2 == 0 and x % 2 == 0
        cy, cx = y//2, x//2
        ycord, xcord = np.arange(y) - cy + 0.5, np.arange(x) - cx + 0.5
        xx, yy = np.meshgrid(xcord, ycord)
        rr = (xx**2 + (yy * distortion_factor)**2) ** (1/2)
        phiphi = np.arctan2(yy * distortion_factor, xx)
        phiphi[phiphi < 0] = phiphi[phiphi < 0] + 2 * np.pi
        phiphi = phiphi / np.pi * 180
        #plot(phiphi)
        #plt.contour(phiphi, colors='w', levels=np.linspace(0, 2*np.pi, 25))
        #plot(zoom(phiphi, [distortion_factor, 1]))
        #plt.contour(zoom(phiphi, [distortion_factor, 1]), colors='w', levels=np.linspace(0, 2*np.pi, 25))
        if np.any(phi_range):
            phi_mask = True
            for [phi_lo, phi_hi] in phi_range:
                phi_mask &= ~ ( (phiphi > phi_lo) & (phiphi < phi_hi) )
            if 0:
                phiphi_plot = phiphi.copy()
                phiphi_plot[~phi_mask] = -10
                plot(phiphi_plot)

        #r_bounds = np.arange(0, cy + 0.1, 10)
        r_centre = (r_bounds[1:] + r_bounds[:-1]) / 2
        N = len(r_bounds) - 1

        binned_profile = np.zeros(N)
        for i in range(N):
            r_lower = r_bounds[i]
            r_upper = r_bounds[i + 1]
            mask = (rr > r_lower) & (rr < r_upper)
            if np.any(phi_range):
                mask &= phi_mask

            #plot(mask)
            values = im[mask]
            if len(values) == 0:
                print('Warning: R_bounds too closely spaced! Trying slower method')
                "When inclination is large and r_bounds is closely spaced, there may be no pixels that fall within an annulus. In this case, use the slower method which first transforms the image, effectively interpolating between the pixels."
                return get_binned_azimuthal_profile(im, r_bounds, inclination=inclination, phi_range=phi_range)
            else:
                binned_profile[i] = values.mean() / distortion_factor
            "Adjusted by distortion_factor to get value per pixel if viewed face-on"

        return binned_profile

def get_binned_azimuthal_profile3(im, r_bounds, inclination=0):
    '''Transforms image into polar coordinates first.'''

    azimuthally_averaged_radial_profile = polar_transform_profile(im, inclination=inclination, mode='radial_profile')

    return bin_linecut(azimuthally_averaged_radial_profile, r_bounds)

def azimuthal_rings(rings, inclination=0):
    '''Apply MAKE_LINECUT to each element of the list RINGS'''
    return np.array([get_azimuthal_profile2(rings[i], inclination) for i in range(len(rings))])

def azimuthal_bin_rings(rings, r_bounds, inclination=0, phi_range=None, interpolate_subpixel=False):
    '''Apply MAKE_LINECUT then BIN_LINECUT to each element of the array RINGS.'''
    return np.array([get_binned_azimuthal_profile2(rings[i], r_bounds, inclination, phi_range=phi_range, interpolate_subpixel=interpolate_subpixel) for i in range(len(rings))])

def distort_image(image, inclination=0):
    if inclination:
        distortion_factor = 1/abs(np.cos(inclination/180*pi))
        distorted_image = zoom(image, [distortion_factor, 1])
        distorted_image = reshape_image(distorted_image, *image.shape) / distortion_factor
        return distorted_image
    else:
        return image.copy()

def polar_transform(image, inclination=0, phi_range=None):
    '''Transforms image from (x, y) to (r, phi) coordinates. phi=0 is to the right and goes counter-clockwise.'''
    distorted_image = distort_image(image, inclination=inclination)

    from skimage.transform import warp_polar
    y, x = distorted_image.shape
    transformed_image = warp_polar(distorted_image, radius=min(x, y)//2)

    if np.any(phi_range):
        mask_out = np.zeros(transformed_image.shape, dtype=bool)
        for interval in phi_range:
            mask_out[interval[0]:interval[1]] = 1

        return transformed_image, mask_out

    return transformed_image

def polar_transform_profile(image, inclination=0, mode='radial_profile', phi_range=None):
    '''Gives the azimuthally averaged or radially averaged profile using the polar transformed image.'''
    if np.any(phi_range):
        transformed_image, mask_out = polar_transform(image, inclination=inclination, phi_range=phi_range)

        if mode == 'radial_profile':
            include_rows = [irow for irow in range(len(mask_out)) if not np.any(mask_out[irow])]
            transformed_image = transformed_image[include_rows]
        else:
            transformed_image[mask_out] = np.nan

    else:
        transformed_image = polar_transform(image, inclination=inclination, phi_range=phi_range)

    if mode == 'radial_profile':
        return transformed_image.mean(0)
    elif mode == 'azimuthal_profile':
        return transformed_image.mean(1)
    else:
        raise KeyError('Mode for polar transformed profile not recognised')
