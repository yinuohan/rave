from .lib import *

## Brightness functions
def brightness(r):
    '''Defines the brightness of a particle as a function of distance from the star ("brightness function"). 
    Setting the brightness of each particle in the annulus to be the same, regardless of their distance from the star, is equivalent to fitting to the face-on surface *brightness* of the observation.
    Setting it to anything else is equivalent to fitting to the face-on surface *density* under the assumption that the brightness function is true.'''
    return np.ones(r.size)
    #return fancy_brightness(r)

def fancy_brightness(r):
    '''One possible brightness function. 
    This is not used in Rave by default.'''
    # Temperature
    star_luminosity = 8.7
    delta = 0.14
    A = 0.5
    #delta = 0.5
    #A = 1
    # Old A = 0.5, delta = 0.14
    # BB A = 1, delta = 0.5

    temperature = A * 278.3 * star_luminosity**0.25 * r**(-delta)

    # Blackbody (see P76 for more details)
    brightness = bnuw(wavelength, temperature)

    # Convert from Jy/sr to mJy/pixel
    brightness = brightness * (pixel*np.pi/648000)**2 * 1e3
    return brightness

def bnuw(wavelength, temperature):
    '''Blackbody function.'''
    # Give the constants for this function, where:
    # h = 6.6260755*10^{-34} Js is Planck's constant,
    # k = 1.380658*10^{-23} J/K is Boltzmann's constant,
    # c = 2.997924580*10^8 m/s is the speed of light.
    # wavelength in microns
    # temperature in K
    # Output is in Jy/sr

    k1 = 3.9728949e19 # = 2hc
    k2 = 14387.69     # = hc/k
    fact1 = k1 / wavelength**3
    fact2 = k2 / (wavelength*temperature)
    return fact1 / (np.exp(fact2) - 1)


## Rotation
def rotate_x(angle):
    '''Rotate (x, y, z) coordinates about the x axis'''
    C2 = np.zeros((3,3))

    C2[:,0] = [1, 0, 0]
    C2[:,1] = [0, np.cos(angle), np.sin(angle)]
    C2[:,2] = [0, -np.sin(angle), np.cos(angle)]
    return C2

def rotate_y(angle):
    '''Rotate (x, y, z) coordinates about the y axis'''
    C2 = np.zeros((3,3))

    C2[:,0] = [np.cos(angle), 0, -np.sin(angle)]
    C2[:,1] = [0, 1, 0]
    C2[:,2] = [np.sin(angle), 0,  np.cos(angle)]
    return C2

def rotate_z(angle):
    '''Rotate (x, y, z) coordinates about the z axis'''
    C2 = np.zeros((3,3))

    C2[:,0] = [np.cos(angle), np.sin(angle), 0]
    C2[:,1] = [-np.sin(angle), np.cos(angle), 0]
    C2[:,2] = [0, 0, 1]
    return C2


## Moving window
def moving_window(x, size=5):
    '''
    Input
        X: input array to smooth
        SIZE: size of moving window
    Returns 2 arrays of size len(x) - size + 1
        INDICES: The indices of the input array for which there is an output
        SMOOTHED: smoothed array
    '''
    x = np.array(x)
    dim = len(x)
    assert len(x.shape) == 1, 'Array to smooth must be 1D!'
    assert size <= dim, 'Window size must be at least length of array!'

    if size % 2 != 0:
        size = int(size)
        print('Adjusted window size to odd number!')

    left = size - 1
    right = dim

    shifted_x = np.zeros([size, dim + size - 1])
    for i in range(size):
        shifted_x[i, i:i+dim] = x

    smoothed = np.mean(shifted_x, 0)
    smoothed = smoothed[left:right]
    indices = np.arange(left - size//2, right - size//2) + 1
    return indices, smoothed


## Weighted sum
def weighted_sum(w, x):
    '''Sum of X weighted according to W'''
    assert len(w) == len(x)
    weighted = np.array([w[i] * x[i] for i in range(len(x))])
    return np.sum(weighted, 0)

weighted_sum_simple = weighted_sum

def weighted_sum(w, x):
    '''Sum of X weighted according to W.
    If X and/or W is a dictionary with keys 'left' and 'right', then also return a dictionary that calculates 'left' and 'right' independently.'''
    if type(w) is dict:
        if type(x) is dict:
            return {
            'left': weighted_sum_simple(w['left'], x['left']),
            'right': weighted_sum_simple(w['right'], x['right'])
            }
        else:
            return {
            'left': weighted_sum_simple(w['left'], x),
            'right': weighted_sum_simple(w['right'], x)
            }
    elif type(x) is dict:
        return {
        'left': weighted_sum_simple(w, x['left']),
        'right': weighted_sum_simple(w, x['right'])
        }
    else:
        return weighted_sum_simple(w, x)


## Dictionaries
def handlelr(function):
    '''Returns a new function such that, if there is an argument that is a dictionary, which consists of keys LEFT and RIGHT, the new function repeats the function for each of the values and outputs a dictionary with keys LEFT and RIGHT'''
    def new_function(*args, **kwargs):
        left_args = list(args)
        right_args = list(args)

        #d = None
        for i in range(len(args)):
            if type(args[i]) is dict:
                left_args[i], right_args[i] = args[i]['left'], args[i]['right']
        #assert d is not None, "Can't find dictionary to handle!"
        
        return {
        'left': function(*left_args, **kwargs),
        'right': function(*right_args, **kwargs)
        }
        
    return new_function

import matplotlib.pyplot as plt
plotlr = handlelr(plt.plot)
steplr = handlelr(plt.step)

def pltstep(x, y, *args, **kwargs):
    '''Modified version of plt.step
    X defines the boundaries of the steps
    Y defines the height of each step
    Requires SIZE(X) = SIZE(Y) + 1'''
    return plt.step(x, np.r_[y[0], y], *args, **kwargs)

def createlr(*args):
    '''Creates a dictionary with keys LEFT and RIGHT.
    If input one argument, then both keys have the same value. 
    If input two arguments, then assign them to LEFT and RIGHT respectively.'''
    assert len(args) <= 2
    if len(args) == 1:
        return {
        'left': args[0],
        'right': args[0]
        }
    else:
        return {
        'left': args[0],
        'right': args[1]
        }

def meanlr(x):
    '''Takes in a dictionary with keys LEFT and RIGHT and returns the mean of the values of the two keys'''
    return (x['left'] + x['right']) / 2

def divlr(x, y):
    '''Takes in a dictionary X with keys LEFT and RIGHT and divides the value of each key by Y'''
    return {
        'left': x['left']/y,
        'right': x['right']/y
        }

## Other functions
def yzoom(image, factor):
    '''Elongate the image along the y axis by a factor of FACTOR.'''
    image2 = zoom(image, [factor, 1])
    ydim, xdim = image2.shape
    if factor >= 1:
        cy, cx = ydim//2, xdim//2
        image2 = image2[cy - cx:cy + cx]
        assert image.shape == image2.shape
    else:
        #print(cy, cx)
        pass
    return image2


def divide(x, y):
    '''Takes in two arrays X and Y. 
    Returns Xi/Yi for each element only if Yi is not a very small number.
    Otherwise return 0 for that element. '''
    assert len(x) == len(y)
    ans = np.zeros(len(x))
    for i in range(len(x)):
        ans[i] = x[i] / y[i] if abs(y[i]) > 1e-10 else 0
    return ans


def expand(x, n_elements):
    '''If X is only a number, duplicate X into an array of length N_ELEMENTS.
    If X is already an array of length N_ELEMENTS then don't touch it. 
    Otherwise, raise an error. '''
    if type(x) in [int, float, np.float64, np.float32, np.float16, np.int64, np.int32]:
        return np.ones(n_elements) * x
    else:
        assert len(x) == n_elements
        return x

def floor(x):
    '''Floors all values in a dictionary with keys LEFT and RIGHT to 0'''
    if type(x) == dict:
        return createlr(floor(x['left']), floor(x['right']))
    assert len(x) # Make sure x has a length
    x[x < 0] = 0
    return x

def cut(image, ymax):
    '''Cut out the region of image within YMAX away from the central horizontal line'''
    y, x = image.shape
    image[ :y//2 - ymax] = 0
    image[y//2 + ymax: ] = 0
    return image

def store(variable, filename):
    '''Store VARIABLE into FILENAME with pickle'''
    import pickle
    import os
    
    f = open(filename, mode='wb')
    pickle.dump(variable, f)
    f.close()
    print('Stored at:', os.getcwd()+ '\\' + filename)


def load(filename):
    '''Load a pickled variable'''
    import pickle
    f = open(filename, mode='rb')
    variable = pickle.load(f)
    f.close()
    return variable
    

## Make kernel
def make_kernel(dim, sigma):
    '''Makes a circular Gaussian kernel with dimensions of DIM * DIM and a standard deviation of SIGMA'''
    from scipy.ndimage import gaussian_filter
    
    assert dim % 2 == 1, 'Kernel dimensions must be odd!'
    kernel = np.zeros([dim, dim])
    
    # Add a point and smooth
    kernel[dim//2, dim//2] = 1
    kernel = gaussian_filter(kernel, sigma)
    
    # Normalise
    kernel = kernel/kernel.sum()
    
    return kernel

def make_kernel2(dim, fwhm_x, fwhm_y, theta=0):
    '''Makes an elliptical Gaussian kernel with dimensions DIM * DIM and a fwhm of FWHM_X and FWHM_Y in the x and y directions respectively. 
    Then rotates the whole kernel by THETA degrees.'''
    def g(x, y, sx, sy):
        return np.exp( - (x/sx)**2/2 - (y/sy)**2/2 )
    
    # Grid coordinates
    dim2 = 2 * dim
    x = np.arange(-dim2//2, dim2//2+1)
    xx, yy = np.meshgrid(x, x)
    
    # Make kernel
    kernel = g(xx, yy, fwhm_x/2.355, fwhm_y/2.355)
    
    # Rotate
    kernel = rotate(kernel, theta)
    
    # Re-centre
    y, x = kernel.shape        
    by, bx = brightest_pixel(kernel)
    #if y % 2 == 0 and x % 2 == 0:
    #    from scipy.ndimage import shift
    #    kernel = shift(kernel, ())
    kernel = kernel[by-dim//2:by+dim//2+1, bx-dim//2:bx+dim//2+1]
    kernel /= kernel.sum()
    
    return kernel

def calculate_beam_area(fwhm_x, fwhm_y):
    '''Calculates the equivalent area of an ellitical Gaussian kernel. 
    FWHM_X can be a list of length 2, in which case FWHM_Y is ignored.'''
    if type(fwhm_x) == list:
        assert len(fwhm_x) == 2
        return calculate_beam_area(fwhm_x[0], fwhm_x[1])
    return np.pi / 4 / np.log(2) * fwhm_x * fwhm_y

def pythagoras(a, b):
    '''Calculates the length of the hypotenuse'''
    return np.sqrt(a**2 + b**2)


## Interpolate
def interpolate(RATIOS, R_BOUNDS, newpoints=1000):
    '''
    Turns a step-like function defined by only a few points into a more continuous functions by adding more sample points. 
    Input
        RATIOS: a dictionary with keys LEFT and RIGHT. Each key has an array of size N_ITERATIONS * NRINGS. 
        R_BOUNDS: a dictionary with keys LEFT and RIGHT. Each key has an array of size (NRINGS + 1). 
    Output
        RNEW: an array of length NEWPOINTS linearly spaced between the smallest and largest values of R_BOUNDS. 
        INTERPOLATED: interpolated version of RATIOS. Each key has an array of size N_ITERATIONS * NEWPOINTS.''' 
    rnew = np.linspace(R_BOUNDS[:,0].max(), R_BOUNDS[:,-1].min(), newpoints)
    assert type(RATIOS) is dict, 'RATIOS needs to be dict'
    n_iterations = RATIOS['left'].shape[0]
    INTERPOLATED = handlelr(np.zeros)([n_iterations, newpoints])
    
    for i in range(n_iterations):                
        f = interp1d(R_BOUNDS[i], np.r_[RATIOS['left'][i], RATIOS['left'][i, -1]], kind='zero')
        INTERPOLATED['left'][i] = f(rnew)
    
        f = interp1d(R_BOUNDS[i], np.r_[RATIOS['right'][i], RATIOS['right'][i, -1]], kind='zero')
        INTERPOLATED['right'][i] = f(rnew)
    return rnew, INTERPOLATED
