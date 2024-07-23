#from .lib import *
#from .short_functions import *
#from .linecut_functions import *


## Making rings
def make_ring(R, inner_radius, outer_radius, height=10, inclination=0, dim=200, kernel=None, n_points_per_pixel=200, scattering_phase=None):
    '''Returns an image of a ring (equivalently called an annulus). 
    All units are in pixels. 
    Input:
        R: a list containing the radius of each particle in the Monte-Carlo simulation. If this is input as NONE, then the code draws R from a uniform distribution. 
        INNER_RADIUS: inner boundary of ring. 
        OUTER_RADIUS: outer boundary of ring. 
        HEIGHT: standard deviation of the Gaussian distribution of the ring's vertical height. 
        INCLINATION: inclination for viewing the ring relative to face-on. 0 is face-on. 90 is edge-on. 
        DIM: dimensions of the image. Output image is always square. 
        KERNEL: the kernel to concolve the image with. None means don't convolve with anything. 
        N_POINTS_PER_PIXEL: how many points to add to the image per pixel if the ring were to be viewed face-on. The more points you add, the higher the accuracy of the image, but the slower the code will run. THIS MUST BE CONSISTENT WITH R IF R IS GIVEN. 
        SCATTERING_PHASE: scattering phase function. 2 by N array where N is the numbero of sample points to interpolate between. First row is azimuthal angle. Second row is scattering phase. Must have been normalised already at input. 
    '''
    
    if R is None:
        n_points = calculate_n_points(n_points_per_pixel, [inner_radius, outer_radius])[0]
        R = np.random.uniform(inner_radius, outer_radius, n_points)
        # Make face-on brightness uniform
        B_adjust = R / ((inner_radius + outer_radius)/2) / n_points_per_pixel
    else:
        n_points = len(R)
        B_adjust = 1/n_points_per_pixel
        
    
    Theta = np.random.uniform(0, 360, n_points)/180*np.pi
    "Avoid all points being binned to row of pixels above midplane when h = 0"
    if height == 0:
        height = 0.000001 
    Z = np.random.normal(0, height, n_points)
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    B = brightness(R) * B_adjust
    
    if inclination:
        zip = np.array((X, Y, Z))
        zip2 = rotate_x(inclination/180*np.pi) @ zip
    
        X, Y, Z = zip2[0], zip2[1], zip2[2]
    
    # Scattering phase function
    if np.any(scattering_phase):
        SPF = interpolate_SPF(Theta, inclination/180*np.pi, scattering_phase)
        B *= SPF
    else:
        pass
    
    # Bin into image
    image = np.zeros([dim, dim])
    
    if dim % 2 == 0:
        image_X = (np.round(X + dim//2 - 0.5)).astype(int)
        image_Y = (np.round(Y + dim//2 - 0.5)).astype(int)
    else:
        image_X = (np.round(X + dim//2)).astype(int)
        image_Y = (np.round(Y + dim//2)).astype(int)
    
    k = np.where((image_X >= 0) & (image_Y >= 0))
    image_X = image_X[k]
    image_Y = image_Y[k]
    for i in range(n_points):
        try:
            image[image_Y[i], image_X[i]] += B[i]
        except IndexError:
            pass
    
    # Convolve ring image
    if kernel is not None:
        image = convolve(image, kernel)
    
    return image


def calculate_n_points(n_points_per_pixel, r_bounds):
    '''Calculate number of points required for each ring defined by R_BOUNDS in order to achieve N_POINTS_PER_PIXEL if the image was viewed face-on. '''
    
    nrings = len(r_bounds) - 1
    ring_area = [np.pi*(r_bounds[i+1]**2 - r_bounds[i]**2) for i in range(nrings)]
    points_per_ring = np.array([int(ring_area[i] * n_points_per_pixel) for i in range(nrings)])
    
    return points_per_ring


def generate_R(n_points_per_pixel, r_bounds):
    '''The R distribution is a bit tricky to make up, so can achieve it by chopping up a set of points drawn from a triangular distribution. 
    This function generates r-coordinates for points in order to achieve N_POINTS_PER_PIXEL if the image was viewed face-on.'''
    
    points_per_ring = calculate_n_points(n_points_per_pixel, r_bounds)
    nrings = len(r_bounds) - 1
    
    # Try more points than total points to make sure each ring has enough points to match points_per_ring
    try_n_points = int(1.5 * n_points_per_pixel
        * np.pi * r_bounds[-1]**2)
    
    # Add more points if needed
    add_n_points = int(0.2 * try_n_points)
    
    # Draw from a triangular distribution
    Rs = np.random.triangular(0, r_bounds[-1], r_bounds[-1], try_n_points)
    R_per_ring = []
    
    for iring in range(nrings):
        # Don't try too many times
        while len(Rs) <= 10 * try_n_points:
            
            # Pick out points in ring
            R = Rs[(Rs >= r_bounds[iring]) & (Rs < r_bounds[iring+1])]
            
            # Take first points_per_ring if enough
            if len(R) >= points_per_ring[iring]:
                R_per_ring.append(R[ :points_per_ring[iring]])
                break
            
            # Otherwise add more points
            else:
                add_points = np.random.triangular(r_bounds[0], r_bounds[-1], r_bounds[-1], add_n_points)
                Rs = np.append(Rs, add_points)
                print('Added points')
                if len(Rs) > 10 * try_n_points:
                    print('Added too many points!')
    
    return R_per_ring


def make_all_rings(r_bounds, heights, inclination, dim, n_points_per_pixel=200, kernel=None, scattering_phase=None, verbose=True):
    '''Makes all rings defined by R_BOUNDS.
    Input variables are defined in MAKE_RING.'''
    
    # Adjust parameters
    nrings = len(r_bounds) - 1
    heights = expand(heights, nrings)
    
    # Get R coordinates
    R_per_ring = generate_R(n_points_per_pixel, r_bounds)
        
    # Storage parameters
    all_ring_images = np.zeros([nrings, dim, dim])
    
    # Make rings
    for iring in range(nrings-1, -1, -1):
        if verbose:
            print(iring, sep='', end=' ')
        ring_image = make_ring(R_per_ring[iring], None, None, heights[iring], inclination, dim, kernel, n_points_per_pixel, scattering_phase)
        all_ring_images[iring] = ring_image
    
    return all_ring_images



## Other rings functions
def generate_r_bounds(nrings, right_edge_pixel, left_edge_pixel=0, feature_region=None, min_width=None):
    '''Makes boundaries for NRINGS number of annuli between LEFT_EDGE_PIXEL and RIGHT_EDGE_PIXEL.
    Satisfied the conditions that make sure each annulus isn't too wide or narrow.
    FEATURE_REGION is a list with 3 elements. The first two are the lower and upper bounds of the region. The third is the number of annuli to split it into.'''
    assert right_edge_pixel > left_edge_pixel
    range_pixels = right_edge_pixel - left_edge_pixel

    #width_floor = max(0.1 * range_pixels / nrings, 1)
    #width_ceil = 5 * range_pixels / nrings
    '''
    while True:
        r_bounds = np.linspace(left_edge_pixel, right_edge_pixel, nrings+1)
        gap = (right_edge_pixel - left_edge_pixel) / nrings
        r_bounds[1:-1] += np.random.normal(0, gap, nrings-1)
        r_bounds = np.sort(r_bounds)
        diff = np.diff(r_bounds)
        if np.all(diff > 1) and r_bounds[0] == left_edge_pixel and r_bounds[-1] == right_edge_pixel:
            break
    '''
    
    if min_width == None:
        if nrings <= 20 and range_pixels/nrings >= 5:
            width_floor = max(0.3 * range_pixels / nrings, 1)
            width_ceil = 2 * range_pixels / nrings
        elif nrings <= 30:
            width_floor = max(0.2 * range_pixels / nrings, 1)
            width_ceil = 2.5 * range_pixels / nrings
        else:
            width_floor = 1
            width_ceil = 3 * range_pixels / nrings
    else:
        width_floor = min_width
        width_ceil = np.inf
    
    
    while True:
        r_bounds = np.random.uniform(left_edge_pixel, right_edge_pixel, nrings-1)
        r_bounds.sort()
        r_bounds = np.r_[left_edge_pixel, r_bounds, right_edge_pixel]
        r_widths = np.diff(r_bounds)
        if len(np.where((r_widths > width_ceil)|(r_widths < width_floor))[0]) == 0:
            break
    
    
    '''
    if np.any(feature_region):
        feature_lo, feature_hi = feature_region[0], feature_region[1]
        r_bounds_new = [r_bounds[0]]
        for i in range(len(r_bounds)-1):
            if not( np.all(r_bounds[i:i+2] < feature_lo) or np.all(r_bounds[i:i+2] > feature_hi) ) and (r_bounds[i+1] - r_bounds[i] > 2):
                "If the interval overlaps with feature_region by any amount at all"
                nsplit = 3
                r_bounds_new += list(np.linspace(r_bounds[i], r_bounds[i+1], num=nsplit+1)[1:])
            else:
                r_bounds_new.append(r_bounds[i+1])
        r_bounds = np.array(r_bounds_new)
    '''
    if np.any(feature_region):
        feature_lo, feature_hi = feature_region[0], feature_region[1]
        i_lower, i_upper = [], []
        for i in range(len(r_bounds)-1):
            if r_bounds[i] <= feature_lo and r_bounds[i+1] >= feature_lo:
                i_lower.append(i)
            if r_bounds[i] <= feature_hi and r_bounds[i+1] >= feature_hi:
                i_upper.append(i+1)
        assert len(i_lower) == 1 and len(i_upper) == 1
        
        nsplit = feature_region[2]
        assert (feature_region[1] - feature_region[0]) / nsplit >= 1, 'R_bounds too closely spaced'
        
        insert_r_bounds = generate_r_bounds(nsplit, right_edge_pixel=r_bounds[i_upper[0]], left_edge_pixel=r_bounds[i_lower[0]], feature_region=None, min_width=0.5)
        
        r_bounds = np.r_[r_bounds[:i_lower[0]], insert_r_bounds, r_bounds[i_upper[0]+1:]]
    
    return r_bounds


def get_r_bounds(nrings, right_edge_pixel, n_iterations, timeit=True, use_flux_range=False, feature_region=None):
    '''Reads in pre-generated R_BOUNDS if it exists. Otherwise makes a new set of R_BOUNDS with the right conditions and stores it. '''
    
    # Try stored rbounds
    import pickle
    prefix = 'Rbounds_'
    suffix = '.python'
    found_cache = 0
    
    if use_flux_range:
        assert len(right_edge_pixel) == 2
        if right_edge_pixel[0] == 0:
            left_edge_pixel = 0
            right_edge_pixel = right_edge_pixel[1]
            pixel_range = right_edge_pixel
        else:
            pixel_range = right_edge_pixel
            left_edge_pixel, right_edge_pixel = pixel_range
    else:
        left_edge_pixel = 0
        pixel_range = right_edge_pixel
    
    if not np.any(feature_region):
        for filename in os.listdir():
            if prefix in filename:
                
                # Remove prefix and suffix and get parameters
                filename2 = filename.replace(prefix, '').replace(suffix, '')
                nrings2, pixel_range2, n_interations2 = filename2.split('_')
                
                # Compare parameters
                if nrings == float(nrings2) and str(pixel_range) == pixel_range2 and n_iterations <= float(n_interations2):
                    print('    Found stored rbounds', filename)
                    found_cache = 1
                    file = open(filename, 'rb')
                    R_BOUNDS = pickle.load(file)
                    file.close()
                    break
    
    # Otherwise make rbounds
    if not found_cache and not np.any(feature_region):
        print('    Making rbounds')
        t0 = time.time()
        
        R_BOUNDS = np.zeros([n_iterations, nrings+1])
        for i in range(n_iterations):
            if nrings > 30:
                print(i, sep='', end=' ')
            R_BOUNDS[i] = generate_r_bounds(nrings, right_edge_pixel, left_edge_pixel, feature_region=feature_region)
        
        filename = f'{prefix}{nrings}_{pixel_range}_{n_iterations}{suffix}' 
        f = open(filename, 'wb')
        pickle.dump(R_BOUNDS, f)
        f.close()
        path = os.getcwd()
        print('    Cached to', path + '/' + filename)
        
        if timeit:
            print('    Time taken:', f'{(time.time() - t0):.0f}')
    
    elif np.any(feature_region):
        print('    Making rbounds')
        t0 = time.time()
        
        R_BOUNDS = []
        for i in range(n_iterations):
            if nrings > 30:
                print(i, sep='', end=' ')
            R_BOUNDS.append(generate_r_bounds(nrings, right_edge_pixel, left_edge_pixel, feature_region=feature_region))
        
        filename = f'{prefix}{nrings}_{pixel_range}_{n_iterations}{suffix}' 
        print(filename, 'R_bounds not cached because feature_region is used')
        
        if timeit:
            print('    Time taken:', f'{(time.time() - t0):.0f}')
    
    return R_BOUNDS


def get_narrow_annuli(r_outer, dr, height, inclination, dim, n_points_per_pixel=200, timeit=True, h_over_r=False, interpolate_height=False, scattering_phase=None):
    '''Narrow annuli are used to speed up simulating ring images, because adding up narrow rings is faster than making up new rings. 
    This function reads in pre-generated narrow annuli if they exist. Otherwise it makes a new set of annuli with the right conditions and stores them. '''
    
    if interpolate_height:
        # Vertically stretch/re-scale the set of annuli with the most similar height to approximate the annuli with the required height.
        
        "Print warning if disk is not edge-on"
        if np.abs(inclination - 90) > 5:
            print('Interpolating height over face-on disks, which gives wrong geometry!')
    
        "Find all available height values"
        hr_available = []
        H_available = []
    
        for filename in os.listdir():
            if 'RapidAnnuli' in filename:
                heightname = filename.split('_')[3]
                if 'hr' in filename:
                    hr_available.append(float(heightname[2:]))
                else:
                    H_available.append(float(heightname))
    
        hr_available = np.sort(hr_available)
        H_available = np.sort(H_available)
    
        if h_over_r:
            use_height = hr_available
            tag = 'h/r'
        else:
            use_height = H_available
            tag = 'H'
    
        "Find nearest available height value"
        nearest_heightval = use_height[abs(height - use_height) == min(abs(height - use_height))][0]
        print(f"Approximating {tag} = {height} using {nearest_heightval}")
        
        "Shuffle variables"
        required_height = height
        height = nearest_heightval
    
    # Try stored rings
    prefix = 'RapidAnnuli_'
    if np.any(scattering_phase):
        suffix = '_spf.python'
    else:
        suffix = '.python'
    found_cache = 0
    
    if h_over_r:
        heightname = f'hr{height}'
    else:
        if height % 1 == 0:
            height = int(height)
        heightname = str(height)
    
    if not np.any(scattering_phase):
        "Storing rapid annuli with scattering phase function not supported"
        
        import pickle
        
        for filename in os.listdir():
            if prefix in filename:
    
                # Remove prefix and suffix and get parameters
                filename2 = filename.replace(prefix, '').replace(suffix, '')
                dim2, n_points_per_pixel2, height2, inclination2, r_outer2, dr2 = [element for element in filename2.split('_')]
                
                # Compare parameters
                if [dim, heightname, inclination, dr] == [float(dim2), height2, float(inclination2), float(dr2)] and r_outer <= float(r_outer2):
                    print('    Found stored rings', filename2)
                    found_cache = 1
                    file = open(filename, 'rb')
                    RINGS = pickle.load(file)
                    file.close()
                    break
    
    # Otherwise make rings
    if np.any(scattering_phase) or not found_cache:
        print('    Making rapid rings')
        filename = f'{prefix}{dim}_{n_points_per_pixel}_{heightname}_{inclination}_{r_outer}_{dr}{suffix}'
        print('    ' + filename)
        t0 = time.time()
        
        r_bounds_make = np.arange(0, r_outer+dr, dr)
        r_centre = (r_bounds_make[1: ] + r_bounds_make[ :-1]) / 2
        
        if h_over_r:
            RINGS = make_all_rings(r_bounds_make, r_centre * height, inclination, dim, n_points_per_pixel, kernel=None, scattering_phase=scattering_phase)
        else:
            RINGS = make_all_rings(r_bounds_make, height, inclination, dim, n_points_per_pixel, kernel=None, scattering_phase=scattering_phase)
        
        # Store rings
        if np.any(scattering_phase):
            print('Rapid annnli not stored because scattering phase function is used')
            
        else:
            f = open(filename, 'wb')
            pickle.dump(RINGS, f)
            f.close()
            path = os.getcwd()
            print('Cached to', path + '/' + filename)
        
        if timeit:
            print('    Time taken:', f'{(time.time() - t0):.0f}')
    
    if interpolate_height:
        "Vertically scale the image of a ring while retaining the total flux"
        def zoom_height(image, old_height, new_height):
            ratio = new_height / old_height
            new_image = zoom(image, [ratio, 1])
            if new_image.shape[0] % 2 != image.shape[0] % 2:
                from scipy.ndimage import shift
                new_image = shift(new_image, [-0.5, 0])
            new_image = new_image / ratio
            new_image = reshape_image(new_image, *image.shape)
            return new_image
    
        "Apply to all rings"
        for i in range(len(RINGS)):
            RINGS[i] = zoom_height(RINGS[i], height, required_height)
    
    
    # Define ring-generating functions
    def rapid_ring(r_inner, r_outer, kernel=None):
        index_inner = round(r_inner/dr)
        index_outer = round(r_outer/dr)
        assert 0 <= index_inner <= len(RINGS) and 0 <= index_outer <= len(RINGS), 'Outside rapid range!'
        ring = np.sum(RINGS[index_inner:index_outer], axis=0)
        if callable(kernel):
            ring = kernel(ring)
        elif kernel is not None:
            ring = convolve(ring, kernel)
        return ring
    
    def rapid_rings(r_bounds, kernel=None):
        ring_images = []
        for i in range(1, len(r_bounds)):
            ring_images.append(rapid_ring(r_bounds[i-1], r_bounds[i], kernel))
        return ring_images
    
    return rapid_rings
    


## Fitting rings
def iterative_fit(target, components):
    '''Output should be identical to the matrix inversion method below but implemented in an iterative way. '''
    if type(target) is dict:
        assert type(components) is dict, 'Check input dictionary!'
        left = iterative_fit(target['left'], components['left'])
        right = iterative_fit(target['right'], components['right'])
        ratios = {'left': left[0], 'right': right[0]}
        residuals = {'left': left[1],'right': right[1]}
        return ratios, residuals
    
    n_iterations = 100
    fraction = 0.3 # How much closer each iteration gets
    
    nrings = len(components)
    all_ratios = np.zeros([n_iterations, nrings])
    residual = target.copy()
    
    for iteration in range(n_iterations):
        for iring in range(nrings-1, -1, -1):
            
            # Find ratio between image linecut and model linecut
            all_ratios[iteration, iring] = fraction * residual[iring] / components[iring, iring]
            
            # Modify redisual for next inner ring
            residual -= all_ratios[iteration, iring] * components[iring]
        
    ratios = np.sum(all_ratios, 0)
    #plot(all_ratios)
    return ratios, residual


def matrix_fit(target, components, show_matrix=False):
    '''Solves for the discretised face-on radial surface brightness given the discretised edge-on flux of modelled annuli and the observed edge-on flux of a disk. 
    EDGE-ON BRIGHTNESS OF ANNULI matrix * FACE-ON BRIGHTNESS column vector = EDGE-ON BRIGHTNESS column vector
    Input
        TARGET: a vector. Contains the observed discretised 1D flux. 
        COMPONENTS: a matrix. Contains all the modelled discretised 1D fluxes. Every column of the matrix is the discretised linecut of a ring. 
    Output
        WEIGHTS: the inverse of COMPONENTS multiplied with TARGET. 
        RESIDUALS: Always a vector of 0's since the matrix inversion method is exact. It is here to be syntactically consistent with ITERATIVE_FIT, which may have small residuals. 
    '''
    if type(target) is dict:
        assert type(components) is dict, 'Check input dictionary!'
        left = matrix_fit(target['left'], components['left'])
        right = matrix_fit(target['right'], components['right'])
        ratios = {'left': left[0], 'right': right[0]}
        residuals = {'left': left[1],'right': right[1]}
        return ratios, residuals
    
    nrings = len(components)
    matrix = np.zeros([nrings, nrings])
    
    for i in range(nrings):
        matrix[:, i] = components[i]
    
    inverted_matrix = np.linalg.inv(matrix)
    
    if False or show_matrix:
        plt.figure()
        plt.imshow(matrix)
        plt.title('Matrix')
        plt.colorbar()
        plt.show()
        
        plt.figure()
        plt.imshow(inverted_matrix)
        plt.title('Inverted matrix')
        plt.colorbar()
        plt.show()
    
    weights = inverted_matrix @ np.array(target)
    residual = np.zeros(nrings)
    return weights, residual
