from .lib import *
from .short_functions import *
from .ring_functions import *
from .linecut_functions import *


## Height calibrator and part generator
def make_generators(r_bounds, interp_heights, dim, RAPID_RINGS, kernel, cut_height=10, testing=True):
    '''This is run in preparation for FIT_HEIGHT2. 
    Returns
        HEIGHT_CALIBRATORS: a list containing the height-calibrating function of each annulus
        PART_GENERATORS: a list containing a function for each annulus that can quickly generate its midplane flux given a height. 
    
    Input
        R_BOUNDS: boundaries defining the annuli
        INTERP_HEIGHTS: the height values that are used to compute properly in order for the HEIGHT_CALIBRATORS and PART_GENERATORS to interpolate over. 
        DIM: dimensions of the image
        RAPID_RINGS: contains all the functions that let you rapidly generate images of rings. 
        KERNEL: kernel of the observation being fitted to. 
        CUT_HEIGHT: distance from midplaning defining the midplane region. 
        TESTING: always set this to TRUE. '''
    
    # Initialise
    nrings_make = len(r_bounds) - 1
    PART = handlelr(np.zeros)([len(interp_heights), nrings_make, dim//2])
    intp_range = handlelr(np.zeros)([nrings_make, 2])
    
    # Get data for each test height
    for i in range(len(interp_heights)):
        rings = RAPID_RINGS[interp_heights[i]](r_bounds, kernel)
        
        for j in range(nrings_make):
            ring = rings[j]
            part = make_linecut(ring, cut_height)
            PART['left'][i, j, :], PART['right'][i, j, :] = part['left'], part['right']

    # What height gives the right flux
    height_calibrators = createlr([], [])
    
    for j in range(nrings_make):
        maxparts = handlelr(np.zeros)(len(interp_heights))
        
        for i in range(len(interp_heights)):
            part = createlr(PART['left'][i, j, :], PART['left'][i, j, :])
            binned_part = bin_linecut(part, r_bounds)
            maxparts['left'][i], maxparts['right'][i] = binned_part['left'][j], binned_part['right'][j]
            
        f_height_left = interp1d(maxparts['left'], interp_heights, kind='linear')
        f_height_right = interp1d(maxparts['right'], interp_heights, kind='linear')
        height_calibrators['left'].append(f_height_left)
        height_calibrators['right'].append(f_height_right)
        
        intp_range['left'][j] = maxparts['left'][[0, -1]]
        intp_range['right'][j] = maxparts['right'][[0, -1]]

    # Generate linecut when height varied
    pixels, height = np.arange(dim//2), interp_heights
    part_generators = createlr([], [])
    
    for iring in range(nrings_make):
        part2d_left = np.array(PART['left'][:, iring, :])
        f_part2d_left = interp2d(pixels, height, part2d_left, kind='linear')
        part_generators['left'].append(f_part2d_left)
        
        part2d_right = np.array(PART['right'][:, iring, :])
        f_part2d_right = interp2d(pixels, height, part2d_right, kind='linear')
        part_generators['right'].append(f_part2d_right)
    
    if testing:
        return height_calibrators, part_generators, intp_range
    else:
        return height_calibrators, part_generators


## Apply ratio to flux, not height
def fit_height2(lim, radial_profile, rnew, r_bounds, height_calibrators, part_generators, H0, dim, intp_range=None):
    '''Fits the height given a set of annuli. 
    Returns
        H: a list containing the fitted height of each annulus
        LMOD: discretised midplane flux of the final fitted model
    
    Input
        LIM: discretised midplane flux of the image being fitted to 
        RADIAL_PROFILE: fitted surface brightness profile 
        RNEW: r-coordinates for RADIAL_PROFILE
        R_BOUNDS: boundaries defining the annuli
        HEIGHT_CALIBRATORS: see MAKE_GENERATORS
        PART_GENERATORS: see MAKE_GENERATORS
        H0: initial height at start of agorithm
        DIM: dimensions of the image
        INTP_RANGE: the "interpolation range", or the minimum and maximum fluxes of each annulus than can be mapped to its height. Mostly used for testing. '''
    
    if type(lim) is dict:
        left = fit_height2(lim['left'], radial_profile['left'], rnew, r_bounds, height_calibrators['left'], part_generators['left'], H0, dim, intp_range['left'])
        right = fit_height2(lim['right'], radial_profile['right'], rnew, r_bounds, height_calibrators['right'], part_generators['right'], H0, dim, intp_range['right'])
        H = createlr(left[0], right[0])
        lmod = createlr(left[1], right[1])
        return H, lmod
    
    # Interpolate from median radial profile
    delta = rnew[1] - rnew[0]
    radial_profile[radial_profile < 0] = 0
    F = bin_linecut(radial_profile * rnew, r_bounds/delta) / bin_linecut(rnew, r_bounds/delta)
    F2 = bin_linecut(radial_profile, r_bounds/delta)
    #print(F)
    #print(F2)
    
    # Parameters
    niterations = 100
    step_fraction = 0.3
    
    nrings = len(r_bounds) - 1
    H = np.zeros([niterations+1, nrings])
    lmod = np.zeros([niterations+1, nrings])
    pixels = np.arange(dim)
    
    # Initialise
    H[0, :] = H0
    previous_li = np.array([bin_linecut(part_generators[iring](pixels, H[0, iring]), r_bounds) for iring in range(nrings)])
    lmod[0, :] = weighted_sum(F, previous_li)
    
    
    for iter in range(1, niterations+1):
        # Before looping through rings, copy lmod from iter-1
        lmod[iter] = lmod[iter-1]
        
        for iring in range(nrings-1, -1, -1):
            # Calculate flux required of this ring if step_fraction of the residual flux comes from this ring
            target_flux = F[iring] * previous_li[iring, iring] + step_fraction * (lim[iring] - lmod[iter, iring])
            
            # Calculate height required to reach this flux
            try:
                if F[iring] > 0:
                    target_height = height_calibrators[iring](target_flux / F[iring])
                else:
                    #print(f'-> {iter}, {iring+1}, {H[iter-1, iring]} | Flux is {F[iring]}')
                    target_height = H[iter-1, iring]
            except ValueError:
                #print(f'-> {iter} r-{iring+1} | {H[iter-1, iring]:.1f}, {previous_li[iring, iring]:.1f} | {target_flux / F[iring]:.1f} [{intp_range[iring][0]:.1f}, {intp_range[iring][1]:.1f}]')
                if target_flux / F[iring] >= intp_range[iring][0]:
                    target_height = 0
                else:
                    target_height = H[iter-1, iring]
                '''
                if lmod[iter, iring] < lim[iring]:
                    target_height = H[iter-1, iring] - 1/step_fraction
                else:
                    target_height = H[iter-1, iring] + 1/step_fraction
                '''
            
            # Update height
            H[iter, iring] = target_height
            
            # Generate linecut for ring
            li = part_generators[iring](pixels, H[iter, iring])
            binned_li = bin_linecut(li, r_bounds)
            
            # Replace previous ring flux by current ring flux
            lmod[iter] = lmod[iter] - F[iring] * previous_li[iring] + F[iring] * binned_li
            previous_li[iring] = binned_li
    
    if 0:
        plt.figure()
        plt.plot(H)
        plt.show()
        
        plt.figure()
        plt.plot(lmod, alpha=0.8)
        plt.hlines(lim, 0, len(lmod)-1, ls='--')
        plt.show()
        
    return H, lmod