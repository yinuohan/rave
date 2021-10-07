from .lib import *
from .short_functions import *
from .height_functions import *
from .ring_functions import *
from .linecut_functions import *


class Image():
    '''Use this class to fit to an observation. '''
    
    def __init__(self, image, kernel, scale=1, distance=None, pixel=None):
        '''Create image object and specify SCALE. 
        Input
            IMAGE: 2D array representing observed image
            KERNEL: 2D array representing the PSF/beam of the observation
            SCALE: AU per pixel
            DISTANCE: distance to target in pc
            PIXEL: arcsec per pixel'''
        
        self.image = image
        self.kernel = kernel
        
        # Scale
        if distance and pixel:
            scale = distance * pixel
        
        self.scale = scale # AU/pixel
        self.distance = distance # pc
        self.pixel = pixel # arcsec/pixel
        
        # Initialise
        if image is not None:
            self.initialise()
            self.is_fake_image = False
    
    def initialise(self):
        '''Initialises IMAGE object by normalising the kernel and setting up a few variables'''
        # Dimensions
        self.ydim, self.xdim = self.image.shape # pixels
        self.cx, self.cy = self.xdim//2, self.ydim//2 # pixels
        if self.xdim % 2 != 0 or self.ydim % 2 != 0:
            print('Image dimensions not even')
        self.dim = self.xdim
        
        # Normalise kernel
        if not np.any(self.kernel == None):
            self.kernel /= self.kernel.sum()
        
        # Distance in AU
        self.R = (0.5 + np.arange(self.cx)) * self.scale # AU
        
        # Linecut
        self.linecut = make_linecut(self.image, 'full')
        
        # Other status
        if not hasattr(self, 'add_blur'):
            self.add_blur = 0
        if not hasattr(self, 'flipped'):
            self.flipped = False
    
    def add_flip(self, rotate=True):
        '''Sums the image and kernel with their repestive 180-degree rotated images'''
        if rotate:
            self.image = (self.image + rotate180(self.image)) / 2
            self.kernel = (self.kernel + rotate180(self.kernel)) / 2
        else:
            self.image = (self.image + np.fliplr(self.image)) / 2
            self.kernel = (self.kernel + np.fliplr(self.kernel)) / 2
        self.initialise()
        self.flipped = True
    
    def smooth(self, sigma=2):
        '''Smoothes the image and kernel with a circular Gaussian kernel with a standard deviation of SIGMA'''
        if sigma > 0:
            assert self.add_blur == 0, 'Already blurred!'
            self.image = blur(self.image, sigma)
            self.kernel = blur(self.kernel, sigma)
            self.kernel /= self.kernel.sum()
            self.initialise() # redo linecut
            self.add_blur = sigma
        else:
            print('Skipped smoothing')
    
    def flux_region(self, y_max, mode='cut'):
        '''Defines the region of the image that actually has flux. This is specified as the region within Y_MAX from midplane. 
        MODE can be "cut" or "specify".
            If MODE is 'SPECIFY', then only specify this region, which is subsequently used for noise calculation.
            If MODE is 'CUT', then also sets everthing outside the flux region to 0. This ignores the region with only background noise to improve S/N for subsequent fitting. '''
        self.y_max = y_max
        
        if mode == 'cut':
            self.image[ :self.cy - self.y_max] = 0
            self.image[self.cy + self.y_max: ] = 0
        
    
    def noise_by_snr(self, snr_per_beam, beam_fwhm, mode='add'):
        '''Calculates noise/pixel and noise/beam given snr/beam and beam_fwhm.
        MODE can be "add" or "specify".
            If MODE is 'SPECIFY', then only perform noise calculation.
            If MODE is 'ADD', then add this amount of noise to the image. '''
        self.snr_per_beam = snr_per_beam
        self.beam_fwhm = beam_fwhm
        
        # Number of beams required to cover the disk
        self.beam_area = calculate_beam_area(beam_fwhm, beam_fwhm)
        self.flux_pixels = self.xdim * 2 * self.y_max
        self.n_beams = self.flux_pixels / self.beam_area
        
        # Flux per beam
        total_flux = self.image.sum()
        flux_per_beam = total_flux / self.n_beams
        
        # RMS noise per beam
        self.noise_per_beam = flux_per_beam / self.snr_per_beam
        
        # RMS noise per pixel
        self.noise_per_pixel = self.noise_per_beam / np.sqrt(self.beam_area)
        
        # Add noise
        if mode == 'add':
            if self.noise_per_pixel > 0:
                noise = np.random.normal(0, self.noise_per_pixel, self.image.shape)
                self.image += noise
        
        self.kernel_noise = 0
        self.initialise()
    
    def noise_by_rms(self, image_noise, beam_fwhm=None, reference_area='pixel', mode='add'):
        '''Calculates snr/beam given (noise/pixel or noise/beam) and  beam_fwhm.
        REFERENCE_AREA can be "pixel" or "beam". 
        MODE can be "add" or "specify".
            If MODE is 'SPECIFY', then only perform noise calculation.
            If MODE is 'ADD', then add this amount of noise to the image.'''
        
        self.beam_fwhm = beam_fwhm
        
        if self.beam_fwhm:
            # Number of beams required to cover the disk
            self.beam_area = calculate_beam_area(beam_fwhm, beam_fwhm)
            self.flux_pixels = self.xdim * 2 * self.y_max
            self.n_beams = self.flux_pixels / self.beam_area
            
            # Flux per beam
            total_flux = self.image.sum()
            flux_per_beam = total_flux / self.n_beams
            
            # RMS noise per beam
            if reference_area == 'pixel':
                self.noise_per_pixel = image_noise
                self.noise_per_beam = self.noise_per_pixel * np.sqrt(self.beam_area)
            else:
                assert reference_area == 'beam'
                self.noise_per_beam = image_noise
                self.noise_per_pixel = self.noise_per_beam / np.sqrt(self.beam_area)
            
            # SNR per beam
            self.snr_per_beam = flux_per_beam / self.noise_per_beam
        
        else:
            assert reference_area == 'pixel'
            self.noise_per_pixel = image_noise
        
        # Add noise
        if mode == 'add':
            if self.noise_per_pixel > 0:
                noise = np.random.normal(0, self.noise_per_pixel, self.image.shape)
                self.image += noise
        
        self.kernel_noise = 0
        self.initialise()

    def store(self, filename):
        '''Stores IMAGE OBJECT with pickle'''
        store_image = StoreImage(self)
        store(store_image, filename)
    
    def plot(self, unit='pixel'):
        '''Plots image'''
        cy2, cx2 = self.cy + 0.5, self.cx + 0.5
        unit_label = 'pixels'
        
        if unit == 'au':
            cx2 *= self.scale
            cy2 *= self.scale
            unit_label = 'au'
        elif unit == 'beam':
            cx2 /= self.beam_fwhm
            cy2 /= self.beam_fwhm
            unit_label = 'beam FWHMs'
        plt.figure()
        plt.imshow(self.image, extent=[-cx2, cx2, -cy2, cy2], origin='lower')
        
        plt.xlabel(f'Relative RA ({unit_label})')
        plt.ylabel(f'Relative Dec ({unit_label})')
        #plt.tight_layout()
        plt.show()
    
    def make_model(self, inclination=90, heights=0, h_over_r=True, n_points_per_pixel=200, rapid=True, use_kernel=True, add_before_convolve=True, default_height=None):
        '''Makes a model of the image.
        Must be performed after radial profile is fitted so that SELF.RADIAL.PROFILE and SELF.RADIAL.RNEW exist. 
        Input
            INCLINATION: inclination to view the model image from. 0 is face-on. 90 is edge-on. 
            HEIGHTS: height of each annulus. 
                If HEIGHT is only one number, it means the height is the same everywhere. 
                If HEIGHT is NONE, the fitted height is used. Must have already obtained SELF.HEIGHT.PROFILE in this case. 
            H_OVER_R: TRUE or FALSE. If TRUE, use HEIGHTS as the aspect ratio instead of the absolute height. 
            POINTS_PER_PIXEL: number of sample points per pixel if the image was viewed edge-on for the Monte Carlo method to simulate annuli images. 
            RAPID: TRUE or FALSE. If TRUE, make model by summing up pre-generated narrow annuli. If FALSE, use Monte Carlo method. 
            USE_KERNEL: TRUE or FALSE. If True, convolve with the PSF/beam.
            ADD_BEFORE_CONVOLVE: TRUE or FALSE. If TRUE, add unconvolved narrow annuli first before convolving the final image with the PSF/beam.
            DEFAULT_HEIGHT: if HEIGHTS is NONE and SELF.HEIGHTS.PROFILE is not fitted for the entire range in r, then use DEFAULT_HEIGHT as the height everywhere not fitted to. 
            '''
        
        # Use bin_linecut as a binning function
        r_bounds_make = np.arange(0, self.cx + 1)
        profile = (self.radial.profile['left'] + self.radial.profile['right']) / 2
        weights_make = bin_linecut(profile, r_bounds_make * len(profile) / (len(r_bounds_make) - 1))
        if heights != None:
            heights_make = heights
            if h_over_r:
                rapid = False
                r = (r_bounds_make[:-1] + r_bounds_make[1:]) / 2
                heights_make = heights * r
        else:
            rapid = False
            r = self.height.rnew
            r_bounds_2 = np.arange(0, int(r[-1]) - int(r[0]) + 1)
            H = (self.height.profile['left'] + self.height.profile['right']) / 2
            H = bin_linecut(H, r_bounds_2 * len(H) / (len(r_bounds_2) - 1))
            heights_make = np.zeros(self.cx)
            heights_make[int(r[0]):int(r[-1])] = H
            if default_height == None:
                heights_make[ :int(r[0])] = self.height.starting_height
                heights_make[int(r[-1]): ] = self.height.starting_height
            elif default_height == 'interpolate':
                heights_make[ :int(r[0])] = H[0]
                heights_make[int(r[-1]): ] = H[-1]
            else:
                heights_make[ :int(r[0])] = default_height
                heights_make[int(r[-1]): ] = default_height
        
        if use_kernel:
            kernel = self.kernel
        else:
            kernel = None
        
        self.model = MakeImage(r_bounds_make, weights_make, heights_make, inclination, self.xdim, n_points_per_pixel, kernel, scale=self.scale, rapid=rapid, add_before_convolve=add_before_convolve)
    
    def plot_compare(self, cut_height=10, unit='pixel'):
        '''Makes a plot to compare the full 1D flux and midplane flux between the observation and best-fit model.'''
        if unit == 'pixel':
            r = np.arange(self.cx)
            unit_label = 'pixels'
        elif unit == True or unit == 'au':
            r = np.arange(self.cx) * self.scale
            unit_label = 'au'
        elif unit == 'beam':
            assert self.beam_fwhm != None
            r = np.arange(self.cx) / self.beam_fwhm
            unit_label = 'beam FWHMs'
        
        plt.figure()
        plt.plot(r, meanlr(make_linecut(self.image, 'full')), '--', label='Data: full flux')
        plt.plot(r, meanlr(make_linecut(self.model.image, 'full')), label='Model: full flux')
        plt.plot(r, meanlr(make_linecut(self.image, cut_height)), '--', label='Data: midplane flux')
        plt.plot(r, meanlr(make_linecut(self.model.image, cut_height)), label='Model: midplane flux')
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel('1D flux')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    
    def fit_inclination(self, cut_height=10, inc_range=[0, 90], guess=58, delta=1, unit='pixel'):
        '''Finds a lower bound for the inclination.
        Must have already fitted SELF.RADIAL.PROFILE.'''
        
        if unit == 'pixel':
            r = np.arange(self.cx)
            unit_label = 'pixels'
        elif use_au == True:
            r = np.arange(self.cx) * self.image.scale
            unit_label = 'au'
        elif use_au == 'beam':
            assert self.beam_fwhm != None
            r = np.arange(self.cx) / self.beam_fwhm
            unit_label = 'beam FWHMs'
        
        lim = make_linecut(self.image, cut_height)
        lim = meanlr(lim)
        plt.figure()
        plt.plot(r, lim, label='Data')
        
        while inc_range[1] - inc_range[0] > 2:
            self.make_model(inclination=guess, heights=0, add_before_convolve=True)
            lmod = make_linecut(self.model.image, cut_height)
            lmod = meanlr(lmod)
            plt.plot(r, lmod, label=guess)
            
            if np.sum(lim > lmod + delta) > 0:
                inc_range[0] = guess
            else:
                inc_range[1] = guess
            guess = int(np.round(np.mean(inc_range)))
        
        print('Lower bound for inclination:', inc_range)
        self.inclination = inc_range[1]
        
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel('Midplane flux')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
    
    def estimate_inclination(self, cut_height=10, inc_range=[78, 90], di=2, unit='pixel'):
        '''Finds a lower bound for the inclination.
        Must have already fitted SELF.RADIAL.PROFILE.'''
        
        if unit == 'pixel':
            r = np.arange(self.cx)
            unit_label = 'pixels'
        elif use_au == True:
            r = np.arange(self.cx) * self.image.scale
            unit_label = 'au'
        elif use_au == 'beam':
            assert self.beam_fwhm != None
            r = np.arange(self.cx) / self.beam_fwhm
            unit_label = 'beam FWHMs'
        
        lim = make_linecut(self.image, cut_height)
        lim = meanlr(lim)
        plt.figure()
        plt.plot(r, lim, label='Data')
        
        inc_vals = np.arange(inc_range[0], inc_range[1]+di, di)
        
        for inc in inc_vals:
            self.make_model(inclination=inc, heights=0, add_before_convolve=True)
            lmod = make_linecut(self.model.image, cut_height)
            lmod = meanlr(lmod)
            plt.plot(r, lmod, label=inc)
        
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel('Midplane flux')
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()


class MakeImage(Image):
    '''Use this class to make up an image for testing'''
    
    def __init__(self, r_bounds_make, weights_make, heights_make, inclination_make, dim, n_points_per_pixel, kernel, scale=1, rapid=False, add_before_convolve=True, verbose=True):
        '''Specify parameters used to make up the image.
        All units are in pixels. 
        Input:
            R_BOUNDS_MAKE: boundaries defining the annuli. 
            WEIGHTS_MAKE: face-on surface brightness of each annulus. 
            HEIGHTS_MAKE: standard deviation of the Gaussian distribution of  the vertical height of each annulus. 
            INCLINATION_MAKE: inclination to view the image from. 0 is face-on. 90 is edge-on. 
            DIM: dimensions of the image. Output image is always square. 
            N_POINTS_PER_PIXEL: how many points to add to the image per pixel if the ring were to be viewed face-on. The more points you add, the higher the accuracy of the image, but the slower the code will run. THIS MUST BE CONSISTENT WITH R IF R IS GIVEN. 
            KERNEL: the kernel to concolve the image with. None means don't convolve with anything. 
            SCALE: AU per pixel.  
            RAPID: TRUE or FALSE. If TRUE, make model by summing up pre-generated narrow annuli. If FALSE, use Monte Carlo method. 
            ADD_BEFORE_CONVOLVE: TRUE or FALSE. If TRUE, add unconvolved narrow annuli first before convolving the final image with the PSF/beam.
        '''
            
        self.r_bounds_make = r_bounds_make
        self.weights_make = weights_make
        self.heights_make = heights_make
        self.inclination_make = inclination_make
        self.dim = dim
        self.n_points_per_pixel = n_points_per_pixel
        self.scale = scale
        
        # Make image
        if verbose:
            print('Making image')
        if not rapid:
            t0 = time.time()
            if not add_before_convolve:
                rings_make = make_all_rings(r_bounds_make, heights_make, inclination_make, dim, n_points_per_pixel, kernel, verbose=verbose)
            else:
                rings_make = make_all_rings(r_bounds_make, heights_make, inclination_make, dim, n_points_per_pixel, None, verbose=verbose)
            if verbose:
                print('Time taken:', f'{(time.time() - t0):.0f}')
        else:
            self.get_rapid_annuli()
            if not add_before_convolve:
                rings_make = self.rapid_rings(r_bounds_make, kernel)
            else:
                rings_make = self.rapid_rings(r_bounds_make, None)
        
        image = weighted_sum(weights_make, rings_make)
        if add_before_convolve and not np.any(kernel == None):
            image = convolve(image, kernel)
        
        self.image = image
        self.kernel = kernel
        self.rings_make = rings_make
        
        # Initialise
        self.initialise()
        self.is_fake_image = True
    
    def get_rapid_annuli(self, dr=0.1):
        '''Load the rapid annuli used to generate the fake image.'''
        print('\n----- Make image annuli -----')
        
        self.rapid_rings = get_narrow_annuli(self.dim//2, dr, self.heights_make, self.inclination_make, self.dim, self.n_points_per_pixel)
    

class LoadImage(MakeImage):
    '''Use this class to load a stored Image Object'''

    def __init__(self, filename, scale=None, kernel=None, fitted=False):
        '''Specify SCALE and KERNEL. 
        Input
            FILENAME: the location and name of the stored image to load. 
            SCALE: AU per pixel. 
            KERNEL: PSF/beam of the observation. 
            FITTED: whether or not this image object contains fitted profiles. '''
        
        loaded_image = load(filename)
        self.image = loaded_image.image
        self.kernel = loaded_image.kernel
        
        # Store variables
        if not fitted:
            self.r_bounds_make = loaded_image.r_bounds_make
            self.weights_make = loaded_image.weights_make
            self.heights_make = loaded_image.heights_make
            self.inclination_make = loaded_image.inclination_make
        self.scale = scale if scale else loaded_image.scale
        
        # Convolve
        if np.any(kernel != None):
            assert np.all(loaded_image.kernel == None)
            self.kernel = kernel / kernel.sum()
            self.image = convolve(self.image, self.kernel)
        
        # Initialise
        self.initialise()
        self.is_fake_image = True


class Empty():
    '''Empty class'''
    pass


class StoreImage():
    '''Use this class to create an image object that can be stored.'''
    
    def __init__(self, image_object):
        '''Creates a new image object with the same attributes except ones that can't be pickled'''
        for x in image.__dict__.keys():
            if x == 'radial':
                self.radial = Empty()
                for y in image.radial.__dict__.keys():
                    if y not in ['rapid_rings', 'image']:
                        val = getattr(image.radial, y)
                        setattr(self.radial, y, val)
            
            elif x == 'height':
                self.height = Empty()
                for z in image.height.__dict__.keys():
                    if z not in ['RAPID_RINGS', 'image']:
                        val = getattr(image.height, z)
                        setattr(self.height, z, val)
            
            else:
                if x not in ['model', 'rings_make']:
                    val = getattr(image, x)
                    setattr(self, x, val)


class RadialProfile():
    '''Radial Profile Fitter class. Here "Radial Profile" = "surface brightness profile".'''
    
    def __init__(self, image):
        '''Links the Image object with the Radial Profile Fitter obeject'''
        image.radial = self
        self.image = image
    
    def quick_fit(self, nrings=7, r_outer=None, fit_method='matrix', r_bounds='even', ring_method='rapid'):
        '''Does a quick fit to the image with only one iteration. 
        Input
            NRINGS: number of annuli to use. 
            R_OUTER: fit up to this radial location. 
            FIT_METHOD: 'matrix' or 'iterative'.
            R_BOUNDS: boundaries defining the annuli. If 'even', then evenly spaces the annuli. 
            RING_METHOD: if 'rapid' then makes rings by summing up narrow annuli. Otherwise makes up rings from scratch using Monte Carlo. '''
        print('\n----- Radial Quick Fit -----')
        
        t0 = time.time()
        
        if r_bounds == 'even':
            if r_outer == None:
                if hasattr(self, 'r_outer'):
                    r_outer = self.r_outer
                else:
                    r_outer = self.image.cx
            r_bounds = np.linspace(0, r_outer, nrings+1)
        
        if fit_method == 'matrix':
            function = matrix_fit
        else:
            function = iterative_fit
        
        if ring_method == 'rapid':
            rings = self.rapid_rings(r_bounds, self.image.kernel)
        else:
            print('    Making fitting annuli')
            t0 = time.time()
            rings = make_all_rings(r_bounds, heights=10, inclination=90, dim=self.image.xdim, n_points_per_pixel=200, kernel=self.image.kernel)
            print('    Time taken:', f'{(time.time() - t0):.0f}')
        
        self.quick_abrl = bin_rings(rings, r_bounds, 'full')
        L = bin_linecut(self.image.linecut, r_bounds)
        self.quick_profile, _ = function(L, self.quick_abrl)
        self.quick_r_bounds = r_bounds
        self.quick_nrings = nrings
        
        print('Time taken:', f'{(time.time() - t0):.0f}')
    
    def quick_plot(self, unit='pixel', average=False):
        '''Makes a quick plot of the QUICK_FIT. 
        UNIT can be 'pixel', 'au' or 'beam'. '''
        
        if unit == 'pixel':
            r1 = self.quick_r_bounds
            r2 = self.image.r_bounds_make
            unit_label = 'pixels'
        if unit == 'au':
            r1 = self.quick_r_bounds * self.image.scale
            r2 = self.image.r_bounds_make * self.image.scale
            unit_label = 'au'
        elif unit == 'beam':
            r1 = self.quick_r_bounds / self.image.beam_fwhm
            r2 = self.image.r_bounds_make / self.image.beam_fwhm
            unit_label = 'beam FWHMs'
            
        plt.figure()
        if not average:
            pltstep(r1, self.quick_profile['left'], label='Left')
            pltstep(r1, self.quick_profile['right'], label='Right')
        else:
            pltstep(r1, meanlr(self.quick_profile), label='Fitted')
        if self.image.is_fake_image:
            pltstep(r2, self.image.weights_make, linestyle='--', label='True', color=colours[3], alpha=0.8)
        plt.legend(frameon=False)
        plt.title('Quick radial profile')
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel('Surface brightness (Jy/beam)')
        plt.tight_layout()
        plt.show()
    
    def get_rapid_annuli(self, r_outer, dr=0.1, height=10, inclination=90, points_per_pixel=200):
        '''Gets narrow annuli with the required properties. 
        Input:
            R_OUTER: outermost radial location with flux to fit to. 
            DR: width of each narrow annulus. 
            HEIGHT: height of each narrow annulus. 
            INCLINATION: inclination of each narrow annulus.'''
        print('\n----- Radial annuli -----')
        
        self.r_outer = r_outer
        self.dr = dr
        self.height = height
        self.inclination = inclination
        self.points_per_pixel = points_per_pixel
        
        # Get function that makes rings rapidly
        self.rapid_rings = get_narrow_annuli(r_outer, dr, height, inclination, self.image.xdim, points_per_pixel)
    
    def fit(self, nrings, n_iterations=100, extra_noise=0, random=True, verbose=True, fit_star=False, floor_to_0=True):
        '''Fits the radial surface brightness profile.
        Input
            NRINGS: number of annuli to use. 
            N_ITERATIONS: how many sets of fits to perform. Each set has a different set of annuli boundaries. 
            EXTRA_NOISE: how much noise to add to the best-fit model to repeat the fitting procedure on. If 0, the code will skip this step. 
            RANDOM: whether or not to randomly pick annuli boundaries if there are more sets of stored annuli boundaries than needed. 
            FIT_STAR: if True, then adds an annulus that occupies the first pixel to fit to the flux of the star. 
            FLOOR_TO_0: whether or not to set all fitted values below 0 to 0. '''
        
        ## Preparation
        print('\n----- Radial Fit -----')
        
        # Set up attributes
        self.nrings = nrings
        self.n_iterations = n_iterations
        self.extra_noise = extra_noise
        self.fit_star = fit_star
        
        # Set up variables
        dr = self.dr
        kernel = self.image.kernel
        image_linecut = self.image.linecut
        ndim = nrings + fit_star
        
        # Get annuli boundaries
        if not fit_star:
            self.R_BOUNDS = get_r_bounds(self.nrings, self.r_outer, n_iterations)
        else:
            "Include an additional ring with radius = 1 pixel"
            R_BOUNDS = get_r_bounds(self.nrings, [1, self.r_outer], n_iterations, use_flux_range=True)
            R_BOUNDS2 = np.zeros([len(R_BOUNDS), ndim+1])
            for i in range(len(R_BOUNDS)):
                R_BOUNDS2[i] = np.r_[0, R_BOUNDS[i]]
            self.R_BOUNDS = R_BOUNDS2
        
        # Set up storing variables
        self.R_BOUNDS = np.round(self.R_BOUNDS / dr) * dr
        self.MTXRATIOS = handlelr(np.zeros)([n_iterations, ndim])
        "fitted with extra noise"
        self.RATIOS_2 = handlelr(np.zeros)([n_iterations, ndim]) 
        
        # Pick subset of boundaries to use 
        if random:
            indices = np.random.choice(len(self.R_BOUNDS), n_iterations, replace=False)
            self.R_BOUNDS = self.R_BOUNDS[indices]
        else:
            self.R_BOUNDS = self.R_BOUNDS[0:n_iterations]
        
        # Time it
        t0 = time.time()
        
        ## Fit for each set of annuli boundaries
        for i in range(n_iterations):
            
            # Print status
            if verbose: print(i, end=' ')
            
            # Make annuli
            r_bounds = self.R_BOUNDS[i]
            rings = self.rapid_rings(r_bounds, kernel)
            abrl = bin_rings(rings, r_bounds, 'full')
            
            # Fit with matrix and iterative method
            L = bin_linecut(image_linecut, r_bounds)
            mtxratios, _ = matrix_fit(L, abrl)
        
            # Store values
            self.MTXRATIOS['left'][i], self.MTXRATIOS['right'][i] = mtxratios['left'], mtxratios['right']
            
        # Interpolate points
        self.rnew, INTERPOLATED = interpolate(self.MTXRATIOS, self.R_BOUNDS)
        
        ## Make noiseless model
        # *** Store profile ***
        self.profile = handlelr(np.median)(INTERPOLATED, axis=0)
        
        # Generate model
        if extra_noise:
            self.image.make_model(inclination=90, heights=0)
            noiseless_model = self.image.model.image
            
        ## Fit again with noise
        for i in range(n_iterations):
            
            # Don't fit with extra noise if no extra noise specified
            if not extra_noise: continue
            
            # Print status
            if verbose: print('n', i, end=' ')
            
            # Make annuli
            r_bounds = self.R_BOUNDS[i]
            rings = self.rapid_rings(r_bounds, kernel)
            abrl = bin_rings(rings, r_bounds, 'full')
        
            # Add noise to model
            noise = np.random.normal(0, extra_noise, self.image.image.shape)
            if self.image.add_blur:
                noise = blur(noise, self.image.add_blur)
            noisy_model = noiseless_model + noise
            noisy_model = cut(noisy_model, self.image.y_max)
            if self.image.flipped:
                noisy_model = (noisy_model + rotate180(noisy_model)) / 2
            "Make linecut"
            noisy_linecut = make_linecut(noisy_model, 'full')
            
            # Fit with matrix method
            L2 = bin_linecut(noisy_linecut, r_bounds)
            mtxratios2, _ = matrix_fit(L2, abrl)
        
            # Store values
            self.RATIOS_2['left'][i], self.RATIOS_2['right'][i] = mtxratios2['left'], mtxratios2['right']
        
        if not extra_noise:
            self.RATIOS_2 = self.MTXRATIOS
            
        # Interpolate points        
        self.rnew, INTERPOLATED2 = interpolate(self.RATIOS_2, self.R_BOUNDS)
        
        ## Calculate percentiles
        quantile_range = 0.34
        
        # Uncertainty without noise [first shade of uncertainty]
        self.profile_up = handlelr(np.quantile)(INTERPOLATED, 0.5 + quantile_range, axis=0)
        self.profile_down = handlelr(np.quantile)(INTERPOLATED, 0.5 - quantile_range, axis=0)
        
        # Uncertainty with noise [not used]
        self.profile2 = handlelr(np.median)(INTERPOLATED2, axis=0)
        
        # Uncertainty without noise [second shade of uncertainty]
        self.profile_up2 = handlelr(np.quantile)(INTERPOLATED2, 0.5 + quantile_range, axis=0)
        self.profile_down2 = handlelr(np.quantile)(INTERPOLATED2, 0.5 - quantile_range, axis=0)
        
        # Set negative values to 0
        if floor_to_0:
            self.profile = floor(self.profile)
            self.profile_up = floor(self.profile_up)
            self.profile_down = floor(self.profile_down)
            
            self.profile = floor(self.profile)
            self.profile_up2 = floor(self.profile_up2)
            self.profile_down2 = floor(self.profile_down2)
        
        print('Time taken:', f'{(time.time() - t0):.0f}')
        
    def plot(self, smooth=0, average=True, factor=1, use_au=True, floor_to_0=True):
        '''Plots the fitted radial profile. 
        Input
            SMOOTH: how much to smooth the fitted curves by. Recommend trying 101 if you want to smooth. 
            AVERAGE: whether or not to average the fits to the left and right halves of the image. In most cases you probably have averaged the left and right halves of the image already, so the two fits should be the same. Recommend using TRUE.
            FACTOR: how much to vertical scale the fit by. Used to avoid numbers that are too small or big on the axis labels when they the absolute value of the values doesn't matter. 
            USE_AU: this keyword is identical to UNIT in other methods. Can be 'pixel', 'au' or 'beam'.
            FLOOR_TO_0: whether or not to set everything below 0 to 0 when plotting. '''
        
        if smooth:
            from scipy.signal import savgol_filter
            window_length = smooth
            polyorder = 3
            
            yup = handlelr(savgol_filter)(self.profile_up, window_length, polyorder)
            ymedian = handlelr(savgol_filter)(self.profile, window_length, polyorder)
            ydown = handlelr(savgol_filter)(self.profile_down, window_length, polyorder)
            
            yup2 = handlelr(savgol_filter)(self.profile_up2, window_length, polyorder)
            ymedian2 = handlelr(savgol_filter)(self.profile2, window_length, polyorder)
            ydown2 = handlelr(savgol_filter)(self.profile_down2, window_length, polyorder)
            
        else:
            yup = self.profile_up
            ymedian = self.profile
            ydown = self.profile_down
            
            yup2 = self.profile_up2
            ymedian2 = self.profile2
            ydown2 = self.profile_down2
        
        yup = divlr(yup, 1/factor)
        ymedian = divlr(ymedian, 1/factor)
        ydown = divlr(ydown, 1/factor)
        
        yup2 = divlr(yup2, 1/factor)
        ymedian2 = divlr(ymedian2, 1/factor)
        ydown2 = divlr(ydown2, 1/factor)
        
        if not use_au:
            r = self.rnew
            if self.image.is_fake_image:
                r2 = self.image.r_bounds_make
            unit_label = 'pixels'
        elif use_au == True:
            r = self.rnew * self.image.scale
            if self.image.is_fake_image:
                r2 = self.image.r_bounds_make * self.image.scale
            unit_label = 'au'
        elif use_au == 'beam':
            assert self.image.beam_fwhm != None
            r = self.rnew / self.image.beam_fwhm
            if self.image.is_fake_image:
                r2 = self.image.r_bounds_make / self.image.beam_fwhm
            unit_label = 'beam FWHMs'
        
        # floor to 0
        if floor_to_0:
            ymedian = floor(ymedian)
            yup = floor(yup)
            ydown = floor(ydown)
            
            ymedian2 = floor(ymedian2)
            yup2 = floor(yup2)
            ydown2 = floor(ydown2)
        
        plt.figure()
        if not average:
            plt.plot(r, ymedian['left'], alpha=0.8, linewidth=2, label='Left')
            plt.plot(r, ymedian['right'], alpha=0.8, linewidth=2, label='Right')
            handlelr(plt.fill_between)(r, yup, ydown, alpha=0.5, linewidth=2)
        else:
            plt.plot(r, meanlr(ymedian), alpha=0.8, linewidth=2, label='Fitted from obs.')
            plt.fill_between(r, meanlr(yup), meanlr(ydown), alpha=0.5, linewidth=0)
            if self.extra_noise:
                plt.plot(r, meanlr(ymedian2), alpha=0.8, linewidth=2, label='Fitted from model')
                plt.fill_between(r, meanlr(yup2), meanlr(ydown2), color='C1', alpha=0.2, linewidth=0)
        #plt.ylim([None, np.max(meanlr(yup)[100:])*1.2])
        
        if self.image.is_fake_image:
            pltstep(r2, self.image.weights_make, color=colours[3], alpha=0.8, linestyle='--', label='True')
        
        if not average or self.image.is_fake_image:
            plt.legend(frameon=False)
        plt.title(f'{self.nrings} annuli, {self.image.noise_per_pixel/self.image.image.max()*100:.0f}% noise ({self.n_iterations} iterations)')
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel(r'Surface brightness ($\mu$Jy/beam)')
        plt.tight_layout()
        plt.show()
    
        
class HeightProfile():
    '''Height Profile Fitter class. '''
    
    def __init__(self, image):
        '''Links the Image object with the Height Profile Fitter obeject'''
        image.height = self
        self.image = image
    
    def get_rapid_annuli(self, r_outer, dr=0.1, hrange=[10, 30], dh=3, inclination=90):
        '''Gets narrow annuli with the required properties. 
        Input:
            R_OUTER: outermost radial location with flux to fit to. 
            DR: width of each narrow annulus. 
            HRANGE: minimum and maximum height of annuli to load.
            DH: spacing in heights between each set of narrow annuli to load. 
            INCLINATION: inclination of each narrow annulus.'''
        print('\n----- Height annuli -----')
        
        self.r_outer = r_outer
        self.dr = dr
        self.points_per_pixel = self.image.radial.points_per_pixel
        
        self.interp_heights = np.arange(hrange[0], hrange[-1] + dh, dh)
        self.dh = dh
        self.inclination = inclination
        
        # Storing parameter
        self.RAPID_RINGS = dict()
        
        for height in self.interp_heights:
            self.RAPID_RINGS[height] = get_narrow_annuli(self.r_outer, self.dr, height, inclination, self.image.xdim, self.points_per_pixel)
        
    def quick_fit(self, cut_height, starting_height):
        '''Does a quick fit to the image with only one iteration. Must have done a QUICK_FIT to the radial profile already. 
        Input
            CUT_HEIGHT: distance from the midplane defining the midplane region.
            STARTING_HEIGHT: intial height of the annuli to initiate the algorithm.  '''
        print('\n----- Height Quick Fit -----')
        
        r_bounds = self.image.radial.quick_r_bounds
        mtxratios = self.image.radial.quick_profile
        
        t0 = time.time()
        
        # Get calibrators and generators
        height_calibrators, part_generators = make_generators(r_bounds, self.interp_heights, self.image.xdim, self.RAPID_RINGS, self.image.kernel, cut_height)
        
        # Fit height
        self.image.part = make_linecut(self.image.image, cut_height)
        self.lim = bin_linecut(self.image.part, r_bounds)
        H, lmod = fit_height(self.lim, mtxratios, r_bounds, height_calibrators, part_generators, starting_height, self.image.xdim)
            
        self.quick_profile = createlr(H['left'][-1, :], H['right'][-1, :])
        self.quick_lmod = createlr(lmod['left'][-1, :], lmod['right'][-1, :])
        self.H = H
        self.lmod = lmod
        self.quick_r_bounds = r_bounds
        self.height_calibrators = height_calibrators
        self.part_generators = part_generators
        print('Time taken:', f'{(time.time() - t0):.0f}')
    
    def quick_plot(self, unit='pixel'):
        '''Makes a quick plot of the QUICK_FIT. 
        UNIT can be 'pixel', 'au' or 'beam'. '''
        
        if unit == 'pixel':
            r1 = self.quick_r_bounds
            r2 = self.image.r_bounds_make
            unit_label = 'pixels'
        if unit == 'au':
            r1 = self.quick_r_bounds * self.image.scale
            r2 = self.image.r_bounds_make * self.image.scale
            unit_label = 'au'
        elif unit == 'beam':
            r1 = self.quick_r_bounds / self.image.beam_fwhm
            r2 = self.image.r_bounds_make / self.image.beam_fwhm
            unit_label = 'beam FWHMs'
        
        plt.figure()
        pltstep(r1, self.quick_profile['left'], label='Left')
        pltstep(r1, self.quick_profile['right'], label='Right')
        plt.legend(frameon=False)
        if self.image.is_fake_image:
            pltstep(r2, self.image.heights_make, linestyle='--', label='True')
        plt.legend(frameon=False)
        plt.title('Quick height profile')
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel(f'Height ({unit_label})')
        plt.tight_layout()
        
        
        plt.figure()
        pltstep(r1, self.quick_lmod['left'], label='Left')
        pltstep(r1, self.quick_lmod['right'], label='Right')
        pltstep(r1, self.lim['right'], label='Observed')
        plt.legend(frameon=False)
        plt.title('Quick model partial flux')
        plt.xlabel(f'Radial distance ({unit_label})')
        plt.ylabel('Partial flux')
        plt.tight_layout()
        
    
    def fit(self, nrings, cut_height, starting_height, n_iterations=100, extra_noise=0, flux_range=None, random=True, verbose=True, remove_default=True):
        '''Fits the radial height profile.
        Input
            NRINGS: number of annuli to use. 
            CUT_HEIGHT: distance from the midplane defining the midplane region.
            STARTING_HEIGHT: intial height of the annuli to initiate the algorithm.  
            N_ITERATIONS: how many sets of fits to perform. Each set has a different set of annuli boundaries. 
            EXTRA_NOISE: how much noise to add to the best-fit model to repeat the fitting procedure on. If 0, the code will skip this step. 
            FLUX_RANGE: the range in radial distance from the star that actually has substantial flux. Other regions can't be fitted to since there is not enough flux to generate a robust fit. The heights of those regions is assumed to be STARTING_HEIGHT. 
            RANDOM: whether or not to randomly pick annuli boundaries if there are more sets of stored annuli boundaries than needed. 
            REMOVE_DEFAULT: whether or not to remove the regions without substantial flux by creating a model of those regions. Recommending using TRUE.'''
        
        ## Preparation
        print('\n----- Height Fit -----')
        
        # Set up attributes
        self.cut_height = cut_height
        self.starting_height = starting_height
        self.n_iterations = n_iterations
        self.extra_noise = extra_noise
        self.nrings = nrings
        
        "CUT_HEIGHT is used to to define the partial linecut"
        self.image.part = make_linecut(self.image.image, cut_height)
        
        "Height fitting is only done to radial locations where the fitted surface brightness is within FLUX_RANGE"
        if flux_range == None:
            self.flux_range = [0, self.r_outer]
        else:
            self.flux_range = flux_range
        assert n_iterations <= self.image.radial.n_iterations
        
        # Time it
        t0 = time.time()
        
        ## Default model
        "The 'default model' is used to remove the midplane flux contribution from regions where the surface brightness is low"
        "Fitting directly to their height will influence regions where there actually is flux"
        
        if remove_default and (flux_range[0] != 0 or flux_range[1] != self.r_outer):
            
            print('Removing default model from midplane flux')
            
            # Radial bins for making default model
            rb = np.arange(self.r_outer+1)
            r = (rb[1:] + rb[:-1]) / 2
            
            # Get surface brightness
            rnew = self.image.radial.rnew
            delta = rnew[1] - rnew[0]
            radial_profile = meanlr(self.image.radial.profile)
            radial_profile[radial_profile < 0] = 0
            w = bin_linecut(radial_profile, rb/delta)
            
            # Remove contributions from regions about to be fitted to
            w[flux_range[0]:flux_range[1]] = 0
            "Using starting height as default model height"
            h = starting_height
            default_model = MakeImage(rb, w, h, inclination_make=self.inclination, dim=self.image.dim, n_points_per_pixel=self.points_per_pixel, kernel=self.image.kernel, scale=1, rapid=False, add_before_convolve=True)
            self.default_model = default_model
            
            # Remove midplane flux contribution from default model
            default_part = make_linecut(default_model.image, cut_height)
            self.image.part['left'] -= default_part['left']
            self.image.part['right'] -= default_part['right']
        
        ## More preparation
        # Get annuli boundaries
        self.R_BOUNDS = get_r_bounds(self.nrings, self.flux_range, n_iterations, use_flux_range=True)
        if random:
            indices = np.random.choice(len(self.R_BOUNDS), n_iterations, replace=False)
            self.R_BOUNDS = self.R_BOUNDS[indices]
        else:
            self.R_BOUNDS = self.R_BOUNDS[0:n_iterations]
        
        # Storing variables
        self.HEIGHT = handlelr(np.zeros)([n_iterations, nrings])
        self.HEIGHT_2 = handlelr(np.zeros)([n_iterations, nrings])
        
        ## Fit height for each set of annuli boundaries
        for i in range(n_iterations):
            
            # Print status
            if verbose >= 1: print(i, end=' ')
            
            # Get annuli boundaries
            r_bounds = self.R_BOUNDS[i]
            
            # Get HEIGHT_CALIBRATORS and PART_GENERATORS for all annuli
            "HEIGHT_CALIBRATORS: mapping between h and midplane flux"
            "PART_GENERATORS: generates midplane flux given height"
            height_calibrators, part_generators, intp_range = make_generators(r_bounds, self.interp_heights, self.image.xdim, self.RAPID_RINGS, self.image.kernel, cut_height, testing=True)
            
            "Store these functions for testing"
            self.height_calibrators = height_calibrators
            self.part_generators = part_generators
            self.intp_range = intp_range
            
            # Fit height
            lim = bin_linecut(self.image.part, r_bounds)
            H, lmod = fit_height2(lim, self.image.radial.profile, self.image.radial.rnew, r_bounds, height_calibrators, part_generators, starting_height, self.image.xdim, intp_range)
            
            # Store values
            self.HEIGHT['left'][i, :], self.HEIGHT['right'][i, :] = H['left'][-1, :], H['right'][-1, :]
        
        # Interpolate height
        self.rnew, HEIGHT_INT = interpolate(self.HEIGHT, self.R_BOUNDS)
        
        ## Make noiseless model
        # *** Store profile ***
        self.profile = handlelr(np.median)(HEIGHT_INT, axis=0)
        
        # Generate model
        if extra_noise:
            self.image.make_model(inclination=self.inclination, heights=None, h_over_r=False, n_points_per_pixel=200, rapid=True, use_kernel=True, default_height=starting_height)

            noiseless_model = self.image.model.image
        
        ## Fit again with noise
        for i in range(n_iterations):
            
            # Don't fit with extra noise if no extra noise specified
            if not extra_noise: continue
            
            # Print status
            if verbose: print('n', i, end=' ')
            
            # Get annuli boundaries
            r_bounds = self.R_BOUNDS[i]
            
            # Get HEIGHT_CALIBRATORS and PART_GENERATORS for all annuli
            "HEIGHT_CALIBRATORS: mapping between h and midplane flux"
            "PART_GENERATORS: generates midplane flux given height"
            height_calibrators, part_generators, intp_range = make_generators(r_bounds, self.interp_heights, self.image.xdim, self.RAPID_RINGS, self.image.kernel, cut_height, testing=True)
            
            "Store these functions for testing"
            self.height_calibrators = height_calibrators
            self.part_generators = part_generators
            self.intp_range = intp_range
            
            # Add noise to model
            noise = np.random.normal(0, extra_noise, self.image.image.shape)
            if self.image.add_blur:
                noise = blur(noise, self.image.add_blur)
            noisy_model = noiseless_model + noise
            noisy_model = cut(noisy_model, self.image.y_max)
            if self.image.flipped:
                noisy_model = (noisy_model + rotate180(noisy_model)) / 2
            
            "Make linecut"
            #plot(noisy_model)
            part2 = make_linecut(noisy_model, cut_height)
            if remove_default and (flux_range[0] != 0 or flux_range[1] != self.r_outer):
                part2['left'] -= default_part['left']
                part2['right'] -= default_part['right']
            lim2 = bin_linecut(part2, r_bounds)
            
            # Fit height
            H2, lmod2 = fit_height2(lim2, self.image.radial.profile, self.image.radial.rnew, r_bounds, height_calibrators, part_generators, starting_height, self.image.xdim, intp_range)
            
            # Store values
            self.HEIGHT_2['left'][i, :], self.HEIGHT_2['right'][i, :] = H2['left'][-1, :], H2['right'][-1, :]
        
        # Interpolate height
        self.rnew, HEIGHT_2_INT = interpolate(self.HEIGHT_2, self.R_BOUNDS)
        
        if not extra_noise:
            HEIGHT_2_INT = HEIGHT_INT
        
        ## Calculate percentiles
        quantile_range = 0.34
        self.profile_up = handlelr(np.quantile)(HEIGHT_INT, 0.5 + quantile_range, axis=0)
        self.profile_down = handlelr(np.quantile)(HEIGHT_INT, 0.5 - quantile_range, axis=0)
        
        self.profile2 = handlelr(np.median)(HEIGHT_2_INT, axis=0)
        self.profile_up2 = handlelr(np.quantile)(HEIGHT_2_INT, 0.5 + quantile_range, axis=0)
        self.profile_down2 = handlelr(np.quantile)(HEIGHT_2_INT, 0.5 - quantile_range, axis=0)
         
        print(f'Time taken: {(time.time() - t0):.0f}')

    def plot(self, smooth=0, F_lim=0, average=True, use_au=False, h_over_r=False):
        '''Plots the fitted radial profile. 
        Input
            SMOOTH: how much to smooth the fitted curves by. Recommend trying 101 if you want to smooth. 
            F_LIM: only plots the heights for regions with flux above F_LIM. Recommending setting to 0 if using REMOVE_DEFAULT when fitting. 
            AVERAGE: whether or not to average the fits to the left and right halves of the image. In most cases you probably have averaged the left and right halves of the image already, so the two fits should be the same. Recommend using TRUE.
            USE_AU: this keyword is identical to UNIT in other methods. Can be 'pixel', 'au' or 'beam'.
            FLOOR_TO_0: whether or not to set everything below 0 to 0 when plotting. 
            H_OVER_R: if TRUE, plots y axis as h/r instead of h. '''
        
        # Smoothing
        if smooth:
            from scipy.signal import savgol_filter
            window_length = smooth
            polyorder = 3
            
            yup = handlelr(savgol_filter)(self.profile_up, window_length, polyorder)
            ymedian = handlelr(savgol_filter)(self.profile, window_length, polyorder)
            ydown = handlelr(savgol_filter)(self.profile_down, window_length, polyorder)
            
            yup2 = handlelr(savgol_filter)(self.profile_up2, window_length, polyorder)
            ymedian2 = handlelr(savgol_filter)(self.profile2, window_length, polyorder)
            ydown2 = handlelr(savgol_filter)(self.profile_down2, window_length, polyorder)
            
        else:
            yup = self.profile_up
            ymedian = self.profile
            ydown = self.profile_down
            
            yup2 = self.profile_up2
            ymedian2 = self.profile2
            ydown2 = self.profile_down2
        
        # Units
        if not use_au:
            r1 = self.rnew
            if self.image.is_fake_image:
                r2 = self.image.r_bounds_make
                heights_make = self.image.heights_make
            unit_label = 'pixels'
        elif use_au == True:
            r1 = self.rnew * self.image.scale
            if self.image.is_fake_image:
                r2 = self.image.r_bounds_make * self.image.scale
            if not h_over_r:
                yup = divlr(yup, 1/self.image.scale)
                ymedian = divlr(ymedian, 1/self.image.scale)
                ydown = divlr(ydown, 1/self.image.scale)
                yup2 = divlr(yup2, 1/self.image.scale)
                ymedian2 = divlr(ymedian2, 1/self.image.scale)
                ydown2 = divlr(ydown2, 1/self.image.scale)
                if self.image.is_fake_image:
                    heights_make = self.image.heights_make * self.image.scale
            unit_label = 'au'
        elif use_au == 'beam':
            assert self.image.beam_fwhm != None
            r1 = self.rnew / self.image.beam_fwhm
            if self.image.is_fake_image:
                r2 = self.image.r_bounds_make / self.image.beam_fwhm
            if not h_over_r:
                yup = divlr(yup, self.image.beam_fwhm)
                ymedian = divlr(ymedian, self.image.beam_fwhm)
                ydown = divlr(ydown, self.image.beam_fwhm)
                yup2 = divlr(yup2, self.image.beam_fwhm)
                ymedian2 = divlr(ymedian2, self.image.beam_fwhm)
                ydown2 = divlr(ydown2, self.image.beam_fwhm)
                if self.image.is_fake_image:
                    heights_make = self.image.heights_make / self.image.beam_fwhm
            unit_label = 'beam FWHMs'
        
        # Determine if plotting h or h/r
        if h_over_r:
            yup = divlr(yup, r1)
            ymedian = divlr(ymedian, r1)
            ydown = divlr(ydown, r1)
            if self.image.is_fake_image:
                r = (self.image.r_bounds_make[:-1] + self.image.r_bounds_make[1:])/2
                heights_make = self.image.heights_make / r
        
        # Don't plot regions with low flux
        for y in [yup, ymedian, ydown]:
            y['left'][self.image.radial.profile['left'] < F_lim * max(self.image.radial.profile['left'])] = np.nan
            y['right'][self.image.radial.profile['right'] < F_lim * max(self.image.radial.profile['right'])] = np.nan
        
        # floor to 0
        if 1:
            ymedian = floor(ymedian)
            yup = floor(yup)
            ydown = floor(ydown)
            
            ymedian2 = floor(ymedian2)
            yup2 = floor(yup2)
            ydown2 = floor(ydown2)
        
        # Plot fitted profiles
        plt.figure()
        if average:
            plt.plot(r1, meanlr(ymedian), color='C0', alpha=0.8, linewidth=2, label='Fitted from obs.')
            plt.fill_between(r1, meanlr(yup), meanlr(ydown), color='C0', alpha=0.5, linewidth=0)
            
            if self.extra_noise:
                plt.plot(r1, meanlr(ymedian2), color='C1', alpha=0.8, linewidth=2, label='Fitted from model')
                plt.fill_between(r1, meanlr(yup2), meanlr(ydown2), color='C1', alpha=0.3, linewidth=0)
        
        else:
            plt.plot(r1, ymedian['left'], alpha=0.8, linewidth=2, label='Left')
            plt.plot(r1, ymedian['right'], alpha=0.8, linewidth=2, label='Right')
            handlelr(plt.fill_between)(r1, yup, ydown, alpha=0.5)
        
        # Plot true profiles
        if self.image.is_fake_image:
            pltstep(r2, heights_make, color=colours[2], linestyle='--', label='True')
            plt.ylim([0, heights_make.max()*1.2])
        
        # Legend
        plt.legend(frameon=False)
        plt.title(f'{self.nrings} annuli, {self.image.noise_per_pixel/self.image.image.max()*100:.0f}% noise ({self.n_iterations} iterations)\n {self.inclination} inclination, {self.cut_height} cut height, {self.starting_height} starting height, {self.dh} dh')
        plt.xlabel(f'Radial distance ({unit_label})')
        if h_over_r:
            plt.ylabel('H/r')
        else:
            plt.ylabel(f'Height ({unit_label})')
        plt.tight_layout()
        plt.show()

