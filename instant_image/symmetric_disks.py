import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import binned_statistic_2d, norm
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates


## Function that generates disk images
def make_axisymmetric_disk(fr, fz, dim, inc=0, pa=90, h=0.05):
    "Define midplane vector"
    midplane_unit_vector = np.array([0, 0, 1])

    "x is minus east, y is north, z is minus line of sight (towards observer)"
    midplane_unit_vector = R.from_euler('y', inc, degrees=True).apply(midplane_unit_vector)
    midplane_unit_vector = R.from_euler('z', pa, degrees=True).apply(midplane_unit_vector)

    "Use centre as reference point"
    cx, cy, cz = dim/2-0.5, dim/2-0.5, dim/2-0.5

    "Create coordinates of lattice"
    x = np.arange(0, dim, 1) - cx
    y = np.arange(0, dim, 1) - cy
    z = np.arange(0, dim, 1) - cz

    xxx, yyy, zzz = np.meshgrid(x, y, z, indexing='xy')
    coords = np.c_[xxx.flatten(), yyy.flatten(), zzz.flatten()]
    del xxx, yyy, zzz

    "Calculate radius from centre"
    rc = np.linalg.norm(coords, axis=1)

    "Calculate height by projecting onto midplane unit vector"
    z_disk = coords @ midplane_unit_vector
    del coords

    "Calculate radius in midplane"
    r_disk = (rc**2 - z_disk**2) ** 0.5
    del rc

    "Create lattice in polar coordinates from which to interpolate (directly calculating values causes numerical nomalisation issues when scale height is small and the vertical distribution narrowly centred at a fraction of a pixel"
    r_disk_polar = np.arange(0, np.ceil(r_disk.max()+1), 1)
    z_disk_polar = np.arange(-np.ceil(z_disk.max()+1), np.ceil(z_disk.max()+2), 1)
    rr_polar, zz_polar = np.meshgrid(r_disk_polar, z_disk_polar, indexing='xy')

    values_rr_polar = fr(rr_polar)
    #"Stacking the same array doesn't speed it up in the 2D case as np is already fast"
    #values_r_polar = fr(r_disk_polar)
    #values_rr_polar = np.broadcast_to(values_r_polar, (len(z_disk_polar), *values_r_polar.shape))
    values_zz_polar = fz(rr_polar, zz_polar)

    "Normalise columns such that the vertical distribution sums to 0.5 and fix values_zz_polar where H = 0"
    for icolumn in range(values_zz_polar.shape[1]):
        if np.any(np.isnan(values_zz_polar[:,icolumn])):
            values_zz_polar[:, icolumn] = 0
            values_zz_polar[values_zz_polar.shape[0]//2, icolumn] = 1
        values_zz_polar[:,icolumn] = values_zz_polar[:,icolumn] / values_zz_polar[:,icolumn].sum() * 1

    "Combine the r and z dimensions"
    values_polar = values_rr_polar * values_zz_polar

    "Interpolate from polar lattice to Cartesian lattice"
    "If order = 1, alisaing can occur at certain inclinations, e.g., inc = 30 deg. Order = 2 improves this and order = 3 looks very smooth."
    values = map_coordinates(values_polar, [np.abs(z_disk + len(z_disk_polar)//2), r_disk], order=3, cval=0).reshape(dim, dim, dim)

    "Get image"
    im = values.sum(2)
    return im


## Generate image (everything is in units of pixels)
"Define radial profile"
def fr(r_disk):
    mu_r = 20
    sigma_r = 10
    return np.exp( - (r_disk - mu_r)**2 / (2 * sigma_r**2) )

"Define vertical profile"
def fz(r_disk, z_disk):
    sigma_h = h * r_disk
    sigma_h[sigma_h == 0] = np.nan

    fz = np.exp( - (z_disk - 0)**2 / (2 * sigma_h**2) )
    norm = 1 / (sigma_h * np.sqrt(2 * np.pi))
    return norm * fz

"Other parameters"
dim = 100
inc = 30
pa = 90
h = 0.0

"Generate image"
t0 = time.time()
im = make_axisymmetric_disk(fr=fr, fz=fz, dim=dim, inc=inc, pa=pa, h=h)
t1 = time.time()
print(t1 - t0)

"Display the image"
if 1:
    if 0:
        kernel = make_kernel2(31, 10, 10, 0)
        im = convolve(im, kernel)
    else:
        kernel = None
    plt.figure()
    plt.imshow(im, origin='lower', cmap='viridis', extent=[-dim/2, dim/2, -dim/2, dim/2])
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
