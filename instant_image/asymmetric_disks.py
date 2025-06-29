import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import binned_statistic_2d, norm
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates


## Function to simulate disk images
def make_disk(fr, fz, fphi, dim, inc=0, pa=90, h=0.05):
    "Define midplane vector"
    midplane_unit_vector = np.array([0, 0, 1])
    major_axis_unit_vector = np.array([0, 1, 0])
    minor_unit_vector = np.array([1, 0, 0])

    "x is minus east, y is north, z is minus line of sight (towards observer)"
    midplane_unit_vector = R.from_euler('y', inc, degrees=True).apply(midplane_unit_vector)
    midplane_unit_vector = R.from_euler('z', pa, degrees=True).apply(midplane_unit_vector)

    major_axis_unit_vector = R.from_euler('y', inc, degrees=True).apply(major_axis_unit_vector)
    major_axis_unit_vector = R.from_euler('z', pa, degrees=True).apply(major_axis_unit_vector)

    minor_unit_vector = R.from_euler('y', inc, degrees=True).apply(minor_unit_vector)
    minor_unit_vector = R.from_euler('z', pa, degrees=True).apply(minor_unit_vector)

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

    "Calculate radius and angle in midplane"
    r_disk = (rc**2 - z_disk**2) ** 0.5
    del rc

    "Calculate x and y coordinates in disk frame by projecting onto x and y unit vectors"
    y_disk = coords @ minor_unit_vector
    x_disk = coords @ major_axis_unit_vector
    del coords

    "Calculate azimuth in midplane from x and y coordinates in disk frame"
    phi_disk = np.arctan2(-x_disk, y_disk)
    del x_disk, y_disk

    "Create lattice in polar coordinates from which to interpolate (directly calculating values causes numerical nomalisation issues when scale height is small and the vertical distribution narrowly centred at a fraction of a pixel"
    r_disk_polar = np.arange(0, np.ceil(r_disk.max()+1), 1)
    z_disk_polar = np.arange(-np.ceil(z_disk.max()+1), np.ceil(z_disk.max()+2), 1)
    phi_disk_polar = np.linspace(-np.pi, np.pi, 100)

    #r3_polar, z3_polar, phi3_polar = np.meshgrid(r_disk_polar, z_disk_polar, phi_disk_polar, indexing='xy')

    # values_r3_polar = fr(r3_polar_simple)
    # values_z3_polar = fz(r3_polar, z3_polar)
    # values_phi3_polar = fphi(phi3_polar_simple)

    # "Normalise columns such that the vertical distribution sums to 0.5 and fix z3_polar where H = 0"
    # for iphi in range(len(phi_disk_polar)):
    #     for ir in range(len(r_disk_polar)):
    #         if np.any(np.isnan(values_z3_polar[:, ir, iphi])):
    #             values_z3_polar[:, ir, iphi] = 0
    #             values_z3_polar[len(z_disk_polar)//2, ir, iphi] = 1
    #         values_z3_polar[:, ir, iphi] = values_z3_polar[:, ir, iphi] / values_z3_polar[:, ir, iphi].sum() * 1

    "In the r and phi directions, only one dimension determines the values, so the rest is just copying and pasting, rather than having re-calculate each copy"

    values_r3_polar_r = fr(r_disk_polar)
    values_r3_polar = np.broadcast_to(values_r3_polar_r[np.newaxis, ..., np.newaxis], (len(z_disk_polar), len(r_disk_polar), len(phi_disk_polar)))

    values_phi3_polar_phi = fphi(phi_disk_polar)
    values_phi3_polar = np.broadcast_to(values_phi3_polar_phi[np.newaxis, np.newaxis, ...], (len(z_disk_polar), len(r_disk_polar), len(phi_disk_polar)))

    "In the z direction, just make one array and copy"
    values_z3_polar_rz = fz(*np.meshgrid(r_disk_polar, z_disk_polar))

    "Normalise columns such that the vertical distribution sums to 0.5 and fix z3_polar where H = 0"
    for ir in range(len(r_disk_polar)):
        if np.any(np.isnan(values_z3_polar_rz[:, ir])):
            values_z3_polar_rz[:, ir] = 0
            values_z3_polar_rz[len(z_disk_polar)//2, ir] = 1
        values_z3_polar_rz[:, ir] = values_z3_polar_rz[:, ir] / values_z3_polar_rz[:, ir].sum() * 1

    "Copy 2D z array to 3D"
    values_z3_polar = np.broadcast_to(values_z3_polar_rz[..., np.newaxis], (len(z_disk_polar), len(r_disk_polar), len(phi_disk_polar)))

    "Combine the r and z dimensions"
    values_polar = values_r3_polar * values_z3_polar * values_phi3_polar

    "Interpolate from polar lattice to Cartesian lattice"
    "If order = 1, alisaing can occur at certain inclinations, e.g., inc = 30 deg. Order = 2 improves this and order = 3 looks very smooth."
    values = map_coordinates(values_polar, [np.abs(z_disk + len(z_disk_polar)//2), r_disk, (phi_disk + np.pi) / (2 * np.pi) * 99], order=3, cval=0).reshape(dim, dim, dim)

    "Get image"
    im = values.sum(2)
    return im


## Generate disk image
"Define radial profile (Jy as a function of radius)"
def fr(r_disk):
    mu_r = 20
    sigma_r = 10
    return np.exp( - (r_disk - mu_r)**2 / (2 * sigma_r**2) )

"Define vertical profile (PDF as a function of height in disk frame, with integral normalised to 1"
def fz(r_disk, z_disk):
    sigma_h = h * r_disk
    sigma_h[sigma_h == 0] = np.nan

    fz = np.exp( - (z_disk - 0)**2 / (2 * sigma_h**2) )
    norm = 1 / (sigma_h * np.sqrt(2 * np.pi))
    return norm * fz

"Define azimuthal profile (some dimensionless modulatory function of azimuth in disk frame)"
"This is just a test"
def fphi_gaussian(phi_disk):
    mu_phi = 0 / 180 * np.pi
    sigma_phi = 90 / 180 * np.pi

    fphi = np.exp( - (phi_disk - mu_phi)**2 / (2 * sigma_phi**2) )
    return fphi

"Now use a Henyey-Greenstein scattering phase function"
def azimuth_to_scattering_angle(phi, inclination):
    scattering_angle = np.arccos( np.cos(phi) * np.cos(np.pi/2 - inclination) )
    return scattering_angle

def hg_scattering_phase_function(scattering_angle, g):
    return (1 - g**2) / ( 4 * np.pi * (1 - 2 * g * np.cos(scattering_angle) + g**2 )**(3/2) )

def fphi(phi_disk):
    scattering_angle = azimuth_to_scattering_angle(phi_disk, -inc/180*np.pi)
    fphi = hg_scattering_phase_function(scattering_angle, g=0.2)
    return fphi

"Other parameters"
dim = 100
inc = 30
pa = 120
h = 0.01

"Generate image"
t0 = time.time()
im = make_disk(fr=fr, fz=fz, fphi=fphi, dim=dim, inc=inc, pa=pa, h=h)
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
