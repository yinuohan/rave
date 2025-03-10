r = np.arange(150) + 0.5
rp = 0.5 * np.exp( - (r - 30)**2 / (2 * 3**2) ) + np.exp( - (r - 50)**2 / (2 * 15**2) )

xdim = 200
ydim = 200

# Version 1: create image straight from inclined coordinates
t1 = time.time()

rr1 = create_r_grid(ydim, xdim, inclination=60)

im1 = np.zeros([ydim, xdim])

for i in range(ydim):
    for j in range(xdim):
        adjust_factor = np.cos(60/180*np.pi)
        im1[i, j] = np.interp(rr1[i, j], r, rp) * adjust_factor

print(time.time() - t1)

plot(im1)

# Version 2: create face-on image then squish to right inclination
t0 = time.time()

rr2 = create_r_grid(ydim, xdim, inclination=0)

im2 = np.zeros([ydim, xdim])

for i in range(ydim):
    for j in range(xdim):
        im2[i, j] = np.interp(rr2[i, j], r, rp)

adjust_factor = np.cos(60/180*np.pi)
im2 = zoom(im2, [np.cos(60/180*np.pi), 1]) * adjust_factor
im2 = reshape_image(im2, ydim, xdim)

print(time.time() - t0)

plot(im2)

# rave version h = 0
t0 = time.time()

imageh0 = MakeImage(r_bounds_make=centre_to_bin(r[:100]), weights_make=rp[:100], heights_make=0, inclination_make=60, dim=ydim, n_points_per_pixel=200, kernel=None, h_over_r=True, rapid=False)

print(time.time() - t0)

plot(imageh0.image)

# rave version h = 0.01
t0 = time.time()

imageh1 = MakeImage(r_bounds_make=centre_to_bin(r[:100]), weights_make=rp[:100], heights_make=0.01, inclination_make=60, dim=ydim, n_points_per_pixel=200, kernel=None, h_over_r=True, rapid=False)

print(time.time() - t0)

plot(imageh1.image)

# Compare
plot((im2 - im1) / im1)

plot((im1 - imageh0.image) / imageh0.image)

plot((imageh1.image - imageh0.image) / imageh1.image)