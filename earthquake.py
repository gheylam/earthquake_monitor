import numpy as np
import matplotlib.pyplot as plt

def value_cal(xs, ys, x_sensor_values, y_sensor_values):
    print(xs.shape())
    print(xs.shape())
    print(x_sensor_values.shape())
    print(y_sensor_values.shape())
    return 1

def single_explosion_signal_truth(station_x, station_y, truth_x, truth_y):
    distance_sqr = (station_x - truth_x)**2 + (station_y - truth_y)**2
    signal = 1 / (distance_sqr + 0.1)
    return signal

def single_explosion_signal_noisy(station_x, station_y, truth_x, truth_y, sd):
    truth = single_explosion_signal_truth(station_x, station_y, truth_x, truth_y)
    noise = np.random.normal(0, sd)
    return truth + noise

def calculate_location_posterior(pts_x, pts_y, noisy_station_signal, station_x, station_y, sd):
    pts_dim = pts_x.shape[0]
    # Start with uniform prior (as specified in question)
    prior_xy = np.ones(pts_dim)
    # Initialise posterior with zeros
    posterior_xy = np.zeros(pts_dim)

    # For every point we assign the posterior probability value
    for pt in range(pts_dim):
        x = pts_x[pt]
        y = pts_y[pt]

        # Calculate the posterior for this point
        num_stations = station_x.shape[0]

        for station in range(num_stations):
            # Get the expected signal
            expected_signal = single_explosion_signal_truth(station_x[station], station_y[station], x, y)
            # P(V = z | D = d)
            normal_coffe = 1 / np.sqrt(2 * np.pi * sd**2)
            exp_coffe = -1 / (2 * sd**2)
            sqr_diff = (noisy_station_signal[station] - expected_signal)**2
            prob_of_sig_z_given_d = normal_coffe * np.exp(exp_coffe * sqr_diff)
            # Bayesian Updating:
            # Posterior = Prior * Likelihood
            posterior_xy[pt] = prior_xy[pt] * prob_of_sig_z_given_d

    return posterior_xy


# plotting the basic graph
# Compute areas and colors
np.random.seed(19680801)

num_pts = 5000
rate = 25 # The angular rate of the spiral
theta = np.zeros(num_pts)
r = np.zeros(num_pts)
pt_colors = np.zeros(num_pts)
pt_x = np.zeros(num_pts)
pt_y = np.zeros(num_pts)
for pt in range(num_pts):
    theta[pt] = rate * 2 * np.pi * pt / num_pts
    r[pt] = pt / num_pts
    pt_x[pt] = r[pt] * np.cos(theta[pt])
    pt_y[pt] = r[pt] * np.sin(theta[pt])
pt_area = 1

# Plotting stations
num_stations = 10
station_theta = np.zeros(num_stations)
station_r = np.ones(num_stations) * 1.05
station_colors = np.ones(num_stations)
station_x = np.zeros(num_stations)
station_y = np.zeros(num_stations)
station_area = np.ones(num_stations) * 100

for station in range(num_stations):
    station_theta[station] = station * (2 * np.pi / num_stations)
    station_x[station] = station_r[station] * np.cos(station_theta[station])
    station_y[station] = station_r[station] * np.sin(station_theta[station])

# Populating sensor readings
true_sensor_reading = np.zeros(num_stations)
noisy_sensor_reading = np.zeros(num_stations)
ground_truth_x = 0.2
ground_truth_y = 0.2

sd = 0.2
for station in range(num_stations):
    true_sensor_reading[station] = single_explosion_signal_truth(station_x[station], station_y[station], ground_truth_x, ground_truth_y)
    noisy_sensor_reading[station] = single_explosion_signal_noisy(station_x[station], station_y[station], ground_truth_x, ground_truth_y, sd)


# Generate posterior
pt_colors = calculate_location_posterior(pt_x, pt_y, noisy_sensor_reading, station_x, station_y, sd)

# Generating station sensor plots (both true and noisy sensor value)
sensor_size = 1000
true_sensor_reading_thetas = []
noisy_sensor_reading_thetas = []
true_sensor_reading_rs = []
noisy_sensor_reading_rs = []
true_sensor_reading_scaled = true_sensor_reading / np.amax(true_sensor_reading)
noisy_sensor_reading_scaled = noisy_sensor_reading / np.amax(noisy_sensor_reading)

for reading in range(num_stations):
    temp_r_val = station_r[reading]
    # Building the true sensor reading points
    true_theta = station_theta[reading] + 0.1
    num_points = int(np.floor(sensor_size * true_sensor_reading_scaled[reading]))

    for pt in range(num_points):
        r_val = temp_r_val + (1 / sensor_size)
        temp_r_val = r_val
        true_sensor_reading_rs = np.append(true_sensor_reading_rs, r_val)
        true_sensor_reading_thetas = np.append(true_sensor_reading_thetas, true_theta)

    true_sensor_reading_rs = np.array(true_sensor_reading_rs)
    true_sensor_reading_thetas = np.array(true_sensor_reading_thetas)
    # Building the noisy sensor reading points
    noisy_theta = station_theta[reading] - 0.1
    num_points = int(np.floor(sensor_size * noisy_sensor_reading_scaled[reading]))
    temp_r_val = station_r[reading] # reset the r value back to original station r value
    for pt in range(num_points):
        r_val = temp_r_val + (1 / sensor_size)
        temp_r_val = r_val
        noisy_sensor_reading_rs = np.append(noisy_sensor_reading_rs, r_val)
        noisy_sensor_reading_thetas = np.append(noisy_sensor_reading_thetas, noisy_theta)

# Generate the x and y coordinates for the sensor readings
true_sensor_reading_dim = true_sensor_reading_rs.shape[0]
true_sensor_reading_x = np.zeros(true_sensor_reading_dim)
true_sensor_reading_y = np.zeros(true_sensor_reading_dim)

for reading in range(true_sensor_reading_dim):
    true_sensor_reading_x[reading] = true_sensor_reading_rs[reading] * np.cos(true_sensor_reading_thetas[reading])
    true_sensor_reading_y[reading] = true_sensor_reading_rs[reading] * np.sin(true_sensor_reading_thetas[reading])

noisy_sensor_reading_dim = noisy_sensor_reading_rs.shape[0]
noisy_sensor_reading_x = np.zeros(noisy_sensor_reading_dim)
noisy_sensor_reading_y = np.zeros(noisy_sensor_reading_dim)

for reading in range(noisy_sensor_reading_dim):
    noisy_sensor_reading_x[reading] = noisy_sensor_reading_rs[reading] * np.cos(noisy_sensor_reading_thetas[reading])
    noisy_sensor_reading_y[reading] = noisy_sensor_reading_rs[reading] * np.sin(noisy_sensor_reading_thetas[reading])


# ground truth plot
ground_truth_r = np.sqrt(ground_truth_x**2 + ground_truth_y**2)
ground_truth_theta = np.arctan(ground_truth_y/ground_truth_x)

true_sensor_dim = true_sensor_reading_thetas.shape
true_sensor_reading_color = np.ones(true_sensor_dim)

noisy_sensor_dim = noisy_sensor_reading_thetas.shape
noisy_sensor_reading_color = np.ones(noisy_sensor_dim)

sensor_reading_area = 0.2

fig = plt.figure()

# ax for points
ax = fig.add_subplot(111, projection='polar')
ax.grid(False)
ax.set_axis_off()
ax.set_frame_on(True)

ax.scatter(theta, r, c=pt_colors, s=pt_area, cmap='gray_r', alpha=0.5)
ax.scatter(station_theta, station_r, c=station_colors, s=station_area, cmap='hsv', alpha=0.5)
ax.scatter(true_sensor_reading_thetas, true_sensor_reading_rs, c=true_sensor_reading_color, s=sensor_reading_area, cmap="hsv")
ax.scatter(noisy_sensor_reading_thetas, noisy_sensor_reading_rs, c=noisy_sensor_reading_color, s=sensor_reading_area, cmap="viridis")

plt.plot(ground_truth_theta, ground_truth_r, 'r+', mew=2, ms=10)

# annotating the stations by their id
for station in range(num_stations):
    plt.annotate(str(station), xy=(station_theta[station], station_r[station]))

plt.show()

# Plotting the cartesian version of the graph
fig_cartesian = plt.figure()
ax2 = fig_cartesian.add_subplot(111)
ax2.scatter(pt_x, pt_y, s=0.5, c=pt_colors, cmap="gray_r")
ax2.scatter(station_x, station_y, s=100, c=station_colors, cmap='hsv')
ax2.scatter(true_sensor_reading_x, true_sensor_reading_y, c=true_sensor_reading_color, s=1, cmap="hsv")
ax2.scatter(noisy_sensor_reading_x, noisy_sensor_reading_y, c=noisy_sensor_reading_color, s=1, cmap="viridis")
plt.plot(ground_truth_x, ground_truth_y, 'r+', mew=2, ms=10)
station_circle = plt.Circle((0, 0), 1.0, color='black', fill=False)
ax2.add_artist(station_circle)

# Plotting the maximum likelihood
max_posterior = np.amax(pt_colors)
max_index = np.where(pt_colors == max_posterior)
max_x = pt_x[max_index]
max_y = pt_y[max_index]
plt.plot(max_x, max_y, 'b+', mew=2, ms=10)

plt.show()




