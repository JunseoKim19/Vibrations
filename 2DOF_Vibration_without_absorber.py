import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv, eig

# Constants and parameters
m = 7.6  # mass of the pipe (kg)
l1 = 0.45  # length of the left segment of the pipe (m)
l2 = 0.15  # length of the right segment of the pipe (m)
l3 = 0.45
c1 = 45  # damping coefficient (kg/s)
c2 = 45
k1 = 45000  # equivalent stiffness of support bearings (N/m)
k2 = 45000
F0 = 5  # amplitude of the lateral concentrated load (N)
r0 = 0.168
ri = 0.167
forcing_freq_range = np.linspace(650, 1500, 1000)  # forcing frequency range (rpm)
# Convert rpm to rad/s
forcing_freq_range_rad_s = forcing_freq_range * (2 * np.pi / 60)

# Time interval
t = np.linspace(0, 0.3, 1000)

# Create an empty array to store the resulting x(t) and theta(t)
xt_results = np.zeros((len(forcing_freq_range_rad_s), len(t)))
theta_results = np.zeros((len(forcing_freq_range_rad_s), len(t)))

# Mass matrix
M = np.array([[m, 0],
              [0, (r0**2+ri**2)/2*m]])

# Damping matrix
C = np.array([[c1+c2, l2*c2-l1*c1],
              [l2*c2-l1*c1, l2**2*c2+l1**2*c1]])

# Stiffness matrix
K = np.array([[k1+k2, k2*l2-k1-l1],
              [k2*l2-k1*l1, l1**2*k1+l2**2*k2]])

# Compute M^(-1/2)
M_inv_sqrt = inv(sqrtm(M))

# Compute C_tilda and K_tilda
C_tilda = M_inv_sqrt @ C @ M_inv_sqrt
K_tilda = M_inv_sqrt @ K @ M_inv_sqrt

# Compute eigenvalues and eigenvectors of K_tilda
eigenvalues, P = eig(K_tilda)

# Compute the transpose of P
P_T = np.transpose(P)

# Compute P^T * K_tilda * P
result1 = P_T @ K_tilda @ P

# Compute P^T * C_tilda * P
result2 = P_T @ C_tilda @ P

# Find two natural frequencies
w1, w2 = np.sqrt(eigenvalues)

# Find two damping ratios
z1 = result2[0, 0] / (2 * w1)
z2 = result2[1, 1] / (2 * w2)

# Find damped natural frequencies
wd1 = w1 * np.sqrt(1 - z1**2)
wd2 = w2 * np.sqrt(1 - z2**2)

# Rearrange if needed
if w1 > w2:
    w1, w2 = w2, w1
    z1, z2 = z2, z1
    wd1, wd2 = wd2, wd1

# Desired forcing frequencies in rpm
desired_frequencies = [650, 800,950, 1200, 1500]

# Initialize arrays to store maximum displacements and maximum angular displacements
x_max_values = np.zeros(len(forcing_freq_range_rad_s))
theta_max_values = np.zeros(len(forcing_freq_range_rad_s))

for idx, omega in enumerate(forcing_freq_range_rad_s):
    # Time-varying force function
    F_t = F0 * np.cos(omega * t)

    # Force vector
    F = np.array([[0],
                  [F_t]])

    # Find P^T * M^(-1/2) * F
    result3 = P_T @ M_inv_sqrt @ F

    # Calculate r1(t) and r2(t)
    r1_t = (result3[0, 0] / (m * wd1)) * np.exp(-z1 * w1 * t) * np.sin(wd1 * t)
    r2_t = (result3[1, 0] / (wd2)) * np.exp(-z2 * w2 * t) * np.sin(wd2 * t)

    # Create r(t) matrix
    r_t = np.vstack((r1_t, r2_t))

    # Calculate x(t) and theta(t)
    xt_theta_t = M_inv_sqrt @ P @ r_t

    # Store the results
    xt_results[idx, :] = xt_theta_t[0, :]
    theta_results[idx, :] = xt_theta_t[1, :]
    
    # Calculate maximum displacement and maximum angular displacement for the current forcing frequency
    x_max = np.max(np.abs(xt_results[idx, :]))
    theta_max = np.max(np.abs(theta_results[idx, :]))

    # Store the maximum displacements and maximum angular displacements
    x_max_values[idx] = x_max
    theta_max_values[idx] = theta_max

# Plot x(t) and theta(t) for each desired forcing frequency
for freq in desired_frequencies:
    # Find the index of the desired frequency
    idx = np.argmin(np.abs(forcing_freq_range - freq))

    plt.figure()
    plt.plot(t, xt_results[idx, :], label="x(t)")
    plt.plot(t, theta_results[idx, :], label="theta(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m) / Angle (rad)")
    plt.title(f"x(t) and theta(t) vs Time at {forcing_freq_range[idx]} rpm")
    plt.legend()
    plt.grid()
    plt.show()
    
# Calculate r values for both natural frequencies
r_w1 = forcing_freq_range_rad_s / w1
r_w2 = forcing_freq_range_rad_s / w2

z_values = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure()
for z in z_values:
    # Calculate TR_D and TR_F for w1
    TR_D_w1 = r_w1**2 / ((1 - r_w1**2)**2 + (2 * z * r_w1)**2)**(1/2)
    TR_F_w1 = r_w1**2 * TR_D_w1

    # Calculate TR_D and TR_F for w2
    TR_D_w2 = r_w2**2 / ((1 - r_w2**2)**2 + (2 * z * r_w2)**2)**(1/2)
    TR_F_w2 = r_w2**2 * TR_D_w2

    # Combine TR_D values for both natural frequencies
    TR_D_combined = np.concatenate((TR_D_w1, TR_D_w2))
    
    # Combine r arrays for both natural frequencies
    r_combined = np.concatenate((r_w1, r_w2))

    # Combine TR_F values for both natural frequencies
    TR_F_combined = np.concatenate((TR_F_w1, TR_F_w2))

    # Sort r_combined and the corresponding TR_D and TR_F values
    sorted_indices = np.argsort(r_combined)
    r_combined = r_combined[sorted_indices]
    TR_D_combined = TR_D_combined[sorted_indices]
    TR_F_combined = TR_F_combined[sorted_indices]

    # Plot combined TR_D vs r
    plt.plot(r_combined, TR_D_combined, label=f"TR_D (z={z})")
plt.xlabel("r")
plt.ylabel("TR_D")
plt.title("Combined TR_D vs r for different z values")
plt.legend()
plt.grid()
plt.xlim(10**-1, 10**1)  # Set the x-axis range
plt.xscale('log')  # Set the x-axis to log scale
plt.show()

plt.figure()
for z in z_values:
    # Calculate TR_D and TR_F for w1
    TR_D_w1 = r_w1**2 / ((1 - r_w1**2)**2 + (2 * z * r_w1)**2)**(1/2)
    TR_F_w1 = r_w1**2 * TR_D_w1

    # Calculate TR_D and TR_F for w2
    TR_D_w2 = r_w2**2 / ((1 - r_w2**2)**2 + (2 * z * r_w2)**2)**(1/2)
    TR_F_w2 = r_w2**2 * TR_D_w2

    # Combine TR_F values for both natural frequencies
    TR_F_combined = np.concatenate((TR_F_w1, TR_F_w2))

    # Sort r_combined and the corresponding TR_F values
    sorted_indices = np.argsort(r_combined)
    r_combined = r_combined[sorted_indices]
    TR_F_combined = TR_F_combined[sorted_indices]

    # Plot combined TR_F vs r
    plt.plot(r_combined, TR_F_combined, label=f"TR_F (z={z})")

plt.xlabel("r")
plt.ylabel("TR_F")
plt.title("Combined TR_F vs r for different z values")
plt.legend()
plt.grid()
plt.xlim(10**-1, 10**1)  # Set the x-axis range
plt.xscale('log')  # Set the x-axis to log scale
plt.show()

# Find the maximum values for x(t), xa(t), and theta(t)
max_xt = np.max(np.abs(xt_results))
max_theta = np.max(np.abs(theta_results))

# Find the indices of the maximum values
idx_max_xt = np.unravel_index(np.argmax(np.abs(xt_results)), xt_results.shape)
idx_max_theta = np.unravel_index(np.argmax(np.abs(theta_results)), theta_results.shape)

# Find the corresponding forcing frequencies for the maximum values
freq_max_xt = forcing_freq_range[idx_max_xt[0]]
freq_max_theta = forcing_freq_range[idx_max_theta[0]]

# Print the maximum values and the corresponding forcing frequencies
print(f"Maximum x(t) value: {max_xt:.6f} m at {freq_max_xt:.2f} rpm")
print(f"Maximum theta(t) value: {max_theta:.6f} rad at {freq_max_theta:.2f} rpm")


frequencies = np.linspace(0, 500, 5000)  # Define a range of frequencies between 0 to 300

plt.figure()
for z in z_values:
    TR_D_values_w1 = []
    TR_D_values_w2 = []
    for freq in frequencies:
        r_w1 = freq / w1
        r_w2 = freq / w2
        TR_D_w1 = r_w1**2 / ((1 - r_w1**2)**2 + (2 * z * r_w1)**2)**(1/2)
        TR_D_w2 = r_w2**2 / ((1 - r_w2**2)**2 + (2 * z * r_w2)**2)**(1/2)
        TR_D_values_w1.append(TR_D_w1)
        TR_D_values_w2.append(TR_D_w2)

    TR_D_combined = np.array(TR_D_values_w1) + np.array(TR_D_values_w2)
    plt.plot(frequencies, TR_D_combined, label=f"TR_D (z={z})")

plt.xlabel("Frequency (rad/s)")
plt.ylabel("TR_D")
plt.title("TR_D vs Frequency for different z values")
plt.legend()
plt.grid()
plt.xlim(0, 500)  # Set the x-axis range
plt.show()

print(f"Natural frequency w1: {w1:.2f} rad/s")
print(f"Natural frequency w2: {w2:.2f} rad/s")

frequencies = np.linspace(0, 300, 1000)

plt.figure()
for z in z_values:
    TR_F_values_w1 = []
    TR_F_values_w2 = []
    for freq in frequencies:
        r_w1 = freq / w1
        r_w2 = freq / w2
        TR_D_w1 = r_w1**2 / ((1 - r_w1**2)**2 + (2 * z * r_w1)**2)**(1/2)
        TR_D_w2 = r_w2**2 / ((1 - r_w2**2)**2 + (2 * z * r_w2)**2)**(1/2)
        TR_F_w1 = r_w1**2 * TR_D_w1
        TR_F_w2 = r_w2**2 * TR_D_w2
        TR_F_values_w1.append(TR_F_w1)
        TR_F_values_w2.append(TR_F_w2)

    TR_F_combined = np.array(TR_F_values_w1) + np.array(TR_F_values_w2)
    plt.plot(frequencies, TR_F_combined, label=f"TR_F (z={z})")

plt.xlabel("Frequency (rad/s)")
plt.ylabel("TR_F")
plt.title("TR_F vs Frequency for different z values")
plt.legend()
plt.grid()
plt.xlim(0, 300)
plt.show()