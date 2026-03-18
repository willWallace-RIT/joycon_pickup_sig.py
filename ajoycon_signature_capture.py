import time
import numpy as np
import os
from datetime import datetime
from ajoycon import discover_joycons
import asyncio
#from pyjoycon import JoyCon, get_L_id  # or get_R_id

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 60  # Hz
WINDOW_SECONDS = 3
SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

PICKUP_THRESHOLD = 1.5
SETTLE_THRESHOLD = 0.5

SAVE_DIR = "signatures"

# =========================
# SETUP
# =========================
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

#joycon_id = get_L_id()
#jc = JoyCon(*joycon_id)

print("Connected to Joy-Con")

# =========================
# IMU FUNCTIONS
# =========================
def read_imu(joycon):
    status=joycon.status
    imu= status.imu
    gyro = (imu.gyro.x,imu.gyro.y,imu.gyro.z)
    accel = (imu.accel.x,imu.accel.y,imu.accel.x)    # (x, y, z)
    return np.array(gyro), np.array(accel)

def detect_pickup(gyro):
    magnitude = np.linalg.norm(gyro)
    return magnitude > PICKUP_THRESHOLD

# =========================
# CAPTURE
# =========================
def wait_for_pickup(joycon):
    print("Waiting for pickup...")
    while True:
        gyro, _ = read_imu()
        if detect_pickup(gyro):
            print("Pickup detected!")
            return
        time.sleep(1 / SAMPLE_RATE)

def capture_window():
    gyro_buffer = []
    accel_buffer = []

    print("Recording window...")

    for _ in range(SAMPLES):
        gyro, accel = read_imu()
        gyro_buffer.append(gyro)
        accel_buffer.append(accel)
        time.sleep(1 / SAMPLE_RATE)

    return np.array(gyro_buffer), np.array(accel_buffer)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(gyro_data, accel_data):
    features = []

    # --- Gyro magnitude ---
    gyro_mag = np.linalg.norm(gyro_data, axis=1)

    features.append(np.mean(gyro_mag))
    features.append(np.std(gyro_mag))
    features.append(np.max(gyro_mag))
    features.append(np.min(gyro_mag))

    # --- FFT features ---
    fft = np.fft.fft(gyro_mag)
    power = np.abs(fft)

    features.append(np.mean(power))
    features.append(np.std(power))

    # --- Acceleration magnitude ---
    accel_mag = np.linalg.norm(accel_data, axis=1)

    features.append(np.mean(accel_mag))
    features.append(np.std(accel_mag))
    features.append(np.max(accel_mag))

    # --- Stabilization time ---
    below_thresh = np.where(gyro_mag < SETTLE_THRESHOLD)[0]
    settle_time = below_thresh[0] if len(below_thresh) > 0 else len(gyro_mag)

    features.append(settle_time)

    # --- Axis-specific features ---
    for axis in range(3):
        features.append(np.mean(gyro_data[:, axis]))
        features.append(np.std(gyro_data[:, axis]))

    return np.array(features)

# =========================
# SAVE
# =========================
def save_sample(user_label, features):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/{user_label}_{timestamp}.npy"
    np.save(filename, features)
    print(f"Saved: {filename}")

# =========================
# MAIN LOOP
# =========================
async def main():
    joycons = discover_joycons()
    if not joycons:
        print("No joy-cons found!")
        return
    async with joycons[0].connect() as joycon:


        user_label = input("Enter user label (e.g., userA): ").strip()

        while True:
           input("\nPress ENTER when ready to perform pickup...")

           wait_for_pickup()
           gyro_data, accel_data = capture_window()

           features = extract_features(gyro_data, accel_data)
           save_sample(user_label, features)

           print("Feature vector:", features)

           cont = input("Capture another? (y/n): ").strip().lower()
           if cont != 'y':
               break

if __name__ == "__main__":
    main()
