import serial
import csv
import threading
import os
from datetime import datetime

PORT = "COM5"
BAUD = 115200

SESSION_TYPES = {
    "1": "Baseline",
    "2": "Static",
    "3": "Dynamic",
    "4": "Fatigue",
    "5": "Recovery",
}

COMMANDS = {
    "1": "START",
    "2": "STOP",
    "3": "CALIBRATE_MVC",
    "4": "RESET_MVC",
    "5": "ZERO_ANGLE",
    "6": "RESET_ANGLE",
    "7": "STATUS",
    "8": "HEADER",
    "9": "LED_ON",
    "0": "LED_OFF",
}

def print_menu():
    print("\n--- Commands ---")
    for key, cmd in COMMANDS.items():
        print(f"  [{key}] {cmd}")
    print("  [q] Quit")
    print("----------------")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_collection")
os.makedirs(DATA_DIR, exist_ok=True)

# Filename is finalised (with session type) when START is pressed.
# The serial_reader opens the file lazily on the first data line,
# so the filename is always correct by the time writing begins.
filename = os.path.join(DATA_DIR, f"imu_emg_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Connected to {PORT} at {BAUD} baud")
except serial.SerialException as e:
    print(f"Failed to open port: {e}")
    exit(1)

stop_event       = threading.Event()
recording_active = threading.Event()   # set only after START is pressed
auto_stop_timer  = None


# --- Auto STOP after duration ---
def auto_stop(minutes):
    seconds = int(minutes * 60)
    print(f"\n[TIMER] Auto STOP will trigger in {minutes} minute(s)...")

    for remaining in range(seconds, 0, -1):
        if stop_event.is_set():
            return
        if remaining in [60, 30, 10, 5]:
            print(f"\n[TIMER] Auto STOP in {remaining} seconds...")
        threading.Event().wait(1)

    if not stop_event.is_set():
        print("\n[TIMER] Time's up! Sending STOP...")
        try:
            ser.write(b"STOP\n")
        except:
            pass
        stop_event.set()
        ser.close()  # Forces readline() to throw and break out of reader loop


# --- Thread 1: Reads serial data and writes to CSV ---
# File is opened lazily on the first real data line so that the
# filename (set during START) is finalised before writing begins.
def serial_reader():
    header_written = False
    f = None
    writer = None
    try:
        while not stop_event.is_set():
            try:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if line and not stop_event.is_set():
                    print(f"\n[DATA] {line}")
                    if line.startswith("#"):
                        continue  # skip comment lines
                    # Discard all data until START is pressed
                    if not recording_active.is_set():
                        continue
                    if line.startswith("timestamp_ms"):
                        if header_written:
                            continue  # skip duplicate headers
                        header_written = True
                    # Open file on first real data line (filename is set by START handler)
                    if f is None:
                        f = open(filename, "w", newline="")
                        writer = csv.writer(f)
                    fields = line.split(",")
                    # Replace ESP32 relative timestamp with absolute Unix ms
                    if header_written and not line.startswith("timestamp_ms"):
                        fields[0] = str(int(datetime.now().timestamp() * 1000))
                    writer.writerow(fields)
                    f.flush()
            except Exception:
                break  # Port closed or error — exit cleanly
    finally:
        if f:
            f.close()


# --- Thread 2: Handles user command input ---
def command_input():
    global auto_stop_timer, filename

    print_menu()
    while not stop_event.is_set():
        try:
            choice = input("\nEnter command number: ").strip().lower()

            if choice == "q":
                if auto_stop_timer and auto_stop_timer.is_alive():
                    print("[TIMER] Auto stop cancelled.")
                try:
                    ser.write(b"STOP\n")
                    print("Sent: STOP")
                except:
                    pass
                stop_event.set()
                ser.close()
                break

            elif choice == "1":  # START — ask for duration then session type
                try:
                    # Step 1: Duration
                    minutes_input = input("Run for how many minutes? (press Enter to run indefinitely): ").strip()

                    # Step 2: Session type
                    print("\nWhich of the following five options would you like to select?")
                    for key, label in SESSION_TYPES.items():
                        print(f"  [{key}] {label}")
                    type_choice = input("Enter option (1-5): ").strip()

                    if type_choice in SESSION_TYPES:
                        session_type = SESSION_TYPES[type_choice]
                    else:
                        print("Invalid choice. Defaulting to 'Static'.")
                        session_type = "Static"

                    # Optional degree value for filename
                    deg_input = input("Enter degree value for filename (e.g. 8, 45) or press Enter to skip: ").strip()
                    deg_suffix = ""
                    if deg_input:
                        try:
                            deg_suffix = f"_{int(deg_input):02d}deg"
                        except ValueError:
                            print("Invalid degree value, skipping.")

                    # Update filename to include session type and optional degree
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = os.path.join(DATA_DIR, f"imu_emg_log_{timestamp_str}_{session_type}{deg_suffix}.csv")

                    # Send session type BEFORE START so label is set before any rows stream
                    ser.write(f"SESSION_TYPE:{session_type}\n".encode())
                    print(f"Sent: SESSION_TYPE:{session_type}")
                    ser.write(b"START\n")
                    print("Sent: START")
                    print(f"\n>>> Saving data to: {filename}\n")
                    recording_active.set()

                    # Step 3: Start auto-stop timer if duration given
                    if minutes_input:
                        minutes = float(minutes_input)
                        if minutes > 0:
                            auto_stop_timer = threading.Thread(
                                target=auto_stop,
                                args=(minutes,),
                                daemon=True
                            )
                            auto_stop_timer.start()
                        else:
                            print("Invalid duration, running indefinitely.")
                    else:
                        print("Running indefinitely. Send STOP manually.")

                except ValueError:
                    print("Invalid input, running indefinitely.")

            elif choice == "2":  # Manual STOP — cancel timer too
                if auto_stop_timer and auto_stop_timer.is_alive():
                    stop_event.set()
                    print("[TIMER] Auto stop cancelled.")
                    stop_event.clear()  # Clear so logging keeps going until quit
                recording_active.clear()
                try:
                    ser.write(b"STOP\n")
                    print("Sent: STOP")
                except:
                    pass

            elif choice in COMMANDS:
                cmd = COMMANDS[choice]
                try:
                    ser.write(f"{cmd}\n".encode())
                    print(f"Sent: {cmd}")
                except serial.SerialException as e:
                    print(f"Failed to send command: {e}")

            else:
                print("Invalid choice.")
                print_menu()

        except (EOFError, KeyboardInterrupt):
            stop_event.set()
            break


# --- Start both threads ---
reader_thread = threading.Thread(target=serial_reader, daemon=True)
input_thread = threading.Thread(target=command_input, daemon=True)

reader_thread.start()
input_thread.start()

# Unblocks the moment any thread sets stop_event
stop_event.wait()

try:
    ser.close()
except:
    pass

print(f"\nPort closed. Data saved to {filename}")
