/*
 * CervicalLogger.ino
 * ─────────────────────────────────────────────────────────────────────────────
 * Biomechanics Data Acquisition System
 * ESP32-S3-WROOM | MPU6050 IMU | MyoWare 2.0 EMG
 *
 * Science Fair: Cervical Neuromuscular Stabilization Experiment
 *
 * Computes: neck flexion angle, angular velocity, cervical torque,
 *           EMG RMS, EMG %MVC
 *
 * Output:   100 Hz CSV stream via USB Serial at 115200 baud
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Preferences.h>
#include <math.h>

// ─── Pin Assignments ──────────────────────────────────────────────────────────
#define PIN_SDA         8
#define PIN_SCL         9
#define PIN_EMG_ENV     4
#define PIN_EMG_RAW     5
#define PIN_LED         10

// ─── Sampling ─────────────────────────────────────────────────────────────────
#define SAMPLE_RATE_HZ          100
#define SAMPLE_INTERVAL_US      (1000000UL / SAMPLE_RATE_HZ)   // 10 000 µs

// ─── Complementary Filter ─────────────────────────────────────────────────────
#define ALPHA                   0.98f

// ─── Biomechanical Constants ──────────────────────────────────────────────────
#define HEAD_WEIGHT_N           50.0f
#define MOMENT_ARM_M            0.08f

// ─── EMG RMS ──────────────────────────────────────────────────────────────────
#define RMS_WINDOW              20      // 200 ms @ 100 Hz

// ─── MVC Calibration ──────────────────────────────────────────────────────────
#define MVC_CAL_DURATION_MS     6000UL  // 6-second calibration window
#define PLATEAU_WINDOW          40      // 400 ms plateau window (samples @ 100 Hz)
#define PLATEAU_STABILITY_RATIO 0.10f   // max (max-min)/mean allowed for stable plateau

// ─── Gyro Bias Calibration ────────────────────────────────────────────────────
#define GYRO_CAL_SAMPLES        500

// ─── State Machine ────────────────────────────────────────────────────────────
typedef enum {
  STATE_IDLE,
  STATE_CALIBRATING_MVC,
  STATE_STREAMING,
  STATE_ERROR
} SystemState;

// ─── Objects ──────────────────────────────────────────────────────────────────
Adafruit_MPU6050 mpu;
Preferences      prefs;

// ─── Global State ─────────────────────────────────────────────────────────────
SystemState state = STATE_IDLE;

// IMU
float gyroBiasY_dps           = 0.0f;
float angle_deg               = 0.0f;
float angle_offset            = 0.0f;  // set by ZERO_ANGLE, subtracted from output
float accel_angle_deg         = 0.0f;
float angular_velocity_dps    = 0.0f;
float prev_angular_velocity_dps = 0.0f;  // for acceleration derivation
float angular_accel_dps2      = 0.0f;    // degrees per second squared
float cervical_torque_nm      = 0.0f;
float ax, ay, az;                         // accelerometer m/s²
float gx, gy, gz;                         // gyroscope rad/s
bool  firstSample             = true;     // suppress invalid accel on first cycle

// EMG
int   emg_env_raw        = 0;
int   emg_raw_raw        = 0;
float emgBuffer[RMS_WINDOW];
int   emgBufIdx          = 0;
float emg_rms            = 0.0f;
float emg_percent_mvc    = -1.0f;
float mvc_rms_reference  = 0.0f;
bool  mvc_calibrated     = false;

// Timing
unsigned long lastSampleTime_us = 0;
float         dt                = 0.01f; // seconds, updated each cycle

// LED manual override flag
bool ledManualOn = false;

// Session type label — set by SESSION_TYPE command from host, printed in status column
char session_label[16] = "BASELINE";

// ─── Forward Declarations ─────────────────────────────────────────────────────
void        initIMU();
void        calibrateGyroBias();
void        readIMU();
float       computeAccelAngle();
void        computeAngularAcceleration(float dt_s);
void        updateComplementaryFilter(float dt_s);
float       computeCervicalTorque();
void        readEMG();
void        updateEMGRMS();
void        handleSerialCommands();
void        runMVCCalibration();
void        printCSVHeader();
void        printCSVRow(unsigned long ts_ms);
void        updateLEDState();
const char* stateString();


// ═════════════════════════════════════════════════════════════════════════════
// SETUP
// ═════════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  // GPIO
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  // ADC configuration
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // I2C
  Wire.begin(PIN_SDA, PIN_SCL);

  // IMU
  initIMU();

  // Load persisted MVC reference from NVS flash
  prefs.begin("cervical", false);
  mvc_rms_reference = prefs.getFloat("mvc_ref", 0.0f);
  if (mvc_rms_reference > 0.0f) {
    mvc_calibrated = true;
    Serial.printf("# MVC reference loaded from flash: %.4f\n", mvc_rms_reference);
  } else {
    Serial.println("# MVC not calibrated. Send CALIBRATE_MVC to calibrate.");
  }

  angle_offset = prefs.getFloat("angle_offset", 0.0f);
  if (angle_offset != 0.0f) {
    Serial.printf("# Angle offset loaded from flash: %.4f deg\n", angle_offset);
  }

  // Gyro bias calibration (sensor must be still)
  calibrateGyroBias();

  // Clear EMG circular buffer
  memset(emgBuffer, 0, sizeof(emgBuffer));

  // Print header and initial status
  printCSVHeader();
  Serial.println("# System ready. Send START to begin streaming.");

  lastSampleTime_us = micros();
  state = STATE_IDLE;
}


// ═════════════════════════════════════════════════════════════════════════════
// LOOP
// ═════════════════════════════════════════════════════════════════════════════
void loop() {
  // 1. Handle incoming serial commands
  handleSerialCommands();

  // 2. MVC calibration is a blocking routine; run and return
  if (state == STATE_CALIBRATING_MVC) {
    runMVCCalibration();
    return;
  }

  // 3. Enforce 100 Hz timing window
  unsigned long now_us = micros();
  if ((now_us - lastSampleTime_us) < SAMPLE_INTERVAL_US) return;

  dt = (float)(now_us - lastSampleTime_us) / 1e6f;
  lastSampleTime_us = now_us;

  // 4. Only run sampling sequence when STREAMING
  if (state != STATE_STREAMING) {
    updateLEDState();
    return;
  }

  // ── Synchronized Sampling Sequence (one CSV row per cycle) ───────────────

  // Step 1 — Timestamp
  unsigned long timestamp_ms = millis();

  // Step 2 — Read IMU (sets ax, ay, az, angular_velocity_dps)
  readIMU();

  // Step 3 — Accelerometer tilt estimate
  accel_angle_deg = computeAccelAngle();

  // Step 4 — Angular velocity already computed in readIMU()

  // Step 5 — Angular acceleration (derived from previous angular velocity)
  computeAngularAcceleration(dt);

  // Step 6 — Update complementary filter angle
  updateComplementaryFilter(dt);

  // Step 7 — Cervical torque
  cervical_torque_nm = computeCervicalTorque();

  // Step 8 — Read EMG channels
  readEMG();

  // Steps 9 & 10 — Update circular buffer and compute RMS
  updateEMGRMS();

  // Step 11 — Compute %MVC
  if (mvc_calibrated && mvc_rms_reference > 0.0f) {
    emg_percent_mvc = 100.0f * (emg_rms / mvc_rms_reference);
  } else {
    emg_percent_mvc = -1.0f;
  }

  // Step 12 — Store angular velocity for next cycle's acceleration computation
  prev_angular_velocity_dps = angular_velocity_dps;
  firstSample = false;

  // Step 13 — Output one CSV row
  printCSVRow(timestamp_ms);

  updateLEDState();
}


// ═════════════════════════════════════════════════════════════════════════════
// FUNCTION IMPLEMENTATIONS
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// initIMU
// Initialise MPU6050 with required sensor ranges and filter bandwidth.
// Halts in ERROR state if the device is not detected.
// ─────────────────────────────────────────────────────────────────────────────
void initIMU() {
  if (!mpu.begin()) {
    Serial.println("# ERROR: MPU6050 not detected. Check wiring on SDA/GPIO8 and SCL/GPIO9.");
    state = STATE_ERROR;
    while (true) {
      digitalWrite(PIN_LED, HIGH); delay(150);
      digitalWrite(PIN_LED, LOW);  delay(150);
    }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("# MPU6050 initialised (±8g, ±500°/s, 21 Hz DLPF).");
}

// ─────────────────────────────────────────────────────────────────────────────
// calibrateGyroBias
// Collects GYRO_CAL_SAMPLES readings on the pitch axis (Y) while the sensor
// is stationary and stores the mean as the bias to subtract at runtime.
// ─────────────────────────────────────────────────────────────────────────────
void calibrateGyroBias() {
  Serial.println("# Gyro bias calibration — keep sensor completely still...");
  double sumY = 0.0;
  for (int i = 0; i < GYRO_CAL_SAMPLES; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    sumY += (double)g.gyro.y;   // pitch axis in rad/s
    delay(2);                    // ~500 Hz during calibration pass
  }
  // Convert mean rad/s → deg/s and store
  gyroBiasY_dps = (float)(sumY / GYRO_CAL_SAMPLES) * (180.0f / PI);
  Serial.printf("# Gyro bias (pitch axis): %.5f deg/s\n", gyroBiasY_dps);
}

// ─────────────────────────────────────────────────────────────────────────────
// readIMU
// Fetches one sensor event from the MPU6050.
// Stores accelerometer axes and computes bias-corrected angular velocity.
// ─────────────────────────────────────────────────────────────────────────────
void readIMU() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  ax = a.acceleration.x;   // m/s²
  ay = a.acceleration.y;
  az = a.acceleration.z;

  // Gyro Y = pitch axis; convert rad/s → deg/s, subtract calibrated bias
  angular_velocity_dps = (g.gyro.y * (180.0f / PI)) - gyroBiasY_dps;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeAccelAngle
// Returns neck pitch angle from accelerometer data alone (degrees).
// Formula: atan2(az, sqrt(ax² + ay²))
// Breadboard mounted vertically on back of neck:
//   0°  = head upright (breadboard vertical, az ≈ 0)
//  90°  = head fully flexed forward (breadboard horizontal, az ≈ g)
// If angles read negative during forward flexion, change az to -az.
// ─────────────────────────────────────────────────────────────────────────────
float computeAccelAngle() {
  return atan2f(az, sqrtf(ax * ax + ay * ay)) * (180.0f / PI);
}

// ─────────────────────────────────────────────────────────────────────────────
// computeAngularAcceleration
// Derives angular acceleration from the change in angular velocity over dt.
// On the first sample there is no valid previous value, so outputs 0.
// Units: degrees per second squared (°/s²)
// ─────────────────────────────────────────────────────────────────────────────
void computeAngularAcceleration(float dt_s) {
  if (firstSample || dt_s <= 0.0f) {
    angular_accel_dps2 = 0.0f;
  } else {
    angular_accel_dps2 = (angular_velocity_dps - prev_angular_velocity_dps) / dt_s;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// updateComplementaryFilter
// Blends gyro integration with accelerometer tilt using a 0.98 / 0.02 split.
// Result stored in global angle_deg.
// ─────────────────────────────────────────────────────────────────────────────
void updateComplementaryFilter(float dt_s) {
  float angle_gyro = angle_deg + angular_velocity_dps * dt_s;
  angle_deg = ALPHA * angle_gyro + (1.0f - ALPHA) * accel_angle_deg;
}

// ─────────────────────────────────────────────────────────────────────────────
// computeCervicalTorque
// Returns estimated gravitational torque on the cervical spine (Newton-metres).
// τ = F × d × sin(θ)
// ─────────────────────────────────────────────────────────────────────────────
float computeCervicalTorque() {
  return HEAD_WEIGHT_N * MOMENT_ARM_M * sinf(angle_deg * PI / 180.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// readEMG
// Reads both MyoWare 2.0 outputs via 12-bit ADC.
// ENV  → emg_env_raw  (processed envelope)
// RAW  → emg_raw_raw  (used for RMS computation)
// ─────────────────────────────────────────────────────────────────────────────
void readEMG() {
  emg_env_raw = analogRead(PIN_EMG_ENV);
  emg_raw_raw = analogRead(PIN_EMG_RAW);
}

// ─────────────────────────────────────────────────────────────────────────────
// updateEMGRMS
// Inserts the latest RAW sample into the 20-sample circular buffer and
// recomputes the windowed RMS value.
// ─────────────────────────────────────────────────────────────────────────────
void updateEMGRMS() {
  emgBuffer[emgBufIdx] = (float)emg_raw_raw;
  emgBufIdx = (emgBufIdx + 1) % RMS_WINDOW;

  // Remove DC offset before computing RMS so the result reflects
  // actual AC muscle signal amplitude rather than the ADC bias level.
  float mean = 0.0f;
  for (int i = 0; i < RMS_WINDOW; i++) mean += emgBuffer[i];
  mean /= (float)RMS_WINDOW;

  float sumSq = 0.0f;
  for (int i = 0; i < RMS_WINDOW; i++) {
    float centered = emgBuffer[i] - mean;
    sumSq += centered * centered;
  }
  emg_rms = sqrtf(sumSq / (float)RMS_WINDOW);
}

// ─────────────────────────────────────────────────────────────────────────────
// printCSVHeader
// Outputs the column header row to Serial.
// ─────────────────────────────────────────────────────────────────────────────
void printCSVHeader() {
  Serial.println("timestamp_ms,angle_deg,angular_velocity_deg_s,angular_accel_deg_s2,"
                 "cervical_torque_nm,emg_env_raw,emg_raw_raw,emg_rms,emg_percent_mvc,"
                 "accel_angle_deg,status");
}

// ─────────────────────────────────────────────────────────────────────────────
// printCSVRow
// Outputs one complete, synchronised data row to Serial.
// All values originate from the same sampling cycle.
// ─────────────────────────────────────────────────────────────────────────────
void printCSVRow(unsigned long ts_ms) {
  Serial.print(ts_ms);
  Serial.print(',');
  Serial.print(angle_deg - angle_offset, 2);
  Serial.print(',');
  Serial.print(angular_velocity_dps, 2);
  Serial.print(',');
  Serial.print(angular_accel_dps2, 2);
  Serial.print(',');
  Serial.print(cervical_torque_nm, 4);
  Serial.print(',');
  Serial.print(emg_env_raw);
  Serial.print(',');
  Serial.print(emg_raw_raw);
  Serial.print(',');
  Serial.print(emg_rms, 4);
  Serial.print(',');
  if (emg_percent_mvc < 0.0f) {
    Serial.print("-1");
  } else {
    Serial.print(emg_percent_mvc, 2);
  }
  Serial.print(',');
  Serial.print(accel_angle_deg - angle_offset, 2);
  Serial.print(',');
  Serial.println(session_label);
}

// ─────────────────────────────────────────────────────────────────────────────
// handleSerialCommands
// Parses single-line commands from the host computer.
// Non-command lines starting with '#' are silently ignored.
// ─────────────────────────────────────────────────────────────────────────────
void handleSerialCommands() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  if (cmd == "START") {
    if (state == STATE_IDLE || state == STATE_ERROR) {
      state = STATE_IDLE;         // reset ERROR if applicable
      memset(emgBuffer, 0, sizeof(emgBuffer));
      emgBufIdx = 0;
      lastSampleTime_us = micros();
      state = STATE_STREAMING;
      digitalWrite(PIN_LED, HIGH);  // LED on immediately when streaming starts
      printCSVHeader();
    } else if (state == STATE_STREAMING) {
      Serial.println("# Already streaming.");
    } else {
      Serial.println("# Cannot start: calibration in progress.");
    }

  } else if (cmd == "STOP") {
    if (state == STATE_STREAMING) {
      state = STATE_IDLE;
      digitalWrite(PIN_LED, LOW);   // LED off immediately when streaming stops
      strncpy(session_label, "BASELINE", sizeof(session_label) - 1);
      Serial.println("# Streaming stopped.");
    } else {
      Serial.println("# Not currently streaming.");
    }

  } else if (cmd.startsWith("SESSION_TYPE:")) {
    String label = cmd.substring(13);  // extract text after "SESSION_TYPE:"
    label.trim();
    if (label == "BASELINE" || label == "STATIC" || label == "DYNAMIC" || label == "FATIGUE" || label == "RECOVERY") {
      label.toCharArray(session_label, sizeof(session_label));
      Serial.printf("# Session type set to: %s\n", session_label);
    } else {
      Serial.println("# Unknown session type. Use: Baseline, Static, Dynamic, Fatigue, Recovery.");
    }

  } else if (cmd == "CALIBRATE_MVC") {
    if (state == STATE_STREAMING || state == STATE_IDLE) {
      state = STATE_CALIBRATING_MVC;
      // runMVCCalibration() will be called at the top of loop()
    } else {
      Serial.println("# Cannot calibrate in current state.");
    }

  } else if (cmd == "RESET_MVC") {
    mvc_rms_reference = 0.0f;
    mvc_calibrated    = false;
    emg_percent_mvc   = -1.0f;
    prefs.putFloat("mvc_ref", 0.0f);
    Serial.println("# MVC reference cleared and removed from flash.");

  } else if (cmd == "ZERO_ANGLE") {
    if (state == STATE_STREAMING) {
      // Use the live complementary filter angle — already stable
      angle_offset = angle_deg;
    } else {
      // IDLE: take a fresh accelerometer reading on the spot
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      float fresh_ax = a.acceleration.x;
      float fresh_ay = a.acceleration.y;
      float fresh_az = a.acceleration.z;
      angle_offset = atan2f(fresh_az, sqrtf(fresh_ax * fresh_ax + fresh_ay * fresh_ay))
                    * (180.0f / PI);
    }
    prefs.putFloat("angle_offset", angle_offset);
    Serial.printf("# Angle zeroed. Offset = %.4f deg (saved to flash).\n", angle_offset);

  } else if (cmd == "RESET_ANGLE") {
    angle_offset = 0.0f;
    prefs.putFloat("angle_offset", 0.0f);
    Serial.println("# Angle offset cleared and removed from flash.");

  } else if (cmd == "STATUS") {
    Serial.printf("# State        : %s\n", stateString());
    Serial.printf("# MVC ref      : %.4f\n", mvc_rms_reference);
    Serial.printf("# Calibrated   : %s\n", mvc_calibrated ? "YES" : "NO");
    Serial.printf("# Gyro bias    : %.5f deg/s\n", gyroBiasY_dps);
    Serial.printf("# angle_deg    : %.2f\n", angle_deg);
    Serial.printf("# angle_offset : %.4f deg\n", angle_offset);
    Serial.printf("# angle_output : %.2f deg\n", angle_deg - angle_offset);

  } else if (cmd == "HEADER") {
    printCSVHeader();

  } else if (cmd == "LED_ON") {
    ledManualOn = true;
    digitalWrite(PIN_LED, HIGH);
    Serial.println("# LED on.");

  } else if (cmd == "LED_OFF") {
    ledManualOn = false;
    digitalWrite(PIN_LED, LOW);
    Serial.println("# LED off.");

  } else {
    Serial.printf("# Unknown command: %s\n", cmd.c_str());
    Serial.println("# Valid: START | STOP | CALIBRATE_MVC | RESET_MVC | ZERO_ANGLE | RESET_ANGLE | STATUS | HEADER | LED_ON | LED_OFF");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// runMVCCalibration  (blocking)
//
// Algorithm:
//   • Runs for MVC_CAL_DURATION_MS (6 seconds) at 100 Hz.
//   • Maintains a PLATEAU_WINDOW-sample sliding window of RMS values.
//   • A "stable plateau" is detected when (max − min) / mean ≤ 10 %.
//   • Tracks the highest such plateau RMS found during the session.
//   • Saves the result to NVS flash via Preferences.
// ─────────────────────────────────────────────────────────────────────────────
void runMVCCalibration() {
  Serial.println("# ── MVC CALIBRATION STARTED ─────────────────────────────");
  Serial.println("# Perform 3 maximal shoulder shrugs over the next 6 seconds.");
  Serial.println("# Hold each shrug firmly for ~0.5 s.  Send STOP to abort.");
  Serial.println("# ─────────────────────────────────────────────────────────");

  // Local EMG circular buffer (independent of streaming buffer)
  float localBuf[RMS_WINDOW];
  int   localIdx = 0;
  memset(localBuf, 0, sizeof(localBuf));

  // Plateau detection sliding window
  float plateauBuf[PLATEAU_WINDOW];
  int   plateauIdx   = 0;
  int   plateauFill  = 0;    // counts up to PLATEAU_WINDOW before we start evaluating
  float bestMVC      = 0.0f;

  memset(plateauBuf, 0, sizeof(plateauBuf));

  unsigned long calStart  = millis();
  unsigned long lastSampleUs = micros();
  unsigned long lastPrintMs  = 0;

  while ((millis() - calStart) < MVC_CAL_DURATION_MS) {

    // Allow aborting via STOP command
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim(); cmd.toUpperCase();
      if (cmd == "STOP") {
        Serial.println("# MVC calibration aborted.");
        digitalWrite(PIN_LED, LOW);
        state = STATE_IDLE;
        return;
      }
    }

    // Enforce 100 Hz
    unsigned long nowUs = micros();
    if ((nowUs - lastSampleUs) < SAMPLE_INTERVAL_US) continue;
    lastSampleUs = nowUs;

    // Sample RAW EMG
    int rawSample = analogRead(PIN_EMG_RAW);
    localBuf[localIdx] = (float)rawSample;
    localIdx = (localIdx + 1) % RMS_WINDOW;

    // Compute current window RMS (DC-subtracted, consistent with updateEMGRMS)
    float localMean = 0.0f;
    for (int i = 0; i < RMS_WINDOW; i++) localMean += localBuf[i];
    localMean /= (float)RMS_WINDOW;

    float sumSq = 0.0f;
    for (int i = 0; i < RMS_WINDOW; i++) {
      float c = localBuf[i] - localMean;
      sumSq += c * c;
    }
    float currentRMS = sqrtf(sumSq / (float)RMS_WINDOW);

    // Update plateau detection buffer
    plateauBuf[plateauIdx] = currentRMS;
    plateauIdx = (plateauIdx + 1) % PLATEAU_WINDOW;
    if (plateauFill < PLATEAU_WINDOW) plateauFill++;

    // Evaluate plateau once buffer is full
    if (plateauFill == PLATEAU_WINDOW) {
      float pSum = 0.0f, pMax = 0.0f, pMin = 1e9f;
      for (int i = 0; i < PLATEAU_WINDOW; i++) {
        pSum += plateauBuf[i];
        if (plateauBuf[i] > pMax) pMax = plateauBuf[i];
        if (plateauBuf[i] < pMin) pMin = plateauBuf[i];
      }
      float pMean = pSum / (float)PLATEAU_WINDOW;

      // Stable plateau: low variation and non-trivial signal
      if (pMean > 10.0f) {   // ignore noise floor (< ~10 ADC counts RMS)
        float variation = (pMean > 0.0f) ? (pMax - pMin) / pMean : 1.0f;
        if (variation <= PLATEAU_STABILITY_RATIO && pMean > bestMVC) {
          bestMVC = pMean;
        }
      }
    }

    // Progress report every second
    unsigned long elapsed = millis() - calStart;
    if (elapsed >= lastPrintMs + 1000) {
      lastPrintMs = elapsed;
      Serial.printf("# t = %lu s | RMS = %.1f | Best plateau = %.1f\n",
                    elapsed / 1000UL, currentRMS, bestMVC);
    }

    // LED: fast blink during calibration (250 ms period)
    digitalWrite(PIN_LED, ((elapsed / 250) % 2 == 0) ? HIGH : LOW);
  }

  digitalWrite(PIN_LED, LOW);

  if (bestMVC > 0.0f) {
    mvc_rms_reference = bestMVC;
    mvc_calibrated    = true;
    prefs.putFloat("mvc_ref", mvc_rms_reference);
    Serial.printf("# ✓ MVC calibration complete. Reference RMS = %.4f (saved to flash)\n",
                  mvc_rms_reference);
  } else {
    Serial.println("# WARNING: No stable plateau detected.");
    Serial.println("# Try stronger, held contractions. MVC reference was NOT updated.");
  }

  state = STATE_IDLE;
  Serial.println("# Send START to resume streaming.");
}

// ─────────────────────────────────────────────────────────────────────────────
// updateLEDState
// STREAMING : solid ON (set immediately by START command)
// IDLE      : LED off (unless manually overridden by LED_ON command)
// ERROR     : handled inside initIMU() — fast alternating blink, never exits
// ─────────────────────────────────────────────────────────────────────────────
void updateLEDState() {
  if (state != STATE_STREAMING) {
    // Respect manual LED command when not streaming
    digitalWrite(PIN_LED, ledManualOn ? HIGH : LOW);
  }
  // During STREAMING the LED is already HIGH from the START handler;
  // no action needed here — it stays on until STOP turns it off.
}

// ─────────────────────────────────────────────────────────────────────────────
// stateString — returns human-readable label for the current state
// ─────────────────────────────────────────────────────────────────────────────
const char* stateString() {
  switch (state) {
    case STATE_IDLE:            return "IDLE";
    case STATE_CALIBRATING_MVC: return "CALIBRATING_MVC";
    case STATE_STREAMING:       return "STREAMING";
    case STATE_ERROR:           return "ERROR";
    default:                    return "UNKNOWN";
  }
}
