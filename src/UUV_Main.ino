#include <Wire.h>
#include <SparkFun_BMI270_Arduino_Library.h>
#include <BasicLinearAlgebra.h>
#include "StateEstimator.h"

// --- Hardware Objects ---
BMI270 imu;
#define gpsSerial Serial1

// --- State Estimator ---
StateEstimator ekf;

// --- Timing ---
unsigned long lastUpdate = 0;

float getPressureZ() {
    // TODO: Read actual pressure sensor
    // Return depth (positive underwater)
    // For demo, we return 5.0m (underwater) then 0.0m (surface) to test logic
    if (millis() > 10000 && millis() < 15000) return 0.0; // Surface event
    return 5.0; 
}

struct GPSData { float x; float y; bool valid; };

GPSData readGPS() {
    // TODO: Parse NMEA sentences from gpsSerial
    // For now, return a dummy valid coord if surfacing
    GPSData d = {0,0, false};
    if (gpsSerial.available()) {
        // If we successfully parse a lat/lon...
        d.x = 10.0; // Local tangent plane coord
        d.y = 10.0;
        d.valid = true;
        // clear buffer
        while(gpsSerial.available()) gpsSerial.read();
    }
    return d;
}

void setup() {
    Serial.begin(115200);
    gpsSerial.begin(9600);
    
    // IMU Setup
    Wire.setSDA(4); Wire.setSCL(5); Wire.begin();
    if (imu.beginI2C(0x68) != BMI2_OK) Serial.println("IMU Error");
    else Serial.println("IMU Ready");
    
    // Initialize EKF with surface start
    ekf.x(IDX_Z) = 0; 
}

void loop() {
    unsigned long now = millis();
    float dt = (now - lastUpdate) / 1000.0;

    if (dt >= 0.05) {
        lastUpdate = now;

        // 1. READ SENSORS
        imu.getSensorData();
        float z_meas = getPressureZ();
        GPSData gps = readGPS();
        
        float accel_L = imu.data.accelX;
        float accel_R = imu.data.accelX; 
        float accel_Z = imu.data.accelZ;

        // 2. PREDICTION
        ekf.predict(accel_L, accel_R, accel_Z, dt);

        // 3. CORRECTION
        // Always correct Z with pressure
        ekf.updatePressure(z_meas);

        // Check surface condition (z approx 0)
        bool isSurface = (abs(ekf.x(IDX_Z)) < 0.5); 

        if (isSurface && gps.valid) {
            Serial.println(">> SURFACE LOCK: Updating with GPS");
            ekf.updateGPS(gps.x, gps.y);
            
            // Trigger RTS Smoother to fix history
            Serial.println(">> Running RTS Smoother on history...");
            ekf.runRTS_Smoother(); 
        }

        // 4. DEBUG OUTPUT
        Serial.print("State [X,Y,Z]: ");
        Serial.print(ekf.x(IDX_X)); Serial.print(", ");
        Serial.print(ekf.x(IDX_Y)); Serial.print(", ");
        Serial.println(ekf.x(IDX_Z));
    }
}