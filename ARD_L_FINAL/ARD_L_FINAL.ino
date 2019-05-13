#include <Adafruit_Sensor.h>
#include <Adafruit_MMA8451.h>
#include <SoftwareSerial.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

SoftwareSerial ble(7,8); //RX,TX for Bluetooth

unsigned long read_attempts = 20;
unsigned long timeout = 20*1000; //milliseconds
double accel_x, accel_y, accel_z;
double pi = 3.14159265359;
char buff[6];
String state;
String send_pitch;
Adafruit_MMA8451 mma = Adafruit_MMA8451();

String read_BLE(int wake) {
  char term;
  unsigned long t_start = millis();
  unsigned long t_cur = 0;
  String read_val;
  if (wake == 1) {
    term = 'N';
  } else {
    term = '?';
  }
  while (1) {
    if (ble.available()){
      read_val = ble.readStringUntil(term);
      break;
    } else {
      t_cur = millis();
      if ((t_cur - t_start) > (read_attempts*timeout)){
        break;
      }
    }
  }
  Serial.println(read_val);
  return read_val;
}

void BLE_flush() {
  while(1) {
    if (ble.available()){
      ble.read();
    } else {
      break;
    }
  }
}

double calc_pitch(double x, double y, double z){
  double pitch = -1*atan2((-x),sqrt(y*y+z*z))*(180/2/pi);
  return pitch;
}

void laser_ON(){
  digitalWrite(13,HIGH);
  digitalWrite(12,LOW);
}

void laser_OFF(){
  digitalWrite(13,LOW);
  digitalWrite(12,HIGH);
}

void setup() {
  pinMode(13,OUTPUT); //pins for Laser Relay
  pinMode(12,OUTPUT); // 13 HIGH + 12 LO = ON

  laser_OFF(); //initialize laser OFF
  
  Serial.begin(9600);
  ble.begin(9600);
  ble.setTimeout(timeout);
  delay(1000);
  
  ble.print("AT");
  delay(1000);
  ble.print("AT+NAMELAS2");
  delay(1000);
  ble.print("AT+NOTI1");
  delay(1000);
  ble.print("AT+POWE3");
  delay(1000);
  ble.print("AT+RESET");
  
  state = "IDLE";
}

void loop() {

  if (state == "IDLE"){
    Serial.println(state);
    if (read_BLE(1) == "OK+CO"){
      laser_ON();
      state = "WAIT_I";
    }
  } else if (state == "WAIT_I") {
    Serial.println(state);
    BLE_flush();
    if (read_BLE(0) == "TURN_OFF_LASER") {
      laser_OFF();
      ble.print("?LASER_OFF?");
      state = "ACCEL";
    } else {
      state = "SHUTDOWN";
    }
  } else if (state == "ACCEL"){
    Serial.println(state);
    if(!mma.begin()){
      state = "SHUTDOWN";
    }else{
      mma.read();
      accel_x = mma.x;
      accel_y = mma.y;
      accel_z = mma.z;
      double pitch = calc_pitch(accel_x,accel_y,accel_z);
      send_pitch = '?' + String(pitch,DEC) +'?';
      state = "WAIT_II";
    }
  } else if (state == "SEND_ACCEL"){
    Serial.println(state);
    ble.print(send_pitch);
    Serial.println(send_pitch);  
    state = "SHUTDOWN";
  } else if (state == "WAIT_II"){
    Serial.println(state);
    if (read_BLE(0) == "TURN_OFF"){
      state = "SEND_ACCEL";
    } else {
      state = "SHUTDOWN";
    }
  } else if (state == "SHUTDOWN"){
    Serial.println(state);
    laser_OFF();
    state = "IDLE";
  }
}
