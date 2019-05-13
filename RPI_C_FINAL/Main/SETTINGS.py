import os

BASE_DIR = "/home/pi/Desktop/"

IMG_DIR = os.path.join(BASE_DIR, "images")
ACCELEROMETER = os.path.join(BASE_DIR, "accel_module")
IMG_REC = os.path.join(BASE_DIR, "cv_module")
IMG_CAM = os.path.join(BASE_DIR, "RPiCam")
COMM = os.path.join(BASE_DIR,"communication_module")
RUN_LOG_PATH = os.path.join(BASE_DIR, "Main/run_log.txt")
DATA_LOG_PATH = os.path.join(BASE_DIR, "Main/data_log.txt")
NAME_SCHEME = "img"
NS = "img_%d_{}"
FILE_FORMAT = "png"
RAW_IMAGE = False
NUM_SAMPLES = 2
BUFFER = 100

BRIDGE_NAME = 'bridge-1'
SERVER = "http://apps2.peer.berkeley.edu/caltrans/sensors/"+BRIDGE_NAME+"/update/"
PASSWORD = "djioewfj34jod2jdoi3jr0jl983jsa"

pi = 3.1415926535897932
query_time = 10 #seconds
query_multipler = 4 #*query_time for total wait time

pMap = {3:2,5:3,7:4,8:14,10:15,11:17,12:18,13:27,15:22,16:23,18:24,19:10,21:9,22:25,23:11,24:8,26:7}
power_pin = 24;

#for BLE

LASER_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
SCAN_FIELD = "Complete Local Name"
SCAN_VALUE = "LAS"
marker = "?"
TEMP = None
DATA = None
WRITE = None
NEW_MSG = False

#for FONA
reset_pin = 26
batt_pin = 22
num_tries = 3
host_name = "www.google.com"

