from SETTINGS import *
import SETTINGS
from bluepy import btle
import traceback
import sys
import time as t
import binascii
import struct
import logging
import subprocess
import time


class ScanDelegate(btle.DefaultDelegate):
    def __init__(self):
        btle.DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            print("Discovered device", dev.addr)
        elif isNewData:
            print("Received new data from", dev.addr)

class BLEDelegate(btle.DefaultDelegate):
    def __init__(self,params):
        btle.DefaultDelegate.__init__(self)
        self.lock = params

    def handleNotification(self,cHandle,data):
        print("handling notification...")
        print("ble.py: " + str(data))
        SETTINGS.TEMP = data
        t_start = t.time()
        t_end = t_start + query_time*query_multipler
        while (t.time() < t_end):
            have_data_lock = self.lock.acquire(0)
            if have_data_lock:
                SETTINGS.NEW_MSG = True
                SETTINGS.DATA = SETTINGS.TEMP
                logging.debug("SETTINGS.DATA: " + str(SETTINGS.DATA))
                self.lock.release()
                have_data_lock = False
                break

def scan_ble(time_scan):
    scanner =  btle.Scanner().withDelegate(ScanDelegate())
    devices = scanner.scan(time_scan)
    found = None
    try:
        if len(devices) != 0:
            for dev in devices:
                print("Found: Device %s (%s), RSSI=%d dB" % (dev.addr, dev.addrType, dev.rssi))
                for (adtype, desc, value) in dev.getScanData():
                    if desc == SCAN_FIELD and value[0:3] == SCAN_VALUE:
                        found = dev
                        print("Found Dev: " + dev.getValueText(9))
                        break
        else:
            print("Found No Devices...")
        return found
    except btle.BTLEException as e:
        print("Scanning Error. Error:")
        print(e)

def connect_ble(scan_obj,lock):
    if scan_obj is None:
        char_obj = None
        per_obj = None
        return char_obj, per_obj, scan_obj
    else:
        try:
            device = btle.Peripheral(scan_obj).withDelegate(BLEDelegate(lock))
            print("Connected to "+scan_obj.getValueText(9))
            chars = device.getCharacteristics(uuid = LASER_UUID)
            if len(chars) != 1:
                print("More than 1 char...disconnect...")
                device.disconnect()
                return
            print("Subscribing to characteristic...")
            device.writeCharacteristic(chars[0].getHandle() + 1,'\x01\x00')
            print("Subscribed")
            return chars[0],device,scan_obj
        except btle.BTLEException as e:
            traceback.print_exc(file = sys.stdout)
            print("Unable to Connect. Error:")
            print(e)
            return None, None, None
            
def disconnect_ble(per_obj):
    try:
        per_obj.disconnect();
    except btle.BTLEException as e:
        traceback.print_exc(file = sys.stdout);
        print("Unable to Disconnect. Error: ");
        print(e);
    except:
        print("Unable to Disconnect");

def write_ble(char_obj,data_string):
    try:
        char_obj.write(data_string)
        print("Writing Successful")
    except:
        traceback.print_exc(file = sys.stdout)
        print("Writing Failed...")

def wait_notif(obj,time_wait):
    try:
        obj.waitForNotifications(time_wait);
    except:
        traceback.print_exc(file = sys.stdout)
        print("Waiting for Notif. Failed...")
        
def reset_BLE():
    check_0 = subprocess.call("sudo hciconfig hci0 down", shell = True);
    time.sleep(0.5);
    check_1 = subprocess.call("sudo hciconfig hci0 up", shell = True);
    if check_0 == 0 and check_1 == 0:
        flag = True;
    else: 
        flag = None;
    
    return flag



