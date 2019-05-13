from os import system
import serial
import subprocess
import time
import RPi.GPIO as GPIO
import socket
from SETTINGS import *

query_time = 15;

#start PPPD
def openPPPD():
    flag = None;
    count = 0;
    while count <= num_tries:
        count = count + 1;
        #check if PPPD is already running
        check = subprocess.call("sudo pon fona-config", shell = True);
        if check == 8: #connect script failed -- FONA probably still asleep and not accessing serial port
            fona_reset();
        elif check == 6: #serial port not accessible, PPP probably already connected
            flag = True;
            break
        elif check == 0: #successfully connected
            flag = True;
            break
        else:
            time.sleep(query_time);

    return flag

#check PPPD:
def checkPPPD():
    flag = None;
    try:
        host = socket.gethostbyname(host_name); #see if we can resolve host name -- if there is DNS
        s = socket.create_connection((host,80),query_time); #see if we can reach host and connect to it
        flag = True;
    except:
        flag = None;

    return flag

#close PPPD
def closePPPD():
    flag = None;
    count = 0;
    while count <= num_tries:
        count = count + 1;
        check = subprocess.call("sudo poff", shell = True);
        if check == 0:
            flag = True;
            break
        elif check == 1: #PPPD already closed
            flag = True;
            break
        else:
            time.sleep(query_time);

    return flag

#put FONA to sleep, remember to disconnect PPPD first to free up serial connection
def fona_sleep():
    try:
        ser = serial.Serial('/dev/serial0',115200,timeout = 10);
        ser.read(128); #flush
        cmd = "AT+CSCLK=1\r";
        ser.write(cmd.encode());
        recv = ser.read(256);
        print("recv: " + recv);
        if "OK" in recv:
            flag = True;
        else:
            flag = None;
    except:
        flag = None;

    return flag

#return battery level from FONA
def fona_batt():
    fona_reset();
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout = 10);
        ser.read(128); #flush
        cmd = "AT+CBC\r";
        ser.write(cmd.encode());
        recv = ser.read(256);
        print("recv: " + str(recv));
        if "CBC" in recv:
            msg = recv.split(":")[1];
            msg = msg.split(",")[2];
            msg = msg.split("\r\n")[0];
            print("FONA Charge: " + msg);
            try:
                if float(msg) > 3800:
                    flag = True;
                else:
                    flag = False;
            except:
                flag = None;
        else:
            flag = None;
    except:
        flag = None;

    return flag

#reset FONA module
def fona_reset():
    flag = None;
    try:
        GPIO.setmode(GPIO.BCM);
        GPIO.setup(pMap[reset_pin],GPIO.OUT);
        GPIO.output(pMap[reset_pin],True);
        time.sleep(1);
        GPIO.output(pMap[reset_pin],False);
        time.sleep(1);
        GPIO.output(pMap[reset_pin],True);
        time.sleep(10);
        flag = True;
    except:
        flag = None;

    return flag

#erase SMS messages saved on fona
def fona_delete_SMS():
    flag = None;
    try:
        ser = serial.Serial('/dev/serial0', 115200, timeout = 10);
        ser.read(128); #flush
        cmd = "AT+CMGD = 1,4\r";
        ser.write(cmd.encode());
        recv = ser.read(640);
        print("recv: " + str(recv));
        if "OK" in recv:
            flag = True;
        else:
            flag = None;
    except:
        flag = None;
    return flag
