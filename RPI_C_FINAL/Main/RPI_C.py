from SETTINGS import *
from ble import *
import SETTINGS
import fona
import subprocess
import traceback
import RPi.GPIO as GPIO
import sys
import time
import threading
import logging
from Read_Accel import read_accel
sys.path.insert(0,IMG_REC)
from sig_proc import center_detect
sys.path.insert(0,IMG_CAM)
from img_capture import ImgCollector
sys.path.insert(0,COMM)
from communication import send_data_to_server

def ble_worker(locks,event,event1):
    ble_state = 'INIT';
    scan_entry = None;
    char_obj = None;
    per_obj = None;
    while True:
        if ble_state == 'INIT':
            have_lock_0 = locks[0].acquire(0);
            scan_entry = scan_ble(query_time);
            char_obj, per_obj, scan_entry = connect_ble(scan_entry,locks[0]);
            if char_obj is None:
                logging.debug("Connection with RPI_L Not Successful");
                if have_lock_0:
                    SETTINGS.DATA = "?NO_CONNECTION?";
                    SETTINGS.NEW_MSG = True;
                    locks[0].release();
                    have_lock_0 = False;
            else:
                if have_lock_0:
                    SETTINGS.DATA = "?CONNECTED?";
                    SETTINGS.NEW_MSG = True;
                    locks[0].release();
                    have_lock_0 = False;
                    ble_state = "READ_WRITE";
        elif ble_state == 'READ_WRITE':
            logging.debug("Entered READ_WRITE");
            ble_w_thread = threading.Thread(target = ble_writer, args =(char_obj,locks[1],event,event1,), name = 'ble_write_thread');
            ble_w_thread.setDaemon(True);
            ble_w_thread.start();
            while not event1.is_set():
                have_b_lock = locks[1].acquire(0);
                if have_b_lock:
                    wait_notif(per_obj,0.1);
                    locks[1].release();
                    have_b_lock = False;
                time.sleep(0.5);
            break
    try:
        disconnect_ble(per_obj);
    except:
        logging.debug("Disconnect error");


def ble_writer(obj,b_lock,b_event,b_event_kill):
    while not b_event_kill.is_set():
        b_event.wait();
        logging.debug("Trying to get b_lock");
        b_lock.acquire();
        logging.debug("b_lock Acquired");
        write_ble(obj,SETTINGS.WRITE);
        b_lock.release();
        logging.debug("b_lock Released");
        time.sleep(1);

def wait_for_msg(locks):
    data_access = False;
    msg = None;
    t_start = time.time();
    t_end = t_start + query_multipler*query_time;
    while (time.time() < t_end):
        try:
            have_lock_0 = locks[0].acquire(0);
            if have_lock_0:
                if SETTINGS.NEW_MSG:
                    logging.debug("wait_for_msg: New MSG");
                    msg = SETTINGS.DATA;
                    SETTINGS.NEW_MSG = False;
                else:
                    locks[0].release();
                    have_lock_0 = False;
        finally:
            if have_lock_0:
                locks[0].release();
                logging.debug("wait_for_msg: Lock 0 Released");
                break
        time.sleep(1);

    if msg is None:
        msg = None;
    else:
        try:
            msg = msg.split("?",2)[1];
        except:
            msg = "PARSE_ERROR";

    logging.debug("wait_for_msg: " + str(msg));

    return msg

def run_log(r_w_flg,path,msg):
    run_counter = -1;
    if r_w_flg == 1:
        if os.path.exists(path):
            rfile = open(path,"r");
            try:
                run_counter = int(rfile.read());
            except ValueError:
                run_counter = -1;
        else:
            rfile = open(path,"w");
            run_counter = 0;
            rfile.write(str(run_counter));
        rfile.close();
    else:
        wfile = open(path, "w");
        wfile.write(str(msg));
        wfile.close();

    return run_counter

def data_log(r_w_flg,path,msg):
    #flag = 1: read in oldest line, delete it
    #flag = 2: write new line
    num_data = 0;
    run = [];
    x_p = [];
    y_p = [];
    p_C = [];
    p_L = [];
    err = [];
    if r_w_flg ==1:
        if os.path.exists(path):
            rfile = open(path,"r");
            for line in open(path).xreadlines():
                num_data = num_data + 1;
            if num_data > 0:
                for line in rfile:
                    s = line.strip("\n").split(",",5);
                    run.append(s[0]);
                    x_p.append(s[1]);
                    y_p.append(s[2]);
                    p_C.append(s[3]);
                    p_L.append(s[4]);
                    err.append(s[5]);
                rfile.close();
                wfile = open(path,"w");
                for i in range(1,num_data):
                    wfile.write(str(run[i]) + "," + str(x_p[i]) + "," + str(y_p[i]) + "," + str(p_C[i]) + "," + str(p_L[i]) + "," + str(err[i]) + "\n");
                wfile.close();
        else:
            rfile = open(path,"w");
            rfile.close();
    else:
        wfile = open(path,"a");
        wfile.write(msg);
        wfile.close();

    if num_data == 0:
        run = [None];
        x_p = [None];
        y_p = [None];
        p_C = [None];
        p_L = [None];
        err = [None];

    return num_data,run[0],x_p[0],y_p[0],p_C[0],p_L[0],err[0]

def main():

    locks = [];
    errors = [];
    pos = [];
    pitch_C = None;
    pitch_L = None;
    error_flag = False;

    run_counter = run_log(1,RUN_LOG_PATH,None);
    logging.debug("run_counter: " + str(run_counter));

    if run_counter >= 0:
        try:
            GPIO.setmode(GPIO.BCM);
            GPIO.setup(pMap[power_pin], GPIO.OUT);
            GPIO.setup(pMap[batt_pin], GPIO.OUT);
            GPIO.output(pMap[power_pin], GPIO.HIGH);
            logging.debug("Pin 24 High");
        except:
            logging.debug("PIN ERROR");
            errors.append(1);
            error_flag = True;
        state = "INIT_0";
    else:
        logging.debug("Issue with Run_File Read, Proceeding to Shutdown");
        errors.append(31);
        error_flag = True;
        state = "SHUTDOWN";

    while True:

        try:
            if state == "INIT_0":
                check = reset_BLE();
                if check is True:
                    state = "INIT";
                else:
                    state = "SHUTDOWN";
                    errors.append(20);
                    error_flag = True;

            elif state == "INIT":

                ble_event = threading.Event();
                ble_kill_event = threading.Event();
                data_lock = threading.Lock();
                ble_lock = threading.Lock();
                locks.append(data_lock);
                locks.append(ble_lock);
                ble_thread = threading.Thread(target = ble_worker, args = (locks,ble_event,ble_kill_event, ), name = 'ble_thread');
                ble_thread.setDaemon(True);
                ble_thread.start();
                time.sleep(3) #wait for BLE to begin and obtain write lock
                logging.debug("Connecting to RPI_L");
                try:
                    msg = wait_for_msg(locks);
                    if (msg == "CONNECTED"):
                        logging.debug("Proceeding to Image Acquisition");
                        state = "IMAGE_I";
                    else:
                        logging.debug("Connection with RPI_L Not Successful");
                        logging.debug("Proceeding to Safe Shutdown");
                        state = "SHUTDOWN";
                        errors.append(11);
                        error_flag = True;

                    msg = None;
                except:
                    logging.debug("INIT ERROR");
                    traceback.print_exc(file = sys.stdout);
                    errors.append(32);
                    error_flag = True;
                    state = "SHUTDOWN";

            elif state == "IMAGE_I":
                try:

                    img_collector = ImgCollector(IMG_DIR, NAME_SCHEME, FILE_FORMAT, RAW_IMAGE, buffer=SETTINGS.BUFFER);
                    logging.debug("Taking Camera Picture of Laser Beam");
                    img_collector.capture();
                    img_collector.shutdown();
                    logging.debug("Done Taking Photo");

                    state = "IMAGE_I_COMM"
                except:
                    traceback.print_exc(file = sys.stdout);
                    logging.debug("IMAGE_1 CAM ERROR");
                    state = "SHUTDOWN";
                    errors.append(2);
                    error_flag = True;

            elif state == "IMAGE_I_COMM":

                try:
                    SETTINGS.WRITE = "TURN_OFF_LASER?";
                    ble_event.set();
                    logging.debug("write_event set");
                    time.sleep(1);
                    ble_event.clear();
                    logging.debug("write_event cleared");

                    msg = wait_for_msg(locks);
                    if msg != None:
                        logging.debug("SETTINGS.DATA: " + str(msg));
                        if msg == "LASER_OFF":
                            logging.debug("Continuing to Ambient Image");
                            state = "IMAGE_II";
                        else:
                            logging.debug("No Laser Off Ack Error");
                            errors.append(12);
                            error_flag = True;
                            state = "SHUTDOWN";
                    else:
                        logging.debug("RPI_L Communication Error");
                        errors.append(13);
                        error_flag = True;
                        state = "SHUTDOWN";
                    msg = None;
                except:
                    traceback.print_exc(file = sys.stdout);
                    logging.debug("IMAGE_I_COMM: ERROR");
                    errors.append(33);
                    error_flag = True;
                    state = "SHUTDOWN";

            elif state == "IMAGE_II":
                try:

                    logging.debug("Taking Ambient Camera Picture");
                    img_collector.init_cam();
                    img_collector.capture();
                    img_collector.shutdown();
                    img_collector.clean_dir();
                    logging.debug("Done Taking Photo");

                    state = "L_ACCEL";
                except:
                    traceback.print_exc(file = sys.stdout);
                    logging.debug("IMAGE_II ERROR");
                    errors.append(3);
                    error_flag = True;
                    state = "SHUTDOWN";

            elif state == "L_ACCEL":
                try:

                    SETTINGS.WRITE = "TURN_OFF?";
                    ble_event.set();
                    logging.debug("write_event set");
                    time.sleep(1);
                    ble_event.clear();
                    logging.debug("write_event cleared");
                    msg = wait_for_msg(locks);

                    if msg != None:
                        logging.debug("SETTINGS.DATA: " + str(msg));
                        try:
                            if any(char.isdigit() for char in msg):
                                pitch_L = msg;
                            else:
                                logging.debug("Pitch_L Not Numeric");
                                errors.append(35);
                                error_flag = True;
                        except:
                            logging.debug("Pitch_L Data Type Error Suspected");
                            errors.append(15);
                            error_flag = True;
                    else:
                        logging.debug("Pitch_L Never Received");
                        errors.append(14);
                        error_flag = True;
                    msg = None;

                    try:
                        ble_kill_event.set(); #end all threads
                        logging.debug("BLE Processes Killed");
                    except:
                        logging.debug("Error setting event to kill BLE threads");
                        errors.append(37);
                        error_flag = True;

                    state = "C_ACCEL";
                except:
                    traceback.print_exc(file = sys.stdout);
                    logging.debug("L_ACCEL ERROR");
                    errors.append(34);
                    error_flag = True;
                    state = "C_ACCEL";

            elif state == "C_ACCEL":
                try:
                    logging.debug("Taking Acceleration 1 Readings at Camera Module");
                    pitch_C = read_accel();
                    logging.debug("pitch_C: " + str(pitch_C));
                    state = "PROCESS";
                except:
                    logging.debug("C_ACCEL ERROR");
                    errors.append(4);
                    error_flag = True
                    state = "PROCESS";

            elif state == "PROCESS":
                try:
                    print("RPICam: Performing Image Processing of Image");
                    pos = center_detect(img_collector.get_last_meas());
                    print("pos = [" + str(pos[0]) + "," + str(pos[1]) + "]");
                    logging.debug("Processed Image");
                    state = "SERVER";
                except:
                    traceback.print_exc(file = sys.stdout);
                    logging.debug("PROCESS ERROR");
                    errors.append(26);
                    error_flag = True;
                    state = "SHUTDOWN";
                    
            elif state == "SERVER":
                logging.debug("Connecting to Internet");
                flag_open = fona.openPPPD();
                logging.debug("PPPD opened: " + str(flag_open));
                time.sleep(2);
                flag_active = fona.checkPPPD();
                logging.debug("PPPD active: " + str(flag_active));
                if flag_active is True: # if a connection is active
                    logging.debug("Sending Data to Server");
                    try:
                        send_data_to_server(x = str(pos[0]), y = str(pos[1]), z = 0.0, theta = str(pitch_C), phi = str(pitch_L), psi = 0.0, errors = str(errors).replace(" ","")[1:-1], counter = run_counter); 
                        n_lines,run,x_p,y_p,p_C,p_L,err = data_log(1,DATA_LOG_PATH,None);
                        while n_lines !=0:
                            send_data_to_server(x= x_p, y= y_p, z=0.0, theta= p_C, phi= p_L, psi=0.0, errors=err, counter = run);
                            logging.debug("Sent line...");
                            n_lines,run,x_p,y_p,p_C,p_L,err = data_log(1,DATA_LOG_PATH,None);
                        error_flag = False;
                        logging.debug("Sent Data");
                    except RuntimeWarning as e:
                        traceback.print_exc(file = sys.stdout);
                        logging.debug("Runtime Warning: Value Not Pushed to Server");
                        errors.append(16);
                        error_flag = True;
                    except:
                        traceback.print_exc(file = sys.stdout);
                        logging.debug("Value Not Pushed to Server");
                        errors.append(17);
                        error_flag = True;

                fona.closePPPD();

                flag_batt = fona.fona_batt();

                if flag_batt:
                    try:
                        GPIO.setmode(GPIO.BCM);
                        GPIO.setup(pMap[batt_pin], GPIO.OUT);
                        GPIO.output(pMap[batt_pin],GPIO.HIGH);
                    except:
                        logging.debug("Error actuating GPIO pin");
                        errors.append(6);
                        error_flag = True;
                elif flag_batt == None:
                    errors.append(5);
                    error_flag = True;
                    
                logging.debug("FONA battery charged: " + str(flag_batt));

                flag_sms = fona.fona_delete_SMS();
                logging.debug("FONA SMS deleted: " + str(flag_sms));

                flag_off = fona.fona_sleep();
                logging.debug("FONA put to sleep: " + str(flag_off));

                if flag_active is None and flag_off is None:
                    errors.append(18);
                    error_flag = True;
                elif flag_active is True and flag_off is None:
                    errors.append(36);
                    error_flag = True;
                elif flag_active is None and flag_off is True:
                    errors.append(19);
                    error_flag = True;

                if flag_sms is None:
                    errors.append(21);
                    error_flag = True;

                state = "SHUTDOWN";

            elif state == "SHUTDOWN":

                try:
                    img_collector.shutdown()
                except:
                    logging.debug("Cant close camera")

                if error_flag:
                    if len(pos) == 0:
                        pos = [None,None];
                    msg = str(run_counter) + "," + str(pos[0]) + "," + str(pos[1]) + "," + str(pitch_C) + "," + str(pitch_L) + "," + str(errors).replace(" ","")[1:-1] + "\n";
                    data_log(2,DATA_LOG_PATH,msg);
                    logging.debug("Data Saved to File");
                    
                run_log(2, RUN_LOG_PATH, run_counter + 1);

                GPIO.cleanup();
                subprocess.call("sudo shutdown -h now", shell = True);
                sys.exit(0);

        except KeyboardInterrupt:

            try:
                img_collector.shutdown()
            except:
                logging.debug("Cant close camera")

            wfile.close();
            GPIO.cleanup();
            logging.debug("Proceeding to Shutdown");
            subprocess.call("sudo shutdown -h now", shell = True);
            sys.exit(0);

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format = '(%(threadName)-10s) %(message)s',)
    main()
