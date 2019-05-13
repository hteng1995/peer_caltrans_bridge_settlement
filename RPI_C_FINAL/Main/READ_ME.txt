Deployable version of Camera Module (Raspberry Pi) Code: 4/23/2019

Prepare Raspberry Pi 3B+ by flashing most recent version of Raspbian onto a SD card. This can be done through "NOOBS" or by downloading a software called "Etcher," and flashing the .zip file containing Raspbian (as downloaded from the RPI website) onto it. The latter is quicker.

Insert SD card into Pi, and boot with keyboard, mouse, and HDMI inserted. Upon boot, initialize the keyboard to US and connect to the internet using the desktop tools. You are ready to install required packages. 

Update the Raspbian firmware using: "sudo apt-get update" and "sudo apt-get disc-upgrade"
Source: "https://www.raspberrypi.org/documentation/raspbian/updating.md"

Install: bluepy (go to bluepy GitHub page for installation steps), opencv (sudo apt-get install python-opencv), scipy, picamera

Using the command: "sudo rasps-config" you may turn on SSH, SPI, I2C, Camera, and Serial (hardware, not the login option)

Save contents: ble.py, fona.py, RPI_C.py, SETTINGS.py, Read_Accel.py in a "Main" folder on Desktop
Save contents: fona-chat, fona-config in a folder /etc/ppp/peers/ (use "sudo su cp ---")
Save other folders: communication_module, cv_module, accel_module on Desktop

Errors.txt enumerates error codes

For Server: 
Go into the SETTINGS.py file, and edit the bridge name
You may also change the "buffer" variable, which controls how many images can be saved on the Pi's local memory before FIFO erasure

for FONA: 
edit /boot/config.txt by adding "enable_uart=1" at end of file
install: "sudo apt-get install ppp"
Source: "https://github.com/initialstate/fona-raspberry-pi-3"

for all other files: 
install: bluepy (go to GitHub page for installation steps), opencv (sudo apt-get install python-opencv), scipy, picamera

After ensuring all components function correctly (do a few test runs with the laser module), you can edit the rc.local file to auto-run the main program from boot
Source: "https://www.raspberrypi.org/documentation/linux/usage/rc-local.md"