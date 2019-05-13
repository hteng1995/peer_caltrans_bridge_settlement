// Global Variables
unsigned long max_time = 480000;
unsigned long cur_time = 0;
unsigned long start_time = 0;
int rpi_on = 0;
int fona_charged = 0;
int state = 0;

int CurrentStatePin2 = LOW;
int PrevStatePin2 = LOW;
int CurrentStatePin3 = LOW;
int PrevStatePin3 = LOW;
int CurrentStatePin4 = LOW;
int PrevStatePin4 = LOW;

void setup() {
  Serial.begin(9600);
  pinMode(4,INPUT);//BATTERY INDICATOR
  pinMode(3,INPUT);//RING INDICATOR INPUT
  pinMode(2,INPUT);//RPI POWER INDICATOR
  
  pinMode(12,OUTPUT);
  pinMode(11,OUTPUT); //12 HI +  11 LOW = PWR ON, 12 LO + 11 HI = PWR OFF

  pinMode(8,OUTPUT);
  pinMode(7,OUTPUT); //8 HI + 7 LOW = PWR ON, 8 LO + 7 HI = PWR OFF

  digitalWrite(12,LOW); //Initialize RPI off
  digitalWrite(11,HIGH);

  digitalWrite(8,HIGH); //Initialize FONA on
  digitalWrite(7,LOW); 
}

void pi_ON() {
  digitalWrite(11,LOW);
  digitalWrite(12,HIGH);
  rpi_on = 1;
  fona_charged = 0;
}

void pi_OFF() {
  delay(1000*10); //wait for RPI to properly shutdown before cutting power
  digitalWrite(11,HIGH);
  digitalWrite(12,LOW);
  rpi_on = 0;
}

void fona_ON() {
  digitalWrite(7,LOW);
  digitalWrite(8,HIGH);
}

void fona_OFF() {
  digitalWrite(7,HIGH);
  digitalWrite(8,LOW);
}


void loop() {
  
  Serial.println(cur_time-start_time);
      
  PrevStatePin2 = CurrentStatePin2;
  PrevStatePin3 = CurrentStatePin3;
  PrevStatePin4 = CurrentStatePin4;

  delay(10);
  
  CurrentStatePin2 = digitalRead(2);
  CurrentStatePin3 = digitalRead(3);
  CurrentStatePin4 = digitalRead(4);

  if ((PrevStatePin2 == HIGH) && (CurrentStatePin2 == LOW) && (rpi_on == 1)){
    pi_OFF();
    start_time = 0;
    cur_time = 0;
    if (fona_charged == 1){
      fona_OFF();
    }else {
      fona_ON();
    }
  }
  if ((PrevStatePin3 == HIGH) && (CurrentStatePin3 == LOW) && (rpi_on == 0)){
    fona_OFF();
    delay(1000);
    pi_ON();
    start_time = millis();
  }
  if ((rpi_on == 1) && (PrevStatePin4 == LOW) && (CurrentStatePin4 == HIGH)){
    fona_charged = 1;
  }
  if (rpi_on == 1){
    cur_time = millis();
  } 
  if (((cur_time - start_time) > max_time) && (rpi_on == 1)){
    Serial.println(start_time);
    Serial.println(cur_time);
    pi_OFF();
    start_time = 0;
    cur_time = 0;
    if (fona_charged == 1){
      fona_OFF();
    } else{
      fona_ON();
    }
  }
}


