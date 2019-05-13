// Global Variables
int timer = 0;
int minInterval = 5;
int syncInterval = 1;
int flag_sync = 0;

int CurrentStatePin2 = LOW;
int PrevStatePin2 = LOW;
int CurrentStatePin3 = LOW;
int PrevStatePin3 = LOW;

void setup() {
  pinMode(3,INPUT);
  pinMode(2,INPUT);
  
  pinMode(13,OUTPUT);
  pinMode(12,OUTPUT);

  digitalWrite(13,HIGH); // this flips a relay energizing the RPi
  digitalWrite(12,LOW);

  Serial.begin(9600);
  Serial.println("Running...");
}

// timer1 interrupt service routine
ISR(TIMER1_COMPA_vect){
  // turn on the RPi, as the time interval has elapsed
  if(timer == 60*minInterval && flag_sync == 0){
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    timer = 1;
    Serial.println(timer);
  // turn on the RPi, as the sync interval has elapsed and the RPi wants to sync
  } else if (timer == 60*syncInterval && flag_sync == 1){
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    timer = 1;
    Serial.println(timer);
    flag_sync = 0;
    Serial.println("flag_sync off");
  // increment timer
  } else {
    timer += 1;
    Serial.println(timer);
  }
}


void pi_OFF(){
  // turn off RPi, as the signal pin has lifted saying it is shutdown
  delay(1000);
  digitalWrite(13,LOW);
  digitalWrite(12,HIGH);
  Serial.println("pin 2 lifted");
}

// pin3 interrup service routine
void timer_BEGIN(){
  // start the timer1, as the signal pin has been lifted saying the clocks are synchronized
  cli();
  TCCR1A = 0;// set entire TCCR1A register to 0
  TCCR1B = 0;// same for TCCR1B
  TCNT1  = 0;//initialize counter value to 0
  // set compare match register for 1hz increments
  OCR1A = 15624;// = (16*10^6) / (1*1024) - 1 (must be <65536)
  //OCR1A = 15614;// for Laser Module
  // Set CS12 and CS10 bits for 1024 prescaler
  TCCR1B |= (1 << CS12) | (1 << CS10);  
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  TCCR1B |= (1 << WGM12); 
  sei();
  timer = 0; // initialize or reset counter (first execution, or re-sync)
  Serial.println("pin 3 lifted");
}


void loop() {
  
  PrevStatePin2 = CurrentStatePin2;
  PrevStatePin3 = CurrentStatePin3;

  delay (10);
  
  CurrentStatePin2 = digitalRead(2);
  CurrentStatePin3 = digitalRead(3);

  if ((PrevStatePin2 == HIGH) && (CurrentStatePin2 == LOW)){
    pi_OFF();
  }
  if ((PrevStatePin3 == LOW) && (CurrentStatePin3 == HIGH)){
    timer_BEGIN();
    flag_sync = 1;
    Serial.println("flag_sync on");
  }
}
