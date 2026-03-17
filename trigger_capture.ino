int IR_pin = 5;
int sensor_value = 0;
int photo_taken = false;


void setup() {
  // Set all the motor control pins to outputs
  pinMode(IR_pin,INPUT);
  Serial.begin(9600);
}

void loop() {
  // photo capture
  sensor_value = digitalRead(IR_pin);
  if (sensor_value == 0 && photo_taken == false){
    Serial.println("C");
    delay(50);
    photo_taken = true;
    
  }
  if (sensor_value == 1){
    photo_taken = false;
  }

}
  


