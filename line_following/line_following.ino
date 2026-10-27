/*
 * EmberBot — line-following controller (Arduino Uno)
 * ----------------------------------------------------
 * Drives EmberBot along a black line using three digital IR sensors
 * (right, middle, left) and an L298 dual motor driver.
 *
 * Sensor logic (1 = sensor is over the black line, 0 = over white):
 *   middle only on line      -> drive forward
 *   right side on line       -> turn right
 *   left side on line        -> turn left
 *   all three on line        -> stop (junction / end marker)
 *   none on line             -> stop (line lost)
 *
 * After every decision the loop calls Stop() and waits 50 ms, so the
 * robot moves in short pulses. This acts as a crude speed limiter, since
 * the L298 enable pins are not PWM-controlled in this build.
 *
 * Wiring
 *   IR sensors : right = D8, middle = D9, left = D10
 *   L298 inputs: IN1 = D6, IN2 = D7 (right motor), IN3 = D2, IN4 = D3 (left motor)
 *
 * Note: transcribed from the team report screenshot; logic unchanged.
 */

// ---- L298 motor driver input pins ----
#define in1 6   // right motor, backward
#define in2 7   // right motor, forward
#define in3 2   // left motor, backward
#define in4 3   // left motor, forward

void forward();
void turnLeft();
void turnRight();
void Stop();

// ---- IR sensor pins ----
int sensor0 = 8;   // right sensor
int sensor1 = 9;   // middle sensor
int sensor2 = 10;  // left sensor

// ---- latest sensor readings ----
int value0 = 0;    // right
int value1 = 0;    // middle
int value2 = 0;    // left

void setup() {
  pinMode(sensor0, INPUT);
  pinMode(sensor1, INPUT);
  pinMode(sensor2, INPUT);
  // NOTE: the original code never sets the motor pins (in1–in4) as OUTPUT.
  // Adding pinMode(inX, OUTPUT) for each would make the motor drive more reliable.
}

void loop() {
  // Read all three sensors
  value0 = digitalRead(sensor0);  // right
  value1 = digitalRead(sensor1);  // middle
  value2 = digitalRead(sensor2);  // left

  // Decide which way to move
  if (value0 == 0 && value1 == 1 && value2 == 0) { forward();   }  // centred on the line
  if (value0 == 1 && value2 == 0)                { turnRight(); }  // line has drifted right
  if (value0 == 0 && value2 == 1)                { turnLeft();  }  // line has drifted left
  if (value0 == 1 && value1 == 1 && value2 == 1) { Stop();      }  // all on black: junction
  if (value0 == 0 && value1 == 0 && value2 == 0) { Stop();      }  // all on white: line lost

  // Pause between pulses (limits speed so the sensors can keep up)
  Stop();
  delay(50);
}

// Both motors forward
void forward() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  delay(20);
}

// Right motor stopped, left motor forward -> pivot right
void turnRight() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
  delay(30);
}

// Left motor stopped, right motor forward -> pivot left
void turnLeft() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  delay(30);
}

// All motor inputs low -> both motors stop
void Stop() {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}
