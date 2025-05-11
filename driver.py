import msgParser
import carState
import carControl
import csv
import threading
from pynput import keyboard
import torch
import torch.nn as nn
import numpy as np

class RegressionModel(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Steer, Accel, Brake
        )
    
    def forward(self, x):
        return torch.tanh(self.net(x))

class ClassificationModel(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 8)  # 8 classes for Gear_output (-1, 0, 1, 2, 3, 4, 5, 6)
        )
    
    def forward(self, x):
        return self.net(x)

class Driver:
    '''
    A driver object for the SCRC
    '''
    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage
        
        self.parser = msgParser.MsgParser()
        self.state = carState.CarState()
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None
        self.stuck_counter = 0  # Count consecutive stuck frames
        
        self.manual_mode = True  # Start in manual mode
        self.manual_controls = {"steer": 0.0, "accel": 0.0, "gear": 1}  # Default manual controls
        
        # Load neural network models
        self.reg_model = RegressionModel(input_size=25)
        self.cls_model = ClassificationModel(input_size=25)
        try:
            self.reg_model.load_state_dict(torch.load("torcs_reg_controller.pth"))
            self.cls_model.load_state_dict(torch.load("torcs_cls_controller.pth"))
            self.reg_model.eval()
            self.cls_model.eval()
            print("Models loaded successfully.")
        except FileNotFoundError:
            print("Warning: Model files not found. Autonomous mode will fail.")
        
        # Gear mapping for classification
        self.gear_map = [-1, 0, 1, 2, 3, 4, 5, 6]
        
        # Recovery state variables
        self.recovery_state = "normal"  # States: normal, reversing, stopping, turning, accelerating
        self.recovery_timer = 0
        self.recovery_direction = 0  # -1 for left, 1 for right
        self.last_position = None
        self.last_speed = 0
        self.stuck_time = 0
        self.last_recovery_time = 0
        self.stuck_threshold = 3  # Number of consecutive frames to consider car stuck
        
        # Gear control
        self.gear_timer = 0
        self.last_gear_change = 0
        self.last_predicted_gear = 1
        self.gear_change_threshold = 5  # Minimum frames between gear changes
        
        # Start keyboard listener in a separate thread
        self.listener_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.listener_thread.start()

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for _ in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def is_car_stuck(self, speed_x, angle, track_pos, rpm):
        """Determine if the car is stuck based on various metrics."""
        # Car is stuck if it's moving very slowly and has high RPM (wheels spinning)
        # or if it's at a sharp angle off the track
        
        is_slow = abs(speed_x) < 3.0
        high_rpm = rpm > 5000
        bad_angle = abs(angle) > 0.8
        off_track = abs(track_pos) > 0.9
        
        # Increment stuck counter if conditions met
        if (is_slow and high_rpm) or (is_slow and bad_angle) or (is_slow and off_track):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        # Car is considered stuck if it's been stuck for several consecutive frames
        return self.stuck_counter > self.stuck_threshold
    
    def manage_recovery(self, angle, track_pos, speed_x, current_time):
        """Handle recovery from stuck situations using a state machine."""
        # Don't start a new recovery if one was completed recently
        if current_time - self.last_recovery_time < 5.0 and self.recovery_state == "normal":
            return False
            
        if self.recovery_state == "normal":
            # Start recovery process
            self.recovery_state = "reversing"
            self.recovery_timer = current_time
            self.recovery_direction = -np.sign(angle) if abs(angle) > 0.3 else -np.sign(track_pos)
            self.control.setGear(-1)  # Reverse gear
            print(f"Starting recovery: reversing at time {current_time}")
            return True
            
        elif self.recovery_state == "reversing":
            # Reverse for a short time
            time_in_state = current_time - self.recovery_timer
            
            # Apply reverse throttle and opposite steering
            self.control.setGear(-1)
            self.control.setAccel(0.7)
            self.control.setBrake(0.0)
            self.control.setSteer(self.recovery_direction * 0.5)
            
            # Transition to stopping state after sufficient reversing
            if (time_in_state > 2.0 and abs(speed_x) > 3.0) or time_in_state > 3.5:
                self.recovery_state = "stopping"
                self.recovery_timer = current_time
                print(f"Recovery: now stopping at time {current_time}")
            return True
            
        elif self.recovery_state == "stopping":
            # Come to a complete stop before changing direction
            time_in_state = current_time - self.recovery_timer
            
            # Apply brakes and neutral gear to stop
            self.control.setGear(0)
            self.control.setAccel(0.0)
            self.control.setBrake(1.0)
            self.control.setSteer(0.0)
            
            # Transition to turning state after car is stopped or timeout
            if (abs(speed_x) < 0.5) or time_in_state > 1.5:
                self.recovery_state = "turning"
                self.recovery_timer = current_time
                print(f"Recovery: now turning at time {current_time}")
            return True
            
        elif self.recovery_state == "turning":
            # Turn to face in a better direction
            time_in_state = current_time - self.recovery_timer
            
            # Set forward gear, full steering and moderate throttle
            self.control.setGear(1)
            self.control.setSteer(-self.recovery_direction * 0.8)  # Opposite of reverse direction
            self.control.setAccel(0.3)
            self.control.setBrake(0.0)
            
            # Transition to accelerating after short turning period
            if time_in_state > 1.5:
                self.recovery_state = "accelerating"
                self.recovery_timer = current_time
                print(f"Recovery: now accelerating at time {current_time}")
            return True
            
        elif self.recovery_state == "accelerating":
            # Accelerate to get back on track
            time_in_state = current_time - self.recovery_timer
            
            # Full throttle with steering towards track center
            self.control.setGear(1)
            self.control.setSteer(-np.sign(track_pos) * 0.3)  # Steer towards track center
            self.control.setAccel(0.8)
            self.control.setBrake(0.0)
            
            # End recovery after acceleration period or when back on track
            if time_in_state > 2.0 or (abs(track_pos) < 0.5 and abs(angle) < 0.3 and speed_x > 10):
                self.recovery_state = "normal"
                self.last_recovery_time = current_time
                print(f"Recovery: completed at time {current_time}")
                return False
            return True
            
        return False
    
    def manage_gears(self, current_gear, rpm, speed_x, predicted_gear, current_time):
        """Intelligent gear management logic."""
        # Don't change gears too frequently
        if current_time - self.last_gear_change < self.gear_change_threshold/10:
            return current_gear
            
        # Handle reverse gear specially
        if current_gear == -1:
            if speed_x > 5.0:  # Moving forward in reverse gear
                self.last_gear_change = current_time
                return 1
            return -1
            
        # Special case for neutral
        if current_gear == 0:
            self.last_gear_change = current_time
            return 1
            
        # Apply upshift logic: higher RPM thresholds for lower gears
        upshift_threshold = 8500 if current_gear == 1 else 8000
        if rpm > upshift_threshold and current_gear < 6:
            self.last_gear_change = current_time
            return current_gear + 1
            
        # Apply downshift logic: lower RPM thresholds for higher gears
        downshift_threshold = 3000
        if rpm < downshift_threshold and current_gear > 1:
            self.last_gear_change = current_time
            return current_gear - 1
            
        # Consider neural network's prediction if it's reasonable
        if predicted_gear > 0:  # Ignore reverse or neutral predictions
            gear_diff = predicted_gear - current_gear
            if abs(gear_diff) == 1:  # Only accept incremental changes
                # Validate prediction makes sense for current RPM
                if (gear_diff > 0 and rpm > 7000) or (gear_diff < 0 and rpm < 4000):
                    self.last_gear_change = current_time
                    return predicted_gear
                    
        # Default: keep current gear
        return current_gear
        
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        reg_actions = None
        gear_idx = None
        current_time = self.state.getCurLapTime()

        if self.manual_mode:
            self.control.setSteer(self.manual_controls["steer"])
            accel = self.manual_controls["accel"]
            self.control.setAccel(max(0, accel))  # Map positive to accel
            self.control.setBrake(max(0, -accel))  # Map negative to brake
            self.control.setGear(self.manual_controls["gear"])
        else:
            # Autonomous mode: Get car state
            track = self.state.getTrack()
            track_pos = self.state.getTrackPos()
            angle = self.state.getAngle()
            speed_x = self.state.getSpeedX()
            current_gear = self.state.getGear()
            rpm = self.state.getRpm()
            opponents = self.state.getOpponents()
            min_opponent = min(opponents)
            
            # Check if car is stuck
            car_stuck = self.is_car_stuck(speed_x, angle, track_pos, rpm)
            
            # Handle recovery if car is stuck
            if car_stuck:
                if self.manage_recovery(angle, track_pos, speed_x, current_time):
                    # If in recovery mode, don't use neural network outputs
                    reg_actions = [self.control.getSteer(), self.control.getAccel(), self.control.getBrake()]
                    gear_idx = self.gear_map.index(self.control.getGear())
                    self.save_data(reg_actions, gear_idx)
                    return self.control.toMsg()
            elif self.recovery_state != "normal":
                # Continue recovery process if it's not completed
                if self.manage_recovery(angle, track_pos, speed_x, current_time):
                    reg_actions = [self.control.getSteer(), self.control.getAccel(), self.control.getBrake()]
                    gear_idx = self.gear_map.index(self.control.getGear())
                    self.save_data(reg_actions, gear_idx)
                    return self.control.toMsg()
            
            # Normal driving: prepare input for neural networks
            # Normalize inputs
            track_norm = [min(v/200, 1) for v in track]
            track_pos_norm = (track_pos + 1) / 2
            angle_norm = (angle + np.pi) / (2 * np.pi)
            speed_x_norm = min(speed_x/100, 1)
            gear_norm = (current_gear + 1) / 7
            rpm_norm = min(rpm/10000, 1)
            min_opponent_norm = min(min_opponent/200, 1)
            
            obs = np.array(track_norm + [track_pos_norm, angle_norm, speed_x_norm, gear_norm, rpm_norm, min_opponent_norm])
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            # Predict actions
            with torch.no_grad():
                reg_actions = self.reg_model(obs_tensor).numpy()
                cls_logits = self.cls_model(obs_tensor)
                gear_idx = torch.argmax(cls_logits).item()
                predicted_gear = self.gear_map[gear_idx]
                
                # Debug info
                if current_time % 1 < 0.1:  # Print every ~1 second
                    print(f"T: {current_time:.1f}, RPM: {rpm:.0f}, Speed: {speed_x:.1f}, "
                          f"Angle: {angle:.2f}, TrackPos: {track_pos:.2f}, "
                          f"Gear: {current_gear}, PredGear: {predicted_gear}")
            
            # Set steering with track position awareness
            steer_correction = -track_pos * 0.5  # Steer towards center
            steer_value = reg_actions[0] + steer_correction
            
            # Apply stronger steering when off track
            if abs(track_pos) > 0.7:
                steer_value = np.clip(steer_value * 1.5, -1, 1)
                
            # Reduce speed on sharp corners
            closest_track_sensors = min(track[8:11])  # Front-facing sensors
            if closest_track_sensors < 50 and speed_x > 50:
                reg_actions[1] *= 0.5  # Reduce accelerator
                reg_actions[2] = max(reg_actions[2], 0.1)  # Apply some brake
            
            # Set controls
            self.control.setSteer(np.clip(steer_value, -1, 1))
            self.control.setAccel(max(0, reg_actions[1]))
            self.control.setBrake(max(0, reg_actions[2]))
            
            # Apply intelligent gear management
            new_gear = self.manage_gears(current_gear, rpm, speed_x, predicted_gear, current_time)
            self.control.setGear(new_gear)
            
            # Special case for hill starts or when car is almost stopped
            if speed_x < 5 and current_gear <= 1 and self.control.getAccel() > 0.5:
                self.control.setGear(1)  # Force first gear for starting
                self.control.setClutch(0.5)  # Apply clutch to prevent stalling
                # Gradually release clutch
                if self.state.getDistRaced() % 3 < 0.1:
                    self.control.setClutch(0)
        
        return self.control.toMsg()
    
    def save_data(self, reg_actions=None, gear_idx=None):
        """Save all available car state and control data to a CSV file."""
        data = [
            # Positional and Orientation Data
            self.state.getAngle(),
            self.state.getTrackPos(),
            self.state.getDistFromStart(),
            self.state.getDistRaced(),
            self.state.getZ(),
            # Speed Components
            self.state.getSpeedX(),
            self.state.getSpeedY(),
            self.state.getSpeedZ(),
            # Vehicle Status
            self.state.getGear(),
            self.state.getRpm(),
            self.state.getFuel(),
            self.state.getDamage(),
            self.state.getRacePos(),
            self.state.getCurLapTime(),
            self.state.getLastLapTime(),
            # Additional Sensors
            self.state.getFocus(),
            self.state.getTrack(),
            self.state.getOpponents(),
            self.state.getWheelSpinVel(),
            # Car Control
            self.control.getSteer(),
            self.control.getAccel(),
            self.control.getBrake(),
            self.control.getClutch(),
            self.control.getFocus(),
            self.control.getGear(),
            self.control.getMeta(),
            # Predicted actions (if autonomous mode)
            reg_actions[0] if reg_actions is not None else None,  # Pred_Steer
            reg_actions[1] if reg_actions is not None else None,  # Pred_Accel
            reg_actions[2] if reg_actions is not None else None,  # Pred_Brake
            self.gear_map[gear_idx] if gear_idx is not None else None  # Pred_Gear
        ]
        
        with open("Lancer_Round2.csv", "a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                header = [
                    "Angle", "TrackPos", "DistFromStart", "DistRaced", "Z",
                    "SpeedX", "SpeedY", "SpeedZ",
                    "Gear", "RPM", "Fuel", "Damage", "RacePos",
                    "CurLapTime", "LastLapTime",
                    "Focus", "Track", "Opponents", "WheelSpinVel",
                    "Steer", "Accel", "Brake", "Clutch", "ControlFocus",
                    "Gear_output", "Meta",
                    "Pred_Steer", "Pred_Accel", "Pred_Brake", "Pred_Gear"
                ]
                writer.writerow(header)
            writer.writerow(data)

    def enable_manual_mode(self, enable):
        """Enable or disable manual mode."""
        self.manual_mode = enable
        self.recovery_state = "normal"  # Reset recovery state when switching modes
        print(f"Mode switched to: {'Manual' if enable else 'Autonomous'}")
    
    def listen_keyboard(self):
        """Listen for keyboard inputs and update manual controls."""
        def on_press(key):
            try:
                if key.char == 'u':
                    self.manual_controls["accel"] = 1.0  # Accelerate
                elif key.char == 'j':
                    self.manual_controls["accel"] = -1.0  # Brake
                elif key.char == 'h':
                    self.manual_controls["steer"] = 0.5  # Steer left
                elif key.char == 'k':
                    self.manual_controls["steer"] = -0.5  # Steer right
                elif key.char == 'e':
                    self.manual_controls["gear"] += 1  # Gear up
                    self.manual_controls["gear"] = min(self.manual_controls["gear"], 6)
                elif key.char == 'w':
                    self.manual_controls["gear"] -= 1  # Gear down
                    self.manual_controls["gear"] = max(self.manual_controls["gear"], -1)
                elif key.char == 'm':  # Toggle manual/autonomous mode
                    self.enable_manual_mode(not self.manual_mode)
                elif key.char == 'r':  # Force recovery mode
                    if not self.manual_mode:
                        self.recovery_state = "normal"  # Reset state machine
                        self.is_car_stuck(0, 1.0, 1.0, 6000)  # Force stuck to true
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in ['u', 'j']:
                    self.manual_controls["accel"] = 0.0  # Stop acceleration
                if key.char in ['h', 'k']:
                    self.manual_controls["steer"] = 0.0  # Stop steering
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def onShutDown(self):
        pass
    
    def onRestart(self):
        self.recovery_state = "normal"
        self.stuck_counter = 0