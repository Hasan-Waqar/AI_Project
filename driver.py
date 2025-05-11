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
        
        self.manual_mode = False  # Start in manual mode
        self.manual_controls = {"steer": 0.0, "accel": 0.0, "gear": 1}  # Default manual controls
        
        # Load neural network models
        self.reg_model = RegressionModel(input_size=25)
        self.cls_model = ClassificationModel(input_size=25)
        try:
            self.reg_model.load_state_dict(torch.load("torcs_reg_controller.pth"))
            self.cls_model.load_state_dict(torch.load("torcs_cls_controller.pth"))
            self.reg_model.eval()
            self.cls_model.eval()
        except FileNotFoundError:
            print("Warning: Model files not found. Autonomous mode will fail.")
        
        # Gear mapping for classification
        self.gear_map = [-1, 0, 1, 2, 3, 4, 5, 6]
        
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
    
    def drive(self, msg):
        self.state.setFromMsg(msg)
        
        reg_actions = None
        gear_idx = None

        if self.manual_mode:
            self.control.setSteer(self.manual_controls["steer"])
            accel = self.manual_controls["accel"]
            self.control.setAccel(max(0, accel))  # Map positive to accel
            self.control.setBrake(max(0, -accel))  # Map negative to brake
            self.control.setGear(self.manual_controls["gear"])
        else:
            # Autonomous mode: Use neural networks
            # Prepare input (match preprocessing.py)
            track = self.state.getTrack()
            track_pos = self.state.getTrackPos()
            angle = self.state.getAngle()
            speed_x = self.state.getSpeedX()
            gear = self.state.getGear()
            rpm = self.state.getRpm()
            opponents = self.state.getOpponents()
            min_opponent = min(opponents)
            # Normalize inputs
            track_norm = [min(v/200, 1) for v in track]
            track_pos_norm = (track_pos + 1) / 2
            angle_norm = (angle + np.pi) / (2 * np.pi)
            speed_x_norm = min(speed_x/100, 1)
            gear_norm = (gear + 1) / 7
            rpm_norm = min(rpm/10000, 1)
            min_opponent_norm = min(min_opponent/200, 1)
            obs = np.array(track_norm + [track_pos_norm, angle_norm, speed_x_norm, gear_norm, rpm_norm, min_opponent_norm])
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            # Predict actions
            with torch.no_grad():
                reg_actions = self.reg_model(obs_tensor).numpy()
                cls_logits = self.cls_model(obs_tensor)
                gear_idx = torch.argmax(cls_logits).item()
                print(f"RPM: {rpm}, SpeedX: {speed_x}, cls_logits: {cls_logits.numpy()}, gear_idx: {gear_idx}, Pred_Gear: {self.gear_map[gear_idx]}")
            # Set controls
            self.control.setSteer(reg_actions[0])  # [-1, 1]
            self.control.setAccel(max(0, reg_actions[1]))  # [0, 1]
            self.control.setBrake(max(0, reg_actions[2]))  # [0, 1]
            self.control.setGear(self.gear_map[gear_idx])  # -1 to 6
        
        # Save data
        self.save_data(reg_actions, gear_idx)
        
        return self.control.toMsg()
    
    def steer(self):
        # Kept for reference but unused in autonomous mode
        angle = self.state.angle
        dist = self.state.trackPos
        self.control.setSteer((angle - dist * 0.5) / self.steer_lock)
    
    def gear(self):
        # Kept for reference but unused in autonomous mode
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        if self.prev_rpm is None:
            up = True
        else:
            up = (self.prev_rpm - rpm) < 0
        if up and rpm > 7000:
            gear += 1
        if not up and rpm < 3000:
            gear -= 1
        self.control.setGear(gear)
    
    def speed(self):
        # Kept for reference but unused in autonomous mode
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        if speed < self.max_speed:
            accel += 0.1
            accel = min(accel, 1.0)
        else:
            accel -= 0.1
            accel = max(accel, 0.0)
        self.control.setAccel(accel)
    
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
        
        with open("road3.csv", "a", newline="") as file:
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
    
    def listen_keyboard(self):
        """Listen for keyboard inputs and update manual controls."""
        def on_press(key):
            try:
                if key.char == 'w':
                    self.manual_controls["accel"] = 1.0  # Accelerate
                elif key.char == 's':
                    self.manual_controls["accel"] = -1.0  # Brake
                elif key.char == 'a':
                    self.manual_controls["steer"] = 0.5  # Steer left
                elif key.char == 'd':
                    self.manual_controls["steer"] = -0.5  # Steer right
                elif key.char == 'e':
                    self.manual_controls["gear"] += 1  # Gear up
                elif key.char == 'q':
                    self.manual_controls["gear"] -= 1  # Gear down
                elif key.char == 'm':  # Toggle manual/autonomous mode
                    self.enable_manual_mode(not self.manual_mode)
                    print(f"Mode: {'Manual' if self.manual_mode else 'Autonomous'}")
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in ['w', 's']:
                    self.manual_controls["accel"] = 0.0  # Stop acceleration
                if key.char in ['a', 'd']:
                    self.manual_controls["steer"] = 0.0  # Stop steering
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def onShutDown(self):
        pass
    
    def onRestart(self):
        pass