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
        # define a simple feedforward network for regression
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # steer, accel, brake
        )
    
    def forward(self, x):
        # output is passed through tanh for normalization
        return torch.tanh(self.net(x))

class ClassificationModel(nn.Module):
    def __init__(self, input_size=25):
        super().__init__()
        # define a simple feedforward network for classification
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 8)  # 8 classes for gear_output (-1, 0, 1, 2, 3, 4, 5, 6)
        )
    
    def forward(self, x):
        # output is logits for classification
        return self.net(x)

class Driver:
    '''
    a driver object for the scrc
    '''
    def __init__(self, stage):
        '''constructor'''
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
        self.stuck_counter = 0  # count consecutive stuck frames
        
        self.manual_mode = True  # start in manual mode
        self.manual_controls = {"steer": 0.0, "accel": 0.0, "gear": 1}  # default manual controls
        
        # load neural network models
        self.reg_model = RegressionModel(input_size=25)
        self.cls_model = ClassificationModel(input_size=25)
        try:
            self.reg_model.load_state_dict(torch.load("torcs_reg_controller.pth"))
            self.cls_model.load_state_dict(torch.load("torcs_cls_controller.pth"))
            self.reg_model.eval()
            self.cls_model.eval()
            print("models loaded successfully.")
        except FileNotFoundError:
            print("warning: model files not found. autonomous mode will fail.")
        
        # gear mapping for classification
        self.gear_map = [-1, 0, 1, 2, 3, 4, 5, 6]
        
        # recovery state variables
        self.recovery_state = "normal"  # states: normal, reversing, stopping, turning, accelerating
        self.recovery_timer = 0
        self.recovery_direction = 0  # -1 for left, 1 for right
        self.last_position = None
        self.last_speed = 0
        self.stuck_time = 0
        self.last_recovery_time = 0
        self.stuck_threshold = 3  # number of consecutive frames to consider car stuck
        
        # gear control
        self.gear_timer = 0
        self.last_gear_change = 0
        self.last_predicted_gear = 1
        self.gear_change_threshold = 5  # minimum frames between gear changes
        
        # start keyboard listener in a separate thread
        self.listener_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.listener_thread.start()

    def init(self):
        '''return init string with rangefinder angles'''
        self.angles = [0 for _ in range(19)]
        
        # set angles for rangefinder sensors
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def is_car_stuck(self, speed_x, angle, track_pos, rpm):
        """determine if the car is stuck based on various metrics."""
        # car is stuck if it's moving very slowly and has high rpm (wheels spinning)
        # or if it's at a sharp angle off the track
        
        is_slow = abs(speed_x) < 3.0
        high_rpm = rpm > 5000
        bad_angle = abs(angle) > 0.8
        off_track = abs(track_pos) > 0.9
        
        # increment stuck counter if conditions met
        if (is_slow and high_rpm) or (is_slow and bad_angle) or (is_slow and off_track):
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        # car is considered stuck if it's been stuck for several consecutive frames
        return self.stuck_counter > self.stuck_threshold
    
    def manage_recovery(self, angle, track_pos, speed_x, current_time):
        """handle recovery from stuck situations using a state machine."""
        # don't start a new recovery if one was completed recently
        if current_time - self.last_recovery_time < 5.0 and self.recovery_state == "normal":
            return False
            
        if self.recovery_state == "normal":
            # start recovery process
            self.recovery_state = "reversing"
            self.recovery_timer = current_time
            self.recovery_direction = -np.sign(angle) if abs(angle) > 0.3 else -np.sign(track_pos)
            self.control.setGear(-1)  # reverse gear
            print(f"starting recovery: reversing at time {current_time}")
            return True
            
        elif self.recovery_state == "reversing":
            # reverse for a short time
            time_in_state = current_time - self.recovery_timer
            
            # apply reverse throttle and opposite steering
            self.control.setGear(-1)
            self.control.setAccel(0.7)
            self.control.setBrake(0.0)
            self.control.setSteer(self.recovery_direction * 0.5)
            
            # transition to stopping state after sufficient reversing
            if (time_in_state > 2.0 and abs(speed_x) > 3.0) or time_in_state > 3.5:
                self.recovery_state = "stopping"
                self.recovery_timer = current_time
                print(f"recovery: now stopping at time {current_time}")
            return True
            
        elif self.recovery_state == "stopping":
            # come to a complete stop before changing direction
            time_in_state = current_time - self.recovery_timer
            
            # apply brakes and neutral gear to stop
            self.control.setGear(0)
            self.control.setAccel(0.0)
            self.control.setBrake(1.0)
            self.control.setSteer(0.0)
            
            # transition to turning state after car is stopped or timeout
            if (abs(speed_x) < 0.5) or time_in_state > 1.5:
                self.recovery_state = "turning"
                self.recovery_timer = current_time
                print(f"recovery: now turning at time {current_time}")
            return True
            
        elif self.recovery_state == "turning":
            # turn to face in a better direction
            time_in_state = current_time - self.recovery_timer
            
            # set forward gear, full steering and moderate throttle
            self.control.setGear(1)
            self.control.setSteer(-self.recovery_direction * 0.8)  # opposite of reverse direction
            self.control.setAccel(0.3)
            self.control.setBrake(0.0)
            
            # transition to accelerating after short turning period
            if time_in_state > 1.5:
                self.recovery_state = "accelerating"
                self.recovery_timer = current_time
                print(f"recovery: now accelerating at time {current_time}")
            return True
            
        elif self.recovery_state == "accelerating":
            # accelerate to get back on track
            time_in_state = current_time - self.recovery_timer
            
            # full throttle with steering towards track center
            self.control.setGear(1)
            self.control.setSteer(-np.sign(track_pos) * 0.3)  # steer towards track center
            self.control.setAccel(0.8)
            self.control.setBrake(0.0)
            
            # end recovery after acceleration period or when back on track
            if time_in_state > 2.0 or (abs(track_pos) < 0.5 and abs(angle) < 0.3 and speed_x > 10):
                self.recovery_state = "normal"
                self.last_recovery_time = current_time
                print(f"recovery: completed at time {current_time}")
                return False
            return True
            
        return False
    
    def manage_gears(self, current_gear, rpm, speed_x, predicted_gear, current_time):
        """intelligent gear management logic."""
        # don't change gears too frequently
        if current_time - self.last_gear_change < self.gear_change_threshold/10:
            return current_gear
            
        # handle reverse gear specially
        if current_gear == -1:
            if speed_x > 5.0:  # moving forward in reverse gear
                self.last_gear_change = current_time
                return 1
            return -1
            
        # special case for neutral
        if current_gear == 0:
            self.last_gear_change = current_time
            return 1
            
        # apply upshift logic: higher rpm thresholds for lower gears
        upshift_threshold = 8500 if current_gear == 1 else 8000
        if rpm > upshift_threshold and current_gear < 6:
            self.last_gear_change = current_time
            return current_gear + 1
            
        # apply downshift logic: lower rpm thresholds for higher gears
        downshift_threshold = 3000
        if rpm < downshift_threshold and current_gear > 1:
            self.last_gear_change = current_time
            return current_gear - 1
            
        # consider neural network's prediction if it's reasonable
        if predicted_gear > 0:  # ignore reverse or neutral predictions
            gear_diff = predicted_gear - current_gear
            if abs(gear_diff) == 1:  # only accept incremental changes
                # validate prediction makes sense for current rpm
                if (gear_diff > 0 and rpm > 7000) or (gear_diff < 0 and rpm < 4000):
                    self.last_gear_change = current_time
                    return predicted_gear
                    
        # default: keep current gear
        return current_gear


    # def manage_gears(self, current_gear, rpm, speed_x, predicted_gear, current_time):
    #     """ML-based intelligent gear management logic."""
    #     # don't change gears too frequently
    #     if current_time - self.last_gear_change < self.gear_change_threshold / 10:
    #         return current_gear

    #     # special case: reverse gear handling
    #     if current_gear == -1:
    #         if speed_x > 5.0:
    #             self.last_gear_change = current_time
    #             return 1
    #         return -1

    #     # prepare features and predict using the trained ML model
    #     features = [[current_gear, rpm, speed_x]]
    #     ml_predicted_gear = int(self.cls_model.predict(features)[0])

    #     # smooth gear changes: allow only +/-1 difference
    #     if ml_predicted_gear > 0:  # ignore reverse/neutral predictions
    #         gear_diff = ml_predicted_gear - current_gear
    #         if abs(gear_diff) == 1:
    #             self.last_gear_change = current_time
    #             return ml_predicted_gear

    #     # fallback to current gear
    #     return current_gear
  
    def drive(self, msg):
        # update car state from message
        self.state.setFromMsg(msg)
        
        reg_actions = None
        gear_idx = None
        current_time = self.state.getCurLapTime()

        if self.manual_mode:
            # manual mode: set controls from manual_controls
            self.control.setSteer(self.manual_controls["steer"])
            accel = self.manual_controls["accel"]
            self.control.setAccel(max(0, accel))  # map positive to accel
            self.control.setBrake(max(0, -accel))  # map negative to brake
            self.control.setGear(self.manual_controls["gear"])
        else:
            # autonomous mode: get car state
            track = self.state.getTrack()
            track_pos = self.state.getTrackPos()
            angle = self.state.getAngle()
            speed_x = self.state.getSpeedX()
            current_gear = self.state.getGear()
            rpm = self.state.getRpm()
            opponents = self.state.getOpponents()
            min_opponent = min(opponents)
            
            # check if car is stuck
            car_stuck = self.is_car_stuck(speed_x, angle, track_pos, rpm)
            
            # handle recovery if car is stuck
            if car_stuck:
                if self.manage_recovery(angle, track_pos, speed_x, current_time):
                    # if in recovery mode, don't use neural network outputs
                    reg_actions = [self.control.getSteer(), self.control.getAccel(), self.control.getBrake()]
                    gear_idx = self.gear_map.index(self.control.getGear())
                    self.save_data(reg_actions, gear_idx)
                    return self.control.toMsg()
            elif self.recovery_state != "normal":
                # continue recovery process if it's not completed
                if self.manage_recovery(angle, track_pos, speed_x, current_time):
                    reg_actions = [self.control.getSteer(), self.control.getAccel(), self.control.getBrake()]
                    gear_idx = self.gear_map.index(self.control.getGear())
                    self.save_data(reg_actions, gear_idx)
                    return self.control.toMsg()
            
            # normal driving: prepare input for neural networks
            # normalize inputs
            track_norm = [min(v/200, 1) for v in track]
            track_pos_norm = (track_pos + 1) / 2
            angle_norm = (angle + np.pi) / (2 * np.pi)
            speed_x_norm = min(speed_x/100, 1)
            gear_norm = (current_gear + 1) / 7
            rpm_norm = min(rpm/10000, 1)
            min_opponent_norm = min(min_opponent/200, 1)
            
            obs = np.array(track_norm + [track_pos_norm, angle_norm, speed_x_norm, gear_norm, rpm_norm, min_opponent_norm])
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            # predict actions
            with torch.no_grad():
                reg_actions = self.reg_model(obs_tensor).numpy()
                cls_logits = self.cls_model(obs_tensor)
                gear_idx = torch.argmax(cls_logits).item()
                predicted_gear = self.gear_map[gear_idx]
                
                # debug info
                if current_time % 1 < 0.1:  # print every ~1 second
                    print(f"t: {current_time:.1f}, rpm: {rpm:.0f}, speed: {speed_x:.1f}, "
                          f"angle: {angle:.2f}, trackpos: {track_pos:.2f}, "
                          f"gear: {current_gear}, predgear: {predicted_gear}")
            
            # set steering with track position awareness
            steer_correction = -track_pos * 0.5  # steer towards center
            steer_value = reg_actions[0] + steer_correction
            
            # apply stronger steering when off track
            if abs(track_pos) > 0.7:
                steer_value = np.clip(steer_value * 1.5, -1, 1)
                
            # reduce speed on sharp corners
            closest_track_sensors = min(track[8:11])  # front-facing sensors
            if closest_track_sensors < 50 and speed_x > 50:
                reg_actions[1] *= 0.5  # reduce accelerator
                reg_actions[2] = max(reg_actions[2], 0.1)  # apply some brake
            
            # set controls
            self.control.setSteer(np.clip(steer_value, -1, 1))
            self.control.setAccel(max(0, reg_actions[1]))
            self.control.setBrake(max(0, reg_actions[2]))
            
            # apply intelligent gear management
            new_gear = self.manage_gears(current_gear, rpm, speed_x, predicted_gear, current_time)
            self.control.setGear(new_gear)
            
            # special case for hill starts or when car is almost stopped
            if speed_x < 5 and current_gear <= 1 and self.control.getAccel() > 0.5:
                self.control.setGear(1)  # force first gear for starting
                self.control.setClutch(0.5)  # apply clutch to prevent stalling
                # gradually release clutch
                if self.state.getDistRaced() % 3 < 0.1:
                    self.control.setClutch(0)
        
        return self.control.toMsg()
    
    def save_data(self, reg_actions=None, gear_idx=None):
        """save all available car state and control data to a csv file."""
        data = [
            # positional and orientation data
            self.state.getAngle(),
            self.state.getTrackPos(),
            self.state.getDistFromStart(),
            self.state.getDistRaced(),
            self.state.getZ(),
            # speed components
            self.state.getSpeedX(),
            self.state.getSpeedY(),
            self.state.getSpeedZ(),
            # vehicle status
            self.state.getGear(),
            self.state.getRpm(),
            self.state.getFuel(),
            self.state.getDamage(),
            self.state.getRacePos(),
            self.state.getCurLapTime(),
            self.state.getLastLapTime(),
            # additional sensors
            self.state.getFocus(),
            self.state.getTrack(),
            self.state.getOpponents(),
            self.state.getWheelSpinVel(),
            # car control
            self.control.getSteer(),
            self.control.getAccel(),
            self.control.getBrake(),
            self.control.getClutch(),
            self.control.getFocus(),
            self.control.getGear(),
            self.control.getMeta(),
            # predicted actions (if autonomous mode)
            reg_actions[0] if reg_actions is not None else None,  # pred_steer
            reg_actions[1] if reg_actions is not None else None,  # pred_accel
            reg_actions[2] if reg_actions is not None else None,  # pred_brake
            self.gear_map[gear_idx] if gear_idx is not None else None  # pred_gear
        ]
        
        with open("Lancer_Round2.csv", "a", newline="") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                header = [
                    "angle", "trackpos", "distfromstart", "distraced", "z",
                    "speedx", "speedy", "speedz",
                    "gear", "rpm", "fuel", "damage", "racepos",
                    "curlaptime", "lastlaptime",
                    "focus", "track", "opponents", "wheelspinvel",
                    "steer", "accel", "brake", "clutch", "controlfocus",
                    "gear_output", "meta",
                    "pred_steer", "pred_accel", "pred_brake", "pred_gear"
                ]
                writer.writerow(header)
            writer.writerow(data)

    def enable_manual_mode(self, enable):
        """enable or disable manual mode."""
        self.manual_mode = enable
        self.recovery_state = "normal"  # reset recovery state when switching modes
        print(f"mode switched to: {'manual' if enable else 'autonomous'}")
    
    def listen_keyboard(self):
        """listen for keyboard inputs and update manual controls."""
        def on_press(key):
            try:
                if key.char == 'u':
                    self.manual_controls["accel"] = 1.0  # accelerate
                elif key.char == 'j':
                    self.manual_controls["accel"] = -1.0  # brake
                elif key.char == 'h':
                    self.manual_controls["steer"] = 0.5  # steer left
                elif key.char == 'k':
                    self.manual_controls["steer"] = -0.5  # steer right
                elif key.char == 'e':
                    self.manual_controls["gear"] += 1  # gear up
                    self.manual_controls["gear"] = min(self.manual_controls["gear"], 6)
                elif key.char == 'w':
                    self.manual_controls["gear"] -= 1  # gear down
                    self.manual_controls["gear"] = max(self.manual_controls["gear"], -1)
                elif key.char == 'm':  # toggle manual/autonomous mode
                    self.enable_manual_mode(not self.manual_mode)
                elif key.char == 'r':  # force recovery mode
                    if not self.manual_mode:
                        self.recovery_state = "normal"  # reset state machine
                        self.is_car_stuck(0, 1.0, 1.0, 6000)  # force stuck to true
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key.char in ['u', 'j']:
                    self.manual_controls["accel"] = 0.0  # stop acceleration
                if key.char in ['h', 'k']:
                    self.manual_controls["steer"] = 0.0  # stop steering
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def onShutDown(self):
        # shutdown hook (empty)
        pass
    
    def onRestart(self):
        # restart hook: reset recovery state and stuck counter
        self.recovery_state = "normal"
        self.stuck_counter = 0