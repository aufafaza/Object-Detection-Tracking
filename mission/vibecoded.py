#!/usr/bin/env python3

from pymavlink import mavutil
import time
import math
import threading

CONNECTION_STRING = 'udp:127.0.0.1:14550'
TAKEOFF_ALT = 20
TARGET_LAT = -6.123456
TARGET_LON = 106.123456
LOITER_RADIUS = 50

class DroneController:
    def __init__(self, connection_string):
        print(f"Connecting to {connection_string}...")
        self.master = mavutil.mavlink_connection(connection_string)
        
        self.master.wait_heartbeat()
        print(f"Heartbeat from system {self.master.target_system} component {self.master.target_component}")
        
        self.monitoring = False
        self.monitor_thread = None
        
    def arm(self):
        print("Arming motors...")
        self.master.arducopter_arm()
        self.master.motors_armed_wait()
        print("✓ Motors armed!")
        
    def disarm(self):
        print("Disarming motors...")
        self.master.arducopter_disarm()
        self.master.motors_disarmed_wait()
        print("✓ Motors disarmed!")
        
    def set_mode(self, mode):
        mode_mapping = {
            'STABILIZE': 0,
            'ACRO': 1,
            'ALT_HOLD': 2,
            'AUTO': 3,
            'GUIDED': 4,
            'LOITER': 5,
            'RTL': 6,
            'CIRCLE': 7,
            'LAND': 9,
            'DRIFT': 11,
            'SPORT': 13,
            'FLIP': 14,
            'AUTOTUNE': 15,
            'POSHOLD': 16,
            'BRAKE': 17,
        }
        
        if mode not in mode_mapping:
            print(f"Mode {mode} not recognized!")
            return False
            
        mode_id = mode_mapping[mode]
        
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        
        print(f"✓ Mode set to {mode}")
        return True
    
    def set_guided_options(self, options):
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'GUIDED_OPTIONS',
            options,
            mavutil.mavlink.MAV_PARAM_TYPE_INT32
        )
        print(f"✓ GUIDED_OPTIONS set to {options}")
        time.sleep(0.5)
        
    def takeoff(self, altitude):
        print(f"Taking off to {altitude} meters...")
        
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0,
            altitude
        )
        
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack and ack.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("✓ Takeoff command accepted")
                return True
            else:
                print(f"✗ Takeoff failed with result: {ack.result}")
                return False
        
        return False
        
    def wait_altitude(self, target_alt, tolerance=1.0):
        print(f"Waiting for altitude {target_alt}m (tolerance ±{tolerance}m)...")
        
        while True:
            msg = self.master.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=True,
                timeout=1
            )
            
            if msg:
                current_alt = msg.relative_alt / 1000.0
                print(f"  Altitude: {current_alt:.2f}m / {target_alt}m", end='\r')
                
                if abs(current_alt - target_alt) < tolerance:
                    print(f"\n✓ Target altitude reached: {current_alt:.2f}m")
                    return True
                    
            time.sleep(0.5)
            
    def goto_position(self, lat, lon, alt):
        print(f"Navigating to LAT:{lat}, LON:{lon}, ALT:{alt}m")
        
        self.master.mav.set_position_target_global_int_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,
            0, 0, 0,
            0, 0
        )
        
        print("✓ Navigation command sent")
        
    def wait_position_reached(self, target_lat, target_lon, tolerance=2.0):
        print(f"Waiting for position reached (tolerance {tolerance}m)...")
        
        while True:
            msg = self.master.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=True,
                timeout=1
            )
            
            if msg:
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
                
                distance = self.haversine_distance(
                    current_lat, current_lon,
                    target_lat, target_lon
                )
                
                print(f"  Distance to target: {distance:.2f}m", end='\r')
                
                if distance < tolerance:
                    print(f"\n✓ Target position reached! ({distance:.2f}m)")
                    return True
                    
            time.sleep(0.5)
            
    def land(self):
        print("Starting landing...")
        
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0, 0, 0, 0, 0, 0, 0
        )
        
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack and ack.command == mavutil.mavlink.MAV_CMD_NAV_LAND:
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print("✓ Landing command accepted")
                return True
            else:
                print(f"✗ Landing failed with result: {ack.result}")
                return False
        
        return False
        
    def get_current_position(self):
        msg = self.master.recv_match(
            type='GLOBAL_POSITION_INT',
            blocking=True,
            timeout=2
        )
        
        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0
            return lat, lon, alt
        
        return None, None, None
    
    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        while self.monitoring:
            msg = self.master.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=True,
                timeout=1
            )
            if msg:
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.relative_alt / 1000.0
                print(f"[Monitor] LAT:{lat:.6f} LON:{lon:.6f} ALT:{alt:.2f}m")
            time.sleep(2)
        
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371000
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_phi/2) ** 2 + 
             math.cos(phi1) * math.cos(phi2) * 
             math.sin(delta_lambda/2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

def print_menu():
    print("\n" + "="*60)
    print("DRONE CONTROL MENU")
    print("="*60)
    print("a - ARM             : Arm motors (do this first!)")
    print("g - GUIDED OPTIONS  : Set pilot control in GUIDED mode")
    print("1 - TAKEOFF         : Takeoff to altitude")
    print("2 - MOVE TO TARGET  : Navigate to target coordinates")
    print("3 - LOITER          : Stop and loiter at position")
    print("4 - LAND            : Land")
    print("d - DISARM          : Disarm motors")
    print()
    print("e - EMERGENCY       : Switch to STABILIZE (FULL PILOT CONTROL)")
    print("m - Start/Stop position monitoring")
    print("p - Show current position")
    print("s - Show status")
    print("q - Quit")
    print("="*60)

def show_status(drone):
    lat, lon, alt = drone.get_current_position()
    if lat is not None:
        print("\n--- DRONE STATUS ---")
        print(f"Position: LAT {lat:.6f}, LON {lon:.6f}")
        print(f"Altitude: {alt:.2f}m")
        print("--------------------")
    else:
        print("✗ Cannot read position")

def main():
    
    print("="*60)
    print("DRONE CONTROL - Interactive Mode")
    print("="*60)
    print(f"Connection: {CONNECTION_STRING}")
    print(f"Takeoff altitude: {TAKEOFF_ALT}m")
    print(f"Target: LAT {TARGET_LAT}, LON {TARGET_LON}")
    print("="*60)
    
    drone = DroneController(CONNECTION_STRING)
    monitoring = False
    
    try:
        while True:
            print_menu()
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'a':
                print("\n>>> EXECUTING: ARM <<<")
                print("⚠️  WARNING: Motors will arm!")
                confirm = input("Type 'yes' to confirm arm: ").strip().lower()
                if confirm == 'yes':
                    drone.set_mode('GUIDED')
                    time.sleep(1)
                    drone.arm()
                    print("✓ ARM COMPLETE! Motors ready for takeoff.")
                else:
                    print("✗ Arm cancelled")
            
            elif command == 'g':
                print("\n>>> GUIDED OPTIONS <<<")
                print("0 = No pilot control")
                print("1 = Allow pilot position override")
                print("2 = Allow pilot altitude override")
                print("4 = Allow pilot yaw override")
                print("7 = Allow all (position + altitude + yaw)")
                print()
                try:
                    option = int(input("Enter GUIDED_OPTIONS value (0-7): ").strip())
                    drone.set_guided_options(option)
                except ValueError:
                    print("✗ Invalid input. Must be a number.")
            
            elif command == '1':
                print("\n>>> EXECUTING: TAKEOFF <<<")
                drone.takeoff(TAKEOFF_ALT)
                drone.wait_altitude(TAKEOFF_ALT, tolerance=1.5)
                print("✓ TAKEOFF COMPLETE!")
                
            elif command == '2':
                print("\n>>> EXECUTING: MOVE TO TARGET <<<")
                drone.goto_position(TARGET_LAT, TARGET_LON, TAKEOFF_ALT)
                drone.wait_position_reached(TARGET_LAT, TARGET_LON, tolerance=3.0)
                print("✓ MOVE TO TARGET COMPLETE!")
                
            elif command == '3':
                print("\n>>> EXECUTING: LOITER <<<")
                drone.set_mode('LOITER')
                print("✓ LOITER MODE ACTIVE!")
                
            elif command == '4':
                print("\n>>> EXECUTING: LAND <<<")
                drone.set_mode('LAND')
                print("Waiting for landing to complete...")
                time.sleep(5)
                print("✓ LANDING COMPLETE!")
                
            elif command == 'd':
                print("\n>>> EXECUTING: DISARM <<<")
                confirm = input("Type 'yes' to confirm disarm: ").strip().lower()
                if confirm == 'yes':
                    drone.disarm()
                    print("✓ DISARM COMPLETE!")
                else:
                    print("✗ Disarm cancelled")
            
            elif command == 'e':
                print("\n>>> EMERGENCY: STABILIZE MODE <<<")
                print("⚠️⚠️⚠️  SWITCHING TO STABILIZE MODE!")
                print("⚠️⚠️⚠️  PILOT WILL HAVE FULL CONTROL!")
                drone.set_mode('STABILIZE')
                print("\n✓✓✓ STABILIZE MODE ACTIVATED ✓✓✓")
                print("✓✓✓ PILOT HAS FULL CONTROL ✓✓✓")
                print("Control with your RC transmitter now!")
                
            elif command == 'm':
                if not monitoring:
                    print("\n✓ Starting monitoring...")
                    drone.start_monitoring()
                    monitoring = True
                else:
                    print("\n✓ Stopping monitoring...")
                    drone.stop_monitoring()
                    monitoring = False
                    
            elif command == 'p':
                show_status(drone)
                
            elif command == 's':
                show_status(drone)
                
            elif command == 'q':
                print("\nExiting program...")
                if monitoring:
                    drone.stop_monitoring()
                break
                
            else:
                print("✗ Command not recognized!")
            
            input("\nPress ENTER to continue...")
            
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if monitoring:
            drone.stop_monitoring()
        print("\nDone")

if __name__ == "__main__":
    main()