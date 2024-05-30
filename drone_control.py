# Import the Tello class from the djitellopy library
from djitellopy import Tello 

# Import the time module for time-related operations
import time

# Create an instance of the Tello class
drone = Tello()

# Connect to the Tello drone
drone.connect()

# Get the battery level of the drone
battery_level = drone.get_battery()

# Print the battery level
print("battery level:", battery_level, "%")

# Print the type of the battery_level variable
print("the battery variable type is:", type(battery_level))

# Command the drone to take off
drone.takeoff()

# Print a message indicating successful takeoff
print("take off successful")

# Pause the execution for 8 seconds
time.sleep(8)

# Command the drone to land
drone.land()

# Print a message indicating successful landing
print("landing successful")

# End the connection with the drone
drone.end()
