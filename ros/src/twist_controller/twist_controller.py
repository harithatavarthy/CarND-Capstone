import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband,
                 decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio

        self.last_velocity = 0.

        self.lpf_fuel = LowPassFilter(60.0, 0.1)
        self.lpf_accel = LowPassFilter(0.5, 0.02)
	#self.lpf_angv = LowPassFilter(1.0, 1)

        self.accel_pid = PID(0.4, 0.1, 0.0, 0.0, 1.0)
        self.speed_pid = PID(2.0, 0.0, 0.0, self.decel_limit,
                             self.accel_limit)

        self.yaw_control = YawController(wheel_base, steer_ratio, 4. * ONE_MPH,
                                         max_lat_accel, max_steer_angle)
        self.last_ts = None

    def set_fuel(self, level):
        self.lpf_fuel.filt(level)

    def get_vehicle_mass(self):
        return self.vehicle_mass + \
               self.lpf_fuel.get() / 100.0 * self.fuel_capacity * GAS_DENSITY

    def time_elasped(self, msg=None):
        now = rospy.get_time()
        if self.last_ts is None:
            self.last_ts = now
        elasped, self.last_ts = now - self.last_ts,  now
        return elasped

    def control(self,current_velocity,dbw_enabled,linear_velocity,angular_velocity):
	#rospy.logwarn("Current Velocity = %f, Linear Velocity = %f, Angular Velocity = %f", current_velocity, linear_velocity, angular_velocity)
        time_elasped = self.time_elasped()
	given_av = angular_velocity
        if time_elasped > 1./5 or time_elasped < 1e-4:
            self.speed_pid.reset()
            self.accel_pid.reset()
            self.last_velocity = current_velocity
            return 0., 0., 0.

        vehicle_mass = self.get_vehicle_mass()
        vel_error = linear_velocity - current_velocity
	#rospy.logwarn("Velocity Error : %s",vel_error)

        if abs(linear_velocity) < ONE_MPH:
            self.speed_pid.reset()

        accel_cmd = self.speed_pid.step(vel_error, time_elasped)
	#rospy.logwarn("Velocity step : %s",accel_cmd)
	
        min_speed = ONE_MPH * 5
        if linear_velocity < 0.01:
	    #rospy.logwarn("Current Velocity = %f, Linear Velocity = %f, Angular Velocity = %f", current_velocity, linear_velocity, angular_velocity)
	    #rospy.logwarn("Velocity Error : %s",vel_error)
	    #rospy.logwarn("Velocity step Before: %s",accel_cmd)
            accel_cmd = min(accel_cmd,
                            -530. / vehicle_mass / self.wheel_radius)
	    #rospy.logwarn("computed accel_cmd : %s",accel_cmd)
	    #rospy.logwarn("Velocity step After: %s",accel_cmd)
        elif linear_velocity < min_speed:
            angular_velocity *= min_speed/linear_velocity
            linear_velocity = min_speed
	    #rospy.logwarn("updated angular_velocity : %s", angular_velocity)
	    #rospy.logwarn("updated linear velocity: %s",linear_velocity)

        accel = (current_velocity - self.last_velocity) / time_elasped
        #rospy.logwarn("Current accel = %f, Last = %f, New = %f", accel, self.last_velocity, current_velocity)
        self.lpf_accel.filt(accel)
        self.last_velocity = current_velocity

        throttle, brake, steering = 0., 0., 0.
        if dbw_enabled:
            if accel_cmd >= 0:
                throttle = self.accel_pid.step(accel_cmd - self.lpf_accel.get(),
                                               time_elasped)
            else:
                self.accel_pid.reset()
            if (accel_cmd < -self.brake_deadband) or \
               (linear_velocity < min_speed):
              brake = -accel_cmd * vehicle_mass * self.wheel_radius
	    #angular_velocity_filt = self.lpf_angv.filt(angular_velocity)
            steering = self.yaw_control.get_steering(linear_velocity,
                                                     angular_velocity,
                                                     current_velocity)
        else:
            self.speed_pid.reset()
            self.accel_pid.reset()
	#rospy.logwarn("Throttle = %f, Brake = %f, Steering = %f" , throttle, brake, steering)
	#rospy.logwarn("given_av = %f,  comp_av = %f, filt_av = %f, Steering = %f" , given_av, angular_velocity, angular_velocity_filt, steering)
        return throttle, brake, steering
