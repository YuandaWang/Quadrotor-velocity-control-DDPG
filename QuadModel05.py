"""
Quanser Quadrotor Model

Input is the thrust/PWM input of motors, output is all states of the quadrotor;
including:
attitude: phi, theta, psi, the rates
position: X, Y, Z, and the speeds



         X
         1
         
   Y 2       4
   
         3


@author: Yuanda Wang

Created on Jan 10 2017


E-1. The height Z is added. 12/05/2017 ---Yuanda Wang
E-2. The position X Y is added. 12/08/2017  ---Yuanda Wang

---- For desired velocity  ----
E-3. All new model, all random velocity, attitude, and angular velocity

"""

import numpy as np
import matplotlib.pyplot as plt

# RK4 function for simulation
def RK4(ufunc, x0, u, h):
    k1 = ufunc(x0, u)
    k2 = ufunc(x0 + h*k1/2, u)
    k3 = ufunc(x0 + h*k2/2, u)
    k4 = ufunc(x0 + h*k3, u)
    x1 = x0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return x1
    
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


# first-order dynamic
def FO_dynamic(y, u):
    y = u
    return y

def generate_init_atti(max_angle, max_angle_vel):
    co = 0.5
    
    phi   = (2 * np.random.random() - 1) * max_angle * co
    theta = (2 * np.random.random() - 1) * max_angle * co
    psi   = (2 * np.random.random() - 1) * max_angle * co
    
    phidot   = (2 * np.random.random() - 1) * max_angle_vel * co
    thetadot = (2 * np.random.random() - 1) * max_angle_vel * co
    psidot   = (2 * np.random.random() - 1) * max_angle_vel * co
    
    return np.array([phi, theta, psi, phidot, thetadot, psidot])

def generate_init_velocity(max_velocity):
    co = 0.5
    
    velX = (2 * np.random.random() - 1) * max_velocity * co
    velY = (2 * np.random.random() - 1) * max_velocity * co
    velZ = (2 * np.random.random() - 1) * max_velocity * co
    
    return np.array([velX, velY, velZ])
    

class QuadModel():
    
    def __init__(self, d_vel):
        # limits
        self.max_angle_vel = 4.5
        self.max_angle = np.pi/2
        self.max_vel = 10.0
        self.max_pos = 10.0 # no use 
        
        # simulation
        self.Ts = 0.01
        self.U = [0, 0, 0, 0]
        self.g = 9.81
        self.done = False
        
        # initial states
        self.atti = generate_init_atti(self.max_angle, self.max_angle_vel)
        self.vel  = generate_init_velocity(self.max_vel)
        self.pos  = np.array([0, 0, 0])

        # desired velocity
        self.d_vel = d_vel
        


    def change_dvel(self, d_vel):
        # change desired velocity
        self.d_vel = d_vel
        
        
    # attitude model from u to attitude    
    def atti_model(self, u):
        Ts = self.Ts
        atti = self.atti
        
        phi, theta, psi = atti[0], atti[1], atti[2]
        phidot, thetadot, psidot = atti[3], atti[4], atti[5]

        # u to angular velocity
        phidot = RK4(FO_dynamic, phidot, u[1], Ts)
        thetadot = RK4(FO_dynamic, thetadot, u[2], Ts)
        psidot = RK4(FO_dynamic, psidot, u[3], Ts)
        # lock Yaw
        # psidot = 0

        # angular velocity to attitude
        phi = RK4(FO_dynamic, phi, phidot, Ts)
        theta = RK4(FO_dynamic, theta, thetadot,Ts)
        psi = RK4(FO_dynamic, psi, psidot, Ts)
        # lock Yaw
        # psi = 0

        self.atti = [phi, theta, psi, phidot, thetadot, psidot]

    # pos dynamics X, Y, Z
    def pos_dynamic_X(self, velX, u):
        phi, theta, psi = self.atti[0], self.atti[1], self.atti[2]
        velXdot = u * (np.sin(psi) * np.sin(phi) + np.cos(psi) * np.sin(theta) * np.cos(phi))
        return velXdot
    
    def pos_dynamic_Y(self, velY, u):
        phi, theta, psi = self.atti[0], self.atti[1], self.atti[2]
        velYdot = u * (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi))
        return velYdot
    
    def pos_dynamic_Z(self, velZ, u):
        phi, theta = self.atti[0], self.atti[1]
        velZdot = u * np.cos(theta) * np.cos(phi) - self.g
        return velZdot
        
        
    
    def pos_model(self, u):
        posX, posY, posZ = self.pos[0], self.pos[1], self.pos[2]
        velX, velY, velZ = self.vel[0], self.vel[1], self.vel[2]
        
        velX = RK4(self.pos_dynamic_X, velX, u[0], self.Ts)
        posX = RK4(FO_dynamic, posX, velX, self.Ts)
        
        velY = RK4(self.pos_dynamic_Y, velY, u[0], self.Ts)
        posY = RK4(FO_dynamic, posY, velY, self.Ts)
    
        velZ = RK4(self.pos_dynamic_Z, velZ, u[0], self.Ts)
        posZ = RK4(FO_dynamic, posZ, velZ, self.Ts)
             
        self.pos = [posX, posY, posZ]
        self.vel = [velX, velY, velZ]
    
    def check_fail(self):
        
        CHECK_ANGLE      = True
        CHECK_ANGLESPEED = True
        CHECK_VELOCITY   = True
        CHECK_POSITION  = False
        
        # angular speed fail
        phidot, thetadot, psidot = self.atti[3], self.atti[4], self.atti[5]
        if CHECK_ANGLESPEED and (np.max(np.absolute([phidot, thetadot, psidot])) > self.max_angle_vel):
            self.done = True
        
        # angle fail
        phi, theta, psi = self.atti[0], self.atti[1], self.atti[2]
        if CHECK_ANGLE and (np.max(np.absolute([phi, theta, psi])) > self.max_angle):
            self.done = True
            
        # velocity fail
        velX, velY, velZ = self.vel[0], self.vel[1], self.vel[2]
        if CHECK_VELOCITY and (np.max(np.absolute([velX, velY, velZ])) > self.max_vel):
            self.done = True
            
        # position fail
        posX, posY, posZ = self.pos[0], self.pos[1], self.pos[2]
        if CHECK_POSITION and np.max(np.absolute([posX, posY, posZ])) > self.max_pos:
            self.done = True

    def motor_angle_acc(self, U):
        # parameters
        K_m = self.g * 1.79 / 2    # the horvering throttle is 0.5 in [0 -- 1]
        omega_motor = 100
        K_p = 1 #?
        # frame params
        R = 0.2 # motor arm length
        M = 1.79
        Jx = 0.04
        Jy = 0.04
        Jz = 0.03
        Jp = 0.002
        
        self.U = np.array(U)
        
        u1 = (K_m*U[0] + K_m*U[1] + K_m*U[2] + K_m*U[3]) / M
        u2 = K_m * R * (U[1] - U[3]) / Jx
        u3 = K_m * R * (U[0] - U[2]) / Jy
        u4 = K_p * (U[0] - U[1] + U[2] - U[3]) / Jz
        u = [u1, u2, u3, u4]
        
        return u
        
    def observe(self):
        # attitude
        atti = self.atti
        phi, theta, psi = atti[0], atti[1], atti[2]
        phidot, thetadot, psidot = atti[3], atti[4], atti[5]
        
        # normalize dot and height for observation
        phidot /= self.max_angle_vel
        thetadot /= self.max_angle_vel
        psidot /= self.max_angle_vel
        
        # pos --normalized
        pos = self.pos
        posX, posY, posZ = pos[0]/self.max_pos, pos[1]/self.max_pos, pos[2]/self.max_pos
        
        # vel
        vel = self.vel
        velX, velY, velZ = vel[0]/self.max_vel, vel[1]/self.max_vel, vel[2]/self.max_vel
        
        # d_vel
        d_vel = self.d_vel
        dX, dY, dZ = d_vel[0]/self.max_vel, d_vel[1]/self.max_vel, d_vel[2]/self.max_vel
        
        # velocity error  --normalized
        e_velX, e_velY, e_velZ = (dX-velX), (dY-velY), (dZ-velZ)

        
        # easy to use in give reward
        self.vel_errors = [e_velX, e_velY, e_velZ]
        
        return np.array([phi, phidot, 
                         theta, thetadot, 
                         psi, psidot,
                         velX, velY, velZ,
                         dX, dY, dZ])
    
    def rmap(self, x):
        tau = 500
        y = np.log((x*tau+1)) / np.log(tau+1)
        
        return y
    
    def givereward(self):
        
        # attitude
        atti = self.atti
        phi, theta, psi = atti[0], atti[1], atti[2]
        phi = angle_normalize(phi) / np.pi * 2
        theta = angle_normalize(theta) / np.pi * 2
        psi = angle_normalize(psi) / np.pi * 2
        
        
        # velocity error reward  --normalized
        e_vel = self.vel_errors
        eX, eY, eZ = e_vel[0], e_vel[1], e_vel[2]
        
        cost =   self.rmap(eX**2) + self.rmap(eY**2) + self.rmap(eZ**2)  + self.rmap(psi**2)

        cost = 4 - cost
        
        return cost
        
    def step(self, U):
        # 1. convert motor PWM (U -- 4-dim) to angular acc
        self.U = U
        u = self.motor_angle_acc(U)
        
        # 2.1 attitude model, from angular acc to angular velocity and attitude angles
        self.atti_model(u)
        
        # 2.2 position model, from attitude thrust to velocity and positions
        self.pos_model(u)
        
        # 2.3 check fail
        self.check_fail()
        
        # 3. generate observations
        ob = self.observe()
        
        # 4. generate reward
        reward = self.givereward()
        

        
        return ob, reward, self.done
    
        
    
    
    
    
    
