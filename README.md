# Autonomous-Driving-Car-MLis-2

## Group Information

PENDING

## Overview of Project

The aim of this assignment was to develop a cars with autonomous driving capabilities where it is capable of lane navigation and collision avoidance. ​The cars should be able to predict speed and steering angle based on the current view captured by its camera.

In this project, our group have developed a convolutional neural network (CNN) based on Nvidia's CNN Architecture. We also performed various types of data augmentation, image preprocessing and feature engineering to further enhance the performance of our model.

Other than that, we also developed a simulation using Unity to combat the issue of lack of data. Due to the Covid-19 situation, collecting real world data using PiCar is no longer possible. Therefore, this simulation allowed us to collect, in theory unlimited training data remotely.

## PiCar Simulation Overview

PiCar Simulation are developed using Unity, and scripts are written in C#. Models appear in this simulation are created using Blender while refering to real-wrold models provided by MLiS 2 course. This simulation allowed us to collect simulation data, with a similiar setting following the actual PiCar provided by faculty. A demo of this simulation can be found in this [video](https://youtu.be/5SC681vJocY).

![Screen capture of PiCar Simulation](/images/sim-snapshot.PNG)   
![Screen capture of PiCar Simulation](/images/snapshot-sim-just-in-case.PNG)

Capabilities of PiCar Simulation
- Recording data​
- Enable self-driving mode​
- Configure camera angle​
- Configure image capture interval​
- Switch between tracks​
- Switch between scenes​
- Ability to configure environment​
- Spawning object

## Simulation Setup Guide

1. Extract piCar Sim Build.rar
2. Run PiCar Simulation.exe
3. Select Start Up option
4. Start using PiCar simulation

## Simulation User Guide

- 'W', 'S' to control speed
- 'A', 'D' to control steering angle
- 'R' to start recording simulation data (data are saved in \PiCar Simulation_Data\Image Data)
- Hold Left Mouse Button to pick up object
- 'Q' to rotate object

- Click 'Next Scene' to switch scene (currently contains 3 scenes)
- Click 'Next Track' to switch track (currently contains 5 tracks)
- Click 'Toggle Self Driving' button to start autonomous driving (refers to below section to setup)
- Click 'Toggle Camera' button to switch view
- Click 'Spawn Menu' button to spawn object in drop down (Spawning multiple objects at once will cause object to fall)

- Drag 'Camera Angle' slider to change camera angle
- Drag 'Time Interval' slider to change rate of recorded data being save per second, also affect the output rate of stream view for self driving capability. (default is 1 image per second)

## Setup for Self-Driving Model in simulation

1. Activate Conda environment with 'picar_env.yml'
2. Run \PiCar Simulation_Data\PyScript\steering_angle_prediction.py
3. Run PiCar Simulation.exe
4. Click 'Toggle Self Driving' button in simulation

the summary of the working mechanism of this feature are shown in below figure.
![Mechanism of Self Driving Mode](/images/self-driving-mode-mech.PNG)

## Neural Network Model Overview

PENDING

## Result

PENDING