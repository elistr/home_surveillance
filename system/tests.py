import SurveillanceSystem
import Camera
import json
import logging
from logging.handlers import RotatingFileHandler
import threading
import time
from random import random
import os
import sys
import cv2
import psutil

gather_mode=True

# Initialises system variables, this object is the heart of the application
HomeSurveillance = SurveillanceSystem.SurveillanceSystem(gather_mode=gather_mode)
# Threads used to continuously push data to the client
alarmStateThread = threading.Thread()
facesUpdateThread = threading.Thread()
monitoringThread = threading.Thread()

camURL = "http://10.0.0.173/videostream.cgi?user=admin&pwd="
camURL_tests = "file:///host/hacks/dan.png"
application = "JingleBot"
detectionMethod = False
fpsTweak = False

test = True

if test:
    camURL = camURL_tests


with HomeSurveillance.camerasLock:
    HomeSurveillance.add_camera(SurveillanceSystem.Camera.IPCamera(camURL, application, detectionMethod, fpsTweak))




