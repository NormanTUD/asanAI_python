#!/usr/bin/env python3
# This generated code is licensed under CC-BY.

# First, click 'Download model data' (or 'Modelldaten downloaden') and place the files you get in the same folder as this script.
# Then, run this script like this:
# python3 scriptname.py
# - or -
# python3 scriptname.py 1.jpg 2.jpg 3.jpg

import sys
import re
import os
import subprocess

try:
    import venv
except ModuleNotFoundError:
    print("venv not found. Is python3-venv installed?")
    sys.exit(1)
from pathlib import Path

VENV_PATH = Path.home() / ".asanai_venv"
PYTHON_BIN = VENV_PATH / "bin" / "python"

def create_and_setup_venv():
    print(f"Creating virtualenv at {VENV_PATH}")
    venv.create(VENV_PATH, with_pip=True)
    subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "asanai"])

def restart_with_venv():
    os.execv(PYTHON_BIN, [str(PYTHON_BIN)] + sys.argv)

try:
    import asanai
except ModuleNotFoundError:
    if not VENV_PATH.exists():
        create_and_setup_venv()
    else:
        subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "-q", "asanai"])
    restart_with_venv()

tf = asanai.install_tensorflow(sys.argv)

# This code converts the tensorflow.js image from the browser to the tensorflow image for usage with python
if not os.path.exists('model.h5'):
    asanai.convert_to_keras_if_needed()

if not os.path.exists('model.h5'):
    print('model.h5 cannot be found')
    sys.exit(1)

model = tf.keras.models.load_model('model.h5')

model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

labels = ['fire', 'mandatory', 'prohibition', 'rescue', 'warning']
height = 40
width = 40
divide_by = 255

for a in range(1, len(sys.argv)):
    filename = sys.argv[a]
    image = asanai.load(filename, height, width, divide_by)
    if image is not None:
        print(f'{filename}:')
        prediction = model.predict(image, verbose=0)
        for i in range(0, len(prediction)):
            nr_labels = len(prediction[i])
            if len(labels) < nr_labels:
                print(f'Cannot continue. Has only {len(labels)} labels, but needs at least {nr_labels}')
                sys.exit(1)
            for j in range(0, nr_labels):
                print(labels[j] + ': ' + str(prediction[i][j]))

# If no command line arguments were given, try to predict the current webcam:
if len(sys.argv) == 1:
    try:
        import cv2

        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                import sys
                print("Could not load frame from webcam. Is the webcam currently in use?")
                sys.exit(1)

            image = asanai.load_frame(frame, height, width, divide_by)

            if image is not None:
                predictions = model.predict(image, verbose=0)

                frame = asanai.annotate_frame(frame, predictions, labels)

                asanai.print_predictions_line(predictions, labels)

                if frame is not None:
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                        print("Window was closed.")
                        break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print("You pressed CTRL-c. Program will end.")
        sys.exit(0)
