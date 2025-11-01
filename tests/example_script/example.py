#!/usr/bin/env python3
# This generated code is licensed under CC-BY.

# First, click 'Download model data' (or 'Modelldaten downloaden') and place the file you get in the same folder as this script.
# Then, run this script like this:
# python3 scriptname.py
# - or -
# python3 scriptname.py 1.jpg 2.jpg 3.jpg

import sys
import re
import platform
import shutil
import os
import subprocess

try:
    import venv
except ModuleNotFoundError:
    print("venv not found. Is python3-venv installed?")
    sys.exit(1)

from pathlib import Path

VENV_PATH = Path.home() / ".asanai_venv"
PYTHON_BIN = VENV_PATH / ("Scripts" if platform.system() == "Windows" else "bin") / ("python.exe" if platform.system() == "Windows" else "python")

def create_and_setup_venv():
    print(f"Creating virtualenv at {VENV_PATH}")
    venv.create(VENV_PATH, with_pip=True)
    subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "git+https://github.com/NormanTUD/asanAI_python.git"])

def restart_with_venv():
    try:
        result = subprocess.run(
            [str(PYTHON_BIN)] + sys.argv,
            text=True,
            check=True,
            env=dict(**os.environ)
        )
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error while restarting python: {e}")
        sys.exit(1)

try:
    import asanai
except ModuleNotFoundError:
    if not VENV_PATH.exists():
        create_and_setup_venv()
    else:
        try:
            subprocess.check_call([PYTHON_BIN, "-m", "pip", "install", "-q", "--upgrade", "asanai"])
        except subprocess.CalledProcessError:
            shutil.rmtree(VENV_PATH)
            create_and_setup_venv()
            restart_with_venv()
    try:
        restart_with_venv()
    except KeyboardInterrupt:
        print("You cancelled installation")
        sys.exit(0)

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

import rich
from rich.table import Table

if asanai.output_is_simple_image(model) or asanai.output_is_complex_image(model):
    if len(sys.argv) == 1:
        asanai.visualize_webcam(model, height, width, divide_by)
    else:
        for a in range(1, len(sys.argv)):
            filename = sys.argv[a]
            asanai.visualize(model, filename)
elif asanai.model_is_simple_classification(model):
    for a in range(1, len(sys.argv)):
        filename = sys.argv[a]
        image = asanai.load(filename, height, width, divide_by)

        if image is None:
            asanai.console.print(f"[bold red]Error:[/] Could not load image: [italic]{filename}[/]")
            continue

        prediction = model.predict(image, verbose=0)

        for prediction_idx in range(len(prediction)):
            nr_labels = len(prediction[prediction_idx])
            if len(labels) < nr_labels:
                asanai.console.print(
                    rich.Panel.fit(
                        f"[bold red]Aborted:[/] Model returned [bold]{nr_labels}[/] labels,\n"
                        f"but only [bold]{len(labels)}[/] labels are defined.",
                        title="Error",
                        border_style="red"
                    )
                )
                sys.exit(1)

            table = Table(show_lines=True)

            table.add_column("Label", style="cyan", justify="right")
            table.add_column("Probability/Output", style="magenta", justify="left")

            for nr_idx in range(nr_labels):
                table.add_row(labels[nr_idx], f"{prediction[prediction_idx][nr_idx]:.4f}")

            asanai.console.print(table)

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
                        try:
                            cv2.imshow('frame', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                            if cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                                print("\nWindow was closed.")
                                break
                        except cv2.error:
                            print("")
                            sys.exit(1)

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("You pressed CTRL-c. Program will end.")
            sys.exit(0)
else:
    output = model.predict(dummy_input, verbose=0)

    print("Raw Output:")
    print(output)
