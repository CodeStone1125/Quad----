# script.py

import os
import sys

# Get the directory of the current script or module
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the "build" folder
build_path = os.path.join(current_dir, 'build')

# Add the "build" folder to sys.path
sys.path.append(build_path)

# Now you can import the "example" module
import example
print(example.add())

import model
print(model.add())

import quad
print(quad.add())


