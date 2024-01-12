#!/bin/bash

# # Check if pip is installed
# if ! command -v pip &> /dev/null; then
#     echo "Error: pip is not installed. Please install pip first."
#     exit 1
# fi

# Install required Python packages
pip install numpy
pip install pybullet
pip install matplotlib

# It seems like 'utils' is a custom module. If it's not available on PyPI,
# you might need to provide instructions on how to install or include it.

# You can provide additional instructions or commands if needed

# echo "Installation complete."