# OpenBarnes [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
OpenBarnes is a rodent tracking & analysis software for [Barnes Maze](https://en.wikipedia.org/wiki/Barnes_maze) experiments.

# Getting started

Start by cloning the repository:
```
git clone https://github.com/Ferill/OpenBarnes.git
```
Set up a virtual environment
```
cd OpenBarnes
virtualenv env
source env/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```
Run the application
```
python run.py samples/barnes-sample-1.mp4
python run.py samples/barnes-sample-2.mp4
```

# Features
## Correct Image Distortion
![distort](https://user-images.githubusercontent.com/65185610/81633195-d1a45800-9429-11ea-9137-98acfcaa7aa3.jpg)

## Export tracks to SVG format
![small-plot](https://user-images.githubusercontent.com/65185610/81640309-d1ad5380-943b-11ea-86f7-cb9277f2cf95.png)

# Demo
![barnes-demo-1](https://user-images.githubusercontent.com/65185610/81633131-a4f04080-9429-11ea-9564-0b47cad60ec2.gif) ![barnes-demo-2](https://user-images.githubusercontent.com/65185610/81638474-f7842980-9436-11ea-867b-7a138b0b51f5.gif)


# License

This project is licensed under the **GNU GPL v3** License. Read the [LICENSE.md](LICENSE.md) file for details.
