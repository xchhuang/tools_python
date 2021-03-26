## pyrender2

### Introduction
The code enables the efficient and off-screen rendering of depth maps from 3D triangle meshes for research in graphics and vision. The code is only tested on Ubuntu 16.04 with python3.

### Prerequisites
* `pip3 install Cython numpy trimesh shapely`
* `sudo apt-get install freeglut3-dev libglew-dev`

### Build & Run
* First build the project by:
`python3 setup.py build_ext --inplace`

* Then run the example by:
`python3 main.py`

### Acknowledgements
This project contains a Cython version of [librender](http://www.cvlibs.net/software/librender/) and a modified version based on [pyrender](https://github.com/griegler/pyrender). In order to avoid conflicts with another excellent [pyrender](https://github.com/mmatl/pyrender) while installing both renderers, I named it to be pyrender2.

