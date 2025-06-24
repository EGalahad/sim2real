# Quick Start


Follow the instructions here, **must** use mamba to install ros2:
https://robostack.github.io/GettingStarted.html#install-mamba

## Install ROS Humble

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults


mamba install ros-humble-desktop
```
python 3.11 is by default, installed by mamba.
```bash
mamba install -c conda-forge python=3.11
```


## Install unitree sdk2 python



## Run policy

refer to [CheatSheet_Haoyang.md](CheatSheet_Haoyang.md)