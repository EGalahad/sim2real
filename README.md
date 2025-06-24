# Quick Start


Use conda and python 3.9.

## Install unitree sdk2 python


## Install ROS Humble

```bash
# this adds the conda-forge channel to the new created environment configuration 
conda config --env --add channels conda-forge
# and the robostack channel
conda config --env --add channels robostack-staging
# remove the defaults channel just in case, this might return an error if it is not in the list which is ok
conda config --env --remove channels defaults


conda install ros-humble-desktop
```

## Run policy

refer to [CheatSheet_Haoyang.md](CheatSheet_Haoyang.md)