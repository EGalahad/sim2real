# Network Connection

## Hotspot

Open hotspot on laptop via usb adapter.

```bash
## configure hotspot
export SSID=MyHotspot
export PASSWORD=YourDesiredPassword
export USB_IF_NAME=wlxfc221c100233

nmcli device wifi hotspot ifname $USB_IF_NAME con-name $SSID ssid $SSID password $PASSWORD
nmcli connection modify $SSID ipv4.method shared

## start hotspot
nmcli connection up $SSID

# enable ip forwarding
sudo sysctl -w net.ipv4.ip_forward=1

export WIFI_NAME=wlp0s20f3
export USB_IF_NAME=wlxfc221c100233

sudo iptables -F
sudo iptables -t nat -F

sudo iptables -t nat -A POSTROUTING -s 192.168.123.0/24 -o $WIFI_NAME -j MASQUERADE

## allow forwarding from USB to Wi-Fi
sudo iptables -A FORWARD -i $USB_IF_NAME -o $WIFI_NAME -j ACCEPT

## allow forwarding from Wi-Fi to USB
sudo iptables -A FORWARD -i $WIFI_NAME -o $USB_IF_NAME -m state --state RELATED,ESTABLISHED -j ACCEPT
```

Connect to hotspot on orin.

```bash
# connect to hotspot
export SSID=MyHotspot
export PASSWORD=YourDesiredPassword

sudo nmcli connection add type wifi ifname wlan0 con-name $SSID ssid $SSID
sudo nmcli connection modify $SSID wifi-sec.key-mgmt wpa-psk
sudo nmcli connection modify $SSID wifi-sec.psk $PASSWORD

sudo nmcli connection up $SSID

# configue ip route
sudo ip route del default via 10.42.0.1 dev wlan0
sudo ip route add default via 10.42.0.1 dev wlan0 metric 100

# set dns
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

# Vicon
To start replay zmq server on laptop, run:

```bash
python scripts/publishers/vicon_pose_publisher.py
```


# MuJoCo Sim2Sim

## DeepMimic Walk (29 DoF)

```bash
# sim
python sim_env/base_sim.py --robot_config ./config/robot/g1.yaml --scene_config ./config/scene/g1_29dof_nohand.yaml

# policy
python rl_policy/base_policy.py --robot_config ./config/robot/g1.yaml --policy_config checkpoints/exports/G1TrackWalk/policy-xob4chg3-1500.yaml
```

## DeepMimic Push Door (29 DoF)

```bash
# sim
python sim_env/push_door.py --robot_config ./config/robot/g1.yaml --scene_config ./config/scene/g1_29dof_nohand-door.yaml

# policy
python rl_policy/local_tracking.py --robot_config ./config/robot/g1.yaml --policy_config ./checkpoints/exports/G1Track/policy-6s0umflf-final.yaml
```

## DeepMimic Push Box (29 DoF) with Mocap

```bash
python sim_env/push_box.py

python rl_policy/local_tracking.py --robot_config ./config/robot/g1.yaml --motion_path data/motion/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height-zero_waist_pitch-processed --policy_config checkpoints/exports/G1TrackBox/policy-dxx84vic-final.yaml
```