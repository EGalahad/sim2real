# Jetson gs_usb module

The `5.15.148-tegra` kernel on `g1-ygx` has `CONFIG_CAN_GS_USB` disabled even
though the DH116S USB-CANFD adapters require that driver. Build `gs_usb.c` from
the upstream Linux `v5.15` tag against the installed Jetson headers:

```bash
curl -L -o gs_usb.c \
  https://raw.githubusercontent.com/torvalds/linux/v5.15/drivers/net/can/usb/gs_usb.c
make -C /lib/modules/$(uname -r)/build M="$PWD" modules
sudo install -m 0644 gs_usb.ko /lib/modules/$(uname -r)/extra/gs_usb.ko
sudo depmod -a
sudo modprobe gs_usb
```

The downloaded kernel source and compiled objects are runtime artifacts and are
not committed.
