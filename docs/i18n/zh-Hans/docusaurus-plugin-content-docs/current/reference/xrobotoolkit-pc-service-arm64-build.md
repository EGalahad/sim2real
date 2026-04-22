---
title: XRoboToolkit PC Service ARM64 Build
slug: /reference/xrobotoolkit-pc-service-arm64-build
---

# XRoboToolkit PC Service ARM64 Instructions

Target machine:

- `g1-rp`
- `aarch64`
- `Ubuntu 20.04.6 LTS`
- output `.deb`: `~/src/XRoboToolkit-PC-Service-main/RoboticsService/Package/output/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb`

## Why Rebuild

The upstream ARM64 `.deb` installs but does not run on Ubuntu 20.04. It was
built against newer runtime symbols:

```text
GLIBC_2.34 not found
GLIBC_2.33 not found
GLIBC_2.32 not found
GLIBCXX_3.4.29 not found
```

Build on `g1-rp` so the service links against Ubuntu 20.04-compatible runtime
libraries.

## Setup

### 1. system packages

Run on `g1-rp`:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  python3 \
  python3-pip \
  protobuf-compiler \
  libprotobuf-dev
```

Enable proxy if GitHub / Qt downloads are slow:

```bash
source ~/.bashrc
proxy_on
```

### 2. source

On a machine with good GitHub access:

```bash
curl -L \
  -o /tmp/xrobotoolkit-pc-service-main.tar.gz \
  https://codeload.github.com/XR-Robotics/XRoboToolkit-PC-Service/tar.gz/refs/heads/main

scp /tmp/xrobotoolkit-pc-service-main.tar.gz \
  g1-rp:/home/elijah/src/xrobotoolkit-pc-service-main.tar.gz
```

On `g1-rp`:

```bash
mkdir -p ~/src
cd ~/src
rm -rf XRoboToolkit-PC-Service-main
tar xf xrobotoolkit-pc-service-main.tar.gz
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService
```

### 3. Qt 6.7.3

```bash
python3 -m pip install --user --upgrade pip setuptools wheel aqtinstall
source ~/.bashrc
proxy_on

~/.local/bin/aqt install-qt \
  -O ~/Qt6 \
  linux_arm64 desktop 6.7.3 linux_gcc_arm64 \
  -m qt5compat
```

Check:

```bash
~/Qt6/6.7.3/gcc_arm64/bin/qmake -query QT_VERSION
```

Expected:

```text
6.7.3
```

### 4. gRPC 1.16

```bash
mkdir -p ~/pkgprobe ~/opt/grpc16-extract
cd ~/pkgprobe

apt download \
  libgrpc++-dev \
  libgrpc-dev \
  protobuf-compiler-grpc \
  libgrpc++1 \
  libgrpc6 \
  libc-ares2 \
  libssl1.1

cd ~/opt/grpc16-extract
rm -rf ./*

for f in \
  ~/pkgprobe/libgrpc++-dev_1.16.1-1ubuntu5_arm64.deb \
  ~/pkgprobe/libgrpc-dev_1.16.1-1ubuntu5_arm64.deb \
  ~/pkgprobe/protobuf-compiler-grpc_1.16.1-1ubuntu5_arm64.deb \
  ~/pkgprobe/libgrpc++1_1.16.1-1ubuntu5_arm64.deb \
  ~/pkgprobe/libgrpc6_1.16.1-1ubuntu5_arm64.deb \
  ~/pkgprobe/libc-ares2_1.15.0-1ubuntu0.5_arm64.deb \
  ~/pkgprobe/libssl1.1_1.1.1f-1ubuntu2.24_arm64.deb
do
  dpkg-deb -x "$f" ~/opt/grpc16-extract
done

cd ~/opt/grpc16-extract/usr/lib/aarch64-linux-gnu
ln -sf libcares.so.2 libcares.so
```

Check:

```bash
test -x ~/opt/grpc16-extract/usr/bin/grpc_cpp_plugin
ls ~/opt/grpc16-extract/usr/lib/aarch64-linux-gnu/libgrpc++.so.1
```

## Patch

Run from:

```bash
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService
```

### 1. remove Core5Compat

```bash
perl -0pi -e 's/find_package\(Qt6 REQUIRED COMPONENTS Core5Compat\)\n//g;
              s/target_link_libraries\(([^\n]+) PRIVATE Qt6::Core5Compat\)\n//g' \
  CMakeLists.txt \
  CommonUtils/CMakeLists.txt \
  Business/CMakeLists.txt \
  PXREAGRPCServer/CMakeLists.txt \
  RoboticsServiceProcess/CMakeLists.txt

perl -0pi -e 's/#include <QTextCodec>\n//g;
              s/\n\s*QTextCodec\* codec = QTextCodec::codecForName\("utf-8"\);\n\s*QTextCodec::setCodecForLocale\(codec\);\n/\n/g' \
  Business/Business_global.h \
  Business/business.cpp \
  RoboticsServiceProcess/main.cpp
```

### 2. regenerate protobuf

```bash
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService/PXREAService

cp -f linux_aarch64/PXREAService.pb.h linux_aarch64/PXREAService.pb.h.bak
cp -f linux_aarch64/PXREAService.pb.cc linux_aarch64/PXREAService.pb.cc.bak
cp -f linux_aarch64/PXREAService.grpc.pb.h linux_aarch64/PXREAService.grpc.pb.h.bak
cp -f linux_aarch64/PXREAService.grpc.pb.cc linux_aarch64/PXREAService.grpc.pb.cc.bak

protoc \
  -I . \
  -I /usr/include \
  --cpp_out=linux_aarch64 \
  --grpc_out=linux_aarch64 \
  --plugin=protoc-gen-grpc=$HOME/opt/grpc16-extract/usr/bin/grpc_cpp_plugin \
  PXREAService.proto
```

### 3. link local gRPC

```bash
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService

python3 - <<'PY'
from pathlib import Path

p = Path("PXREAGRPCServer/CMakeLists.txt")
text = p.read_text()

text = text.replace(
    """target_include_directories(PXREAGRPCServer PUBLIC
            ../Redistributable/linux_aarch64/grpc/include
            ../PXREAService/linux_aarch64
        )""",
    """target_include_directories(PXREAGRPCServer PUBLIC
            $ENV{HOME}/opt/grpc16-extract/usr/include
            ../PXREAService/linux_aarch64
        )""",
)

text = text.replace(
    "target_link_directories(PXREAGRPCServer PUBLIC ${PROJECT_SOURCE_DIR}/../Redistributable/linux_aarch64/grpc/lib)",
    "target_link_directories(PXREAGRPCServer PUBLIC $ENV{HOME}/opt/grpc16-extract/usr/lib/aarch64-linux-gnu)",
)

lines = text.splitlines()
out = []
i = 0

while i < len(lines):
    if lines[i].startswith("    target_link_libraries(PXREAGRPCServer PUBLIC"):
        out.extend([
            "    target_link_libraries(PXREAGRPCServer PUBLIC",
            "        grpc++",
            "        grpc++_reflection",
            "        grpc",
            "        gpr",
            "        protobuf",
            "        cares",
            "        ssl",
            "        crypto",
            "        z",
            "        Threads::Threads",
            "    )",
            "",
        ])
        i += 1
        while i < len(lines) and lines[i].strip() != ")":
            i += 1
        if i < len(lines) and lines[i].strip() == ")":
            i += 1
        continue
    out.append(lines[i])
    i += 1

p.write_text("\n".join(out) + "\n")
PY
```

## Build

```bash
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService

rm -rf build20
cmake \
  -S . \
  -B build20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$HOME/Qt6/6.7.3/gcc_arm64 \
  -DBUILD_LIB_PATH=$HOME/Qt6/6.7.3/gcc_arm64

cmake --build build20 -- -j"$(nproc)"
```

Check:

```bash
ls -l bin/RoboticsServiceProcess
```

Expected build line:

```text
[100%] Built target RoboticsServiceProcess
```

## Package

### 1. stage runtime libraries

```bash
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService
mkdir -p bin/lib

for lib in \
  libQt6Core.so libQt6Core.so.6 libQt6Core.so.6.7.3 \
  libQt6Network.so libQt6Network.so.6 libQt6Network.so.6.7.3 \
  libQt6DBus.so libQt6DBus.so.6 libQt6DBus.so.6.7.3 \
  libQt6Sql.so libQt6Sql.so.6 libQt6Sql.so.6.7.3
do
  [ -e "$HOME/Qt6/6.7.3/gcc_arm64/lib/$lib" ] && \
    cp -a "$HOME/Qt6/6.7.3/gcc_arm64/lib/$lib" bin/lib/
done

rm -f bin/lib/libicui18n.so.73 bin/lib/libicuuc.so.73 bin/lib/libicudata.so.73
cp -L "$HOME/Qt6/6.7.3/gcc_arm64/lib/libicui18n.so.73" bin/lib/libicui18n.so.73
cp -L "$HOME/Qt6/6.7.3/gcc_arm64/lib/libicuuc.so.73" bin/lib/libicuuc.so.73
cp -L "$HOME/Qt6/6.7.3/gcc_arm64/lib/libicudata.so.73" bin/lib/libicudata.so.73

for lib in \
  libgrpc++.so.1 libgrpc++.so.1.16.1 \
  libgrpc++_reflection.so.1 libgrpc++_reflection.so.1.16.1 \
  libgrpc.so.6 libgrpc.so.6.0.0 \
  libgpr.so.6 libgpr.so.6.0.0 \
  libssl.so.1.1 libcrypto.so.1.1 \
  libcares.so.2 libcares.so.2.3.0
do
  [ -e "$HOME/opt/grpc16-extract/usr/lib/aarch64-linux-gnu/$lib" ] && \
    cp -a "$HOME/opt/grpc16-extract/usr/lib/aarch64-linux-gnu/$lib" bin/lib/
done

[ -d "$HOME/Qt6/6.7.3/gcc_arm64/translations" ] && \
  cp -a "$HOME/Qt6/6.7.3/gcc_arm64/translations" bin/

mkdir -p bin/plugins
for d in generic networkinformation platforminputcontexts platforms sqldrivers tls; do
  [ -d "$HOME/Qt6/6.7.3/gcc_arm64/plugins/$d" ] && \
    cp -a "$HOME/Qt6/6.7.3/gcc_arm64/plugins/$d" bin/plugins/
done
```

Check:

```bash
LD_LIBRARY_PATH=$PWD/bin:$PWD/bin/lib \
  ldd bin/RoboticsServiceProcess | grep 'not found' || true
```

Expected: no output.

### 2. build deb

```bash
cd ~/src/XRoboToolkit-PC-Service-main/RoboticsService
rm -rf Package/debPackAArch64/package_arm64
bash Package/debPackAArch64/setup.sh
```

Check:

```bash
ls -lh Package/output/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb

dpkg-deb -c Package/output/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb | \
  egrep 'RoboticsServiceProcess|libicu|libgrpc|libQt6Core|libQt6Network|runService'
```

## Install

```bash
sudo apt install -y \
  ~/src/XRoboToolkit-PC-Service-main/RoboticsService/Package/output/XRoboToolkit-PC-Service_1.0.0.0_arm64.deb
```

Installed path:

```bash
/opt/apps/roboticsservice
```

## Validate

Check runtime dependencies:

```bash
cd /opt/apps/roboticsservice

LD_LIBRARY_PATH=$PWD:$PWD/lib:$PWD/SDK/arm64 \
  ldd ./RoboticsServiceProcess | grep 'not found' || true
```

Expected: no output.

Start service:

```bash
bash /opt/apps/roboticsservice/runService.sh
```

Expected log:

```text
Synchronous server. Num CQs: 1, Min pollers: 1, Max Pollers: 2
```

## Notes

- `Package/debPackAArch64/setup.sh` prints `cp: cannot stat ... RobotDemoQt`
  and `RobotDataRecorder` before the final `bin/*` copy. The final package
  still includes the files from `bin/`.
- The package intentionally uses Ubuntu 20.04 system `libprotobuf.so.17`.
- The upstream ARM64 `.deb` requires newer `glibc` / `libstdc++` and should not
  be used on Ubuntu 20.04.
