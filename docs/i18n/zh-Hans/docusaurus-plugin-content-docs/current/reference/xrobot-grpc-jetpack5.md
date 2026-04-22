---
title: XRobot gRPC JetPack 5
slug: /reference/xrobot-grpc-jetpack5
---

# XRobot gRPC for JetPack 5

This note builds JetPack 5 compatible gRPC dependencies for
`XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK`.

## Why

The upstream `Redistributable/linux_aarch64/grpc` package can require newer
glibc/libstdc++ symbols than JetPack 5 provides, for example
`__libc_single_threaded`. Build the dependency set directly on the Orin to
match JetPack 5 / Ubuntu 20.04.

## Versions

- gRPC: `v1.66.0`
- Protobuf: `27.2.0`
- Abseil: `20240116`
- Build host: JetPack 5 / L4T R35 / Ubuntu 20.04 / aarch64

## Build On Orin

Run this on `g1-rp`.

```bash
source ~/.bashrc
proxy_on
```

```bash
export work_dir=/tmp/xrobot-grpc-build
export install_dir=/tmp/xrobot-grpc-install
```

```bash
rm -rf "$work_dir" "$install_dir"
mkdir -p "$work_dir"
git clone --depth 1 --branch v1.66.0 https://github.com/grpc/grpc "$work_dir/grpc"
cd "$work_dir/grpc"
```

```bash
git config --global http.version HTTP/1.1
git submodule update --init --jobs 1 \
  third_party/abseil-cpp \
  third_party/protobuf \
  third_party/re2 \
  third_party/zlib \
  third_party/cares/cares \
  third_party/boringssl-with-bazel
```

```bash
cmake -S . -B cmake/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DBUILD_SHARED_LIBS=OFF \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=ON \
  -DgRPC_ABSL_PROVIDER=module \
  -DgRPC_CARES_PROVIDER=module \
  -DgRPC_PROTOBUF_PROVIDER=module \
  -DgRPC_RE2_PROVIDER=module \
  -DgRPC_SSL_PROVIDER=module \
  -DgRPC_ZLIB_PROVIDER=module
```

```bash
cmake --build cmake/build --target install -- -j8
```

## Package For Local Reuse

Run this on `g1-rp` after the build succeeds.

```bash
cd "$install_dir"
tar -czf /tmp/xrobot-grpc-jetpack5-aarch64.tar.gz .
```

Copy the archive to the development machine.

```bash
scp g1-rp:/tmp/xrobot-grpc-jetpack5-aarch64.tar.gz /tmp/
```

Unpack it into a local directory on the development machine.

```bash
rm -rf /tmp/xrobot-grpc-jetpack5-aarch64/grpc
mkdir -p /tmp/xrobot-grpc-jetpack5-aarch64/grpc
tar -xzf /tmp/xrobot-grpc-jetpack5-aarch64.tar.gz \
  -C /tmp/xrobot-grpc-jetpack5-aarch64/grpc
```

You can keep this extracted directory anywhere convenient; do not assume a
`prebuilt/` path is versioned in git.

## Use The Prepared Package

Run this on onboard Orin before building `PXREARobotSDK`.

```bash
export xrobot_root=external/XRoboToolkit-PC-Service
export sdk_grpc="$xrobot_root/RoboticsService/Redistributable/linux_aarch64/grpc"
export local_grpc="prebuilt/jetpack5-aarch64/xrobot-grpc"
```

```bash
rm -rf "$sdk_grpc.upstream"
mv "$sdk_grpc" "$sdk_grpc.upstream"
cp -a "$local_grpc" "$sdk_grpc"
```

Then build the SDK.

```bash
(cd "$xrobot_root/RoboticsService/PXREARobotSDK" && bash build.sh)
```
