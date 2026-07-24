#!/usr/bin/env bash
set -euo pipefail

SDK_VERSION="LHandProLib-API-Linux-20260325"
SOURCE_REPO="Roboparty/RP_teleoperate_ygx"
SOURCE_REL="assets/DH116S_hand/${SDK_VERSION}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor/${SDK_VERSION}"
PYTHON_DIR="${SCRIPT_DIR}/python/lhandprolib_python_sdk"
COMPAT_DIR="${SCRIPT_DIR}/compat"

ARCH="auto"
SOURCE_DIR=""
FORCE=0
WITH_PYTHON_DEPS=0
PYTHON_BIN="python3"

usage() {
    printf '%s\n' \
        "Install the DH116S LHandProLib SDK inside the sim2real repository." \
        "" \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --arch auto|aarch64|x86_64|i386  Target architecture (default: auto)." \
        "  --source DIR                     Use an existing SDK/repository directory." \
        "  --force                          Replace the cached SDK and Python install." \
        "  --with-python-deps               Install python-can into --python." \
        "  --python PATH                    Python interpreter for checks/dependencies." \
        "  -h, --help                       Show this help."
}

while (($#)); do
    case "$1" in
        --arch)
            ARCH="${2:?--arch requires a value}"
            shift 2
            ;;
        --source)
            SOURCE_DIR="${2:?--source requires a directory}"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --with-python-deps)
            WITH_PYTHON_DEPS=1
            shift
            ;;
        --python)
            PYTHON_BIN="${2:?--python requires an interpreter path}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${ARCH}" == "auto" ]]; then
    case "$(uname -m)" in
        aarch64|arm64) ARCH="aarch64" ;;
        x86_64|amd64) ARCH="x86_64" ;;
        i386|i486|i586|i686) ARCH="i386" ;;
        *)
            printf 'Unsupported host architecture: %s\n' "$(uname -m)" >&2
            exit 1
            ;;
    esac
fi
case "${ARCH}" in
    aarch64|x86_64|i386) ;;
    *)
        printf 'Unsupported SDK architecture: %s\n' "${ARCH}" >&2
        exit 1
        ;;
esac

if [[ "${FORCE}" -eq 1 ]]; then
    rm -rf -- "${VENDOR_DIR}" "${PYTHON_DIR}"
fi

if [[ ! -d "${VENDOR_DIR}/${ARCH}" ]]; then
    mkdir -p "$(dirname -- "${VENDOR_DIR}")"
    if [[ -n "${SOURCE_DIR}" ]]; then
        SOURCE_DIR="$(cd -- "${SOURCE_DIR}" && pwd)"
        if [[ -d "${SOURCE_DIR}/${SOURCE_REL}/${ARCH}" ]]; then
            RESOLVED_SOURCE="${SOURCE_DIR}/${SOURCE_REL}"
        elif [[ -d "${SOURCE_DIR}/${SDK_VERSION}/${ARCH}" ]]; then
            RESOLVED_SOURCE="${SOURCE_DIR}/${SDK_VERSION}"
        elif [[ -d "${SOURCE_DIR}/${ARCH}" ]]; then
            RESOLVED_SOURCE="${SOURCE_DIR}"
        else
            printf 'Could not find %s/%s under %s\n' "${SDK_VERSION}" "${ARCH}" "${SOURCE_DIR}" >&2
            exit 1
        fi
        cp -a -- "${RESOLVED_SOURCE}" "${VENDOR_DIR}"
    else
        command -v gh >/dev/null || {
            printf 'gh is required for the default private-repository download.\n' >&2
            exit 1
        }
        gh auth status >/dev/null 2>&1 || {
            printf 'Authenticate first with: gh auth login\n' >&2
            exit 1
        }
        DOWNLOAD_DIR="$(mktemp -d)"
        trap 'rm -rf -- "${DOWNLOAD_DIR}"' EXIT
        gh repo clone "${SOURCE_REPO}" "${DOWNLOAD_DIR}/source" -- \
            --depth 1 --filter=blob:none --sparse
        git -C "${DOWNLOAD_DIR}/source" sparse-checkout set "${SOURCE_REL}"
        cp -a -- "${DOWNLOAD_DIR}/source/${SOURCE_REL}" "${VENDOR_DIR}"
    fi
fi

UPSTREAM_PYTHON="${VENDOR_DIR}/${ARCH}/share/LHandProLib/examples/CANFD_python"
UPSTREAM_LIBRARY="${VENDOR_DIR}/${ARCH}/lib/libLHandProLib.so"
for required in \
    "${UPSTREAM_PYTHON}/canfd_lib.py" \
    "${UPSTREAM_PYTHON}/lhandpro_controller.py" \
    "${UPSTREAM_PYTHON}/lhandprolib_loader.py" \
    "${UPSTREAM_PYTHON}/lhandprolib_wrapper.py" \
    "${UPSTREAM_LIBRARY}"; do
    if [[ ! -f "${required}" ]]; then
        printf 'SDK payload is incomplete; missing %s\n' "${required}" >&2
        exit 1
    fi
done

mkdir -p "${PYTHON_DIR}/lib"
install -m 0644 "${UPSTREAM_PYTHON}/canfd_lib.py" "${PYTHON_DIR}/canfd_lib.py"
install -m 0644 "${UPSTREAM_PYTHON}/lhandpro_controller.py" "${PYTHON_DIR}/lhandpro_controller.py"
install -m 0644 "${UPSTREAM_PYTHON}/lhandprolib_loader.py" "${PYTHON_DIR}/lhandprolib_loader.py"
install -m 0644 "${UPSTREAM_PYTHON}/lhandprolib_wrapper.py" "${PYTHON_DIR}/lhandprolib_wrapper.py"
install -m 0644 "${UPSTREAM_LIBRARY}" "${PYTHON_DIR}/lib/libLHandProLib.so"
install -m 0644 "${COMPAT_DIR}/__init__.py" "${PYTHON_DIR}/__init__.py"
install -m 0644 "${COMPAT_DIR}/config.py" "${PYTHON_DIR}/config.py"
install -m 0644 "${COMPAT_DIR}/controller.py" "${PYTHON_DIR}/controller.py"

sed -i \
    -e 's/^from lhandprolib_wrapper import /from .lhandprolib_wrapper import /' \
    -e 's/^from canfd_lib import /from .canfd_lib import /' \
    -e 's/^from config import /from .config import /' \
    -e 's/^            from canfd_lib import /            from .canfd_lib import /' \
    "${PYTHON_DIR}/lhandpro_controller.py"
sed -i \
    -e 's/^from lhandprolib_loader import (/from .lhandprolib_loader import (/' \
    "${PYTHON_DIR}/lhandprolib_wrapper.py"
sed -i \
    -e 's/os.listdir("\/sys\/class\/net")/sorted(os.listdir("\/sys\/class\/net"))/g' \
    "${PYTHON_DIR}/canfd_lib.py"

printf '%s\n' "${SDK_VERSION}" > "${SCRIPT_DIR}/python/SDK_VERSION"
printf '%s\n' "${ARCH}" > "${SCRIPT_DIR}/python/SDK_ARCH"

"${PYTHON_BIN}" -m py_compile "${PYTHON_DIR}"/*.py
"${PYTHON_BIN}" -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' "${PYTHON_DIR}/lib/libLHandProLib.so"

if [[ "${WITH_PYTHON_DEPS}" -eq 1 ]]; then
    "${PYTHON_BIN}" -m pip install 'python-can~=4.6.1'
fi

printf 'Installed %s (%s) at %s\n' "${SDK_VERSION}" "${ARCH}" "${SCRIPT_DIR}/python"
printf 'Runtime dependency: python-can~=4.6.1\n'
