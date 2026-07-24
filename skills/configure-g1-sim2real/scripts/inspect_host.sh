#!/usr/bin/env bash

set -u

repo_path="${1:-$HOME/sim2real}"

value() {
    printf '%-28s %s\n' "$1" "$2"
}

uv_path="$(command -v uv 2>/dev/null || true)"
if [ -z "$uv_path" ] && [ -x "$HOME/.local/bin/uv" ]; then
    uv_path="$HOME/.local/bin/uv"
fi

os_name="unknown"
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    os_name="${PRETTY_NAME:-unknown}"
fi

value "hostname" "$(hostname)"
value "os" "$os_name"
value "architecture" "$(uname -m)"
value "kernel" "$(uname -r)"
value "python3" "$(python3 --version 2>&1 || true)"
value "uv.path" "${uv_path:-not found}"
value "uv" "$([ -n "$uv_path" ] && "$uv_path" --version 2>&1 || echo unavailable)"
value "repo" "$repo_path"
value "repo.present" "$([ -d "$repo_path" ] && echo yes || echo no)"
value "jetpack.release" "$([ -r /etc/nv_tegra_release ] && head -n 1 /etc/nv_tegra_release || echo unavailable)"
value "cyclonedds.install" "$([ -d "$HOME/cyclonedds/install" ] && echo present || echo missing)"
value "root.venv" "$([ -d "$repo_path/.venv" ] && echo present || echo missing)"
value "pico.venv" "$([ -d "$repo_path/venv/pico/.venv" ] && echo present || echo missing)"
value "dh116s.venv" "$([ -d "$repo_path/venv/dh116s/.venv" ] && echo present || echo missing)"
value "CYCLONEDDS_HOME" "${CYCLONEDDS_HOME:-unset}"
value "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH:-unset}"
value "HF_HUB_OFFLINE" "${HF_HUB_OFFLINE:-unset}"
value "HF_ENDPOINT" "${HF_ENDPOINT:-unset}"

printf '\n[filesystems]\n'
df -h "$repo_path" 2>/dev/null || df -h "$HOME"

printf '\n[interfaces]\n'
ip -br addr 2>/dev/null || true

printf '\n[fresh login shell]\n'
bash -lc '
printf "%-28s %s\n" "uv.path" "$(command -v uv 2>/dev/null || echo not-found)"
printf "%-28s %s\n" "CYCLONEDDS_HOME" "${CYCLONEDDS_HOME:-unset}"
printf "%-28s %s\n" "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH:-unset}"
printf "%-28s %s\n" "HF_HUB_OFFLINE" "${HF_HUB_OFFLINE:-unset}"
printf "%-28s %s\n" "HF_ENDPOINT" "${HF_ENDPOINT:-unset}"
'
