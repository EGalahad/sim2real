# Project Agent Notes

## Python Environments

- Inference / policy / sim work on both PC and onboard Orin uses the root project.
- Teleop work on both PC and onboard Orin uses `venv/teleop`.
- When touching inference/runtime files, verify syntax with the root project when available:
  `uv run python -m py_compile <files>`
- When touching teleop files, verify syntax with the teleop project when available:
  `uv --project venv/teleop run python -m py_compile <files>`

## Documentation Sync

- Keep English and Chinese documentation in sync.
- When changing a doc page, update the corresponding English and Chinese sources in the same change whenever both versions exist.
- For Docusaurus content, treat `docs/` and `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/` as paired sources when both files exist.
- Apply the same sync rule to top-level docs such as `README.md` and `README_zh.md`.

## Runtime Artifact Google Drive Sync

- Runtime artifacts live under the shared Google Drive folder
  `gdrive:sim2real`.
- Available rclone remotes on this machine include `haoyang:`, `gear:`, and
  `gdrive:`. Use `gdrive:` for Google Drive artifact sync.
- Useful commands:
  - `rclone lsd gdrive:`
  - `rclone ls gdrive:sim2real`
  - `rclone copy local_dir gdrive:sim2real/some_folder`
  - `rclone copy gdrive:sim2real/some_folder local_dir`
  - `rclone sync local_dir gdrive:sim2real/some_folder`
- For `checkpoints/`, sync only deploy artifacts converted from other
  codebases. Do not upload checkpoints produced by the user's own training runs
  unless the user explicitly requests it.
- For `third_party/`, avoid syncing large high-file-count directories directly.
  If a directory contains many files, such as `third_party/prebuilt/...`, package
  that directory as a zip and upload the zip to Drive instead. Preserve enough
  path context in the zip name or parent folder so it can be restored to the
  original local layout.
- Never paste rclone OAuth tokens or Google Drive access tokens into chat,
  commits, docs, or logs. The local rclone config is stored under
  `~/.config/rclone/rclone.conf`; if a token is exposed, revoke the rclone Drive
  access in Google Account settings and re-authorize locally.
