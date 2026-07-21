# Migration Notes

This repository uses a hard cutover to `src/tactifoot_vision`.

Use:

- Python API from `tactifoot_vision.*`.
- CLI entrypoint `tactifoot`.
- YAML configs under `configs/`.

Do not use old root packages or scripts for production runtime. They are archived
under `legacy/` for reference while feature parity is completed.

No compatibility API is guaranteed for the legacy layout.
