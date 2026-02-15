#!/usr/bin/env bash

# Choose a Python launcher that exists on the current platform.
# Priority: explicit PYTHON_BIN -> python -> python3 -> py -3 (Windows launcher).
resolve_python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    echo "$PYTHON_BIN"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v py >/dev/null 2>&1; then
    echo "py -3"
    return 0
  fi
  return 1
}

require_python_bin() {
  local py
  py="$(resolve_python_bin)" || {
    echo "[err] Python not found. Set PYTHON_BIN or install python/python3."
    exit 1
  }
  printf '%s\n' "$py"
}
