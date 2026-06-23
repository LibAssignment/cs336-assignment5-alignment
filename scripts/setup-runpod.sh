#!/bin/bash

export-env() {
  echo "export UV_CACHE_DIR=/root/.uv/cache"
  echo 'export PATH=/root/.local/bin:$PATH'
  echo 'export UV_PROJECT_ENVIRONMENT=/tmp/.venv'
}

export-env >> ~/.profile
export-env >> ~/.bashrc

uv tool install uv
mkdir /tmp/.venv
