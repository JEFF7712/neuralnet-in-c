#!/usr/bin/env bash
set -euo pipefail

cc="${CC:-gcc}"
out="tensor_demo"

"$cc" -std=c17 -Wall -Wextra -pedantic main.c tensor.c -o "$out"
./"$out"
