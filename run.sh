#!/usr/bin/env bash
set -euo pipefail

cc="${CC:-gcc}"
out="mnist"

"$cc" -std=c17 -Wall -Wextra -pedantic mnist.c tensor.c nn.c -o "$out" -lm
./"$out" "$@"
