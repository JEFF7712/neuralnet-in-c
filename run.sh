#!/usr/bin/env bash
set -euo pipefail

cc="${CC:-gcc}"
out="main"

"$cc" -std=c17 -Wall -Wextra -pedantic main.c tensor.c nn.c -o "$out" -lm
./"$out"
