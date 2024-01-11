#!/usr/bin/env bash

PYCACHES=$(find . -type d -iname '__pycache__')
while IFS= read -r line; do rm -rvf "$line"; done <<< "$PYCACHES"
