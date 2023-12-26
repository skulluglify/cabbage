#!/usr/bin/env bash

if [ -f 'projekt.zip' ]; then
  rm projekt.zip
fi

/usr/bin/zip -r0q9yuo projekt.zip .
