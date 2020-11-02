#!/bin/sh
for f in classifier.py autoencoder.py; do
  echo "Running $f"
  ipython3 $f
done

