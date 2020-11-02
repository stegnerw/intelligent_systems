#!/bin/sh
for f in test_classifier.py test_autoencoder.py features.py; do
  echo "Running $f"
  ipython3 $f
done

