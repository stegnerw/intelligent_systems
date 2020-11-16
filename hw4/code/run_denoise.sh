#!/bin/sh
for f in auto_params.py autoencoder_noisy.py test_autoencoder_noisy.py features.py classifier_noisy.py test_classifier_noisy.py; do
  echo "Running $f"
  ipython3 $f
done

