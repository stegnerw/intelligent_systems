#!/bin/sh
for f in auto_params.py autoencoder.py test_autoencoder_clean.py autoencoder_noisy.py test_autoencoder_noisy.py features.py; do
  echo "Running $f"
  ipython3 $f
done

