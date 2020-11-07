#!/bin/sh
for f in classifier.py test_classifier_clean.py classifier_noisy.py test_classifier_noisy.py; do
  echo "Running $f"
  ipython3 $f
done

