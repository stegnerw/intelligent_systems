#!/bin/sh
for f in classifier.py test_class.py; do
  echo "Running $f"
  ipython3 $f
done

