#!/bin/sh
for f in sofm.py classifier.py test_class.py; do
  echo "Running $f"
  ipython3 $f
done

