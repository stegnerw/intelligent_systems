###############################################################################
# Imports
###############################################################################
from settings import *
import pathlib
import csv


# Save parameters to CSV
print('Saving training parameters')
csv_rows = list()
csv_rows.append(['Parameter', 'Value', 'Description'])
csv_rows.append(['$\\eta$', str(CLASS_ETA),
    'Learning rate'])
csv_rows.append(['$\\alpha$', str(CLASS_ALPHA),
    'Momentum'])
csv_rows.append(['$\\lambda$', str(CLASS_DECAY),
    'Weight decay'])
csv_rows.append(['$max\\_epochs$', str(CLASS_MAX_EPOCHS),
    'Maximum training epochs'])
csv_rows.append(['$L$', str(CLASS_L),
    'Lower activation threshold'])
csv_rows.append(['$H$', str(CLASS_H),
    'Upper activation threshold'])
csv_rows.append(['$patience$', str(CLASS_PATIENCE),
    'Patience before early stopping'])
csv_rows.append(['$es\\_delta$', str(CLASS_ES_DELTA),
    'Delta value for early stopping'])
with open(str(DATA_DIR.joinpath('class_params.csv')), 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(csv_rows)

