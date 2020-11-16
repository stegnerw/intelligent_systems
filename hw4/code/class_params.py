###############################################################################
# Imports
###############################################################################
from settings import *
import pathlib
import csv


# Save parameters to CSV
print('Saving training parameters')
csv_rows = list()
csv_rows.append(['Parameter', 'Clean', 'Noisy', 'Description'])
csv_rows.append(['$hidden\\_layer\\_size$', str(HIDDEN_LAYER_SIZES[0]),
    str(HIDDEN_LAYER_SIZES[0]), 'Neurons in hidden layer'])
csv_rows.append(['$\\eta$', str(CLASS_CLEAN_ETA), str(CLASS_NOISY_ETA),
    'Learning rate'])
csv_rows.append(['$\\alpha$', str(CLASS_CLEAN_ALPHA), str(CLASS_NOISY_ALPHA),
    'Momentum'])
csv_rows.append(['$\\lambda$', str(CLASS_CLEAN_DECAY), str(CLASS_NOISY_DECAY),
    'Weight decay'])
csv_rows.append(['$max\\_epochs$', str(CLASS_CLEAN_MAX_EPOCHS),
    str(CLASS_NOISY_MAX_EPOCHS), 'Maximum training epochs'])
csv_rows.append(['$L$', str(CLASS_CLEAN_L), str(CLASS_NOISY_L),
    'Lower activation threshold'])
csv_rows.append(['$H$', str(CLASS_CLEAN_H), str(CLASS_NOISY_H),
    'Upper activation threshold'])
csv_rows.append(['$patience$', str(CLASS_CLEAN_PATIENCE),
    str(CLASS_NOISY_PATIENCE), 'Patience before early stopping'])
csv_rows.append(['$es\\_delta$', str(CLASS_CLEAN_ES_DELTA),
    str(CLASS_CLEAN_ES_DELTA), 'Delta value for early stopping'])
with open(str(DATA_DIR.joinpath('class_params.csv')), 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(csv_rows)

