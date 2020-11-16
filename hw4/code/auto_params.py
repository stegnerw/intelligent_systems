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
csv_rows.append(['$\\eta$', str(AUTO_CLEAN_ETA), str(AUTO_NOISY_ETA),
    'Learning rate'])
csv_rows.append(['$\\alpha$', str(AUTO_CLEAN_ALPHA), str(AUTO_NOISY_ALPHA),
    'Momentum'])
csv_rows.append(['$\\lambda$', str(AUTO_CLEAN_DECAY), str(AUTO_NOISY_DECAY),
    'Weight decay'])
csv_rows.append(['$max\\_epochs$', str(AUTO_CLEAN_MAX_EPOCHS),
    str(AUTO_NOISY_MAX_EPOCHS), 'Maximum training epochs'])
csv_rows.append(['$patience$', str(AUTO_CLEAN_PATIENCE),
    str(AUTO_NOISY_PATIENCE), 'Patience before early stopping'])
csv_rows.append(['$es\\_delta$', str(AUTO_CLEAN_ES_DELTA),
    str(AUTO_CLEAN_ES_DELTA), 'Delta value for early stopping'])
with open(str(DATA_DIR.joinpath('auto_params.csv')), 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(csv_rows)

