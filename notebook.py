# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Input data files are available in the read-only "../input/" directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

TEST_CSV = '/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv'
TEST_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'

SUBMIT_CSV = '/kaggle/working/submission.csv'


import os
import numpy as np


def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


def write_csv(patients_id, y, c, output_file):
    make_dir(output_file)
    with open(output_file, 'w') as f:
        f.write('Patient_Week,FVC,Confidence\n')

        for w in range(146):
            for i, p in enumerate(patients_id):
                f.write('{}_{},{},{}\n'.format(p, w - 12, y[i][w], c[i][w]))


def predict(csv_file, image_dir, output_file):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    patients_id = [e[0] for e in content]
    y = np.ones((len(content), 146), np.int16) * 2000
    c = np.ones((len(content), 146), np.int16) * 70

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict(TEST_CSV, TEST_DIR, SUBMIT_CSV)


