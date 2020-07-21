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

        for w in len(y):
            for i, p in enumerate(patients_id):
                f.write('{}_{},{},{}\n'.format(p, w - 12, y[p][w], c[p][w]))


def predict(csv_file, image_dir, output_file):
    with open(csv_file) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    patients_id = [e[0] for e in content]
    y = np.ones((len(content), 146), np.int16) * 2000
    c = np.ones((len(content), 146), np.int16) * 70

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict('raw/test.csv', 'raw/test', 'output/_.csv')

