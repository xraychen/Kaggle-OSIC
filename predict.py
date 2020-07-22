import os, pickle
import numpy as np

from train_osic import OsicModel, ImageDataset
from nets import *
from data_process import codec_fcv


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


def predict(test_pickle, test_csv, model_file, output_file):
    with open(test_csv) as f:
        content = f.read().splitlines()[1:]
        content = [e.split(',') for e in content]

    with open(test_pickle, 'rb') as f:
        images, images_id, x = pickle.load(f)

    test_set = ImageDataset(images, images_id, x, None, val_transform)

    model = OsicModel(net=NetSimple())
    model.load_checkpoint(model_file)

    y = model.predict(test_set, batch_size=16)
    y = codec_fcv(y, decode=True).astype(np.int16)

    patients_id = [e[0] for e in content]
    c = np.ones((len(content), 146), np.int16) * 70

    write_csv(patients_id, y, c, output_file)


if __name__ == '__main__':
    predict('input/test.pickle', 'raw/test.csv', 'model/test_01/e20_v273.8.pickle', 'output/test_01.csv')
