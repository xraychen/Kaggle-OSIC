# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Input data files are available in the read-only "../input/" directory
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

TRAIN_CSV = '/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv'
TEST_CSV = '/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv'


def test():
    with open(TEST_CSV) as f:
        content = f.read().splitlines()

    for line in content:
        print(line)


if __name__ == '__main__':
    test()
