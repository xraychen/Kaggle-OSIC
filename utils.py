import os


WEEK    = (31.861846352485475, 23.240045178171002)
FVC     = (2690.479018721756, 832.5021066817238)
PERCENT = (77.67265350296326, 19.81686156299212)
AGE     = (67.18850871530019, 7.055116199848975)
IMAGE   = (615.48615, 483.8854)

def make_dir(file_path):
    dirname = os.path.dirname(file_path)
    try:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    except FileNotFoundError:
        pass


class Codec:
    def __init__(self, tag='fvc'):
        if tag == 'week':
            self.mean, self.std = WEEK
        elif tag == 'fvc':
            self.mean, self.std = FVC
        elif tag == 'percent':
            self.mean, self.std = PERCENT
        elif tag == 'age':
            self.mean, self.std = AGE
        elif tag == 'image':
            self.mean, self.std = IMAGE
        else:
            raise KeyError

    def encode(self, value, scale_only=False):
        value = float(value) if type(value) == str else value
        if scale_only:
            return value / self.std
        else:
            return (value - self.mean) / self.std

    def decode(self, value, scale_only=False):
        value = float(value) if type(value) == str else value
        if scale_only:
            return value * self.std
        else:
            return value * self.std + self.mean


codec_w = Codec(tag='week')
codec_f = Codec(tag='fvc')
codec_p = Codec(tag='percent')
codec_a = Codec(tag='age')
codec_i = Codec(tag='image')
