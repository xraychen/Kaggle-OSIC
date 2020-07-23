X_LENGTH = 161
Y_LENGTH = 146
Y_OFFSET = -12

FCV_MEAN = 2690.479018721756
FCV_STD = 832.5021066817238
PERCENT_MEAN = 77.67265350296326
PERCENT_STD = 19.81686156299212


class Codec:
    def __init__(self, tag='fcv'):
        if tag == 'fcv':
            self.mean = FCV_MEAN
            self.std = FCV_STD
        elif tag == 'percent':
            self.mean = PERCENT_MEAN
            self.std = PERCENT_STD
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


codec_f = Codec(tag='fcv')
codec_p = Codec(tag='percent')
