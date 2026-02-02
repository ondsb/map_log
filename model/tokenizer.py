import math


class Tokenizer:
    enc: dict
    dec: dict

    def __init__(self, meta):
        self.enc = meta["enc"]
        self.dec = meta["dec"]

    def encode(self, data):
        data_txt = [
            [self.enc[x] if not self.is_number(x) else self.num for x in y]
            for y in data
        ]
        data_num = [[float(x) if self.is_number(x) else 1 for x in y] for y in data]

        return data_txt, data_num

    def decode(self, y_t, y_n):
        out = []
        for i in range(len(y_t)):
            if y_t[i] != self.num:
                out.append(self.dec[y_t[i]])
            else:
                out.append(str(y_n[i]))
        return " ".join(out)

    @staticmethod
    def is_number(s):
        try:
            return math.isfinite(float(s))
        except ValueError:
            return False

    @property
    def pad(self):
        return self.enc["<|pad|>"]

    @property
    def eot(self):
        return self.enc["<|endoftext|>"]

    @property
    def num(self):
        return self.enc["<|num|>"]

    @property
    def side_light(self):
        return self.enc["light"]

    @property
    def side_dark(self):
        return self.enc["dark"]

    @property
    def kill(self):
        return self.enc["killed"]
