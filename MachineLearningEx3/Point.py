from numpy import double


class Point:
    def __init__(self, x: double, y: double, label: int):
        self.x = x
        self.y = y
        self.label = label

    def __str__(self):
        return "(%s,%s), %s" % (self.x, self.y, self.label)
        # Other statements outside the cla
