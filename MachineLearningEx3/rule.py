class rule:

    def __init__(self, m, b, alpha, sighUp, trueError):
        self.m = m
        self.b = b
        self.alpha = alpha
        self.sighUp = sighUp
        self.trueError = trueError

    def __str__(self):
        return "(%s,%s), %s, %s" % (self.m, self.b, self.alpha, self.trueError)
        # Other statements outside the cla
