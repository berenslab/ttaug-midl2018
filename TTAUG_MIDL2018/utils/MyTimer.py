import time


class MyTimer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        # print('Elapsed: %s' % (time.time() - self.tstart))
        print('Elapsed time : %s' % (time.strftime("%j days and %H:%M:%S", time.gmtime(time.time()-self.tstart))))
