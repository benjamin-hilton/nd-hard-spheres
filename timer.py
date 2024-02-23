import time

times = {}

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if method.__name__ in times:
            times[method.__name__].append(te - ts)
        else:
            times[method.__name__] = [te - ts]

        #print '%r %f sec' % \
        #      (method.__name__, te-ts)
        return result

    return timed