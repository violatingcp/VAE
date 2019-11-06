import numpy as np 

def slice_it(li, cols=2):
    """Slicing list"""
    start = 0
    for i in xrange(cols):
        stop = start + len(li[i::cols])
        yield li[start:stop]
        start = stop
