import numpy as np
import math

''' some helper functions'''

def smap(f):
    '''mapping of functions for multiprocessing'''
    return f()

def choice(x,n):
    ''' used as numpy choice with uniform weights'''
    i=int(np.random.rand()*n)
    return x[i]

def init_array(v,size):
    '''
        return 1D numpy array of single value v and length=size
    '''
    #return np.array([v for i in range(size)])
    return np.tile(v, size)


def find_nearest_index(array,value):
    '''
        find the index of element in sorted array, where
        element is closest to value
    '''
    
    # find index where element should be inserted to maintain order.
    idx = np.searchsorted(array, value, side="left")
    
    # decide whether value is closer to idx or idx-1
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
    

def random_choice(array,cumsum=None):
    '''
        return one random sample elements in array, given cumulative sum of weights
    '''
    
    r=np.random.rand()
    i=find_nearest_index(cumsum,r)
    return array[i]
    