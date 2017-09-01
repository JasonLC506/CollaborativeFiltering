import numpy as np
import copy

## extension of numpy.outer() to unlimited number of vectors,
## result[i,j,k] = v1[i] * v2[j] * v3[k] ##
def TensorOuter(vector_list):
    L = len(vector_list)
    dim = [vector_list[i].shape[0] for i in range(L)]
    result = copy.deepcopy(vector_list[0])
    previous_size = 1
    for i in range(1, L):
        previous_size = previous_size * dim[i-1]
        result = np.outer(result,vector_list[i])
        result = result.reshape(previous_size * dim[i])
    result = result.reshape(dim)
    return result

def TensorOuterFull(array_list):
    L = len(array_list)
    shapes = [array_list[i].shape for i in range(L)]
    sizes_flat = [_prod(shapes[i]) for i in range(L)]
    result = copy.deepcopy(array_list[0].reshape(sizes_flat[0]))
    previous_size = 1
    for i in range(1,L):
        previous_size = previous_size * sizes_flat[i-1]
        result = np.outer(result, array_list[i].reshape(sizes_flat[i])).reshape(previous_size * sizes_flat[i])
    return result.reshape(_concatenate(shapes))

def _concatenate(array_list):
    result = np.array([],dtype=np.int64)
    for array in array_list:
        result = np.concatenate((result, array), axis=0)
    return result

def _prod(array):
    result = 1
    for value in array:
        result *= value
    return result

## extention of numpy.multiply (c[i,j] = a[i,j]*b[j]) to c[i,j] = a[i,j]*b[i] ##
def transMultiply(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError("the first axis does not match")
    return np.transpose(np.multiply(np.transpose(a), b))


if __name__ == "__main__":
    a = np.arange(6).reshape([2, 3])
    b = np.arange(4) * 1.1
    c = np.arange(5) * 2.0
    print TensorOuter([np.arange(2)+1, np.arange(3)+10, np.arange(4)+100])
    print TensorOuterFull([np.arange(2)+1, np.arange(3)+10, np.arange(4)+100])
    print TensorOuterFull([b,a,c])
    # print transMultiply(np.arange(6).reshape([2,3]), np.arange(2))
