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


if __name__ == "__main__":
    print TensorOuter([np.arange(2)+1, np.arange(3)+10, np.arange(4)+100])