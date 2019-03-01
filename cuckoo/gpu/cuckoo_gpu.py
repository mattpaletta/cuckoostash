import random
from typing import List
import pycuda.driver as cuda
from pycuda import compiler, gpuarray
import numpy as np

# -- initialize the device
import pycuda.autoinit

# Inheritance for using thread
class GPUThread(object):
    def __init__(self, number, arr):
        print("Init devices")
        # threading.Thread.__init__(self)
        self._number = number
        self._arr = arr

        # self._dev = cuda.Device(self._number)
        self._ctx: cuda.Context = pycuda.autoinit.make_default_context()


        # initialize gpu array and copy from cpu to gpu.
        self._cukoo_gpu = gpuarray.to_gpu(self._arr)
        self._functions = {}
        self._get_cuda_functions()

    def get(self, keys):
        return self._run_function("_get_helper", keys)

    def set(self, keys, values):
        return self._run_function("set", keys, values)

    def _run_function(self, func_name: str, keys: List[int], values: List[int] = None):
        # Get lock to synchronize threads
        output = gpuarray.empty(keys, np.int64)

        params = {
            "grid": (1, 1, 1),
            "block": (len(keys), len(keys), 1)
        }

        my_func = self._functions.get(func_name)
        if values is None:
            my_func(keys, output, **params)
        else:
            my_func(keys, values, output, **params)
        # self._ctx.pop()
        return output

    def __del__(self):
        # delete device,context for saving resources.
        # del self._ctx
        # del self._cukoo_gpu
        pass

    def _get_stash_function(self, stash_size: int):
        p = 334_214_459
        return self._get_next_hash_function(a = random.randint(a = 1, b = p),
                                            b = random.randint(a = 0, b = p),
                                            p = p,
                                            s_t = stash_size,
                                            function_name = "stash_hash_function",
                                            function = "linear")

    def _get_next_hash_function(self, a, b, p, s_t, function_name: str, function: str):
        def linear():
            return """__device__ void ${FUNCTION_NAME}%(int key) {
                       return ((%{VAR_A}% * key) + %{VAR_B}%) % %{VAR_P}%) % %{VAR_ST}%;
                   }""".format(
                FUNCTION_NAME = function_name,
                VAR_A = a,
                VAR_B = b,
                VAR_P = p,
                VAR_ST = s_t
            )

        def xor():
            return """__device__ void ${FUNCTION_NAME}%(int key) {
                       return ((%{VAR_A}% ^ key) + %{VAR_B}%) % %{VAR_P}%) % %{VAR_ST}%;
                   }""".format(
                FUNCTION_NAME = function_name,
                VAR_A = a,
                VAR_B = b,
                VAR_P = p,
                VAR_ST = s_t
            )

        if function == "linear":
            return linear()
        elif function == "xor":
            return xor()
        else:
            raise NotImplementedError("Invalid hash function: " + str(function))

    def _get_cuda_functions(self):
        # define size of blocks and tiles sub-matrix
        # (we assume that the block size is same as tile size)
        TILE_SIZE = 20
        BLOCK_SIZE = TILE_SIZE
        MATRIX_SIZE = 4000
        CUCKOO_SIZE = 4000
        STASH_SIZE = 101

        # _get_helper the kernel code from the template
        with open("cuckoo/gpu/get_value.cu", "r") as f:
            kernel_code_template = "\n".join(f.readlines())

        # by specifying the constants MATRIX_SIZE and BLOCK_SIZE
        kernel_code = kernel_code_template % {
            'MATRIX_SIZE': MATRIX_SIZE,
            'BLOCK_SIZE' : BLOCK_SIZE,
            'CUCKOO_SIZE': CUCKOO_SIZE,
            'STASH_SIZE': STASH_SIZE,
        }

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # # create empty gpu array for the result (C = A * B)
        # c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

        # _get_helper the kernel function from the compiled module
        for func in ["_get_helper", "set"]:
            self._functions.update({func: mod.get_function(func)})


if __name__ == "__main__":
    num = cuda.Device.count()

    gpu_thread_list = []
    num_elements = 100_000
    some_list = []
    for i in range(1):
        some_list.append(np.random.randn(num_elements, 1).astype(np.int64))

    print("Preparing values")
    keys = list(range(num_elements + 1))
    values = list(map(lambda x: x + 100, keys))

    # threadLock = threading.Lock()
    for i, arr in enumerate(some_list):
        gpu_thread = GPUThread(i, arr)
        gpu_thread.set(keys, values)
        gpu_thread_list.append(gpu_thread)
