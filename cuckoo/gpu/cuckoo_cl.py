import os
import random
from typing import List

import numpy as np
import pyopencl as cl

# Inheritance for using thread
# from coriander.coriander import cu_to_cl

class GPUThread(object):
    def __init__(self, number, arr):
        print("Init devices")

        # threading.Thread.__init__(self)
        self._number = number
        self._arr = arr

        platforms = cl.get_platforms()
        devices = platforms[0].get_devices()
        print(devices)

        # Choose the CPU opencl or GPU opencl.
        self._ctx = cl.Context(devices = [devices[-1]])
        print(self._ctx.devices)
        self._cl_queue = cl.CommandQueue(self._ctx)
        self._functions = {}
        self._get_cuda_functions()

    def get(self, keys):
        return self._run_function("_get_helper", keys)

    def set(self, keys, values):
        return self._run_function("set", keys, values)

    def _run_function(self, func_name: str, keys: List[int], values: List[int] = None):
        my_func = self._functions.get(func_name)
        if values is None:
            return my_func(keys)
        else:
            return my_func(keys, values)

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
        kernelsource = """
        __kernel void vadd(
            __global const ulong* a,
            __global const ulong* b,
            __global const ulong* c,
            __global ulong* r)
        {
            int gx = get_global_id(0);
            // int lx = get_local_id(0);
            // * get_global_size(0)
            int index = gx;
            //if (index < ) {
            //    r[index] = 1.0 / (1.0 + exp(-100.f));
            //}
            
            r[index] = a[index] + b[index] + c[index];
            //r[index] = get_local_size(0);
        }
        """

        vadd = cu_to_cl(context = self._ctx,
                        cu_filepath = "get_value.cu",
                        kernelname = "fetch",
                        num_args = 4)
        LENGTH = np.uint64(10000)

        print("Compilied program")
        h_a = np.ones(LENGTH).astype(np.uint64)
        h_b = np.ones(LENGTH).astype(np.uint64)
        h_c = np.zeros(LENGTH).astype(np.uint64)
        h_r = np.empty(LENGTH).astype(np.uint64)  # Host result

        # Create the device buffers
        d_a = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_a)
        d_b = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_b)
        d_c = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = h_c)

        # Create the output (r) array in device memory
        d_r = cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, h_r.nbytes)
        print("Created buffers")

        def helper(keys, values = None):

            # vadd.set_arg(0, d_a)
            # vadd.set_arg(1, d_b)
            # # vadd.set_arg(2, d_c)
            #
            # vadd.set_arg(2, d_r)
            # vadd.set_args(d_a, d_b, d_r)

            print("Setup values")
            # vadd.set_scalar_arg_dtypes([None, None, None, None, np.uint64])

            # vadd(self._cl_queue, h_a.shape, None, d_a, d_b, d_c, d_r, np.uint64(LENGTH * 2))
            # print(h_a.shape)
            cl.enqueue_copy(self._cl_queue, d_a, d_b, d_c, d_r)

            ev = cl.enqueue_nd_range_kernel(self._cl_queue, vadd, h_a.shape, None)
            print("Waiting")
            ev.wait()
            print("waited")
            cl.enqueue_copy(self._cl_queue, h_r, d_r)
            print("Returning results")
            self._cl_queue.finish()
            print("Getting sum")
            print(sum(h_r) // 2)
            return h_r

        # _get_helper the kernel function from the compiled module
        for func in ["_get_helper", "set"]:
            self._functions.update({func: helper})

if __name__ == "__main__":
    # os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

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
        print(gpu_thread.set(keys, values))
        gpu_thread_list.append(gpu_thread)
