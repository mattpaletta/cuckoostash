#include <metal_stdlib>
using namespace metal;

kernel void sqr(
    const device float *vIn [[ buffer(0) ]],
    device float *vOut [[ buffer(1) ]],
    uint id[[ thread_position_in_grid ]]) {
        vOut[id] = vIn[id] * vIn[id];
}