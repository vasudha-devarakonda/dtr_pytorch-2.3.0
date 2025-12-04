#include <c10/cuda/CUDACachingAllocator.h>

extern "C" long _bridge_current_memory(int device) {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    return stats.allocated_bytes[0].current;
}