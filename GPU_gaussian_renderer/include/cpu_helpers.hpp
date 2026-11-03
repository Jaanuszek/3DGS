#pragma once

#include <cstdint>
#include <bit>

namespace H_GS
{
    inline int get_msb_index(uint32_t n)
    {
        if(n == 0) return -1;
        // for older c++ <20 __builtin_clz can be used instead of std::countl_zero
        return 31 - std::countl_zero(n);
    }
}
