#pragma once

#include <cstdint>

namespace H_GS
{
    uint32_t getMSB(uint32_t n)
    {
        int step = sizeof(n) * 4;
        uint32_t msb = 0;

        if(n >= (1U << step)) { msb += step; n >>= step; step >>= 1; }

        return msb;
    }
}
