#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <ml/core/random.hpp>

int main() {
    // 1. 相同 seed 应产生相同随机序列
    {
        ml::RandomGenerator rng1(42);
        ml::RandomGenerator rng2(42);

        for (int i = 0; i < 100; ++i) {
            assert(rng1.uniform_double(-1.0, 1.0)
                == rng2.uniform_double(-1.0, 1.0));
        }
    }

    // 2. uniform_double() 生成的数应该落在区间内
    {
        ml::RandomGenerator rng(123);

        for (int i = 0; i < 10000; ++i) {
            double x = rng.uniform_double(-5.0, 3.0);

            assert(x >= -5.0);
            assert(x <= 3.0);
        }
    }

    // 3. uniform_int() 生成的整数应该落在闭区间 [left, right]
    {
        ml::RandomGenerator rng(456);

        for (int i = 0; i < 10000; ++i) {
            std::int64_t x = rng.uniform_int(-10, 20);

            assert(x >= -10);
            assert(x <= 20);
        }
    }

    // 4. reseed() 后应该重新产生同样的序列
    {
        ml::RandomGenerator rng(42);

        double a1 = rng.uniform_double(0.0, 1.0);
        double a2 = rng.uniform_double(0.0, 1.0);

        rng.reseed(42);

        double b1 = rng.uniform_double(0.0, 1.0);
        double b2 = rng.uniform_double(0.0, 1.0);

        assert(a1 == b1);
        assert(a2 == b2);
    }

    // 5. uniform_double() 非法区间应该抛异常
    {
        ml::RandomGenerator rng(42);

        bool thrown = false;

        try {
            rng.uniform_double(10.0, -10.0);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    // 6. uniform_int() 非法区间应该抛异常
    {
        ml::RandomGenerator rng(42);

        bool thrown = false;

        try {
            rng.uniform_int(10, -10);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    // 7. normal() 的 stddev 不能小于 0
    {
        ml::RandomGenerator rng(42);

        bool thrown = false;

        try {
            rng.normal(0.0, -1.0);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    std::cout << "test_random passed\n";
}