#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <ml/core/random.hpp>

int main() {
    {
        ml::RandomGenerator rng1(42);
        ml::RandomGenerator rng2(42);

        for (int i = 0; i < 100; ++i) {
            assert(rng1.uniform_double(-1.0, 1.0)
                == rng2.uniform_double(-1.0, 1.0));
        }
    }

    {
        ml::RandomGenerator rng(123);

        for (int i = 0; i < 10000; ++i) {
            double x = rng.uniform_double(-5.0, 3.0);

            assert(x >= -5.0);
            assert(x <= 3.0);
        }
    }

    {
        ml::RandomGenerator rng(456);

        for (int i = 0; i < 10000; ++i) {
            std::int64_t x = rng.uniform_int(-10, 20);

            assert(x >= -10);
            assert(x <= 20);
        }
    }

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
