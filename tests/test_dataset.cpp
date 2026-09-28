#include <cassert>
#include <iostream>

#include <ml/core/dataset.hpp>

int main() {
    ml::Dataset1D data{
        {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        }
    };

    assert(data.size() == 3);
    assert(!data.empty());

    assert(data[0].x == 1.0);
    assert(data[0].y == 2.0);

    data[1].x = 100.0;
    assert(data[1].x == 100.0);

    const ml::Dataset1D& const_data = data;
    assert(const_data[2].y == 6.0);

    std::cout << "test_dataset passed\n";
}
