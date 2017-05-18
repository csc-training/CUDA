#include <iostream>

inline void indToZorder(unsigned int &x, unsigned int &y, unsigned int ind)
{
    x = 0;
    y = 0;
    unsigned int mask = 1;
    while (ind != 0)
    {
        y |= ind & mask;
        ind = ind>>1;
        x |= ind & mask;
        mask = mask << 1;
    }
}

int main()
{
    //

    unsigned int x;
    unsigned int y;

    int xMax = 10;
    int yMax = 10;

    indToZorder(x,y, 10);

    std::cout << x << " " << y << std::endl; 
}