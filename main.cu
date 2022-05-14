#include <iostream>
#include <vector>
#include <cuda.h>

#include "./cuVector.h"
#include "./cuCheck.h"
#include "./hull.h"
#include "./prefixsum.h"
using namespace std;

int hull_test()
{
    Andrew jar;
    vector<vector<int>> input = {{1, 1}, {2, 2}, {2, 0}, {2, 4}, {3, 3}, {4, 2}};
    auto ansgpu = jar.outerTreesGPU(input);
    for (auto &it : ansgpu)
    {
        for (auto &i : it)
        {
            cout << i << " ";
        }
        cout << endl;
    }
    auto anscpu = jar.outerTrees(input);
    for (auto &it : anscpu)
    {
        for (auto &i : it)
        {
            cout << i << " ";
        }
        cout << endl;
    }
}
int main()
{
    prefixsum p;
    vector<vector<int>> input = {{100, 200, 100}, {200, 50, 200}, {100, 200, 100}};
    p.imageSmootherGPU(input);
    p.imageSmoother(input); 
}