#include <iostream>
#include <vector>
#include <cuda.h>

#include "./cuVector.h"
#include "./cuCheck.h"
#include "./hull.h"
#include "./prefixsum.h"
#include "./triangleArea.h"
using namespace std;

void hull_test()
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

void prefixsum_test()
{
    prefixsum p;
    vector<vector<int>> input = {{100, 200, 100}, {200, 50, 200}, {100, 200, 100}};
    p.imageSmootherGPU(input);
    p.imageSmoother(input);
}
int main()
{
    vector<vector<int>> input = {{0, 0}, {0, 1}, {1, 0}, {0, 2}, {2, 0}};
    TriangleArea ta;
    cout << ta.largestTriangleAreaCPU(input) << endl;
    cout << ta.largestTriangleAreaGPU(input) << endl;
}