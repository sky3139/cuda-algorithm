#include <iostream>
#include <vector>
#include <cuda.h>

#include "./cuVector.h"
#include "./cuCheck.h"
#include "./hull.h"

using namespace std;

int main()
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
