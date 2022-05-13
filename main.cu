#include <cuda.h>
#include "./cuVector.h"
#include "./cuCheck.h"
#include <iostream>
#include <vector>
using namespace std;
typedef int2 caltype;
__device__ int cross(caltype p, caltype q, caltype r)
{
    return (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x);
}
__global__ void kernel(cuVector<caltype> trees, cuVector<bool> visit, cuVector<caltype> out)
{
    int leftMost = 0;
    //最坐边
    auto n = trees.cap();
    for (int i = 0; i < n; i++)
    {
        if (trees[i].x < trees[leftMost].x ||
            (trees[i].x == trees[leftMost].x &&
             trees[i].y < trees[leftMost].y))
        {
            leftMost = i;
        }
    }
    int p = leftMost;
    do
    {
        int q = (p + 1) % n; // p的下一个点
        for (int r = 0; r < n; r++)
        {
            /* 如果 r 在 pq 的右侧，则 q = r */
            if (cross(trees[p], trees[q], trees[r]) < 0)
            {
                q = r;
            }
        }
        /* 是否存在点 i, 使得 p 、q 、i 在同一条直线上 */
        for (int i = 0; i < n; i++)
        {
            if (visit[i] || i == p || i == q)
            {
                continue;
            }
            if (cross(trees[p], trees[q], trees[i]) == 0)
            {
                out.push_back(trees[i]);
                visit[i] = true;
            }
        }
        if (!visit[q])
        {
            out.push_back(trees[q]);
            visit[q] = true;
        }
        p = q;
    } while (p != leftMost);
}

class Jarvis
{
public:
    int cross(vector<int> &p, vector<int> &q, vector<int> &r)
    {
        return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]);
    }
    vector<vector<int>> outerTrees(vector<vector<int>> &trees)
    {
        int n = trees.size();
        if (n < 4)
        {
            return trees;
        }
        int leftMost = 0;
        for (int i = 0; i < n; i++)
        {
            if (trees[i][0] < trees[leftMost][0] ||
                (trees[i][0] == trees[leftMost][0] &&
                 trees[i][1] < trees[leftMost][1]))
            {
                leftMost = i;
            }
        }
        vector<vector<int>> res;
        vector<bool> visit(n, false);
        int p = leftMost;
        do
        {
            int q = (p + 1) % n; // p的下一个点
            for (int r = 0; r < n; r++)
            {
                /* 如果 r 在 pq 的右侧，则 q = r */
                if (cross(trees[p], trees[q], trees[r]) < 0)
                {
                    q = r;
                }
            }
            /* 是否存在点 i, 使得 p 、q 、i 在同一条直线上 */
            for (int i = 0; i < n; i++)
            {
                if (visit[i] || i == p || i == q)
                {
                    continue;
                }
                if (cross(trees[p], trees[q], trees[i]) == 0)
                {
                    res.emplace_back(trees[i]);
                    visit[i] = true;
                }
            }
            if (!visit[q])
            {
                res.emplace_back(trees[q]);
                visit[q] = true;
            }
            p = q;
        } while (p != leftMost);
        return res;
    }

    vector<vector<int>> outerTreesGPU(vector<vector<int>> &trees)
    {
        if (trees.size() < 4)
            return trees;
        cuVector<caltype> arr(trees.size());
        cuVector<caltype> ans(trees.size());
        for (int i = 0; i < trees.size(); i++)
            arr.push_back(make_int2(trees[i][0], trees[i][1]));
        cuVector<bool> mask(trees.size(), false);
        kernel<<<1, 1>>>(arr, mask, ans);
        CK(cudaDeviceSynchronize());
        vector<vector<int>> res;
        for (int i = 0; i < ans.size(); i++)
            res.push_back(vector<int>{ans[i].x, ans[i].y});
        return res;
    }
};
int main()
{
    Jarvis jar;
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
