#pragma once

#include <vector>

#include <cuda.h>
#include "./cuVector.h"
#include "./cuCheck.h"

__global__ void calsum_kernel(cuVector2D<int> pre, cuVector2D<int> cuimg)
{

    int j = threadIdx.x; // + threadIdx.y * blockDim.x;
    int i = blockIdx.x;  // + threadIdx.y * blockDim.x;
    int n = pre.cols;
    int m = pre.rows;
    int left = max(j - 1, 0);
    int right = min(j + 1, n - 2);
    int top = max(i - 1, 0);
    int bottom = min(i + 1, m - 2);
    int cnt = (bottom - top + 1) * (right - left + 1);
    int sum = pre[bottom + 1][right + 1] - pre[top][right + 1] - pre[bottom + 1][left] + pre[top][left];
    cuimg[i][j] = sum / cnt;
}
using namespace std;

class prefixsum
{
public:
    vector<vector<int>> imageSmoother(vector<vector<int>> &img)
    {
        int m = img.size();
        int n = img[0].size();
        vector<vector<int>> pre(m + 1, vector<int>(n + 1, 0));

        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                pre[i][j] = pre[i - 1][j] + pre[i][j - 1] - pre[i - 1][j - 1] + img[i - 1][j - 1];
            }
        }
        vector<vector<int>> ans(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int left = max(j - 1, 0);
                int right = min(j + 1, n - 1);
                int top = max(i - 1, 0);
                int bottom = min(i + 1, m - 1);
                int cnt = (bottom - top + 1) * (right - left + 1);
                int sum = pre[bottom + 1][right + 1] - pre[top][right + 1] - pre[bottom + 1][left] + pre[top][left];
                ans[i][j] = sum / cnt;
            }
        }
        // for (int i = 0; i < m; i++)
        // {
        //     for (int j = 0; j < n; j++)
        //     {
        //         cout << ans[i][j] << " ";
        //     }
        //     cout << endl;
        // }
        return ans;
    }
    vector<vector<int>> imageSmootherGPU(vector<vector<int>> &img)
    {
        int m = img.size();
        int n = img[0].size();

        cuVector2D<int> cupre(m + 1, n + 1, 0);
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                cupre[i][j] = cupre[i - 1][j] + cupre[i][j - 1] - cupre[i - 1][j - 1] + img[i - 1][j - 1];
            }
        }
        cuVector2D<int> cuimg(m, n, 0);
        vector<vector<int>> ans(m, vector<int>(n, 0));
        calsum_kernel<<<m, n>>>(cupre, cuimg);
        CK(cudaDeviceSynchronize());
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                ans[i][j] = cuimg[i][j];
                // cout << ans[i][j] << " ";
            }
            // cout << endl;
        }
        return ans;
    }
};
