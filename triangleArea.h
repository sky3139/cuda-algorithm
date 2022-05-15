#include <vector>

#include <cuda.h>
#include "./cuVector.h"
#include "./cuCheck.h"

//行列式形式的面积公式
__host__ __device__ double triangleAreaG(double2 p1, double2 p2, double2 p3)
{
    return 0.5 * abs(p1.x * p2.y + p2.x * p3.y + p3.x * p1.y - p1.x * p3.y - p2.x * p1.y - p3.x * p2.y);
}

__global__ void tri_kernel(cuVector<double2> point, cuVector2D<double> ans)
{

    int i = threadIdx.x; // 
    int j = blockIdx.x;  //
    int n = point.size();
    // printf("%d %d %d %d,%d,%d\n", i, j, n, 0, 0, 0);
    if (j <= i)
        return;
    for (int k = j + 1; k < n; k++)
    {
        ans[i][j] = triangleAreaG(point[i], point[j], point[k]);
    }
    __syncthreads();      //等待同一block内所有线程执行到这里
    if (i == 1 && j == 2) //用一个线程找最大值
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                ans[n][n] = max(ans[i][j], ans[n][n]);
            }
        }
    }
    // unsigned int val = atomicInc(&point_buf->dev_points_num, 0xffffff);
}
class TriangleArea
{
public:
    double triangleArea(int x1, int y1, int x2, int y2, int x3, int y3)
    {
        return 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2);
    }

    double largestTriangleAreaCPU(vector<vector<int>> &points)
    {
        int n = points.size();
        double ret = 0.0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                for (int k = j + 1; k < n; k++)
                {
                    ret = max(ret, triangleArea(points[i][0], points[i][1], points[j][0], points[j][1], points[k][0], points[k][1]));
                }
            }
        }
        return ret;
    }

    double largestTriangleAreaGPU(vector<vector<int>> &points)
    {
        cuVector<double2> point(points.size());
        for (auto &pt : points)
        {
            point.push_back(make_double2(pt[0], pt[1]));
        }
        int n = point.size();
        double ret = 0.0;
        cuVector2D<double> ans(n + 1, n + 1, 0);
        tri_kernel<<<n, n>>>(point, ans);
        CK(cudaDeviceSynchronize());

        return ans[n][n];
    }
};
