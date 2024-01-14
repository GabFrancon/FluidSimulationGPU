#include "SphSolverGPU.cuh"

#include <Time/Profiler.h>
#include <Debug/Assert.h>

// CUDA library
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK_ERROR(_cudaCmd)                                                              \
{                                                                                               \
    cudaError_t cudaStatus;                                                                     \
    cudaStatus = _cudaCmd;                                                                      \
    AVA_ASSERT(cudaStatus == cudaSuccess, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));  \
}


namespace sph
{
    cudaEvent_t s_fluidGridResetComplete, s_boundaryGridResetComplete;
    cudaEvent_t s_fPositionTransferComplete, s_fVelocityTransferComplete, s_fDensityTransferComplete;

    //__________________________________________________________________________
    // Math Functions

    __device__ float square(float _a)
    {
        return _a * _a;
    }

    __device__ float cube(float _a)
    {
        return _a * _a * _a;
    }

    __device__ float dot(const float2& _vecA, const float2& _vecB)
    {
        return _vecA.x * _vecB.x + _vecA.y * _vecB.y;
    }

    __device__ float equal(const float2& _vecA, const float2& _vecB)
    {
        return _vecA.x == _vecB.x && _vecA.y == _vecB.y;
    }

    __device__ float length(const float2& _vec)
    {
        return sqrtf(dot(_vec, _vec));
    }

    __device__ float lengthSquared(const float2& _vec)
    {
        return dot(_vec, _vec);
    }


    //__________________________________________________________________________
    // Grid Functions

    __device__ void GridGpu::GatherNeighborCells(u32& _cellCount, u32 _cellIndices[kMaxNeighborCells], const float2& _particle, float _radius) const
    {
        float2 low;
        low.x = _particle.x - _radius;
        low.y = _particle.y - _radius;
        const int2 minCell = CellPosition(low);

        float2 high;
        high.x = _particle.x + _radius;
        high.y = _particle.y + _radius;
        const int2 maxCell = CellPosition(high);

        const int iMin = minCell.x > 0 ? minCell.x : 0;
        const int iMax = maxCell.x < m_gridCells.x - 1 ? maxCell.x : m_gridCells.x - 1;
        const int jMin = minCell.y > 0 ? minCell.y : 0;
        const int jMax = maxCell.y < m_gridCells.y - 1 ? maxCell.y : m_gridCells.y - 1;

        int count = 0;
        _cellCount = (iMax - iMin + 1) * (jMax - jMin + 1);

        for (int i = iMin; i <= iMax; ++i)
        {
            for (int j = jMin; j <= jMax; ++j)
            {
                _cellIndices[count] = CellIndex(i, j);
                count++;
            }
        }
    }

    __device__ int2 GridGpu::CellPosition(const float2& _particle) const
    {
        int2 cell;
        cell.x = (u32)floor(_particle.x / m_cellSize);
        cell.y = (u32)floor(_particle.y / m_cellSize);
        return cell;
    }

    __device__ u32 GridGpu::CellIndex(const float2& _particle) const
    {
        const int2 cell = CellPosition(_particle);
        return CellIndex(cell.x, cell.y);
    }

    __device__ u32 GridGpu::CellIndex(int _i, int _j) const
    {
        return _i + _j * m_gridCells.x;
        // return _i * m_gridCells.y + _j;
    }

    __device__ bool GridGpu::Contains(const float2& _particle) const
    {
        const u32 idx = CellIndex(_particle);
        return Contains(idx);
    }

    __device__ bool GridGpu::Contains(int _cellIdx) const
    {
        return _cellIdx >= 0 && _cellIdx < CellCount();
    }

    __host__ __device__ u32 GridGpu::CellCount() const
    {
        return m_gridCells.x * m_gridCells.y;
    }


    //__________________________________________________________________________
    // Kernel Functions

    __device__ float KernelGpu::f(float _length) const
    {
        const float q = _length / m_h;

        if (q <= 1.f)
        {
            return m_coeff * (1.f - 1.5f * square(q) + 0.75f * cube(q));
        }

        if (q <= 2.f)
        {
            return m_coeff * 0.25f * cube(2.f - q);
        }

        return 0.f;
    }

    __device__ float KernelGpu::derivativeF(float _length) const
    {
        const float q = _length / m_h;

        if (q <= 1.f) 
        {
            return m_derivCoeff * (-3.f * q + 2.25f * square(q));
        }

        if (q <= 2.f) 
        {
            return -m_derivCoeff * 0.75f * square(2.f - q);
        }

        return 0.f;
    }

    __device__ float KernelGpu::W(const float2& _rij) const
    {
        const float len = length(_rij);
        return f(len);
    }

    __device__ float2 KernelGpu::GradW(const float2& _rij) const
    {
        const float len = length(_rij);
        const float derivative = derivativeF(len);

        float2 res;
        res.x = derivative * _rij.x / len;
        res.y = derivative * _rij.y / len;
        return res;
    }


    //___________________________________________________________
    // Memory Allocation

    __host__ void SceneGpu::Allocate(int _fCount, int _bCount)
    {
        m_fluidCount = _fCount;
        m_boundaryCount = _bCount;

        // Fluid state
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fPosition, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_fPosition, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fDensity, sizeof(float) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_fDensity, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fVelocity, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_fVelocity, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fPressure, sizeof(float) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_fPressure, 0, sizeof(float) * m_fluidCount));

        // Boundary state
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_bPosition, sizeof(float2) * m_boundaryCount));
        CUDA_CHECK_ERROR(cudaMemset(m_bPosition, 0, sizeof(float2) * m_boundaryCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Psi, sizeof(float) * m_boundaryCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Psi, 0, sizeof(float) * m_boundaryCount));

        // Internal data
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Dii, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Dii, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Aii, sizeof(float) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Aii, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_sumDijPj, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_sumDijPj, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Vadv, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Vadv, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Dadv, sizeof(float) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Dadv, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Pl, sizeof(float) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Pl, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Dcorr, sizeof(float) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Dcorr, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Fadv, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Fadv, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Fp, sizeof(float2) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_Fp, 0, sizeof(float2) * m_fluidCount));

        // Fluid grid
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fluidPerCell, sizeof(u32) * m_grid.CellCount() * kMaxFluidPerCell));
        CUDA_CHECK_ERROR(cudaMemset(m_fluidPerCell, 0, sizeof(u32) * m_grid.CellCount() * kMaxFluidPerCell));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_nbFluidPerCell, sizeof(int) * m_grid.CellCount()));
        CUDA_CHECK_ERROR(cudaMemset(m_nbFluidPerCell, 0, sizeof(int) * m_grid.CellCount()));

        // Boundary grid
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_boundaryPerCell, sizeof(u32) * m_grid.CellCount() * kMaxBoundaryPerCell));
        CUDA_CHECK_ERROR(cudaMemset(m_boundaryPerCell, 0, sizeof(u32) * m_grid.CellCount() * kMaxBoundaryPerCell));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_nbBoundaryPerCell, sizeof(int) * m_grid.CellCount()));
        CUDA_CHECK_ERROR(cudaMemset(m_nbBoundaryPerCell, 0, sizeof(int) * m_grid.CellCount()));

        // Fluid neighbors
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fNeighbors, sizeof(u32) * m_fluidCount * kMaxFluidNeighbors));
        CUDA_CHECK_ERROR(cudaMemset(m_fNeighbors, 0, sizeof(u32) * m_fluidCount * kMaxFluidNeighbors));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fNeighborsCount, sizeof(int) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_fNeighborsCount, 0, sizeof(int) * m_fluidCount));

        // Boundary neighbors
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_bNeighbors, sizeof(u32) * m_fluidCount * kMaxBoundaryNeighbors));
        CUDA_CHECK_ERROR(cudaMemset(m_bNeighbors, 0, sizeof(u32) * m_fluidCount * kMaxBoundaryNeighbors));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_bNeighborsCount, sizeof(int) * m_fluidCount));
        CUDA_CHECK_ERROR(cudaMemset(m_bNeighborsCount, 0, sizeof(int) * m_fluidCount));

        // CUDA memory events
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fluidGridResetComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_boundaryGridResetComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fPositionTransferComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fVelocityTransferComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fDensityTransferComplete));

        // GPU dispatch grid
        m_blockSize = 256;
        m_fGridSize = (m_fluidCount + m_blockSize - 1) / m_blockSize;
        m_bGridSize = (m_boundaryCount + m_blockSize - 1) / m_blockSize;
    }

    __host__ void SceneGpu::Deallocate()
    {
        CUDA_CHECK_ERROR(cudaEventDestroy(s_fluidGridResetComplete));
        CUDA_CHECK_ERROR(cudaEventDestroy(s_boundaryGridResetComplete));
        CUDA_CHECK_ERROR(cudaEventDestroy(s_fPositionTransferComplete));
        CUDA_CHECK_ERROR(cudaEventDestroy(s_fVelocityTransferComplete));
        CUDA_CHECK_ERROR(cudaEventDestroy(s_fDensityTransferComplete));

        CUDA_CHECK_ERROR(cudaFree(m_fPosition));
        CUDA_CHECK_ERROR(cudaFree(m_fDensity));
        CUDA_CHECK_ERROR(cudaFree(m_fVelocity));
        CUDA_CHECK_ERROR(cudaFree(m_fPressure));

        CUDA_CHECK_ERROR(cudaFree(m_bPosition));
        CUDA_CHECK_ERROR(cudaFree(m_Psi));

        CUDA_CHECK_ERROR(cudaFree(m_Dii));
        CUDA_CHECK_ERROR(cudaFree(m_Aii));
        CUDA_CHECK_ERROR(cudaFree(m_sumDijPj));
        CUDA_CHECK_ERROR(cudaFree(m_Vadv));
        CUDA_CHECK_ERROR(cudaFree(m_Dadv));
        CUDA_CHECK_ERROR(cudaFree(m_Pl));
        CUDA_CHECK_ERROR(cudaFree(m_Dcorr));
        CUDA_CHECK_ERROR(cudaFree(m_Fadv));
        CUDA_CHECK_ERROR(cudaFree(m_Fp));

        CUDA_CHECK_ERROR(cudaFree(m_fluidPerCell));
        CUDA_CHECK_ERROR(cudaFree(m_boundaryPerCell));
        CUDA_CHECK_ERROR(cudaFree(m_fNeighbors));
        CUDA_CHECK_ERROR(cudaFree(m_bNeighbors));

        m_fluidCount = 0;
        m_boundaryCount = 0;
    }


    //___________________________________________________________
    // Memory Transfer

    __host__ void SceneGpu::UploadBoundaryState(const void* _bPosition, const void* _Psi) const
    {
        CUDA_CHECK_ERROR(cudaMemcpy(m_bPosition, _bPosition, sizeof(float2) * m_boundaryCount, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(m_Psi, _Psi, sizeof(float) * m_boundaryCount, cudaMemcpyHostToDevice));
    }

    __host__ void SceneGpu::UploadFluidState(const void* _fPosition, const void* _fVelocity, const void* _fDensity) const
    {
        CUDA_CHECK_ERROR(cudaMemcpy(m_fPosition, _fPosition, sizeof(float2) * m_fluidCount, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(m_fVelocity, _fVelocity, sizeof(float2) * m_fluidCount, cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(m_fDensity, _fDensity, sizeof(float) * m_fluidCount, cudaMemcpyHostToDevice));
    }

    __host__ void SceneGpu::FetchFluidState(void* _fPosition, void* _fVelocity, void* _fDensity) const
    {
        AUTO_CPU_MARKER("Retrieve Fluid State");

        // Asynchronously memcpy fluid position from device to host
        CUDA_CHECK_ERROR(cudaMemcpyAsync(_fPosition, m_fPosition, sizeof(float2) * m_fluidCount, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaEventRecord(s_fPositionTransferComplete));

        // Asynchronously memcpy fluid velocity from device to host
        CUDA_CHECK_ERROR(cudaMemcpyAsync(_fVelocity, m_fVelocity, sizeof(float2) * m_fluidCount, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaEventRecord(s_fVelocityTransferComplete));

        // Asynchronous fluid density memcpy from device to host
        CUDA_CHECK_ERROR(cudaMemcpyAsync(_fDensity, m_fDensity, sizeof(float) * m_fluidCount, cudaMemcpyDeviceToHost));
        CUDA_CHECK_ERROR(cudaEventRecord(s_fDensityTransferComplete));

        // Synchronize with all memcpy events
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fPositionTransferComplete));
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fVelocityTransferComplete));
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fDensityTransferComplete));
    }


    //___________________________________________________________
    // Particle Grid

    __device__ void SceneGpu::AddToFluidGrid(u32 i) const
    {
        const int cellIdx = m_grid.CellIndex(m_fPosition[i]);

        if (m_grid.Contains(cellIdx))
        {
            const int indexInCell = atomicAdd(&m_nbFluidPerCell[cellIdx], 1);
            m_fluidPerCell[cellIdx * kMaxFluidPerCell + indexInCell] = i;
        }
    }

    __device__ void SceneGpu::AddToBoundaryGrid(u32 i) const
    {
        const int cellIdx = m_grid.CellIndex(m_bPosition[i]);

        if (m_grid.Contains(cellIdx))
        {
            const int indexInCell = atomicAdd(&m_nbBoundaryPerCell[cellIdx], 1);
            m_boundaryPerCell[cellIdx * kMaxBoundaryPerCell + indexInCell] = i;
        }
    }

    __device__ u32 SceneGpu::GetFluidInCell(u32 _cellID, int _fluidIdx) const
    {
        const int index = _cellID * kMaxFluidPerCell + _fluidIdx;
        return m_fluidPerCell[index];
    }

    __device__ u32 SceneGpu::GetBoundaryInCell(u32 _cellID, int _boundaryIdx) const
    {
        const int index = _cellID * kMaxBoundaryPerCell + _boundaryIdx;
        return m_boundaryPerCell[index];
    }


    //___________________________________________________________
    // Neighbor Search

    __device__ void SceneGpu::SearchNeighbors(u32 i) const
    {
        const float searchRadius = 2.f * m_h;
        const float squaredRadius = square(searchRadius);

        // gather neighboring cells
        u32 cellCount;
        u32 neighborCells[kMaxNeighborCells];
        m_grid.GatherNeighborCells(cellCount, neighborCells, m_fPosition[i], searchRadius);

        const int fNeighborOffset = i * kMaxFluidNeighbors;
        const int bNeighborOffset = i * kMaxBoundaryNeighbors;

        int fCurrentNeighbor = 0;
        int bCurrentNeighbor = 0;

        for (int cellIdx = 0; cellIdx < cellCount; cellIdx++)
        {
            const u32 cellID = neighborCells[cellIdx];

            // search for neighboring fluid particles
            for (int idx = 0; idx < m_nbFluidPerCell[cellID]; idx++)
            {
                const u32 j = GetFluidInCell(cellID, idx);

                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                if (lengthSquared(pos_ij) < squaredRadius)
                {
                    m_fNeighbors[fNeighborOffset + fCurrentNeighbor] = j;
                    fCurrentNeighbor++;
                }
            }

            // search for neighboring boundary particles
            for (int idx = 0; idx < m_nbBoundaryPerCell[cellID]; idx++)
            {
                const u32 j = GetBoundaryInCell(cellID, idx);

                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                if (lengthSquared(pos_ij) < squaredRadius)
                {
                    m_bNeighbors[bNeighborOffset + bCurrentNeighbor] = j;
                    bCurrentNeighbor++;
                }
            }
        }

        // update neighbors count
        m_fNeighborsCount[i] = fCurrentNeighbor;
        m_bNeighborsCount[i]  = bCurrentNeighbor;
    }

    __device__ u32 SceneGpu::GetFluidNeighbor(u32 _particleID, int _fNeighborIdx) const
    {
        const int index = _particleID * kMaxFluidNeighbors + _fNeighborIdx;
        return m_fNeighbors[index];
    }

    __device__ u32 SceneGpu::GetBoundaryNeighbor(u32 _particleID, int _bNeighborIdx) const
    {
        const int index = _particleID * kMaxBoundaryNeighbors + _bNeighborIdx;
        return m_bNeighbors[index];
    }

    __device__ void SceneGpu::PrecomputePsi(u32 i) const
    {
        const float searchRadius = m_h;
        const float squaredRadius = square(searchRadius);

        // gather neighboring cells
        u32 neighborCount;
        u32 neighborCells[kMaxNeighborCells];
        m_grid.GatherNeighborCells(neighborCount, neighborCells, m_bPosition[i], searchRadius);

        // compute boundary density number "Psi"
        float sumK = 0.f;

        for (int cellIdx = 0; cellIdx < neighborCount; cellIdx++)
        {
            const u32 cellID = neighborCells[cellIdx];

            // search for neighboring boundary particles
            for (int idx = 0; idx < m_nbBoundaryPerCell[cellID]; idx++)
            {
                const u32 j = GetBoundaryInCell(cellID, idx);

                float2 pos_ij;
                pos_ij.x = m_bPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_bPosition[i].y - m_bPosition[j].y;

                if (lengthSquared(pos_ij) < squaredRadius)
                {
                    sumK += m_kernel.W(pos_ij);
                }
            }
        }

        m_Psi[i] = m_rho0 / sumK;
    }


    //___________________________________________________________
    // Advection Prediction Helpers

    __device__ void SceneGpu::ComputeDensity(u32 i) const
    {
        m_fDensity[i] = 0.f;

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            float2 pos_ij;
            pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
            pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

            m_fDensity[i] += m_m0 * m_kernel.W(pos_ij);
        }

        for (int idx = 0; idx < m_bNeighborsCount[i]; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            float2 pos_ij;
            pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
            pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

            m_fDensity[i] += m_Psi[j] * m_kernel.W(pos_ij);
        }
    }

    __device__ void SceneGpu::ComputeAdvectionForces(u32 i) const
    {
        m_Fadv[i] = { 0.f, 0.f };

        // add body force
        m_Fadv[i].y += m_m0 * m_gY;

        // add viscous force
        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                float2 vel_ij;
                vel_ij.x = m_fVelocity[i].x - m_fVelocity[j].x;
                vel_ij.y = m_fVelocity[i].y - m_fVelocity[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = 2.f * m_nu * (square(m_m0) / m_fDensity[j]) * dot(vel_ij, pos_ij) / (lengthSquared(pos_ij) + 0.01f * square(m_h));

                m_Fadv[i].x += coeff * gradW.x;
                m_Fadv[i].y += coeff * gradW.y;
            }
        }
    }

    __device__ void SceneGpu::PredictVelocity(u32 i) const
    {
        m_Vadv[i].x = m_fVelocity[i].x + m_dt * m_Fadv[i].x / m_m0;
        m_Vadv[i].y = m_fVelocity[i].y + m_dt * m_Fadv[i].y / m_m0;
    }

    __device__ void SceneGpu::StoreDii(u32 i) const
    {
        m_Dii[i] = { 0.f, 0.f };

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_m0 / square(m_fDensity[i]);

                m_Dii[i].x += coeff * gradW.x;
                m_Dii[i].y += coeff * gradW.y;
            }
        }

        for (int idx = 0; idx < m_bNeighborsCount[i]; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            if (!equal(m_bPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_Psi[j] / square(m_fDensity[i]);

                m_Dii[i].x += coeff * gradW.x;
                m_Dii[i].y += coeff * gradW.y;
            }
        }

        const float dt2 = square(m_dt);
        m_Dii[i].x *= dt2;
        m_Dii[i].y *= dt2;
    }

    __device__ void SceneGpu::PredictDensity(u32 i) const
    {
        m_Dadv[i] = 0.f;

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                float2 vel_adv_ij;
                vel_adv_ij.x = m_Vadv[i].x - m_Vadv[j].x;
                vel_adv_ij.y = m_Vadv[i].y - m_Vadv[j].y;

                m_Dadv[i] += m_m0 * dot(vel_adv_ij, m_kernel.GradW(pos_ij));
            }
        }

        for (int idx = 0; idx < m_bNeighborsCount[i]; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            if (!equal(m_bPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                m_Dadv[i] += m_Psi[j] * dot(m_Vadv[i], m_kernel.GradW(pos_ij));
            }
        }

        m_Dadv[i] *= m_dt;
        m_Dadv[i] += m_fDensity[i];
    }

    __device__ void SceneGpu::InitPressure(u32 i) const
    {
        m_Pl[i] = 0.5f * m_fPressure[i];
    }

    __device__ void SceneGpu::StoreAii(u32 i) const
    {
        m_Aii[i] = 0.f;

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -square(m_dt) * m_m0 / square(m_fDensity[i]);

                float2 D_ji;
                D_ji.x = coeff * (-gradW.x);
                D_ji.y = coeff * (-gradW.y);

                float2 temp;
                temp.x = m_Dii[i].x - D_ji.x;
                temp.y = m_Dii[i].y - D_ji.y;

                m_Aii[i] += m_m0 * dot(temp, gradW);
            }
        }

        for (int idx = 0; idx < m_bNeighborsCount[i]; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            if (!equal(m_bPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                m_Aii[i] += m_Psi[j] * dot(m_Dii[i], m_kernel.GradW(pos_ij));
            }
        }
    }


    //___________________________________________________________
    // Pressure Solving Helpers

    __device__ void SceneGpu::StoreSumDijPj(u32 i) const
    {
        m_sumDijPj[i] = { 0.f, 0.f };

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_m0 * m_Pl[j] / square(m_fDensity[j]);

                m_sumDijPj[i].x += coeff * gradW.x;
                m_sumDijPj[i].y += coeff * gradW.y;
            }
        }

        const float dt2 = square(m_dt);
        m_sumDijPj[i].x *= dt2;
        m_sumDijPj[i].y *= dt2;
    }

    __device__ void SceneGpu::ComputePressure(u32 i) const
    {
        m_Dcorr[i] = 0.f;

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -square(m_dt) * m_m0 / square(m_fDensity[i]);

                float2 D_ji;
                D_ji.x = coeff * (-gradW.x);
                D_ji.y = coeff * (-gradW.y);

                float2 temp;
                temp.x = m_sumDijPj[i].x - m_Dii[j].x * m_Pl[j] - (m_sumDijPj[j].x - D_ji.x * m_Pl[i]);
                temp.y = m_sumDijPj[i].y - m_Dii[j].y * m_Pl[j] - (m_sumDijPj[j].y - D_ji.y * m_Pl[i]);

                m_Dcorr[i] += m_m0 * dot(temp, gradW);
            }
        }

        for (int idx = 0; idx < m_bNeighborsCount[i]; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            if (!equal(m_bPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                m_Dcorr[i] += m_Psi[j] * dot(m_sumDijPj[i], m_kernel.GradW(pos_ij));
            }
        }

        m_Dcorr[i] += m_Dadv[i];

        if (fabs(m_Aii[i]) > FLT_EPSILON)
        {
            m_fPressure[i] = fmax(0.f, (1 - m_omega) * m_Pl[i] + (m_omega / m_Aii[i]) * (m_rho0 - m_Dcorr[i]));
        }
        else
        {
            m_fPressure[i] = 0.f;
        }
    }

    __device__ void SceneGpu::SavePressure(u32 i) const
    {
        m_Dcorr[i] += m_Aii[i] * m_Pl[i];

        // save pressure for next iteration
        m_Pl[i] = m_fPressure[i];
    }


    //___________________________________________________________
    // Integration Helpers

    __device__ void SceneGpu::ComputePressureForces(u32 i) const
    {
        m_Fp[i] = { 0.f, 0.f };

        for (int idx = 0; idx < m_fNeighborsCount[i]; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -square(m_m0) * (m_fPressure[i] / square(m_fDensity[i]) + m_fPressure[j] / square(m_fDensity[j]));

                m_Fp[i].x += coeff * gradW.x;
                m_Fp[i].y += coeff * gradW.y;
            }
        }

        for (int idx = 0; idx < m_bNeighborsCount[i]; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            if (!equal(m_bPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_m0 * m_Psi[j] * (m_fPressure[i] / square(m_fDensity[i]));

                m_Fp[i].x += coeff * gradW.x;
                m_Fp[i].y += coeff * gradW.y;
            }
        }
    }

    __device__ void SceneGpu::UpdateVelocity(u32 i) const
    {
        m_fVelocity[i].x = m_Vadv[i].x + m_dt * m_Fp[i].x / m_m0;
        m_fVelocity[i].y = m_Vadv[i].y + m_dt * m_Fp[i].y / m_m0;
    }

    __device__ void SceneGpu::UpdatePosition(u32 i) const
    {
        m_fPosition[i].x += m_dt * m_fVelocity[i].x;
        m_fPosition[i].y += m_dt * m_fVelocity[i].y;
    }


    //___________________________________________________________
    // CUDA Compute Kernels

    __global__ void addToFluidGridKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.AddToFluidGrid(particleID);
    }

    __global__ void addToBoundaryGridKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_boundaryCount)
        {
            return;
        }

        _scene.AddToBoundaryGrid(particleID);
    }

    __global__ void searchNeighborKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.SearchNeighbors(particleID);
    }

    __global__ void precomputePsiKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_boundaryCount)
        {
            return;
        }

        _scene.ComputeDensity(particleID);
    }

    __global__ void computeDensityKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.ComputeDensity(particleID);
    }

    __global__ void computeDiiKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.ComputeAdvectionForces(particleID);
        _scene.PredictVelocity(particleID);
        _scene.StoreDii(particleID);
    }

    __global__ void computeAiiKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.PredictDensity(particleID);
        _scene.InitPressure(particleID);
        _scene.StoreAii(particleID);
    }

    __global__ void computeSumDijPjKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.StoreSumDijPj(particleID);
    }

    __global__ void computePressureKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.ComputePressure(particleID);
    }

    __global__ void savePressureKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.SavePressure(particleID);
    }

    __global__ void computePressureForcesKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.ComputePressureForces(particleID);
    }

    __global__ void integratePositionKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-of-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.UpdateVelocity(particleID);
        _scene.UpdatePosition(particleID);
    }


    //___________________________________________________________
    // Main simulation Steps

    __host__ void SceneGpu::BuildParticleGrid() const
    {
        AUTO_CPU_MARKER("Build Particle Grid");

        // reset fluid grid
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_nbFluidPerCell, 0, sizeof(int) * m_grid.CellCount()));
        CUDA_CHECK_ERROR(cudaEventRecord(s_fluidGridResetComplete));

        // reset boundary grid
        CUDA_CHECK_ERROR(cudaMemsetAsync(m_nbBoundaryPerCell, 0, sizeof(int) * m_grid.CellCount()));
        CUDA_CHECK_ERROR(cudaEventRecord(s_boundaryGridResetComplete));

        // build fluid grid
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fluidGridResetComplete));
        addToFluidGridKernel<<<m_fGridSize, m_blockSize>>>(*this);

        // build boundary grid
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_boundaryGridResetComplete));
        addToBoundaryGridKernel<<<m_bGridSize, m_blockSize>>>(*this);

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    __host__ void SceneGpu::SearchNeighbors() const
    {
        AUTO_CPU_MARKER("Search Neighbors");

        searchNeighborKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    __host__ void SceneGpu::PredictAdvection() const
    {
        AUTO_CPU_MARKER("Predict Advection");

        computeDensityKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        computeDiiKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        computeAiiKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    __host__ void SceneGpu::SolvePressure() const
    {
        AUTO_CPU_MARKER("Pressure Solve");

        int iteration = 0;

        while (iteration < kMaxPressureSolveIteration)
        {
            computeSumDijPjKernel<<<m_fGridSize, m_blockSize>>>(*this);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            computePressureKernel<<<m_fGridSize, m_blockSize>>>(*this);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            savePressureKernel<<<m_fGridSize, m_blockSize>>>(*this);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            iteration++;
        }
    }

    __host__ void SceneGpu::Integrate() const
    {
        AUTO_CPU_MARKER("Integration");

        computePressureForcesKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        integratePositionKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }


    //___________________________________________________________
    // Entry Point

    __host__ void SceneGpu::Prepare()
    {
        BuildParticleGrid();
        SearchNeighbors();

        precomputePsiKernel<<<m_bGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        computeDensityKernel<<<m_fGridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    __host__ void SceneGpu::Simulate(float _dt)
    {
        AUTO_CPU_MARKER("Simulation");
        m_dt = _dt;

        BuildParticleGrid();
        SearchNeighbors();
        PredictAdvection();
        SolvePressure();
        Integrate();
    }
}
