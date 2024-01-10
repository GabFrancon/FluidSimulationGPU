#include "SphSolverGPU.cuh"

#include <Time/Profiler.h>
#include <Debug/Assert.h>

// GPU multi-threading
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sph
{
    //__________________________________________________________________________
    // Kernel function

    __device__ float KernelGpu::f(float _len) const
    {
        const float q = _len / m_h;

        if (q <= 1.f)
        {
            return m_coeff * (1.f - 1.5f * square(q) + 0.75f * cube(q));
        }

        if (q <= 2.f)
        {
            return m_coeff * (0.25f * cube(2.f - q));
        }

        return 0.f;
    }

    __device__ float KernelGpu::derivativeF(float _len) const
    {
        const float q = _len / m_h;

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
        cudaMalloc((void**)&m_fPosition, m_fluidCount * sizeof(float2));
        cudaMemset(m_fPosition, 0, sizeof(float2) * m_fluidCount);

        cudaMalloc((void**)&m_fDensity, m_fluidCount * sizeof(float));
        cudaMemset(m_fDensity, 0, sizeof(float) * m_fluidCount);

        cudaMalloc((void**)&m_fVelocity, m_fluidCount * sizeof(float2));
        cudaMemset(m_fVelocity, 0, sizeof(float2) * m_fluidCount);

        cudaMalloc((void**)&m_fPressure, m_fluidCount * sizeof(float));
        cudaMemset(m_fPressure, 0, sizeof(float) * m_fluidCount);

        // Boundary state
        cudaMalloc((void**)&m_bPosition, m_boundaryCount * sizeof(float2));
        cudaMemset(m_bPosition, 0, sizeof(float2) * m_boundaryCount);

        cudaMalloc((void**)&m_Psi, m_boundaryCount * sizeof(float));
        cudaMemset(m_Psi, 0, sizeof(float) * m_boundaryCount);

        // Internal data
        cudaMalloc((void**)&m_Dii, m_fluidCount * sizeof(float2));
        cudaMemset(m_Dii, 0, sizeof(float2) * m_fluidCount);

        cudaMalloc((void**)&m_Aii, m_fluidCount * sizeof(float));
        cudaMemset(m_Aii, 0, sizeof(float) * m_fluidCount);

        cudaMalloc((void**)&m_sumDijPj, m_fluidCount * sizeof(float2));
        cudaMemset(m_sumDijPj, 0, sizeof(float2) * m_fluidCount);

        cudaMalloc((void**)&m_Vadv, m_fluidCount * sizeof(float2));
        cudaMemset(m_Vadv, 0, sizeof(float2) * m_fluidCount);

        cudaMalloc((void**)&m_Dadv, m_fluidCount * sizeof(float));
        cudaMemset(m_Dadv, 0, sizeof(float) * m_fluidCount);

        cudaMalloc((void**)&m_Pl, m_fluidCount * sizeof(float));
        cudaMemset(m_Pl, 0, sizeof(float) * m_fluidCount);

        cudaMalloc((void**)&m_Dcorr, m_fluidCount * sizeof(float));
        cudaMemset(m_Dcorr, 0, sizeof(float) * m_fluidCount);

        cudaMalloc((void**)&m_Fadv, m_fluidCount * sizeof(float2));
        cudaMemset(m_Fadv, 0, sizeof(float2) * m_fluidCount);

        cudaMalloc((void**)&m_Fp, m_fluidCount * sizeof(float2));
        cudaMemset(m_Fp, 0, sizeof(float2) * m_fluidCount);

        // Neighboring structures
        cudaMalloc((void**)&m_fNeighborsFlat, m_fluidCount * (kMaxFluidNeighbors + 1) * sizeof(u32));
        cudaMemset(m_fNeighborsFlat, 0, sizeof(u32) * (kMaxFluidNeighbors + 1) * m_fluidCount);

        cudaMalloc((void**)&m_bNeighborsFlat, m_fluidCount * (kMaxBoundaryNeighbors + 1) * sizeof(u32));
        cudaMemset(m_bNeighborsFlat, 0, sizeof(u32) * (kMaxBoundaryNeighbors + 1) * m_fluidCount);
    }

    __host__ void SceneGpu::Deallocate()
    {
        cudaFree(m_fPosition);
        cudaFree(m_fDensity);
        cudaFree(m_fVelocity);
        cudaFree(m_fPressure);

        cudaFree(m_bPosition);
        cudaFree(m_Psi);

        cudaFree(m_Dii);
        cudaFree(m_Aii);
        cudaFree(m_sumDijPj);
        cudaFree(m_Vadv);
        cudaFree(m_Dadv);
        cudaFree(m_Pl);
        cudaFree(m_Dcorr);
        cudaFree(m_Fadv);
        cudaFree(m_Fp);

        cudaFree(m_fNeighborsFlat);
        cudaFree(m_bNeighborsFlat);

        m_fluidCount = 0;
        m_boundaryCount = 0;
    }

    __device__ u32 SceneGpu::GetFluidNeighbor(u32 _particleID, int _neighborIdx) const
    {
        return m_fNeighborsFlat[get_idx(_particleID, _neighborIdx, kMaxFluidNeighbors)];
    }

    __device__ u32 SceneGpu::GetBoundaryNeighbor(u32 _particleID, int _neighborIdx) const
    {
        return m_bNeighborsFlat[get_idx(_particleID, _neighborIdx, kMaxBoundaryNeighbors)];
    }


    //___________________________________________________________
    // Advection Prediction Helpers

    __device__ void SceneGpu::ComputeDensity(u32 i) const
    {
        m_fDensity[i] = 0.f;

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            float2 pos_ij;
            pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
            pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

            m_fDensity[i] += m_m0 * m_kernel.W(pos_ij);
        }

        for (int idx = 0; GetBoundaryNeighbor(i, idx) != kInvalidIdx; idx++)
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
        m_Fadv[i] = { 0.f, 0.f}; 

        // add body force
        m_Fadv[i].y += m_m0 * m_gY;

        // add viscous force
        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
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

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
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

        for (int idx = 0; GetBoundaryNeighbor(i, idx) != kInvalidIdx; idx++)
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

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
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

        for (int idx = 0; GetBoundaryNeighbor(i, idx) != kInvalidIdx; idx++)
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

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -square(m_dt) * m_m0 / square(m_fDensity[i]);

                float2 d_ji;
                d_ji.x = coeff * (-gradW.x);
                d_ji.y = coeff * (-gradW.y);

                float2 temp;
                temp.x = m_Dii[i].x - d_ji.x;
                temp.y = m_Dii[i].y - d_ji.y;

                m_Aii[i] += m_m0 * dot(temp, gradW);
            }
        }

        for (int idx = 0; GetBoundaryNeighbor(i, idx) != kInvalidIdx; idx++)
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
        m_sumDijPj[i] = {0.f, 0.f};

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_m0 * m_fPressure[j] / square(m_fDensity[j]);

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

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
        {
            const u32 j = GetFluidNeighbor(i, idx);

            if (!equal(m_fPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_fPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_fPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -square(m_dt) * m_m0 / square(m_fDensity[i]);

                float2 d_ji;
                d_ji.x = coeff * (-gradW.x);
                d_ji.y = coeff * (-gradW.y);

                float2 temp;
                temp.x = m_sumDijPj[i].x - m_Dii[j].x * m_Pl[j] - (m_sumDijPj[j].x - d_ji.x * m_Pl[i]);
                temp.y = m_sumDijPj[i].y - m_Dii[j].y * m_Pl[j] - (m_sumDijPj[j].y - d_ji.y * m_Pl[i]);

                m_Dcorr[i] += m_m0 * dot(temp, gradW);
            }
        }

        for (int idx = 0; GetBoundaryNeighbor(i, idx) != kInvalidIdx; idx++)
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
        float previousPl = m_Pl[i];

        if (abs(m_Aii[i]) > FLT_MIN)
        {
            m_Pl[i] = (1 - m_omega) * previousPl + (m_omega / m_Aii[i]) * (m_rho0 - m_Dcorr[i]);
        }
        else
        {
            m_Pl[i] = 0.f;
        }

        m_fPressure[i] = fmax(m_Pl[i], 0.f);

        m_Pl[i] = m_fPressure[i];
        m_Dcorr[i] += m_Aii[i] * previousPl;
    }


    //___________________________________________________________
    // Integration Helpers

    __device__ void SceneGpu::ComputePressureForces(u32 i) const
    {
        m_Fp[i] = { 0.f, 0.f };

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
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

        for (int idx = 0; GetBoundaryNeighbor(i, idx) != kInvalidIdx; idx++)
        {
            const u32 j = GetBoundaryNeighbor(i, idx);

            if (!equal(m_bPosition[j], m_fPosition[i]))
            {
                float2 pos_ij;
                pos_ij.x = m_fPosition[i].x - m_bPosition[j].x;
                pos_ij.y = m_fPosition[i].y - m_bPosition[j].y;

                const float2 gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_m0 * m_Psi[j] * m_fPressure[i] / square(m_fDensity[i]);

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
    // CUDA compute kernels

    __global__ void predictAdvectionKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-ob-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.ComputeDensity(particleID);

        __syncthreads();

        _scene.ComputeAdvectionForces(particleID);
        _scene.PredictVelocity(particleID);
        _scene.StoreDii(particleID);

        __syncthreads();

        _scene.PredictDensity(particleID);
        _scene.InitPressure(particleID);
        _scene.StoreAii(particleID);
    }

    __global__ void pressureSolveKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-ob-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.StoreSumDijPj(particleID);

        __syncthreads();

        _scene.ComputePressure(particleID);
    }

    __global__ void integrationKernel(SceneGpu _scene)
    {
        const u32 particleID = blockIdx.x * blockDim.x + threadIdx.x;

        // out-ob-bound discard
        if (particleID >= _scene.m_fluidCount)
        {
            return;
        }

        _scene.ComputePressureForces(particleID);

        __syncthreads();

        _scene.UpdateVelocity(particleID);
        _scene.UpdatePosition(particleID);
    }

    __host__ void simulate(SceneGpu _scene)
    {
        constexpr u32 blockSize = 256;
        const u32 gridSize = (_scene.m_fluidCount + blockSize - 1) / blockSize;

        {
            AUTO_CPU_MARKER("Predict Advection");
            predictAdvectionKernel<<<gridSize, blockSize>>>(_scene);

            cudaDeviceSynchronize();
            const cudaError_t cudaError = cudaGetLastError();
            AVA_ASSERT(cudaError == cudaSuccess, "CUDA error: %s", cudaGetErrorString(cudaError));
        }
        {
            AUTO_CPU_MARKER("Pressure Solve");
            pressureSolveKernel<<<gridSize, blockSize>>>(_scene);

            cudaDeviceSynchronize();
            const cudaError_t cudaError = cudaGetLastError();
            AVA_ASSERT(cudaError == cudaSuccess, "CUDA error: %s", cudaGetErrorString(cudaError));
        }
        {
            AUTO_CPU_MARKER("Integration");
            integrationKernel<<<gridSize, blockSize>>>(_scene);

            cudaDeviceSynchronize();
            const cudaError_t cudaError = cudaGetLastError();
            AVA_ASSERT(cudaError == cudaSuccess, "CUDA error: %s", cudaGetErrorString(cudaError));
        }
    }

}
