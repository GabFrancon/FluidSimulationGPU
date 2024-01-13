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
    cudaEvent_t s_fNeighborsTransferComplete, s_bNeighborsTransferComplete;
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
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fPosition, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_fPosition, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fDensity, m_fluidCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_fDensity, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fVelocity, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_fVelocity, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fPressure, m_fluidCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_fPressure, 0, sizeof(float) * m_fluidCount));

        // Boundary state
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_bPosition, m_boundaryCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_bPosition, 0, sizeof(float2) * m_boundaryCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Psi, m_boundaryCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_Psi, 0, sizeof(float) * m_boundaryCount));

        // Internal data
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Dii, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_Dii, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Aii, m_fluidCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_Aii, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_sumDijPj, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_sumDijPj, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Vadv, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_Vadv, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Dadv, m_fluidCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_Dadv, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Pl, m_fluidCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_Pl, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Dcorr, m_fluidCount * sizeof(float)));
        CUDA_CHECK_ERROR(cudaMemset(m_Dcorr, 0, sizeof(float) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Fadv, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_Fadv, 0, sizeof(float2) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_Fp, m_fluidCount * sizeof(float2)));
        CUDA_CHECK_ERROR(cudaMemset(m_Fp, 0, sizeof(float2) * m_fluidCount));

        // Neighboring structures
        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_fNeighborsFlat, m_fluidCount * (kMaxFluidNeighbors + 1) * sizeof(u32)));
        CUDA_CHECK_ERROR(cudaMemset(m_fNeighborsFlat, 0, sizeof(u32) * (kMaxFluidNeighbors + 1) * m_fluidCount));

        CUDA_CHECK_ERROR(cudaMalloc((void**)&m_bNeighborsFlat, m_fluidCount * (kMaxBoundaryNeighbors + 1) * sizeof(u32)));
        CUDA_CHECK_ERROR(cudaMemset(m_bNeighborsFlat, 0, sizeof(u32) * (kMaxBoundaryNeighbors + 1) * m_fluidCount));

        // CUDA memory events
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fNeighborsTransferComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_bNeighborsTransferComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fPositionTransferComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fVelocityTransferComplete));
        CUDA_CHECK_ERROR(cudaEventCreate(&s_fDensityTransferComplete));

        // GPU dispatch grid
        m_blockSize = 256;
        m_gridSize = (m_fluidCount + m_blockSize - 1) / m_blockSize;
    }

    __host__ void SceneGpu::Deallocate()
    {
        CUDA_CHECK_ERROR(cudaEventDestroy(s_fNeighborsTransferComplete));
        CUDA_CHECK_ERROR(cudaEventDestroy(s_bNeighborsTransferComplete));
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

        CUDA_CHECK_ERROR(cudaFree(m_fNeighborsFlat));
        CUDA_CHECK_ERROR(cudaFree(m_bNeighborsFlat));

        m_fluidCount = 0;
        m_boundaryCount = 0;
    }


    //___________________________________________________________
    // Memory Transfer

    __host__ void SceneGpu::UploadNeighbors(const void* _fNeighbors, const void* _bNeighbors) const
    {
        AUTO_CPU_MARKER("Upload Neighbors");

        // Asynchronous fluid neighbors memcpy from host to device
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_fNeighborsFlat, _fNeighbors, sizeof(u32) * (kMaxFluidNeighbors + 1) * m_fluidCount, cudaMemcpyHostToDevice));

        // Record an event after the memcpy is complete
        CUDA_CHECK_ERROR(cudaEventRecord(s_fNeighborsTransferComplete));

        // Asynchronous boundary neighbors memcpy from host to device
        CUDA_CHECK_ERROR(cudaMemcpyAsync(m_bNeighborsFlat, _bNeighbors, sizeof(u32) * (kMaxBoundaryNeighbors + 1) * m_fluidCount, cudaMemcpyHostToDevice));

        // Record an event after the memcpy is complete
        CUDA_CHECK_ERROR(cudaEventRecord(s_bNeighborsTransferComplete));

        // Synchronize with both memcpy events
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fNeighborsTransferComplete));
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_bNeighborsTransferComplete));

    }

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

    __host__ void SceneGpu::RetrieveFluidState(void* _fPosition, void* _fVelocity, void* _fDensity) const
    {
        AUTO_CPU_MARKER("Retrieve Fluid State");

        // Asynchronous fluid position memcpy from device to host
        CUDA_CHECK_ERROR(cudaMemcpyAsync(_fPosition, m_fPosition, sizeof(float2) * m_fluidCount, cudaMemcpyDeviceToHost));

        // Record an event after the memcpy is complete
        CUDA_CHECK_ERROR(cudaEventRecord(s_fPositionTransferComplete));

        // Asynchronous fluid velocity memcpy from device to host
        CUDA_CHECK_ERROR(cudaMemcpy(_fVelocity, m_fVelocity, sizeof(float2) * m_fluidCount, cudaMemcpyDeviceToHost));

        // Record an event after the memcpy is complete
        CUDA_CHECK_ERROR(cudaEventRecord(s_fVelocityTransferComplete));

        // Asynchronous fluid density memcpy from device to host
        CUDA_CHECK_ERROR(cudaMemcpy(_fDensity, m_fDensity, sizeof(float) * m_fluidCount, cudaMemcpyDeviceToHost));

        // Record an event after the memcpy is complete
        CUDA_CHECK_ERROR(cudaEventRecord(s_fDensityTransferComplete));

        // Synchronize with all memcpy events
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fPositionTransferComplete));
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fVelocityTransferComplete));
        CUDA_CHECK_ERROR(cudaEventSynchronize(s_fDensityTransferComplete));
    }


    //___________________________________________________________
    // Neighbor Access

    __device__ u32 SceneGpu::GetFluidNeighbor(u32 _particleID, int _neighborIdx) const
    {
        const int index = _particleID * (kMaxFluidNeighbors + 1) + _neighborIdx;
        return m_fNeighborsFlat[index];
    }

    __device__ u32 SceneGpu::GetBoundaryNeighbor(u32 _particleID, int _neighborIdx) const
    {
        const int index = _particleID * (kMaxBoundaryNeighbors + 1) + _neighborIdx;
        return m_bNeighborsFlat[index];
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
        m_Fadv[i] = { 0.f, 0.f };

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

                float2 D_ji;
                D_ji.x = coeff * (-gradW.x);
                D_ji.y = coeff * (-gradW.y);

                float2 temp;
                temp.x = m_Dii[i].x - D_ji.x;
                temp.y = m_Dii[i].y - D_ji.y;

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
        m_sumDijPj[i] = { 0.f, 0.f };

        for (int idx = 0; GetFluidNeighbor(i, idx) != kInvalidIdx; idx++)
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

                float2 D_ji;
                D_ji.x = coeff * (-gradW.x);
                D_ji.y = coeff * (-gradW.y);

                float2 temp;
                temp.x = m_sumDijPj[i].x - m_Dii[j].x * m_Pl[j] - (m_sumDijPj[j].x - D_ji.x * m_Pl[i]);
                temp.y = m_sumDijPj[i].y - m_Dii[j].y * m_Pl[j] - (m_sumDijPj[j].y - D_ji.y * m_Pl[i]);

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

    __host__ void SceneGpu::PredictAdvection() const
    {
        AUTO_CPU_MARKER("Predict Advection");

        computeDensityKernel<<<m_gridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        computeDiiKernel<<<m_gridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        computeAiiKernel<<<m_gridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    __host__ void SceneGpu::SolvePressure() const
    {
        AUTO_CPU_MARKER("Pressure Solve");

        int iteration = 0;

        while (iteration < kMaxPressureSolveIteration)
        {
            computeSumDijPjKernel<<<m_gridSize, m_blockSize>>>(*this);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            computePressureKernel<<<m_gridSize, m_blockSize>>>(*this);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            savePressureKernel<<<m_gridSize, m_blockSize>>>(*this);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            iteration++;
        }
    }

    __host__ void SceneGpu::Integrate() const
    {
        AUTO_CPU_MARKER("GPU Integration");

        computePressureForcesKernel<<<m_gridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        integratePositionKernel<<<m_gridSize, m_blockSize>>>(*this);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
}
