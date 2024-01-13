#pragma once

#include "SphAlloc.h"

namespace sph
{
    // Simulation smooth kernel
    struct KernelGpu
    {
        float m_h;
        float m_coeff;             // kernel coefficient
        float m_derivCoeff;        // kernel gradient coefficient

        __device__ float f(float _length) const;
        __device__ float derivativeF(float _length) const;

        __device__ float W(const float2& _rij) const;
        __device__ float2 GradW(const float2& _rij) const;
    };

    // Simulation scene
    struct SceneGpu
    {
    // smooth kernel
        KernelGpu m_kernel;

    // simulation settings
        float m_dt;                      // simulation timestep
        float m_nu;                      // kinematic viscosity
        float m_eta;                     // compressibility
        float m_rho0;                    // rest density
        float m_h;                       // particle spacing
        float m_gY;                      // gravity Y component
        float m_m0;                      // particles rest mass
        float m_omega;                   // Jacobi's relaxed coefficient

    // boundary particle states
        float2* m_bPosition = nullptr;   // boundary particles positions
        int m_boundaryCount = 0;         // number of boundary particles

    // fluid particle states
        float2* m_fPosition = nullptr;   // fluid particles positions
        float2* m_fVelocity = nullptr;   // fluid particles velocities
        float* m_fPressure = nullptr;    // fluid particles pressure
        float* m_fDensity = nullptr;     // fluid particles density
        int m_fluidCount = 0;            // number of fluid particles

    // internal data
        float* m_Psi = nullptr;          // boundary density numbers
        float2* m_Dii = nullptr;         // intermediate computation coefficient
        float* m_Aii = nullptr;          // intermediate computation coefficient 
        float2* m_sumDijPj = nullptr;    // intermediate computation coefficient
        float2* m_Vadv = nullptr;        // velocity X component without pressure forces
        float* m_Dadv = nullptr;         // density without pressure forces
        float* m_Pl = nullptr;           // corrected pressure at last iteration
        float* m_Dcorr = nullptr;        // corrected density
        float2* m_Fadv = nullptr;        // non-pressure forces X component
        float2* m_Fp = nullptr;          // pressure forces X component

    // neighboring structures
        u32* m_fNeighborsFlat = nullptr; // list of fluid particles neighboring each fluid particle
        u32* m_bNeighborsFlat = nullptr; // list of boundary particles neighboring each fluid particle

    // GPU dispatch
        u32 m_blockSize = 256;
        u32 m_gridSize = 0;

    // main simulation steps
        __host__ void PredictAdvection() const;
        __host__ void SolvePressure() const;
        __host__ void Integrate() const;

    // memory allocation
        __host__ void Allocate(int _fCount, int _bCount);
        __host__ void Deallocate();

    // memory transfer
        __host__ void UploadNeighbors(const void* _fNeighbors, const void* _bNeighbors) const;
        __host__ void UploadBoundaryState(const void* _bPosition, const void* _Psi) const;
        __host__ void UploadFluidState(const void* _fPosition, const void* _fVelocity, const void* _fDensity) const;
        __host__ void RetrieveFluidState(void* _fPosition, void* _fVelocity, void* _fDensity) const;

    // neighbors access
        __device__ u32 GetFluidNeighbor(u32 _particleID, int _neighborIdx) const;
        __device__ u32 GetBoundaryNeighbor(u32 _particleID, int _neighborIdx) const;

    // advection prediction
        __device__ void ComputeDensity(u32 i) const;
        __device__ void ComputeAdvectionForces(u32 i) const;
        __device__ void PredictVelocity(u32 i) const;
        __device__ void StoreDii(u32 i) const;
        __device__ void PredictDensity(u32 i) const;
        __device__ void InitPressure(u32 i) const;
        __device__ void StoreAii(u32 i) const;

    // pressure solving
        __device__ void StoreSumDijPj(u32 i) const;
        __device__ void ComputePressure(u32 i) const;
        __device__ void SavePressure(u32 i) const;

    // integration
        __device__ void ComputePressureForces(u32 i) const;
        __device__ void UpdateVelocity(u32 i) const;
        __device__ void UpdatePosition(u32 i) const;
    };

}