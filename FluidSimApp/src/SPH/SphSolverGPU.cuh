#pragma once

#include "SphAlloc.h"

namespace sph
{
    // Simulation grid
    struct GridGpu
    {
        int2 m_gridCells;
        float2 m_gridSize;
        float  m_cellSize;

        __device__ void GatherNeighborCells(u32& _cellCount, u32 _cellIndices[kMaxNeighborCells], const float2& _particle, float _radius) const;
        __device__ int2 CellPosition(const float2& _particle) const;

        __device__ u32 CellIndex(const float2& _particle) const;
        __device__ u32 CellIndex(int _i, int _j) const;

        __device__ bool Contains(const float2& _particle) const;
        __device__ bool Contains(int _cellIdx) const;

        __host__ __device__ u32 CellCount() const;
    };

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
    // scene grid
        GridGpu m_grid;

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

    // fluid grid
        u32* m_fluidPerCell = nullptr;      // list of fluid particles contained in each grid cell
        int* m_nbFluidPerCell = nullptr;    // number of fluid particles contained in each grid cell

    // boundary grid
        u32* m_boundaryPerCell = nullptr;   // list of boundary particles contained in each grid cell
        int* m_nbBoundaryPerCell = nullptr; // number of boundary particles contained in each grid cell

    // fluid neighbors
        u32* m_fNeighbors = nullptr;        // list of fluid particles neighboring each fluid particle
        int* m_fNeighborsCount = nullptr;   // number of fluid particles neighboring each fluid particle

    // boundary neighbors
        u32* m_bNeighbors = nullptr;        // list of boundary particles neighboring each fluid particle
        int* m_bNeighborsCount = nullptr;   // number of boundary particles neighboring each fluid particle

    // GPU dispatch
        u32 m_blockSize = 256;
        u32 m_fGridSize = 0;
        u32 m_bGridSize = 0;

    // Entry Point
        void Prepare();
        void Simulate(float _dt);

    // main simulation steps
        __host__ void BuildParticleGrid() const;
        __host__ void SearchNeighbors() const;
        __host__ void PredictAdvection() const;
        __host__ void SolvePressure() const;
        __host__ void Integrate() const;

    // memory allocation
        __host__ void Allocate(int _fCount, int _bCount);
        __host__ void Deallocate();

    // memory transfer
        __host__ void UploadBoundaryState(const void* _bPosition, const void* _Psi) const;
        __host__ void UploadFluidState(const void* _fPosition, const void* _fVelocity, const void* _fDensity) const;
        __host__ void FetchFluidState(void* _fPosition, void* _fVelocity, void* _fDensity) const;

    // particle grid
        __device__ void AddToFluidGrid(u32 i) const;
        __device__ void AddToBoundaryGrid(u32 i) const;
        __device__ u32 GetFluidInCell(u32 _cellID, int _fluidIdx) const;
        __device__ u32 GetBoundaryInCell(u32 _cellID, int _boundaryIdx) const;

    // neighbors search
        __device__ void SearchNeighbors(u32 i) const;
        __device__ u32 GetFluidNeighbor(u32 _particleID, int _fNeighborIdx) const;
        __device__ u32 GetBoundaryNeighbor(u32 _particleID, int _bNeighborIdx) const;

    // advection prediction
        __device__ void PrecomputePsi(u32 i) const;
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