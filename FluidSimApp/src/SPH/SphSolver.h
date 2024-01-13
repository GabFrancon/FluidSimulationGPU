#pragma once

#include "SphGrid.h"
#include "SphKernel.h"
#include "SphAlloc.h"

namespace sph
{
    struct SceneGpu;

    struct SphSettings
    {
        float spacing = 0.25f;
        float restDensity = 1000.f;
        float viscosity = 0.08f;
        float compressibility = 0.01f;
        float jacobiCoeff = 0.5f;
        Vec2f gravity = Vec2f(0.f, -9.81f);
        Vec2f dimensions = Vec2f(25.f, 14.f);

        int nbFluidParticles = 200;
        int nbBoundaryParticles = 200;
    };

    class SphSolver
    {
    public:
        SphSolver(const SphSettings& _settings);
        ~SphSolver();

        void Prepare();
        void Simulate(float _dt, bool _onGpu = false);

    // fluid particles
        int GetFluidCount() const;
        void SetFluidPosition(u32 _particleID, const Vec2f& _position) const;
        Vec2f GetFluidPosition(u32 _particleID) const;

        void SetFluidVelocity(u32 _particleID, const Vec2f& _velocity) const;
        Vec2f GetFluidVelocity(u32 _particleID) const;

        void SetFluidDensity(u32 _particleID, float _density) const;
        float GetFluidDensity(u32 _particleID) const;

        float AverageFluidDensity() const;
        float GetParticleSpacing() const;

    // boundary particles
        int GetBoundaryCount() const;
        void SetBoundaryPosition(u32 _particleID, const Vec2f& _position) const;
        Vec2f GetBoundaryPosition(u32 _particleID) const;

    private:
    // memory allocation
        void _Allocate(int _fCount, int _bCount);
        void _Deallocate();

    // main simulation steps
        void _BuildParticleGrid();
        void _SearchNeighbors();
        void _PredictAdvection();
        void _SolvePressure();
        void _Integrate();

    // advection prediction helpers
        void _PreComputePsi(u32 i) const;
        void _ComputeDensity(u32 i) const;
        void _ComputeAdvectionForces(u32 i) const;
        void _PredictVelocity(u32 i) const;
        void _StoreDii(u32 i) const;
        void _PredictDensity(u32 i) const;
        void _InitPressure(u32 i) const;
        void _StoreAii(u32 i) const;

    // pressure solving helpers
        void _StoreSumDijPj(u32 i) const;
        void _ComputePressure(u32 i) const;
        void _SavePressure(u32 i) const;

    // integration helpers
        void _ComputePressureForces(u32 i) const;
        void _UpdateVelocity(u32 i) const;
        void _UpdatePosition(u32 i) const;

    // data helpers
        SphKernel m_kernel;  // smooth kernel helper
        SphGrid m_grid;      // spatial grid helper

        SceneGpu* m_sceneGpu;

    // simulation settings
        float m_dt;          // simulation timestep
        float m_nu;          // kinematic viscosity
        float m_eta;         // compressibility
        float m_rho0;        // rest density
        float m_h;           // particle spacing
        Vec2f m_g;           // gravity
        float m_m0;          // particles rest mass
        float m_omega;       // Jacobi's relaxed coefficient

    // boundary particle states
        Vec2f* m_bPosition = nullptr;  // boundary particles positions
        int m_boundaryCount = 0;       // number of boundary particles

    // fluid particle states
        Vec2f* m_fPosition = nullptr;  // fluid particles positions
        Vec2f* m_fPositionGpu = nullptr;
        Vec2f* m_fVelocity = nullptr;  // fluid particles velocities
        float* m_fPressure = nullptr;  // fluid particles pressure
        float* m_fDensity = nullptr;   // fluid particles density
        int m_fluidCount = 0;          // number of fluid particles

    // internal data
        float* m_Psi = nullptr;        // boundary density numbers
        Vec2f* m_Dii = nullptr;        // intermediate computation coefficient
        float* m_Aii = nullptr;        // intermediate computation coefficient 
        Vec2f* m_sumDijPj = nullptr;   // intermediate computation coefficient
        Vec2f* m_Vadv = nullptr;       // velocity without pressure forces
        float* m_Dadv = nullptr;       // density without pressure forces
        float* m_Pl = nullptr;         // corrected pressure at previous iteration
        float* m_Dcorr = nullptr;      // corrected density
        Vec2f* m_Fadv = nullptr;       // non-pressure forces
        Vec2f* m_Fp = nullptr;         // pressure forces

    // fluid grid
        u32** m_fluidPerCell;        // list of fluid particles contained in each grid cell
        u32* m_fluidPerCellFlat;     // flat array version of m_fluidInGrid
        int* m_nbFluidPerCell;       // number of fluid particles per grid cell

    // boundary grid
        u32** m_boundaryPerCell;     // list of boundary particles contained in each grid cell
        u32* m_boundaryPerCellFlat;  // flat array version of m_boundaryInGrid
        int* m_nbBoundaryPerCell;    // number of boundary particles per grid cell

    // fluid neighbors
        u32** m_fNeighbors;          // list of fluid particles neighboring each fluid particle
        u32* m_fNeighborsFlat;       // flat array version of m_fNeighbors
        int* m_fNeighborsCount;      // number of fluid neighbors per particle

    // boundary neighbors
        u32** m_bNeighbors;          // list of boundary particles neighboring each fluid particle
        u32* m_bNeighborsFlat;       // flat array version of m_bNeighbors
        int* m_bNeighborsCount;      // number of boundary neighbors per particle
    };
}