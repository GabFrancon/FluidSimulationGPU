#pragma once

// CPU multi-threading
#include "omp.h"

// CPU profiling
#include <Time/Profiler.h>

#include "SphGrid.h"
#include "SphKernel.h"

namespace sph
{
    struct SphSettings
    {
        float spacing = 0.25f;
        float restDensity = 1000.f;
        float viscosity = 0.08f;
        float compressibility = 0.01f;
        float jacobiCoeff = 0.5f;
        int maxIteration = 2;
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
        void Simulate(float _dt);

    // fluid particles
        int GetFluidCount() const;

        void SetFluidPosition(int _particleID, const Vec2f& _position) const;
        Vec2f GetFluidPosition(int _particleID) const;

        void SetFluidVelocity(int _particleID, const Vec2f& _velocity) const;
        Vec2f GetFluidVelocity(int _particleID) const;

        void SetFluidDensity(int _particleID, float _density) const;
        float GetFluidDensity(int _particleID) const;

    // boundary particles
        int GetBoundaryCount() const;

        void SetBoundaryPosition(int _particleID, const Vec2f& _position) const;
        Vec2f GetBoundaryPosition(int _particleID) const;


    private:
    // memory allocation
        void _Allocate(int _fCount, int _bCount);
        void _Deallocate();

    // main simulation steps
        void _BuildParticleGrid();
        void _SearchNeighbors();
        void _PredictAdvection();
        void _SolvePressure();
        void _CorrectIntegration();

    // advection prediction helpers
        void _PreComputePsi(int i) const;
        void _ComputeDensity(int i) const;
        void _ComputeAdvectionForces(int i) const;
        void _PredictVelocity(int i) const;
        void _StoreDii(int i) const;
        void _PredictDensity(int i) const;
        void _InitPressure(int i) const;
        void _StoreAii(int i) const;

    // pressure solving helpers
        void _StoreSumDijPj(int i) const;
        void _ComputePressure(int i) const;
        float _GetAverageDensity() const;

    // integration helpers
        void _ComputePressureForces(int i) const;
        void _UpdateVelocity(int i) const;
        void _UpdatePosition(int i) const;

    // data helpers
        SphKernel m_kernel;  // smooth kernel helper
        SphGrid m_grid;      // spatial grid helper

    // simulation settings
        float m_dt;          // simulation timestep
        float m_nu;          // kinematic viscosity
        float m_eta;         // compressibility
        float m_rho0;        // rest density
        float m_h;           // particle spacing
        Vec2f m_g;           // gravity
        float m_m0;          // particles rest mass
        float m_omega;       // Jacobi's relaxed coefficient
        int m_maxIter;       // max pressure solve iteration

    // boundary particle states
        Vec2f* m_bPosition = nullptr;  // boundary particles positions
        int m_boundaryCount = 0;       // number of boundary particles

    // fluid particle states
        Vec2f* m_fPosition = nullptr;  // fluid particles positions
        Vec2f* m_fVelocity = nullptr;  // fluid particles velocities
        float* m_fPressure = nullptr;  // fluid particles pressure
        float* m_fDensity = nullptr;   // fluid particles density
        int m_fluidCount = 0;          // number of fluid particles

    // internal data
        float* m_Psi = nullptr;        // boundary density numbers
        Vec2f* m_Dii = nullptr;        // computation coefficient
        float* m_Aii = nullptr;        // computation coefficient 
        Vec2f* m_sumDijPj = nullptr;   // computation coefficient
        Vec2f* m_Vadv = nullptr;       // advection velocity
        float* m_Dadv = nullptr;       // advection density
        float* m_Pl = nullptr;         // corrected pressure at iteration l
        float* m_Dcorr = nullptr;      // corrected density
        Vec2f* m_Fadv = nullptr;       // non-pressure forces
        Vec2f* m_Fp = nullptr;         // pressure forces

    // neighboring structures
        std::vector<IndexList> m_fGrid;      // fluid particles contained in each grid cell
        std::vector<IndexList> m_fNeighbors; // fluid neighbors of each fluid particle
        std::vector<IndexList> m_bGrid;      // boundary particles contained in each grid cell
        std::vector<IndexList> m_bNeighbors; // boundary neighbors of each fluid particle
    };
}