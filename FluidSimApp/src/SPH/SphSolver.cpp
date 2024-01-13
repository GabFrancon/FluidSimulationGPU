#include "SphSolver.h"

// CPU multi-threading
#include "omp.h"

// GPU multi-threading
#include "SphSolverGPU.cuh"

// Ava
#include <Debug/Log.h>
#include <Time/Profiler.h>

namespace sph
{
    SphSolver::SphSolver(const SphSettings& _settings)
    {
        m_rho0 = _settings.restDensity;
        m_nu = _settings.viscosity;
        m_eta = _settings.compressibility;
        m_omega = _settings.jacobiCoeff;
        m_g  = _settings.gravity;
        m_h = _settings.spacing;

        m_m0 = m_rho0 * Math::square(m_h);
        m_kernel = SphKernel(m_h, 2);
        m_grid = SphGrid(m_h * 2.f, _settings.dimensions);

        _Allocate(_settings.nbFluidParticles, _settings.nbBoundaryParticles);

        m_sceneGpu = new SceneGpu();
        m_sceneGpu->m_rho0 = m_rho0;
        m_sceneGpu->m_nu = m_nu;
        m_sceneGpu->m_eta = m_eta;
        m_sceneGpu->m_omega = m_omega;
        m_sceneGpu->m_gY = m_g.y;
        m_sceneGpu->m_h = m_h;
        m_sceneGpu->m_m0 = m_m0;
        m_sceneGpu->m_kernel.m_h = m_h;
        m_sceneGpu->m_kernel.m_coeff = m_kernel.GetCoeff();
        m_sceneGpu->m_kernel.m_derivCoeff = m_kernel.GetDerivCoeff();

        m_sceneGpu->Allocate(_settings.nbFluidParticles, _settings.nbBoundaryParticles);
    }

    SphSolver::~SphSolver()
    {
        _Deallocate();

        m_sceneGpu->Deallocate();
        delete m_sceneGpu;
    }

    void SphSolver::Prepare()
    {
        _BuildParticleGrid();
        _SearchNeighbors();

    #pragma omp parallel for
        for (int i = 0; i < m_boundaryCount; i++)
        {
            _PreComputePsi(i);
        }

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            _ComputeDensity(i);
        }

        m_sceneGpu->UploadFluidState(m_fPosition, m_fVelocity, m_fDensity);
        m_sceneGpu->UploadBoundaryState(m_bPosition, m_Psi);
    }

    void SphSolver::Simulate(float _dt, bool _onGpu)
    {
        AUTO_CPU_MARKER("Simulate");

        _BuildParticleGrid();
        _SearchNeighbors();

        if (_onGpu)
        {
            AUTO_CPU_MARKER("GPU");
        
            m_sceneGpu->m_dt = _dt;
            m_sceneGpu->UploadNeighbors(m_fNeighborsFlat, m_bNeighborsFlat);

            m_sceneGpu->PredictAdvection();
            m_sceneGpu->SolvePressure();
            m_sceneGpu->Integrate();

            m_sceneGpu->RetrieveFluidState(m_fPosition, m_fVelocity, m_fDensity);
        }
        else
        {
            AUTO_CPU_MARKER("CPU");
            m_dt = _dt;

            _PredictAdvection();
            _SolvePressure();
            _Integrate();
        }
    }


    //___________________________________________________________
    // Fluid Particles

    int SphSolver::GetFluidCount() const
    {
        return m_fluidCount;
    }

    void SphSolver::SetFluidPosition(u32 _particleID, const Vec2f& _position) const
    {
        m_fPosition[_particleID] = _position;
    }

    Vec2f SphSolver::GetFluidPosition(u32 _particleID) const
    {
        return m_fPosition[_particleID];
    }

    void SphSolver::SetFluidVelocity(u32 _particleID, const Vec2f& _velocity) const
    {
        m_fVelocity[_particleID] = _velocity;
    }

    Vec2f SphSolver::GetFluidVelocity(u32 _particleID) const
    {
        return m_fVelocity[_particleID];
    }

    void SphSolver::SetFluidDensity(u32 _particleID, float _density) const
    {
        m_fDensity[_particleID] = _density;
    }

    float SphSolver::GetFluidDensity(u32 _particleID) const
    {
        return m_fDensity[_particleID];
    }

    float SphSolver::AverageFluidDensity() const
    {
        float avgDensity = 0.f;

        for (int i = 0; i < m_fluidCount; i++)
        {
            avgDensity += m_Dcorr[i];
        }

        avgDensity /= m_fluidCount;
        return avgDensity;
    }


    //___________________________________________________________
    // Boundary Particles

    int SphSolver::GetBoundaryCount() const
    {
        return m_boundaryCount;
    }

    void SphSolver::SetBoundaryPosition(u32 _particleID, const Vec2f& _position) const
    {
        m_bPosition[_particleID] = _position;
    }

    Vec2f SphSolver::GetBoundaryPosition(u32 _particleID) const
    {
        return m_bPosition[_particleID];
    }


    //___________________________________________________________
    // Memory Allocation

    void SphSolver::_Allocate(int _fCount, int _bCount)
    {
        m_fluidCount = _fCount;
        m_boundaryCount = _bCount;

        // fluid state
        m_fPosition = newArr1<Vec2f>(m_fluidCount);
        memset(m_fPosition, 0, sizeof(Vec2f) * m_fluidCount);

        m_fVelocity = newArr1<Vec2f>(m_fluidCount);
        memset(m_fVelocity, 0, sizeof(Vec2f) * m_fluidCount);

        m_fPressure = newArr1<float>(m_fluidCount);
        memset(m_fPressure, 0, sizeof(float) * m_fluidCount);

        m_fDensity = newArr1<float>(m_fluidCount);
        memset(m_fDensity, 0, sizeof(float) * m_fluidCount);

        // boundary state
        m_bPosition = newArr1<Vec2f>(m_boundaryCount);
        memset(m_bPosition, 0, sizeof(Vec2f) * m_boundaryCount);

        m_Psi = newArr1<float>(m_boundaryCount);
        memset(m_Psi, 0, sizeof(float) * m_boundaryCount);

        // internal data
        m_Dii = newArr1<Vec2f>(m_fluidCount);
        memset(m_Dii, 0, sizeof(Vec2f) * m_fluidCount);

        m_Aii = newArr1<float>(m_fluidCount);
        memset(m_Aii, 0, sizeof(float) * m_fluidCount);

        m_sumDijPj = newArr1<Vec2f>(m_fluidCount);
        memset(m_sumDijPj, 0, sizeof(Vec2f) * m_fluidCount);

        m_Vadv = newArr1<Vec2f>(m_fluidCount);
        memset(m_Vadv, 0, sizeof(Vec2f) * m_fluidCount);

        m_Dadv = newArr1<float>(m_fluidCount);
        memset(m_Dadv, 0, sizeof(float) * m_fluidCount);

        m_Pl = newArr1<float>(m_fluidCount);
        memset(m_Pl, 0, sizeof(float) * m_fluidCount);

        m_Dcorr = newArr1<float>(m_fluidCount);
        memset(m_Dcorr, 0, sizeof(float) * m_fluidCount);

        m_Fadv = newArr1<Vec2f>(m_fluidCount);
        memset(m_Fadv, 0, sizeof(Vec2f) * m_fluidCount);

        m_Fp = newArr1<Vec2f>(m_fluidCount);
        memset(m_Fp, 0, sizeof(Vec2f) * m_fluidCount);

        // neighboring structures
        m_fluidInGrid = newArr2<u32>(&m_fluidInGridFlat, m_grid.CellCount(), kMaxFluidInCell + 1);
        std::fill_n(m_fluidInGridFlat, m_grid.CellCount() * (kMaxFluidInCell + 1), kInvalidIdx);

        m_fNeighbors = newArr2<u32>(&m_fNeighborsFlat, m_fluidCount, kMaxFluidNeighbors + 1);
        std::fill_n(m_fNeighborsFlat, m_fluidCount * (kMaxFluidNeighbors + 1), kInvalidIdx);

        m_boundaryInGrid = newArr2<u32>(&m_boundaryInGridFlat, m_grid.CellCount(), kMaxBoundaryInCell + 1);
        std::fill_n(m_boundaryInGridFlat, m_grid.CellCount() * (kMaxBoundaryInCell + 1), kInvalidIdx);

        m_bNeighbors = newArr2<u32>(&m_bNeighborsFlat, m_fluidCount, kMaxBoundaryNeighbors + 1);
        std::fill_n(m_bNeighborsFlat, m_fluidCount * (kMaxBoundaryNeighbors + 1), kInvalidIdx);
    }

    void SphSolver::_Deallocate()
    {
        delArray1(m_fPosition);
        delArray1(m_fVelocity);
        delArray1(m_fPressure);
        delArray1(m_fDensity);

        delArray1(m_bPosition);
        delArray1(m_Psi);

        delArray1(m_Dii);
        delArray1(m_Aii);
        delArray1(m_sumDijPj);
        delArray1(m_Vadv);
        delArray1(m_Dadv);
        delArray1(m_Pl);
        delArray1(m_Dcorr);
        delArray1(m_Fadv);
        delArray1(m_Fp);

        delArray2(m_fluidInGrid);
        delArray2(m_fNeighbors);

        delArray2(m_boundaryInGrid);
        delArray2(m_bNeighbors);

        m_fluidCount = 0;
        m_boundaryCount = 0;
    }


    //___________________________________________________________
    // Main Simulation Steps

    void SphSolver::_BuildParticleGrid()
    {
        AUTO_CPU_MARKER("Build Particle Grid");

        // reset particles in grid
        std::fill_n(m_fluidInGridFlat, m_grid.CellCount() * (kMaxFluidInCell + 1), kInvalidIdx);
        std::fill_n(m_boundaryInGridFlat, m_grid.CellCount() * (kMaxBoundaryInCell + 1), kInvalidIdx);

        // identify grid cells occupied by fluid particles
        for (int i = 0; i < m_fluidCount; i++)
        {
            const int cellIdx = m_grid.CellIndex(m_fPosition[i]);

            if (m_grid.Contains(cellIdx))
            {
                for (int idx = 0; idx < kMaxFluidInCell; idx++)
                {
                    if (m_fluidInGrid[cellIdx][idx] == kInvalidIdx)
                    {
                        m_fluidInGrid[cellIdx][idx] = i;
                        break;
                    }
                }
            }
        }

        // identify grid cells occupied by boundary particles
        for (int i = 0; i < m_boundaryCount; i++)
        {
            const int cellIdx = m_grid.CellIndex(m_bPosition[i]);

            if (m_grid.Contains(cellIdx))
            {
                for (int idx = 0; idx < kMaxBoundaryInCell; idx++)
                {
                    if (m_boundaryInGrid[cellIdx][idx] == kInvalidIdx)
                    {
                        m_boundaryInGrid[cellIdx][idx] = i;
                        break;
                    }
                }
            }
        }
    }

    void SphSolver::_SearchNeighbors()
    {
        AUTO_CPU_MARKER("Search Neighbors");

        const float searchRadius = 2.f * m_h;
        const float squaredRadius = Math::square(searchRadius);

        // reset neighbors
        std::fill_n(m_fNeighborsFlat, m_fluidCount * (kMaxFluidNeighbors + 1), kInvalidIdx);
        std::fill_n(m_bNeighborsFlat, m_fluidCount * (kMaxBoundaryNeighbors + 1), kInvalidIdx);

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            // gather neighboring cells
            std::vector<u32> neighborCells;
            const Vec2f& position = m_fPosition[i];
            m_grid.GatherNeighborCells(neighborCells, position, searchRadius);

            int fCurrentNeighbor = 0;
            int bCurrentNeighbor = 0;

            for (const u32 cellIdx : neighborCells)
            {
                // search for neighboring fluid particles
                for (int idx = 0; m_fluidInGrid[cellIdx][idx] != kInvalidIdx; idx++)
                {
                    const u32 j = m_fluidInGrid[cellIdx][idx];
                    const float d2 = Math::normSquared(m_fPosition[j] - position);

                    if (d2 < squaredRadius)
                    {
                        m_fNeighbors[i][fCurrentNeighbor] = j;
                        fCurrentNeighbor++;
                    }
                }

                // search for neighboring boundary particles
                for (int idx = 0; m_boundaryInGrid[cellIdx][idx] != kInvalidIdx; idx++)
                {
                    const u32 j = m_boundaryInGrid[cellIdx][idx];
                    const float d2 = Math::normSquared(m_bPosition[j] - position);

                    if (d2 < squaredRadius)
                    {
                        m_bNeighbors[i][bCurrentNeighbor] = j;
                        bCurrentNeighbor++;
                    }
                }
            }
        }
    }

    void SphSolver::_PredictAdvection()
    {
        AUTO_CPU_MARKER("Predict Advection");

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            _ComputeDensity(i);
        }

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            _ComputeAdvectionForces(i);
            _PredictVelocity(i);
            _StoreDii(i);
        }

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            _PredictDensity(i);
            _InitPressure(i);
            _StoreAii(i);
        }
    }

    void SphSolver::_SolvePressure()
    {
        AUTO_CPU_MARKER("Pressure Solve");

        int iteration = 0;
        // float error = 1.f;

        while (iteration < kMaxPressureSolveIteration)
        {
    #pragma omp parallel for
            for (int i = 0; i < m_fluidCount; i++)
            {
                _StoreSumDijPj(i);
            }

    #pragma omp parallel for
            for (int i = 0; i < m_fluidCount; i++)
            {
                _ComputePressure(i);
            }

    #pragma omp parallel for
            for (int i = 0; i < m_fluidCount; i++)
            {
                _SavePressure(i);
            }

            // error = abs(AverageFluidDensity() - m_rho0);
            iteration++;
        }
    }

    void SphSolver::_Integrate()
    {
        AUTO_CPU_MARKER("Integration");

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            _ComputePressureForces(i);
        }

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            _UpdateVelocity(i);
            _UpdatePosition(i);
        }
    }


    //___________________________________________________________
    // Advection Prediction Helpers

    void SphSolver::_PreComputePsi(u32 i) const
    {
        const float searchRadius = m_h;
        const float squaredRadius = Math::square(searchRadius);

        // gather neighboring cells
        std::vector<u32> neighborCells;
        m_grid.GatherNeighborCells(neighborCells, m_bPosition[i], searchRadius);

        // search for neighboring boundary particles
        std::vector<u32> neighbors;

        for (const u32 cellIdx : neighborCells)
        {
            // search for neighboring boundary particles
            for (int idx = 0; m_boundaryInGrid[cellIdx][idx] != kInvalidIdx; idx++)
            {
                const u32 j = m_boundaryInGrid[cellIdx][idx];
                const float d2 = Math::normSquared(m_bPosition[j] - m_bPosition[i]);

                if (d2 < squaredRadius)
                {
                    neighbors.push_back(j);
                }
            }
        }

        // compute density number "Psi"
        float sumK = 0.f;
        Vec2f pos_ij;
        
        for (const u32 j : neighbors)
        {
            pos_ij = m_bPosition[i] - m_bPosition[j];
            sumK += m_kernel.W(pos_ij);
        }

        m_Psi[i] = m_rho0 / sumK;
    }

    void SphSolver::_ComputeDensity(u32 i) const
    {
        m_fDensity[i] = 0.f;

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];
            const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
            m_fDensity[i] += m_m0 * m_kernel.W(pos_ij);
        }

        for (int idx = 0; m_bNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_bNeighbors[i][idx];
            const Vec2f pos_ij = m_fPosition[i] - m_bPosition[j];
            m_fDensity[i] += m_Psi[j] * m_kernel.W(pos_ij);
        }
    }

    void SphSolver::_ComputeAdvectionForces(u32 i) const
    {
        m_Fadv[i] = { 0.f, 0.f };

        // add body force
        m_Fadv[i] += m_m0 * m_g;

        // add viscous force
        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
                const Vec2f vel_ij = m_fVelocity[i] - m_fVelocity[j];

                const Vec2f gradW = m_kernel.GradW(pos_ij);
                const float coeff = 2.f * m_nu * (Math::square(m_m0) / m_fDensity[j]) * dot(vel_ij, pos_ij) / (Math::normSquared(pos_ij) + 0.01f * Math::square(m_h));

                m_Fadv[i] += coeff * gradW;
            }
        }
    }

    void SphSolver::_PredictVelocity(u32 i) const
    {
        m_Vadv[i] = m_fVelocity[i] + m_dt * m_Fadv[i] / m_m0;
    }

    void SphSolver::_StoreDii(u32 i) const
    {
        m_Dii[i] = { 0.f, 0.f };

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
                m_Dii[i] += (-m_m0 / Math::square(m_fDensity[i])) * m_kernel.GradW(pos_ij);
            }
        }

        for (int idx = 0; m_bNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_bNeighbors[i][idx];

            if (m_bPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Dii[i] += (-m_Psi[j] / Math::square(m_fDensity[i])) * m_kernel.GradW(pos_ij);
            }
        }

        m_Dii[i] *= Math::square(m_dt);
    }

    void SphSolver::_PredictDensity(u32 i) const
    {
        m_Dadv[i] = 0.f;

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
                const Vec2f vel_adv_ij = m_Vadv[i] - m_Vadv[j];
                m_Dadv[i] += m_m0 * dot(vel_adv_ij, m_kernel.GradW(pos_ij));
            }
        }

        for (int idx = 0; m_bNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_bNeighbors[i][idx];

            if (m_bPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Dadv[i] += m_Psi[j] * dot(m_Vadv[i], m_kernel.GradW(pos_ij));
            }
        }

        m_Dadv[i] *= m_dt;
        m_Dadv[i] += m_fDensity[i];
    }

    void SphSolver::_InitPressure(u32 i) const
    {
        m_Pl[i] = 0.5f * m_fPressure[i];
    }

    void SphSolver::_StoreAii(u32 i) const
    {
        m_Aii[i] = 0.f;

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
                const Vec2f D_ji = -(Math::square(m_dt) * m_m0 / Math::square(m_fDensity[i])) * (-m_kernel.GradW(pos_ij));
                m_Aii[i] += m_m0 * dot(m_Dii[i] - D_ji, m_kernel.GradW(pos_ij));
            }
        }

        for (int idx = 0; m_bNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_bNeighbors[i][idx];

            if (m_bPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Aii[i] += m_Psi[j] * dot(m_Dii[i], m_kernel.GradW(pos_ij));
            }
        }
    }


    //___________________________________________________________
    // Pressure Solving Helpers

    void SphSolver::_StoreSumDijPj(u32 i) const
    {
        m_sumDijPj[i] = { 0.f, 0.f };

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
                const Vec2f gradW = m_kernel.GradW(pos_ij);
                const float coeff = -m_m0 * m_Pl[j] / Math::square(m_fDensity[j]);
                m_sumDijPj[i] += coeff * gradW;
            }
        }

        m_sumDijPj[i] *= Math::square(m_dt);
    }

    void SphSolver::_ComputePressure(u32 i) const
    {
        m_Dcorr[i] = 0.f;

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];

                const Vec2f gradW = m_kernel.GradW(pos_ij);
                const float coeff = -Math::square(m_dt) * m_m0 / Math::square(m_fDensity[i]);

                const Vec2f D_ji = coeff * (-gradW);
                const Vec2f temp = m_sumDijPj[i] - m_Dii[j] * m_Pl[j] - (m_sumDijPj[j] - D_ji * m_Pl[i]);

                m_Dcorr[i] += m_m0 * dot(temp, gradW);
            }
        }

        for (int idx = 0; m_bNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_bNeighbors[i][idx];

            if (m_bPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_bPosition[j];
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

    void SphSolver::_SavePressure(u32 i) const
    {
        m_Dcorr[i] += m_Aii[i] * m_Pl[i];

        // save pressure for next iteration
        m_Pl[i] = m_fPressure[i];
    }


    //___________________________________________________________
    // Integration Helpers

    void SphSolver::_ComputePressureForces(u32 i) const
    {
        m_Fp[i] = { 0.f, 0.f };

        for (int idx = 0; m_fNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_fNeighbors[i][idx];

            if (m_fPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_fPosition[j];
                m_Fp[i] += -Math::square(m_m0) * (m_fPressure[i] / Math::square(m_fDensity[i]) + m_fPressure[j] / Math::square(m_fDensity[j])) * m_kernel.GradW(pos_ij);
            }
        }

        for (int idx = 0; m_bNeighbors[i][idx] != kInvalidIdx; idx++)
        {
            const u32 j = m_bNeighbors[i][idx];

            if (m_bPosition[j] != m_fPosition[i])
            {
                const Vec2f pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Fp[i] += -m_m0 * m_Psi[j] * (m_fPressure[i] / Math::square(m_fDensity[i])) * m_kernel.GradW(pos_ij);
            }
        }
    }

    void SphSolver::_UpdateVelocity(u32 i) const
    {
        m_fVelocity[i] = m_Vadv[i] + m_dt * m_Fp[i] / m_m0;
    }

    void SphSolver::_UpdatePosition(u32 i) const
    {
        m_fPosition[i] += m_dt * m_fVelocity[i];
        AVA_ASSERT(m_grid.Contains(m_fPosition[i]), "Particle outside simulation grid");
    }
}