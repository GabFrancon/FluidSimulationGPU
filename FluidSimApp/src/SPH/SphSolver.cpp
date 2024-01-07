#include "SphSolver.h"

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
        m_maxIter = _settings.maxIteration;
        m_m0 = m_rho0 * Math::square(m_h);
        m_kernel = SphKernel(m_h, 2);
        m_grid = SphGrid(m_h * 2.f, _settings.dimensions);

        _Allocate(_settings.nbFluidParticles, _settings.nbBoundaryParticles);
    }

    SphSolver::~SphSolver()
    {
        _Deallocate();
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
    }

    void SphSolver::Simulate(float _dt)
    {
        AUTO_CPU_MARKER("Solver::Advance");
        m_dt = _dt;

        _BuildParticleGrid();
        _SearchNeighbors();
        _PredictAdvection();
        _SolvePressure();
        _CorrectIntegration();
    }


    //___________________________________________________________
    // Fluid Particles

    int SphSolver::GetFluidCount() const
    {
        return m_fluidCount;
    }

    void SphSolver::SetFluidPosition(int _particleID, const Vec2f& _position) const
    {
        m_fPosition[_particleID] = _position;
    }

    Vec2f SphSolver::GetFluidPosition(int _particleID) const
    {
        return m_fPosition[_particleID];
    }

    void SphSolver::SetFluidVelocity(int _particleID, const Vec2f& _velocity) const
    {
        m_fVelocity[_particleID] = _velocity;
    }

    Vec2f SphSolver::GetFluidVelocity(int _particleID) const
    {
        return m_fVelocity[_particleID];
    }

    void SphSolver::SetFluidDensity(int _particleID, float _density) const
    {
        m_fDensity[_particleID] = _density;
    }

    float SphSolver::GetFluidDensity(int _particleID) const
    {
        return m_fDensity[_particleID];
    }


    //___________________________________________________________
    // Boundary Particles

    int SphSolver::GetBoundaryCount() const
    {
        return m_boundaryCount;
    }

    void SphSolver::SetBoundaryPosition(int _particleID, const Vec2f& _position) const
    {
        m_bPosition[_particleID] = _position;
    }

    Vec2f SphSolver::GetBoundaryPosition(int _particleID) const
    {
        return m_bPosition[_particleID];
    }


    //___________________________________________________________
    // Memory Allocation

    void SphSolver::_Allocate(int _fCount, int _bCount)
    {
        m_fluidCount = _fCount;
        m_boundaryCount = _bCount;

        m_fPosition = new Vec2f[m_fluidCount];
        memset(m_fPosition, 0, sizeof(Vec2f) * m_fluidCount);

        m_fDensity = new float[m_fluidCount];
        memset(m_fDensity, 0, sizeof(float) * m_fluidCount);

        m_fVelocity = new Vec2f[m_fluidCount];
        memset(m_fVelocity, 0, sizeof(Vec2f) * m_fluidCount);

        m_fPressure = new float[m_fluidCount];
        memset(m_fPressure, 0, sizeof(float) * m_fluidCount);

        m_bPosition = new Vec2f[m_boundaryCount];
        memset(m_bPosition, 0, sizeof(Vec2f) * m_boundaryCount);

        m_Psi = new float[m_boundaryCount];
        memset(m_Psi, 0, sizeof(float) * m_boundaryCount);

        m_Dii = new Vec2f[m_fluidCount];
        memset(m_Dii, 0, sizeof(Vec2f) * m_fluidCount);

        m_Aii = new float[m_fluidCount];
        memset(m_Aii, 0, sizeof(float) * m_fluidCount);

        m_sumDijPj = new Vec2f[m_fluidCount];
        memset(m_sumDijPj, 0, sizeof(Vec2f) * m_fluidCount);

        m_Vadv = new Vec2f[m_fluidCount];
        memset(m_Vadv, 0, sizeof(Vec2f) * m_fluidCount);

        m_Dadv = new float[m_fluidCount];
        memset(m_Dadv, 0, sizeof(float) * m_fluidCount);

        m_Pl = new float[m_fluidCount];
        memset(m_Pl, 0, sizeof(float) * m_fluidCount);

        m_Dcorr = new float[m_fluidCount];
        memset(m_Dcorr, 0, sizeof(float) * m_fluidCount);

        m_Fadv = new Vec2f[m_fluidCount];
        memset(m_Fadv, 0, sizeof(Vec2f) * m_fluidCount);

        m_Fp = new Vec2f[m_fluidCount];
        memset(m_Fp, 0, sizeof(Vec2f) * m_fluidCount);

        m_fGrid.resize(m_grid.CellCount());
        m_bGrid.resize(m_grid.CellCount());

        m_fNeighbors.resize(m_fluidCount);
        m_bNeighbors.resize(m_fluidCount);
    }

    void SphSolver::_Deallocate()
    {
        delete[] m_fPosition;
        delete[] m_fDensity;
        delete[] m_fVelocity;
        delete[] m_fPressure;

        delete[] m_bPosition;
        delete[] m_Psi;

        delete[] m_Dii;
        delete[] m_Aii;
        delete[] m_sumDijPj;
        delete[] m_Vadv;
        delete[] m_Dadv;
        delete[] m_Pl;
        delete[] m_Dcorr;
        delete[] m_Fadv;
        delete[] m_Fp;

        m_fGrid.clear();
        m_bGrid.clear();

        m_fNeighbors.clear();
        m_bNeighbors.clear();

        m_fluidCount = 0;
        m_boundaryCount = 0;
    }


    //___________________________________________________________
    // Main Simulation Steps

    void SphSolver::_BuildParticleGrid()
    {
        AUTO_CPU_MARKER("Solver::BuildParticleGrid");

        // reserve memory
        for (auto& fIndices : m_fGrid)
        {
            const size_t previousAlloc = fIndices.size();
            fIndices.clear();
            fIndices.reserve(previousAlloc);
        }

        // identify grid cells occupied by fluid particles
        for (int i = 0; i < m_fluidCount; i++)
        {
            const int cellID = m_grid.CellIndex(m_fPosition[i]);

            if (m_grid.Contains(cellID))
            {
                m_fGrid[cellID].push_back(i);
            }
        }

        // reserve memory
        for (auto& bIndices : m_bGrid)
        {
            const size_t previousAlloc = bIndices.size();
            bIndices.clear();
            bIndices.reserve(previousAlloc);
        }

        // identify grid cells occupied by boundary particles
        for (int i = 0; i < m_boundaryCount; i++)
        {
            const int cellID = m_grid.CellIndex(m_bPosition[i]);

            if (m_grid.Contains(cellID))
            {
                m_bGrid[cellID].push_back(i);
            }
        }
    }

    void SphSolver::_SearchNeighbors()
    {
        AUTO_CPU_MARKER("Solver::SearchNeighbors");

        const float searchRadius = 2.f * m_h;
        const float squaredRadius = Math::square(searchRadius);

    #pragma omp parallel for
        for (int i = 0; i < m_fluidCount; i++)
        {
            // reserve memory
            const size_t fPreviousAlloc = m_fNeighbors[i].size();
            m_fNeighbors[i].clear();
            m_fNeighbors[i].reserve(fPreviousAlloc);

            const size_t bPreviousAlloc = m_bNeighbors[i].size();
            m_bNeighbors[i].clear();
            m_bNeighbors[i].reserve(bPreviousAlloc);    

            // gather neighboring cells
            IndexList neighborCells;
            const Vec2f& position = m_fPosition[i];
            m_grid.GatherNeighborCells(neighborCells, position, searchRadius);

            for (const u32 cellID : neighborCells)
            {
                // search for neighboring fluid particles
                for (const u32 neighborID : m_fGrid[cellID]) 
                {
                    const float d2 = Math::normSquared(m_fPosition[neighborID] - position);

                    if (d2 < squaredRadius)
                    {
                        m_fNeighbors[i].push_back(neighborID);
                    }
                }

                // search for neighboring boundary particles
                for (const u32 neighborID : m_bGrid[cellID]) 
                {
                    const float d2 = Math::normSquared(m_bPosition[neighborID] - position);

                    if (d2 < squaredRadius)
                    {
                        m_bNeighbors[i].push_back(neighborID);
                    }
                }
            }
        }
    }

    void SphSolver::_PredictAdvection()
    {
        AUTO_CPU_MARKER("Solver::PredictAdvection");

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
        AUTO_CPU_MARKER("Solver::SolvePressure");

        int iteration = 0;
        float avgDensity = m_rho0 + 2.f * m_eta;

        while (abs(avgDensity - m_rho0) > m_eta && iteration < m_maxIter)
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

            avgDensity = _GetAverageDensity();
            iteration++;
        }
    }

    void SphSolver::_CorrectIntegration()
    {
        AUTO_CPU_MARKER("Solver::CorrectIntegration");

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

    void SphSolver::_PreComputePsi(int i) const
    {
        const float searchRadius = m_h;
        const float squaredRadius = Math::square(searchRadius);

        // gather neighboring cells
        IndexList neighborCells;
        m_grid.GatherNeighborCells(neighborCells, m_bPosition[i], searchRadius);

        // search for neighboring boundary particles
        IndexList neighbors;

        for (const u32 cellID : neighborCells)
        {
            for (const u32 neighborID : m_bGrid[cellID]) 
            {
                const float d2 = Math::normSquared(m_bPosition[neighborID] - m_bPosition[i]);

                if (d2 < squaredRadius)
                {
                    neighbors.push_back(neighborID);
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

    void SphSolver::_ComputeDensity(int i) const
    {
        m_fDensity[i] = 0.f;
        Vec2f pos_ij;

        for (const u32 j : m_fNeighbors[i])
        {
            pos_ij = m_fPosition[i] - m_fPosition[j];
            m_fDensity[i] += m_m0 * m_kernel.W(pos_ij);
        }

        for (const u32 j : m_bNeighbors[i])
        {
            pos_ij = m_fPosition[i] - m_bPosition[j];
            m_fDensity[i] += m_Psi[j] * m_kernel.W(pos_ij);
        }
    }

    void SphSolver::_ComputeAdvectionForces(int i) const
    {
        m_Fadv[i] = Vec2f(0.f, 0.f);

        // add body force
        m_Fadv[i] += m_m0 * m_g;

        // add viscous force
        Vec2f pos_ij;
        Vec2f vel_ij;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                vel_ij = m_fVelocity[i] - m_fVelocity[j];
                m_Fadv[i] += 2.f * m_nu * (Math::square(m_m0) / m_fDensity[j]) * dot(vel_ij, pos_ij) * m_kernel.GradW(pos_ij) / (Math::normSquared(pos_ij) + 0.01f * Math::square(m_h));
            }
        }
    }

    void SphSolver::_PredictVelocity(int i) const
    {
        m_Vadv[i] = m_fVelocity[i] + m_dt * m_Fadv[i] / m_m0;
    }

    void SphSolver::_StoreDii(int i) const
    {
        m_Dii[i] = Vec2f(0.f, 0.f);
        Vec2f pos_ij;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                m_Dii[i] += (-m_m0 / Math::square(m_fDensity[i])) * m_kernel.GradW(pos_ij);
            }
        }

        for (const u32 j : m_bNeighbors[i])
        {
            if (m_bPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Dii[i] += (-m_Psi[j] / Math::square(m_fDensity[i])) * m_kernel.GradW(pos_ij);
            }
        }

        m_Dii[i] *= Math::square(m_dt);
    }

    void SphSolver::_PredictDensity(int i) const
    {
        m_Dadv[i] = 0.f;
        Vec2f pos_ij;
        Vec2f vel_adv_ij;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                vel_adv_ij = m_Vadv[i] - m_Vadv[j];
                m_Dadv[i] += m_m0 * dot(vel_adv_ij, m_kernel.GradW(pos_ij));
            }
        }

        for (const u32 j : m_bNeighbors[i])
        {
            if (m_bPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_bPosition[j];
                vel_adv_ij = m_Vadv[i];
                m_Dadv[i] += m_Psi[j] * dot(vel_adv_ij, m_kernel.GradW(pos_ij));
            }
        }

        m_Dadv[i] *= m_dt;
        m_Dadv[i] += m_fDensity[i];
    }

    void SphSolver::_InitPressure(int i) const
    {
        m_Pl[i] = 0.5f * m_fPressure[i];
    }

    void SphSolver::_StoreAii(int i) const
    {
        m_Aii[i] = 0.f;
        Vec2f pos_ij;
        Vec2f d_ji;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                d_ji = -(Math::square(m_dt) * m_m0 / Math::square(m_fDensity[i])) * (-m_kernel.GradW(pos_ij));
                m_Aii[i] += m_m0 * dot(m_Dii[i] - d_ji, m_kernel.GradW(pos_ij));
            }
        }

        for (const u32 j : m_bNeighbors[i])
        {
            if (m_bPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Aii[i] += m_Psi[j] * dot(m_Dii[i], m_kernel.GradW(pos_ij));
            }
        }
    }


    //___________________________________________________________
    // Pressure Solving Helpers

    void SphSolver::_StoreSumDijPj(int i) const
    {
        m_sumDijPj[i] = Vec2f(0.f, 0.f);
        Vec2f pos_ij;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                m_sumDijPj[i] += -(m_m0 * m_fPressure[j] / Math::square(m_fDensity[j])) * m_kernel.GradW(pos_ij);
            }
        }

        m_sumDijPj[i] *= Math::square(m_dt);
    }

    void SphSolver::_ComputePressure(int i) const
    {
        m_Dcorr[i] = 0.f;
        Vec2f pos_ij;
        Vec2f d_ji;
        Vec2f temp;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                d_ji = -(Math::square(m_dt) * m_m0 / Math::square(m_fDensity[i])) * (-m_kernel.GradW(pos_ij));
                temp = m_sumDijPj[i] - m_Dii[j] * m_Pl[j] - (m_sumDijPj[j] - d_ji * m_Pl[i]);
                m_Dcorr[i] += m_m0 * dot(temp, m_kernel.GradW(pos_ij));
            }
        }

        for (const u32 j : m_bNeighbors[i])
        {
            if (m_bPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Dcorr[i] += m_Psi[j] * dot(m_sumDijPj[i], m_kernel.GradW(pos_ij));
            }
        }

        m_Dcorr[i] += m_Dadv[i];
        float previousPl = m_Pl[i];

        if (std::abs(m_Aii[i]) > FLT_MIN)
        {
            m_Pl[i] = (1 - m_omega) * previousPl + (m_omega / m_Aii[i]) * (m_rho0 - m_Dcorr[i]);
        }
        else
        {
            m_Pl[i] = 0.f;
        }

        m_fPressure[i] = std::fmax(m_Pl[i], 0.f);

        m_Pl[i] = m_fPressure[i];
        m_Dcorr[i] += m_Aii[i] * previousPl;
    }

    float SphSolver::_GetAverageDensity() const
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
    // Integration Helpers

    void SphSolver::_ComputePressureForces(int i) const
    {
        m_Fp[i] = Vec2f(0.f, 0.f);
        Vec2f pos_ij;

        for (const u32 j : m_fNeighbors[i])
        {
            if (m_fPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_fPosition[j];
                m_Fp[i] += -Math::square(m_m0) * (m_fPressure[i] / Math::square(m_fDensity[i]) + m_fPressure[j] / Math::square(m_fDensity[j])) * m_kernel.GradW(pos_ij);
            }
        }

        for (const u32 j : m_bNeighbors[i])
        {
            if (m_bPosition[j] != m_fPosition[i])
            {
                pos_ij = m_fPosition[i] - m_bPosition[j];
                m_Fp[i] += -m_m0 * m_Psi[j] * (m_fPressure[i] / Math::square(m_fDensity[i])) * m_kernel.GradW(pos_ij);
            }
        }
    }

    void SphSolver::_UpdateVelocity(int i) const
    {
        m_fVelocity[i] = m_Vadv[i] + m_dt * m_Fp[i] / m_m0;
    }

    void SphSolver::_UpdatePosition(int i) const
    {
        m_fPosition[i] += m_dt * m_fVelocity[i];
        AVA_ASSERT(m_grid.Contains(m_fPosition[i]), "Particle outside simulation grid");
    }
}