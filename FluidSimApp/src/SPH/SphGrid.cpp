#include "SphGrid.h"

namespace sph
{
    SphGrid::SphGrid(float _cellSize, const Vec2f& _gridSize)
    {
        m_cellSize = _cellSize;
        m_gridSize = _gridSize;

        m_gridCells.x = (int)std::floor(_gridSize.x / _cellSize);
        m_gridCells.y = (int)std::floor(_gridSize.y / _cellSize);
    }

    u32 SphGrid::CellIndex(const Vec2f& _particle) const
    {
        const Vec2i cell = CellPosition(_particle);
        return CellIndex(cell.x, cell.y);
    }

    u32 SphGrid::CellIndex(int _i, int _j) const
    {
        return _i + _j * m_gridCells.x;
        // return _i * m_gridCells.y + _j;
    }

    bool SphGrid::Contains(const Vec2f& _particle) const
    {
        const u32 idx = CellIndex(_particle);
        return Contains(idx);
    }

    bool SphGrid::Contains(int _cellIdx) const
    {
        return _cellIdx >= 0 && _cellIdx < CellCount();
    }

    Vec2i SphGrid::CellPosition(const Vec2f& _particle) const
    {
        Vec2i cell;
        cell.x = (int)std::floor(_particle.x / m_cellSize);
        cell.y = (int)std::floor(_particle.y / m_cellSize);
        return cell;
    }

    int SphGrid::CellCount() const
    {
        return m_gridCells.x * m_gridCells.y;
    }

    float SphGrid::CellSize() const
    {
        return m_cellSize;
    }

    Vec2f SphGrid::GridSize() const
    {
        return m_gridSize;
    }

    Vec2i SphGrid::GridCells() const
    {
        return m_gridCells;
    }

    void SphGrid::GatherNeighborCells(std::vector<u32>& _cellIndices, const Vec2f& _particle, float _radius) const
    {
        AVA_ASSERT(Contains(_particle), "Particle outisde the grid");
        _cellIndices.clear();

        const Vec2i minCell = CellPosition(_particle - _radius);
        const Vec2i maxCell = CellPosition(_particle + _radius);

        const int iMin = std::max(minCell.x, 0);
        const int iMax = std::min(maxCell.x, m_gridCells.x - 1);
        const int jMin = std::max(minCell.y, 0);
        const int jMax = std::min(maxCell.y, m_gridCells.y - 1);

        int count = 0;
        const int nbNeighbors = (jMax - jMin + 1) * (iMax - iMin + 1);
        AVA_ASSERT(nbNeighbors >= 0);

        _cellIndices.resize(nbNeighbors);

        for (int i = iMin; i <= iMax; ++i)
        {
            for (int j = jMin; j <= jMax; ++j)
            {
                _cellIndices[count] = CellIndex(i, j);
                count++;
            }
        }
    }
}
