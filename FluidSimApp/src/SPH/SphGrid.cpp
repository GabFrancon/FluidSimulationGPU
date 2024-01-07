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

    void SphGrid::GatherNeighborCells(IndexList& _neighbors, const Vec2f& _particle, float _radius) const
    {
        _neighbors.clear();

        if (!Contains(_particle))
        {
            return;
        }

        const Vec2i minCell = CellPosition(_particle - _radius);
        const Vec2i maxCell = CellPosition(_particle + _radius);

        const int iMin = std::max(minCell.x, 0);
        const int iMax = std::min(maxCell.x, m_gridCells.x - 1);
        const int jMin = std::max(minCell.y, 0);
        const int jMax = std::min(maxCell.y, m_gridCells.y - 1);

        int count = 0;
        const int nbNeighbors = (jMax - jMin + 1) * (iMax - iMin + 1);

        _neighbors.resize(nbNeighbors);

        for (int j = jMin; j <= jMax; ++j)
        {
            for (int i = iMin; i <= iMax; ++i)
            {
                _neighbors[count] = CellIndex(i, j);
                count++;
            }
        }
    }
}
