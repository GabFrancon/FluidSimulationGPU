#pragma once

#include <Math/Math.h>
using namespace Ava;

namespace sph
{
    using IndexList = std::vector<u32>;

    class SphGrid
    {
    public:
        SphGrid() = default;
        SphGrid(float _cellSize, const Vec2f& _gridSize);

        u32 CellIndex(const Vec2f& _particle) const;
        u32 CellIndex(int _i, int _j) const;

        bool Contains(const Vec2f& _particle) const;
        bool Contains(int _cellIdx) const;

        Vec2i CellPosition(const Vec2f& _particle) const;
        int CellCount() const;
        float CellSize() const;

        void GatherNeighborCells(IndexList& _neighbors, const Vec2f& _particle, float _radius) const;

    private:
        Vec2i m_gridCells;
        Vec2f m_gridSize;
        float  m_cellSize;
    };
}