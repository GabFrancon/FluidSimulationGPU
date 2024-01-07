#pragma once
/// @file Quadtree.h
/// @brief

/*******************************
// Generic linear quadtree.
//
// IndexType is the type used for indexing nodes and determines the absolute max
// level of subdivision possible. This should be a unsigned integer type (uint8,
// uint16, uint32, uint64).
//
// NodeType is the node type. Typically this will be a pointer or index into a
// separate node data pool. Use the _initNodeValue argument of the constructor
// to init the quadtree with 'invalid' nodes.
//
// Internally, each level is stored sequentially with the root level at index 0.
// Within each level, nodes are laid out in Morton order:
//  +---+---+
//  | 0 | 2 |
//  +---+---+
//  | 1 | 3 |
//  +---+---+
********************************/

#include <Core/Base.h>
#include <Math/Math.h>

namespace Ava {

    /// @brief Generic linear quadtree data structure
    template <typename IndexType, typename NodeType>
    class Quadtree
    {
    public:
        // To make sure provided IndexType is valid
        static_assert(std::is_integral_v<IndexType>, "IndexType must be integral.");
        static_assert(std::is_unsigned_v<IndexType>, "IndexType must be unsigned.");
        static_assert(sizeof(IndexType) <= 4, "IndexType must be <= 4 bytes");

        using Index = IndexType;
        using Node = NodeType;

        // To mark invalid indices
        static constexpr Index kNodeIdxInvalid = ~Index(0);

        Quadtree(int _levelCount = _GetAbsoluteMaxLevelCount(), Node _initNodeValue = Node());
        ~Quadtree();

        /// @brief Returns the width in leaf nodes at given quadtree level.
        Index GetNodeWidth(const int _level) { return _GetNodeWidth(Math::max(m_levelCount - _level - 1, 0)); }
        /// @brief Returns the number of nodes at the given quadtree level.
        Index GetNodeCount(const int _level) const { return _GetNodeCount(_level); }
        /// @brief Returns the total number of nodes in the quadtree.
        int GetTotalNodeCount() const { return _GetTotalNodeCount(m_levelCount); }
        /// @brief Returns the number of levels in the quadtree.
        int GetLevelCount() const { return m_levelCount; }

        /// @brief Returns the index of the parent of _childIndex.
        Index GetParentIndex(Index _childIndex, int _childLevel) const;
        /// @brief Returns the index of the first child of _parentIndex.
        Index GetFirstChildIndex(Index _parentIndex, int _parentLevel) const;
        /// @brief Searches a valid neighbor place at (_offsetX, _offsetY) from _nodeIndex.
        Index FindValidNeighbor(Index _nodeIndex, int _level, int _offsetX, int _offsetY, Node _invalidNodeValue);

        /// @brief Performs a depth-first traversal of the quadtree, starting at _rootIndex and calling _onVisit for each node.
        /// The traversal proceeds to the children nodes only if _onVisit(parentNode) returns true.
        template <typename OnVisitFunc>
        void Traverse(OnVisitFunc&& _onVisit, Index _rootIndex = 0);

        // Level access.
        const Node* GetLevel(const int _levelIndex) const     { return m_nodes + _GetLevelStartIndex(_levelIndex); }
        Node*       GetLevel(const int _levelIndex)           { return m_nodes + _GetLevelStartIndex(_levelIndex); }

        // Node access.
        Node&       operator[](Index _index)          { return m_nodes[_index]; }
        const Node& operator[](Index _index) const    { return m_nodes[_index]; }
        Index       GetIndex(const Node& _node) const { return &_node - m_nodes; }

    private:
        std::vector<Node> m_nodes;
        int m_levelCount;

        /// @brief Returns the max number of levels given the number of index bits.
        static constexpr int _GetAbsoluteMaxLevelCount() { return static_cast<int>(sizeof(Index) * CHAR_BIT) / 2; }
        /// @brief Returns the width in leaf nodes at given quadtree level [ = sqrt(4^_level) ]
        static constexpr Index _GetNodeWidth(const int _level) { return 1 << _level; }
        /// @brief Returns the number of nodes at given quadtree level [ = 4^_level ]
        static constexpr Index _GetNodeCount(const int _level) { return 1 << (2 * _level); }
        /// @brief Returns the total number of nodes in a quadtree with given level depth [ = 1 + 4 * (leafCount-1) / 3 ]
        static constexpr Index _GetTotalNodeCount(const int _levelCount) { return 1 + 4 * (_GetNodeCount(_levelCount - 1) - 1) / 3; }
        /// @brief Returns the index of the first node at given quadtree level
        static constexpr Index _GetLevelStartIndex(const int _level) { return (_level > 0) ? _GetTotalNodeCount(_level) : 0; }

        /// @brief Converts _nodeIndex to a cartesian offset relative to quadtree root.
        static Vec2i _ToCartesian(Index _nodeIndex, int _level);
        /// @brief Converts (_x, _y) cartesian coordinates to a node index.
        static Index _ToIndex(Index _x, Index _y, int _level);
        /// @brief Returns the neighbor placed at (_offsetX, _offsetY) from _nodeIndex, or kNodeIdxInvalid if we are outside the tree.
        static Index _FindNeighbor(Index _nodeIndex, int _level, int _offsetX, int _offsetY);
        /// @brief Returns the quadtree level where is _nodeIndex is located.
        static int _FindLevel(Index _nodeIndex);
    };

    // ---- Quadtree static helpers -------------------------------------------------------------

    template <typename IndexType, typename NodeType>
    Vec2i Quadtree<IndexType, NodeType>::_ToCartesian(Index _nodeIndex, const int _level)
    {
        // traverse the index LSB -> MSB summing node width (= number of leaf nodes covered)
        Index nodeWidth = 1;
        Index nodeIndex = _nodeIndex - _GetLevelStartIndex(_level);
        Vec2i cartesianOffset = {0, 0};

        for (int i = 0; i < _level; i++, nodeWidth *= 2)
        {
            cartesianOffset.y += (nodeIndex & 1) * nodeWidth;
            nodeIndex = nodeIndex >> 1;

            cartesianOffset.x += (_nodeIndex & 1) * nodeWidth;
            nodeIndex = nodeIndex >> 1;
        }

        return cartesianOffset;
    }

    template <typename IndexType, typename NodeType>
    typename Quadtree<IndexType, NodeType>::Index Quadtree<IndexType, NodeType>::_ToIndex(Index _x, Index _y, const int _level)
    {
        // checks we are inside the quadtree
        Index width = GetNodeWidth(_level);
        if (_x > width || _y > width)
        {
            return kNodeIdxInvalid;
        }

        // interleaves _x and _y to produce Morton code
        Index nodeIndex = 0;
        for (int i = 0; i < sizeof(Index) * CHAR_BIT; i++)
        {
            nodeIndex = nodeIndex | (_y & 1 << i) << i | (_x & 1 << i) << (i + 1);
        }

        return nodeIndex + _GetLevelStartIndex(_level);
    }

    template <typename IndexType, typename NodeType>
    typename Quadtree<IndexType, NodeType>::Index Quadtree<IndexType, NodeType>::_FindNeighbor(Index _nodeIndex, int _level, const int _offsetX, const int _offsetY)
    {
        if (_nodeIndex == kNodeIdxInvalid)
        {
            return kNodeIdxInvalid;
        }

        Vec2i offset = _ToCartesian(_nodeIndex, _level) + Vec2i(_offsetX, _offsetY);
        return _ToIndex(offset.x, offset.y, _level);
    }

    template <typename IndexType, typename NodeType>
    int Quadtree<IndexType, NodeType>::_FindLevel(Index _nodeIndex)
    {
        for (int i = 0; i < _GetAbsoluteMaxLevelCount(); i++)
        {
            if (_nodeIndex < _GetLevelStartIndex(i + 1))
            {
                return i;
            }
        }

        return -1;
    }


    // ---- Quadtree implementation -------------------------------------------------------------

    template <typename IndexType, typename NodeType>
    Quadtree<IndexType, NodeType>::Quadtree(const int _levelCount, Node _initNodeValue) : m_levelCount(_levelCount)
    {
        AVA_ASSERT(_levelCount <= _GetAbsoluteMaxLevelCount(), 
            "Not enough bits available in IndexType to hold %d levels", _levelCount);

        const Index totalNodeCount = _GetTotalNodeCount(_levelCount);
        m_nodes.reserve(totalNodeCount);

        for (Index i = 0; i < totalNodeCount; ++i)
        {
            m_nodes.emplace_back(_initNodeValue);
        }
    }

    template <typename IndexType, typename NodeType>
    Quadtree<IndexType, NodeType>::~Quadtree()
    {
        m_nodes.clear();
    }

    template <typename IndexType, typename NodeType>
    typename Quadtree<IndexType, NodeType>::Index Quadtree<IndexType, NodeType>::GetParentIndex(Index _childIndex, const int _childLevel) const
    {
        // root node doesn't have a parent
        if (_childIndex == 0)
        {
            return kNodeIdxInvalid;
        }

        Index childOffset = _GetLevelStartIndex(_childLevel);
        Index parentOffset = _GetLevelStartIndex(Math::max(_childLevel - 1, 0));

        return parentOffset + ((_childIndex - childOffset) >> 2);
    }

    template <typename IndexType, typename NodeType>
    typename Quadtree<IndexType, NodeType>::Index Quadtree<IndexType, NodeType>::GetFirstChildIndex(Index _parentIndex, const int _parentLevel) const
    {
        // leaf nodes don't have children
        if (_parentLevel >= m_levelCount - 1)
        {
            return kNodeIdxInvalid;
        }

        Index parentOffset = _GetLevelStartIndex(_parentLevel);
        Index childOffset = _GetLevelStartIndex(_parentLevel + 1);

        return childOffset + ((_parentIndex - parentOffset) << 2);
    }

    template <typename IndexType, typename NodeType>
    typename Quadtree<IndexType, NodeType>::Index Quadtree<IndexType, NodeType>::FindValidNeighbor(Index _nodeIndex, const int _level, const int _offsetX, const int _offsetY, Node _invalidNodeValue)
    {
        // First, search for a valid neighbor at the same level
        int level = _level;
        Index neighbor = _FindNeighbor(_nodeIndex, level, _offsetX, _offsetY);

        // Then search-up the quadtree until a valid neighbor is found
        while (neighbor != kNodeIdxInvalid && m_nodes[neighbor] == _invalidNodeValue)
        {
           neighbor = GetParentIndex(neighbor, level--);
        }

        return neighbor;
    }

    template <typename IndexType, typename NodeType>
    template <typename OnVisitFunc>
    void Quadtree<IndexType, NodeType>::Traverse(OnVisitFunc&& _onVisit, Index _rootIndex)
    {
        struct NodeAddress
        {
            Index index;
            int level;
        };

        std::vector<NodeAddress> nodeStack;
        nodeStack.push_back({ _rootIndex, _FindLevel(_rootIndex) });

        while (!nodeStack.empty())
        {
            auto node = nodeStack.back();
            nodeStack.pop_back();

            if (_onVisit(node.index, node.level) && node.level < m_levelCount - 1)
            {
                Index i = GetFirstChildIndex(node.index, node.level);

                nodeStack.push_back({ Index(i + 0), node.level + 1 });
                nodeStack.push_back({ Index(i + 1), node.level + 1 });
                nodeStack.push_back({ Index(i + 2), node.level + 1 });
                nodeStack.push_back({ Index(i + 3), node.level + 1 });
            }
        }
    }

}
