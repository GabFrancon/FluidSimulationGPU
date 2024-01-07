#pragma once
/// @file ByteAllocator.h
/// @brief

#include <Core/Base.h>

namespace Ava {

    /// @brief Helper to dynamically allocate bytes in a buffer
    /// and get the result offset in whatever granularity you want.
    class ByteAllocatorBase
    {
    public:
        ByteAllocatorBase(const int _allocGranularity)
            : m_allocGranularity(_allocGranularity)
        {
        }

        void Init(u32 _byteSize, int _deallocWaitingFrames);
        void Shutdown();
        void NextFrame();

        u32 Allocate(u32 _byteSize);
        void Deallocate(u32 _offset);

        // expressed in bytes
        u32 Size() const;
        u32 Capacity() const;

    private:
        void _ProcessDealloc(u32 _offset);

        struct Node
        {
            u32 offset;
            u32 size;

            struct ByOffset { bool operator() (const Node& _a, const Node& _b) const { return _a.offset < _b.offset; } };
            struct BySize { bool operator() (const Node& _a, const Node& _b) const { return _a.size != _b.size ? _a.size < _b.size : _a.offset < _b.offset; } };
        };

        std::set<Node, Node::ByOffset> m_freeByOffset;
        std::set<Node, Node::BySize>   m_freeBySize;
        std::set<Node, Node::ByOffset> m_allocated;

        std::vector<u32>* m_waitingDealloc = nullptr;
        std::vector< std::vector<u32> > m_waitingLists;

        int m_bytesCapacity = 0;
        int m_allocGranularity = 4;
    };


    /// @brief Templated version of ByteAllocatorBase.
    template<typename T>
    class ByteAllocator : public ByteAllocatorBase
    {
    public:
        ByteAllocator() : ByteAllocatorBase(sizeof T)
        {
        }
    };

}