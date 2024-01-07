#pragma once
/// @file Pool.h
/// @brief

#include <Memory/Memory.h>
#include <Debug/Assert.h>

namespace Ava {

    ///////////////////////////////////////////////////////////////////////////////////////////
    /// The PoolBase class is a common implementation base for the templated Pool<T> class. ///
    /// If you are looking for a pool container, take a look at Pool<T> instead.            ///
    ///////////////////////////////////////////////////////////////////////////////////////////
    template <size_t ItemSize, size_t Alignment>
    class PoolBase
    {
        inline static const size_t kItemSize = ItemSize;

    public:
        PoolBase(size_t _itemsPerBlock);
        ~PoolBase();

        PoolBase(PoolBase&& _other) noexcept;
        PoolBase(PoolBase const& _other) = delete;

        PoolBase& operator=(PoolBase&& _other) noexcept;
        PoolBase& operator=(const PoolBase& _other) = delete;

        u8* New();
        void Delete(const u8* _item);

        void clear();
        size_t size() const { return capacity() - m_freeItemCount; }
        size_t capacity() const { return m_blocks.size() * m_itemsPerBlock; }

        static size_t GetItemSize() { return kItemSize; }
        size_t GetItemsPerBlock() const { return m_itemsPerBlock; }

        bool IsAllocated  (const u8* _item) const;

        // low level accessors
        size_t GetBlockCount() const { return m_blocks.size(); }
        u8* GetItem(const size_t _blockIdx, const size_t _itemIdx) { return m_blocks[_blockIdx] + _itemIdx * kItemSize; }
        const u8* GetItem(const size_t _blockIdx, const size_t _itemIdx) const { return m_blocks[_blockIdx] + _itemIdx * kItemSize; }
        bool IsAllocated(size_t _blockIdx, size_t _itemIdx) const;

        /// Number of items per block can only be changed before the Pool is used (ie. capacity is 0)
        void SetItemsPerBlock(size_t _itemsPerBlock);

    protected:
        void _AddBlock();
        void _Delete(size_t _blockIdx, size_t _itemIdx);
        bool _GetItemIdx(const u8* _item, size_t& _blockIdx, size_t& _itemIdx) const;
        void _SetAllocBit(const u8* _item, bool _allocated);
        void _SetAllocBit(size_t _blockIdx, size_t _itemIdx, bool _allocated);

        /// Returns the number of bytes per block required to store allocation bits
        size_t _GetAllocationBytesPerBlock() const;

         /// Stores 1 bit per item to know if the item is allocated
        std::vector<u8>  m_allocatedItems;
        std::vector<u8*> m_blocks;
        size_t m_itemsPerBlock;

        struct FreeItem
        {
            FreeItem* next;
        };

        // chained list of free items
        FreeItem* m_nextFreeItem;
        size_t m_freeItemCount;
    };


    // ----- PoolBase implementation ----------------------------------------------------------

    template <size_t ItemSize, size_t Alignment>
    u8* PoolBase<ItemSize, Alignment>::New()
    {
        if (!m_nextFreeItem)
        {
            _AddBlock();
        }

        FreeItem* item = m_nextFreeItem;
        m_nextFreeItem = item->next;

        m_freeItemCount--;
        _SetAllocBit((u8*)item, true);

        return (u8*)item;
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::Delete(const u8* _item)
    {
        size_t blockIdx, itemIdx;
        if (AVA_VERIFY(_GetItemIdx(_item, blockIdx, itemIdx))) // This item pointer is not in this pool!
        {
            _Delete(blockIdx, itemIdx);
        }
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::_Delete(const size_t _blockIdx, const size_t _itemIdx)
    {
        m_freeItemCount++;
        _SetAllocBit(_blockIdx, _itemIdx, false);

        FreeItem* item = (FreeItem*)(m_blocks[_blockIdx] + _itemIdx * ItemSize);
        item->next = m_nextFreeItem;
        m_nextFreeItem = item;
    }

    template <size_t ItemSize, size_t Alignment>
    bool PoolBase<ItemSize, Alignment>::IsAllocated(const size_t _blockIdx, const size_t _itemIdx) const
    {
        AVA_ASSERT(_blockIdx < m_blocks.size());
        AVA_ASSERT(_itemIdx < m_itemsPerBlock);

        // Number of bytes per block used to store allocation bits
        const size_t allocBytesPerBlock = _GetAllocationBytesPerBlock();

        // Allocation bits for this block start at this index
        const size_t blockAllocIdx = allocBytesPerBlock * _blockIdx;

        // Location of the bit to change
        const size_t byteIdx = _itemIdx / 8;
        const size_t bitIdx  = _itemIdx % 8;

        const u8 allocByte = m_allocatedItems[blockAllocIdx + byteIdx];

        return (allocByte & (1u << bitIdx)) != 0;
    }

    template <size_t ItemSize, size_t Alignment>
    bool PoolBase<ItemSize, Alignment>::IsAllocated(const u8* _item) const
    {
        size_t blockIdx, itemIdx;
        if (AVA_VERIFY(_GetItemIdx(_item, blockIdx, itemIdx))) // This item pointer is not in this pool!
        {
            return IsAllocated(blockIdx, itemIdx);
        }
        return false;
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::_SetAllocBit(const u8* _item, const bool _allocated)
    {
        size_t blockIdx, itemIdx;
        if (AVA_VERIFY(_GetItemIdx(_item, blockIdx, itemIdx))) // This item pointer is not in this pool!
        {
            _SetAllocBit(blockIdx, itemIdx, _allocated);
        }
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::_SetAllocBit(const size_t _blockIdx, const size_t _itemIdx, const bool _allocated)
    {
        // Number of bytes per block used to store allocation bits
        const size_t allocBytesPerBlock = _GetAllocationBytesPerBlock();

        // Allocation bits for this block start at this byte index
        const size_t blockAllocIdx = allocBytesPerBlock * _blockIdx;

        // Location of the bit to change
        const size_t byteIdx = _itemIdx / 8;
        const size_t bitIdx  = _itemIdx % 8;

        u8& allocByte = m_allocatedItems[blockAllocIdx + byteIdx];

        if (_allocated)
        {
            // Set the bit to 1 (= allocated)
            AVA_ASSERT((allocByte & (1u << bitIdx)) == 0); // This item was already allocated!
            allocByte |= (1u << bitIdx);
        }
        else
        {
            // Set the bit to 0 (= free)
            AVA_ASSERT((allocByte & (1u << bitIdx)) != 0); // This item was already destroyed!
            allocByte &= ~(1u << bitIdx);
        }
    }

    template <size_t ItemSize, size_t Alignment>
    bool PoolBase<ItemSize, Alignment>::_GetItemIdx(const u8* _item, size_t& _blockIdx, size_t& _itemIdx) const
    {
        // Find the block containing this item
        for (auto& block : m_blocks)
        {
            if (_item >= block
                && _item < (block + ItemSize * m_itemsPerBlock))
            {
                // Found the block
                _blockIdx = &block - m_blocks.data();
                _itemIdx = (_item - block) / ItemSize;

                return true;
            }
        }
        return false;
    }

    template <size_t ItemSize, size_t Alignment>
    size_t PoolBase<ItemSize, Alignment>::_GetAllocationBytesPerBlock() const
    {
        return (m_itemsPerBlock / 8u) + 1u;
    }

    template <size_t ItemSize, size_t Alignment>
    PoolBase<ItemSize, Alignment>::PoolBase(const size_t _itemsPerBlock)
        : m_itemsPerBlock(_itemsPerBlock)
        , m_nextFreeItem(nullptr)
        , m_freeItemCount(0)
    {
        AVA_ASSERT(ItemSize >= sizeof(u8*));
        AVA_ASSERT(_itemsPerBlock > 0);
    }

    template <size_t ItemSize, size_t Alignment>
    PoolBase<ItemSize, Alignment>::~PoolBase()
    {
        // No need to call clear() here, just free the blocks
        for (size_t i = 0; i < m_blocks.size(); ++i)
        {
            AVA_FREE_ALIGNED(m_blocks[i]);
        }
    }

    template <size_t ItemSize, size_t Alignment>
    PoolBase<ItemSize, Alignment>::PoolBase(PoolBase&& _other) noexcept
    {
        m_allocatedItems = std::move(_other.m_allocatedItems);
        m_blocks = std::move(_other.m_blocks);

        m_itemsPerBlock = _other.m_itemsPerBlock;
        m_freeItemCount = _other.m_freeItemCount;

        m_nextFreeItem = _other.m_nextFreeItem;
        _other.m_nextFreeItem = nullptr;
    }

    template <size_t ItemSize, size_t Alignment>
    PoolBase<ItemSize, Alignment>& PoolBase<ItemSize, Alignment>::operator=(PoolBase&& _other) noexcept
    {
        m_allocatedItems = std::move(_other.m_allocatedItems);
        m_blocks = std::move(_other.m_blocks);

        m_itemsPerBlock = _other.m_itemsPerBlock;
        m_freeItemCount = _other.m_freeItemCount;

        m_nextFreeItem = _other.m_nextFreeItem;
        _other.m_nextFreeItem = nullptr;

        return *this;
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::SetItemsPerBlock(const size_t _itemsPerBlock)
    {
        AVA_ASSERT(capacity() == 0);
        m_itemsPerBlock = _itemsPerBlock;
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::clear()
    {
        for (size_t blockIdx = 0; blockIdx < m_blocks.size(); ++blockIdx)
        {
            for (size_t itemIdx = 0; itemIdx < m_itemsPerBlock; ++itemIdx)
            {
                if (IsAllocated(blockIdx, itemIdx))
                {
                    _Delete(blockIdx, itemIdx);
                }
            }
        }
    }

    template <size_t ItemSize, size_t Alignment>
    void PoolBase<ItemSize, Alignment>::_AddBlock()
    {
        m_freeItemCount += m_itemsPerBlock;
        m_allocatedItems.resize(m_allocatedItems.size() + _GetAllocationBytesPerBlock(), 0);

        auto* newBlock = (u8*)AVA_MALLOC_ALIGNED(ItemSize * m_itemsPerBlock, Alignment);
        m_blocks.push_back(newBlock);

        for (size_t i = 0; i < m_itemsPerBlock - 1; ++i)
        {
            FreeItem* item0 = (FreeItem*)(newBlock + i * ItemSize);
            FreeItem* item1 = (FreeItem*)(newBlock + (i + 1) * ItemSize);

            if (item0)
            {
                item0->next = item1;
            }
        }

        if (auto* lastItem = (FreeItem*)(newBlock + (m_itemsPerBlock - 1) * ItemSize))
        {
            lastItem->next = m_nextFreeItem;
        }

        m_nextFreeItem = (FreeItem*)newBlock;
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /// The Pool class allows contiguous storage of items with fast and                                   ///
    /// (almost) constant time allocation and deallocation of items.                                      ///
    /// It also allows to iterate over allocated items with iterators.                                    ///
    ///                                                                                                   ///
    /// Implementation info:                                                                              ///
    /// Items are stored inside "blocks" of memory. Blocks are not contiguous in memory,                  ///
    /// but items inside them are. The pool also allocates 1 bit per item in a separate buffer            ///
    /// to keep track of allocated items. For simplicity reasons, blocks are never de-allocated           ///
    /// (except when the pool is destroyed). Free items are aliased with pointers to the next             ///
    /// free item for fast allocation/deallocation.                                                       ///
    ///                                                                                                   ///
    /// @note For best performance, choose a value of _itemsPerBlock that keeps the number of blocks low. ///
    /// @note Iteration can be relatively slow if capacity() is much greater than size()                  ///
    /// because the iterators internally iterate over all items to find allocated ones.                   ///
    /// @note Storing items smaller than a pointer is not supported (and would waste a lot of memory).    ///
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <class T, size_t Alignment = alignof(T)>
    class Pool 
        : public PoolBase<
        sizeof(T) - (sizeof(T) % Alignment) + ((sizeof(T) % Alignment) ? Alignment : 0), // sizeof(T) + padding to make each element aligned
        Alignment
        >
    {
        typedef PoolBase<
            sizeof(T) - (sizeof(T) % Alignment) + ((sizeof(T) % Alignment) ? Alignment : 0),
            Alignment
        > Base;

    public:
        Pool(size_t _itemsPerBlock);
        ~Pool();

        Pool(Pool&& _other) noexcept;
        Pool(const Pool& _other) = delete;

        Pool& operator=(Pool&& _other) noexcept;
        Pool& operator=(const Pool& _other)  = delete;

        class iterator;
        class const_iterator;

        T* Allocate();
        void Deallocate(T* _item);
        T* New();
        template <class ... Args>
        T* New(Args... _args);
        void Delete(T* _item);

        void clear();
        size_t size() const;
        size_t capacity() const;

        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;

        void DeleteItems(iterator _it);
        void DeleteItems(T* _item);

        bool IsAllocated (const T* _item) const { return Base::IsAllocated((const u8*)_item); }
        bool IsAllocated (const_iterator _it)   const;
        bool IsAllocated(size_t _blockIdx, size_t _itemIdx) const { return Base::IsAllocated(_blockIdx, _itemIdx); }

        // Low level accessors - do NOT use items that are not allocated
        T&       GetItem (size_t _blockIdx, size_t _itemIdx) { return *(T*)Base::GetItem(_blockIdx, _itemIdx); }
        const T& GetItem (size_t _blockIdx, size_t _itemIdx) const { return *(const T*)Base::GetItem(_blockIdx, _itemIdx); }
        size_t   GetBlockCount() const { return Base::GetBlockCount(); }
        size_t   GetItemsPerBlock() const { return Base::GetItemsPerBlock(); }
    };


    // ----- Pool implementation ----------------------------------------------------------------------------

    template <class T, size_t Alignment>
    class Pool<T, Alignment>::iterator
    {
        friend class Pool<T, Alignment>;

    public:
        iterator()
            : m_pool(nullptr)
            , m_blockIdx(0)
            , m_itemIdx(0)
        {}

        iterator(Pool<T, Alignment>* _pool, const size_t _blockIdx, const size_t _itemIdx)
            : m_pool(_pool)
            , m_blockIdx(_blockIdx)
            , m_itemIdx(_itemIdx)
        {}

        T& operator* () const
        {
            AVA_ASSERT(m_pool->IsAllocated(*this));
            return m_pool->GetItem(m_blockIdx, m_itemIdx);
        }

        T* operator->() const
        {
            AVA_ASSERT(m_pool->IsAllocated(*this));
            return &m_pool->GetItem(m_blockIdx, m_itemIdx);
        }

        iterator operator++ (int)
        {
            iterator it = *this;
            ++(*this);
            return it;
        }

        iterator& operator++ ()
        {
            AVA_ASSERT(m_blockIdx < m_pool->GetBlockCount() && m_itemIdx < m_pool->GetItemsPerBlock());

            const size_t itemsPerBlock = m_pool->GetItemsPerBlock();
            const size_t blockCount = m_pool->GetBlockCount();

            size_t itemIdx  = m_itemIdx;
            size_t blockIdx = m_blockIdx;

            do 
            {
                itemIdx++;
                if (itemIdx == itemsPerBlock)
                {
                    itemIdx = 0;
                    blockIdx++;
                }
            }
            while (blockIdx < blockCount // reached end()
                && !m_pool->IsAllocated(blockIdx, itemIdx));

            m_itemIdx = itemIdx;
            m_blockIdx = blockIdx;

            return *this;
        }

        bool operator==(const iterator& _rhs) const
        {
            AVA_ASSERT(m_pool == _rhs.m_pool);
            return m_blockIdx == _rhs.m_blockIdx 
                && m_itemIdx == _rhs.m_itemIdx;
        }

        bool operator!=(const iterator& _rhs) const
        {
            return !(*this == _rhs);
        }

    private:
        Pool<T, Alignment>* m_pool;
        size_t m_blockIdx;
        size_t m_itemIdx;
    };

    template <class T, size_t Alignment>
    class Pool<T, Alignment>::const_iterator
    {
        friend class Pool<T, Alignment>;

    public:
        const_iterator()
            : m_pool(nullptr)
            , m_blockIdx(0)
            , m_itemIdx(0)
        {}

        const_iterator(iterator _iterator)
            : m_pool(_iterator.m_pool)
            , m_blockIdx(_iterator.m_blockIdx)
            , m_itemIdx(_iterator.m_itemIdx)
        {}

        const_iterator(Pool<T, Alignment> const* _pool, const size_t _blockIdx, const size_t _itemIdx)
            : m_pool(_pool)
            , m_blockIdx(_blockIdx)
            , m_itemIdx(_itemIdx)
        {}

        T const& operator* () const
        {
            AVA_ASSERT(m_pool->IsAllocated(*this));
            return m_pool->GetItem(m_blockIdx, m_itemIdx);
        }

        T const* operator->() const
        {
            AVA_ASSERT(m_pool->IsAllocated(*this));
            return &m_pool->GetItem(m_blockIdx, m_itemIdx);
        }

        const_iterator operator++ (int)
        {
            const_iterator it = *this;
            ++(*this);
            return it;
        }

        const_iterator& operator++ ()
        {
            AVA_ASSERT(m_blockIdx < m_pool->GetBlockCount() && m_itemIdx < m_pool->GetItemsPerBlock());

            const size_t itemsPerBlock = m_pool->GetItemsPerBlock();
            const size_t blockCount = m_pool->GetBlockCount();

            size_t itemIdx = m_itemIdx;
            size_t blockIdx = m_blockIdx;

            do
            {
                itemIdx++;
                if (itemIdx == itemsPerBlock)
                {
                    itemIdx = 0;
                    blockIdx++;
                }
            } while (blockIdx < blockCount // reached end()
                && !m_pool->isAllocated(blockIdx, itemIdx));

            m_itemIdx = itemIdx;
            m_blockIdx = blockIdx;

            return *this;
        }

        bool operator==(const const_iterator& _other) const
        {
            AVA_ASSERT(m_pool == _other.m_pool);
            return m_blockIdx == _other.m_blockIdx
                && m_itemIdx == _other.m_itemIdx;
        }

        bool operator!=(const const_iterator& _rhs) const
        {
            return !(*this == _rhs);
        }

    private:
        Pool<T, Alignment> const*  m_pool;
        size_t m_blockIdx;
        size_t m_itemIdx;
    };

    template <class T, size_t Alignment>
    Pool<T, Alignment>::Pool( size_t _itemsPerBlock ) : Base(_itemsPerBlock) {}

    template <class T, size_t Alignment>
    Pool<T, Alignment>::~Pool() { clear(); }

    template <class T, size_t Alignment>
    Pool<T, Alignment>::Pool(Pool&& _other) noexcept : Base(std::move(_other)) {}

    template <class T, size_t Alignment>
    Pool<T, Alignment>& Pool<T, Alignment>::operator=(Pool&& _other) noexcept
    {
        return static_cast<Pool&>(Base::operator=(std::move(_other)));
    }

    template <class T, size_t Alignment>
    void Pool<T, Alignment>::clear()
    {
        for (iterator it = begin(); it != end(); ++it)
        {
            DeleteItems(it);
        }
    }

    template <class T, size_t Alignment>
    T* Pool<T, Alignment>::Allocate()
    {
        return reinterpret_cast<T*>(Base::New());
    }

    template <class T, size_t Alignment>
    void Pool<T, Alignment>::Deallocate(T* _item)
    {
        Base::Delete(reinterpret_cast<u8*>(_item));
    }

    template <class T, size_t Alignment>
    T* Pool<T, Alignment>::New()
    {
        T* item = Allocate();
        new (item) T; // no brackets = no zero initialization
        return item;
    }

    template <class T, size_t Alignment>
    template <class ... Args>
    T* Pool<T, Alignment>::New( Args ... _args )
    {
        T* item = Allocate();
        new (item) T(_args...);
        return item;
    }

    template <class T, size_t Alignment>
    void Pool<T, Alignment>::Delete( T* _item )
    {
        _item->~T();
        return Deallocate(_item);
    }

    template <class T, size_t Alignment>
    size_t Pool<T, Alignment>::capacity() const
    {
        return Base::capacity();
    }

    template <class T, size_t Alignment>
    size_t Pool<T, Alignment>::size() const
    {
        return Base::size();
    }

    template <class T, size_t Alignment>
    typename Pool<T, Alignment>::iterator Pool<T, Alignment>::begin()
    {
        // If pool empty just return end()
        if (size() == 0)
            return end();

        // Else, search the first allocated item
        iterator it(this, 0, 0);

        if (!IsAllocated(it))
        {
            ++it;
        }
        
        return it;
    }

    template <class T, size_t Alignment>
    typename Pool<T, Alignment>::iterator Pool<T, Alignment>::end()
    {
        return iterator(this, GetBlockCount(), 0);
    }

    template <class T, size_t Alignment>
    typename Pool<T, Alignment>::const_iterator Pool<T, Alignment>::begin() const
    {
        // If pool empty just return end()
        if (size() == 0)
            return end();

        // Else, search the first allocated item
        const_iterator it(this, 0, 0);

        if (!IsAllocated(it))
        {
            ++it;
        }

        return it;
    }

    template <class T, size_t Alignment>
    typename Pool<T, Alignment>::const_iterator Pool<T, Alignment>::end() const
    {
        return const_iterator(this, GetBlockCount(), 0);
    }

    template <class T, size_t Alignment>
    void Pool<T, Alignment>::DeleteItems(iterator _it)
    {
        T* item = &*_it;
        item->~T();
        return Base::_Delete(_it.m_blockIdx, _it.m_itemIdx);
    }

    template <class T, size_t Alignment>
    void Pool<T, Alignment>::DeleteItems(T* _item)
    {
        _item->~T();
        return Base::Delete(reinterpret_cast<u8*>(_item));
    }

    template <class T, size_t Alignment>
    bool Pool<T, Alignment>::IsAllocated(const_iterator _it) const
    {
        return Base::IsAllocated(_it.m_blockIdx, _it.m_itemIdx);
    }

    template <typename T, size_t Alignment, typename MutexType>
    auto begin(Pool<T, Alignment>& _pool)
    {
        return _pool.begin();
    }
    
    template <typename T, size_t Alignment, typename MutexType>
    auto end(Pool<T, Alignment>& _pool)
    {
        return _pool.end();
    }

    template <typename T, size_t Alignment, typename MutexType>
    auto begin(Pool<T, Alignment> const& _pool)
    {
        return _pool.begin();
    }

    template <typename T, size_t Alignment, typename MutexType>
    auto end(Pool<T, Alignment> const& _pool)
    {
        return _pool.end();
    }
}