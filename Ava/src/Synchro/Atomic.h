#pragma once
/// @file Atomic.h
/// @brief

#include <Core/Base.h>
#include <atomic>

// Different memory order access :
//    - Relaxed         --> no ordering constraints imposed on other reads or writes.
//    - Acquire         --> no reads or writes in the current thread can be reordered before this load.
//    - Release         --> no reads or writes in the current thread can be reordered after this store.
//    - Seq. consistent --> a load performs an acquire operation, a store performs a release operation.

// Different atomic operations :
//    - Load      --> atomically obtains the value of the atomic object.
//    - Store     --> atomically replaces the value of the atomic object with a non-atomic argument.
//    - Fetch add --> atomically adds the argument to the value stored in the atomic object and obtains the value held previously.
//    - Exchange  --> atomically replaces the value of the atomic object and obtains the value held previously.
//    - Compare   --> atomically compares the value of the atomic object with non-atomic argument and performs exchange if equal or load if not.

namespace Ava {

    /********************************
     *        Atomic integer        *
     ********************************/

    /// @brief Atomic integer.
    template <class Integer>
    class Atomic
    {
    public:
        explicit Atomic(Integer _value = 0) : m_atomic(_value) {}

        // Load
        Integer Load_Relaxed() const { return m_atomic.load(std::memory_order_relaxed); }
        Integer Load_Acquire() const { return m_atomic.load(std::memory_order_acquire); }
        Integer Load_SeqCst () const { return m_atomic.load(std::memory_order_seq_cst); }

        // Store
        void Store_Relaxed(Integer _value) { m_atomic.store(_value, std::memory_order_relaxed); }
        void Store_Release(Integer _value) { m_atomic.store(_value, std::memory_order_release); }
        void Store_SeqCst (Integer _value) { m_atomic.store(_value, std::memory_order_seq_cst); }

        // Fetch
        Integer FetchAdd_Relaxed(Integer _value) { return m_atomic.fetch_add(_value, std::memory_order_relaxed); }
        Integer FetchAdd_Acquire(Integer _value) { return m_atomic.fetch_add(_value, std::memory_order_acquire); }
        Integer FetchAdd_Release(Integer _value) { return m_atomic.fetch_add(_value, std::memory_order_release); }
        Integer FetchAdd_SeqCst (Integer _value) { return m_atomic.fetch_add(_value, std::memory_order_seq_cst); }

        // Exchange
        Integer Exchange_Relaxed(Integer _value) { return m_atomic.exchange(_value, std::memory_order_relaxed); }
        Integer Exchange_Acquire(Integer _value) { return m_atomic.exchange(_value, std::memory_order_acquire); }
        Integer Exchange_Release(Integer _value) { return m_atomic.exchange(_value, std::memory_order_release); }
        Integer Exchange_SeqCst (Integer _value) { return m_atomic.exchange(_value, std::memory_order_seq_cst); }

        // Compare
        bool CompareExchange_Relaxed(Integer& _expected, Integer _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_relaxed); }
        bool CompareExchange_Acquire(Integer& _expected, Integer _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_acquire); }
        bool CompareExchange_Release(Integer& _expected, Integer _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_release, std::memory_order_relaxed); }
        bool CompareExchange_SeqCst (Integer& _expected, Integer _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_seq_cst); }

        // Weak compare (for loops)
        bool CompareExchangeWeak_Relaxed(Integer& _expected, Integer _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_relaxed); }
        bool CompareExchangeWeak_Acquire(Integer& _expected, Integer _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_acquire); }
        bool CompareExchangeWeak_Release(Integer& _expected, Integer _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_release, std::memory_order_relaxed); }
        bool CompareExchangeWeak_SeqCst (Integer& _expected, Integer _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_seq_cst); }

        /// Copy constructor is available for convenience but should not be used in multi-threaded code.
        Atomic(const Atomic<Integer>& _other) : Atomic(_other.Load_Relaxed()) {}

        /// Copying another atomic is possible for convenience but should not be done in multi-threaded code.
        Atomic<Integer>& operator=(const Atomic<Integer>& _other)
        {
            Store_Relaxed(_other.Load_Relaxed());
            return *this;
        }

        Integer operator=(Integer _value)
        {
            Store_SeqCst(_value);
            return _value;
        }

        operator Integer() const
        {
            return Load_SeqCst();
        }

        // increment / decrement operators
        Integer operator++ (int)            { return FetchAdd_SeqCst((Integer) 1); }
        Integer operator-- (int)            { return FetchAdd_SeqCst((Integer)-1); }
        Integer operator+= (Integer _value) { return FetchAdd_SeqCst(_value) + _value; }
        Integer operator-= (Integer _value) { return FetchAdd_SeqCst(_value) - _value; }

    private:
        std::atomic<Integer> m_atomic;
    };


    /********************************
     *        Atomic pointer        *
     ********************************/

    /// @brief Atomic pointer.
    template <class TypePtr>
    class AtomicPtr
    {
    public:
        explicit AtomicPtr(TypePtr* _value) : m_atomic(_value) {}

        TypePtr* Load_Relaxed() const { return m_atomic.load(std::memory_order_relaxed); }
        TypePtr* Load_Acquire() const { return m_atomic.load(std::memory_order_acquire); }
        TypePtr* Load_SeqCst () const { return m_atomic.load(std::memory_order_seq_cst); }

        void Store_Relaxed(TypePtr* _value) { m_atomic.store(_value, std::memory_order_relaxed); }
        void Store_Release(TypePtr* _value) { m_atomic.store(_value, std::memory_order_release); }
        void Store_SeqCst (TypePtr* _value) { m_atomic.store(_value, std::memory_order_seq_cst); }

        TypePtr* Exchange_Relaxed(TypePtr* _value) { return m_atomic.exchange(_value, std::memory_order_relaxed); }
        TypePtr* Exchange_Acquire(TypePtr* _value) { return m_atomic.exchange(_value, std::memory_order_acquire); }
        TypePtr* Exchange_Release(TypePtr* _value) { return m_atomic.exchange(_value, std::memory_order_release); }
        TypePtr* Exchange_SeqCst (TypePtr* _value) { return m_atomic.exchange(_value, std::memory_order_seq_cst); }

        bool CompareExchange_Relaxed(TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_relaxed); }
        bool CompareExchange_Acquire(TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_acquire); }
        bool CompareExchange_Release(TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_release, std::memory_order_relaxed); }
        bool CompareExchange_SeqCst (TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_strong(_expected, _value, std::memory_order_seq_cst); }

        bool CompareExchangeWeak_Relaxed(TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_relaxed); }
        bool CompareExchangeWeak_Acquire(TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_acquire); }
        bool CompareExchangeWeak_Release(TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_release, std::memory_order_relaxed); }
        bool CompareExchangeWeak_SeqCst (TypePtr*& _expected, TypePtr* _value) { return m_atomic.compare_exchange_weak(_expected, _value, std::memory_order_seq_cst); }

        /// Copy constructor is available for convenience but should not be used in multi-threaded code.
        AtomicPtr(const AtomicPtr& _other) : AtomicPtr(_other.Load_Relaxed()) {}

        /// Copying another atomic is possible for convenience but should not be done in multi-threaded code.
        AtomicPtr& operator=(const AtomicPtr& _other)
        {
            Store_Relaxed(_other.Load_Relaxed());
            return *this;
        }

        TypePtr operator=(TypePtr* _value)
        {
            Store_SeqCst(_value);
            return _value;
        }

        operator TypePtr() const
        {
            return Load_SeqCst();
        }

    private:
        std::atomic<TypePtr*> m_atomic;
    };

}