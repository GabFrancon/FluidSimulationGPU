#pragma once
/// @file Synchro.h
/// @brief file implementing common synchronization primitives.

#include <Synchro/Atomic.h>
#include <Debug/Assert.h>

// Different primitive types :
//    - Mutex          --> Ensures that only one thread at a time can access specific data.
//    - ReadWriteMutex --> Several reader threads or one write thread can access the data at the same time.
//    - Semaphore      --> Gives access to a certain number of threads (in opposite to mutex which only handles one thread).

// Different primitive variations :
//     - Synchronization primitives --> Puts threads to sleep when locked (not performance free).
//     - Light-weight primitives    --> Does some busy waiting before actually putting thread to sleep.
//     - Spinning primitives        --> Just atomics we use to perform spin waiting (i.e looping on them).

namespace Ava {

    class Mutex;
    class LwMutex;
    class SpinMutex;

    class Semaphore;
    class LwSemaphore;
    class SpinSemaphore;

    class ReadWriteMutex;
    class LwReadWriteMutex;
    class SpinReadWriteMutex;

    static constexpr u32 kWaitInfinite = ~0;


    /********************************
     *       Base primitives        *
     ********************************/

    /// @brief Automatic lock for Mutex. Locks when created, unlocks when destroyed.
    template <class MutexType>
    class AutoLock
    {
        friend class ConditionVariable;

    public:
        AutoLock(MutexType& _mutex)
        {
            m_mutex = &_mutex;
            m_mutex->Lock();
        }

        AutoLock(AutoLock<MutexType>&& _lock) noexcept
        {
            m_mutex = _lock.m_mutex;
            _lock.m_mutex = nullptr;
        }

        ~AutoLock()
        {
            if (m_mutex)
            {
                m_mutex->Unlock();
            }
        }

        AutoLock(const AutoLock<MutexType>& _lock) = delete;
        AutoLock<MutexType>& operator=(const AutoLock<MutexType>& _lock) = delete;

    private:
        MutexType* m_mutex;
    };

    /// @brief Blocks one or several threads until a notification is received, a timeout expires or a spurious wakeup occurs.
    class ConditionVariable
    {
    public:
        ConditionVariable();
        ~ConditionVariable();

        /// @brief Waits for _timeoutMS milliseconds or until NotifyOne/NotifyAll is called.
        bool Wait(const AutoLock<Mutex>& _lock, u32 _timeoutMS = kWaitInfinite);
        /// @brief Wakes up one thread.
        void NotifyOne();
        /// @brief Wakes all waiting threads.
        void NotifyAll();

    private:
        std::condition_variable m_condition;
    };

    /// @brief When locked, the mutex ensures that only one thread access some data at a time.
    class Mutex
    {
        friend class ConditionVariable;

    public:
        Mutex();
        ~Mutex();

        /// @brief Locks the mutex, blocks if the mutex is not available.
        void Lock();
        /// @brief Tries to lock the mutex, returns if the mutex is not available.
        bool TryLock();
        /// @brief Unlocks the mutex.
        void Unlock();
        /// @brief Builds an auto lock with this mutex.
        [[nodiscard]] AutoLock<Mutex> MakeAutoLock();

    private:
        std::mutex m_mutex;
    };

    /// @brief Gives access to a certain number of threads.
    class Semaphore
    {
    public:
        Semaphore(int _initialCount = 0);
        ~Semaphore();

        /// @brief Decrements the internal counter or blocks until it can.
        void Wait();
        /// @brief Increments the internal counter and unblocks acquirers.
        void Signal(s32 _count = 1);
        /// @brief Returns the number of thread currently waiting for the semaphore to signal.
        int GetWaitingCount() const;

    private:
        s32 m_count;
        s32 m_waitingCount;
        Mutex m_mutex;
        ConditionVariable m_condition;
    };

    /// @brief Blocks one or several threads until an event is raised.
    class ThreadSignal
    {
    public:
        ThreadSignal(bool _manualReset = false);
        ~ThreadSignal();

        /// @brief The current thread will wait until the signal is raised.
        bool Wait(u32 _timeoutMS = kWaitInfinite);
        /// @brief Raises the event.
        void Raise();
        /// @brief Resets the signal.
        void Clear();

    private:
        const bool m_manualReset;
        bool m_signaled;
        ConditionVariable m_condition;
        Mutex m_mutex;
    };


    /********************************
     *     Semaphore variations     *
     ********************************/

    /// @brief Light-weight version of the Semaphore.
    class LwSemaphore
    {
    public:
        LwSemaphore(int _initialCount = 0, int _spinCount = 10000);
        ~LwSemaphore();

        /// @brief Tries decrementing the internal counter without blocking.
        bool TryWait();
        /// @brief Decrements the internal counter or blocks until it can.
        void Wait();
        /// @brief Increments the internal counter and unblocks acquirers.
        void Signal(s32 _count = 1);
        /// @brief Returns the number of thread currently waiting for the semaphore to signal.
        int GetWaitingCount() const;

        void SetSpinCount(const s32 _spinCount) { m_spinCount = _spinCount; }
        s32 GetSpinCount() const { return m_spinCount; }

    private:
        void _WaitWithPartialSpinning();

        Atomic<s32> m_count;
        Semaphore m_semaphore;
        s32 m_spinCount;
    };

    /// @brief Spinning version of the Semaphore.
    class SpinSemaphore
    {
    public:
        SpinSemaphore(int _initialCount = 0);
        ~SpinSemaphore();

        /// @brief Tries decrementing the internal counter without blocking.
        bool TryWait();
        /// @brief Decrements the internal counter or blocks until it can.
        void Wait();
        /// @brief Increments the internal counter and unblocks acquirers.
        void Signal(s32 _count = 1);

    private:
        void _WaitWithSpinning();

        Atomic<s32> m_count;
    };


    /********************************
     *       Mutex variations       *
     ********************************/

    /// @brief Light-weight version of the Mutex (= benaphore).
    class LwMutex
    {
    public:
        LwMutex(int _spinCount = 10000);
        ~LwMutex();

        /// @brief Locks the mutex, blocks if the mutex is not available.
        void Lock();
        /// @brief Tries to lock the mutex, returns false if the mutex is not available.
        bool TryLock();
        /// @brief Unlocks the mutex.
        void Unlock();
        /// @brief Builds an auto lock with this mutex.
        [[nodiscard]] AutoLock<LwMutex> MakeAutoLock();

    private:
        Atomic<s32> m_contentionCount;
        LwSemaphore m_semaphore;
    };

    /// @brief Spinning version of the Mutex.
    class SpinMutex
    {
    public:
        SpinMutex();
        ~SpinMutex();

        /// @brief Locks the mutex, blocks if the mutex is not available.
        void Lock();
        /// @brief Tries to lock the mutex, returns if the mutex is not available.
        bool TryLock();
        /// @brief Unlocks the mutex.
        void Unlock();
        /// @brief Builds an auto lock with this mutex.
        [[nodiscard]] AutoLock<SpinMutex> MakeAutoLock();

    private:
        Atomic<s32> m_spinLock;
    };


    /********************************
     *      Read/write mutex        *
     ********************************/

    /// @brief Automatic lock for ReadWriteMutex. Locks readers when created, unlocks when destroyed.
    template <class MutexType>
    class ReaderAutoLock
    {
    public:
        ReaderAutoLock(MutexType& _mutex)
        {
            m_mutex = &_mutex;
            m_mutex->LockReader();
        }

        ReaderAutoLock(ReaderAutoLock<MutexType>&& _lock) noexcept
        {
            m_mutex = _lock.m_mutex;
            _lock.m_mutex = NULL;
        }

        ~ReaderAutoLock()
        {
            if (m_mutex)
                m_mutex->UnlockReader();
        }

        ReaderAutoLock(const ReaderAutoLock<MutexType>& _lock) = delete;
        ReaderAutoLock<MutexType>& operator=(const ReaderAutoLock<MutexType>& _lock) = delete;

    private:
        MutexType* m_mutex;
    };

    /// @brief Automatic lock for ReadWriteMutex. Locks writer when created, unlocks when destroyed.
    template <class MutexType>
    class WriterAutoLock
    {
    public:
        WriterAutoLock(MutexType& _mutex)
        {
            m_mutex = &_mutex;
            m_mutex->LockWriter();
        }

        WriterAutoLock(WriterAutoLock<MutexType>&& _lock) noexcept
        {
            m_mutex = _lock.m_mutex;
            _lock.m_mutex = NULL;
        }

        ~WriterAutoLock()
        {
            if (m_mutex)
                m_mutex->UnlockWriter();
        }

        WriterAutoLock(const WriterAutoLock<MutexType>& _lock) = delete;
        WriterAutoLock<MutexType>& operator=(const WriterAutoLock<MutexType>& _lock) = delete;

    private:
        MutexType* m_mutex;
    };

    /// @brief Base implementation for read/write synchronization objects.
    template <class SemaphoreType>
    class RWMutexBase
    {
    public:
        RWMutexBase()
            : m_status(0)
        {
        }

        void LockReader()
        {
            Status oldStatus = m_status.Load_Relaxed();
            Status newStatus;
            do
            {
                newStatus = oldStatus;
                if (oldStatus.writers > 0)
                {
                    ++newStatus.waitToRead;
                }
                else
                {
                    ++newStatus.readers;
                }
                // CAS until successful. On failure, oldStatus will be updated with the latest value.
            }
            while (!m_status.CompareExchangeWeak_Acquire(oldStatus, newStatus));

            if (oldStatus.writers > 0)
            {
                m_readSemaphore.Wait();
            }
        }

        bool TryLockReader()
        {
            Status oldStatus = m_status.Load_Relaxed();
            Status newStatus;
            do
            {
                newStatus = oldStatus;
                if (oldStatus.writers > 0)
                {
                    return false;
                }
                else
                {
                    ++newStatus.readers;
                }
                // CAS until successful. On failure, oldStatus will be updated with the latest value.
            } while (!m_status.CompareExchangeWeak_Acquire(oldStatus, newStatus));

            return true;
        }

        void UnlockReader()
        {
            Status oldStatus = m_status.FetchAdd_Release(-(s32)Status::ReadersOne());
            AVA_ASSERT(oldStatus.readers > 0);
            if (oldStatus.readers == 1 && oldStatus.writers > 0)
            {
                m_writeSemaphore.Signal();
            }
        }

        void LockWriter()
        {
            Status oldStatus = m_status.FetchAdd_Acquire(Status::WritersOne());
            AVA_ASSERT(oldStatus.writers + 1 <= (u32)Status::WritersMax());
            if (oldStatus.readers > 0 || oldStatus.writers > 0)
            {
                m_writeSemaphore.Wait();
            }
        }

        bool TryLockWriter()
        {
            Status oldStatus = m_status.Load_Relaxed();
            if (oldStatus.readers > 0 || oldStatus.writers > 0)
                return false;

            Status newStatus = oldStatus;
            newStatus.writers = 1;
            return m_status.CompareExchange_Acquire(oldStatus, newStatus);
        }

        void UnlockWriter()
        {
            Status oldStatus = m_status.Load_Relaxed();
            Status newStatus;
            u32 waitToRead = 0;
            do
            {
                AVA_ASSERT(oldStatus.readers == 0);
                newStatus = oldStatus;
                --newStatus.writers;

                waitToRead = oldStatus.waitToRead;
                if (waitToRead > 0)
                {
                    newStatus.waitToRead = 0;
                    newStatus.readers = waitToRead;
                }
                // CAS until successful. On failure, oldStatus will be updated with the latest value.
            }
            while (!m_status.CompareExchangeWeak_Release(oldStatus, newStatus));

            if (waitToRead > 0)
            {
                m_readSemaphore.Signal(waitToRead);
            }
            else if (oldStatus.writers > 1)
            {
                m_writeSemaphore.Signal();
            }
        }

    private:
        struct Status
        {
            Status(const u32 _status = 0) { allBits = _status; }
            operator u32&() { return allBits; }

            static Status ReadersOne()
            {
                Status status(0);
                status.readers = 1;
                return status;
            }

            static Status ReadersMax()
            {
                Status status(0);
                status.readers = (1 << 10) - 1;
                return status;
            }

            static Status WritersOne()
            {
                Status status(0);
                status.writers = 1;
                return status;
            }

            static Status WritersMax()
            {
                Status status(0);
                status.writers = (1 << 10) - 1;
                return status;
            }

            union
            {
                struct
                {
                    u32 readers : 10;
                    u32 waitToRead : 10;
                    u32 writers : 10;
                };

                u32 allBits;
            };
        };

        Atomic<u32> m_status;
        SemaphoreType m_readSemaphore;
        SemaphoreType m_writeSemaphore;
    };

    /// @brief Simple read/write mutex class.
    class ReadWriteMutex : public RWMutexBase<Semaphore>
    {
    public:
        [[nodiscard]] ReaderAutoLock<ReadWriteMutex> AutoLockReader() { return ReaderAutoLock(*this); }
        [[nodiscard]] WriterAutoLock<ReadWriteMutex> AutoLockWriter() { return WriterAutoLock(*this); }
    };

    /// @brief Light-weight version of the ReadWriteMutex.
    class LwReadWriteMutex : public RWMutexBase<LwSemaphore>
    {
    public:
        [[nodiscard]] ReaderAutoLock<LwReadWriteMutex> AutoLockReader() { return ReaderAutoLock(*this); }
        [[nodiscard]] WriterAutoLock<LwReadWriteMutex> AutoLockWriter() { return WriterAutoLock(*this); }
    };

    /// @brief Spinning version of the ReadWriteMutex.
    class SpinReadWriteMutex : public RWMutexBase<SpinSemaphore>
    {
    public:
        [[nodiscard]] ReaderAutoLock<SpinReadWriteMutex> AutoLockReader() { return ReaderAutoLock(*this); }
        [[nodiscard]] WriterAutoLock<SpinReadWriteMutex> AutoLockWriter() { return WriterAutoLock(*this); }
    };
}