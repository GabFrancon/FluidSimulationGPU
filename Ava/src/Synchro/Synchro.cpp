#include <avapch.h>
#include "Synchro.h"

#include <Math/Math.h>

namespace Ava {

    // ---- Condition variable ----------------------------

    ConditionVariable::ConditionVariable()
    {
    }

    ConditionVariable::~ConditionVariable()
    {
    }

    bool ConditionVariable::Wait(const AutoLock<Mutex>& _lock, const u32 _timeoutMS/*= kWaitInfinite*/)
    {
        // Temporarily transfers ownership of the mutex to a std::unique_lock
        std::unique_lock stdLock(_lock.m_mutex->m_mutex, std::adopt_lock);

        if (_timeoutMS == kWaitInfinite)
        {
            m_condition.wait(stdLock);
            stdLock.release(); // don't unlock, give back the ownership to AutoLock
            return true;
        }

        const std::cv_status status = m_condition.wait_for(stdLock, std::chrono::milliseconds(_timeoutMS));
        stdLock.release(); // don't unlock, give back the ownership to AutoLock
        return status == std::cv_status::no_timeout;
    }

    void ConditionVariable::NotifyOne()
    {
        m_condition.notify_one();
    }

    void ConditionVariable::NotifyAll()
    {
        m_condition.notify_all();
    }


    // ---- Mutex -----------------------------------------

    Mutex::Mutex()
    {
    }

    Mutex::~Mutex()
    {
        if (!TryLock()) {
            AVA_ASSERT(false, "Deleting a mutex while it's still locked is undefined behavior.");
        }
        else {
            Unlock();
        }
    }

    void Mutex::Lock()
    {
        m_mutex.lock();
    }

    bool Mutex::TryLock()
    {
        return m_mutex.try_lock();
    }

    void Mutex::Unlock()
    {
        m_mutex.unlock();
    }

    AutoLock<Mutex> Mutex::MakeAutoLock()
    {
        return AutoLock(*this);
    }


    // ---- Semaphore -------------------------------------

    Semaphore::Semaphore(const int _initialCount/*= 0*/)
        : m_count(_initialCount), m_waitingCount(0)
    {
        AVA_ASSERT(m_count >= 0);
    }

    Semaphore::~Semaphore()
    {
    }

    void Semaphore::Wait()
    {
        bool wakeUpAnother = false;

        // mutex is locked inside this scope
        {
            const auto lock = m_mutex.MakeAutoLock();

            while (m_count == 0)
            {
                m_waitingCount++;
                m_condition.Wait(lock, 1);
                m_waitingCount--;
            }

            m_count--;
            if (m_count > 0 && m_waitingCount > 0)
            {
                wakeUpAnother = true;
            }
        }

        if (wakeUpAnother)
        {
            m_condition.NotifyOne();
        }
    }

    void Semaphore::Signal(const s32 _count/*= 1*/)
    {
        bool wakeUp = false;

        // mutex is locked inside this scope
        {
            const auto lock = m_mutex.MakeAutoLock();

            m_count += _count;
            if (m_waitingCount > 0)
            {
                wakeUp = true;
            }
        }

        if (wakeUp)
        {
            m_condition.NotifyOne();
        }
    }

    int Semaphore::GetWaitingCount() const
    {
        return m_waitingCount;
    }


    // ---- Signal ----------------------------------------

    ThreadSignal::ThreadSignal(const bool _manualReset)
        : m_manualReset(_manualReset), m_signaled(false)
    {
    }

    ThreadSignal::~ThreadSignal()
    {
    }

    bool ThreadSignal::Wait(const u32 _timeoutMS/*= kWaitInfinite*/)
    {
        const auto lock = m_mutex.MakeAutoLock();
        bool timeout = false;

        // Start by checking if the signal is already raised (we don't want to wait in this case).
        // The loop is necessary because condition variable can wake up spuriously.
        while (!m_signaled)
        {
            if (_timeoutMS == kWaitInfinite)
            {
                m_condition.Wait(lock);
            }
            else
            {
                timeout = !m_condition.Wait(lock, _timeoutMS);

                // If it's a timeout, it's not a spurious wakeup, don't loop.
                if (timeout)
                {
                    break;
                }
            }
        }

        // Don't clear if woken up by a timeout.
        if (!m_manualReset && !timeout)
        {
            m_signaled = false;
        }

        return !timeout;
    }

    void ThreadSignal::Raise()
    {
        const auto lock = m_mutex.MakeAutoLock();

        // Already signaled, nothing to do
        if (m_signaled == true)
        {
            return;
        }
        m_signaled = true;

        if (m_manualReset)
        {
            m_condition.NotifyAll();
        }
        else
        {
            m_condition.NotifyOne();
        }
    }

    void ThreadSignal::Clear()
    {
        const auto lock = m_mutex.MakeAutoLock();
        m_signaled = false;
    }


    // ---- Light-weight semaphore ------------------------

    LwSemaphore::LwSemaphore(const int _initialCount/*= 0*/, const int _spinCount/*= 10000*/)
        : m_count(_initialCount), m_spinCount(_spinCount)
    {
        AVA_ASSERT(m_count >= 0);
    }

    LwSemaphore::~LwSemaphore()
    {
    }

    bool LwSemaphore::TryWait()
    {
        int oldCount = m_count.Load_Relaxed();
        return oldCount > 0 && m_count.CompareExchange_Acquire(oldCount, oldCount  - 1);
    }

    void LwSemaphore::Wait()
    {
        if (!TryWait())
        {
            _WaitWithPartialSpinning();
        }
    }

    void LwSemaphore::Signal(const s32 _count/*= 1*/)
    {
        const int oldCount = m_count.FetchAdd_Release(_count);
        const int toRelease = Math::min(-oldCount, _count);

        if (toRelease > 0)
        {
            m_semaphore.Signal(toRelease);
        }
    }

    int LwSemaphore::GetWaitingCount() const
    {
        const int count = m_count.Load_Relaxed();
        if (count < 0)
        {
            return -count;
        }
        return 0;
    }

    void LwSemaphore::_WaitWithPartialSpinning()
    {
        int spin = m_spinCount;
        int oldCount;

        while (spin--)
        {
            oldCount = m_count.Load_Relaxed();
            if (oldCount > 0 && m_count.CompareExchange_Acquire(oldCount, oldCount - 1))
            {
                return;
            }

            AVA_YIELD_PROCESSOR(); // Improves the speed at which the code detects the release of the lock.
            AVA_BARRIER(); // Prevents the compiler from collapsing the loop.
        }

        oldCount = m_count.FetchAdd_Acquire(-1);

        if (oldCount <= 0)
        {
            m_semaphore.Wait();
        }
    }


    // ---- Spin semaphore -------------------------------

    SpinSemaphore::SpinSemaphore(const int _initialCount/*= 0*/)
        : m_count(_initialCount)
    {
        AVA_ASSERT(m_count >= 0);
    }

    SpinSemaphore::~SpinSemaphore()
    {
    }

    bool SpinSemaphore::TryWait()
    {
        int oldCount = m_count.Load_Relaxed();
        return oldCount > 0 && m_count.CompareExchange_Acquire(oldCount, oldCount - 1);
    }

    void SpinSemaphore::Wait()
    {
        if (!TryWait())
        {
            _WaitWithSpinning();
        }
    }

    void SpinSemaphore::Signal(const s32 _count/*= 1*/)
    {
        m_count.FetchAdd_Release(_count);
    }

    void SpinSemaphore::_WaitWithSpinning()
    {
        int oldCount = 1;

        while (true)
        {
            oldCount = m_count.Load_Relaxed();
            if (oldCount > 0 && m_count.CompareExchange_Acquire(oldCount, oldCount - 1))
            {
                return;
            }

            AVA_YIELD_PROCESSOR(); // Improves the speed at which the code detects the release of the lock.
            AVA_BARRIER(); // Prevents the compiler from collapsing the loop.
        }
    }


    // ---- Light-weight mutex ---------------------------

    LwMutex::LwMutex(const int _spinCount/*= 10000*/)
        : m_contentionCount(0), m_semaphore(0, _spinCount)
    {
    }

    LwMutex::~LwMutex()
    {
    }

    void LwMutex::Lock()
    {
        if (m_contentionCount.FetchAdd_Acquire(1) > 0)
        {
            m_semaphore.Wait();
        }
    }

    bool LwMutex::TryLock()
    {
        if (m_contentionCount.Load_Relaxed() != 0)
        {
            return false;
        }

        int expected = 0;
        return m_contentionCount.CompareExchange_Acquire(expected, 1);
    }

    void LwMutex::Unlock()
    {
        const int oldCount = m_contentionCount.FetchAdd_Release(-1);
        AVA_ASSERT(oldCount > 0);

        if (oldCount > 1)
        {
            m_semaphore.Signal();
        }
    }

    AutoLock<LwMutex> LwMutex::MakeAutoLock()
    {
        return AutoLock(*this);
    }


    // ---- Spin mutex -----------------------------------

    SpinMutex::SpinMutex()
        : m_spinLock(0)
    {
    }

    SpinMutex::~SpinMutex()
    {
    }

    void SpinMutex::Lock()
    {
        while (true)
        {
            if (m_spinLock.Load_Acquire() == 0)
            {
                s32 expected = 0;
                constexpr s32 store = 1;

                if (m_spinLock.CompareExchange_Acquire(expected, store))
                {
                    return;
                }
            }

            AVA_YIELD_PROCESSOR();
        }
    }

    bool SpinMutex::TryLock()
    {
        s32 expected = 0;
        constexpr s32 store = 1;

        if (m_spinLock.CompareExchange_Acquire(expected, store))
        {
            return true;
        }

        return false;
    }

    void SpinMutex::Unlock()
    {
        m_spinLock.Store_Release(0);
    }

    AutoLock<SpinMutex> SpinMutex::MakeAutoLock()
    {
        return AutoLock(*this);
    }
    
}
