#pragma once
/// @file Signal.h
/// @brief

#include <Synchro/Delegate.h>
#include <Debug/Assert.h>

//----- Signal class -----------------------------------------------------------

namespace Ava
{
    template <class Signature>
    class Signal;

    template <class R, class ... Args>
    class Signal<R(Args...)>
    {
    public:
        typedef Delegate<R(Args...)> Delegate;

    private:
        using Slot = Delegate;
        using SlotList = std::vector<Slot>;
        using SlotIt = typename SlotList::const_iterator;

        static SlotIt FindDelegateItem(const SlotList& _slotList, const Slot& _slot)
        {
            return std::find(_slotList.begin(), _slotList.end(), _slot);
        }

        static void InsertDelegateItem(SlotList& _slotList, const Slot& _slot)
        {
            _slotList.push_back(_slot);
        }

        static void RemoveDelegateItem(SlotList& _slotList, const SlotIt& _slotIt)
        {
            _slotList.erase(_slotIt);
        }

        static void CallDelegateItem(const Slot& _slot, Args... _args)
        {
            _slot(_args...);
        }

    public:
        Signal() : m_slots(nullptr) {}
        ~Signal() { Clear(); }

        Signal(const Signal& _other)
        {
            if (_other.m_slots)
            {
                m_slots = new SlotList();
                *m_slots = *_other.m_slots;
            }
        }

        Signal(Signal&& _other) noexcept
        {
            Clear();
            m_slots = _other.m_slots;
            _other.m_slots = nullptr;
        }

        template <class T, class Mptr>
        void Connect(T* _obj, Mptr _func)
        {
            Connect(Delegate(_obj, _func));
        }

        void Connect(R(*_func)(Args...))
        {
            Connect(Delegate(_func));
        }

        void Connect(Slot const& _delegate)
        {
            if (m_slots == nullptr)
            {
                m_slots = new SlotList();
            }
            AVA_ASSERT(FindDelegateItem(*m_slots, _delegate) == m_slots->end());
            InsertDelegateItem(*m_slots, _delegate);
        }

        template <class T, class Mptr>
        void Disconnect(T* _obj, Mptr _func)
        {
            Disconnect(Delegate(_obj, _func));
        }

        void Disconnect(Slot const& _delegate)
        {
            AVA_ASSERT(m_slots);
            auto found = FindDelegateItem(*m_slots, _delegate);
            AVA_ASSERT(!(found == m_slots->end()));
            RemoveDelegateItem(*m_slots, found);
            if (m_slots->empty())
            {
                delete m_slots;
                m_slots = nullptr;
            }
        }

        void Emit(Args... _args)
        {
            if (m_slots)
            {
                SlotList slotsCopy = *m_slots;

                for (auto& slot : slotsCopy)
                {
                    CallDelegateItem(slot, _args...);
                }
            }
        }

        bool HasReceivers() const
        {
            return m_slots != nullptr;
        }

        Signal& operator=(const Signal& _other)
        {
            Clear();
            if (_other.m_slots)
            {
                m_slots = new SlotList();
                *m_slots = *_other.m_slots;
            }
            return *this;
        }

        Signal& operator=(Signal&& _other) noexcept
        {
            Clear();
            m_slots = _other.m_slots;
            _other.m_slots = nullptr;
            return *this;
        }

        void operator()(Args... _args)
        {
            Emit(_args...);
        }

    private:
        void Clear()
        {
            if (m_slots)
            {
                delete m_slots;
                m_slots = nullptr;
            }
        }
        SlotList* m_slots = nullptr;
    };
}


//----- Signal macros ----------------------------------------------------------

#define CONNECT_SIGNAL(emitterPtr, signal, receiverPtr, func) \
    ((emitterPtr)->signal).Connect(receiverPtr, &std::remove_pointer_t<std::remove_reference_t<decltype(receiverPtr)>>::func)

#define DISCONNECT_SIGNAL(emitterPtr, signal, receiverPtr, func) \
    ((emitterPtr)->signal).Disconnect(receiverPtr, &std::remove_pointer_t<std::remove_reference_t<decltype(receiverPtr)>>::func)

#define CONNECT_SIGNAL_STATIC(emitterPtr, signal, func) ((emitterPtr)->signal).Connect(func)

#define DISCONNECT_SIGNAL_STATIC(emitterPtr, signal, func) ((emitterPtr)->signal).Disconnect(func)
