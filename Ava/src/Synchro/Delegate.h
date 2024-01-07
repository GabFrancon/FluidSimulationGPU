#pragma once
/// @file Delegate.h
/// @brief

#include <Core/Base.h>

namespace Ava
{
    // Delegate class highly inspired by "FastDelegate" using modern C++11 variadic templates.

    namespace detail
    {
        template <class OutputClass, class InputClass>
        OutputClass implicit_cast(InputClass _input)
        {
            return _input;
        }
        template <typename DestType, typename SourceType>
        DestType union_cast(SourceType _sourceValue)
        {
            union {
                SourceType sourceValue;
                DestType   destValue;
            } u{};
            u.sourceValue = _sourceValue;
            return u.destValue;
        }

    #if defined(_MSC_VER)
        class __single_inheritance DelegateGenericClass;
        class DelegateGenericClass {};
    #else
        class DelegateGenericClass;
    #endif

        constexpr int delegate_generic_class_mfptr_size = sizeof(void (DelegateGenericClass::*)());

        template <int N>
        struct MFHacker {
            template <class T, class FP, class MFP>
            static DelegateGenericClass* Convert(T* _this, FP _functionToBind, MFP &_boundFunc)
            {
                static_assert(N < 0, "delegates not supported on this compiler");
                return nullptr;
            }
        };

        template <>
        struct MFHacker<delegate_generic_class_mfptr_size>
        {
            template <class T, class FP, class MFP>
            static DelegateGenericClass* Convert(T* _pthis, FP _functionToBind, MFP &_boundFunc)
            {
                _boundFunc = reinterpret_cast<MFP>(_functionToBind);
                return reinterpret_cast<DelegateGenericClass*>(_pthis);
            }
        };

    #if defined(_MSC_VER)

        template <>
        struct MFHacker<delegate_generic_class_mfptr_size + sizeof(int)>
        {
            template <class T, class FP, class MFP>
            static DelegateGenericClass* Convert(T* _pthis, FP _functionToBind, MFP &_boundFunc)
            {
                union {
                    FP func;
                    struct {
                        MFP funcAddress;
                        int delta;
                    }s;
                } u{};

                static_assert(sizeof(_functionToBind) == sizeof(u.s), "cannot use union_cast");
                u.func = _functionToBind;
                _boundFunc = u.s.funcAddress;
                return reinterpret_cast<DelegateGenericClass*>(reinterpret_cast<char *>(_pthis) + u.s.delta);
            }
        };

        struct MicrosoftVirtualMFP {
            void (DelegateGenericClass::*codeptr)();
            int delta;
            int vtableIndex;
        };

        struct DelegateGenericVirtualClass : virtual DelegateGenericClass
        {
            typedef DelegateGenericVirtualClass * (DelegateGenericVirtualClass::*ProbePtrType)();
            DelegateGenericVirtualClass * this_() { return this; }
        };

        template <>
        struct MFHacker<delegate_generic_class_mfptr_size + 2 * sizeof(int) >
        {

            template <class T, class FP, class MFP>
            static DelegateGenericClass* Convert(T* _pthis, FP _functionToBind, MFP &_boundFunc)
            {
                union {
                    FP func;
                    DelegateGenericClass* (T::*hackedMfp)();
                    MicrosoftVirtualMFP s;
                } u;
                u.func = _functionToBind;
                _boundFunc = reinterpret_cast<MFP>(u.s.codeptr);
                union {
                    DelegateGenericVirtualClass::ProbePtrType virtualFunc;
                    MicrosoftVirtualMFP s;
                } u2;
                static_assert(sizeof(_functionToBind) == sizeof(u.s)
                    && sizeof(_functionToBind) == sizeof(u.hackedMfp)
                    && sizeof(u2.virtualFunc) == sizeof(u2.s), "cannot use union_cast");
                u2.virtualFunc = &DelegateGenericVirtualClass::this_;
                u.s.codeptr = u2.s.codeptr;
                return (_pthis->*u.hackedMfp)();
            }
        };

        template <>
        struct MFHacker<delegate_generic_class_mfptr_size + sizeof(ptrdiff_t) + 2 * sizeof(int)>
        {
            template <class T, class FP, class MFP>
            static DelegateGenericClass* Convert(T* _pthis, FP _functionToBind, MFP &_boundFunc) {
                union {
                    FP func;
                    struct {
                        MFP funcAddress;
                        int delta;
                        int vtorDisp;
                        int vtableIndex;
                    } s;
                } u;
                static_assert(sizeof(FP) == sizeof(u.s), "cannot use union_cast");
                u.func = _functionToBind;
                _boundFunc = u.s.funcAddress;
                int virtual_delta = 0;
                if (u.s.vtableIndex)
                {
                    const int * vtable = *reinterpret_cast<const int *const*>(
                        reinterpret_cast<const char *>(_pthis) + u.s.vtorDisp);

                    virtual_delta = u.s.vtorDisp + *reinterpret_cast<const int *>(
                        reinterpret_cast<const char *>(vtable) + u.s.vtableIndex);
                }
                return reinterpret_cast<DelegateGenericClass*>(
                    reinterpret_cast<char *>(_pthis) + u.s.delta + virtual_delta);
            };
        };

    #endif // defined(_MSC_VER)

    } // namespace detail

    class DelegateMemento
    {
    public:
        typedef void (detail::DelegateGenericClass::*MFP)();

        DelegateMemento() : m_this(nullptr), m_function(nullptr) {}
        void clear() { m_this = nullptr; m_function = nullptr; }
        void* getThis() const { return m_this; }
        MFP getFunction() const { return m_function; }

        bool operator ==(const DelegateMemento& _other) const
        {
            return m_this == _other.m_this && m_function == _other.m_function;
        }

        bool operator !=(const DelegateMemento& _other) const
        {
            return !operator ==(_other);
        }

        bool operator !() const
        {
            return m_this == nullptr && m_function == nullptr;
        }

        bool empty() const
        {
            return m_this == nullptr && m_function == nullptr;
        }

        DelegateMemento & operator =(const DelegateMemento &_right)
        {
            setMementoFrom(_right);
            return *this;
        }

        bool operator <(const DelegateMemento &_right) const
        {
            if (m_this != _right.m_this)
                return m_this < _right.m_this;
            return memcmp(&m_function, &_right.m_function, sizeof(m_function)) < 0;
        }

        bool operator >(const DelegateMemento &_right) const
        {
            return _right.operator <(*this);
        }

        DelegateMemento(const DelegateMemento &_right) :
            m_this(_right.m_this), m_function(_right.m_function) {}

    protected:
        void setMementoFrom(const DelegateMemento &_right)
        {
            m_function = _right.m_function;
            m_this = _right.m_this;
        }

        detail::DelegateGenericClass* m_this;
        MFP m_function;
    };

    namespace detail {

        template <class t_GenericMFP, class t_StaticFP>
        class ClosurePtr : public DelegateMemento
        {
        public:
            template <class T, class t_AnyMFPtr >
            void bindMF(T* _pthis, t_AnyMFPtr _functionToBind)
            {
                m_this = MFHacker< sizeof(_functionToBind) >
                    ::Convert(_pthis, _functionToBind, m_function);
            }

            template <class T, class t_AnyMFPtr>
            void bindMF_const(const T* _pthis, t_AnyMFPtr _functionToBind)
            {
                m_this = MFHacker< sizeof(_functionToBind) >
                    ::Convert(const_cast<T*>(_pthis), _functionToBind, m_function);
            }

            DelegateGenericClass *getThis() const { return m_this; }
            t_GenericMFP getMFPtr() const { return reinterpret_cast<t_GenericMFP>(m_function); }

            template < class t_DerivedClass >
            void copyFrom(t_DerivedClass* _pParent, const DelegateMemento &_right)
            {
                setMementoFrom(_right);
            }

            template <class t_DerivedClass, class t_ParentInvokerSig>
            void bindF(t_DerivedClass* _pParent, t_ParentInvokerSig _staticFuncInvoker, t_StaticFP _functionToBind)
            {
                if (_functionToBind == 0) { // cope with assignment to 0
                    m_function = nullptr;
                }
                else {
                    bindMF(_pParent, _staticFuncInvoker);
                }

                static_assert(sizeof(DelegateGenericClass *) == sizeof(_functionToBind), "size differs between data pointers and function pointers, cannot use delegates");
                m_this = detail::union_cast<DelegateGenericClass *>(_functionToBind);
            }

            t_StaticFP getStaticFP() const
            {
                static_assert(sizeof(t_StaticFP) == sizeof(this), "size differs between data pointers and function pointers, cannot use delegates");
                return detail::union_cast<t_StaticFP>(this);
            }

            bool contains(t_StaticFP _funcptr)
            {
                if (_funcptr == 0)
                {
                    return empty();
                }
                return _funcptr == reinterpret_cast<t_StaticFP>(getStaticFP());
            }
        };


    } // namespace detail

    template <class S>
    class Delegate;

    template <class R, class... Params>
    class Delegate<R(Params...)>
    {
        typedef R(*FuncPtr)(Params...);
        typedef R(detail::DelegateGenericClass::*MemberFuncPtr)(Params...);
        typedef detail::ClosurePtr<MemberFuncPtr, FuncPtr> ClosurePointerType;

    public:
        typedef Delegate<R(Params...)> ThisType;

        Delegate() { clear(); }

        Delegate(const ThisType& _other)
        {
            m_closure.copyFrom(this, _other.m_closure);
        }

        void operator =(const ThisType& _other)
        {
            m_closure.copyFrom(this, _other.m_closure);
        }

        bool operator ==(const ThisType& _other) const
        {
            return m_closure == _other.m_closure;
        }

        bool operator !=(const ThisType& _other) const
        {
            return m_closure != _other.m_closure;
        }

        bool operator <(const ThisType& _other) const
        {
            return m_closure < _other.m_closure;
        }

        bool operator >(const ThisType& _other) const
        {
            return _other.m_closure < m_closure;
        }

        template <class T, class Y >
        Delegate(Y* _pthis, R(T::* _functionToBind)(Params...))
        {
            m_closure.bindMF(detail::implicit_cast<T*>(_pthis), _functionToBind);
        }

        template <class T, class Y >
        void bind(Y* _pthis, R(T::* _functionToBind)(Params...))
        {
            m_closure.bindMF(detail::implicit_cast<T*>(_pthis), _functionToBind);
        }

        template <class T, class Y >
        Delegate(const Y* _pthis, R(T::* _functionToBind)(Params...) const)
        {
            m_closure.bindMF_const(detail::implicit_cast<const T*>(_pthis), _functionToBind);
        }

        template <class T, class Y >
        void bind(const Y* _pthis, R(T::* _functionToBind)(Params...) const)
        {
            m_closure.bindMF_const(detail::implicit_cast<const T *>(_pthis), _functionToBind);
        }

        Delegate(R(*_functionToBind)(Params...))
        {
            bind(_functionToBind);
        }

        void operator =(R(*_functionToBind)(Params...))
        {
            bind(_functionToBind);
        }

        void bind(R(*_functionToBind)(Params...))
        {
            m_closure.bindF(this, &ThisType::callStaticF,
                _functionToBind);
        }

        R operator ()(Params... _params) const
        {
            return (m_closure.getThis()->*(m_closure.getMFPtr()))(_params...);
        }

        // necessary to allow == 0 to work despite the safe_bool idiom
        bool operator ==(FuncPtr _funcptr)
        {
            return m_closure.contains(_funcptr);
        }

        bool operator !=(FuncPtr _funcptr)
        {
            return !m_closure.contains(_funcptr);
        }

        bool operator !() const
        {
            return !m_closure;
        }

        bool empty() const
        {
            return !m_closure;
        }

        void clear() { m_closure.clear(); }
        const DelegateMemento & getMemento() const { return m_closure; }
        void setMemento(const DelegateMemento& _any) { m_closure.copyFrom(this, _any); }

    private:
        R callStaticF(Params... _params) const {
            return (*m_closure.getStaticFP())(_params...);
        }

        ClosurePointerType m_closure;

    };

    template <class T, class Y, class R, class... Params>
    Delegate<R(Params...)> bind(Y* x, R(T::* _func)(Params...))
    {
        return Delegate<R(Params...)>(x, _func);
    }

    template <class T, class Y, class R, class... Params>
    Delegate<R(Params...)> bind(Y* x, R(T::* _func)(Params...) const)
    {
        return Delegate<R(Params...)>(x, _func);
    }

}
