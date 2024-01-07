#pragma once
/// @file Span.h
/// @brief

#include <Core/Base.h>
#include <Debug/Assert.h>

namespace Ava {

    template <class T>
    class Span
    {
    public:
        Span() = default;
        Span(const T* _begin, const T* _end) : m_begin(_begin), m_end(_end) {}
        Span(const T* _begin, const u32 _count) : m_begin(_begin), m_end(_begin + _count) {}
        Span(std::initializer_list<T> _list) : m_begin(_list.begin()), m_end(_list.end()) {}

        template <size_t N>
        explicit Span(const T (&_array)[N]) : Span(_array, N) {}

        template <class U>
        explicit Span(const U& _container) : Span(_container.data(), (u32)_container.size()) {}

        const T* data()  const { return m_begin; }
        const T* begin() const { return m_begin; }
        const T* end()   const { return m_end;   }

        size_t size() const { return m_end - m_begin; }
        bool empty() const  { return size() == 0; }

        const T& operator[](const size_t _index) const
        {
            AVA_ASSERT(_index < size(), "Invalid span index.");
            return m_begin[_index];
        }

    private:
        const T* m_begin = nullptr;
        const T* m_end = nullptr;
    };

    template <class T>
    Span<T> MakeSpan(std::initializer_list<T> _list) { return Span<T>(_list.begin(), _list.end()); }

    template <class T>
    Span<T> MakeSpan(const T* _begin, const T* _end) { return Span<T>(_begin, _end); }

    template <class T>
    Span<T> MakeSpan(const T* _begin, u32 _count) { return Span<T>(_begin, _count); }

    template <class T, size_t N>
    Span<T> MakeSpan(const T (&_array)[N]) { return Span<T>(_array, N); }

    template <class U>
    Span<typename U::value_type> MakeSpan(U& _container) { return Span<typename U::value_type>(_container); }

    template <class T>
    auto begin(Span<T> _container) { return _container.begin(); }

    template <class T>
    auto end(Span<T> _container) { return _container.end(); }

}