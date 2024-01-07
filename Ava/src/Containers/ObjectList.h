#pragma once
/// @file ObjectList.h
/// @brief

#include <Containers/Span.h>
#include <Debug/Assert.h>

namespace Ava {

    template <typename T>
    class ObjectList
    {
    public:
        ObjectList() = default;
        ObjectList(const ObjectList<T>& _other) { *this = _other; }
        ObjectList(ObjectList<T>&& _other) noexcept { swap(_other); }
        ~ObjectList() { if (m_objects) delete[] m_objects; }

        T* begin() { return m_objects; }
        T* end() { return m_objects + m_objectCount; }
        T* data() { return begin(); }
        T& operator[](const int _index) { return *(begin() + _index); }

        const T* begin() const { return m_objects; }
        const T* end() const { return m_objects + m_objectCount; }
        const T* data() const { return begin(); }
        const T& operator[](const int _index) const { return *(begin() + _index); }

        u32 size() const { return m_objectCount; }

        T& operator=(const T& _other)
        {
            resize(_other.size());
            for (int i = 0; i < size(); ++i)
            {
                data()[i] = _other[i];
            }
            return *this;
        }

        T& operator=(T&& _other) noexcept
        {
            swap(_other);
            return *this;
        }

        void resize(const u32 _size)
        {
            if (m_objectCount == _size)
            {
                return;
            }

            ObjectList<T> old = std::move(*this);

            if (_size > 0)
            {
                m_objects = new T[_size];
            }
            m_objectCount = _size;

            memcpy(data(), old.data(), std::min(size(), old.size()) * sizeof(T));
        }

        void push_back(const Span<T>& _objects)
        {
            const auto oldSize = size();
            resize(oldSize + (u32)_objects.size());

            memcpy(data() + oldSize, _objects.begin(), _objects.size() * sizeof(T));
        }

        void pop_back(const u32 _count)
        {
            AVA_ASSERT(size() >= _count);
            resize(size() - _count);
        }

        void swap(ObjectList<T>& _other) noexcept
        {
            std::swap(m_objects, _other.m_objects);
            std::swap(m_objectCount, _other.m_objectCount);
        }

    private:
        T* m_objects = nullptr;
        u32 m_objectCount = 0;
    };

}
