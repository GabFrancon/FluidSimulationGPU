#pragma once
/// @file Types.h
/// @brief File defining the base mathematical types (vectors, matrices, quaternions, etc).

#include <Core/Base.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/hash.hpp>

AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    // floating point vectors
    using Vec2f = glm::vec2;
    using Vec3f = glm::vec3;
    using Vec4f = glm::vec4;

    // signed integer vectors
    using Vec2i = glm::ivec2;
    using Vec3i = glm::ivec3;
    using Vec4i = glm::ivec4;

    // unsigned integer vectors
    using Vec2u = glm::uvec2;
    using Vec3u = glm::uvec3;
    using Vec4u = glm::uvec4;

    // floating point matrices
    using Mat2 = glm::mat2;
    using Mat3 = glm::mat3;
    using Mat4 = glm::mat4;

    // floating point quaternions
    using Quat = glm::quat;


    //---------- Pointer operator --------------------------------------------------------------------------------------------------------

    template <glm::length_t Length, typename Type, glm::qualifier Qualifier>
    Type* operator&(glm::vec<Length, Type, Qualifier>& _vector)
    {
        return glm::value_ptr(_vector);
    }

    template <glm::length_t Length, typename Type, glm::qualifier Qualifier>
    Type const* operator&(const glm::vec<Length, Type, Qualifier>& _vector)
    {
        return glm::value_ptr(_vector);
    }

    template <glm::length_t Column, glm::length_t Row, typename Type, glm::qualifier Qualifier>
    Type* operator&(glm::mat<Column, Row, Type, Qualifier>& _matrix)
    {
        return glm::value_ptr(_matrix);
    }

    template <glm::length_t Column, glm::length_t Row, typename Type, glm::qualifier Qualifier>
    Type const* operator&(const glm::mat<Column, Row, Type, Qualifier>& _matrix)
    {
        return glm::value_ptr(_matrix);
    }

    template <typename Type, glm::qualifier Qualifier>
    Type* operator&(glm::qua<Type, Qualifier>& _quaternion)
    {
        return glm::value_ptr(_quaternion);
    }

    template <typename Type, glm::qualifier Qualifier>
    Type const* operator&(const glm::qua<Type, Qualifier>& _quaternion)
    {
        return glm::value_ptr(_quaternion);
    }


    //----------- Log operator -----------------------------------------------------------------------------------------------------------

    template <typename OStream, glm::length_t Length, typename Type, glm::qualifier Qualifier>
    OStream& operator<<(OStream& _os, const glm::vec<Length, Type, Qualifier>& _vector)
    {
        return _os << glm::to_string(_vector);
    }

    template <typename OStream, glm::length_t Column, glm::length_t Row, typename Type, glm::qualifier Qualifier>
    OStream& operator<<(OStream& _os, const glm::mat<Column, Row, Type, Qualifier>& _matrix)
    {
        return _os << glm::to_string(_matrix);
    }

    template <typename OStream, typename Type, glm::qualifier Qualifier>
    OStream& operator<<(OStream& _os, const glm::qua<Type, Qualifier>& _quaternion)
    {
        return _os << glm::to_string(_quaternion);
    }


    // --------- Math constants ----------------------------------------------------------------------------------------------------------

    namespace Math {

        static constexpr float PI     = 3.1415927f;
        static constexpr float HalfPI = PI * 0.5f;
        static constexpr float TwoPI  = PI * 2.f;

        static const Vec3f Origin     = { 0.f, 0.f, 0.f };
        static const Vec3f AxisX      = { 1.f, 0.f, 0.f };
        static const Vec3f AxisY      = { 0.f, 1.f, 0.f };
        static const Vec3f AxisZ      = { 0.f, 0.f, 1.f };

        static const Vec3f NullVector = { 0.f, 0.f, 0.f };
        static const Vec3f UnitVector = { 1.f, 1.f, 1.f };

        static constexpr Mat2 Identity2    = glm::identity<Mat2>();
        static constexpr Mat3 Identity3    = glm::identity<Mat3>();
        static constexpr Mat4 Identity4    = glm::identity<Mat4>();
        static constexpr Quat QuatIdentity = glm::identity<Quat>();

    }

}