#pragma once
/// @file Geometry.h
/// @brief File defining the main geometric primitives.

#include <Math/Types.h>

namespace Ava {

    struct Rect;
    struct BoundingBox;
    struct BoundingSphere;
    struct Plane;
    struct Frustum;

    // ----- 2D Rect ------------------------

    //   +------------------+
    //   |                  |
    //   |                  |
    //   |                  |
    //   |                  |
    //   +------------------+

    /// @brief Represents a rectangle with width, height, and position offsets.
    /// @details Provides methods for rectangle manipulation and point containment checks.
    struct Rect
    {
        u32 x = 0;
        u32 y = 0;
        u32 width = 0;
        u32 height = 0;

        Rect() = default;
        Rect(u32 _x, u32 _y, u32 _width, u32 _height);
        Rect(u32 _width, u32 _height);

        bool operator==(const Rect& _other) const;
        bool operator!=(const Rect& _other) const;

        float GetAspectRatio() const;
        Vec2f GetRelativePos(const Vec2f& _absolutePos) const;
        bool Contains(const Vec2f& _absolutePos) const;
    };

    // ----- Bounding box -------------------

    //      +------------+
    //     /            /|
    //    /            / |
    //   +------------+  |
    //   |            |  |
    //   |            |  +
    //   |            | /
    //   |            |/
    //   +------------+

    struct BoundingBox
    {
        Vec3f min = {-0.5f,-0.5f,-0.5f };
        Vec3f max = { 0.5f, 0.5f, 0.5f };

        BoundingBox() = default;
        BoundingBox(const Vec3f& _min, const Vec3f& _max);
        BoundingBox(const BoundingSphere& _sphere);
        BoundingBox(const Frustum& _frustum);
        BoundingBox(const BoundingBox& _box, const Mat4& _transform);

        bool operator==(const BoundingBox& _other) const;
        bool operator!=(const BoundingBox& _other) const;

        Vec3f GetMin() const;
        Vec3f GetMax() const;
        Vec3f GetOrigin() const;
        Vec3f GetExtent() const;

        void Grow(const Vec3f& _extent);
        void Grow(const BoundingBox& _other);
        void Transform(const Mat4& _transform);
        bool Contains(const Vec3f& _point) const;

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
    };


    // ----- Bounding sphere ----------------

    //        .-------.
    //     .'           '.
    //    /               \
    //   |                 |
    //  |\                 /|
    //  | '__     +     __' |
    //   |    ---------    |
    //    \               /
    //     '.           .'
    //        '-------'

    struct BoundingSphere
    {
        Vec3f origin = { 0.f, 0.f, 0.f };
        float radius = 0.5f;

        BoundingSphere() = default;
        BoundingSphere(const Vec3f& _origin, float _radius);
        BoundingSphere(const BoundingBox& _box);
        BoundingSphere(const Frustum& _frustum);
        BoundingSphere(const BoundingSphere& _sphere, const Mat4& _transform);

        bool operator==(const BoundingSphere& _other) const;
        bool operator!=(const BoundingSphere& _other) const;

        Vec3f GetOrigin() const;
        float GetRadius() const;

        void Grow(float _radius);
        void Transform(const Mat4& _transform);
        bool Contains(const Vec3f& _point) const;

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
    };


    // ----- Plane --------------------------

    //       +---------------+
    //      /               /
    //     /               /
    //    /               /
    //   /               /
    //  +---------------+

    struct Plane
    {
        Vec3f normal = Math::AxisY;
        float offset = 0.f;

        Plane() = default;
        Plane(const Vec3f& _normal, float _offset);
        Plane(const Vec3f& _normal, const Vec3f& _origin);
        Plane(float _a, float _b, float _c, float _d);
        Plane(const Vec3f& _p1, const Vec3f& _p2, const Vec3f& _p3);

        bool operator==(const Plane& _other) const;
        bool operator!=(const Plane& _other) const;

        float GetOffset() const;
        Vec3f GetNormal() const;
        Vec3f GetOrigin() const;

        /// @brief Returns a negative distance when point is behind the plane.
        float Distance(const Vec3f& _point) const;
        /// @brief Returns the point on the plane closest to the given point.
        Vec3f Closest(const Vec3f& _point) const;
        /// @brief Applies the given affine transformation to the plane.
        void Transform(const Mat4& _transform);

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
    };


    // ----- Frustum ------------------------

    //   4------------5
    //   |\          /|
    //   7-\--------/-6
    //    \ 0------1 /
    //     \|      |/
    //      3------2

    struct Frustum
    {
        enum FrustumPlane
        {
            NearPlane,
            FarPlane,
            TopPlane,
            RightPlane,
            BottomPlane,
            LeftPlane,

            PlaneCount
        };

        Plane planes[PlaneCount];
        Vec3f vertices[8];

        Frustum() = default;
        Frustum(float _top, float _down, float _right, float _left, float _near, float _far);
        Frustum(float _fovRadians, float _aspectRatio, float _near, float _far);
        Frustum(const Frustum& _frustum, float _nearOffset, float _farOffset);
        Frustum(const Frustum& _frustum, const Mat4& _transform);

        bool Contains(const Vec3f& _point) const;
        bool Contains(const BoundingBox& _box) const;
        bool Contains(const BoundingSphere& _sphere) const;

        void SetNearFar(float _near, float _far);
        void Transform(const Mat4& _transform);
        void InitPlanesFromVertices();

        SerializeError Serialize(Serializer& _serializer, const char* _tag);
    };

}
