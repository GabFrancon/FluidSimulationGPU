#include <avapch.h>
#include "Geometry.h"

#include <Files/Serializer.h>
#include <Math/Math.h>

namespace Ava {

    // --------- Rect ----------------------------------------------

    Rect::Rect(const u32 _x, const u32 _y, const u32 _width, const u32 _height)
        : x(_x), y(_y), width(_width), height(_height)
    {
    }

    Rect::Rect(const u32 _width, const u32 _height)
        : width(_width), height(_height)
    {
    }

    bool Rect::operator==(const Rect& _other) const
    {
        return x == _other.x && y == _other.y && width == _other.width && height == _other.height;
    }

    bool Rect::operator!=(const Rect& _other) const
    {
        return !(*this == _other);
    }

    float Rect::GetAspectRatio() const
    {
        return static_cast<float>(width) / static_cast<float>(height);
    }

    Vec2f Rect::GetRelativePos(const Vec2f& _absolutePos) const
    {
        Vec2f uv{};
        uv.x = (_absolutePos.x - static_cast<float>(x)) / static_cast<float>(width);
        uv.y = (_absolutePos.y - static_cast<float>(y)) / static_cast<float>(height);
        return uv;
    }

    bool Rect::Contains(const Vec2f& _absolutePos) const
    {
        return _absolutePos.x >= x && _absolutePos.x <= width && _absolutePos.y >= y && _absolutePos.y <= height;
    }


    // --- Bounding box ----------------------------------------------------------------

    BoundingBox::BoundingBox(const Vec3f& _min, const Vec3f& _max)
    {
        min = _min;
        max = _max;
    }

    BoundingBox::BoundingBox(const BoundingSphere& _sphere)
    {
        min = _sphere.GetOrigin() - _sphere.GetRadius();
        max = _sphere.GetOrigin() + _sphere.GetRadius();
    }

    BoundingBox::BoundingBox(const Frustum& _frustum)
    {
        min = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
        max =-Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);

        for (int i = 0; i < 8; ++i) {
            min = Math::minPerElement(min, _frustum.vertices[i]);
            max = Math::maxPerElement(max, _frustum.vertices[i]);
        }
    }

    BoundingBox::BoundingBox(const BoundingBox& _box, const Mat4& _transform)
    {
        min = _box.min;
        max = _box.max;
        Transform(_transform);
    }

    bool BoundingBox::operator==(const BoundingBox& _other) const
    {
        return min == _other.min && max == _other.max;
    }

    bool BoundingBox::operator!=(const BoundingBox& _other) const
    {
        return !(*this == _other);
    }

    Vec3f BoundingBox::GetMin() const
    {
        return min;
    }

    Vec3f BoundingBox::GetMax() const
    {
        return max;
    }

    Vec3f BoundingBox::GetOrigin() const
    {
        return 0.5f * (min + max);
    }

    Vec3f BoundingBox::GetExtent() const
    {
        return max - min;
    }

    void BoundingBox::Grow(const Vec3f& _extent)
    {
        min = Math::minPerElement(min, _extent);
        max = Math::maxPerElement(max, _extent);
    }

    void BoundingBox::Grow(const BoundingBox& _other)
    {
        min = Math::minPerElement(min, _other.min);
        max = Math::maxPerElement(max, _other.max);
    }

    void BoundingBox::Transform(const Mat4& _transform)
    {
        const Vec3f mx = Math::column(_transform, 0);
        const Vec3f my = Math::column(_transform, 1);
        const Vec3f mz = Math::column(_transform, 2);
        const Vec3f mw = Math::column(_transform, 3);

        const Vec3f xa = mx * min.x;
        const Vec3f xb = mx * max.x;

        const Vec3f ya = my * min.y;
        const Vec3f yb = my * max.y;

        const Vec3f za = mz * min.z;
        const Vec3f zb = mz * max.z;

        min = Math::minPerElement(xa, xb) + Math::minPerElement(ya, yb) + Math::minPerElement(za, zb) + mw;
        max = Math::maxPerElement(xa, xb) + Math::maxPerElement(ya, yb) + Math::maxPerElement(za, zb) + mw;
    }

    bool BoundingBox::Contains(const Vec3f& _point) const
    {
        const Vec3f center = (min + max) / 2.f;
        const Vec3f extent = (max - min) / 2.f;

        const Vec3f localPosition = _point - center;

        return    abs(localPosition.x) <= extent.x
            &&    abs(localPosition.y) <= extent.y
            &&    abs(localPosition.z) <= extent.z;
    }

    SerializeError BoundingBox::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("min", min);
            _serializer.Serialize("max", max);
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }



    // --- Bounding sphere -------------------------------------------------------------

    BoundingSphere::BoundingSphere(const Vec3f& _origin, const float _radius)
    {
        origin = _origin;
        radius = _radius;
    }

    BoundingSphere::BoundingSphere(const BoundingBox& _box)
    {
        origin = _box.GetOrigin();
        radius = 0.5f * Math::length(_box.GetMax() - _box.GetMin());
    }

    BoundingSphere::BoundingSphere(const Frustum& _frustum)
    {
        origin = Math::Origin;
        radius = 0.f;

        // computes sphere origin
        for (int i = 0; i < 8; i++)
        {
            origin += _frustum.vertices[i];
        }
        origin /= 8.f;

        // deduces sphere radius
        for (int i = 0; i < 8; i++)
        {
            radius = Math::max(radius, Math::normSquared(_frustum.vertices[i] - origin));
        }
        radius = sqrtf(radius);
    }

    BoundingSphere::BoundingSphere(const BoundingSphere& _sphere, const Mat4& _transform)
    {
        origin = _sphere.origin;
        radius = _sphere.radius;
        Transform(_transform);
    }

    bool BoundingSphere::operator==(const BoundingSphere& _other) const
    {
         return
            origin == _other.origin
            && fabs(radius - _other.radius) < FLT_MIN; 
    }

    bool BoundingSphere::operator!=(const BoundingSphere& _other) const
    {
        return !(*this == _other);
    }

    Vec3f BoundingSphere::GetOrigin() const
    {
        return origin;
    }

    float BoundingSphere::GetRadius() const
    {
        return radius;
    }

    void BoundingSphere::Grow(const float _radius)
    {
        radius = Math::max(radius, _radius);
    }

    void BoundingSphere::Transform(const Mat4& _transform)
    {
        const Vec3f scale = Math::getScale(_transform);
        const float maxScale = Math::max3(scale);

        radius *= maxScale;
        origin *= maxScale;

        origin += Math::getTranslation(_transform);
    }

    bool BoundingSphere::Contains(const Vec3f& _point) const
    {
        const float distanceToOrigin = Math::distance(_point, origin);
        return distanceToOrigin < radius;
    }

    SerializeError BoundingSphere::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("origin", origin);
            _serializer.Serialize("radius", radius);
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }



    // --- Plane -----------------------------------------------------------------------

    Plane::Plane(const Vec3f& _normal, const float _offset)
    {
        normal = _normal;
        offset = _offset;
    }

    Plane::Plane(const Vec3f& _normal, const Vec3f& _origin)
    {
        normal = _normal;
        offset = Math::dot(_normal, _origin);
    }

    Plane::Plane(const float _a, const float _b, const float _c, const float _d)
    {
        normal = {_a, _b, _c};
        AVA_ASSERT(Math::length(normal) > 0);
        Math::normalize(normal);

        offset = _d / Math::normSquared(normal);
    }

    Plane::Plane(const Vec3f& _p1, const Vec3f& _p2, const Vec3f& _p3)
    {
        const Vec3f u(_p2 - _p1);
        const Vec3f v(_p3 - _p1);

        normal = Math::cross(u, v);
        normal = Math::normalize(normal);

        offset = Math::dot(normal, (_p1 + _p2 + _p3) / 3.f);
    }

    bool Plane::operator==(const Plane& _other) const
    {
        return
            normal == _other.normal
            && fabs(offset - _other.offset) < FLT_MIN;
    }

    bool Plane::operator!=(const Plane& _other) const
    {
        return !(*this == _other);
    }

    float Plane::GetOffset() const
    {
        return offset;
    }

    Vec3f Plane::GetNormal() const
    {
        return normal;
    }

    Vec3f Plane::GetOrigin() const
    {
        return normal * offset;
    }

    float Plane::Distance(const Vec3f& _point) const
    {
        return Math::dot(normal, _point) - offset;
    }

    Vec3f Plane::Closest(const Vec3f& _point) const
    {
        return _point - Distance(_point) * normal;
    }

    void Plane::Transform(const Mat4& _transform)
    {
        const Vec3f origin = Math::transformPosition(_transform, GetOrigin());
        normal = Math::transformDirection(_transform, normal);
        offset = Math::dot(normal, origin);
    }

    SerializeError Plane::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.Serialize("normal", normal);
            _serializer.Serialize("offset", offset);
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }


    // --- Frustum ---------------------------------------------------------------------

    Frustum::Frustum(const float _top, const float _down, const float _right, const float _left, const float _near, const float _far)
    {
        // near plane
        vertices[0] = Vec3f(_left, _top, -_near);
        vertices[1] = Vec3f(_right, _top, -_near);
        vertices[2] = Vec3f(_right, _down, -_near);
        vertices[3] = Vec3f(_left, _down, -_near);

        // far plane
        vertices[4] = Vec3f(_left, _top, -_far);
        vertices[5] = Vec3f(_right, _top, -_far);
        vertices[6] = Vec3f(_right, _down, -_far);
        vertices[7] = Vec3f(_left, _down, -_far);

        InitPlanesFromVertices();
    }

    Frustum::Frustum(const float _fovRadians, const float _aspectRatio, const float _near, const float _far)
    {
        const float tanHalfFov = tanf(_fovRadians * 0.5f);

        // near plane
        {
            const float top = tanHalfFov * _near;
            const float down = -tanHalfFov * _near;
            const float right = _aspectRatio * tanHalfFov * _near;
            const float left = -_aspectRatio * tanHalfFov * _near;

            vertices[0] = Vec3f(left, top, -_near);
            vertices[1] = Vec3f(right, top, -_near);
            vertices[2] = Vec3f(right, down, -_near);
            vertices[3] = Vec3f(left, down, -_near);
        }

        // far plane
        {
            const float top = tanHalfFov * _far;
            const float down = -tanHalfFov * _far;
            const float right = _aspectRatio * tanHalfFov * _far;
            const float left = -_aspectRatio * tanHalfFov * _far;

            vertices[4] = Vec3f(left, top, -_far);
            vertices[5] = Vec3f(right, top, -_far);
            vertices[6] = Vec3f(right, down, -_far);
            vertices[7] = Vec3f(left, down, -_far);
        }

        InitPlanesFromVertices();
    }

    Frustum::Frustum(const Frustum& _frustum, const float _nearOffset, const float _farOffset)
    {
        const Vec3f depth = _frustum.planes[FarPlane].GetOrigin() - _frustum.planes[NearPlane].GetOrigin();
        const float d = Math::length(depth);
        const float n = _nearOffset / d;
        const float f = _farOffset  / d;

        for (int i = 0; i < 4; ++i) 
        {
            vertices[i]     = Math::lerp(_frustum.vertices[i], _frustum.vertices[i + 4], n);
            vertices[i + 4] = Math::lerp(_frustum.vertices[i], _frustum.vertices[i + 4], 1.f + f);
        }

        InitPlanesFromVertices();
    }

    Frustum::Frustum(const Frustum& _frustum, const Mat4& _transform)
    {
        for (int i = 0; i < 8; ++i)
        {
            vertices[i] = _frustum.vertices[i];
        }

        InitPlanesFromVertices();
        Transform(_transform);
    }

    bool Frustum::Contains(const Vec3f& _point) const
    {
        for (int i = 0; i < PlaneCount; ++i)
        {
            // the frustum planes are oriented towards the outside the frustum,
            // so distance > 0 means the point is beyond the frustum plane.
            if (planes[i].Distance(_point) > 0)
            {
                return false;
            }
        }
        return true;
    }

    bool Frustum::Contains(const BoundingBox& _box) const
    {
        const Vec3f boxMin = _box.GetMin();
        const Vec3f boxMax = _box.GetMax();

        for (int i = 0; i < PlaneCount; i++)
        {
            const Vec3f normal = planes[i].normal;
            const float offset = planes[i].offset;

            const float distance = 
                  Math::min(boxMin.x * normal.x, boxMax.x * normal.x)
                + Math::min(boxMin.y * normal.y, boxMax.y * normal.y)
                + Math::min(boxMin.z * normal.z, boxMax.z * normal.z)
                - offset;

            // the frustum planes are oriented towards outside the frustum,
            // so distance > 0 means the point is beyond the frustum plane.
            if (distance > 0)
            {
                return false;
            }
        }
        return true;
    }

    bool Frustum::Contains(const BoundingSphere& _sphere) const
    {
        const Vec3f origin = _sphere.GetOrigin();
        const float radius = _sphere.GetRadius();

        for (int i = 0; i < PlaneCount; ++i) 
        {
            // the frustum planes are oriented towards the outside the frustum, so distance > radius
            // means the closest point of the sphere to the frustum plane is beyond the plane.
            if (planes[i].Distance(origin) > radius)
            {
                return false;
            }
        }
        return true;
    }

    void Frustum::SetNearFar(const float _near, const float _far)
    {
        for (int i = 0; i < 4; ++i) 
        {
            vertices[i].z = _near;
            vertices[i + 4].z = _far;
        }
        planes[NearPlane].offset = _near;
        planes[FarPlane].offset = _far;
    }

    void Frustum::InitPlanesFromVertices()
    {
        planes[NearPlane]   = Plane(vertices[2], vertices[1], vertices[0]);
        planes[FarPlane]    = Plane(vertices[4], vertices[5], vertices[6]);
        planes[TopPlane]    = Plane(vertices[1], vertices[5], vertices[4]);
        planes[RightPlane]  = Plane(vertices[6], vertices[5], vertices[1]);
        planes[BottomPlane] = Plane(vertices[3], vertices[7], vertices[6]);
        planes[LeftPlane]   = Plane(vertices[3], vertices[0], vertices[4]);
    }

    void Frustum::Transform(const Mat4& _transform)
    {
        for (int i = 0; i < PlaneCount; ++i)
        {
             planes[i].Transform(_transform);
        }

        for (int i = 0; i < 8; ++i) 
        {
            vertices[i] = Math::transformPosition(_transform, vertices[i]);
        }
    }

    SerializeError Frustum::Serialize(Serializer& _serializer, const char* _tag)
    {
        if (_serializer.OpenSection(_tag) == SerializeError::None)
        {
            _serializer.SerializeArray("vertices", vertices, 8);

            if (_serializer.IsReading())
            {
                InitPlanesFromVertices();
            }
        }

        _serializer.CloseSection(_tag);
        return SerializeError::None;
    }

}
