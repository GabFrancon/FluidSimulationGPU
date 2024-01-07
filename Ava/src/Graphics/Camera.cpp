#include <avapch.h>
#include "Camera.h"

#include <Math/Math.h>

namespace Ava {

    Camera::Camera(const CameraSettings& _settings)
        : m_projSettings(_settings)
    {
    }

    Camera::Camera(const Camera& _other)
    {
        m_world = _other.m_world;
        m_view = _other.m_view;

        m_projSettings = _other.m_projSettings;
        m_localFrustum = _other.m_localFrustum;
        m_proj = _other.m_proj;
    }

    Camera::Camera(Camera&& _other) noexcept
    {
        m_world = _other.m_world;
        m_view = _other.m_view;

        m_projSettings = _other.m_projSettings;
        m_localFrustum = _other.m_localFrustum;
        m_proj = _other.m_proj;
    }

    Camera& Camera::operator=(const Camera& _other)
    {
        m_world = _other.m_world;
        m_view = _other.m_view;

        m_projSettings = _other.m_projSettings;
        m_localFrustum = _other.m_localFrustum;
        m_proj = _other.m_proj;

        return *this;
    }

    Camera& Camera::operator=(Camera&& _other) noexcept
    {
        m_world = _other.m_world;
        m_view = _other.m_view;

        m_projSettings = _other.m_projSettings;
        m_localFrustum = _other.m_localFrustum;
        m_proj = _other.m_proj;

        return *this;
    }

    const CameraSettings& Camera::GetSettings() const
    {
        return m_projSettings;
    }

    Mat4 Camera::GetWorld() const
    {
        return m_world;
    }

    Mat4 Camera::GetView() const
    {
        return m_view; 
    }

    Mat4 Camera::GetProj() const
    {
        return m_proj; 
    }

    Vec3f Camera::GetPosition() const
    {
        return Math::column(m_world, 3);
    }

    Vec3f Camera::GetViewVector() const
    {
        return Math::column(m_world, 2); 
    }

    Vec3f Camera::GetRightDirection() const
    {
         return Math::column(m_world, 0);
    }

    Vec3f Camera::GetUpDirection() const
    {
        return Math::column(m_world, 1); 
    }

    Vec3f Camera::GetFrontDirection() const
    {
        return -Math::column(m_world, 2);
    }

    Frustum Camera::GetLocalFrustum() const
    {
        return m_localFrustum;
    }

    Frustum Camera::GetWorldFrustum() const
    {
        return Frustum(m_localFrustum, m_world);
    }

    Vec3f Camera::GetViewFrustumRay(const Vec2f& _ndc) const
    {
        const float tanHalfFov = tanf(0.5f * m_projSettings.verticalFov);
        const float aspectRatio = m_projSettings.aspectRatio;

        Vec3f frustumRayV{};
        frustumRayV.x = _ndc.x * tanHalfFov * aspectRatio;
        frustumRayV.y = -_ndc.y * tanHalfFov;
        frustumRayV.z = -1.f;

        return frustumRayV;
    }

    Vec3f Camera::GetWorldFrustumRay(const Vec2f& _ndc) const
    {
        const Vec3f frustumRayV = GetViewFrustumRay(_ndc);
        return Mat3(m_world) * frustumRayV;
    }

    void Camera::SetPosition(const Vec3f& _position)
    {
        m_world[3] = Vec4f(_position, 1.f);
        m_view = Math::inverse(m_world);
    }

    void Camera::SetViewVector(const Vec3f& _viewVector)
    {
        m_world[2] = Vec4f(_viewVector, 1.f);
        m_view = Math::inverse(m_world);
    }

    void Camera::SetLookAt(const Vec3f& _from, const Vec3f& _to, const Vec3f& _up/*= Math::AxisY*/)
    {
        m_view = Math::lookAt(_from, _to, _up);
        m_world = Math::inverse(m_view);
    }

    void Camera::SetPerspective(const float _fovRadians, const float _aspect, const float _near, const float _far, const u32 _projFlags/*= 0*/)
    {
        // update projection settings
        m_projSettings.verticalFov = _fovRadians;
        m_projSettings.aspectRatio = _aspect;
        m_projSettings.nearPlane = _near;
        m_projSettings.farPlane = _far;

        // recompute local frustum
        m_localFrustum = Frustum(m_projSettings.verticalFov, m_projSettings.aspectRatio, m_projSettings.nearPlane, m_projSettings.farPlane);

        // recompute projection matrix
        const u32 projFlags = _projFlags ? _projFlags : DefaultProjectionFlags;
        const bool isInfinite = projFlags & AVA_CAMERA_INFINITE;
        const bool isReversed = projFlags & AVA_CAMERA_REVERSED;

        if (isInfinite && isReversed) {
            m_proj = Math::perspectiveInfiniteReversed(m_projSettings.verticalFov, m_projSettings.aspectRatio, m_projSettings.nearPlane);
        }
        else if (isInfinite) {
            m_proj = Math::perspectiveInfinite(m_projSettings.verticalFov, m_projSettings.aspectRatio, m_projSettings.nearPlane);
        }
        else if (isReversed) {
            m_proj = Math::perspectiveReversed(m_projSettings.verticalFov, m_projSettings.aspectRatio, m_projSettings.nearPlane, m_projSettings.farPlane);
        }
        else {
            m_proj = Math::perspective(m_projSettings.verticalFov, m_projSettings.aspectRatio, m_projSettings.nearPlane, m_projSettings.farPlane);
        }
    }

    void Camera::SetOrtho(const float _left, const float _right, const float _down, const float _top, const float _near, const float _far, const u32 _projFlags/*= 0*/)
    {
        // update projection settings
        m_projSettings.left = _left;
        m_projSettings.right = _right;
        m_projSettings.down = _down;
        m_projSettings.top = _top;
        m_projSettings.nearPlane = _near;
        m_projSettings.farPlane = _far;
        m_projSettings.aspectRatio = fabs(_right - _left) / fabs(_top - _down);

        // recompute local frustum
        m_localFrustum = Frustum(m_projSettings.top, m_projSettings.down, m_projSettings.right, m_projSettings.left, m_projSettings.nearPlane, m_projSettings.farPlane);

        // recompute projection matrix
        const u32 projFlags = _projFlags ? _projFlags : DefaultProjectionFlags;
        const bool isInfinite = projFlags & AVA_CAMERA_INFINITE;
        const bool isReversed = projFlags & AVA_CAMERA_REVERSED;

        AVA_ASSERT(!isInfinite, "[Camera] infinite orthographic projection is not supported.");

        if (isReversed) {
            m_proj = Math::orthoReversed(m_projSettings.left, m_projSettings.right, m_projSettings.down, m_projSettings.top, m_projSettings.nearPlane, m_projSettings.farPlane);
        }
        else {
            m_proj = Math::ortho(m_projSettings.left, m_projSettings.right, m_projSettings.down, m_projSettings.top, m_projSettings.nearPlane, m_projSettings.farPlane);
        }
    }

}
