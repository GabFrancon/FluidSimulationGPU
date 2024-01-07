#pragma once
/// @file Camera.h
/// @brief

#include <Math/Types.h>
#include <Math/Geometry.h>

namespace Ava {

    enum CamProjectionFlags
    {
        AVA_CAMERA_NONE = 0,

        AVA_CAMERA_ORTHOGRAPHIC = AVA_BIT(0),
        AVA_CAMERA_PERSPECTIVE = AVA_BIT(1),
        AVA_CAMERA_INFINITE = AVA_BIT(2),
        AVA_CAMERA_REVERSED = AVA_BIT(3),
    };

    /// @brief Camera projection settings.
    struct CameraSettings
    {
        // ortho
        float left = 0.f;
        float right = 800.f;
        float down = 0.f;
        float top = 600.f;

        // perspective
        float verticalFov = 0.785398f; // 45°
        float aspectRatio = 19.f / 16.f; // 16:19

        // common
        float nearPlane = 0.1f;
        float farPlane = 600.1f;
    };

    class Camera
    {
    public:
        Camera() = default;
        explicit Camera(const CameraSettings& _settings);
        virtual ~Camera() = default;

        Camera(const Camera& _other);
        Camera(Camera&& _other) noexcept;

        Camera& operator=(const Camera& _other);
        Camera& operator=(Camera&& _other) noexcept;

        const CameraSettings& GetSettings() const;

        Mat4 GetWorld() const;
        Mat4 GetView() const;
        Mat4 GetProj() const;

        Vec3f GetPosition() const;
        Vec3f GetViewVector() const;
        Vec3f GetRightDirection() const;
        Vec3f GetUpDirection() const;
        Vec3f GetFrontDirection() const;

        Frustum GetLocalFrustum() const;
        Frustum GetWorldFrustum() const;

        Vec3f GetViewFrustumRay(const Vec2f& _ndc) const;
        Vec3f GetWorldFrustumRay(const Vec2f& _ndc) const;

        virtual void SetPosition(const Vec3f& _position);
        virtual void SetViewVector(const Vec3f& _viewVector);
        virtual void SetLookAt(const Vec3f& _from, const Vec3f& _to, const Vec3f& _up = Math::AxisY);

        virtual void SetPerspective(float _fovRadians, float _aspect, float _near, float _far, u32 _projFlags = 0);
        virtual void SetOrtho(float _left, float _right, float _down, float _top, float _near, float _far, u32 _projFlags = 0);

        // Default projection flags, can be statically changed by the client application.
        inline static u32 DefaultProjectionFlags = AVA_CAMERA_PERSPECTIVE;

     protected:
        // Cam view
        Mat4 m_world = Math::Identity4;
        Mat4 m_view = Math::Identity4;

        // Cam projection
        CameraSettings m_projSettings;
        Mat4 m_proj = Math::Identity4;
        Frustum m_localFrustum{};
    };


}
