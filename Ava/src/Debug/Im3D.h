#pragma once
/// @file Im3D.h
/// @brief

#include <Math/Types.h>
#include <Math/Geometry.h>

namespace Ava {

    using Im3dColType = u32;

    #define IM3D_COL_CURRENT        (0x12345678)
    #define IM3D_COL_NONE           (0x00000000)
    #define IM3D_COL_BLACK          (0xFF000000)
    #define IM3D_COL_WHITE          (0xFFFFFFFF)
    #define IM3D_COL_GRAY           (0xFF808080)
    #define IM3D_COL_DARK_GRAY      (0xFF404040)
    #define IM3D_COL_RED            (0xFF0000FF)
    #define IM3D_COL_GREEN          (0xFF00FF00)
    #define IM3D_COL_BLUE           (0xFFFF0000)
    #define IM3D_COL_YELLOW         (0xFF00FFFF)
    #define IM3D_COL_MAGENTA        (0xFFFF00FF)
    #define IM3D_COL_CYAN           (0xFFFFFF00)
    #define IM3D_COL_ORANGE         (0xFF0080FF)
    #define IM3D_COL_LIGHT_GREEN    (0xFF55FF55)

    static constexpr float kDefaultThickness = 4.f;

    struct Im3dVertex
    {
        Vec3f position;
        u32 color;
    };

    struct Im3dLineVertex
    {
        Vec3f position;
        Vec3f tangent;
        u32 color;
    };

    enum Im3dDepthTest
    {
        Im3dDepthTest_Disable = 0,
        Im3dDepthTest_Enable,
        Im3dDepthTest_TransparentWhenBehind,

        Im3dDepthTest_Count
    };

    struct Im3dDrawSettings
    {
        float thickness = 1.f;
        Im3dColType color = IM3D_COL_CURRENT;
        Im3dDepthTest depthTest = Im3dDepthTest_TransparentWhenBehind;
    };

    class Im3dDrawList
    {
    public:
        Im3dDrawList() = default;
        ~Im3dDrawList() { Clear(); }

        // ------- Primitives ------------------------------------------------------------------

        void AddLine(const Vec3f& _from, const Vec3f& _to, const Im3dDrawSettings& _settings);
        void AddLines(const Vec3f* _points, u32 _count, const Im3dDrawSettings& _settings);
        void AddLines(const Vec3f* _points, const Im3dColType* _colors, u32 _count, const Im3dDrawSettings& _settings);

        void AddLineStrip(const Vec3f* _points, u32 _count, bool _closed, const Im3dDrawSettings& _settings);
        void AddLineStrip(const Vec3f* _points, const Im3dColType* _colors, u32 _count, bool _closed, const Im3dDrawSettings& _settings);

        void AddTriangle(const Vec3f _points[3], const Im3dDrawSettings& _settings);
        void AddTriangleFilled(const Vec3f _points[3], const Im3dDrawSettings& _settings);

        // ------- Complex shapes --------------------------------------------------------------

        void AddRect(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddRect(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddBox(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddBox(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddCircle(const Vec3f& _center, float _radius, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddCircle(const Vec3f& _center, float _radius, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddSphere(const Vec3f& _center, float _radius, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddSphere(const Vec3f& _center, float _radius, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddCapsule(const Vec3f& _center, float _radius, float _height, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings);
        void AddCapsule(const Vec3f& _center, float _radius, float _height, const Mat4& _transform, const Im3dDrawSettings& _settings);

        void AddArrow(const Vec3f& _from, const Vec3f& _to, const Im3dDrawSettings& _settings,
            float _headBaseSize = 0.f, float _headLengthSize = 0.f);

        void AddGuizmo(const Mat4& _transform, float _size, const Im3dDrawSettings& _settings, 
            Im3dColType _xColor = IM3D_COL_RED, Im3dColType _yColor = IM3D_COL_GREEN, Im3dColType _zColor = IM3D_COL_BLUE);

        void AddBoundingBox(const BoundingBox& _bBox, const Im3dDrawSettings& _settings);
        void AddBoundingSphere(const BoundingSphere& _bSphere, const Im3dDrawSettings& _settings);
        void AddPlane(const Plane& _plane, const Im3dDrawSettings& _settings);
        void AddFrustum(const Frustum& _frustum, const Im3dDrawSettings& _settings);

        // ------- Utilities -------------------------------------------------------------------

        void Clear();
        void ClearMemory();

        u32 GetTotalLineVertexCount() const;
        u32 GetTotalTriangleVertexCount() const;

        std::vector<Im3dLineVertex> m_lineVtxBuffer[Im3dDepthTest_Count];
        std::vector<Im3dVertex> m_triangleVtxBuffer[Im3dDepthTest_Count];
    };

    struct Im3dDrawData
    {
        bool            valid;                  // Only valid after Render() is called and before the next NewFrame() is called.
        int             cmdListsCount;          // Number of ImDrawList* to render
        int             totalLineVtxCount;      // Sum of all Im3dDrawList's lineVtxBuffer size
        int             totalTriangleVtxCount;  // Sum of all Im3dDrawList's triangleVtxBuffer size
        Im3dDrawList**  cmdLists;               // Array of ImDraw3dList* to render.

        Im3dDrawData() { Clear(); }
        void Clear() { memset(this, 0, sizeof(*this)); }
    };

    namespace Im3d
    {
        void CreateContext();
        void DestroyContext();

        void NewFrame();
        void Render();
        Im3dDrawData* GetDrawData();

        void PushColor(Im3dColType _color);
        void PopColor();

        void PushDepthTest(Im3dDepthTest _depthTest);
        void PopDepthTest();

        // ------- Primitives ------------------------------------------------------------------

        void AddLine(const Vec3f& _from, const Vec3f& _to, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddLines(const Vec3f* _points, u32 _count,Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddLines(const Vec3f* _points, const Im3dColType* _colors, u32 _count, float _thickness = kDefaultThickness);

        void AddLineStrip(const Vec3f* _points, u32 _count, bool _closed, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddLineStrip(const Vec3f* _points,const Im3dColType* _colors, u32 _count, bool _closed, float _thickness = kDefaultThickness);

        void AddTriangle(const Vec3f _points[3], Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddTriangleFilled(const Vec3f _points[3], Im3dColType _color = IM3D_COL_CURRENT);

        // ------- Complex shapes --------------------------------------------------------------

        void AddRect(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddRect(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);

        void AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT);
        void AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT);

        void AddBox(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddBox(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);

        void AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT);
        void AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT);

        void AddCircle(const Vec3f& _center, float _radius, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddCircle(const Vec3f& _center, float _radius, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);

        void AddSphere(const Vec3f& _center, float _radius, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddSphere(const Vec3f& _center, float _radius, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);

        void AddCapsule(const Vec3f& _center, float _radius, float _height, const Vec3f& _front, const Vec3f& _up, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddCapsule(const Vec3f& _center, float _radius, float _height, const Mat4& _transform, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);

        void AddArrow(const Vec3f& _from, const Vec3f& _to, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness,
            float _headBaseSize = 0.f, float _headBaseLength = 0.f);

        void AddGuizmo(const Mat4& _transform, float _size = 1.f, float _thickness = 4.f, 
            Im3dColType _xColor = IM3D_COL_RED, Im3dColType _yColor = IM3D_COL_GREEN, Im3dColType _zColor = IM3D_COL_BLUE);

        void AddBoundingBox(const BoundingBox& _bBox, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddBoundingSphere(const BoundingSphere& _bSphere, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddPlane(const Plane& _plane, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
        void AddFrustum(const Frustum& _frustum, Im3dColType _color = IM3D_COL_CURRENT, float _thickness = kDefaultThickness);
    }

}
