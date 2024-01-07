#include <avapch.h>
#include "Im3D.h"

#include <Math/Math.h>

namespace Ava {

    //------ Im3d context data ------------------------------------------------------------------------------------------------------

    static constexpr int kCircleSegmentCount = 24;
    static constexpr int kSphereSegmentCount = 4;

    static constexpr int kColorStackSize = 32;
    static constexpr int kDepthTestStackSize = 32;

    static Im3dDrawData* s_drawData = nullptr;
    static Im3dDrawList* s_drawList = nullptr;

    static Im3dColType s_colorStack[kColorStackSize];
    static Im3dColType s_currentColor;
    static u32 s_currentColorIndex;

    static Im3dDepthTest s_depthTestStack[kDepthTestStackSize];
    static Im3dDepthTest s_currentDepthTest;
    static u32 s_currentDepthTestIndex;

    Mat4 BuildRotationMatrix(const Vec3f& _front, const Vec3f& _up)
    {
        constexpr float kEpsilon = 0.001f;

        const Vec3f nFront = Math::normalize(_front);
        Vec3f nUp = Math::normalize(_up);
        Vec3f right = Math::cross(nFront, nUp);

        if (Math::length(right) < kEpsilon)
        {
            const Vec3f tempNUp = Vec3f(nUp.y, nUp.z, -nUp.x);
            right = Math::cross(tempNUp, nFront);
            AVA_ASSERT(Math::length(right) > kEpsilon);
        }
        const Vec3f nRight = Math::normalize(right);
        nUp = Math::cross(nFront, nRight);
        nUp = Math::normalize(nUp);

        Mat4 mat = Math::Identity4;
        mat[0] = Vec4f(nRight, 0.f);
        mat[1] = Vec4f(nFront, 0.f);
        mat[2] = Vec4f(nUp, 0.f);
        return mat;
    }

    void BuildFlatTriangleLink(std::vector<Im3dLineVertex>& _vtxBuffer, const Im3dLineVertex& _nextVtx)
    {
        if (!_vtxBuffer.empty())
        {
            const Im3dLineVertex back = _vtxBuffer.back();
            _vtxBuffer.push_back(back);
            _vtxBuffer.push_back(_nextVtx);
        }
    }

    void BuildLineVertices(std::vector<Im3dLineVertex>& _vtxBuffer, const Vec3f* _points, int _count, Im3dColType _color, float _thickness)
    {
        _vtxBuffer.reserve(_vtxBuffer.size() + _count);

        for (int i = 0; i < _count; i += 2)
        {
            Vec3f lineDir = _points[i+1] - _points[i];
            lineDir = Math::normalize(lineDir);

            Im3dLineVertex vtx{};
            vtx.position = _points[i];
            vtx.tangent = lineDir * _thickness;
            vtx.color = _color;

            BuildFlatTriangleLink(_vtxBuffer, vtx);
            _vtxBuffer.push_back(vtx);

            vtx.tangent = -vtx.tangent;
            _vtxBuffer.push_back(vtx);

            vtx.position = _points[i+1];
            vtx.tangent = -vtx.tangent;
            _vtxBuffer.push_back(vtx);

            vtx.tangent = -vtx.tangent;
            _vtxBuffer.push_back(vtx);
        }
    }

    void BuildLineVertices(std::vector<Im3dLineVertex>& _vtxBuffer, const Vec3f* _points, const Im3dColType* _colors, int _count, float _thickness)
    {
        _vtxBuffer.reserve(_vtxBuffer.size() + _count);

        for (int i = 0; i < _count; i += 2)
        {
            Vec3f lineDir = _points[i+1] - _points[i];
            lineDir = Math::normalize(lineDir);

            Im3dLineVertex vtx{};
            vtx.position = _points[i];
            vtx.tangent = lineDir * _thickness;
            vtx.color = _colors[i];

            BuildFlatTriangleLink(_vtxBuffer, vtx);
            _vtxBuffer.push_back(vtx);

            vtx.tangent = -vtx.tangent;
            _vtxBuffer.push_back(vtx);

            vtx.position = _points[i+1];
            vtx.tangent = -vtx.tangent;
            _vtxBuffer.push_back(vtx);

            vtx.tangent = -vtx.tangent;
            _vtxBuffer.push_back(vtx);
        }
    }


    //------ Im3dDrawList primitives ----------------------------------------------------------------------------------------------

    void Im3dDrawList::AddLine(const Vec3f& _from, const Vec3f& _to, const Im3dDrawSettings& _settings)
    {
        const Vec3f points[2] = { _from, _to };
        BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], points, 2, _settings.color, _settings.thickness);
    }

    void Im3dDrawList::AddLines(const Vec3f* _points, const u32 _count, const Im3dDrawSettings& _settings)
    {
        BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], _points, _count, _settings.color, _settings.thickness);
    }

    void Im3dDrawList::AddLines(const Vec3f* _points, const Im3dColType* _colors, const u32 _count, const Im3dDrawSettings& _settings)
    {
        BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], _points, _colors, _count, _settings.thickness);
    }

    void Im3dDrawList::AddLineStrip(const Vec3f* _points, const u32 _count, const bool _closed, const Im3dDrawSettings& _settings)
    {
        for (u32 i = 0; i < _count - 1; i++)
        {
            const Vec3f line[2] = { _points[i], _points[i+1] };
            BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], line, 2, _settings.color, _settings.thickness);
        }
        if (_closed)
        {
            const Vec3f lastLine[2] = { _points[_count - 1], _points[0] };
            BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], lastLine, 2, _settings.color, _settings.thickness);
        }
    }

    void Im3dDrawList::AddLineStrip(const Vec3f* _points, const Im3dColType* _colors, const u32 _count, const bool _closed, const Im3dDrawSettings& _settings)
    {
        for (u32 i = 0; i < _count - 1; i++)
        {
            const Vec3f line[2] = { _points[i], _points[i+1] };
            const Im3dColType lineColor[2] = { _colors[i], _colors[i+1] };
            BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], line, lineColor, 2, _settings.thickness);
        }
        if (_closed)
        {
            const Vec3f lastLine[2] = { _points[_count - 1], _points[0] };
            const Im3dColType lastLineColor[2] = { _colors[_count - 1], _colors[0] };
            BuildLineVertices(m_lineVtxBuffer[_settings.depthTest], lastLine, lastLineColor, 2, _settings.thickness);
        }
    }

    void Im3dDrawList::AddTriangle(const Vec3f _points[3], const Im3dDrawSettings& _settings)
    {
        AddLineStrip(_points, 3, true, _settings);
    }

    void Im3dDrawList::AddTriangleFilled(const Vec3f _points[3], const Im3dDrawSettings& _settings)
    {
        m_triangleVtxBuffer[_settings.depthTest].push_back({ _points[0], _settings.color });
        m_triangleVtxBuffer[_settings.depthTest].push_back({ _points[1], _settings.color });
        m_triangleVtxBuffer[_settings.depthTest].push_back({ _points[2], _settings.color });
    }


    //------ Im3dDrawList complex shapes -----------------------------------------------------------------------------------------

    void Im3dDrawList::AddRect(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddRect(_center, _size, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddRect(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        const Vec3f translation = Math::getTranslation(_transform);
        const Mat4 rotationMatrix = Math::getRotation(_transform);
        const Vec3f scale = Math::getScale(_transform);

        Vec2f halfSize = _size / 2.f;
        halfSize.x *= scale.x;
        halfSize.y *= scale.y;

        const Vec3f widthVector = Math::transformPosition(rotationMatrix, {halfSize.x, 0.f, 0.f});
        const Vec3f heightVector = Math::transformPosition(rotationMatrix, {0.f, halfSize.y, 0.f});

        const Vec3f offsettedCenter = _center + translation;

        Vec3f points[4]{};
        points[0] = offsettedCenter + widthVector + heightVector;
        points[1] = offsettedCenter + widthVector - heightVector;
        points[2] = offsettedCenter - widthVector - heightVector;
        points[3] = offsettedCenter - widthVector + heightVector;

        AddLineStrip(points, 4, true, _settings);
    }

    void Im3dDrawList::AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddRectFilled(_center, _size, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        const Vec3f translation = Math::getTranslation(_transform);
        const Mat4 rotationMatrix = Math::getRotation(_transform);
        const Vec3f scale = Math::getScale(_transform);

        Vec2f halfSize = _size / 2.f;
        halfSize.x *= scale.x;
        halfSize.y *= scale.y;

        const Vec3f widthVector = Math::transformPosition(rotationMatrix, {halfSize.x, 0.f, 0.f});
        const Vec3f heightVector = Math::transformPosition(rotationMatrix, {0.f, halfSize.y, 0.f});

        const Vec3f offsettedCenter = _center + translation;

        Vec3f points[4]{};
        points[0] = offsettedCenter + widthVector + heightVector;
        points[1] = offsettedCenter + widthVector - heightVector;
        points[2] = offsettedCenter - widthVector - heightVector;
        points[3] = offsettedCenter - widthVector + heightVector;

        const Vec3f firstTriangle[3] = { points[0], points[1], points[2] };
        AddTriangleFilled(firstTriangle, _settings);

        const Vec3f secondTriangle[3] = { points[0], points[2], points[3] };
        AddTriangleFilled(secondTriangle, _settings);
    }

    void Im3dDrawList::AddBox(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddBox(_center, _size, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddBox(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        const Vec3f translation = Math::getTranslation(_transform);
        const Mat4 rotationMatrix = Math::getRotation(_transform);
        const Vec3f scale = Math::getScale(_transform);

        Vec3f halfSize = _size / 2.f;
        halfSize.x *= scale.x;
        halfSize.y *= scale.y;
        halfSize.z *= scale.z;

        const Vec3f widthVector = Math::transformPosition(rotationMatrix, {halfSize.x, 0.f, 0.f});
        const Vec3f heightVector = Math::transformPosition(rotationMatrix, {0.f, halfSize.y, 0.f});
        const Vec3f depthVector = Math::transformPosition(rotationMatrix, {0.f, 0.f, halfSize.z});

        const Vec3f offsettedCenter = _center + translation;

        Vec3f upPoints[4]{};
        upPoints[0] = offsettedCenter + heightVector + widthVector + depthVector;
        upPoints[1] = offsettedCenter + heightVector + widthVector - depthVector;
        upPoints[2] = offsettedCenter + heightVector - widthVector - depthVector;
        upPoints[3] = offsettedCenter + heightVector - widthVector + depthVector;

        Vec3f downPoints[4]{};
        downPoints[0] = offsettedCenter - heightVector + widthVector + depthVector;
        downPoints[1] = offsettedCenter - heightVector + widthVector - depthVector;
        downPoints[2] = offsettedCenter - heightVector - widthVector - depthVector;
        downPoints[3] = offsettedCenter - heightVector - widthVector + depthVector;

        AddLineStrip(upPoints, 4, true, _settings);
        AddLineStrip(downPoints, 4, true, _settings);

        for (int i = 0; i < 4; i++)
        {
            AddLine(upPoints[i], downPoints[i], _settings);
        }
    }

    void Im3dDrawList::AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddBoxFilled(_center, _size, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        const Vec3f translation = Math::getTranslation(_transform);
        const Mat4 rotationMatrix = Math::getRotation(_transform);
        const Vec3f scale = Math::getScale(_transform);

        Vec3f halfSize = _size / 2.f;
        halfSize.x *= scale.x;
        halfSize.y *= scale.y;
        halfSize.z *= scale.z;

        const Vec3f widthVector = Math::transformPosition(rotationMatrix, {halfSize.x, 0.f, 0.f});
        const Vec3f heightVector = Math::transformPosition(rotationMatrix, {0.f, halfSize.y, 0.f});
        const Vec3f depthVector = Math::transformPosition(rotationMatrix, {0.f, 0.f, halfSize.z});

        const Vec3f offsettedCenter = _center + translation;

        Vec3f upPoints[4]{};
        upPoints[0] = offsettedCenter + heightVector + widthVector + depthVector;
        upPoints[1] = offsettedCenter + heightVector + widthVector - depthVector;
        upPoints[2] = offsettedCenter + heightVector - widthVector - depthVector;
        upPoints[3] = offsettedCenter + heightVector - widthVector + depthVector;

        Vec3f downPoints[4]{};
        downPoints[0] = offsettedCenter - heightVector + widthVector + depthVector;
        downPoints[1] = offsettedCenter - heightVector + widthVector - depthVector;
        downPoints[2] = offsettedCenter - heightVector - widthVector - depthVector;
        downPoints[3] = offsettedCenter - heightVector - widthVector + depthVector;

        // UP
        Vec3f pts[3]{};
        for (int i = 0; i < 3; i++) pts[i] = upPoints[i];
        AddTriangleFilled(pts, _settings);

        pts[0] = upPoints[0];
        pts[1] = upPoints[2];
        pts[2] = upPoints[3];
        AddTriangleFilled(pts, _settings);

        // DOWN
        for (int i = 0; i < 3; i++) pts[i] = downPoints[i];
        AddTriangleFilled(pts, _settings);

        pts[0] = downPoints[0];
        pts[1] = downPoints[2];
        pts[2] = downPoints[3];
        AddTriangleFilled(pts, _settings);

        // SIDES
        for (int i = 0; i < 4; ++i)
        {
            pts[0] = downPoints[i];
            pts[1] = upPoints[i];
            pts[2] = upPoints[(i + 1) % 4];
            AddTriangleFilled(pts, _settings);

            pts[0] = upPoints[(i + 1) % 4];
            pts[1] = downPoints[(i + 1) % 4];
            pts[2] = downPoints[i];
            AddTriangleFilled(pts, _settings);
        }
    }

    void Im3dDrawList::AddCircle(const Vec3f& _center, const float _radius, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddCircle(_center, _radius, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddCircle(const Vec3f& _center, const float _radius, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        constexpr float angleStep = Math::TwoPI / kCircleSegmentCount;
        float angle = 0.f;

        const Vec3f translation = Math::getTranslation(_transform);
        const Mat4 rotationMatrix = Math::getRotation(_transform);

        const Vec3f transformedCenter = _center + translation;
        Vec3f points[kCircleSegmentCount]{};

        for (int i = 0; i < kCircleSegmentCount; i++)
        {
            const Vec3f direction = Math::transformDirection(rotationMatrix, { cosf(angle), sinf(angle), 0.f});
            points[i] = transformedCenter + _radius * direction;
            angle += angleStep;
        }

        AddLineStrip(points, kCircleSegmentCount, true, _settings);
    }

    void Im3dDrawList::AddSphere(const Vec3f& _center, const float _radius, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddSphere(_center, _radius, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddSphere(const Vec3f& _center, const float _radius, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        constexpr float hAngleStep = Math::PI / (float)kSphereSegmentCount;
        constexpr float angleStep = Math::TwoPI / kCircleSegmentCount;

        float hAngle = 0.f;
        for (u8 i = 0; i < kSphereSegmentCount; i++)
        {
            Mat4 rotationMatrix = Math::Identity4;
            Math::rotate(rotationMatrix, Math::AxisY, hAngle);

            float angle = 0.f;
            Vec3f points[kCircleSegmentCount]{};

            for (int j = 0; j < kCircleSegmentCount; j++)
            {
                Vec3f position = { cosf(angle), sinf(angle), 0.f };
                position *= _radius;

                points[j] = _center + Math::transformPosition(_transform * rotationMatrix, position);
                angle += angleStep;
            }
            AddLineStrip(points, kCircleSegmentCount, true, _settings);
            hAngle += hAngleStep;
        }

        Mat4 rotationMatrix = Math::Identity4;
        Math::rotate(rotationMatrix, Math::AxisX, Math::HalfPI);

        AddCircle(_center, _radius, _transform * rotationMatrix, _settings);
    }

    void Im3dDrawList::AddCapsule(const Vec3f& _center, const float _radius, const float _height, const Vec3f& _front, const Vec3f& _up, const Im3dDrawSettings& _settings)
    {
        AddCapsule(_center, _radius, _height, BuildRotationMatrix(_front, _up), _settings);
    }

    void Im3dDrawList::AddCapsule(const Vec3f& _center, const float _radius, const float _height, const Mat4& _transform, const Im3dDrawSettings& _settings)
    {
        const auto halfExtents = Vec3f(0.f, _height * 0.5f, 0.f);
        constexpr float hAngleStep = Math::PI / (float)kSphereSegmentCount;
        constexpr float angleStep = Math::PI / (float)kCircleSegmentCount;
        float hAngle = 0.f;

        for (int i = 0; i < kSphereSegmentCount; ++i)
        {
            Mat4 halfCircleTransformMatrix = Math::Identity4;
            Math::rotate(halfCircleTransformMatrix, Math::AxisY, hAngle);
            halfCircleTransformMatrix = _transform * halfCircleTransformMatrix;

            Vec3f upPoints[kCircleSegmentCount + 1]{};
            Vec3f downPoints[kCircleSegmentCount + 1]{};

            float angle = 0.f;
            for (int j = 0; j < kCircleSegmentCount + 1; ++j)
            {
                Vec3f upPosition = { cosf(angle), sinf(angle), 0.f };
                upPosition *= _radius;
                upPosition += halfExtents;

                Vec3f downPosition = { cosf(angle + Math::PI), sinf(angle + Math::PI), 0.f };
                downPosition *= _radius;
                downPosition -= halfExtents;

                upPoints[j] = _center + Math::transformPosition(halfCircleTransformMatrix, upPosition);
                downPoints[j] = _center + Math::transformPosition(halfCircleTransformMatrix, downPosition);

                angle += angleStep;
            }

            AddLineStrip(upPoints, kCircleSegmentCount + 1, false, _settings);
            AddLineStrip(downPoints, kCircleSegmentCount + 1, false, _settings);

            AddLine(upPoints[0], downPoints[kCircleSegmentCount], _settings);
            AddLine(upPoints[kCircleSegmentCount], downPoints[0], _settings);

            hAngle += hAngleStep;
        }

        Mat4 edgeCircleRotationMatrix = Math::Identity4;
        Math::rotate(edgeCircleRotationMatrix, Math::AxisX, Math::HalfPI);
        edgeCircleRotationMatrix = _transform * edgeCircleRotationMatrix;

        const Vec3f transformedHalfExtents = Mat3(Math::getRotation(_transform)) * halfExtents;
        AddCircle(_center + transformedHalfExtents, _radius, edgeCircleRotationMatrix, _settings);
        AddCircle(_center - transformedHalfExtents, _radius, edgeCircleRotationMatrix, _settings);
    }

    void Im3dDrawList::AddArrow(const Vec3f& _from, const Vec3f& _to, const Im3dDrawSettings& _settings, float _headBaseSize, float _headLengthSize)
    {
        if (_headBaseSize == 0.f)
        {
            _headBaseSize = 0.2f * Math::length(_to - _from);
        }
        if (_headLengthSize == 0.f)
        {
            _headLengthSize = _headBaseSize;
        }

        const Vec3f front = normalize(_to - _from);
        Vec3f worldUp = Math::AxisY;

        // The given up vector is aligned with the front vector, which will result in NaN.
        // To workaround this, we rotate the up vector in the perpendicular plane.
        if (abs(dot(front, worldUp)) > 1.f -  FLT_EPSILON)
        {
            worldUp = Math::AxisX;
        }

        const Vec3f right = normalize(cross(front, worldUp));

        AddLine(_from, _to - front * _headLengthSize, _settings);

        Vec3f triangle[3]{};
        triangle[0] = _to;
        triangle[1] = triangle[0] - front * _headLengthSize + right * 0.5f * _headBaseSize;
        triangle[2] = triangle[0] - front * _headLengthSize - right * 0.5f * _headBaseSize;

        AddTriangleFilled(triangle, _settings);
    }

    void Im3dDrawList::AddGuizmo(const Mat4& _transform, const float _size, const Im3dDrawSettings& _settings, const Im3dColType _xColor, const Im3dColType _yColor, const Im3dColType _zColor)
    {
        Im3dDrawSettings settings = _settings;
        const Vec3f offset = Math::getTranslation(_transform);

        settings.color = _xColor;
        const Vec3f xDirection = Math::transformDirection(_transform, Math::AxisX);
        AddArrow(offset, offset + xDirection * _size, settings);

        settings.color = _yColor;
        const Vec3f yDirection = Math::transformDirection(_transform, Math::AxisY);
        AddArrow(offset, offset + yDirection * _size, settings);

        settings.color = _zColor;
        const Vec3f zDirection = Math::transformDirection(_transform, Math::AxisZ);
        AddArrow(offset, offset + zDirection * _size, settings);
    }

    void Im3dDrawList::AddBoundingBox(const BoundingBox& _bBox, const Im3dDrawSettings& _settings)
    {
        const Vec3f max = _bBox.GetMax();
        const Vec3f min = _bBox.GetMin();

        // Create all the 8 points:
        Vec3f points[8]{};
        points[0] = Vec3f(min.x, max.y, max.z);
        points[1] = Vec3f(min.x, max.y, min.z);
        points[2] = Vec3f(max.x, max.y, min.z);
        points[3] = Vec3f(max.x, max.y, max.z);
        points[4] = Vec3f(min.x, min.y, max.z);
        points[5] = Vec3f(min.x, min.y, min.z);
        points[6] = Vec3f(max.x, min.y, min.z);
        points[7] = Vec3f(max.x, min.y, max.z);

        // Front plane
        AddLineStrip(points, 4, true, _settings);

        // Back plane
        AddLineStrip(points + 4, 4, true, _settings);

        // Sides
        for (int i = 0; i < 4; ++i) {
            AddLine(points[i], points[i + 4], _settings);
        }
    }

    void Im3dDrawList::AddBoundingSphere(const BoundingSphere& _bSphere, const Im3dDrawSettings& _settings)
    {
        return AddSphere(_bSphere.GetOrigin(), _bSphere.GetRadius(), Math::Identity4, _settings);
    }

    void Im3dDrawList::AddPlane(const Plane& _plane, const Im3dDrawSettings& _settings)
    {
        return AddRectFilled(_plane.GetOrigin(), Vec2f(FLT_MAX), Math::Identity4, _settings);
    }

    void Im3dDrawList::AddFrustum(const Frustum& _frustum, const Im3dDrawSettings& _settings)
    {
        // Front plane
        AddLineStrip(_frustum.vertices, 4, true, _settings);

        // Back plane
        AddLineStrip(_frustum.vertices + 4, 4, true, _settings);

        // Sides
        for (int i = 0; i < 4; ++i) {
            AddLine(_frustum.vertices[i], _frustum.vertices[i + 4], _settings);
        }
    }


    //------ Im3dDrawList utilities -----------------------------------------------------------------------------------------

    void Im3dDrawList::Clear()
    {
        for (int i = 0; i < Im3dDepthTest_Count; i++)
        {
            m_lineVtxBuffer[i].clear();
            m_triangleVtxBuffer[i].clear();
        }
    }

    void Im3dDrawList::ClearMemory()
    {
        for (int i = 0; i < Im3dDepthTest_Count; i++)
        {
            m_lineVtxBuffer[i].clear();
            m_lineVtxBuffer[i].shrink_to_fit();

            m_triangleVtxBuffer[i].clear();
            m_triangleVtxBuffer[i].shrink_to_fit();
        }
    }

    u32 Im3dDrawList::GetTotalLineVertexCount() const
    {
        u32 count = 0;
        for (u8 i = 0; i < Im3dDepthTest_Count; i++)
        {
            count += (u32)m_lineVtxBuffer[i].size();
        }
        return count;
    }

    u32 Im3dDrawList::GetTotalTriangleVertexCount() const
    {
        u32 count = 0;
        for (u8 i = 0; i < Im3dDepthTest_Count; i++)
        {
            count += (u32)m_triangleVtxBuffer[i].size();
        }
        return count;
    }


    //------ Im3d implementation -------------------------------------------------------------------------------------------

    void Im3d::CreateContext()
    {
        s_drawList = new Im3dDrawList();
        s_drawData = new Im3dDrawData();

        s_colorStack[0] = IM3D_COL_WHITE;
        s_currentColor = s_colorStack[0];
        s_currentColorIndex = 0u;

        s_depthTestStack[0] = Im3dDepthTest_TransparentWhenBehind;
        s_currentDepthTest = s_depthTestStack[0];
        s_currentDepthTestIndex = 0u;
    }

    void Im3d::DestroyContext()
    {
        delete s_drawList;
    }

    void Im3d::NewFrame()
    {
        s_drawList->Clear();
        s_drawData->valid = false;
    }

    void Im3d::Render()
    {
        s_drawData->cmdLists = &s_drawList;
        s_drawData->totalLineVtxCount = s_drawList->GetTotalLineVertexCount();
        s_drawData->totalTriangleVtxCount = s_drawList->GetTotalTriangleVertexCount();
        s_drawData->cmdListsCount = s_drawData->totalLineVtxCount + s_drawData->totalTriangleVtxCount > 0 ? 1 : 0;
        s_drawData->valid = true;
    }

    Im3dDrawData* Im3d::GetDrawData()
    {
        return s_drawData;
    }

    void Im3d::PushColor(const Im3dColType _color)
    {
        AVA_ASSERT(s_currentColorIndex < kColorStackSize - 1u);
        s_currentColorIndex++;
        s_colorStack[s_currentColorIndex] = _color;
        s_currentColor = s_colorStack[s_currentColorIndex];
    }

    void Im3d::PopColor()
    {
        AVA_ASSERT(s_currentColorIndex > 0);
        s_currentColorIndex--;
        s_currentColor = s_colorStack[s_currentColorIndex];
    }

    void Im3d::PushDepthTest(const Im3dDepthTest _depthTest)
    {
        AVA_ASSERT(s_currentDepthTestIndex < kDepthTestStackSize - 1u);
        s_currentDepthTestIndex++;
        s_depthTestStack[s_currentDepthTestIndex] = _depthTest;
        s_currentDepthTest = s_depthTestStack[s_currentDepthTestIndex];
    }

    void Im3d::PopDepthTest()
    {
        AVA_ASSERT(s_currentDepthTestIndex > 0);
        s_currentDepthTestIndex--;
        s_currentDepthTest = s_depthTestStack[s_currentDepthTestIndex];
    }

    void Im3d::AddLine(const Vec3f& _from, const Vec3f& _to, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;
        settings.depthTest = s_currentDepthTest;

        s_drawList->AddLine(_from, _to, settings);
    }

    void Im3d::AddLines(const Vec3f* _points, const u32 _count, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;
        settings.depthTest = s_currentDepthTest;

        s_drawList->AddLines(_points, _count, settings);
    }

    void Im3d::AddLines(const Vec3f* _points, const Im3dColType* _colors, const u32 _count, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;

        s_drawList->AddLines(_points, _colors, _count, settings);
    }

    void Im3d::AddLineStrip(const Vec3f* _points, const u32 _count, const bool _closed, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;
        settings.depthTest = s_currentDepthTest;

        s_drawList->AddLineStrip(_points, _count, _closed, settings);
    }

    void Im3d::AddLineStrip(const Vec3f* _points, const Im3dColType* _colors, const u32 _count, const bool _closed, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;

        s_drawList->AddLineStrip(_points, _colors, _count, _closed, settings);
    }

    void Im3d::AddTriangle(const Vec3f _points[3], const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddTriangle(_points, settings);
    }

    void Im3d::AddTriangleFilled(const Vec3f _points[3], const Im3dColType _color/*= IM3D_COL_CURRENT*/)
    {
        Im3dDrawSettings settings;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddTriangleFilled(_points, settings);
    }

    void Im3d::AddRect(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddRect(_center, _size, _front, _up, settings);
    }

    void Im3d::AddRect(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddRect(_center, _size, _transform, settings);
    }

    void Im3d::AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dColType _color)
    {
        Im3dDrawSettings settings;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddRectFilled(_center, _size, _front, _up, settings);
    }

    void Im3d::AddRectFilled(const Vec3f& _center, const Vec2f& _size, const Mat4& _transform, const Im3dColType _color/*= IM3D_COL_CURRENT*/)
    {
        Im3dDrawSettings settings;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddRectFilled(_center, _size, _transform, settings);
    }

    void Im3d::AddBox(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddBox(_center, _size, _front, _up, settings);
    }

    void Im3d::AddBox(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddBox(_center, _size, _transform, settings);
    }

    void Im3d::AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Vec3f& _front, const Vec3f& _up, const Im3dColType _color/*= IM3D_COL_CURRENT*/)
    {
        Im3dDrawSettings settings;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddBoxFilled(_center, _size, _front, _up, settings);
    }

    void Im3d::AddBoxFilled(const Vec3f& _center, const Vec3f& _size, const Mat4& _transform, const Im3dColType _color/*= IM3D_COL_CURRENT*/)
    {
        Im3dDrawSettings settings;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddBoxFilled(_center, _size, _transform, settings);
    }

    void Im3d::AddCircle(const Vec3f& _center, const float _radius, const Vec3f& _front, const Vec3f& _up, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddCircle(_center, _radius, _front, _up, settings);
    }

    void Im3d::AddCircle(const Vec3f& _center, const float _radius, const Mat4& _transform, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddCircle(_center, _radius, _transform, settings);
    }

    void Im3d::AddSphere(const Vec3f& _center, const float _radius, const Vec3f& _front, const Vec3f& _up, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddSphere(_center, _radius, _front, _up, settings);
    }

    void Im3d::AddSphere(const Vec3f& _center, const float _radius, const Mat4& _transform, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddSphere(_center, _radius, _transform, settings);
    }

    void Im3d::AddCapsule(const Vec3f& _center, const float _radius, const float _height, const Vec3f& _front, const Vec3f& _up, const Im3dColType/*= IM3D_COL_CURRENT*/ _color, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddCapsule(_center, _radius, _height, _front, _up, settings);
    }

    void Im3d::AddCapsule(const Vec3f& _center, const float _radius, const float _height, const Mat4& _transform, const Im3dColType/*= IM3D_COL_CURRENT*/ _color, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddCapsule(_center, _radius, _height, _transform, settings);
    }

    void Im3d::AddArrow(const Vec3f& _from, const Vec3f& _to, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/, const float _headBaseSize/*= 0.f*/, const float _headBaseLength/*= 0.f*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddArrow(_from, _to, settings, _headBaseSize, _headBaseLength);
    }

    void Im3d::AddGuizmo(const Mat4& _transform, const float _size, const float _thickness/*= kDefaultThickness*/, const Im3dColType _xColor/*= IM3D_COL_RED*/, const Im3dColType _yColor/*= IM3D_COL_GREEN*/, const Im3dColType _zColor/*= IM3D_COL_BLUE*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;

        s_drawList->AddGuizmo(_transform, _size, settings, _xColor, _yColor, _zColor);
    }

    void Im3d::AddBoundingBox(const BoundingBox& _bBox, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddBoundingBox(_bBox, settings);
    }

    void Im3d::AddBoundingSphere(const BoundingSphere& _bSphere, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddBoundingSphere(_bSphere, settings);
    }

    void Im3d::AddPlane(const Plane& _plane, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddPlane(_plane, settings);
    }

    void Im3d::AddFrustum(const Frustum& _frustum, const Im3dColType _color/*= IM3D_COL_CURRENT*/, const float _thickness/*= kDefaultThickness*/)
    {
        Im3dDrawSettings settings;
        settings.thickness = _thickness;
        settings.depthTest = s_currentDepthTest;
        settings.color = _color == IM3D_COL_CURRENT ? s_currentColor : _color;

        s_drawList->AddFrustum(_frustum, settings);
    }
}
