#pragma once
/// @file Math.h
/// @brief File defining useful math methods.

#include <Math/Types.h>
#include <Debug/Assert.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/matrix_decompose.hpp>

AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    namespace Math {

        //---------- Basic functions -----------------------------------------------------------------------------------------------------

        template <typename T>
        T square(const T& _t) { return _t * _t; }

        template <typename T>
        T cube(const T& _t) { return _t * _t * _t; }

        template <typename T>
        T abs(const T& _t) { return glm::abs(_t); }

        template <typename T>
        T floor(const T& _t) { return glm::floor(_t); }

        template <typename T>
        T trunc(const T& _t) { return glm::trunc(_t); }

        template <typename T>
        T ceil(const T& _t) { return glm::ceil(_t); }

        //---------- Basic conversions ---------------------------------------------------------------------------------------------------

        template <typename T>
        T radians(const T& _angle) { return glm::radians(_angle); }

        template <typename T>
        T degrees(const T& _angle) { return glm::degrees(_angle); }

        template <typename T>
        T ndcToUv(const T& _ndc) { return T(0.5f) * _ndc + T(0.5f); }

        template <typename T>
        T uvToNdc(const T& _uv) { return T(2.f) * _uv - T(1.f); }

        //---------- Basic comparisons ---------------------------------------------------------------------------------------------------

        template <typename T>
        T min(const T& _a, const T& _b) { return std::min(_a, _b); }

        template <typename T>
        T max(const T& _a, const T& _b) { return std::max(_a, _b); }

        template <typename T>
        T minPerElement(const T& _a, const T& _b) { return glm::min(_a, _b); }

        template <typename T>
        T maxPerElement(const T& _a, const T& _b) { return glm::max(_a, _b); }

        inline float min3(const float _a, const float _b, const float _c) { return min(min(_a, _b), _c); }
        inline float min3(const Vec3f& _vector) { return min3(_vector.x, _vector.y, _vector.z); }

        inline float max3(const float _a, const float _b, const float _c) { return max(max(_a, _b), _c); }
        inline float max3(const Vec3f& _vector) { return max3(_vector.x, _vector.y, _vector.z); }

        //---------- Basic thresholds ----------------------------------------------------------------------------------------------------

        template <typename T>
        T clamp(const T& _t, const T& _lower, const T& _higher) { return glm::clamp(_t, _lower, _higher); }

        template <typename T>
        T saturate(const T& _t) { return clamp(_t, T(0.f), T(1.f)); }

        template <typename T>
        T lerp(const T& _lower, const T& _higher, const float _t) { return glm::mix(_lower, _higher, _t); }

        template <typename T>
        T step(const T& _threshold, const T& _t) { return glm::step(_threshold, _t); }

        template <typename T>
        T smoothstep(const T& _lower, const T& _higher, const T& _t) { return glm::smoothstep(_lower, _higher, _t); }

        //---------- Vector operations ---------------------------------------------------------------------------------------------------

        template <typename T>
        float length(const T& _vec) { return glm::length(_vec); }

        template <typename T>
        float distance(const T& _u, const T& _v) { return glm::distance(_u, _v); }

        template <typename T>
        float dot(const T& _u, const T& _v) { return glm::dot(_u, _v); }

        template <typename T>
        float normSquared(const T& _vec) { return glm::dot(_vec, _vec); }

        template <typename T>
        T cross(const T& _u, const T& _v) { return glm::cross(_u, _v); }

        template <typename T>
        T normalize(const T& _vec) { return glm::normalize(_vec); }

        template <typename T>
        T reflect(const T& _ray, const T& _normal) { return glm::reflect(_ray, _normal); }

        template <typename T>
        T refract(const T& _ray, const T& _normal, const float _ior) { return glm::refract(_ray, _normal, _ior); }

        //---------- Quaternion / euler angles / matrix conversion -----------------------------------------------------------------------

        inline Quat toQuaternion(const Vec3f& _eulerAngles) { return glm::quat(_eulerAngles); }
        inline Vec3f toEulerAngles(const Quat& _quat) { return glm::eulerAngles(_quat); }
        inline Mat3 toMat3(const Quat& _quat) { return glm::toMat3(_quat); }
        inline Mat4 toMat4(const Quat& _quat) { return glm::toMat4(_quat); }

        //---------- Basic matrix operations ---------------------------------------------------------------------------------------------

        inline float determinant(const Mat4& _matrix) { return glm::determinant(_matrix); }
        inline Mat4 inverse(const Mat4& _matrix) { return glm::inverse(_matrix); }
        inline Mat4 transpose(const Mat4& _matrix) { return glm::transpose(_matrix); }

        inline Vec3f column(const Mat3& _matrix, const int _index) { return glm::column(_matrix, _index); }
        inline Vec4f column(const Mat4& _matrix, const int _index) { return glm::column(_matrix, _index); }
        inline Vec3f row(const Mat3& _matrix, const int _index) { return glm::row(_matrix, _index); }
        inline Vec4f row(const Mat4& _matrix, const int _index) { return glm::row(_matrix, _index); }

        inline Vec3f transformPosition(const Mat4& _matrix, const Vec3f& _position) { return Vec3f(_matrix * Vec4f(_position, 1.f)); }
        inline Vec3f transformDirection(const Mat4& _matrix, const Vec3f& _direction) { return normalize(Mat3(_matrix) * _direction); }

        //---------- Translation matrix operations ---------------------------------------------------------------------------------------

        inline void translate(Mat4& _mat, const Vec3f& _translateVector)
        {
            _mat = glm::translate(_mat, _translateVector);
        }

        inline Mat4 buildTranslationMatrix(const Vec3f& _translateVector)
        {
            Mat4 mat = Identity4;
            translate(mat, _translateVector);
            return mat;
        }

        inline Vec3f getTranslation(const Mat4& _transform)
        {
            // Translate vector is stored in the last column.
            const Vec3f translation = column(_transform, 3);
            return translation;
        }

        //---------- Scale matrix operations ---------------------------------------------------------------------------------------------

        inline void scale(Mat4& _mat, const Vec3f& _scaleFactors)
        {
            _mat = glm::scale(_mat, _scaleFactors);
        }

        inline Mat4 buildScaleMatrix(const Vec3f& _scaleFactors)
        {
            Mat4 mat = Identity4;
            scale(mat, _scaleFactors);
            return mat;
        }

        inline Vec3f getScale(const Mat4& _transform)
        {
            // Scale factors are the length of the three first columns.
            Vec3f scale = UnitVector;
            scale.x = length(column(_transform, 0));
            scale.y = length(column(_transform, 1));
            scale.z = length(column(_transform, 2));
            return scale;
        }

        //---------- Rotation matrix operations ------------------------------------------------------------------------------------------

        inline void rotate(Mat4& _mat, const Vec3f& _axis, const float _angle)
        {
            _mat = glm::rotate(_mat, _angle, _axis);
        }

        inline Mat4 buildRotationMatrix(const Vec3f& _eulerAngles)
        {
            Mat4 mat = Mat3(1.f);
            rotate(mat, AxisZ, _eulerAngles.z);
            rotate(mat, AxisY, _eulerAngles.y);
            rotate(mat, AxisX, _eulerAngles.x);
            return mat;
        }

        inline Mat4 getRotation(const Mat4& _transform)
        {
            // Rotation matrix is obtained by zero the translation and divide by the inverse of the scale.
            Mat4 rotationMatrix = _transform;
            const Vec3f scale = getScale(_transform);

            rotationMatrix[0] /= scale.x;
            rotationMatrix[1] /= scale.y;
            rotationMatrix[2] /= scale.z;
            rotationMatrix[3] = { 0, 0, 0, 1 };

            return rotationMatrix;
        }

        //---------- Transform matrix composition ----------------------------------------------------------------------------------------

        /// @brief  Compute the matrix to perform a "look at" transformation.
        /// @param _from the position of the viewer
        /// @param _to the point the viewer is looking at
        /// @param _up the world up direction (Y axis by default)
        /// @return a matrix representation of the defined "look at" transformation.
        inline Mat4 lookAt(const Vec3f& _from, const Vec3f& _to, const Vec3f& _up = AxisY)
        {
            const Vec3f front = normalize(_to - _from);
            Vec3f worldUp = _up;

            // The given up vector is aligned with the front vector, which will result in NaN.
            // To workaround this, we rotate the up vector in the perpendicular plane.
            if (abs(dot(front, worldUp)) > 1.f -  FLT_EPSILON)
            {
                worldUp = AxisX;
            }

            const Vec3f right = normalize(cross(front, worldUp));
            AVA_ASSERT(!isnan(right.x) && !isnan(right.y) && !isnan(right.z));

            const Vec3f up = cross(right, front);
            AVA_ASSERT(!isnan(up.x) && !isnan(up.y) && !isnan(up.z));

            // glm stores matrix in column-major order
            const Mat4 lookAtMatrix =
            {
                right.x           , up.x           , -front.x         , 0.0f,

                right.y           , up.y           , -front.y         , 0.0f,

                right.z           , up.z           , -front.z         , 0.0f,

                -dot(right, _from), -dot(up, _from), dot(front, _from), 1.0f
            };
            return lookAtMatrix;
        }

        /// @brief Compute the matrix to perform a "rotate around" transformation.
        /// @param _axisOrigin the origin of the axis around which the rotation is performed
        /// @param _axisDirection the direction of the axis around which the rotation is performed
        /// @param _angle the rotation angle, expressed in radians
        /// @return a matrix representation of the defined "rotate around" transformation.
        inline Mat4 rotateAround(const Vec3f& _axisOrigin, const Vec3f& _axisDirection, const float _angle)
        {
            Mat4 transform = Identity4;
            translate(transform, _axisOrigin);
            rotate(transform, _axisDirection, _angle);
            translate(transform, -_axisOrigin);
            return transform;
        }

        /// @brief Retrieve from a transform matrix its translation, rotation and scaling factors.
        /// @param _transform the transform matrix to decompose
        /// @param _translation the translate vector resulting from the decomposition
        /// @param _rotation the euler angles resulting from the decomposition
        /// @param _scale the scale factors resulting from the decomposition
        /// @return a boolean indicating whether the decomposition was possible or not
        inline bool decomposeTransform(const Mat4& _transform, Vec3f& _translation, Vec3f& _rotation, Vec3f& _scale)
        {
            Mat4 localMatrix = _transform;
            using namespace glm;

            // Transform matrix should not be null.
            if (epsilonEqual(localMatrix[3][3], 0.f, FLT_EPSILON))
            {
                return false;
            }

            // Clears the perspective partition.
            if (
                epsilonNotEqual(localMatrix[0][3], 0.f, FLT_EPSILON) ||
                epsilonNotEqual(localMatrix[1][3], 0.f, FLT_EPSILON) ||
                epsilonNotEqual(localMatrix[2][3], 0.f, FLT_EPSILON))
            {
                localMatrix[0][3] = localMatrix[1][3] = localMatrix[2][3] = 0.f;
                localMatrix[3][3] = 1.f;
            }

            // Extracts translation, then removes it.
            _translation = getTranslation(_transform);
            localMatrix[3] = { 0, 0, 0, 1 };

            // Extracts scale, then removes it.
            _scale = getScale(_transform);
            localMatrix[0] /= _scale.x;
            localMatrix[1] /= _scale.y;
            localMatrix[2] /= _scale.z;

            // Extracts euler angles
            _rotation.x = atan2f(localMatrix[1][2], localMatrix[2][2]);
            _rotation.y = atan2f(-localMatrix[0][2], sqrtf(localMatrix[1][2] * localMatrix[1][2] + localMatrix[2][2] * localMatrix[2][2]));
            _rotation.z = atan2f(localMatrix[0][1], localMatrix[0][0]);

            return true;
        }

        /// @brief Build a transform matrix from given translation, rotation and scaling factors.
        /// @param _translation the translate vector to apply to the transform
        /// @param _rotation the euler angles to apply to the transform
        /// @param _scale the scale factors apply to the transform
        /// @return the matrix representation of the given affine transformations
        inline Mat4 recomposeTransform(const Vec3f& _translation, const Vec3f& _rotation, const Vec3f& _scale)
        {
            Mat4 transform = Identity4;
            Math::translate(transform, _translation);
            rotate(transform, AxisZ, _rotation.z);
            rotate(transform, AxisY, _rotation.y);
            rotate(transform, AxisX, _rotation.x);
            scale(transform, _scale);
            return transform;
        }

        //---------- Projection matrix constructors --------------------------------------------------------------------------------------

        /// @brief Compute the orthographic projection matrix of a frustum.
        /// @param _left the position of the frustum's left plane
        /// @param _right the position of the frustum's right plane
        /// @param _down the position of the frustum's bottom plane
        /// @param _top the position of the frustum's top plane
        /// @param _near the position of the frustum's near plane
        /// @param _far the position of the frustum's far plane
        /// @return the orthographic matrix adapted to the given frustum parameters
        inline Mat4 ortho(const float _left, const float _right, const float _down, const float _top, const float _near, const float _far)
        {
            // glm stores matrix in column-major order
            const Mat4 projectionMatrix =
            {
                2.0f / (_right - _left)             , 0.0f                            , 0.0f                  , 0.0f,

                0.0f                                , 2.0f / (_down - _top)           , 0.0f                  , 0.0f,

                0.0f                                , 0.0f                            , 1.0f / (_near - _far) , 0.0f,

                -(_right + _left) / (_right - _left), -(_down + _top) / (_down - _top), _near / (_near - _far), 1.0f
            };
            return projectionMatrix;
        }

        /// @brief Compute the reversed orthographic projection matrix of a frustum.
        /// @param _left the position of the frustum's left plane
        /// @param _right the position of the frustum's right plane
        /// @param _down the position of the frustum's bottom plane
        /// @param _top the position of the frustum's top plane
        /// @param _near the position of the frustum's near plane
        /// @param _far the position of the frustum's far plane
        /// @return the reversed orthographic matrix adapted to the given frustum parameters
        inline Mat4 orthoReversed(const float _left, const float _right, const float _down, const float _top, const float _near, const float _far)
        {
            // glm stores matrix in column-major order
            const Mat4 projectionMatrix =
            {
                2.0f / (_right - _left)             , 0.0f                            , 0.0f                  , 0.0f,

                0.0f                                , 2.0f / (_down - _top)           , 0.0f                  , 0.0f,

                0.0f                                , 0.0f                            , 1.0f / (_far - _near) , 0.0f,

                -(_right + _left) / (_right - _left), -(_down + _top) / (_down - _top), _far / (_far - _near), 1.0f
            };
            return projectionMatrix;
        }

        /// @brief  Compute the perspective projection matrix of a frustum.
        /// @param _fov the vertical field of view of the frustum, in radians
        /// @param _aspect the aspect ratio of the frustum
        /// @param _near the position of the frustum's near plane
        /// @param _far the position of the frustum's far plane
        /// @return the perspective matrix adapted to the given frustum parameters
        inline Mat4 perspective(const float _fov, const float _aspect, const float _near, const float _far)
        {
            const float focalLength = 1.f / tan(0.5f * _fov);

            // glm stores matrix in column-major order
            const Mat4 projectionMatrix =
            {
                focalLength / _aspect, 0.0f       , 0.0f                         , 0.0f,

                0.0f                 ,-focalLength, 0.0f                         , 0.0f,

                0.0f                 , 0.0f       , _far / (_near - _far)        ,-1.0f,

                0.0f                 , 0.0f       , _far * _near / (_near - _far), 0.0f
            };
            return projectionMatrix;
        }

        /// @brief  Compute the infinite perspective projection matrix of a frustum.
        /// @param _fov the vertical field of view of the frustum, in radians
        /// @param _aspect the aspect ratio of the frustum
        /// @param _near the position of the frustum's near plane
        /// @return the infinite perspective matrix adapted to the given frustum parameters
        inline Mat4 perspectiveInfinite(const float _fov, const float _aspect, const float _near)
        {
            const float focalLength = 1.f / tan(0.5f * _fov);

            // glm stores matrix in column-major order
            const Mat4 projectionMatrix =
            {
                focalLength / _aspect, 0.0f       , 0.0f ,  0.0f,

                0.0f                 ,-focalLength, 0.0f ,  0.0f,

                0.0f                 , 0.0f       ,-1.0f , -1.0f,

                0.0f                 , 0.0f       ,-_near,  0.0f
            };
            return projectionMatrix;
        }

        /// @brief  Compute the reversed perspective projection matrix of a frustum.
        /// @param _fov the vertical field of view of the frustum, in radians
        /// @param _aspect the aspect ratio of the frustum
        /// @param _near the position of the frustum's near plane
        /// @param _far the position of the frustum's far plane
        /// @return the reversed perspective matrix adapted to the given frustum parameters
        inline Mat4 perspectiveReversed(const float _fov, const float _aspect, const float _near, const float _far)
        {
            const float focalLength = 1.f / tan(0.5f * _fov);

            // glm stores matrix in column-major order
            const Mat4 projectionMatrix =
            {
                focalLength / _aspect, 0.0f       , 0.0f                         , 0.0f,

                0.0f                 ,-focalLength, 0.0f                         , 0.0f,

                0.0f                 , 0.0f       , _near / (_far - _near)       ,-1.0f,

                0.0f                 , 0.0f       , _near * _far / (_far - _near), 0.0f
            };
            return projectionMatrix;
        }

        /// @brief  Compute the infinite reversed perspective projection matrix of a frustum.
        /// @param _fov the vertical field of view of the frustum, in radians
        /// @param _aspect the aspect ratio of the frustum
        /// @param _near the position of the frustum's near plane
        /// @return the infinite reversed perspective matrix adapted to the given frustum parameters
        inline Mat4 perspectiveInfiniteReversed(const float _fov, const float _aspect, const float _near)
        {
            const float focalLength = 1.f / tan(0.5f * _fov);

            // glm stores matrix in column-major order
            const Mat4 projectionMatrix =
            {
                focalLength / _aspect, 0.0f       , 0.0f , 0.0f,

                0.0f                 ,-focalLength, 0.0f , 0.0f,

                0.0f                 , 0.0f       , 0.0f ,-1.0f,

                0.0f                 , 0.0f       , _near, 0.0f
            };
            return projectionMatrix;
        }

    }

}
