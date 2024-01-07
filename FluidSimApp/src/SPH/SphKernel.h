#pragma once

#include <Math/Math.h>
using namespace Ava;

namespace sph
{
    /// @brief Cubic Spline Smooth Kernel.
    class SphKernel
    {
    public:
        SphKernel(float _h = 1, u32 _dim = 2);

        float GetSmoothingLen() const { return m_h; }
        u32 GetDimension() const { return m_dim; }

        template<typename VecType>
        float W(const VecType& _rij) const 
        {
            const float len = Math::length(_rij);
            return f(len); 
        }

        template<typename VecType>
        VecType GradW(const VecType& _rij) const
        {
            const float len = Math::length(_rij);
            return derivativeF(len) * _rij / len;
        }

    private:
        float m_h;
        u32 m_dim;

        float m_coeff[3];
        float m_derivCoeff[3];

        float f(float _l) const;
        float derivativeF(float _l) const;
    };
}