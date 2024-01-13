#include "SphKernel.h"

namespace sph
{
    SphKernel::SphKernel(float _h, u32 _dim)
    {
        m_h = _h;
        m_dim = _dim;

        m_coeff[0] = 2.f / (3.f * _h);
        m_coeff[1] = 10.f / (7.f * Math::PI * Math::square(_h));
        m_coeff[2] = 1.f / (Math::PI * Math::cube(_h));

        m_derivCoeff[0] = m_coeff[0] / _h;
        m_derivCoeff[1] = m_coeff[1] / _h;
        m_derivCoeff[2] = m_coeff[2] / _h;
    }

    float SphKernel::f(float _length) const
    {
        const float q = _length / m_h;

        if (q <= 1.f)
        {
            return GetCoeff() * (1.f - 1.5f * Math::square(q) + 0.75f * Math::cube(q));
        }

        if (q <= 2.f)
        {
            return GetCoeff() * 0.25f * Math::cube(2.f - q);
        }

        return 0.f;
    }

    float SphKernel::derivativeF(float _length) const
    {
        const float q = _length / m_h;

        if (q <= 1.f) 
        {
            return GetDerivCoeff() * (-3.f * q + 2.25f * Math::square(q));
        }

        if (q <= 2.f) 
        {
            return -GetDerivCoeff() * 0.75f * Math::square(2.f - q);
        }

        return 0.f;
    }

}
