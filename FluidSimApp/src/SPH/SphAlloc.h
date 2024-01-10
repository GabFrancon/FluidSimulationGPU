#pragma once

#include <cuda_runtime.h>

namespace sph
{
    using u32 = uint32_t;

    __host__ __device__
    inline long get_idx(long v, long w, long x, long y, long z, long stride_w, long stride_x, long stride_y, long stride_z)
    {
        return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
    }

    __host__ __device__
    inline long get_idx(long w, long x, long y, long z, long stride_x, long stride_y, long stride_z)
    {
        return stride_x * stride_y * stride_z * w + stride_y * stride_z * x + stride_z * y + z;
    }

    __host__ __device__
    inline long get_idx(long x, long y, long z, long stride_y, long stride_z)
    {
        return stride_y * stride_z * x + stride_z * y + z;
    }

    __host__ __device__
    inline long get_idx(long x, long y, long s1)
    {
        return x + (y * s1);
    }


    template < class type >
    inline type *newArr1(size_t sz1)
    {
      type *arr = new type[sz1];
      return arr;
    }

    template < class type >
    inline type **newArr2(size_t sz1, size_t sz2)
    {
      type **arr = new type*[sz1]; // new type *[sz1];
      type *ptr = newArr1<type>(sz1*sz2);
      for (size_t i = 0; i < sz1; i++)
      {
        arr[i] = ptr;
        ptr += sz2;
      }
      return arr;
    }

    template < class type >
    inline type ***newArr3(size_t sz1, size_t sz2, size_t sz3)
    {
      type ***arr = new type**[sz1]; // new type **[sz1];
      type **ptr = newArr2<type>(sz1*sz2, sz3);
      for (size_t i = 0; i < sz1; i++)
      {
        arr[i] = ptr;
        ptr += sz2;
      }
      return arr;
    }

    template <class type>
    inline type ****newArr4(size_t sz1, size_t sz2, size_t sz3, size_t sz4)
    {
      type ****arr = new type***[sz1]; //(new type ***[sz1]);
      type ***ptr = newArr3<type>(sz1*sz2, sz3, sz4);
      for (size_t i = 0; i < sz1; i++) {
        arr[i] = ptr;
        ptr += sz2;
      }
      return arr;
    }

    // build chained pointer hierarchy for pre-existing bottom level
    //

    /* Build chained pointer hierachy for pre-existing bottom level                        *
     * Provide a pointer to a contig. 1D memory region which was already allocated in "in" *
     * The function returns a pointer chain to which allows subscript access (x[i][j])     */
    template <class type>
    inline type *****newArr5(type **in, size_t sz1, size_t sz2, size_t sz3, size_t sz4, size_t sz5)
    {
      *in = newArr1<type>(sz1*sz2*sz3*sz4*sz5);

      type*****arr = newArr4<type*>(sz1,sz2,sz3,sz4);
      type**arr2 = ***arr;
      type *ptr = *in;
      size_t szarr2 = sz1*sz2*sz3*sz4;
      for(size_t i=0;i<szarr2;i++) {
        arr2[i] = ptr;
        ptr += sz5;
      }
      return arr;
    }

    template <class type>
    inline type ****newArr4(type **in, size_t sz1, size_t sz2, size_t sz3, size_t sz4)
    {
      *in = newArr1<type>(sz1*sz2*sz3*sz4);

      type****arr = newArr3<type*>(sz1,sz2,sz3);
      type**arr2 = **arr;
      type *ptr = *in;
      size_t szarr2 = sz1*sz2*sz3;
      for(size_t i=0;i<szarr2;i++) {
        arr2[i] = ptr;
        ptr += sz4;
      }
      return arr;
    }

    template <class type>
    inline type ***newArr3(type **in, size_t sz1, size_t sz2, size_t sz3)
    {
      *in = newArr1<type>(sz1*sz2*sz3);

      type***arr = newArr2<type*>(sz1,sz2);
      type**arr2 = *arr;
      type *ptr = *in;
      size_t szarr2 = sz1*sz2;
      for(size_t i=0;i<szarr2;i++) {
        arr2[i] = ptr;
        ptr += sz3;
      }
      return arr;
    }

    template <class type>
    inline type **newArr2(type **in, size_t sz1, size_t sz2)
    {
      *in = newArr1<type>(sz1*sz2);
      type**arr = newArr1<type*>(sz1);
      type *ptr = *in;
      for(size_t i=0;i<sz1;i++) {
        arr[i] = ptr;
        ptr += sz2;
      }
      return arr;
    }

    // methods to deallocate arrays
    //
    template < class type > inline void delArray1(type * arr)
    { delete[](arr); }
    template < class type > inline void delArray2(type ** arr)
    { delArray1(arr[0]); delete[](arr); }
    template < class type > inline void delArray3(type *** arr)
    { delArray2(arr[0]); delete[](arr); }
    template < class type > inline void delArray4(type **** arr)
    { delArray3(arr[0]); delete[](arr); }

    constexpr int kMaxFluidInCell = 10;
    constexpr int kMaxBoundaryInCell = 4;
    constexpr int kMaxFluidNeighbors = 20;
    constexpr int kMaxBoundaryNeighbors = 10;
    constexpr int kMaxPressureSolveIteration = 4;
    constexpr u32 kInvalidIdx = 0xFFFFFFFF;
}