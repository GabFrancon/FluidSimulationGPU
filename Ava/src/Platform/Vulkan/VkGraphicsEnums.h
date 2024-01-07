#pragma once
/// @file VkGraphicsEnums.h
/// @brief file extending GraphicEnums.h to Vulkan.

#include <Graphics/GraphicsEnums.h>

//--- Include third-parties ------------------
AVA_DISABLE_WARNINGS_BEGIN
#include <vulkan/vulkan.h>
AVA_DISABLE_WARNINGS_END
//--------------------------------------------

namespace Ava {

    static VkImageViewType ToVk(const TextureType::Enum _type)
    {
        static constexpr VkImageViewType s_vkTypes[TextureType::Count] = {
            VK_IMAGE_VIEW_TYPE_1D,         // Texture1D
            VK_IMAGE_VIEW_TYPE_2D,         // Texture2D
            VK_IMAGE_VIEW_TYPE_3D,         // Texture3D
            VK_IMAGE_VIEW_TYPE_1D_ARRAY,   // Texture1DArray
            VK_IMAGE_VIEW_TYPE_2D_ARRAY,   // Texture2DArray
            VK_IMAGE_VIEW_TYPE_CUBE        // TextureCube
        };
        return s_vkTypes[_type < TextureType::Count ? _type : 0];
    }

    static VkFormat ToVk(const TextureFormat::Enum _format)
    {
        static constexpr VkFormat s_vkFormats[TextureFormat::Count] = {
            VK_FORMAT_UNDEFINED,            // Undefined

            VK_FORMAT_R8_UNORM,             // R8
            VK_FORMAT_R8G8_UNORM,           // RG8
            VK_FORMAT_R8G8B8_UNORM,         // RGB8
            VK_FORMAT_R8G8B8A8_UNORM,       // RGBA8

            VK_FORMAT_R8_SINT,              // R8_SINT
            VK_FORMAT_R8G8_SINT,            // RG8_SINT
            VK_FORMAT_R8G8B8_SINT,          // RGB8_SINT
            VK_FORMAT_R8G8B8A8_SINT,        // RGBA8_SINT

            VK_FORMAT_R8_UINT,              // R8_UINT
            VK_FORMAT_R8G8_UINT,            // RG8_UINT
            VK_FORMAT_R8G8B8_UINT,          // RGB8_UINT
            VK_FORMAT_R8G8B8A8_UINT,        // RGBA8_UINT

            VK_FORMAT_R16_UNORM,            // R16
            VK_FORMAT_R16G16_UNORM,         // RG16
            VK_FORMAT_R16G16B16_UNORM,      // RGB16
            VK_FORMAT_R16G16B16A16_UNORM,   // RGBA16

            VK_FORMAT_R16_SINT,             // R16_SINT
            VK_FORMAT_R16G16_SINT,          // RG16_SINT
            VK_FORMAT_R16G16B16_SINT,       // RGB16_SINT
            VK_FORMAT_R16G16B16A16_SINT,    // RGBA16_SINT

            VK_FORMAT_R16_UINT,             // R16_UINT
            VK_FORMAT_R16G16_UINT,          // RG16_UINT
            VK_FORMAT_R16G16B16_UINT,       // RGB16_UINT
            VK_FORMAT_R16G16B16A16_UINT,    // RGBA16_UINT

            VK_FORMAT_R16_SFLOAT,           // R16F
            VK_FORMAT_R16G16_SFLOAT,        // RG16F
            VK_FORMAT_R16G16B16_SFLOAT,     // RGB16F
            VK_FORMAT_R16G16B16A16_SFLOAT,  // RGBA16F

            VK_FORMAT_R32_SFLOAT,           // R32F
            VK_FORMAT_R32G32_SFLOAT,        // RG32F
            VK_FORMAT_R32G32B32_SFLOAT,     // RGB32F
            VK_FORMAT_R32G32B32A32_SFLOAT,  // RGBA32F

            VK_FORMAT_BC1_RGBA_UNORM_BLOCK, // BC1
            VK_FORMAT_BC2_UNORM_BLOCK,      // BC2
            VK_FORMAT_BC3_UNORM_BLOCK,      // BC3
            VK_FORMAT_BC4_UNORM_BLOCK,      // BC4
            VK_FORMAT_BC5_UNORM_BLOCK,      // BC5
            VK_FORMAT_BC6H_SFLOAT_BLOCK,    // BC6F
            VK_FORMAT_BC7_UNORM_BLOCK,      // BC7

            VK_FORMAT_D16_UNORM,            // D16
            VK_FORMAT_D32_SFLOAT            // D32F
        };
        return s_vkFormats[_format < TextureFormat::Count ? _format : 0];
    }

    static VkFormat ToVk(const DataType::Enum _type, u8 _count)
    {
        // s_vkFormats begins at index 0.
        _count -= 1;

        // Sometimes treat 3 channels attributes as 4 channels for improved compatibilities.
        static constexpr VkFormat s_vkFormats[DataType::Count][4] = {
            //         R                       R G                      R G B                        R G B A              //
            {    VK_FORMAT_R8_SINT,     VK_FORMAT_R8G8_SINT,      VK_FORMAT_R8G8B8_SINT,       VK_FORMAT_R8G8B8A8_SINT }, // SINT8
            {   VK_FORMAT_R16_SINT,   VK_FORMAT_R16G16_SINT,   VK_FORMAT_R16G16B16_SINT,   VK_FORMAT_R16G16B16A16_SINT }, // SINT16
            {   VK_FORMAT_R32_SINT,   VK_FORMAT_R32G32_SINT,   VK_FORMAT_R32G32B32_SINT,   VK_FORMAT_R32G32B32A32_SINT }, // SINT32

            {    VK_FORMAT_R8_UINT,     VK_FORMAT_R8G8_UINT,    VK_FORMAT_R8G8B8A8_UINT,       VK_FORMAT_R8G8B8A8_UINT }, // UINT8
            {   VK_FORMAT_R16_UINT,   VK_FORMAT_R16G16_UINT,   VK_FORMAT_R16G16B16_UINT,   VK_FORMAT_R16G16B16A16_UINT }, // UINT16
            {   VK_FORMAT_R32_UINT,   VK_FORMAT_R32G32_UINT,   VK_FORMAT_R32G32B32_UINT,   VK_FORMAT_R32G32B32A32_UINT }, // UINT32

            {   VK_FORMAT_R8_SNORM,     VK_FORMAT_R8G8_SNORM,  VK_FORMAT_R8G8B8A8_SNORM,      VK_FORMAT_R8G8B8A8_SNORM }, // SNORM8
            {  VK_FORMAT_R16_SNORM,   VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16B16_SNORM,  VK_FORMAT_R16G16B16A16_SNORM }, // SNORM16

            {   VK_FORMAT_R8_UNORM,     VK_FORMAT_R8G8_UNORM,   VK_FORMAT_R8G8B8A8_UNORM,     VK_FORMAT_R8G8B8A8_UNORM }, // UNORM8
            {  VK_FORMAT_R16_UNORM,   VK_FORMAT_R16G16_UNORM,  VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16A16_UNORM }, // UNORM16

            { VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT }, // FLOAT16
            { VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT }, // FLOAT32
        };
        return s_vkFormats[_type < DataType::Count ? _type : 0][_count <= 3 ? _count : 0];
    }

    static VkShaderStageFlagBits ToVk(const ShaderStage::Enum _stage)
    {
        static constexpr VkShaderStageFlagBits s_vkTypes[ShaderStage::Count] = {
            VK_SHADER_STAGE_VERTEX_BIT,   // Vertex
            VK_SHADER_STAGE_GEOMETRY_BIT, // Geometric
            VK_SHADER_STAGE_FRAGMENT_BIT, // Fragment
            VK_SHADER_STAGE_COMPUTE_BIT   // Compute
        };
        return s_vkTypes[_stage < ShaderStage::Count ? _stage : 0];
    }

    static VkPrimitiveTopology ToVk(const PrimitiveType::Enum _type)
    {
        static constexpr VkPrimitiveTopology s_vkTypes[PrimitiveType::Count] = {
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,  // Triangle
            VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, // TriangleStrip
            VK_PRIMITIVE_TOPOLOGY_LINE_LIST,      // Line
            VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,     // LineStrip
            VK_PRIMITIVE_TOPOLOGY_POINT_LIST      // Point
        };
        return s_vkTypes[_type < PrimitiveType::Count ? _type : 0];
    }

    static VkCullModeFlagBits ToVk(const CullMode::Enum _mode)
    {
        static constexpr VkCullModeFlagBits s_vkModes[CullMode::Count] = {
            VK_CULL_MODE_NONE,           // None
            VK_CULL_MODE_FRONT_BIT,      // Front
            VK_CULL_MODE_BACK_BIT,       // Back
            VK_CULL_MODE_FRONT_AND_BACK  // FrontBack
        };
        return s_vkModes[_mode < CullMode::Count ? _mode : 0];
    }

    static VkCompareOp ToVk(const CompareFunc::Enum _func)
    {
        static constexpr VkCompareOp s_vkFunctions[CompareFunc::Count] = {
                VK_COMPARE_OP_NEVER,            // Never
                VK_COMPARE_OP_NOT_EQUAL,        // NotEqual
                VK_COMPARE_OP_EQUAL,            // Equal
                VK_COMPARE_OP_LESS,             // Less
                VK_COMPARE_OP_LESS_OR_EQUAL,    // LessEqual
                VK_COMPARE_OP_GREATER,          // Greater
                VK_COMPARE_OP_GREATER_OR_EQUAL, // GreaterEqual
                VK_COMPARE_OP_ALWAYS            // Always
        };
        return s_vkFunctions[_func < CompareFunc::Count ? _func : 0];
    }

    static VkBlendOp ToVk(const BlendFunc::Enum _func)
    {
        static constexpr VkBlendOp s_vkFunctions[BlendFunc::Count] = {
            VK_BLEND_OP_ADD,      // Add
            VK_BLEND_OP_SUBTRACT, // Subtract
            VK_BLEND_OP_MIN,      // Min
            VK_BLEND_OP_MAX,      // Max
        };
        return s_vkFunctions[_func < BlendFunc::Count ? _func : 0];
    }

    static VkBlendFactor ToVk(const BlendFactor::Enum _factor)
    {
        static constexpr VkBlendFactor s_vkFactors[BlendFactor::Count] = {
            VK_BLEND_FACTOR_ZERO,                // Zero
            VK_BLEND_FACTOR_ONE,                 // One

            VK_BLEND_FACTOR_SRC_COLOR,           // SrcColor
            VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, // OneMinusSrcColor
            VK_BLEND_FACTOR_DST_COLOR,           // DstColor
            VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, // OneMinusDstColor

            VK_BLEND_FACTOR_SRC_ALPHA,           // SrcAlpha
            VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, // OneMinusSrcAlpha
            VK_BLEND_FACTOR_DST_ALPHA,           // DstAlpha
            VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, // OneMinusDstAlpha
        };
        return s_vkFactors[_factor < BlendFactor::Count ? _factor : 0];
    }

}