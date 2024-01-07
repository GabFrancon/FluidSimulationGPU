#include <avapch.h>
#include "UILayer.h"

#include <Application/GUIApplication.h>
#include <Inputs/InputManager.h>
#include <Events/WindowEvent.h>
#include <Events/MouseEvent.h>
#include <Events/KeyEvent.h>
#include <Resources/ShaderData.h>
#include <Resources/TextureData.h>
#include <Time/TimeManager.h>
#include <Time/Profiler.h>
#include <UI/ImGuiTools.h>
#include <Math/Math.h>

#include <Graphics/Texture.h>
#include <Graphics/GpuBuffer.h>
#include <Graphics/FrameBuffer.h>
#include <Graphics/ShaderProgram.h>
#include <Graphics/GraphicsContext.h>

// pre-compiled data
#include "ImGuiFonts.embed"
#include "ImGuiShaders.embed"

namespace Ava {

    // ----- UI layer lifecycle -----------------------------------------------------------

    UILayer::UILayer(const ImGuiSettings& _settings)
        : Layer("UI Layer")
        , m_settings(_settings)
    {
        // Initializes ImGui library
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        // Loads ImGui config
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = m_settings.configFilePath.c_str();

        // Setups ImGui style
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowTitleAlign.x = 0.5f;
        style.WindowTitleAlign.y = 0.5f;
        style.WindowMinSize.x = 200.f;
        style.WindowMinSize.y = 20.f;
        style.WindowRounding = 12.f;
        style.FrameRounding = 12.f;
        style.GrabRounding = 12.f;
        style.ChildRounding = 12.f;
        style.ScrollbarRounding = 12.f;
        style.PopupRounding = 12.f;
        style.TabRounding = 4.f;
        style.WindowBorderSize = 1.f;
        style.FrameBorderSize = 1.f;
        style.TabBorderSize = 1.f;
        style.PopupBorderSize = 1.f;
        style.ChildBorderSize = 1.f;

        SetTheme(_settings.theme);
        SetConfigFlags(_settings.configFlags);

        // Setups ImGui fonts
        {
            const float regularTextSizePx = 16.f * m_settings.fontScale;
            const float boldTextSizePx = 22.f * m_settings.fontScale;

            // load Nunito regular text font
            ImFont* regularFont = io.Fonts->AddFontFromMemoryCompressedBase85TTF(
                nunito_regular_compressed_data_base85, 
                regularTextSizePx);

            // load Nunito bold text font
            ImFont* boldFont = io.Fonts->AddFontFromMemoryCompressedBase85TTF(
                nunito_bold_compressed_data_base85, 
                boldTextSizePx);

            ImGuiTools::SetFont(UI::FontRegular, regularFont);
            ImGuiTools::SetFont(UI::FontBold, boldFont);

            // FontAwesome fonts need to have their sizes reduced by 2/3 to align correctly
            const float regularIconSizePx = 2.f * regularTextSizePx / 3.f;
            const float solidIconSizePx = 2.f * boldTextSizePx / 3.f;
            static constexpr ImWchar iconsRange[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };

            ImFontConfig iconsConfig;
            iconsConfig.MergeMode = true;
            iconsConfig.PixelSnapH = true;

            // merge icons with regular font atlas
            iconsConfig.GlyphMinAdvanceX = regularIconSizePx;
            iconsConfig.DstFont = regularFont;

            io.Fonts->AddFontFromMemoryCompressedBase85TTF(
                fa_regular_400_compressed_data_base85, 
                regularIconSizePx, &iconsConfig, iconsRange);

            // merge icons with bold font atlas
            iconsConfig.GlyphMinAdvanceX = solidIconSizePx;
            iconsConfig.DstFont = boldFont;
            
            io.Fonts->AddFontFromMemoryCompressedBase85TTF(
                fa_solid_900_compressed_data_base85, 
                solidIconSizePx, &iconsConfig, iconsRange);

            // set regular font as the default one
            io.FontDefault = regularFont;
        }

        // Setups ImGui backend
        {
            AVA_ASSERT(!io.BackendPlatformUserData, "[ImGui] already initialized a platform backend.");

            const Window* window =  GUIApplication::GetInstance().GetWindow();
            const Vec2f windowSize = window->GetExtents();

            io.DisplaySize = ImVec2(windowSize.x, windowSize.y); 
            io.DeltaTime = 1.f / 60.f;

            io.SetClipboardTextFn = _SetClipboardText;
            io.GetClipboardTextFn = _GetClipboardText;
            io.ClipboardUserData = nullptr;

            // Our mouse update function expect PlatformHandle to be filled for the main viewport
            ImGuiViewport* viewport = ImGui::GetMainViewport();
            viewport->PlatformHandle = window->GetWindowHandle();
            viewport->PlatformHandleRaw = window->GetNativeWindowHandle();
        }

        // Setups ImGuizmo
        ImGuizmo::Enable(true);
        ImGuizmo::AllowAxisFlip(false);
    }

    UILayer::~UILayer()
    {
        // Shutdowns GLFW platform backend
        GraphicsContext::WaitIdle();
        // ImGui_ImplGlfw_Shutdown();

        // Destroys ImGui context
        ImGui::DestroyContext();
    }

    void UILayer::OnAttach()
    {
        // Loads font shader
        {
            Shader* stages[ShaderStage::Count]{};
            stages[ShaderStage::Geometric] = nullptr;
            stages[ShaderStage::Compute] = nullptr;

            ShaderData vertexShaderData;
            ShaderLoader::LoadFromMemory(imgui_vs_data, imgui_vs_size, vertexShaderData);
            stages[ShaderStage::Vertex] = GraphicsContext::CreateShader(vertexShaderData);
            ShaderLoader::Release(vertexShaderData);
            
            ShaderData fragmentShaderData;
            ShaderLoader::LoadFromMemory(imgui_fs_data, imgui_fs_size, fragmentShaderData);
            stages[ShaderStage::Fragment] = GraphicsContext::CreateShader(fragmentShaderData);
            ShaderLoader::Release(fragmentShaderData);

            ShaderResourceNames varNames;
            // @warning Must match the resources declared in "shaders/Debug/ImGui_vs.glsl"
            varNames.SetConstantBufferName(ShaderStage::Vertex, 0, "cbParameters");
            // @warning Must match the resources declared in "shaders/Debug/ImGui_fs.glsl"
            varNames.SetSampledTextureName(ShaderStage::Fragment, 0, "txColor");

            m_fontShaderProgram = GraphicsContext::CreateProgram(stages, varNames);
        }

        // Loads font atlas
        {
            const ImGuiIO& io = ImGui::GetIO();
            TextureData fontData;

            int width, height, channels;
            io.Fonts->GetTexDataAsRGBA32((unsigned char**)&fontData.binData, &width, &height, &channels);

            fontData.width = (u16)width;
            fontData.height = (u16)height;
            fontData.binSize = (u32)width * (u32)height * (u32)channels;

            fontData.depth = 1u;
            fontData.mipCount = 1u;
            fontData.imageCount = 1u;
            fontData.format = TextureFormat::RGBA8;

            auto& image = fontData.images.emplace_back();
            image.mipSizes[0] = fontData.binSize;
            image.mipPixels[0] = fontData.binData;

            m_fontTexture = GraphicsContext::CreateTexture(fontData, AVA_TEXTURE_SAMPLED);
            m_fontTexture->SetSamplerState(AVA_TEXTURE_NEAREST | AVA_TEXTURE_CLAMP);
            m_fontTexture->SetDebugName("TX_IMGUI_ATLAS");

            m_fontAtlas.texture = m_fontTexture;
            m_fontAtlas.shader = m_fontShaderProgram;
            io.Fonts->SetTexID(&m_fontAtlas);
        }
    }

    void UILayer::OnUpdate(Timestep& _dt)
    {
        AUTO_CPU_MARKER("IMGUI Update");

        const auto* window = GUIApplication::GetInstance().GetWindow();
        _UpdateInputs(_dt, window);

        if (!window->IsMinimized())
        {
            ImGui::NewFrame();
            ImGuizmo::BeginFrame();
        }
    }

    void UILayer::OnRender(GraphicsContext* _ctx)
    {
        ImGui::Render();

        const ImDrawData* drawCommands = ImGui::GetDrawData();
        AVA_ASSERT(drawCommands->Valid, "[ImGui] invalid draw data, did you call ImGui::Render() ?");

        if (drawCommands->CmdListsCount > 0)
        {
            AUTO_CPU_GPU_MARKER("IMGUI Render");
            _RenderDrawCommands(_ctx, drawCommands);
        }

        // When multi-viewports feature is enabled, we need to update the GLFW context.
        if (_HasConfigFlag(AVA_UI_MULTI_VIEWPORTS))
        {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }
    }

    void UILayer::OnEvent(Event& _event)
    {
        EventDispatcher dispatcher(_event);
        dispatcher.Dispatch<WindowFocusedEvent>(AVA_BIND_FN(_OnWindowFocused));
        dispatcher.Dispatch<MouseButtonPressedEvent>(AVA_BIND_FN(_OnMouseButtonPressed));
        dispatcher.Dispatch<MouseButtonReleasedEvent>(AVA_BIND_FN(_OnMouseButtonReleased));
        dispatcher.Dispatch<MouseMovedEvent>(AVA_BIND_FN(_OnMouseMoved));
        dispatcher.Dispatch<MouseScrolledEvent>(AVA_BIND_FN(_OnMouseScrolled));
        dispatcher.Dispatch<MouseEnteredFrameEvent>(AVA_BIND_FN(_OnMouseEnteredFrame));
        dispatcher.Dispatch<MouseExitedFrameEvent>(AVA_BIND_FN(_OnMouseExitedFrame));
        dispatcher.Dispatch<KeyPressedEvent>(AVA_BIND_FN(_OnKeyPressed));
        dispatcher.Dispatch<KeyReleasedEvent>(AVA_BIND_FN(_OnKeyReleased));
        dispatcher.Dispatch<KeyTypedEvent>(AVA_BIND_FN(_OnKeyTyped));
    }

    void UILayer::OnDetach()
    {
        // Releases font shader
        if (m_fontShaderProgram)
        {
            GraphicsContext::DestroyProgram(m_fontShaderProgram);
            m_fontShaderProgram = nullptr;
        }

        // Releases font atlas
        if (m_fontTexture)
        {
            GraphicsContext::DestroyTexture(m_fontTexture);
            m_fontTexture = nullptr;
        }
    }


    // ----- UI layer helpers -------------------------------------------------------------

    void UILayer::SetConfigFlag(const UIConfigFlags _flag, const bool _enable)
    {
        auto& io = ImGui::GetIO();
        const ImGuiConfigFlags imguiFlag = ImGuiTools::AvaToImGui(_flag);

        if (_enable)
        {
            m_settings.configFlags |= _flag;
            io.ConfigFlags |= imguiFlag;
        }
        else
        {
            m_settings.configFlags &= ~_flag;
            io.ConfigFlags &= ~imguiFlag;
        }
    }

    void UILayer::SetConfigFlags(const u32 _flags)
    {
        m_settings.configFlags = _flags;

        auto& style = ImGui::GetStyle();
        auto& io = ImGui::GetIO();
        io.ConfigFlags = 0;

        if (_HasConfigFlag(AVA_UI_KEYBOARD_NAVIGATION))
        {
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        }
        if (_HasConfigFlag(AVA_UI_GAMEPAD_NAVIGATION))
        {
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
        }
        if (_HasConfigFlag(AVA_UI_NO_MOUSE_CURSOR_CHANGE))
        {
            io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
        }
        if (_HasConfigFlag(AVA_UI_DOCKING))
        {
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        }
        if (_HasConfigFlag(AVA_UI_MULTI_VIEWPORTS))
        {
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

            // When multi-viewports feature is enabled, we have to modify the style to make
            // the transition seamless when moving an ImGui panel inside/outside the main window.
            style.WindowRounding = 0.f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }
    }

    bool UILayer::_HasConfigFlag(const UIConfigFlags _flag) const
    {
        return m_settings.configFlags & _flag;
    }

    void UILayer::SetBlockingEvent(const EventFlags _flag, const bool _enable)
    {
        if (_enable)
        {
            m_settings.blockingEventFlags |= _flag;
        }
        else
        {
            m_settings.blockingEventFlags &= ~_flag;
        }
    }

    void UILayer::SetBlockingEvents(const u32 _flags)
    {
         m_settings.blockingEventFlags = _flags;
    }

    bool UILayer::_ShouldBlockEvent(const EventFlags _flag) const
    {
        return m_settings.blockingEventFlags & _flag;
    }

    void UILayer::SetTheme(const UI::ThemePreset _theme)
    {
        m_settings.theme = _theme;

        switch (m_settings.theme)
        {
            case UI::ThemeClassic:
                ImGui::StyleColorsClassic();
                break;

            case UI::ThemeLight:
                ImGui::StyleColorsLight();
                break;

            case UI::ThemeDark:
                ImGui::StyleColorsDark();
                break;

            default:
                break;
        }
    }

    void UILayer::SetFramebuffer(FrameBuffer* _framebuffer)
    {
         m_targetFramebuffer = _framebuffer;
    }


    // ----- Backend implementation -----------------------------------------------------------

    void UILayer::_UpdateInputs(const Timestep& _dt, const Window* _window) const
    {
        ImGuiIO& io = ImGui::GetIO();
        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        const Vec2f windowSize = _window->GetExtents();

        // Set display size
        io.DisplaySize = ImVec2(windowSize.x, windowSize.y);

        if (windowSize.x > 0 && windowSize.y > 0)
        {
            const FrameBuffer* frameBuffer = m_targetFramebuffer ? m_targetFramebuffer : GraphicsContext::GetMainFramebuffer();
            const Vec2u framebufferSize = frameBuffer->GetSize();

            io.DisplayFramebufferScale.x = (float)framebufferSize.x / windowSize.x;
            io.DisplayFramebufferScale.y = (float)framebufferSize.y / windowSize.y;
        }

        // Set time step
        io.DeltaTime = static_cast<float>(_dt.GetSeconds());

        // No mouse cursor, we can early exit
        if (INPUT_MGR->GetMouseCursorMode() == Mouse::ModeDisabled)
        {
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
            return;
        }

        // Set mouse cursor position
        if (io.WantSetMousePos && _window->IsFocused())
        {
            Vec2f position = INPUT_MGR->GetMouseCursorPosition();
            position.x -= viewport->Pos.x;
            position.y -= viewport->Pos.y;

            INPUT_MGR->SetMouseCursorPosition(position);
        }

        // Set mouse cursor icon and mode
        if (!(io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange))
        {
            const ImGuiMouseCursor imguiCursor = ImGui::GetMouseCursor();

            if (imguiCursor == ImGuiMouseCursor_None || io.MouseDrawCursor)
            {
                // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor
                INPUT_MGR->SetMouseCursorMode(Mouse::ModeHidden);
            }
            else
            {
                const Mouse::CursorIcon icon = ImGuiTools::ImGuiToAva(imguiCursor);
                INPUT_MGR->SetMouseCursorIcon(icon);
                INPUT_MGR->SetMouseCursorMode(Mouse::ModeNormal);
            }
        }
    }

    void UILayer::_RenderDrawCommands(const GraphicsContext* _ctx, const ImDrawData* _drawData) const
    {
        // No ImGui commands to draw
        if (_drawData->CmdListsCount == 0)
        {
            return;
        }

        _ctx->Reset();
        _ctx->SetPrimitiveType(PrimitiveType::Triangle);

        // Blend state enabled
        static const BlendStateID blendState = GraphicsContext::CreateBlendState(BlendState::AlphaBlending);
        _ctx->SetBlendState(blendState);

        // Vertex buffer layout
        static VertexLayout vertexLayout = VertexLayout()
                .AddAttribute(VertexSemantic::Position, DataType::FLOAT32, 2)
                .AddAttribute(VertexSemantic::TexCoord, DataType::FLOAT32, 2)
                .AddAttribute(VertexSemantic::Color, DataType::UNORM8, 4)
                .Build();

        // Framebuffer
        FrameBuffer* frameBuffer = m_targetFramebuffer ? m_targetFramebuffer : _ctx->GetMainFramebuffer();
        _ctx->SetFramebuffer(frameBuffer);

        // Viewport
        const Vec2f renderExtents = frameBuffer->GetSize();
        _ctx->SetViewport((u16)renderExtents.x, (u16)renderExtents.y);

        // ImGui requires right-handed openGL styled view projection matrices.
        const Mat4 viewproj = glm::orthoRH(0.f, renderExtents.x, 0.f, renderExtents.y, -1.f, 1.f);

        // @warning Must match the cbParameters buffer defined in "shaders/Debug/ImGui_vs.glsl"
        struct VertexParamGPU
        {
            Mat4 viewproj;
        };

        // Vertex shader param buffer
        const ConstantBufferRange vertexParamBuffer = _ctx->CreateTransientConstantBuffer(sizeof VertexParamGPU);
        auto* vertexShaderData = static_cast<VertexParamGPU*>(vertexParamBuffer.data);
        vertexShaderData->viewproj = viewproj;

        _ctx->SetConstantBuffer(ShaderStage::Vertex, 0, vertexParamBuffer);

        // ImGui draw commands
        for (int i = 0; i < _drawData->CmdListsCount; i++)
        {
            const ImDrawList* cmdList = _drawData->CmdLists[i];

            // Vertex buffer
            VertexBufferRange vertexBuffer = _ctx->CreateTransientVertexBuffer(vertexLayout, cmdList->VtxBuffer.size());
            vertexBuffer.data = cmdList->VtxBuffer.Data;
            _ctx->SetVertexBuffer(vertexBuffer);

            ImDrawIdx* indices = cmdList->IdxBuffer.Data;

            // Draw commands
            for (const ImDrawCmd* drawCommand = cmdList->CmdBuffer.begin(); drawCommand != cmdList->CmdBuffer.end(); drawCommand++)
            {
                if (drawCommand->UserCallback)
                {
                    drawCommand->UserCallback(cmdList, drawCommand);
                }
                else
                {
                    ImVec4 clipRect = drawCommand->ClipRect;
                    clipRect.x = Math::max(0.f, clipRect.x);
                    clipRect.y = Math::max(0.f, clipRect.y);
                    clipRect.z = Math::max(0.f, clipRect.z);
                    clipRect.w = Math::max(0.f, clipRect.w);

                    const u16 scissorX = (u16)clipRect.x;
                    const u16 scissorY = (u16)clipRect.y;
                    const u16 scissorWidth = (u16)clipRect.z - (u16)clipRect.x;
                    const u16 scissorHeight = (u16)clipRect.w - (u16)clipRect.y;

                    if (scissorWidth > 0 && scissorHeight > 0)
                    {
                        const auto* shadedTexture = static_cast<ShadedTexture*>(drawCommand->TextureId);

                        // Scissor
                        _ctx->SetScissor(scissorWidth, scissorHeight, scissorX, scissorY);

                        // Index buffer
                        IndexBufferRange indexBuffer = _ctx->CreateTransientIndexBuffer(drawCommand->ElemCount);
                        indexBuffer.data = indices;
                        _ctx->SetIndexBuffer(indexBuffer);

                        // Font / User texture
                        Texture* texture = m_fontTexture;
                        if (shadedTexture && shadedTexture->texture)
                        {
                            texture = shadedTexture->texture;
                        }
                        _ctx->SetTexture(ShaderStage::Fragment, 0, texture);

                        // Font / User shader
                        ShaderProgram* shader = m_fontShaderProgram;
                        if (shadedTexture && shadedTexture->shader)
                        {
                            shader = shadedTexture->shader;
                        }
                        _ctx->Draw(shader);
                    }
                }
                indices += drawCommand->ElemCount;
            }
        }
    }


    // ----- Event callbacks -----------------------------------------------------------

    static Vec2f s_lastValidMousePosition = Math::Origin;

    bool UILayer::_OnWindowFocused(const WindowFocusedEvent& _event) const
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddFocusEvent(true);

        return _ShouldBlockEvent(AVA_EVENT_WINDOW);
    }

    bool UILayer::_OnMouseButtonPressed(const MouseButtonPressedEvent& _event) const
    {
        const Mouse::Button button = _event.GetMouseButton();
        const u16 mods = _event.GetModifiers();

        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiMod_Ctrl, mods & Keyboard::AVA_MODIFIER_CTRL);
        io.AddKeyEvent(ImGuiMod_Shift, mods & Keyboard::AVA_MODIFIER_SHIFT);
        io.AddKeyEvent(ImGuiMod_Alt, mods & Keyboard::AVA_MODIFIER_ALT);
        io.AddKeyEvent(ImGuiMod_Super, mods & Keyboard::AVA_MODIFIER_SUPER);
        io.AddMouseButtonEvent(ImGuiTools::AvaToImGui(button), true);

        return io.WantCaptureMouse && _ShouldBlockEvent(AVA_EVENT_MOUSE_BUTTON);
    }

    bool UILayer::_OnMouseButtonReleased(const MouseButtonReleasedEvent& _event) const
    {
        const Mouse::Button button = _event.GetMouseButton();
        const u16 mods = _event.GetModifiers();

        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiMod_Ctrl, mods & Keyboard::AVA_MODIFIER_CTRL);
        io.AddKeyEvent(ImGuiMod_Shift, mods & Keyboard::AVA_MODIFIER_SHIFT);
        io.AddKeyEvent(ImGuiMod_Alt, mods & Keyboard::AVA_MODIFIER_ALT);
        io.AddKeyEvent(ImGuiMod_Super, mods & Keyboard::AVA_MODIFIER_SUPER);
        io.AddMouseButtonEvent(ImGuiTools::AvaToImGui(button), false);

        return io.WantCaptureMouse && _ShouldBlockEvent(AVA_EVENT_MOUSE_BUTTON);
    }

    bool UILayer::_OnMouseEnteredFrame(const MouseEnteredFrameEvent& _event) const
    {
        ImGuiIO& io = ImGui::GetIO();

        if (INPUT_MGR->GetMouseCursorMode() == Mouse::ModeDisabled)
        {
            return false;
        }

        io.AddMousePosEvent(s_lastValidMousePosition.x, s_lastValidMousePosition.y);
        return io.WantCaptureMouse && _ShouldBlockEvent(AVA_EVENT_MOUSE_CURSOR);
    }

    bool UILayer::_OnMouseExitedFrame(const MouseExitedFrameEvent& _event) const
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);

        return io.WantCaptureMouse && _ShouldBlockEvent(AVA_EVENT_MOUSE_CURSOR);
    }

    bool UILayer::_OnMouseMoved(const MouseMovedEvent& _event) const
    {
        ImGuiIO& io = ImGui::GetIO();

        if (INPUT_MGR->GetMouseCursorMode() == Mouse::ModeDisabled)
        {
            return false;
        }

        io.AddMousePosEvent(_event.GetX(), _event.GetY());
        s_lastValidMousePosition = { _event.GetX(), _event.GetY() };

        return io.WantCaptureMouse && _ShouldBlockEvent(AVA_EVENT_MOUSE_CURSOR);
    }

    bool UILayer::_OnMouseScrolled(const MouseScrolledEvent& _event) const
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddMouseWheelEvent(_event.GetXOffset(), _event.GetYOffset());

        return io.WantCaptureMouse && _ShouldBlockEvent(AVA_EVENT_MOUSE_WHEEL);
    }

    bool UILayer::_OnKeyPressed(const KeyPressedEvent& _event) const
    {
        const Keyboard::Key key = _event.GetKeyCode();
        const u16 mods = _event.GetModifiers();

        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiMod_Ctrl, mods & Keyboard::AVA_MODIFIER_CTRL);
        io.AddKeyEvent(ImGuiMod_Shift, mods & Keyboard::AVA_MODIFIER_SHIFT);
        io.AddKeyEvent(ImGuiMod_Alt, mods & Keyboard::AVA_MODIFIER_ALT);
        io.AddKeyEvent(ImGuiMod_Super, mods & Keyboard::AVA_MODIFIER_SUPER);
        io.AddKeyEvent(ImGuiTools::AvaToImGui(key), true);

        return io.WantCaptureKeyboard && _ShouldBlockEvent(AVA_EVENT_KEYBOARD);
    }

    bool UILayer::_OnKeyReleased(const KeyReleasedEvent& _event) const
    {
        const Keyboard::Key key = _event.GetKeyCode();
        const u16 mods = _event.GetModifiers();

        ImGuiIO& io = ImGui::GetIO();
        io.AddKeyEvent(ImGuiMod_Ctrl, mods & Keyboard::AVA_MODIFIER_CTRL);
        io.AddKeyEvent(ImGuiMod_Shift, mods & Keyboard::AVA_MODIFIER_SHIFT);
        io.AddKeyEvent(ImGuiMod_Alt, mods & Keyboard::AVA_MODIFIER_ALT);
        io.AddKeyEvent(ImGuiMod_Super, mods & Keyboard::AVA_MODIFIER_SUPER);
        io.AddKeyEvent(ImGuiTools::AvaToImGui(key), false);

        return io.WantCaptureKeyboard &&_ShouldBlockEvent(AVA_EVENT_KEYBOARD);
    }

    bool UILayer::_OnKeyTyped(const KeyTypedEvent& _event) const
    {
        ImGuiIO& io = ImGui::GetIO();
        io.AddInputCharacter(_event.GetChar());

        return io.WantCaptureKeyboard &&_ShouldBlockEvent(AVA_EVENT_KEYBOARD);
    }


    // ----- Clipboard inputs ----------------------------------------------------------

    void UILayer::_SetClipboardText(void* _userData, const char* _text)
    {
        InputMgr::CopyToClipBoard(_text);
    }

    const char* UILayer::_GetClipboardText(void* _userData)
    {
        return InputMgr::GetClipBoard();
    }

}
