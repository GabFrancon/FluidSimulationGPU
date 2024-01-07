#pragma once
/// @file Forward.h
/// @brief Regroups forward declarations of the main Ava structures.

namespace Ava {

    class Application;
    class Window;
    class Layer;

    class FileMgr;
    class TimeMgr;
    class InputMgr;
    class DrawMgr;

    class Texture;
    class Shader;
    class FrameBuffer;
    class ShaderProgram;
    class GraphicsContext;
    class Camera;

    struct Color;
    struct Rect;
    struct BoundingBox;
    struct BoundingSphere;
    struct Plane;
    struct Frustum;

    class StringHash;
    class StringBuilder;
    class CmdLineParser;

    class Timestep;
    class FrameCounter;

    class Serializer;
    enum class SerializeError;

    class Event;
    class EventDispatcher;

    struct Color;
    struct Rect;

    class WindowResizedEvent;
    class WindowMovedEvent;
    class WindowMinimizedEvent;
    class WindowMaximizedEvent;
    class WindowRestoredEvent;
    class WindowFocusedEvent;
    class WindowClosedEvent;

    class FileChangedEvent;

    class MouseButtonPressedEvent;
    class MouseButtonReleasedEvent;
    class MouseEnteredFrameEvent;
    class MouseExitedFrameEvent;
    class MouseMovedEvent;
    class MouseScrolledEvent;

    class KeyPressedEvent;
    class KeyTypedEvent;
    class KeyReleasedEvent;

    class GamepadButtonPressedEvent;
    class GamepadButtonReleasedEvent;
    class GamepadJoystickMovedEvent;
    class GamepadTriggerMovedEvent;

}
