#pragma once
/// @file Layer.h
/// @brief

#include <Memory/Memory.h>

namespace Ava {

    /// @brief Interface describing a layer attached to the Ava application.
    class Layer
    {
    public:
        explicit Layer(const std::string& _name = "Layer") : m_debugName(_name) {}
        virtual ~Layer() = default;

        // Lifecycle of a layer
        virtual void OnAttach() {}
        virtual void OnUpdate(Timestep& _dt) {}
        virtual void OnRender(GraphicsContext* _ctx) {}
        virtual void OnEvent(Event& _event) {}
        virtual void OnDetach() {}

        const char* GetDebugName() const { return m_debugName.c_str(); }

    private:
        std::string m_debugName;
    };

    /// @brief Stack of layers held by the Ava application.
    class LayerStack
    {
    public:
        LayerStack() = default;
        ~LayerStack() = default;

        void PushLayer(const Ref<Layer>& _layer);
        void PushOverlay(const Ref<Layer>& _overlay);
        void PopLayer(const Ref<Layer>& _layer);
        void PopOverlay(const Ref<Layer>& _overlay);
        void Clear();

        using LayerCollection = std::vector<Ref<Layer>>;

        LayerCollection::iterator begin() { return m_layers.begin(); }
        LayerCollection::iterator end() { return m_layers.end(); }
        LayerCollection::reverse_iterator rbegin() { return m_layers.rbegin(); }
        LayerCollection::reverse_iterator rend() { return m_layers.rend(); }

        LayerCollection::const_iterator begin() const { return m_layers.begin(); }
        LayerCollection::const_iterator end() const { return m_layers.end(); }
        LayerCollection::const_reverse_iterator rbegin() const { return m_layers.rbegin(); }
        LayerCollection::const_reverse_iterator rend() const { return m_layers.rend(); }

    private:
        LayerCollection m_layers;
        u8 m_layerInsertIndex = 0;
    };

}
