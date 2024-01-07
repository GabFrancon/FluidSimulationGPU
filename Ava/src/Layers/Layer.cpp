#include <avapch.h>
#include "Layer.h"

namespace Ava {

    void LayerStack::PushLayer(const Ref<Layer>& _layer)
    {
        const auto it = std::find(m_layers.begin(), m_layers.begin() + m_layerInsertIndex, _layer);
        if (it == m_layers.begin() + m_layerInsertIndex)
        {
            _layer->OnAttach();
            m_layers.emplace(m_layers.begin() + m_layerInsertIndex, _layer);
            m_layerInsertIndex++;
        }
    }

    void LayerStack::PushOverlay(const Ref<Layer>& _overlay)
    {
        const auto it = std::find(m_layers.begin() + m_layerInsertIndex, m_layers.end(), _overlay);
        if (it == m_layers.end())
        {
            _overlay->OnAttach();
            m_layers.emplace_back(_overlay);
        }
    }

    void LayerStack::PopLayer(const Ref<Layer>& _layer)
    {
        const auto it = std::find(m_layers.begin(), m_layers.begin() + m_layerInsertIndex, _layer);
        if (it != m_layers.begin() + m_layerInsertIndex)
        {
            _layer->OnDetach();
            m_layers.erase(it);
            m_layerInsertIndex--;
        }
    }

    void LayerStack::PopOverlay(const Ref<Layer>& _overlay)
    {
        const auto it = std::find(m_layers.begin() + m_layerInsertIndex, m_layers.end(), _overlay);
        if (it != m_layers.end())
        {
            _overlay->OnDetach();
            m_layers.erase(it);
        }
    }

    void LayerStack::Clear()
    {
        for (const auto& layer : m_layers)
        {
            layer->OnDetach();
        }

        m_layers.clear();
        m_layerInsertIndex = 0;
    }

}
