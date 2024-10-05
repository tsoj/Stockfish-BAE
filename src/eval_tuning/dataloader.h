#pragma once

#include <filesystem>
#include <cmath>
#include <memory>
#include <random>
#include <optional>
#include <iostream>

#include "../bae.h"

struct BufferEntry {
    Eval::EvalPosition pos;
    float              targetProbability;
};

template<typename T>
concept Datareader = requires(T t) {
    { t.next() } -> std::same_as<std::optional<BufferEntry>>;
};

class AbstractDataloader {
   public:
    [[nodiscard]] virtual BufferEntry next() = 0;
};

template<Datareader Datareader>
class Dataloader: public AbstractDataloader {
   public:
    explicit Dataloader(const std::filesystem::path& path, size_t bufferSize) :
        m_path(path),
        m_maxBufferSize(bufferSize) {

        assert(m_maxBufferSize > 0);
    }
    virtual ~Dataloader() = default;

    [[nodiscard]] inline BufferEntry next() final {
        if (!m_reader.has_value())
        {
            if (!m_buffer.empty())
            {
                m_uniformDist = std::nullopt;
                auto result   = m_buffer.back();
                m_buffer.pop_back();
                return result;
            }

            std::cout << "\nStarting epoch " << m_epoch << std::endl;
            m_epoch += 1;

            m_reader = Datareader(m_path.string());
            assert(m_buffer.empty());
            while (auto entry = m_reader->next())
            {
                m_buffer.push_back(entry.value());
                if (m_buffer.size() >= m_maxBufferSize)
                {
                    break;
                }
            }
            assert(!m_buffer.empty());
            assert(m_buffer.size() <= m_maxBufferSize);
            std::shuffle(std::begin(m_buffer), std::end(m_buffer), m_e1);
            m_uniformDist = std::uniform_int_distribution<size_t>(0, m_buffer.size() - 1);
        }

        const size_t index  = m_uniformDist.value()(m_e1);
        BufferEntry  result = m_buffer.at(index);

        const auto newEntry = m_reader->next();
        if (newEntry.has_value())
        {
            m_buffer.at(index) = newEntry.value();
        }
        else
        {
            m_reader = std::nullopt;
        }
        return result;
    }

   private:
    std::filesystem::path                                m_path;
    std::vector<BufferEntry>                             m_buffer;
    std::optional<Datareader>                            m_reader{std::nullopt};
    std::default_random_engine                           m_e1;
    std::optional<std::uniform_int_distribution<size_t>> m_uniformDist{std::nullopt};
    size_t                                               m_epoch = 0;
    size_t                                               m_maxBufferSize;
};


class AggregatedDataloader: public AbstractDataloader {
   public:
    explicit AggregatedDataloader(
      std::vector<std::pair<std::shared_ptr<AbstractDataloader>, float>>&& loaders) :
        m_loaders(std::move(loaders)) {
        float totalWeight = 0.0;
        for (auto& [loader, value] : m_loaders)
        {
            const float weight = value;
            value += totalWeight;
            totalWeight += weight;
        }
        m_uniformDist = std::uniform_real_distribution<float>(0.0, totalWeight);
    }

    [[nodiscard]] inline BufferEntry next() final {
        assert(!m_loaders.empty());

        const float index = m_uniformDist(m_e1);

        for (const auto& [loader, value] : m_loaders)
        {
            if (index <= value)
            {
                return loader->next();
            }
        }

        std::cerr << "WARNING: index outside of biggest loader. Index: " << index
                  << ", loader value: " << m_loaders.back().second;

        return m_loaders.back().first->next();
    }


   private:
    std::vector<std::pair<std::shared_ptr<AbstractDataloader>, float>> m_loaders;
    std::default_random_engine                                         m_e1;
    std::uniform_real_distribution<float>                              m_uniformDist;
};
