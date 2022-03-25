#ifndef RL_SAC_H
#define RL_SAC_H

#include "RL_Agent.h"

namespace SAC {
class RL_SAC : public RL_Agent
{
public:
    explicit RL_SAC();
    virtual ~RL_SAC();

    /**
     * @brief train
     * @param cb_stat Called per episode. The statistics of training.
     *                void(frameId, reward, policy loss, Q-network loss, Value loss)
     * @param cb_anim Called on the fly. Performance hit will be high
     *                void(frameId, cartX, pole angle)
     */
    void train( double lr = 0.0001, int batch_size = 64,
                std::function<void (int, double, double, double, double)> *cb_stat = nullptr,
                std::function<void (int,
                                   std::vector<double>,
                                   std::vector<double>)> *cb_anim = nullptr) override;


    void eval(const std::string &checkPoint,
              std::function<bool ()> *noiseSignal = nullptr,
              std::function<bool (int, std::vector<double>, std::vector<double>, std::vector<double>)> *cb_anim = nullptr) override;

    void trainVision(double lr = 0.0001, int batch_size = 64,
                     std::function<void (int, double, double, double, double)> *cb_stat = nullptr,
                     std::function<std::pair<int, int> (std::vector<double>,
                                                        std::vector<double>,
                                                        std::vector<unsigned int> &)> *cb_anim = nullptr) override;


    void evalVision(const std::string &checkPoint,
              std::function<bool ()> *noiseSignal = nullptr,
                    std::function<bool (int,
                                        std::vector<double>,
                                        std::vector<double>,
                                        std::vector<double>)> *cb_anim = nullptr,
                    std::function<std::pair<int, int> (std::vector<double>,
                                                       std::vector<double>,
                                                       std::vector<unsigned int> &)> *cb_render = nullptr) override;

private:
    struct member;
    std::shared_ptr<member> m;
};
};

#endif // RL_SAC_H
