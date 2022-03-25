#include "rl_sac.h"
#include "gym_torch.h"

#include <filesystem>
#include <algorithm>
#include <random>
#include <time.h>
#include <regex>

#include <torch/torch.h>
#include <torch/jit.h>


#if defined(WITH_VISION) && defined(QT_CORE_LIB)
#include <QImage>
#endif


namespace SAC {

int step_counters = 0;

template <typename U>
struct ReplayBuffer {
    ReplayBuffer(int maxBufferSize) :
        maxBufferSize(maxBufferSize){
        assert( 0 != maxBufferSize );
        buffer.reserve(maxBufferSize);
        currentId = 0;
    }
    size_t length(){return buffer.size();}

    void push(const U& data) {
        if ( maxBufferSize > buffer.size()) {
            buffer.push_back(data);
            return;
        }
        buffer[currentId] = data;
        currentId = ++currentId % maxBufferSize;
    }

    auto sample(const int batch_size) {
        std::vector<U> samples;
        samples.reserve(batch_size);
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(samples),
                    batch_size, std::mt19937{std::random_device{}()});
        std::random_shuffle(samples.begin(), samples.end());

        std::vector<torch::Tensor> states(batch_size);
        std::vector<torch::Tensor> actions(batch_size);
        std::vector<torch::Tensor> rewards(batch_size);
        std::vector<torch::Tensor> nextStates(batch_size);
        std::vector<torch::Tensor> dones(batch_size);

        for ( auto i=0; i<batch_size; ++i ) {
            const auto &b = samples.at(i);
            states[i] = std::get<0>(b);
            actions[i] = std::get<1>(b);
            rewards[i] = std::get<2>(b);
            nextStates[i] = std::get<3>(b);
            dones[i] = std::get<4>(b);
        }
        return std::make_tuple<>(states, actions, rewards, nextStates, dones);
    }

    void load(const std::string &prefix) {
        if ( !buffer.empty()) {
            std::cout << "Please clear the buffer explicitly!" << std::endl;
            return;
        }
        try {
            int bufferSize = 0, save_batch = 1;
            std::string tms;
            std::ifstream in;
            char filename[60];
            sprintf(filename, "%s_last.txt", prefix.c_str());
            in.open(filename, std::ios::in);
            in >> bufferSize >> save_batch >> tms;
            in.close();

            std::cout << bufferSize << " "
                      << save_batch << " "
                      << " Memory of : " << tms << std::endl;

            buffer.resize(bufferSize);

            torch::Tensor states, rewards, actions, next_states, dones;

            int i;
            for (i=0; i<bufferSize; i+=save_batch ) {
                int last_batch = (i+1) * save_batch-1;
                if ( last_batch > bufferSize ) {
                    last_batch = bufferSize-1;
                }
                sprintf(filename, "%s_states_%d-%d.pt", prefix.c_str(), i, last_batch);
                torch::load(states, filename);
                sprintf(filename, "%s_actions_%d-%d.pt", prefix.c_str(), i, last_batch);
                torch::load(actions, filename);
                sprintf(filename, "%s_rewards_%d-%d.pt", prefix.c_str(), i, last_batch);
                torch::load(rewards, filename);
                sprintf(filename, "%s_next_states_%d-%d.pt", prefix.c_str(), i, last_batch);
                torch::load(next_states, filename);
                sprintf(filename, "%s_dones_%d-%d.pt", prefix.c_str(), i, last_batch);
                torch::load(dones, filename);

                if ( states.sizes()[0] != actions.sizes()[0] || rewards.sizes()[0] != next_states.sizes()[0]
                     || dones.sizes()[0] != states.sizes()[0] || actions.sizes()[0] != rewards.sizes()[0]) {
                    throw std::runtime_error("Invalid memory data : " + std::string(filename));
                }

                for ( int j=0; j<save_batch; ++j ) {
                    buffer[j+i] = std::make_tuple<>(states[j],actions[j],rewards[j],next_states[j],dones[j]);
                }
            }

            if ( buffer.size() != bufferSize ) {
                 throw std::runtime_error("Memory loading failed!");
            }
            std::cout << "Memory loaded!" << std::endl;
        } catch (const std::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }

    int save(const std::string &prefix, int lastSaved = -1, int save_batch = 512){
        if ( buffer.empty()) {
            std::cout << "Empty buffer!" << std::endl;
            return -1;
        }
        try {
            if ( !std::filesystem::exists(prefix)) {
                std::filesystem::create_directories(prefix);
            }
            const int bufferSize = buffer.size();

            int i=lastSaved;
            if ( i < 0 ) {
                i = 0;
            }
            int j=0;
            char filename[60];
            //Save the every "save_batch" entries
            for ( ; i<bufferSize; i+=save_batch ) {
                std::vector<torch::Tensor> states, rewards, actions, next_states, dones;
                for ( j=i; j<bufferSize; ++j ) {
                    auto &b = buffer.at(j);
                    states.push_back(std::get<0>(b));
                    actions.push_back(std::get<1>(b));
                    rewards.push_back(std::get<2>(b));
                    next_states.push_back(std::get<3>(b));
                    dones.push_back(std::get<4>(b));
                }
                --j;    //The actual index
                sprintf(filename, "%s_states_%d-%d.pt", prefix.c_str(), i, j);
                torch::save(torch::stack(states), filename);
                sprintf(filename, "%s_actions_%d-%d.pt", prefix.c_str(), i, j);
                torch::save(torch::stack(actions), filename);
                sprintf(filename, "%s_rewards_%d-%d.pt", prefix.c_str(), i, j);
                torch::save(torch::stack(rewards), filename);
                sprintf(filename, "%s_next_states_%d-%d.pt", prefix.c_str(), i, j);
                torch::save(torch::stack(next_states), filename);
                sprintf(filename, "%s_dones_%d-%d.pt", prefix.c_str(), i, j);
                torch::save(torch::stack(dones), filename);
            }

            auto tm = std::chrono::system_clock::now();
            auto tm_ = std::chrono::system_clock::to_time_t(tm);
            char tms[64];
#ifdef WIN32
            ctime_s(tms, sizeof tms, &tm_);
#else
            std::string tmp = ctime(&tm_);
            strcpy(tms, tmp.data());
#endif
            sprintf(filename, "%s_last.txt", prefix.c_str());

            std::ofstream out;
            out.open(filename, std::ios::out);
            out << bufferSize << std::endl;
            out << save_batch << std::endl;;
            out << tms;
            out.close();
        } catch (const std::exception &e) {
            std::cout << e.what() << std::endl;
        }
        return buffer.size();
    }

    std::vector<U> buffer;
    const int maxBufferSize;
    int currentId;
};

void initialize_weights(torch::nn::Module& module) {
    torch::NoGradGuard no_grad;
    if (auto* linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::xavier_uniform_(linear->weight, 1.0);
        torch::nn::init::constant_(linear->bias, 0.0);
    } else if (auto* conv = module.as<torch::nn::Conv2d>()) {
        torch::nn::init::xavier_uniform_(conv->weight, 1.0);
    }
}

std::string save_path;
#ifdef WITH_VISION

void toImage(const torch::Tensor &in){
    const auto &g = in[0];//torch::squeeze(in.clone(), 0);
    const auto ss = g.sizes();
    constexpr int gridSep = 2;
    const int gridW = std::ceil(std::sqrt(double(ss[0])));
    const int gridH = std::ceil(double(ss[0]) / gridW);
    QImage img(gridW*( ss[2]+gridSep ), gridH*( ss[1]+gridSep ), QImage::Format_ARGB32);
    img.fill(qRgba(225, 235, 105, 255));
    for ( int c=0; c<ss[0]; ++c ) { //The channel
        const int x_offset = ( c % gridW )*(ss[2]+gridSep);
        const int y_offset = ( c / gridW )*(ss[1]+gridSep);
        const auto &channel = g[c];
        for ( int i=0; i<ss[2]; ++i ){  //Width
            for ( int j=0; j<ss[1]; ++j ) { //Height
                const auto &px = channel[i][j];
                img.setPixel(i + x_offset, j + y_offset,
                             qRgb(px.item().toInt(),
                                  px.item().toInt(),
                                  px.item().toInt()));
            }
        }
    }

    auto fileName = QString("%3/Conv_%1x%1x%2.png").arg(ss[1]).arg(ss[0]).arg(save_path.data());
    std::cout << fileName.toUtf8().data() << " save : " << img.save(fileName, "PNG", 100) << std::endl;
};


struct VisionNetworkImpl : torch::nn::Module {
    VisionNetworkImpl(  int64_t inDim = 128,
                        int64_t outDim = 128):
        m_inDim(inDim),
        m_outDim(outDim)
    {
        //128 * 128
        conv1 = register_module("Conv1", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(4, 8, 4)
//                                    .padding(0).stride(2).bias(true)));
                                    torch::nn::Conv2dOptions(4, 16, 8)
                                    .padding(0).stride(4).bias(true)));
        //64 * 64
        conv2 = register_module("Conv2", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(8, 16, 4)
//                                    .padding(0).stride(2).bias(true)));
                                    torch::nn::Conv2dOptions(16, 32, 4)
                                    .padding(0).stride(2).bias(true)));
        //32 * 32
        conv3 = register_module("Conv3", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(16, 16, 4)
//                                    .padding(0).stride(2).bias(true)));
                                    torch::nn::Conv2dOptions(32, 64, 3)
                                    .padding(1).stride(2).bias(true)));
//        //16 * 16
//        conv4 = register_module("Conv4", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(64, 128, 3)
//                                    .padding(1).stride(2).bias(true)));
//        //8 * 8
//        conv5 = register_module("Conv5", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(128, 256, 3)
//                                    .padding(1).stride(2).bias(true)));
//        //4 * 4
//        conv6 = register_module("Conv6", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(256, 512, 3)
//                                    .padding(1).stride(2).bias(true)));
//        //2 * 2
//        conv7 = register_module("Conv7", torch::nn::Conv2d(
//                                    torch::nn::Conv2dOptions(512, 1024, 2)
//                                    .bias(true)));

//        max_pool1 = register_module("MaxPool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({8,8}).stride({4,4})));
//        max_pool2 = register_module("MaxPool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({6,6}).stride({3,3})));
//        max_pool3 = register_module("MaxPool3", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({4,4}).stride({2,2})));

//        bn1 = register_module("BN1", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1024)));
//        bn2 = register_module("BN2", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)));

//        linear1 = register_module("Linear1", torch::nn::Linear(144, 512));
//        linear2 = register_module("Linear2", torch::nn::Linear(512, m_outDim));

        apply(initialize_weights);
        m_saveOnce = false;
    }

    torch::Tensor forward(torch::Tensor state) {
        auto x = state.view({-1, m_inDim, m_inDim, 4});
        x = torch::permute(x, {0, 3, 2, 1});

        x = conv1(x);
        x = torch::nn::functional::relu(x);
        if ( m_saveOnce ) {
            toImage(x);
        }

        x = conv2(x);
        x = torch::nn::functional::relu(x);
        if ( m_saveOnce ) {
            toImage(x);
        }

        x = conv3(x);
        x = torch::nn::functional::relu(x);
        if ( m_saveOnce ) {
            toImage(x);
        }

        x = torch::permute(x, {0, 1, 2, 3});
//        x = torch::nn::functional::relu(conv4(x));
//        toImage(x);


//        x = torch::nn::functional::relu(conv5(x));
//        toImage(x);

//        x = torch::nn::functional::relu(linear1(x));
//        x = linear2(x);

        m_saveOnce = false;
        return x;
    }

    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::Conv2d conv2 = nullptr;
    torch::nn::Conv2d conv3 = nullptr;
//    torch::nn::Conv2d conv4 = nullptr;
//    torch::nn::Conv2d conv5 = nullptr;

    torch::nn::BatchNorm1d bn1 = nullptr;
    torch::nn::BatchNorm1d bn2 = nullptr;

//    torch::nn::MaxPool2d max_pool1 = nullptr;
//    torch::nn::MaxPool2d max_pool2 = nullptr;
//    torch::nn::MaxPool2d max_pool3 = nullptr;

//    torch::nn::Linear linear1 = nullptr;
//    torch::nn::Linear linear2 = nullptr;

    int64_t m_inDim;
    int64_t m_outDim;
    bool m_saveOnce;
};
TORCH_MODULE(VisionNetwork);

#endif

struct QNetworkImpl : torch::nn::Module {
    QNetworkImpl(  int64_t stateDim,
                       int64_t actionDim,
                       int64_t hiddenDim = 128):
        m_stateDim(stateDim),
        m_actionDim(actionDim),
        m_hiddenDim(hiddenDim)
    {
        linear1 = register_module("Linear1", torch::nn::Linear(m_stateDim+m_actionDim, m_hiddenDim));
        linear2 = register_module("Linear2", torch::nn::Linear(m_hiddenDim, m_hiddenDim));
        linear3 = register_module("Linear3", torch::nn::Linear(m_hiddenDim, 1));

        apply(initialize_weights);
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = torch::cat({state, action}, 1);
        x = torch::nn::functional::relu(linear1->forward(x));
        x = torch::nn::functional::relu(linear2->forward(x));
        x = linear3->forward(x);

        return x;
    }

    torch::nn::Linear linear1 = nullptr;
    torch::nn::Linear linear2 = nullptr;
    torch::nn::Linear linear3 = nullptr;

    int64_t m_stateDim;
    int64_t m_actionDim;
    int64_t m_hiddenDim;
};
TORCH_MODULE(QNetwork);

#ifdef WITH_VISION
struct VQNetworkImpl : torch::nn::Module {
    VQNetworkImpl(  int64_t stateDim,
                    int64_t actionDim,
                    int64_t hiddenDim = 128):
        m_stateDim(stateDim),
        m_actionDim(actionDim),
        m_hiddenDim(hiddenDim)
    {
        vision = register_module("Vision", VisionNetwork(m_stateDim, 3136));
        qnet = register_module("QNet", QNetwork(3136, m_actionDim, m_hiddenDim));

        apply(initialize_weights);
    }

    torch::Tensor forward(torch::Tensor state, torch::Tensor action) {
        auto x = vision(state);
        x = x.view({x.sizes()[0],-1});
        x = qnet(x, action);

        return x;
    }

    VisionNetwork vision = nullptr;
    QNetwork qnet = nullptr;

    int64_t m_stateDim;
    int64_t m_actionDim;
    int64_t m_hiddenDim;
};
TORCH_MODULE(VQNetwork);
#endif

struct BiQNetworkImpl : torch::nn::Module {
    BiQNetworkImpl(   int64_t stateDim,
                      int64_t actionDim,
                      int64_t hiddenDim = 128):
        m_stateDim(stateDim),
        m_actionDim(actionDim),
        m_hiddenDim(hiddenDim)
    {
        Q1 = register_module("Q1", QNetwork(m_stateDim, m_actionDim, m_hiddenDim));
        Q2 = register_module("Q2", QNetwork(m_stateDim, m_actionDim, m_hiddenDim));
    }

    auto forward(torch::Tensor state, torch::Tensor action) {
        auto x1 = Q1(state, action);
        auto x2 = Q2(state, action);

        return std::make_tuple<>(x1, x2);
    }

    void to(torch::Device device, bool non_blocking = false) override
    {
        Q1->to(device, non_blocking);
        Q2->to(device, non_blocking);
        torch::nn::Module::to(device, non_blocking);
    }

    QNetwork Q1 = nullptr;
    QNetwork Q2 = nullptr;

    int64_t m_stateDim;
    int64_t m_actionDim;
    int64_t m_hiddenDim;
};
TORCH_MODULE(BiQNetwork);

#ifdef WITH_VISION
struct BiVQNetworkImpl : torch::nn::Module {
    BiVQNetworkImpl(   int64_t stateDim,
                      int64_t actionDim,
                      int64_t hiddenDim = 128):
        m_stateDim(stateDim),
        m_actionDim(actionDim),
        m_hiddenDim(hiddenDim)
    {
        VQ1 = register_module("VQ1", VQNetwork(m_stateDim, m_actionDim, m_hiddenDim));
        VQ2 = register_module("VQ2", VQNetwork(m_stateDim, m_actionDim, m_hiddenDim));
    }

    auto forward(torch::Tensor state, torch::Tensor action) {
        auto x1 = VQ1(state, action);
        auto x2 = VQ2(state, action);

        return std::make_tuple<>(x1, x2);
    }

    void to(torch::Device device, bool non_blocking = false) override
    {
        VQ1->to(device, non_blocking);
        VQ2->to(device, non_blocking);
        torch::nn::Module::to(device, non_blocking);
    }

    VQNetwork VQ1 = nullptr;
    VQNetwork VQ2 = nullptr;

    int64_t m_stateDim;
    int64_t m_actionDim;
    int64_t m_hiddenDim;

};
TORCH_MODULE(BiVQNetwork);
#endif

struct GaussianPolicyImpl : torch::nn::Module {
    GaussianPolicyImpl(int64_t stateDim,
                       int64_t actionDim,
                       int64_t hiddenDim = 128,
                       double logStdMin = -20.0,
                       double logStdMax = 2.0)
        :m_logStdMin(logStdMin),
         m_logStdMax(logStdMax)
    {
        linear1 = register_module("Linear1", torch::nn::Linear(stateDim, hiddenDim));
        linear2 = register_module("Linear2", torch::nn::Linear(hiddenDim, hiddenDim));

        mean_linear = register_module("MeanLinear", torch::nn::Linear(hiddenDim, actionDim));
        log_std_linear = register_module("LogStdLinear", torch::nn::Linear(hiddenDim, actionDim));

        apply(initialize_weights);

        //# action rescaling
        action_scale = torch::ones({1}) * 1.0;
        action_bias = torch::ones({1}) * 0.0;
    }

    auto forward(torch::Tensor state){
        auto x = torch::nn::functional::relu(linear1(state));
        x = torch::nn::functional::relu(linear2(x));
        auto mean = mean_linear(x);
        auto log_std = log_std_linear(x);
        log_std = torch::clamp(log_std, m_logStdMin, m_logStdMax);
        return std::make_tuple<>(mean, log_std);
    }

    auto sample(torch::Tensor state, double epsilon=1e-6)
    {
        auto &&[mean, log_std] = this->forward(state);
        auto std = log_std.exp();
       // auto normal = at::normal(mean, std);
        static auto rsample = [](torch::Tensor mean, torch::Tensor std){
            //auto eps = at::normal(mean, std);
            auto eps = at::normal(0, 1, mean.sizes()).to(mean.device());
            eps.set_requires_grad(false);
            return mean + eps * std;
        };
        static auto logSqrt2Pi = torch::zeros({1}).to(mean.device());
        static std::once_flag flag;
        std::call_once(flag, [](){
            logSqrt2Pi[0] = 2*M_PI;
            logSqrt2Pi = torch::log(torch::sqrt(logSqrt2Pi));
        });
        static auto log_prob_func = [](torch::Tensor value, torch::Tensor mean, torch::Tensor std){
            auto var = std.pow(2);
            auto log_scale = std.log();
            return -(value - mean).pow(2) / (2 * var) - log_scale - logSqrt2Pi;
        };

        auto x_t = rsample(mean, std);  // for reparameterization trick (mean + std * N(0,1))
        auto y_t = torch::tanh(x_t);
        auto action = y_t * action_scale + action_bias;
        auto log_prob = log_prob_func(x_t, mean, std);
        // Enforcing Action Bound
        log_prob -= torch::log(action_scale * (1 - y_t.pow(2)) + epsilon);

        log_prob = log_prob.sum(1, true);

        mean = torch::tanh(mean) * action_scale + action_bias;
        return std::make_tuple<>(action, log_prob, mean);
    }

    auto select_action(torch::Tensor state, bool eval = false ){
        auto state_ = torch::unsqueeze(state, 0);
        torch::Tensor mean, tmp, action;
        std::tie( action, tmp, mean ) = this->sample(state_);
        if (eval){
            return mean.detach().cpu()[0];
        }
        return action.detach().cpu()[0];
    }

    void to(torch::Device device, bool non_blocking = false) override
    {
        action_scale = action_scale.to(device, non_blocking);
        action_bias = action_bias.to(device, non_blocking);
        torch::nn::Module::to(device, non_blocking);
    }

    torch::nn::Linear linear1 = nullptr;
    torch::nn::Linear linear2 = nullptr;

    torch::nn::Linear mean_linear = nullptr;
    torch::nn::Linear log_std_linear = nullptr;

    torch::Tensor action_scale;
    torch::Tensor action_bias;

    double m_logStdMin;
    double m_logStdMax;
};
TORCH_MODULE(GaussianPolicy);

#ifdef WITH_VISION
struct VGaussianPolicyImpl : torch::nn::Module {
    VGaussianPolicyImpl(int64_t stateDim,
                       int64_t actionDim,
                       int64_t hiddenDim = 128,
                       double logStdMin = -20,
                       double logStdMax = 2)
        :m_logStdMin(logStdMin),
         m_logStdMax(logStdMax)
    {
        vision = register_module("Vision", VisionNetwork(stateDim, 3136));

        linear1 = register_module("Linear1", torch::nn::Linear(3136, hiddenDim));
        linear2 = register_module("Linear2", torch::nn::Linear(hiddenDim, hiddenDim));

        mean_linear = register_module("MeanLinear", torch::nn::Linear(hiddenDim, actionDim));
        log_std_linear = register_module("LogStdLinear", torch::nn::Linear(hiddenDim, actionDim));

        apply(initialize_weights);

        //# action rescaling
        action_scale = torch::ones({1}) * 1.0;
        action_bias = torch::ones({1}) * 0.0;
    }

    auto forward(torch::Tensor state){
        auto x = vision(state);
        x = x.view({x.sizes()[0],-1});
        x = torch::nn::functional::relu(linear1(x));
        x = torch::nn::functional::relu(linear2(x));
        auto mean = mean_linear(x);
        auto log_std = log_std_linear(x);
        log_std = torch::clamp(log_std, m_logStdMin, m_logStdMax);
        return std::make_tuple<>(mean, log_std);
    }

    auto sample(torch::Tensor state, double epsilon=1e-6)
    {
        auto &&[mean, log_std] = this->forward(state);
        auto std = log_std.exp();
        static auto rsample = [](torch::Tensor mean, torch::Tensor std){
            //auto eps = at::normal(mean, std);
            auto eps = at::normal(0, 1, mean.sizes()).to(mean.device());
            eps.set_requires_grad(false);
            return mean + eps * std;
        };
        static auto logSqrt2Pi = torch::zeros({1}).to(mean.device());
        static std::once_flag flag;
        std::call_once(flag, [](){
            logSqrt2Pi[0] = 2*M_PI;
            logSqrt2Pi = torch::log(torch::sqrt(logSqrt2Pi));
        });
        static auto log_prob_func = [](torch::Tensor value, torch::Tensor mean, torch::Tensor std){
            auto var = std.pow(2);
            auto log_scale = std.log();
            return -(value - mean).pow(2) / (2 * var) - log_scale - logSqrt2Pi;
        };
        auto x_t = rsample(mean, std);  // for reparameterization trick (mean + std * N(0,1))
        auto y_t = torch::tanh(x_t);
        auto action = y_t * action_scale + action_bias;
        auto log_prob = log_prob_func(x_t, mean, std);
        // Enforcing Action Bound
        log_prob -= torch::log(action_scale * (1 - y_t.pow(2)) + epsilon);
        log_prob = log_prob.sum(1, true);

        mean = torch::tanh(mean) * action_scale + action_bias;
        return std::make_tuple<>(action, log_prob, mean);
    }

    auto select_action(torch::Tensor state, bool eval = false )
    {
        auto state_ = torch::unsqueeze(state, 0);
        torch::Tensor mean, tmp, action;
        std::tie( action, tmp, mean ) = this->sample(state_);
        if (eval){
            return mean.detach().cpu()[0];
        }
        return action.detach().cpu()[0];
    }

    void to(torch::Device device, bool non_blocking = false) override
    {
        action_scale = action_scale.to(device, non_blocking);
        action_bias = action_bias.to(device, non_blocking);
        torch::nn::Module::to(device, non_blocking);
    }

    torch::nn::Linear linear1 = nullptr;
    torch::nn::Linear linear2 = nullptr;

    VisionNetwork vision = nullptr;

    torch::nn::Linear mean_linear = nullptr;
    torch::nn::Linear log_std_linear = nullptr;

    torch::Tensor action_scale;
    torch::Tensor action_bias;

    double m_logStdMin;
    double m_logStdMax;
};
TORCH_MODULE(VGaussianPolicy);
#endif

struct DeterministicPolicyImpl : torch::nn::Module {
    DeterministicPolicyImpl(int64_t stateDim,
                            int64_t actionDim,
                            int64_t hiddenDim=128)
    {
        linear1 = register_module("Linear1", torch::nn::Linear(stateDim, hiddenDim));
        linear2 = register_module("Linear2", torch::nn::Linear(hiddenDim, hiddenDim));

        mean_linear = register_module("MeanLinear", torch::nn::Linear(hiddenDim, actionDim));
        noise = torch::randn({actionDim});

        apply(initialize_weights);

        //# action rescaling
        action_scale = torch::ones({1});
        action_bias = torch::zeros({1});
    }

    torch::Tensor forward(torch::Tensor state){
        auto x = torch::nn::functional::relu(linear1(state));
        x = torch::nn::functional::relu(linear2(x));
        auto mean = torch::tanh(mean_linear(x)) * action_scale + action_bias;
        return mean;
    }

    auto sample(torch::Tensor state){
        auto mean = this->forward(state);
        auto noise_ = noise.normal_(0., 0.1);
        noise_ = noise_.clamp(-0.25, 0.25);
        auto action = mean + noise_;
        return std::make_tuple<>(action, torch::zeros({1}), mean);
    }

    auto select_action(torch::Tensor state, bool eval = false ){
        auto state_ = torch::unsqueeze(state.clone(), 0);
        torch::Tensor mean, tmp, action;
        std::tie( action, tmp, mean ) = this->sample(state_);
        if (eval){
            return mean.detach().cpu()[0];
        }
        return action.detach().cpu()[0];
    }

    void to(torch::Device device, bool non_blocking = false) override
    {
        action_scale = action_scale.to(device,non_blocking);
        action_bias = action_bias.to(device,non_blocking);
        noise = noise.to(device,non_blocking);
        return torch::nn::Module::to(device,non_blocking);
    }

    torch::nn::Linear linear1 = nullptr;
    torch::nn::Linear linear2 = nullptr;

    torch::nn::Linear mean_linear = nullptr;
    torch::Tensor noise;

    torch::Tensor action_scale;
    torch::Tensor action_bias;
};
TORCH_MODULE(DeterministicPolicy);

using dType = std::tuple<torch::Tensor,
                        torch::Tensor,
                        torch::Tensor,
                        torch::Tensor,
                        torch::Tensor>;

using ReplayMemory = ReplayBuffer<dType>;

struct RL_SAC::member {
    bool eval = false;
    int act_dim = 1;
    int state_dim = 4;
    int hidden_size = 256;
    double gamma = 0.99;
    double tau = 0.005;
    double lr = 0.0002;
    torch::Tensor alpha;
    int seed = 123456;
    int batch_size = 256;
    int num_steps = 1000000;
    int start_steps = 255;//2001;
    int update_per_step = 1;
    int target_update_interval = 1;
    int replay_size = 50000;

    const int max_episode_steps = 500;
    const int save_step = 5000;

    const std::string policy_type = "Gaussian";   //Deterministic

    std::shared_ptr<torch::Device> device;
    std::string trial_path;

    bool auto_entropy_tuning = true;
    double target_entropy = 0.98;
    torch::Tensor log_alpha = torch::zeros({1});
    std::shared_ptr<torch::optim::Adam> alpha_optim;

#ifdef WITH_VISION_TEST
    BiQNetwork critic = nullptr;
    BiQNetwork critic_target = nullptr;
    GaussianPolicy policy = nullptr;
    VisionNetwork vision = nullptr;
#elif defined (WITH_VISION)
    BiVQNetwork critic = nullptr;
    BiVQNetwork critic_target = nullptr;
    VGaussianPolicy policy = nullptr;
#else
    BiQNetwork critic = nullptr;
    BiQNetwork critic_target = nullptr;
    GaussianPolicy policy = nullptr;
#endif
    std::shared_ptr<torch::optim::Adam> critic_optim;
    std::shared_ptr<torch::optim::Adam> policy_optim;

    std::shared_ptr<Gym_Torch> env;
    dType train_internal(ReplayMemory &memory, int batch_size, int updates);
    void save_checkpoint(const std::string &prefix );
    void load_checkpoint(const std::string &prefix );
};

RL_SAC::RL_SAC()
    : m(std::make_shared<member>())
{
    torch::manual_seed(time(nullptr));
    m->device = std::make_shared<torch::Device>(torch::cuda::is_available() ? "cuda" : "cpu");

    //Environment
#ifdef WITH_VISION_TEST
    m->env = std::make_shared< CartPole_ContinuousVision>(true);
    const int vision_outDim = m->env->state_dimension();

    m->state_dim = vision_outDim;
    m->act_dim = m->env->action_dimension();

    m->vision = VisionNetwork(std::sqrt(m->state_dim / 2 / 2), -1);   //Assume aspect ration = 1 and 4 channels
    m->vision->to(*m->device);

    if ( m->policy_type == "Gaussian" ) {
        m->policy = GaussianPolicy(3136, m->act_dim, m->hidden_size);

        m->alpha = torch::ones({1}) * 0.2;

        if ( m->auto_entropy_tuning ) {
            m->target_entropy = -torch::prod(torch::rand({m->act_dim}).to(*m->device)).item().toDouble();
            m->alpha = torch::ones({1});
            m->log_alpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(*m->device));
        }
    } else {
        std::exit(1);
        m->alpha = torch::zeros({1});
//        m->policy = DeterministicPolicy(m->state_dim, m->act_dim, m->hidden_size);
//        m->policy_optim = std::make_shared<torch::optim::Adam>(m->policy->parameters(), m->lr);
    }

    m->critic = BiQNetwork(3136, m->act_dim, m->hidden_size);
    m->critic_target = BiQNetwork(3136, m->act_dim, m->hidden_size);
#elif defined (WITH_VISION)
    m->env = std::make_shared< CartPole_ContinuousVision>(true);
    const int vision_outDim = m->env->state_dimension();

    m->state_dim = vision_outDim;
    m->act_dim = m->env->action_dimension();

//    m->vision = VisionNetwork(std::sqrt(envStateDim / 2 / 2), vision_outDim);   //Assume aspect ration = 1 and 4 channels
//    m->vision_optim = std::make_shared<torch::optim::Adam>(m->vision->parameters(), m->lr);
//    m->vision->to(*m->device);

    if ( m->policy_type == "Gaussian" ) {
        m->policy = VGaussianPolicy(std::sqrt(m->state_dim / 2 / 2), m->act_dim, m->hidden_size);

        m->alpha = torch::ones({1}) * 0.2;

        if ( m->auto_entropy_tuning ) {
            m->target_entropy = -torch::prod(torch::rand({m->act_dim}).to(*m->device)).item().toDouble();
            m->log_alpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(*m->device));
        }
    } else {
        std::exit(1);
        m->alpha = torch::zeros({1});
//        m->policy = DeterministicPolicy(m->state_dim, m->act_dim, m->hidden_size);
//        m->policy_optim = std::make_shared<torch::optim::Adam>(m->policy->parameters(), m->lr);
    }

    m->critic = BiVQNetwork(std::sqrt(m->state_dim / 2 / 2), m->act_dim, m->hidden_size);
    m->critic_target = BiVQNetwork(std::sqrt(m->state_dim / 2 / 2), m->act_dim, m->hidden_size);
#else
    m->env = std::make_shared< CartPole_Continuous>(true);
    m->state_dim = m->env->state_dimension();
    m->act_dim = m->env->action_dimension();

    if ( m->policy_type == "Gaussian" ) {
        m->policy = GaussianPolicy(m->state_dim, m->act_dim, m->hidden_size);

        m->alpha = torch::ones({1}) * 0.2;

        if ( m->auto_entropy_tuning ) {
            m->target_entropy = -torch::prod(torch::rand({m->act_dim}).to(*m->device)).item().toDouble();
            m->log_alpha = torch::zeros({1}, torch::TensorOptions().requires_grad(true).device(*m->device));
        }
    } else {
        std::exit(1);
        m->alpha = torch::zeros({1});
//        m->policy = DeterministicPolicy(m->state_dim, m->act_dim, m->hidden_size);
//        m->policy_optim = std::make_shared<torch::optim::Adam>(m->policy->parameters(), m->lr);
    }

    m->critic = BiQNetwork(m->state_dim, m->act_dim, m->hidden_size);
    m->critic_target = BiQNetwork(m->state_dim, m->act_dim, m->hidden_size);
#endif

    std::cout << "StateDim : " << m->state_dim << std::endl;
    std::cout << "ActDim : " << m->act_dim << std::endl;

    auto &&target_params = m->critic_target->parameters();
    auto &&params = m->critic->parameters();
    const auto n = int(params.size());

    for (int i=0; i<n; ++i ) {
        const auto &v = params.at(i);
        auto &tv = target_params.at(i);
        tv.data().copy_(v.data());
    }


    m->critic->to(*m->device);
    m->critic_target->to(*m->device);
    m->policy->to(*m->device);
    m->alpha = m->alpha.to(*m->device);
}

RL_SAC::~RL_SAC()
{
    m.reset();
}

void RL_SAC::train(double lr, int batch_size, std::function<void (int, double, double, double, double)> *cb_stat,
                   std::function<void (int, std::vector<double>, std::vector<double>)> *cb_anim)
{
    try {
        std::random_device randev;
        std::mt19937 mt(randev());
        std::uniform_int_distribution<int> dist(1,10000);


        //torch::autograd::DetectAnomalyGuard guard;

        char stime[64];
        auto time = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(time);
    #ifdef WIN32
                auto err = ctime_s(stime, 64*sizeof(char), &end_time);
                if ( err != 0 ) {
                    std::cout << "Get current time error" << std::endl;
                }

    #else
                std::string tmp = ctime(&end_time);
                strcpy(stime, tmp.data());
    #endif

        m->trial_path.resize(128);
        int cap = sprintf(m->trial_path.data(), "runs/%s", stime);
        m->trial_path.resize(cap);
        m->trial_path.shrink_to_fit();
        m->trial_path = std::regex_replace(m->trial_path, std::regex(":"), "-");
        m->trial_path = std::regex_replace(m->trial_path, std::regex(" "), "_");
        m->trial_path = std::regex_replace(m->trial_path, std::regex("\n+"), "");

        std::cout << m->trial_path << std::endl;
        std::filesystem::create_directories(m->trial_path);

        m->lr = lr;
        m->batch_size = batch_size;

#ifdef WITH_VISION_TEST
        std::vector<torch::optim::OptimizerParamGroup> vgroup = {m->vision->parameters(),
                                                                m->critic->parameters()};
        std::vector<torch::optim::OptimizerParamGroup> pgroup = {m->vision->parameters(),
                                                                m->policy->parameters()};
#else
        std::vector<torch::optim::OptimizerParamGroup> vgroup = {m->critic->parameters()};
        std::vector<torch::optim::OptimizerParamGroup> pgroup = {m->policy->parameters()};
#endif
        m->critic_optim = std::make_shared<torch::optim::Adam>(vgroup, m->lr);
        m->policy_optim = std::make_shared<torch::optim::Adam>(pgroup, m->lr);
        m->alpha_optim = std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>{m->log_alpha}, m->lr);

        //m->load_checkpoint("chkpt");

        // Training Loop
        auto total_numsteps = 0;
        auto updates = 0;
        auto i_episode = 0;

        auto memory = ReplayMemory(m->replay_size);

        std::vector<double> pos, ang;
        const int dim = m->env->action_dimension();

        int last_save_id = -1;

        while ( true ) {
            auto episode_reward = 0.0;
            auto episode_policy_loss = 0.0;
            auto episode_critic_loss = 0.0;
            auto episode_entropy_loss = 0.0;
            auto episode_steps = 0;
            auto state = m->env->reset().clone();

            while (true) {
                torch::Tensor action;
                if (m->start_steps > total_numsteps) {
                    action = m->env->sample_action();  // Sample random action
                } else {
                    action = m->policy->select_action(state);  // Sample action from policy
                    //action = torch::clamp(action.round(), 0, 1);
    //                if ( 17 == dist(mt)) {   // 0.01% chance of mutation
    //                    std::cout << " ===== Mutate ===== \n" << std::endl;
    //                    action = 1 - action;
    //                }
                }

                if (int(memory.buffer.size()) >= m->batch_size) {
                    // Number of updates per step in environment
                    for ( int i = 0; i<m->update_per_step; ++i) {
                        // Update parameters of all the networks

                        auto &&[critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha] =
                                m->train_internal(memory, m->batch_size, updates);

                        episode_policy_loss += policy_loss.item().toDouble();
                        episode_critic_loss += 0.5*(critic_1_loss+critic_2_loss).item().toDouble();
                        episode_entropy_loss += ent_loss.item().toDouble();

                        updates += 1;
                    }
                }

                auto &&[next_state, reward, done, _] = m->env->step(action); // Step
                episode_steps += 1;
                total_numsteps += 1;
                episode_reward += reward.item().toDouble();

                // Ignore the "done" signal if it comes from hitting the time horizon.
                // (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                auto mask = 1-done;
                if (episode_steps == m->max_episode_steps) {
                    mask = torch::zeros({1});
                }
                auto mem = std::make_tuple<>( state,
                                              action,
                                              reward,
                                              next_state.clone(),
                                              mask); // Append transition to memory
                memory.push(mem);

                if (cb_anim) {
                    pos.clear(); ang.clear();
                    for ( int i=0; i<dim; ++i ) {
                        pos.push_back(state[i*4].item().toDouble());
                        ang.push_back(state[i*4 + 2].item().toDouble());
                    }
                    (*cb_anim)(i_episode, pos, ang);
                }

                if ( 0 == total_numsteps % m->save_step ) {
                    if ( total_numsteps > m->start_steps ) {
                        m->save_checkpoint(m->trial_path + "//chkpt");
                    }
                    last_save_id = memory.save(m->trial_path + "//chkpt", last_save_id, m->save_step);
                }

                //End episode
                if ( done.item().toInt() ||
                     episode_steps >= m->max_episode_steps ) {
                    break;
                }

                state = std::get<3>(mem);
            }

            if (cb_stat) {
                (*cb_stat)(total_numsteps,
                           episode_reward,
                           episode_policy_loss,
                           episode_critic_loss,
                           episode_entropy_loss);
            }

            if (total_numsteps > m->num_steps)
                break;

            printf("Episode: %6d, total numsteps: %6d, episode steps: %6d, reward: %6.2f\n",
                   ++i_episode, total_numsteps, episode_steps, episode_reward);
        }
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "====== Training Finished ======" << std::endl;
}

void RL_SAC::trainVision(double lr, int batch_size, std::function<void (int, double, double, double, double)> *cb_stat,
                         std::function<std::pair<int,int> (std::vector<double>,
                                             std::vector<double>,
                                             std::vector<unsigned int>&)> *cb_anim)
{
    try {
        std::random_device randev;
        std::mt19937 mt(randev());
        std::uniform_int_distribution<int> dist(1,10000);

        //torch::autograd::DetectAnomalyGuard guard;
        char stime[64];
        auto time = std::chrono::system_clock::now();
        std::time_t end_time = std::chrono::system_clock::to_time_t(time);
#ifdef WIN32
        auto err = ctime_s(stime, 64*sizeof(char), &end_time);
        if ( err != 0 ) {
            std::cout << "Get current time error" << std::endl;
        }
#else
        std::string tmp = ctime(&end_time);
        strcpy(stime, tmp.data());
#endif

        m->trial_path.resize(128);
        int cap = sprintf(m->trial_path.data(), "runs/%s", stime);
        m->trial_path.resize(cap);
        m->trial_path.shrink_to_fit();
        m->trial_path = std::regex_replace(m->trial_path, std::regex(":"), "-");
        m->trial_path = std::regex_replace(m->trial_path, std::regex(" "), "_");
        m->trial_path = std::regex_replace(m->trial_path, std::regex("\n+"), "");

        std::cout << m->trial_path << std::endl;
        std::filesystem::create_directories(m->trial_path);

        save_path = m->trial_path;

        m->lr = lr;
        m->batch_size = batch_size;

#ifdef WITH_VISION_TEST
        std::vector<torch::optim::OptimizerParamGroup> vgroup = {m->vision->parameters(),
                                                                m->critic->parameters()};
        std::vector<torch::optim::OptimizerParamGroup> pgroup = {m->vision->parameters(),
                                                                m->policy->parameters()};
#else
        std::vector<torch::optim::OptimizerParamGroup> vgroup = {m->critic->parameters()};
        std::vector<torch::optim::OptimizerParamGroup> pgroup = {m->policy->parameters()};
#endif
        m->critic_optim = std::make_shared<torch::optim::Adam>(vgroup, m->lr);
        m->policy_optim = std::make_shared<torch::optim::Adam>(m->policy->parameters(), m->lr);
        m->alpha_optim = std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>{m->log_alpha}, m->lr);

        m->load_checkpoint("runs/Thu_Mar_10_13-28-13_2022/chkpt");

        auto env = std::static_pointer_cast< CartPole_ContinuousVision>(m->env);
        env->setRender_Callback(cb_anim);

        // Training Loop
        auto total_numsteps = 0;
        auto updates = 0;
        auto i_episode = 0;
        auto last_save_id = 0;

        auto memory = ReplayMemory(m->replay_size);

        std::vector<double> pos, ang;
        const int dim = env->action_dimension();

        std::cout << "Action Dim : " << dim << " "
                  << "State Dim : " << env->state_dimension()
                  << std::endl;

        while ( true ) {
            auto episode_reward = 0.0;
            auto episode_policy_loss = 0.0;
            auto episode_critic_loss = 0.0;
            auto episode_entropy_loss = 0.0;
            auto episode_steps = 0;
            auto state = env->reset().clone();

            while (true) {
                torch::Tensor action;
                if (m->start_steps > total_numsteps) {
                    action = env->sample_action();  // Sample random action
                } else {
#ifdef WITH_VISION_TEST
                    auto vstate = m->vision(state.to(*m->device));
                    vstate = vstate.view({-1});
                    action = m->policy->select_action(vstate);  // Sample action from policy
#else
                    action = m->policy->select_action(state.to(*m->device));  // Sample action from policy
#endif
                    //action = torch::clamp(action.round(), 0, 1);
//                  if ( 17 == dist(mt)) {   // 0.01% chance of mutation
//                      std::cout << " ===== Mutate ===== \n" << std::endl;
//                      action = 1 - action;
//                  }
                }

                if (int(memory.buffer.size()) >= m->batch_size) {
                    // Number of updates per step in environment
                    for ( int i = 0; i<m->update_per_step; ++i) {
                        // Update parameters of all the networks

                        auto &&[critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha] =
                                m->train_internal(memory, m->batch_size, updates);

                        episode_policy_loss += policy_loss.item().toDouble();
                        episode_critic_loss += 0.5*(critic_1_loss+critic_2_loss).item().toDouble();
                        episode_entropy_loss += ent_loss.item().toDouble();

                        updates += 1;
                    }
                }

                auto &&[next_state, reward, done, _] = env->step(action); // Step

                episode_steps += 1;
                total_numsteps += 1;
                episode_reward += reward.item().toDouble();

                step_counters = total_numsteps;

                // Ignore the "done" signal if it comes from hitting the time horizon.
                // (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                auto mask = 1-done;
                if (episode_steps == m->max_episode_steps) {
                    mask = torch::zeros({1});
                }
                auto mem = std::make_tuple<>( state.to(*m->device),
                                              action.to(*m->device),
                                              reward.to(*m->device),
                                              next_state.clone().to(*m->device),
                                              mask.to(*m->device) ); // Append transition to memory
                memory.push(mem);

    //            if (cb_anim) {
    //                pos.clear(); ang.clear();
    //                for ( int i=0; i<dim; ++i ) {
    //                    pos.push_back(state[i*4].item().toDouble());
    //                    ang.push_back(state[i*4 + 2].item().toDouble());
    //                }
    //                (*cb_anim)(i_episode, pos, ang);
    //            }

                if ( 0 == total_numsteps % m->save_step ) {
                    if ( total_numsteps > m->start_steps ) {
                        m->save_checkpoint(m->trial_path + "/chkpt");
                    }
#ifdef WITH_VISION_TEST
                    m->vision->m_saveOnce = true;
#elif defined (WITH_VISION)
                    m->policy->vision->m_saveOnce = true;
#endif
                    //last_save_id = memory.save(m->trial_path + "/memory/", last_save_id, m->save_step);
                }

                //End episode
                if ( done.item().toInt() ||
                     episode_steps >= m->max_episode_steps ) {
                    break;
                }
                state = std::get<3>(mem);
            }

            if (cb_stat) {
                (*cb_stat)(total_numsteps,
                           episode_reward,
                           episode_policy_loss,
                           episode_critic_loss,
                           episode_entropy_loss);
            }

            if (total_numsteps > m->num_steps)
                break;

            printf("Episode: %6d, total numsteps: %6d, episode steps: %6d, reward: %6.2f\n",
                   ++i_episode, total_numsteps, episode_steps, episode_reward);
        }
        std::cout << "====== Training Finished ======" << std::endl;
    }  catch (const std::exception &e) {
        std::cout << "[ Error ]\n"
                  << e.what() << std::endl;
    }
}

void RL_SAC::eval(const std::string &checkPoint, std::function<bool ()> *noiseSignal,
                  std::function<bool (int, std::vector<double>, std::vector<double>,std::vector<double>)> *cb_anim)
{
    torch::Tensor state;

    m->critic_optim = std::make_shared<torch::optim::Adam>(m->critic->parameters(), m->lr);
    m->policy_optim = std::make_shared<torch::optim::Adam>(m->policy->parameters(), m->lr);
    m->alpha_optim = std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>{m->log_alpha}, m->lr);

    m->load_checkpoint(checkPoint+"/chkpt");

    std::random_device randev;
    std::mt19937 mt(randev());
    std::uniform_int_distribution<int> dist(1,100);

    m->policy->eval();

    auto avg_reward = 0.;
    const auto episodes = 10;
    bool bRun = true;

    std::vector<double> pos, ang, acts;
    const int dim = m->env->action_dimension();

    int j;
    for ( j=0; j<episodes && bRun; ++j){
        state = m->env->reset().clone().to(*m->device);
        auto episode_reward = 0.0;
        auto episode_steps = 0;

        while (true){
            auto action = m->policy->select_action(state, true);
            if ( noiseSignal && (*noiseSignal)()) {
                double sign = 1.0 - (dist(mt) % 3);
                action += sign * dist(mt) / 50.0;
            }
            //action = torch::clamp(action.round(), 0, 1);

            auto &&[next_state, reward, done, _] = m->env->step(action);
            episode_reward += reward.item().toDouble();
            state = next_state.clone();
            ++episode_steps;

            if (cb_anim) {
                pos.clear(); ang.clear(); acts.clear();
                for ( int i=0; i<dim; ++i ) {
                    pos.push_back(state[i*4].item().toDouble());
                    ang.push_back(state[i*4 + 2].item().toDouble());
                    acts.push_back(action[i].item().toDouble());
                }
                bRun = (*cb_anim)(episode_steps, pos, ang, acts);
            }
            if ( !bRun || done.item().toBool() ||
                   episode_steps >= m->max_episode_steps) {
                break;
            }
        }
        avg_reward += episode_reward;
    }
    avg_reward /= j;

    printf("----------------------------------------");
    printf("Test Episodes: %d, Avg. Reward: %5.2f",
           episodes, avg_reward);
    printf("----------------------------------------\n");
}

void RL_SAC::evalVision(const std::string &checkPoint, std::function<bool ()> *noiseSignal,
                  std::function<bool (int, std::vector<double>,
                                      std::vector<double>,
                                      std::vector<double>)> *cb_anim,
                  std::function<std::pair<int,int> (std::vector<double>,
                                            std::vector<double>,
                                            std::vector<unsigned int>&)> *cb_render)
{
    try {
        torch::Tensor state;
        m->critic_optim = std::make_shared<torch::optim::Adam>(m->critic->parameters(), m->lr);
        m->policy_optim = std::make_shared<torch::optim::Adam>(m->policy->parameters(), m->lr);
        m->alpha_optim = std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>{m->log_alpha}, m->lr);

        m->load_checkpoint(checkPoint+"/chkpt");

        std::random_device randev;
        std::mt19937 mt(randev());
        std::uniform_int_distribution<int> dist(1,100);

        m->policy->eval();

        auto avg_reward = 0.;
        const auto episodes = 10;
        bool bRun = true;

        auto env = std::static_pointer_cast< CartPole_ContinuousVision>(m->env);
        env->setRender_Callback(cb_render);

        std::vector<double> pos, ang, acts;
        const int dim = m->env->action_dimension();

        int j;
        for ( j=0; j<episodes && bRun; ++j){
            state = m->env->reset().clone();
            auto episode_reward = 0.0;
            auto episode_steps = 0;

            while (true){
#ifdef WITH_VISION_TEST
                auto vstate = m->vision(state.to(*m->device));
                vstate = vstate.view({-1});
                auto action = m->policy->select_action(vstate, true);  // Sample action from policy
#else
                auto action = m->policy->select_action(state.to(*m->device), true);
#endif
                if ( noiseSignal && (*noiseSignal)()) {
                    double sign = 1.0 - (dist(mt) % 3);
                    action += sign * dist(mt) / 100.0;
                }
                //action = torch::clamp(action.round(), 0, 1);

                auto &&[next_state, reward, done, _] = m->env->step(action);
                episode_reward += reward.item().toDouble();
                state = next_state.clone();
                ++episode_steps;

                if (cb_anim) {
                    pos.clear(); ang.clear(); acts.clear();
                    for ( int i=0; i<dim; ++i ) {
                        pos.push_back(env->mState[i*4].item().toDouble());
                        ang.push_back(env->mState[i*4 + 2].item().toDouble());
                        acts.push_back(action[i].item().toDouble());
                    }
                    bRun = (*cb_anim)(episode_steps, pos, ang, acts);
                }
                if ( !bRun || done.item().toBool() ||
                       episode_steps >= m->max_episode_steps) {
                    break;
                }
            }
            avg_reward += episode_reward;
        }
        avg_reward /= j;

        printf("----------------------------------------");
        printf("Test Episodes: %d, Avg. Reward: %5.2f",
               episodes, avg_reward);
        printf("----------------------------------------\n");
    }  catch (const std::exception &e ) {
        std::cout << "[ Error ]\n" << e.what() << std::endl;
    }
}

dType RL_SAC::member::train_internal(ReplayMemory& memory, int batch_size, int updates)
{
    // Sample a batch from memory
    auto &&[states, actions, rewards, next_states, masks] =
            memory.sample(batch_size);

    auto state_batch = torch::stack(states);
    auto next_state_batch = torch::stack(next_states);
    auto action_batch = torch::stack(actions);
    auto reward_batch = torch::stack(rewards);
    auto mask_batch = torch::stack(masks);

    torch::Tensor next_state_action, next_state_log_pi, _1;
    torch::Tensor qf1_next_target, qf2_next_target;
    torch::Tensor min_qf_next_target, next_q_value;

#ifdef WITH_VISION_TEST
    state_batch = vision(state_batch);
    next_state_batch = vision(next_state_batch);

    state_batch = state_batch.view({state_batch.sizes()[0],-1});
    next_state_batch = next_state_batch.view({next_state_batch.sizes()[0],-1});
#endif

    {
        torch::NoGradGuard noguard;
        std::tie (next_state_action, next_state_log_pi, _1) = policy->sample(next_state_batch);
        std::tie (qf1_next_target, qf2_next_target) = critic_target(next_state_batch, next_state_action);

        min_qf_next_target = torch::min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi;
        next_q_value = reward_batch + mask_batch * gamma * min_qf_next_target;
    }
    //torch::AutoGradMode autoGrad(true);

    auto &&[qf1, qf2] = critic(state_batch, action_batch);  // Two Q-functions to mitigate positive bias in the policy improvement step
    auto qf1_loss = torch::nn::functional::mse_loss(qf1, next_q_value);  // JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    auto qf2_loss = torch::nn::functional::mse_loss(qf2, next_q_value);  // JQ = 𝔼(st,at)~D[0.5(Q2(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2];
    auto qf_loss = qf1_loss + qf2_loss;

    critic_optim->zero_grad();
    qf_loss.backward();
    critic_optim->step();

    //Detach the vision head for policy since vision is optimized at critic
    state_batch = state_batch.detach();

    auto &&[pi, log_pi, _2] = policy->sample(state_batch);
    auto &&[qf1_pi, qf2_pi] = critic(state_batch, pi);
    auto min_qf_pi = torch::min(qf1_pi, qf2_pi);

    // Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    auto policy_loss = ((alpha * log_pi) - min_qf_pi).mean();

    policy_optim->zero_grad();
    policy_loss.backward();
    policy_optim->step();

    torch::Tensor alpha_loss, alpha_tlogs;
    if (auto_entropy_tuning) {
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean();

        alpha_optim->zero_grad();
        alpha_loss.backward();
        alpha_optim->step();

        alpha = log_alpha.exp();
        alpha_tlogs = alpha.clone(); // For TensorboardX logs
     } else {
        alpha_loss = torch::exp(log_pi).mean();
        alpha_tlogs = alpha;
    }

    if ( updates % target_update_interval == 0 ) {
        auto &&target_params = critic_target->parameters();
        auto &&params = critic->parameters();
        const auto n = int(target_params.size());
        for (int i=0; i<n; ++i ) {
            auto &v = params.at(i);
            auto &tv = target_params.at(i);
            tv.data().copy_(v * tau + tv*(1.0-tau));
        }
    }

    return std::make_tuple<>(qf1_loss, qf2_loss, policy_loss,
                             alpha_loss, alpha_tlogs);
}

void RL_SAC::member::save_checkpoint(const std::string &prefix)
{
    std::cout << prefix << std::endl;
    torch::save(policy, prefix + "_policy.pt");
    torch::save(critic, prefix + "_critic.pt");
    torch::save(critic_target, prefix + "_critic_target.pt");
    torch::save(*policy_optim, prefix + "_policy_optim.pt");
    torch::save(*critic_optim, prefix + "_critic_optim.pt");
#ifdef WITH_VISION_TEST
    torch::save(vision, prefix + "_vision.pt");
#endif
    std::cout << "Checkpoint saved " << std::endl;

//    constexpr auto description = "Model_Description.txt";

//    std::ofstream out;
//    out.open(description, std::ios::out);
//    if ( !out.is_open()) {
//        return;
//    }
//    policy->pretty_print(out);
//    critic->pretty_print(out);
//    critic_target->pretty_print(out);
//    out.close();
}

void RL_SAC::member::load_checkpoint(const std::string &prefix)
{
    if ( !std::filesystem::exists(prefix + "_policy.pt") ||
         !std::filesystem::exists(prefix + "_critic.pt") ||
         !std::filesystem::exists(prefix + "_critic_target.pt") ||
         !std::filesystem::exists(prefix + "_policy_optim.pt") ||
         !std::filesystem::exists(prefix + "_critic_optim.pt")) {
        std::cout << "Incomplete pretrain checkpoints !" << std::endl;
        return;
    }
    std::cout << "Loading checkpoints : " << prefix << std::endl;
    torch::load(policy, prefix + "_policy.pt");
    torch::load(critic, prefix + "_critic.pt");
    torch::load(critic_target, prefix + "_critic_target.pt");
#ifdef WITH_VISION_TEST
    torch::load(vision, prefix + "_vision.pt");
#endif
    //torch::load(*policy_optim, prefix + "_policy_optim.pt");
    //torch::load(*critic_optim, prefix + "_critic_optim.pt");
}
}
