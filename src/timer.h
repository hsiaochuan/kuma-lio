#pragma once
#include <chrono>
namespace faster_lio {
class Timer {
public:
    Timer();
    void Restart();
    double ElapsedMicroSeconds(bool restart = false);
    double ElapsedSeconds(bool restart = false);
    double ElapsedMiniSeconds(bool restart = false);

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

}  // namespace faster_lio