#include "timer.h"
using namespace faster_lio;
Timer::Timer() { start_time_ = std::chrono::high_resolution_clock::now(); }
void Timer::Restart() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

double Timer::ElapsedMicroSeconds(bool restart) {
    auto time_elapse =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time_)
            .count();
    if (restart) start_time_ = std::chrono::high_resolution_clock::now();
    return time_elapse;
}

double Timer::ElapsedSeconds(bool restart) {
    return ElapsedMicroSeconds(restart) / 1e6;
}
double Timer::ElapsedMiniSeconds(bool restart) {
    return ElapsedMicroSeconds(restart) / 1e3;
}