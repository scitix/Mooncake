#include <gtest/gtest.h>
#include <glog/logging.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "client_service.h"
#include "default_config.h"
#include "master_metric_manager.h"
#include "test_server_helpers.h"
#include "utils.h"

// Reconnect leak-slope regression tests for the worker->master client path.
//
// Background: when the master goes away and comes back (HA failover, or a
// restart on the same address), the long-lived MasterClient must re-establish
// its coro_rpc connections to the master. This must NOT leak per-cycle
// operating-system resources -- neither file descriptors (sockets / eventfd /
// timerfd / epoll) nor OS threads. A regression here is exactly the class of
// bug tracked by issue #588 (reconnect FD/thread accumulation).
//
// Methodology: drive many master stop/restart cycles, sample the live FD count
// (/proc/self/fd) and thread count (/proc/self/status Threads:) once per cycle,
// and compare the late-window average against the early-window average. A real
// per-reconnect leak shows up as a monotonically rising count; transient
// warm-up allocations are tolerated by comparing window averages rather than
// first-vs-last samples.
//
// Two diagnostic controls accompany the main guard so the attribution is not
// ambiguous:
//   * MasterRestartOnlyControl  -- restart the master with NO client, isolating
//                                  server-side restart cost.
//   * ZeroSegmentReconnect      -- a connected client that mounts NO segment,
//                                  exercising the bare ping/reconnect pool path
//                                  without the remount work that runs only for
//                                  clients that own a segment.
// Both controls are expected to be perfectly flat, which is what lets the main
// guard attribute any growth specifically to the worker->master reconnect.

namespace mooncake {
namespace testing {

namespace {

// Live open file descriptors for this process.
long CountOpenFds() {
    long n = 0;
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator("/proc/self/fd", ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        ++n;
    }
    return n;
}

// Live OS thread count for this process, read from /proc/self/status.
long CountThreads() {
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long threads = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::sscanf(line, "Threads: %ld", &threads) == 1) {
            break;
        }
    }
    std::fclose(f);
    return threads;
}

double Mean(const std::vector<long>& v, size_t begin, size_t end) {
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += static_cast<double>(v[i]);
    return sum / static_cast<double>(end - begin);
}

}  // namespace

// Main guard: a connected client (no mounted segment) survives repeated master
// restarts and reconnects each time. Asserts the worker->master reconnect path
// holds FD/thread counts flat across cycles.
TEST(ReconnectLeakSlopeTest, ReconnectDoesNotLeakFdOrThreads) {
    InProcMaster master;
    ASSERT_TRUE(master.Start(InProcMasterConfigBuilder().build()));

    const std::string local_hostname = "127.0.0.1:18021";
    std::string master_addr = master.master_address();

    auto client_opt = Client::Create(local_hostname, "P2PHANDSHAKE", "tcp",
                                     std::nullopt, master_addr);
    ASSERT_TRUE(client_opt.has_value());
    auto client = client_opt.value();

    const int rpc_port = master.rpc_port();
    const int metrics_port = master.http_metrics_port();

    const int kCycles = 12;
    std::vector<long> fd_samples;
    std::vector<long> thread_samples;
    fd_samples.reserve(kCycles);
    thread_samples.reserve(kCycles);

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        // Stop master; let ping failures and reconnect attempts accumulate.
        master.Stop();
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Restart on the SAME ports so the client reconnects to it.
        ASSERT_TRUE(master.Start(InProcMasterConfigBuilder()
                                     .set_rpc_port(rpc_port)
                                     .set_http_metrics_port(metrics_port)
                                     .build()));

        // Give the client time to detect recovery and reconnect (the heartbeat
        // pings on a ~1s cadence and reconnects after a few failures).
        std::this_thread::sleep_for(std::chrono::seconds(1));

        long fds = CountOpenFds();
        long threads = CountThreads();
        fd_samples.push_back(fds);
        thread_samples.push_back(threads);
        LOG(INFO) << "cycle=" << cycle << " fds=" << fds
                  << " threads=" << threads;
    }

    const size_t third = static_cast<size_t>(kCycles) / 3;
    ASSERT_GE(third, 1u);
    double fd_early = Mean(fd_samples, 0, third);
    double fd_late = Mean(fd_samples, fd_samples.size() - third,
                          fd_samples.size());
    double thr_early = Mean(thread_samples, 0, third);
    double thr_late = Mean(thread_samples, thread_samples.size() - third,
                           thread_samples.size());

    LOG(INFO) << "FD early-mean=" << fd_early << " late-mean=" << fd_late
              << " delta=" << (fd_late - fd_early);
    LOG(INFO) << "Thread early-mean=" << thr_early << " late-mean=" << thr_late
              << " delta=" << (thr_late - thr_early);

    // A genuine per-reconnect leak (e.g. a leaked io_context with its driver
    // thread + eventfd/timerfd/epoll) would add several FDs and a thread on
    // every cycle. Allow a tiny slack for transient sockets/timers captured
    // mid-cycle.
    EXPECT_LE(fd_late - fd_early, 4.0)
        << "FD count grows across reconnect cycles -> suspected leak";
    EXPECT_LE(thr_late - thr_early, 1.0)
        << "Thread count grows across reconnect cycles -> suspected leak";
}

// CONTROL: restart the in-process master N times with NO client connected.
// Isolates server-side (coro_rpc_server / admin server) restart cost. Expected
// to be flat; if it were NOT, per-cycle growth could not be attributed to the
// client.
TEST(ReconnectLeakSlopeTest, MasterRestartOnlyControl) {
    InProcMaster master;
    ASSERT_TRUE(master.Start(InProcMasterConfigBuilder().build()));
    const int rpc_port = master.rpc_port();
    const int metrics_port = master.http_metrics_port();

    const int kCycles = 12;
    std::vector<long> fd_samples;
    std::vector<long> thread_samples;
    for (int cycle = 0; cycle < kCycles; ++cycle) {
        master.Stop();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        ASSERT_TRUE(master.Start(InProcMasterConfigBuilder()
                                     .set_rpc_port(rpc_port)
                                     .set_http_metrics_port(metrics_port)
                                     .build()));
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        fd_samples.push_back(CountOpenFds());
        thread_samples.push_back(CountThreads());
        LOG(INFO) << "[control] cycle=" << cycle << " fds=" << fd_samples.back()
                  << " threads=" << thread_samples.back();
    }
    const size_t third = static_cast<size_t>(kCycles) / 3;
    double fd_delta = Mean(fd_samples, fd_samples.size() - third,
                           fd_samples.size()) -
                      Mean(fd_samples, 0, third);
    double thr_delta = Mean(thread_samples, thread_samples.size() - third,
                            thread_samples.size()) -
                       Mean(thread_samples, 0, third);
    LOG(INFO) << "[control] FD delta=" << fd_delta
              << " Thread delta=" << thr_delta;
    EXPECT_LE(fd_delta, 4.0) << "master restart leaks FDs";
    EXPECT_LE(thr_delta, 1.0) << "master restart leaks threads";
}

// DIAGNOSTIC: a connected, zero-segment client across master restarts. This is
// the bare ping/reconnect pool path with no remount work. Expected flat.
TEST(ReconnectLeakSlopeTest, ZeroSegmentReconnect) {
    InProcMaster master;
    ASSERT_TRUE(master.Start(InProcMasterConfigBuilder().build()));
    const std::string local_hostname = "127.0.0.1:18031";
    std::string master_addr = master.master_address();
    auto client_opt = Client::Create(local_hostname, "P2PHANDSHAKE", "tcp",
                                     std::nullopt, master_addr);
    ASSERT_TRUE(client_opt.has_value());
    auto client = client_opt.value();

    const int rpc_port = master.rpc_port();
    const int metrics_port = master.http_metrics_port();
    const int kCycles = 12;
    std::vector<long> fd_samples, thread_samples;
    for (int cycle = 0; cycle < kCycles; ++cycle) {
        master.Stop();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        ASSERT_TRUE(master.Start(InProcMasterConfigBuilder()
                                     .set_rpc_port(rpc_port)
                                     .set_http_metrics_port(metrics_port)
                                     .build()));
        std::this_thread::sleep_for(std::chrono::seconds(1));
        fd_samples.push_back(CountOpenFds());
        thread_samples.push_back(CountThreads());
        LOG(INFO) << "[zeroseg] cycle=" << cycle
                  << " fds=" << fd_samples.back()
                  << " threads=" << thread_samples.back();
    }
    const size_t third = static_cast<size_t>(kCycles) / 3;
    double fd_delta = Mean(fd_samples, fd_samples.size() - third,
                           fd_samples.size()) -
                      Mean(fd_samples, 0, third);
    double thr_delta = Mean(thread_samples, thread_samples.size() - third,
                            thread_samples.size()) -
                       Mean(thread_samples, 0, third);
    LOG(INFO) << "[zeroseg] FD delta=" << fd_delta
              << " Thread delta=" << thr_delta;
    EXPECT_LE(fd_delta, 4.0) << "zero-segment reconnect leaks FDs";
    EXPECT_LE(thr_delta, 1.0) << "zero-segment reconnect leaks threads";
}

}  // namespace testing
}  // namespace mooncake

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    mooncake::init_ylt_log_level();
    return RUN_ALL_TESTS();
}
