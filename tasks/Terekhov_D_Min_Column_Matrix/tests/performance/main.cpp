#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"
#include "Terekhov_D_Min_Column_Matrix/mpi/include/ops_mpi.hpp"
#include "Terekhov_D_Min_Column_Matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace terekhov_d_a_test_task_processes {

class MinColumnRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr int kRows = 15000;
    constexpr int kCols = 15000;

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(1, 1000000);

    input_data_.resize(kRows);
    for (int i = 0; i < kRows; ++i) {
      input_data_[i].resize(kCols);
      for (int j = 0; j < kCols; ++j) {
        input_data_[i][j] = dist(gen);
      }
    }

    expected_.assign(kCols, std::numeric_limits<int>::max());
    for (int i = 0; i < kRows; ++i) {
      for (int j = 0; j < kCols; ++j) {
        expected_[j] = std::min(input_data_[i][j], expected_[j]);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (output_data.size() != expected_.size()) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

TEST_P(MinColumnRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TerekhovDTestTaskMPI, TerekhovDTestTaskSEQ>(
    PPC_SETTINGS_Terekhov_D_Min_Column_Matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MinColumnRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MinColumnRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace terekhov_d_a_test_task_processes
