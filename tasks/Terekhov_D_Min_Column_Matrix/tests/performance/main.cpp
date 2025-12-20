#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <random>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"
#include "Terekhov_D_Min_Column_Matrix/mpi/include/ops_mpi.hpp"
#include "Terekhov_D_Min_Column_Matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace terekhov_d_a_test_task_processes {

class MinColumnRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    constexpr int kMatrixSizeRows = 15000;
    constexpr int kMatrixSizeCols = 15000;
    const int rows = kMatrixSizeRows;
    const int cols = kMatrixSizeCols;

    constexpr unsigned int kSeedValue = 42U;
    std::seed_seq seed_seq{kSeedValue};
    std::mt19937 gen(seed_seq);
    std::uniform_int_distribution<int> dist(1, 1000000);

    input_data_.resize(rows);
    expected_.assign(cols, INT_MAX);

    for (int i = 0; i < rows; ++i) {
      input_data_[i].resize(cols);
      for (int j = 0; j < cols; ++j) {
        const int val = dist(gen);
        input_data_[i][j] = val;
        expected_[j] = std::min(val, expected_[j]);
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (output_data.empty()) {
      return true;
    }

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

  InType GetTestInputData() override {
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
