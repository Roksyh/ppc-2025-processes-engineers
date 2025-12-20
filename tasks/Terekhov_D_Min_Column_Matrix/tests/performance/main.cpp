#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <iostream>
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

    constexpr int kFixedSeed = 42;
    std::mt19937 gen(kFixedSeed);
    std::uniform_int_distribution<int> dist(1, 1000000);

    input_data_.resize(rows);
    expected_.assign(cols, INT_MAX);

    for (int i = 0; i < rows; ++i) {
      input_data_[i].resize(cols);
      for (int j = 0; j < cols; ++j) {
        int val = dist(gen);
        input_data_[i][j] = val;
        if (val < expected_[j]) {
          expected_[j] = val;
        }
      }
    }

#ifdef DEBUG_OUTPUT
    std::cout << "Generated matrix: " << rows << "x" << cols << std::endl;
    std::cout << "Expected size: " << expected_.size() << std::endl;
    if (!expected_.empty()) {
      std::cout << "First few expected values: ";
      for (size_t i = 0; i < std::min(expected_.size(), size_t(5)); ++i) {
        std::cout << expected_[i] << " ";
      }
      std::cout << std::endl;
    }
#endif
  }

  bool CheckTestOutputData(OutType &output_data) override {
    if (output_data.empty()) {
#ifdef DEBUG_OUTPUT
      std::cout << "ERROR: Output data is empty!" << std::endl;
#endif
      return false;
    }

    if (output_data.size() != expected_.size()) {
#ifdef DEBUG_OUTPUT
      std::cout << "ERROR: Size mismatch! Output size: " << output_data.size()
                << ", Expected size: " << expected_.size() << std::endl;
#endif
      return false;
    }

    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_[i]) {
#ifdef DEBUG_OUTPUT
        if (i == 0) {
          std::cout << "ERROR at position " << i << ": Output = " << output_data[i] << ", Expected = " << expected_[i]
                    << std::endl;
        }
#endif
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
