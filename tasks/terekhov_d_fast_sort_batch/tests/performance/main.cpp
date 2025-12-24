#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"
#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"
#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace terekhov_d_fast_sort_batch {

class TerekhovDFastSortBatchPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  void SetUp() override {
    int size = 75000000;
    input_data_.resize(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000000);

    for (int i = 0; i < size; i++) {
      input_data_[i] = dist(gen);
    }

    input_data_[0] = -500000;
    input_data_[size - 1] = 1500000;
    input_data_[size / 2] = 0;

    for (int i = 1; i <= 150; i++) {
      input_data_[(size / 3) + i] = 888888;
    }

    for (int i = 1; i <= 100; i++) {
      input_data_[(2 * size / 3) + i] = 333333;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != input_data_.size()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); i++) {
      if (output_data[i] < output_data[i - 1]) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TerekhovDFastSortBatchPerfTests, BatcherSortPerformance) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TerekhovDFastSortBatchMPI, TerekhovDFastSortBatchSEQ>(
    PPC_SETTINGS_terekhov_d_fast_sort_batch);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TerekhovDFastSortBatchPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(BatcherSortPerf, TerekhovDFastSortBatchPerfTests, kGtestValues, kPerfTestName);

}  // namespace terekhov_d_fast_sort_batch
