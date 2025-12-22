#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"
#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"
#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace terekhov_d_fast_sort_batch {

class TerekhovDFastSortBatchPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kElementCount = 75000000;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    std::vector<int> data_vec(kElementCount);
    for (int i = 0; i < kElementCount; i++) {
      data_vec[i] = kElementCount - i - 1;
    }
    input_data_ = data_vec;
    std::sort(data_vec.begin(), data_vec.end());
    res_ = data_vec;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return res_ == output_data;
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
