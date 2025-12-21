#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"
#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"
#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace terekhov_d_fast_sort_batch {

namespace {

void QuickSort(InType *vec) {
  if (vec->empty()) return;
  
  auto quick_sort_recursive = [](auto& self, std::vector<int>& a, int left, int right) -> void {
    if (left >= right) return;
    
    int pivot = a[(left + right) / 2];
    int i = left, j = right;
    
    while (i <= j) {
      while (a[i] < pivot) ++i;
      while (a[j] > pivot) --j;
      if (i <= j) {
        std::swap(a[i], a[j]);
        ++i;
        --j;
      }
    }
    
    self(self, a, left, j);
    self(self, a, i, right);
  };
  
  quick_sort_recursive(quick_sort_recursive, *vec, 0, static_cast<int>(vec->size()) - 1);
}

}  // namespace

class FastSortBatchRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    constexpr std::size_t kSize = 200000;
    input_data_.resize(kSize);

    int x = 17;
    for (std::size_t i = 0; i < kSize; ++i) {
      x = (x * 1103515245) + 12345;
      input_data_[i] = x;
    }

    expected_ = input_data_;
    QuickSort(&expected_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

TEST_P(FastSortBatchRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TerekhovDFastSortBatchMPI, TerekhovDFastSortBatchSEQ>(
        PPC_SETTINGS_terekhov_d_fast_sort_batch);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = FastSortBatchRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, FastSortBatchRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace terekhov_d_fast_sort_batch
