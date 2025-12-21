#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"
#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"
#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace terekhov_d_fast_sort_batch {

class TerekhovDFastSortBatchRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);

    expected_ = input_data_;
    std::ranges::sort(expected_);
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

namespace {

TEST_P(TerekhovDFastSortBatchRunFuncTestsProcesses, FastSortBatchBasicCases) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(std::vector<int>{}, "empty"),
    std::make_tuple(std::vector<int>{42}, "single"),
    std::make_tuple(std::vector<int>{1, 2, 3, 4, 5, 6}, "already_sorted"),
    std::make_tuple(std::vector<int>{9, 7, 5, 3, 1, 0, -2}, "reverse_sorted"),
    std::make_tuple(std::vector<int>{5, 1, 5, 3, 3, 2, 1, 0, 0}, "duplicates"),
    std::make_tuple(std::vector<int>{10, -1, 7, 7, 2, -100, 50, 3}, "mixed_values"),
    std::make_tuple(std::vector<int>{2, 1}, "two_elems"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TerekhovDFastSortBatchMPI, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch),
    ppc::util::AddFuncTask<TerekhovDFastSortBatchSEQ, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName =
    TerekhovDFastSortBatchRunFuncTestsProcesses::PrintFuncTestName<TerekhovDFastSortBatchRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(TerekhovDFastSortBatchFuncTests, TerekhovDFastSortBatchRunFuncTestsProcesses, kGtestValues,
                         kFuncTestName);

}  // namespace

}  // namespace terekhov_d_fast_sort_batch
