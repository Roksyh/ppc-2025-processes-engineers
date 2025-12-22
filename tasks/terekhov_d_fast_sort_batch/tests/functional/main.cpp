#include <gtest/gtest.h>
#include <stb/stb_image.h>

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

class TerekhovDFastSortBatchFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);

    switch (test_case) {
      case 0:
        input_data_ = {15, 8, 42, 4, 23, 16};
        break;

      case 1:
        input_data_ = {-20, 35, -7, 0, -99, 14};
        break;

      case 2:
        input_data_ = {10, 20, 30, 40, 50, 60};
        break;

      case 3:
        input_data_ = {55, 44, 33, 22, 11, 0};
        break;

      case 4:
        input_data_ = {7, 3, 9, 3, 2, 8, 5, 1, 7, 3};
        break;

      case 5:
        input_data_ = {13, 13, 13, 13, 13};
        break;

      case 6:
        input_data_ = {999};
        break;

      case 7:
        input_data_ = {25, 50};
        break;

      case 8:
        input_data_ = {88, 33};
        break;

      case 9:
        input_data_ = {45, 12, 78, 23, 56, 89, 34, 67, 90, 11};
        break;

      case 10:
        input_data_ = {19, 7, 24, 3, 15, 8, 30, 12, 28, 6, 17};
        break;

      case 11:
        input_data_ = {0, -5, 0, 10, -15, 20};
        break;

      case 12:
        input_data_ = {2000000000, -2000000000, 500, -500, 0};
        break;

      default:
        input_data_ = {150, -75, 25, -25, 100, -100, 50, -50, 0};
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (input_data_.empty() && !output_data.empty()) {
      return false;
    }

    if (!input_data_.empty() && output_data.empty()) {
      return false;
    }

    if (output_data.size() != input_data_.size()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); i++) {
      if (output_data[i] < output_data[i - 1]) {
        return false;
      }
    }

    std::vector<int> sorted_input = input_data_;
    std::vector<int> sorted_output = output_data;

    std::ranges::sort(sorted_input);
    std::ranges::sort(sorted_output);

    return sorted_input == sorted_output;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(TerekhovDFastSortBatchFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 14> kTestParam = {
    std::make_tuple(0, "basic_case"),      std::make_tuple(1, "with_negatives"),  std::make_tuple(2, "already_sorted"),
    std::make_tuple(3, "reverse_sorted"),  std::make_tuple(4, "with_duplicates"), std::make_tuple(5, "all_equal"),
    std::make_tuple(6, "single_element"),  std::make_tuple(7, "two_sorted"),      std::make_tuple(8, "two_unsorted"),
    std::make_tuple(9, "even_size_large"), std::make_tuple(10, "odd_size_large"), std::make_tuple(11, "with_zeros"),
    std::make_tuple(12, "extreme_values"), std::make_tuple(13, "mixed_case")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TerekhovDFastSortBatchMPI, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch),
    ppc::util::AddFuncTask<TerekhovDFastSortBatchSEQ, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TerekhovDFastSortBatchFuncTests::PrintFuncTestName<TerekhovDFastSortBatchFuncTests>;

INSTANTIATE_TEST_SUITE_P(QuickSortTests, TerekhovDFastSortBatchFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace terekhov_d_fast_sort_batch
