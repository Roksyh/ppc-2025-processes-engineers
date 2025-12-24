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
        input_data_ = {15, 8, 3, 12, 6};
        break;

      case 1:
        input_data_ = {-7, 25, 0, -3, 14, -1};
        break;

      case 2:
        input_data_ = {9, 9, 5, 5, 7, 2, 2};
        break;

      case 3:
        input_data_ = {2, 4, 6, 8, 7, 9};
        break;

      case 4:
        input_data_ = {12, 11, 10, 9, 8, 7, 6, 5, 4};
        break;

      case 5:
        input_data_ = {99};
        break;

      case 6:
        input_data_ = {15, 3, 9, 5, 12, 7, 15, 1, 22, 15, 5};
        break;

      case 7:
        input_data_ = {30, -8, 45, -20, 7, 2, 18};
        break;

      case 8:
        input_data_ = {};
        break;

      case 9:
        input_data_ = {33, 17};
        break;

      case 10:
        input_data_ = {25, 8,  14, 2,  -6, 77, 12, 29, 5,  44, 10, -3, 38, 21, 7,  19, 6,
                       31, 13, -9, 23, 20, 4,  35, 28, 46, 18, -5, 50, 41, 27, 33, 37, 34,
                       39, 48, 42, 36, 40, 51, -7, 52, 53, 54, 55, 56, 57, 58, 59, 60};
        break;

      case 11:
        input_data_ = {1500, -750, 345,  1,    1499, -2,   675,  -450,  1125, 30,  -1050, 49,   27,
                       900,  82,   -300, 1332, 1166, 999,  666,  -1500, 482,  185, 333,   166,  -166,
                       -333, -500, 22,   10,   13,   12,   150,  300,   450,  -75, -112,  -188, -30,
                       -15,  750,  752,  753,  -900, 1054, 1056, 1058,  1200, -8,  -3};
        break;

      default:
        input_data_ = {7,  -15, 10,  150, -7, -2, 1,  63, 27, -10, 75, 34, -18, 13, 11, 12,  -30, 49, 66, -3,
                       2,  9,   3,   6,   16, 15, -8, -9, 18, 22,  25, 28, -45, 37, 41, -22, 21,  20, 43, -12,
                       31, 33,  -13, 36,  39, 40, 46, 48, 51, 52,  54, 55, 56,  57, 58, 59,  64,  67, 68, 69};
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

TEST_P(TerekhovDFastSortBatchFuncTests, BatcherSortTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kTestParam = {std::make_tuple(0, "basic_case"),
                                             std::make_tuple(1, "with_negatives"),
                                             std::make_tuple(2, "with_duplicates"),
                                             std::make_tuple(3, "almost_sorted"),
                                             std::make_tuple(4, "reverse_sorted"),
                                             std::make_tuple(5, "single_element"),
                                             std::make_tuple(6, "with_multiple_duplicates"),
                                             std::make_tuple(7, "mixed_values"),
                                             std::make_tuple(8, "empty_array"),
                                             std::make_tuple(9, "two_elements"),
                                             std::make_tuple(10, "large_array_50"),
                                             std::make_tuple(11, "large_array_with_extremes"),
                                             std::make_tuple(12, "very_large_array_60")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TerekhovDFastSortBatchMPI, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch),
    ppc::util::AddFuncTask<TerekhovDFastSortBatchSEQ, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TerekhovDFastSortBatchFuncTests::PrintFuncTestName<TerekhovDFastSortBatchFuncTests>;

INSTANTIATE_TEST_SUITE_P(BatcherSortTestSuite, TerekhovDFastSortBatchFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace terekhov_d_fast_sort_batch
