#include <gtest/gtest.h>
#include <stb/stb_image.h>

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
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    res_ = std::get<2>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (res_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType res_;
};

namespace {

TEST_P(TerekhovDFastSortBatchFuncTests, BatcherSortTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kTestParam = {
    std::make_tuple("case_01", std::vector<int>{15, 8, 3, 12, 6}, std::vector<int>{3, 6, 8, 12, 15}),
    std::make_tuple("case_02", std::vector<int>{-7, 25, 0, -3, 14, -1}, std::vector<int>{-7, -3, -1, 0, 14, 25}),
    std::make_tuple("case_03", std::vector<int>{9, 9, 5, 5, 7, 2, 2}, std::vector<int>{2, 2, 5, 5, 7, 9, 9}),
    std::make_tuple("case_04", std::vector<int>{2, 4, 6, 8, 7, 9}, std::vector<int>{2, 4, 6, 7, 8, 9}),
    std::make_tuple("case_05", std::vector<int>{12, 11, 10, 9, 8, 7, 6, 5, 4},
                    std::vector<int>{4, 5, 6, 7, 8, 9, 10, 11, 12}),
    std::make_tuple("case_06", std::vector<int>{99}, std::vector<int>{99}),
    std::make_tuple("case_07", std::vector<int>{15, 3, 9, 5, 12, 7, 15, 1, 22, 15, 5},
                    std::vector<int>{1, 3, 5, 5, 7, 9, 12, 15, 15, 15, 22}),
    std::make_tuple("case_08", std::vector<int>{30, -8, 45, -20, 7, 2, 18},
                    std::vector<int>{-20, -8, 2, 7, 18, 30, 45}),
    std::make_tuple("case_09", std::vector<int>{}, std::vector<int>{}),
    std::make_tuple("case_10", std::vector<int>{33, 17}, std::vector<int>{17, 33}),
    std::make_tuple("case_11", std::vector<int>{25, 8,  14, 2,  -6, 77, 12, 29, 5,  44, 10, -3, 38, 21, 7,  19, 6,
                                                31, 13, -9, 23, 20, 4,  35, 28, 46, 18, -5, 50, 41, 27, 33, 37, 34,
                                                39, 48, 42, 36, 40, 51, -7, 52, 53, 54, 55, 56, 57, 58, 59, 60},
                    std::vector<int>{-9, -7, -6, -5, -3, 2,  4,  5,  6,  7,  8,  10, 12, 13, 14, 18, 19,
                                     20, 21, 23, 25, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                                     42, 44, 46, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 77}),
    std::make_tuple("case_12",
                    std::vector<int>{1500, -750, 345,  1,    1499, -2,   675,  -450,  1125, 30,  -1050, 49,   27,
                                     900,  82,   -300, 1332, 1166, 999,  666,  -1500, 482,  185, 333,   166,  -166,
                                     -333, -500, 22,   10,   13,   12,   150,  300,   450,  -75, -112,  -188, -30,
                                     -15,  750,  752,  753,  -900, 1054, 1056, 1058,  1200, -8,  -3},
                    std::vector<int>{-1500, -1050, -900, -750, -500, -450, -333, -300, -188, -166, -112, -75, -30,
                                     -15,   -8,    -3,   -2,   1,    10,   12,   13,   22,   27,   30,   49,  82,
                                     150,   166,   185,  300,  333,  345,  450,  482,  666,  675,  750,  752, 753,
                                     900,   999,   1054, 1056, 1058, 1125, 1166, 1200, 1332, 1499, 1500}),
    std::make_tuple(
        "case_13",
        std::vector<int>{7,  -15, 10,  150, -7, -2, 1,  63, 27, -10, 75, 34, -18, 13, 11, 12,  -30, 49, 66, -3,
                         2,  9,   3,   6,   16, 15, -8, -9, 18, 22,  25, 28, -45, 37, 41, -22, 21,  20, 43, -12,
                         31, 33,  -13, 36,  39, 40, 46, 48, 51, 52,  54, 55, 56,  57, 58, 59,  64,  67, 68, 69},
        std::vector<int>{-45, -30, -22, -18, -15, -13, -12, -10, -9, -8, -7, -3, -2, 1,  2,  3,  6,  7,  9,  10,
                         11,  12,  13,  15,  16,  18,  20,  21,  22, 25, 27, 28, 31, 33, 34, 36, 37, 39, 40, 41,
                         43,  46,  48,  49,  51,  52,  54,  55,  56, 57, 58, 59, 63, 64, 66, 67, 68, 69, 75, 150})};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TerekhovDFastSortBatchMPI, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch),
    ppc::util::AddFuncTask<TerekhovDFastSortBatchSEQ, InType>(kTestParam, PPC_SETTINGS_terekhov_d_fast_sort_batch));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TerekhovDFastSortBatchFuncTests::PrintFuncTestName<TerekhovDFastSortBatchFuncTests>;

INSTANTIATE_TEST_SUITE_P(BatcherSortTestSuite, TerekhovDFastSortBatchFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace terekhov_d_fast_sort_batch
