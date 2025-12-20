#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "Terekhov_D_Min_Column_Matrix/common/include/common.hpp"
#include "Terekhov_D_Min_Column_Matrix/mpi/include/ops_mpi.hpp"
#include "Terekhov_D_Min_Column_Matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace terekhov_d_a_test_task_processes {

class MinColumnRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int case_id = std::get<0>(params);
    input_data_ = InType{};
    expected_ = OutType{};

    switch (case_id) {
      case 0:
        input_data_ = {{3, 5, -1}, {0, -2, 10}, {7, 1, -5}};
        expected_ = {0, -2, -5};
        break;
      case 1:
        input_data_ = {{4, -1, 2}};
        expected_ = {4, -1, 2};
        break;
      case 2:
        input_data_ = {{-10, 3}, {-5, 2}, {-20, 4}};
        expected_ = {-20, 2};
        break;
      case 3:
        input_data_ = {{7}};
        expected_ = {7};
        break;
      case 4:
        input_data_ = {{1, 2}, {3, 4}};
        expected_ = {1, 2};
        break;
      case 5:
        input_data_ = {{10, 20}};
        expected_ = {10, 20};
        break;
      case 6:
        input_data_ = {{3}, {1}, {2}};
        expected_ = {1};
        break;
      case 7:
        input_data_ = {{5, 5, 5}, {5, 5, 5}};
        expected_ = {5, 5, 5};
        break;
      case 8:
        input_data_ = {{-3}, {-1}, {-2}, {-5}};
        expected_ = {-5};
        break;
      case 9:
        input_data_ = {{10, 20}, {30, 5}, {15, 25}};
        expected_ = {10, 5};
        break;
      default:
        input_data_ = {{1}};
        expected_ = {1};
        break;
    }
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

TEST_P(MinColumnRunFuncTestsProcesses, MinColumnBasicCases) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple(0, "rectangular"),
    std::make_tuple(1, "single_row"),
    std::make_tuple(2, "negative_numbers"),
    std::make_tuple(3, "single_element_matrix"),
    std::make_tuple(4, "two_by_two_matrix"),
    std::make_tuple(5, "single_row_two_columns"),
    std::make_tuple(6, "three_rows_one_column"),
    std::make_tuple(7, "all_equal_values"),
    std::make_tuple(8, "single_column_negative"),
    std::make_tuple(9, "mixed_values_matrix"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TerekhovDTestTaskMPI, InType>(kTestParam, PPC_SETTINGS_Terekhov_D_Min_Column_Matrix),
    ppc::util::AddFuncTask<TerekhovDTestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_Terekhov_D_Min_Column_Matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = MinColumnRunFuncTestsProcesses::PrintFuncTestName<MinColumnRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(TerekhovDMinColumnFuncTests, MinColumnRunFuncTestsProcesses, kGtestValues, kFuncTestName);

TEST(TerekhovDTestTaskSEQAdditional, AllEqualValuesMatrix) {
  terekhov_d_a_test_task_processes::InType matrix = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
  terekhov_d_a_test_task_processes::TerekhovDTestTaskSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  const auto output = task.GetOutput();
  ASSERT_EQ(output.size(), 3U);
  EXPECT_EQ(output[0], 5);
  EXPECT_EQ(output[1], 5);
  EXPECT_EQ(output[2], 5);
}

TEST(TerekhovDTestTaskSEQAdditional, SingleColumnMultiRow) {
  terekhov_d_a_test_task_processes::InType matrix = {{3}, {1}, {2}, {5}, {0}};
  terekhov_d_a_test_task_processes::TerekhovDTestTaskSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  const auto output = task.GetOutput();
  ASSERT_EQ(output.size(), 1U);
  EXPECT_EQ(output[0], 0);
}

TEST(TerekhovDTestTaskSEQAdditional, MatrixWithMaxIntValues) {
  terekhov_d_a_test_task_processes::InType matrix = {{std::numeric_limits<int>::max(), 100},
                                                     {200, std::numeric_limits<int>::max()}};
  terekhov_d_a_test_task_processes::TerekhovDTestTaskSEQ task(matrix);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
  const auto output = task.GetOutput();
  ASSERT_EQ(output.size(), 2U);
  EXPECT_EQ(output[0], 200);
  EXPECT_EQ(output[1], 100);
}

}  // namespace

}  // namespace terekhov_d_a_test_task_processes
