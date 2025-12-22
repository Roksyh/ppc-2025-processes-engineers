#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"

#include <utility>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

TerekhovDFastSortBatchSEQ::TerekhovDFastSortBatchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TerekhovDFastSortBatchSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool TerekhovDFastSortBatchSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::pair<int, int> TerekhovDFastSortBatchSEQ::PartitionSegment(std::vector<int> &arr, int start_idx, int end_idx) {
  int left = start_idx;
  int right = end_idx;

  // Улучшенный выбор опорного элемента
  int a = arr[left];
  int b = arr[right];
  int c = arr[left + (right - left) / 2];

  // Медиана трех
  int pivot_value;
  if ((a > b) != (a > c)) {
    pivot_value = a;
  } else if ((b > a) != (b > c)) {
    pivot_value = b;
  } else {
    pivot_value = c;
  }

  while (left <= right) {
    while (arr[left] < pivot_value) {
      left++;
    }
    while (arr[right] > pivot_value) {
      right--;
    }
    if (left <= right) {
      std::swap(arr[left], arr[right]);
      left++;
      right--;
    }
  }

  return {left, right};
}

bool TerekhovDFastSortBatchSEQ::RunImpl() {
  std::vector<int> data_array = GetInput();
  if (data_array.empty()) {
    return true;
  }

  struct StackElement {
    int left;
    int right;
  };

  std::vector<StackElement> call_stack;
  call_stack.push_back({0, static_cast<int>(data_array.size()) - 1});

  while (!call_stack.empty()) {
    StackElement current = call_stack.back();
    call_stack.pop_back();

    int l = current.left;
    int r = current.right;

    if (l >= r) {
      continue;
    }

    auto [new_left, new_right] = PartitionSegment(data_array, l, r);

    if (l < new_right) {
      call_stack.push_back({l, new_right});
    }
    if (new_left < r) {
      call_stack.push_back({new_left, r});
    }
  }

  GetOutput().swap(data_array);
  return true;
}

bool TerekhovDFastSortBatchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
