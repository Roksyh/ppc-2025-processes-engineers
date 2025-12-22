#include "terekhov_d_fast_sort_batch/seq/include/ops_seq.hpp"

#include <algorithm>
#include <stack>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

TerekhovDFastSortBatchSEQ::TerekhovDFastSortBatchSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool TerekhovDFastSortBatchSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool TerekhovDFastSortBatchSEQ::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

int TerekhovDFastSortBatchSEQ::Partition(std::vector<int> &arr, int left, int right) {
  int pivot_index = left + ((right - left) / 2);
  int pivot_value = arr[pivot_index];

  std::swap(arr[pivot_index], arr[left]);

  int i = left + 1;
  int j = right;

  while (i <= j) {
    while (i <= j && arr[i] <= pivot_value) {
      i++;
    }

    while (i <= j && arr[j] > pivot_value) {
      j--;
    }

    if (i < j) {
      std::swap(arr[i], arr[j]);
      i++;
      j--;
    } else {
      break;
    }
  }

  std::swap(arr[left], arr[j]);

  return j;
}

void TerekhovDFastSortBatchSEQ::BatcherOddEvenMerge(std::vector<int> &arr, int left, int mid, int right) {
  int n = right - left + 1;

  if (n <= 1) {
    return;
  }

  if (n == 2) {
    if (arr[left] > arr[right]) {
      std::swap(arr[left], arr[right]);
    }
    return;
  }

  std::vector<int> temp(n);

  int i = left;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= right) {
    if (arr[i] <= arr[j]) {
      temp[k] = arr[i];
      i++;
    } else {
      temp[k] = arr[j];
      j++;
    }
    k++;
  }

  while (i <= mid) {
    temp[k] = arr[i];
    i++;
    k++;
  }

  while (j <= right) {
    temp[k] = arr[j];
    j++;
    k++;
  }

  for (int step = 1; step < n; step *= 2) {
    for (int start = 0; start + step < n; start += 2 * step) {
      int end = std::min(start + 2 * step - 1, n - 1);
      int middle = start + step - 1;

      int idx1 = start;
      int idx2 = middle + 1;

      while (idx1 <= middle && idx2 <= end) {
        if (temp[idx1] > temp[idx2]) {
          std::swap(temp[idx1], temp[idx2]);
        }
        idx1++;
        idx2++;
      }
    }
  }

  for (int idx = 0; idx < k; idx++) {
    arr[left + idx] = temp[idx];
  }
}

void TerekhovDFastSortBatchSEQ::QuickSortWithBatcherMerge(std::vector<int> &arr, int left, int right) {
  struct Range {
    int left;
    int right;
  };

  std::stack<Range> stack;
  stack.push({left, right});

  while (!stack.empty()) {
    Range current = stack.top();
    stack.pop();

    int l = current.left;
    int r = current.right;

    if (l >= r) {
      continue;
    }

    int pivot_index = Partition(arr, l, r);

    int left_size = pivot_index - l;
    int right_size = r - pivot_index;

    if (left_size > 1 && right_size > 1) {
      if (left_size > right_size) {
        stack.push({pivot_index + 1, r});
        stack.push({l, pivot_index - 1});
      } else {
        stack.push({l, pivot_index - 1});
        stack.push({pivot_index + 1, r});
      }
    } else if (left_size > 1) {
      stack.push({l, pivot_index - 1});
    } else if (right_size > 1) {
      stack.push({pivot_index + 1, r});
    }

    BatcherOddEvenMerge(arr, l, pivot_index, r);
  }
}

bool TerekhovDFastSortBatchSEQ::RunImpl() {
  QuickSortWithBatcherMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);

  return true;
}

bool TerekhovDFastSortBatchSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
