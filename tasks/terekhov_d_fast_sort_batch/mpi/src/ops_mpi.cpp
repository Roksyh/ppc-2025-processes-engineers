#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <stack>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

TerekhovDFastSortBatchMPI::TerekhovDFastSortBatchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().resize(in.size());
}

bool TerekhovDFastSortBatchMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (GetInput().empty()) {
      return false;
    }
  }

  return true;
}

bool TerekhovDFastSortBatchMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() = GetInput();
  }
  return true;
}

int TerekhovDFastSortBatchMPI::Partition(std::vector<int> &arr, int left, int right) {
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

void TerekhovDFastSortBatchMPI::BatcherOddEvenMerge(std::vector<int> &arr, int left, int mid, int right) {
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
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }

  while (j <= right) {
    temp[k++] = arr[j++];
  }

  for (int idx = 0; idx < n; idx++) {
    arr[left + idx] = temp[idx];
  }
}

void TerekhovDFastSortBatchMPI::QuickSortWithBatcherMerge(std::vector<int> &arr, int left, int right) {
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

TerekhovDFastSortBatchMPI::DistributionInfo TerekhovDFastSortBatchMPI::PrepareDistributionInfo(int total_size,
                                                                                               int world_size,
                                                                                               int world_rank) {
  DistributionInfo info;

  int base_size = total_size / world_size;
  int remainder = total_size % world_size;

  info.local_size = base_size + (world_rank < remainder ? 1 : 0);

  info.send_counts.assign(world_size, 0);
  info.displacements.assign(world_size, 0);

  if (world_rank == 0) {
    int displacement = 0;
    for (int i = 0; i < world_size; ++i) {
      info.send_counts[i] = base_size + (i < remainder ? 1 : 0);
      info.displacements[i] = displacement;
      displacement += info.send_counts[i];
    }
  }

  return info;
}

std::vector<int> TerekhovDFastSortBatchMPI::DistributeData(int world_size, int world_rank) {
  int total_size = 0;

  if (world_rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  DistributionInfo info = PrepareDistributionInfo(total_size, world_size, world_rank);

  std::vector<int> local_buffer(info.local_size);

  MPI_Scatterv(GetOutput().data(), info.send_counts.data(), info.displacements.data(), MPI_INT, local_buffer.data(),
               info.local_size, MPI_INT, 0, MPI_COMM_WORLD);

  return local_buffer;
}

void TerekhovDFastSortBatchMPI::ParallelQuickSort() {
  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size == 1) {
    if (!GetOutput().empty()) {
      QuickSortWithBatcherMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);
    }
    return;
  }

  std::vector<int> local_buffer = DistributeData(world_size, world_rank);

  if (!local_buffer.empty()) {
    QuickSortWithBatcherMerge(local_buffer, 0, static_cast<int>(local_buffer.size()) - 1);
  }

  std::vector<int> all_sizes(world_size);
  int local_size = static_cast<int>(local_buffer.size());

  MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displacements(world_size, 0);
  int total_size = 0;

  if (world_rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      displacements[i] = total_size;
      total_size += all_sizes[i];
    }
    GetOutput().resize(total_size);
  }

  int *recv_data = (world_rank == 0) ? GetOutput().data() : nullptr;
  int *recv_counts = (world_rank == 0) ? all_sizes.data() : nullptr;
  int *recv_displs = (world_rank == 0) ? displacements.data() : nullptr;

  MPI_Gatherv(local_buffer.data(), local_size, MPI_INT, recv_data, recv_counts, recv_displs, MPI_INT, 0,
              MPI_COMM_WORLD);

  if (world_rank == 0 && !GetOutput().empty()) {
    QuickSortWithBatcherMerge(GetOutput(), 0, static_cast<int>(GetOutput().size()) - 1);
  }

  int final_size = 0;
  if (world_rank == 0) {
    final_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (world_rank != 0) {
    GetOutput().resize(final_size);
  }

  MPI_Bcast(GetOutput().data(), final_size, MPI_INT, 0, MPI_COMM_WORLD);
}

bool TerekhovDFastSortBatchMPI::RunImpl() {
  ParallelQuickSort();
  return true;
}

bool TerekhovDFastSortBatchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
