#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

TerekhovDFastSortBatchMPI::TerekhovDFastSortBatchMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TerekhovDFastSortBatchMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool TerekhovDFastSortBatchMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

void TerekhovDFastSortBatchMPI::BroadcastInputData(int rank) {
  int data_size = 0;
  if (rank == 0) {
    data_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    GetInput().resize(data_size);
  }
  MPI_Bcast(GetInput().data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
}

int TerekhovDFastSortBatchMPI::GetLocalSize(int arr_size, int rank, int proc_count) {
  int base_size = arr_size / proc_count;
  if (rank < (arr_size % proc_count)) {
    base_size += 1;
  }
  return base_size;
}

std::vector<int> TerekhovDFastSortBatchMPI::GetProcessSegment(int proc_count, int rank, int arr_size) {
  std::vector<int> bounds(2);
  int segment_base = arr_size / proc_count;
  int remainder = arr_size % proc_count;

  int segment_start = rank * segment_base;
  if (rank < remainder) {
    segment_start += rank;
  } else {
    segment_start += remainder;
  }

  int segment_end = segment_start + segment_base - 1;
  if (rank < remainder) {
    segment_end += 1;
  }

  bounds[0] = segment_start;
  bounds[1] = segment_end;
  return bounds;
}

std::pair<int, int> TerekhovDFastSortBatchMPI::PartitionArray(std::vector<int> &arr, int start_idx, int end_idx) {
  int left = start_idx;
  int right = end_idx;

  // Выбор опорного элемента как медианы трех
  int mid = left + (right - left) / 2;

  // Сортировка left, mid, right
  if (arr[right] < arr[left]) {
    std::swap(arr[left], arr[right]);
  }
  if (arr[mid] < arr[left]) {
    std::swap(arr[left], arr[mid]);
  }
  if (arr[right] < arr[mid]) {
    std::swap(arr[mid], arr[right]);
  }

  int pivot_value = arr[mid];

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

void TerekhovDFastSortBatchMPI::SortLocalData(std::vector<int> &data) {
  if (data.empty()) {
    return;
  }

  struct StackFrame {
    int left;
    int right;
  };

  std::vector<StackFrame> recursion_stack;
  recursion_stack.push_back({0, static_cast<int>(data.size()) - 1});

  while (!recursion_stack.empty()) {
    StackFrame current = recursion_stack.back();
    recursion_stack.pop_back();

    int l = current.left;
    int r = current.right;

    if (l >= r) {
      continue;
    }

    auto [new_left, new_right] = PartitionArray(data, l, r);

    if (l < new_right) {
      recursion_stack.push_back({l, new_right});
    }
    if (new_left < r) {
      recursion_stack.push_back({new_left, r});
    }
  }
}

void TerekhovDFastSortBatchMPI::CombineAndSelect(std::vector<int> &local_vec, std::vector<int> &partner_vec,
                                                 bool take_smaller) {
  std::vector<int> combined(local_vec.size() + partner_vec.size());

  size_t i = 0, j = 0, k = 0;
  while (i < local_vec.size() && j < partner_vec.size()) {
    if (local_vec[i] <= partner_vec[j]) {
      combined[k++] = local_vec[i++];
    } else {
      combined[k++] = partner_vec[j++];
    }
  }

  while (i < local_vec.size()) {
    combined[k++] = local_vec[i++];
  }

  while (j < partner_vec.size()) {
    combined[k++] = partner_vec[j++];
  }

  if (!take_smaller) {
    std::copy(combined.begin(), combined.begin() + local_vec.size(), local_vec.begin());
  } else {
    std::copy(combined.end() - local_vec.size(), combined.end(), local_vec.begin());
  }
}

void TerekhovDFastSortBatchMPI::ProcessExchange(std::vector<int> &local_vec, int arr_size, int rank, int proc_count,
                                                int partner_offset) {
  MPI_Status status;
  int partner_size = GetLocalSize(arr_size, rank + partner_offset, proc_count);
  std::vector<int> partner_data(partner_size);

  MPI_Sendrecv(local_vec.data(), static_cast<int>(local_vec.size()), MPI_INT, rank + partner_offset, 0,
               partner_data.data(), partner_size, MPI_INT, rank + partner_offset, 0, MPI_COMM_WORLD, &status);

  bool take_smaller_part = (partner_offset != 1);
  CombineAndSelect(local_vec, partner_data, take_smaller_part);
}

void TerekhovDFastSortBatchMPI::BatcherEvenStep(std::vector<int> &local_vec, std::vector<int> &segment, int arr_size,
                                                int rank, int proc_count) {
  if ((rank % 2 == 0) && (rank + 1 < proc_count) && (segment[1] != arr_size - 1) && (segment[0] <= segment[1])) {
    ProcessExchange(local_vec, arr_size, rank, proc_count, 1);
  } else if ((segment[0] <= segment[1]) && (rank % 2 == 1)) {
    ProcessExchange(local_vec, arr_size, rank, proc_count, -1);
  }
}

void TerekhovDFastSortBatchMPI::BatcherOddStep(std::vector<int> &local_vec, std::vector<int> &segment, int arr_size,
                                               int rank, int proc_count) {
  if ((rank % 2 == 1) && (rank + 1 < proc_count) && (segment[1] != arr_size - 1) && (segment[0] <= segment[1])) {
    ProcessExchange(local_vec, arr_size, rank, proc_count, 1);
  } else if ((segment[0] <= segment[1]) && (rank != 0) && (rank % 2 == 0)) {
    ProcessExchange(local_vec, arr_size, rank, proc_count, -1);
  }
}

void TerekhovDFastSortBatchMPI::BatcherMergeProcedure(std::vector<int> &local_vec, std::vector<int> &segment,
                                                      int arr_size, int rank, int proc_count) {
  int min_segment_size = arr_size / proc_count;
  if (min_segment_size == 0) {
    min_segment_size = 1;
  }

  int iteration_count = (arr_size + min_segment_size - 1) / min_segment_size;

  for (int iter = 0; iter < iteration_count; iter++) {
    if (iter % 2 == 0) {
      BatcherEvenStep(local_vec, segment, arr_size, rank, proc_count);
    } else {
      BatcherOddStep(local_vec, segment, arr_size, rank, proc_count);
    }
  }
}

void TerekhovDFastSortBatchMPI::DistributeFinalResult(int rank) {
  int final_size = 0;
  if (rank == 0) {
    final_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&final_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(final_size);
  }

  MPI_Bcast(GetOutput().data(), final_size, MPI_INT, 0, MPI_COMM_WORLD);
}

bool TerekhovDFastSortBatchMPI::RunImpl() {
  int proc_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  BroadcastInputData(rank);

  if (rank == 0) {
    int arr_size = static_cast<int>(GetInput().size());
    std::vector<int> process_sizes(proc_count);

    for (int i = 1; i < proc_count; i++) {
      std::vector<int> segment = GetProcessSegment(proc_count, i, arr_size);
      process_sizes[i] = (segment[1] + 1) - segment[0];
      MPI_Send(segment.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    std::vector<int> local_segment = GetProcessSegment(proc_count, 0, arr_size);
    std::vector<int> local_data;

    if (local_segment[0] <= local_segment[1]) {
      local_data = std::vector<int>(GetInput().begin() + local_segment[0], GetInput().begin() + local_segment[1] + 1);
      SortLocalData(local_data);
    }

    BatcherMergeProcedure(local_data, local_segment, arr_size, rank, proc_count);
    GetOutput().insert(GetOutput().end(), local_data.begin(), local_data.end());

    MPI_Status status;
    for (int i = 1; i < proc_count; i++) {
      std::vector<int> received_data(process_sizes[i]);
      MPI_Recv(received_data.data(), process_sizes[i], MPI_INT, i, 2, MPI_COMM_WORLD, &status);
      GetOutput().insert(GetOutput().end(), received_data.begin(), received_data.end());
    }
  } else {
    MPI_Status status;
    std::vector<int> segment_info(2);
    MPI_Recv(segment_info.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    std::vector<int> local_data;
    if (segment_info[0] <= segment_info[1]) {
      local_data = std::vector<int>(GetInput().begin() + segment_info[0], GetInput().begin() + segment_info[1] + 1);
      SortLocalData(local_data);
    }

    int arr_size = static_cast<int>(GetInput().size());
    BatcherMergeProcedure(local_data, segment_info, arr_size, rank, proc_count);
    MPI_Send(local_data.data(), static_cast<int>(local_data.size()), MPI_INT, 0, 2, MPI_COMM_WORLD);
  }

  DistributeFinalResult(rank);
  return true;
}

bool TerekhovDFastSortBatchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
