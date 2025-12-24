#include "terekhov_d_fast_sort_batch/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include "terekhov_d_fast_sort_batch/common/include/common.hpp"

namespace terekhov_d_fast_sort_batch {

TerekhovDFastSortBatchMPI::TerekhovDFastSortBatchMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count_);

  SetTypeOfTask(GetStaticTypeOfTask());
  if (process_rank_ == 0) {
    GetInput() = in;
  }
  GetOutput() = std::vector<int>();
}

bool TerekhovDFastSortBatchMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool TerekhovDFastSortBatchMPI::PreProcessingImpl() {
  return true;
}

bool TerekhovDFastSortBatchMPI::RunImpl() {
  std::size_t original_array_size = 0;
  std::size_t padded_array_size = 0;

  BroadcastArraySizes(original_array_size, padded_array_size);

  if (original_array_size == 0) {
    return true;
  }

  std::vector<int> padded_input_data;
  if (process_rank_ == 0) {
    padded_input_data = GetInput();
    if (padded_array_size > original_array_size) {
      padded_input_data.resize(padded_array_size, std::numeric_limits<int>::max());
    }
  }

  std::vector<int> segment_sizes;
  std::vector<int> segment_offsets;
  std::vector<int> local_data_segment;
  DistributeArrayData(padded_array_size, padded_input_data, segment_sizes, segment_offsets, local_data_segment);

  std::sort(local_data_segment.begin(), local_data_segment.end());

  std::vector<std::pair<int, int>> comparator_pairs;
  GenerateComparatorPairs(comparator_pairs);
  ExecuteComparatorPairs(segment_sizes, local_data_segment, comparator_pairs);

  std::vector<int> collected_result;
  if (process_rank_ == 0) {
    collected_result.resize(padded_array_size);
  }

  MPI_Gatherv(local_data_segment.data(), static_cast<int>(local_data_segment.size()), MPI_INT, collected_result.data(),
              segment_sizes.data(), segment_offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (process_rank_ == 0) {
    collected_result.resize(original_array_size);
    GetOutput() = std::move(collected_result);
  }

  GetOutput().resize(original_array_size);
  MPI_Bcast(GetOutput().data(), static_cast<int>(original_array_size), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void TerekhovDFastSortBatchMPI::BroadcastArraySizes(std::size_t &original_array_size, std::size_t &padded_array_size) {
  if (process_rank_ == 0) {
    original_array_size = GetInput().size();
    const std::size_t remainder_value = original_array_size % process_count_;
    padded_array_size = original_array_size + (remainder_value == 0 ? 0 : (process_count_ - remainder_value));
  }

  MPI_Bcast(&original_array_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&padded_array_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

void TerekhovDFastSortBatchMPI::DistributeArrayData(const std::size_t &padded_array_size,
                                                    const std::vector<int> &padded_input_data,
                                                    std::vector<int> &segment_sizes, std::vector<int> &segment_offsets,
                                                    std::vector<int> &local_data_segment) const {
  const int base_segment_size = static_cast<int>(padded_array_size / process_count_);

  segment_sizes.resize(process_count_);
  segment_offsets.resize(process_count_);

  for (int i = 0, offset_value = 0; i < process_count_; i++) {
    segment_sizes[i] = base_segment_size;
    segment_offsets[i] = offset_value;
    offset_value += base_segment_size;
  }

  const int local_segment_size = segment_sizes[process_rank_];
  local_data_segment.resize(local_segment_size);

  MPI_Scatterv(padded_input_data.data(), segment_sizes.data(), segment_offsets.data(), MPI_INT,
               local_data_segment.data(), local_segment_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void TerekhovDFastSortBatchMPI::GenerateComparatorPairs(std::vector<std::pair<int, int>> &comparator_pairs) const {
  std::vector<int> process_list(process_count_);
  for (int i = 0; i < process_count_; i++) {
    process_list[i] = i;
  }

  ConstructSortNetwork(process_list, comparator_pairs);
}

void TerekhovDFastSortBatchMPI::ConstructSortNetwork(const std::vector<int> &processes,
                                                     std::vector<std::pair<int, int>> &comparator_pairs) {
  std::stack<std::pair<std::vector<int>, bool>> task_stack;
  task_stack.emplace(processes, false);

  while (!task_stack.empty()) {
    auto [current_processes, merge_phase] = task_stack.top();
    task_stack.pop();

    if (current_processes.size() <= 1) {
      continue;
    }

    auto middle_index = static_cast<std::vector<int>::difference_type>(current_processes.size() / 2);
    std::vector<int> left_half(current_processes.begin(), current_processes.begin() + middle_index);
    std::vector<int> right_half(current_processes.begin() + middle_index, current_processes.end());

    if (merge_phase) {
      ConstructMergeNetwork(left_half, right_half, comparator_pairs);
      continue;
    }

    task_stack.emplace(current_processes, true);
    task_stack.emplace(right_half, false);
    task_stack.emplace(left_half, false);
  }
}

void TerekhovDFastSortBatchMPI::ConstructMergeNetwork(const std::vector<int> &upper_processes,
                                                      const std::vector<int> &lower_processes,
                                                      std::vector<std::pair<int, int>> &comparator_pairs) {
  std::stack<std::tuple<std::vector<int>, std::vector<int>, bool>> task_stack;
  task_stack.emplace(upper_processes, lower_processes, false);

  while (!task_stack.empty()) {
    auto [upper_part, lower_part, merge_phase] = task_stack.top();
    task_stack.pop();
    const std::size_t total_processes_count = upper_part.size() + lower_part.size();

    if (total_processes_count <= 1) {
      continue;
    }
    if (total_processes_count == 2) {
      comparator_pairs.emplace_back(upper_part[0], lower_part[0]);
      continue;
    }

    if (!merge_phase) {
      auto [upper_odd, upper_even] = SeparateOddEven(upper_part);
      auto [lower_odd, lower_even] = SeparateOddEven(lower_part);

      task_stack.emplace(upper_part, lower_part, true);
      task_stack.emplace(upper_even, lower_even, false);
      task_stack.emplace(upper_odd, lower_odd, false);
      continue;
    }

    std::vector<int> merged_processes;
    merged_processes.reserve(total_processes_count);
    merged_processes.insert(merged_processes.end(), upper_part.begin(), upper_part.end());
    merged_processes.insert(merged_processes.end(), lower_part.begin(), lower_part.end());

    for (std::size_t i = 1; i < merged_processes.size() - 1; i += 2) {
      comparator_pairs.emplace_back(merged_processes[i], merged_processes[i + 1]);
    }
  }
}

std::pair<std::vector<int>, std::vector<int>> TerekhovDFastSortBatchMPI::SeparateOddEven(
    const std::vector<int> &elements) {
  std::vector<int> odd_elements;
  std::vector<int> even_elements;
  for (std::size_t i = 0; i < elements.size(); i++) {
    if (i % 2 == 0) {
      even_elements.push_back(elements[i]);
    } else {
      odd_elements.push_back(elements[i]);
    }
  }
  return std::make_pair(std::move(odd_elements), std::move(even_elements));
}

void TerekhovDFastSortBatchMPI::ExecuteComparatorPairs(const std::vector<int> &segment_sizes,
                                                       std::vector<int> &local_data_segment,
                                                       const std::vector<std::pair<int, int>> &comparator_pairs) const {
  std::vector<int> partner_buffer;
  std::vector<int> temporary_buffer;

  for (const auto &comparator : comparator_pairs) {
    const int first_process = comparator.first;
    const int second_process = comparator.second;

    if (process_rank_ != first_process && process_rank_ != second_process) {
      continue;
    }

    const int partner_process = (process_rank_ == first_process) ? second_process : first_process;
    const int local_segment_size = segment_sizes[process_rank_];
    const int partner_segment_size = segment_sizes[partner_process];

    partner_buffer.resize(partner_segment_size);
    temporary_buffer.resize(local_segment_size);

    MPI_Status comm_status;
    MPI_Sendrecv(local_data_segment.data(), local_segment_size, MPI_INT, partner_process, 0, partner_buffer.data(),
                 partner_segment_size, MPI_INT, partner_process, 0, MPI_COMM_WORLD, &comm_status);

    MergeDataSegments(local_data_segment, partner_buffer, temporary_buffer, process_rank_ == first_process);
    local_data_segment.swap(temporary_buffer);
  }
}

void TerekhovDFastSortBatchMPI::MergeDataSegments(const std::vector<int> &local_data,
                                                  const std::vector<int> &partner_data, std::vector<int> &result_buffer,
                                                  bool take_smaller_values) {
  const int local_size = static_cast<int>(local_data.size());
  const int partner_size = static_cast<int>(partner_data.size());

  if (take_smaller_values) {
    for (int temp_index = 0, local_index = 0, partner_index = 0; temp_index < local_size; temp_index++) {
      const int local_value = local_data[local_index];
      const int partner_value = partner_data[partner_index];
      if (local_value < partner_value) {
        result_buffer[temp_index] = local_value;
        local_index++;
      } else {
        result_buffer[temp_index] = partner_value;
        partner_index++;
      }
    }
  } else {
    for (int temp_index = local_size - 1, local_index = local_size - 1, partner_index = partner_size - 1;
         temp_index >= 0; temp_index--) {
      const int local_value = local_data[local_index];
      const int partner_value = partner_data[partner_index];
      if (local_value > partner_value) {
        result_buffer[temp_index] = local_value;
        local_index--;
      } else {
        result_buffer[temp_index] = partner_value;
        partner_index--;
      }
    }
  }
}

bool TerekhovDFastSortBatchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
