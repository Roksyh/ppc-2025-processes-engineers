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
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &total_processes_);

  SetTypeOfTask(GetStaticTypeOfTask());
  if (current_rank_ == 0) {
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
  size_t initial_size = 0;
  size_t expanded_size = 0;

  BroadcastArrayDimensions(initial_size, expanded_size);

  if (initial_size == 0) {
    return true;
  }

  std::vector<int> expanded_input;
  if (current_rank_ == 0) {
    expanded_input = GetInput();
    if (expanded_size > initial_size) {
      expanded_input.resize(expanded_size, std::numeric_limits<int>::max());
    }
  }

  std::vector<int> portion_sizes;
  std::vector<int> portion_starts;
  std::vector<int> local_portion;
  DistributeArrayElements(expanded_size, expanded_input, portion_sizes, portion_starts, local_portion);

  // Локальная сортировка
  std::ranges::sort(local_portion);

  std::vector<std::pair<int, int>> comparator_list;
  BuildComparatorSequence(comparator_list);
  ExecuteComparators(portion_sizes, local_portion, comparator_list);

  std::vector<int> collected_result;
  if (current_rank_ == 0) {
    collected_result.resize(expanded_size);
  }

  MPI_Gatherv(local_portion.data(), static_cast<int>(local_portion.size()), MPI_INT, collected_result.data(),
              portion_sizes.data(), portion_starts.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (current_rank_ == 0) {
    collected_result.resize(initial_size);
    GetOutput() = std::move(collected_result);
  }

  GetOutput().resize(initial_size);
  MPI_Bcast(GetOutput().data(), static_cast<int>(initial_size), MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

void TerekhovDFastSortBatchMPI::BroadcastArrayDimensions(size_t &initial_size, size_t &expanded_size) {
  if (current_rank_ == 0) {
    initial_size = GetInput().size();
    const size_t remainder = initial_size % total_processes_;
    expanded_size = initial_size + (remainder == 0 ? 0 : (total_processes_ - remainder));
  }

  MPI_Bcast(&initial_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&expanded_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
}

void TerekhovDFastSortBatchMPI::DistributeArrayElements(const size_t &expanded_size,
                                                        const std::vector<int> &expanded_input,
                                                        std::vector<int> &portion_sizes,
                                                        std::vector<int> &portion_starts,
                                                        std::vector<int> &local_portion) const {
  const int base_portion = static_cast<int>(expanded_size / total_processes_);

  portion_sizes.resize(total_processes_);
  portion_starts.resize(total_processes_);

  for (int i = 0, offset = 0; i < total_processes_; i++) {
    portion_sizes[i] = base_portion;
    portion_starts[i] = offset;
    offset += base_portion;
  }

  const int local_size = portion_sizes[current_rank_];
  local_portion.resize(local_size);

  MPI_Scatterv(expanded_input.data(), portion_sizes.data(), portion_starts.data(), MPI_INT, local_portion.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);
}

void TerekhovDFastSortBatchMPI::BuildComparatorSequence(std::vector<std::pair<int, int>> &comparator_list) const {
  std::vector<int> all_ranks(total_processes_);
  for (int i = 0; i < total_processes_; i++) {
    all_ranks[i] = i;
  }

  ConstructSortStep(all_ranks, comparator_list);
}

void TerekhovDFastSortBatchMPI::ConstructSortStep(const std::vector<int> &process_group,
                                                  std::vector<std::pair<int, int>> &comparator_list) {
  std::stack<std::pair<std::vector<int>, bool>> task_stack;
  task_stack.emplace(process_group, false);

  while (!task_stack.empty()) {
    auto [current_group, is_merge_step] = task_stack.top();
    task_stack.pop();

    if (current_group.size() <= 1) {
      continue;
    }

    auto middle = static_cast<std::vector<int>::difference_type>(current_group.size() / 2);
    std::vector<int> left_half(current_group.begin(), current_group.begin() + middle);
    std::vector<int> right_half(current_group.begin() + middle, current_group.end());

    if (is_merge_step) {
      ConstructMergeStep(left_half, right_half, comparator_list);
      continue;
    }

    task_stack.emplace(current_group, true);
    task_stack.emplace(right_half, false);
    task_stack.emplace(left_half, false);
  }
}

void TerekhovDFastSortBatchMPI::ConstructMergeStep(const std::vector<int> &top_group,
                                                   const std::vector<int> &bottom_group,
                                                   std::vector<std::pair<int, int>> &comparator_list) {
  std::stack<std::tuple<std::vector<int>, std::vector<int>, bool>> task_stack;
  task_stack.emplace(top_group, bottom_group, false);

  while (!task_stack.empty()) {
    auto [upper_part, lower_part, is_merge_phase] = task_stack.top();
    task_stack.pop();
    const size_t total_count = upper_part.size() + lower_part.size();

    if (total_count <= 1) {
      continue;
    }
    if (total_count == 2) {
      comparator_list.emplace_back(upper_part[0], lower_part[0]);
      continue;
    }

    if (!is_merge_phase) {
      auto [upper_odd, upper_even] = SplitByPosition(upper_part);
      auto [lower_odd, lower_even] = SplitByPosition(lower_part);

      task_stack.emplace(upper_part, lower_part, true);
      task_stack.emplace(upper_even, lower_even, false);
      task_stack.emplace(upper_odd, lower_odd, false);
      continue;
    }

    std::vector<int> merged;
    merged.reserve(total_count);
    merged.insert(merged.end(), upper_part.begin(), upper_part.end());
    merged.insert(merged.end(), lower_part.begin(), lower_part.end());

    for (size_t i = 1; i < merged.size() - 1; i += 2) {
      comparator_list.emplace_back(merged[i], merged[i + 1]);
    }
  }
}

std::pair<std::vector<int>, std::vector<int>> TerekhovDFastSortBatchMPI::SplitByPosition(
    const std::vector<int> &items) {
  std::vector<int> odd_items;
  std::vector<int> even_items;
  for (size_t i = 0; i < items.size(); i++) {
    if (i % 2 == 0) {
      even_items.push_back(items[i]);
    } else {
      odd_items.push_back(items[i]);
    }
  }
  return std::make_pair(std::move(odd_items), std::move(even_items));
}

void TerekhovDFastSortBatchMPI::ExecuteComparators(const std::vector<int> &portion_sizes,
                                                   std::vector<int> &local_portion,
                                                   const std::vector<std::pair<int, int>> &comparator_list) const {
  std::vector<int> partner_buffer;
  std::vector<int> temp_buffer;

  for (const auto &comparator : comparator_list) {
    const int first_rank = comparator.first;
    const int second_rank = comparator.second;

    if (current_rank_ != first_rank && current_rank_ != second_rank) {
      continue;
    }

    const int partner_rank = (current_rank_ == first_rank) ? second_rank : first_rank;
    const int local_size = portion_sizes[current_rank_];
    const int partner_size = portion_sizes[partner_rank];

    partner_buffer.resize(partner_size);
    temp_buffer.resize(local_size);

    MPI_Status comm_status;
    MPI_Sendrecv(local_portion.data(), local_size, MPI_INT, partner_rank, 0, partner_buffer.data(), partner_size,
                 MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &comm_status);

    MergePortions(local_portion, partner_buffer, temp_buffer, current_rank_ == first_rank);
    local_portion.swap(temp_buffer);
  }
}

void TerekhovDFastSortBatchMPI::MergePortions(const std::vector<int> &local_data, const std::vector<int> &partner_data,
                                              std::vector<int> &result_buffer, bool take_minimal) {
  const int local_size = static_cast<int>(local_data.size());
  const int partner_size = static_cast<int>(partner_data.size());

  if (take_minimal) {
    for (int idx = 0, local_idx = 0, partner_idx = 0; idx < local_size; idx++) {
      const int local_value = local_data[local_idx];
      const int partner_value = partner_data[partner_idx];
      if (local_value < partner_value) {
        result_buffer[idx] = local_value;
        local_idx++;
      } else {
        result_buffer[idx] = partner_value;
        partner_idx++;
      }
    }
  } else {
    for (int idx = local_size - 1, local_idx = local_size - 1, partner_idx = partner_size - 1; idx >= 0; idx--) {
      const int local_value = local_data[local_idx];
      const int partner_value = partner_data[partner_idx];
      if (local_value > partner_value) {
        result_buffer[idx] = local_value;
        local_idx--;
      } else {
        result_buffer[idx] = partner_value;
        partner_idx--;
      }
    }
  }
}

bool TerekhovDFastSortBatchMPI::PostProcessingImpl() {
  return true;
}

}  // namespace terekhov_d_fast_sort_batch
