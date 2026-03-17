use numpy::PyArray1;
use numpy::{PyReadonlyArray1, PyReadonlyArray4};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

#[derive(Clone)]
struct FlattenedEpisode {
    // We assume fixed shapes for an episode step, but variable length T
    states: Vec<f32>,      // T * N * C * H * W
    actions: Vec<i64>,     // T * N
    rewards: Vec<f32>,     // T * N
    next_states: Vec<f32>, // T * N * C * H * W
    dones: Vec<f32>,       // T * N (using f32 for convenience with tensor conversion)
    global_map: Vec<f32>,  // C * H * W (single global map per episode)

    length: usize,
    // Prefix with _ to suppress unused warning
    _num_agents: usize,
    _obs_shape: (usize, usize, usize),        // C, H, W
    _global_map_shape: (usize, usize, usize), // C, H_g, W_g
}

impl FlattenedEpisode {
    fn from_py(
        py: Python,
        episode_dict: &PyAny,
        num_agents: usize,
        obs_shape: (usize, usize, usize),
        global_map_shape: (usize, usize, usize),
    ) -> PyResult<Self> {
        // Expecting dictionary with lists/arrays
        let states_list: Vec<PyObject> = episode_dict.get_item("states")?.extract()?;
        let actions_list: Vec<Vec<i64>> = episode_dict.get_item("actions")?.extract()?;
        let rewards_list: Vec<Vec<f32>> = episode_dict.get_item("rewards")?.extract()?;
        let next_states_list: Vec<PyObject> = episode_dict.get_item("next_states")?.extract()?;
        let dones_list: Vec<Vec<f32>> = episode_dict.get_item("dones")?.extract()?; // Bool or float? Assuming float for easier handling

        let global_map_obj: PyObject = episode_dict.get_item("global_map")?.extract()?;
        let global_map_arr: &numpy::PyArray3<f32> = global_map_obj.extract(py)?;
        let global_map_slice = unsafe { global_map_arr.as_slice()? };
        let global_map_flat = global_map_slice.to_vec();

        let length = states_list.len();
        let (c, h, w) = obs_shape;
        let obs_size = num_agents * c * h * w;

        let mut states_flat = Vec::with_capacity(length * obs_size);
        let mut next_states_flat = Vec::with_capacity(length * obs_size);

        // Helper to flatten numpy arrays
        for s_obj in states_list {
            let s_arr: &numpy::PyArray4<f32> = s_obj.extract(py)?;
            let s_slice = unsafe { s_arr.as_slice()? };
            states_flat.extend_from_slice(s_slice);
        }

        for ns_obj in next_states_list {
            let ns_arr: &numpy::PyArray4<f32> = ns_obj.extract(py)?;
            let ns_slice = unsafe { ns_arr.as_slice()? };
            next_states_flat.extend_from_slice(ns_slice);
        }

        let actions_flat: Vec<i64> = actions_list.into_iter().flatten().collect();
        let rewards_flat: Vec<f32> = rewards_list.into_iter().flatten().collect();
        let dones_flat: Vec<f32> = dones_list.into_iter().flatten().collect();

        Ok(FlattenedEpisode {
            states: states_flat,
            actions: actions_flat,
            rewards: rewards_flat,
            next_states: next_states_flat,
            dones: dones_flat,
            global_map: global_map_flat,
            length,
            _num_agents: num_agents,
            _obs_shape: obs_shape,
            _global_map_shape: global_map_shape,
        })
    }
}

#[pyclass]
struct RustReplayBuffer {
    capacity: usize,
    seq_len: usize,
    num_agents: usize,
    obs_shape: (usize, usize, usize),
    global_map_shape: (usize, usize, usize),
    buffer: VecDeque<FlattenedEpisode>,
}

#[pymethods]
impl RustReplayBuffer {
    #[new]
    fn new(
        capacity: usize,
        seq_len: usize,
        num_agents: usize,
        obs_shape: (usize, usize, usize),
        global_map_shape: (usize, usize, usize),
    ) -> Self {
        RustReplayBuffer {
            capacity,
            seq_len,
            num_agents,
            obs_shape,
            global_map_shape,
            buffer: VecDeque::with_capacity(capacity),
        }
    }

    fn push(&mut self, py: Python, episode_data: &PyAny) -> PyResult<()> {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        let episode = FlattenedEpisode::from_py(
            py,
            episode_data,
            self.num_agents,
            self.obs_shape,
            self.global_map_shape,
        )?;
        self.buffer.push_back(episode);
        Ok(())
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn __len__(&self) -> usize {
        self.buffer.len()
    }

    fn clear(&mut self) {
        self.buffer.clear();
    }

    fn sample<'py>(&self, py: Python<'py>, batch_size: usize) -> PyResult<&'py PyAny> {
        if self.buffer.len() < batch_size {
            return Ok(py.None().into_ref(py));
        }

        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let sampled_indices: Vec<usize> = indices
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();

        self._construct_batch(py, &sampled_indices)
    }
}

impl RustReplayBuffer {
    fn _construct_batch<'py>(&self, py: Python<'py>, indices: &[usize]) -> PyResult<&'py PyAny> {
        let batch_size = indices.len();
        let (c, h, w) = self.obs_shape;
        let n = self.num_agents;
        let t = self.seq_len;
        let (gc, gh, gw) = self.global_map_shape;

        let obs_dim = n * c * h * w;
        let global_map_dim = gc * gh * gw;

        // Pre-allocate flat vectors for the batch
        let mut batch_states = vec![0.0; batch_size * t * obs_dim];
        let mut batch_next_states = vec![0.0; batch_size * t * obs_dim];
        let mut batch_actions = vec![0i64; batch_size * t * n];
        let mut batch_rewards = vec![0.0; batch_size * t * n];
        let mut batch_dones = vec![0.0; batch_size * t * n];
        let mut batch_global_maps = vec![0.0; batch_size * global_map_dim];

        let mut rng = rand::thread_rng();
        let mut batch_masks = vec![0.0f32; batch_size * t]; // 默认全 0（无效）

        for (i, &idx) in indices.iter().enumerate() {
            let episode = &self.buffer[idx];

            // Random start index
            // Ensure we can take seq_len. Episode must be >= seq_len (handled in python usually, but let's be safe)
            let max_start = if episode.length > t {
                episode.length - t
            } else {
                0
            };
            let start_idx = rng.gen_range(0..=max_start);
            let end_idx = std::cmp::min(start_idx + t, episode.length);
            let actual_len = end_idx - start_idx;

            // Copy data
            // Calculate offsets
            let batch_state_offset = i * t * obs_dim;
            let batch_action_offset = i * t * n;
            let batch_global_map_offset = i * global_map_dim;

            let ep_state_start = start_idx * obs_dim;
            let ep_state_end = end_idx * obs_dim;

            let ep_action_start = start_idx * n;
            let ep_action_end = end_idx * n;

            let mask_offset = i * t;
            for m in 0..actual_len {
                batch_masks[mask_offset + m] = 1.0;
            }

            // Copy slices
            batch_states[batch_state_offset..batch_state_offset + actual_len * obs_dim]
                .copy_from_slice(&episode.states[ep_state_start..ep_state_end]);

            batch_next_states[batch_state_offset..batch_state_offset + actual_len * obs_dim]
                .copy_from_slice(&episode.next_states[ep_state_start..ep_state_end]);

            batch_actions[batch_action_offset..batch_action_offset + actual_len * n]
                .copy_from_slice(&episode.actions[ep_action_start..ep_action_end]);

            batch_rewards[batch_action_offset..batch_action_offset + actual_len * n]
                .copy_from_slice(&episode.rewards[ep_action_start..ep_action_end]);

            batch_dones[batch_action_offset..batch_action_offset + actual_len * n]
                .copy_from_slice(&episode.dones[ep_action_start..ep_action_end]);

            batch_global_maps[batch_global_map_offset..batch_global_map_offset + global_map_dim]
                .copy_from_slice(&episode.global_map[0..global_map_dim]);

            // Zero-padding is implicit since we initialized with 0.0/0
            // If actual_len < t, the rest remains 0
        }

        // Convert to NumPy arrays with correct shapes
        // PyTorch expects:
        // States: (B, T, N, C, H, W)
        // Actions: (B, T, N)
        // global_maps: (B, C, H_g, W_g)

        let states_shape = (batch_size, t, n, c, h, w);
        let actions_shape = (batch_size, t, n);
        let global_maps_shape = (batch_size, gc, gh, gw);

        let masks_shape = (batch_size, t); // [新增] 声明 masks 的形状

        let np_states = PyArray1::from_vec(py, batch_states).reshape(states_shape)?;
        let np_next_states = PyArray1::from_vec(py, batch_next_states).reshape(states_shape)?;
        let np_actions = PyArray1::from_vec(py, batch_actions).reshape(actions_shape)?;
        let np_rewards = PyArray1::from_vec(py, batch_rewards).reshape(actions_shape)?;
        let np_dones = PyArray1::from_vec(py, batch_dones).reshape(actions_shape)?;
        let np_global_maps =
            PyArray1::from_vec(py, batch_global_maps).reshape(global_maps_shape)?;

        let np_masks = PyArray1::from_vec(py, batch_masks).reshape(masks_shape)?; // 转换为 numpy 返回：

        let dict = pyo3::types::PyDict::new(py);

        dict.set_item("masks", np_masks)?;
        dict.set_item("states", np_states)?;
        dict.set_item("next_states", np_next_states)?;
        dict.set_item("actions", np_actions)?;
        dict.set_item("rewards", np_rewards)?;
        dict.set_item("dones", np_dones)?;
        dict.set_item("global_maps", np_global_maps)?;

        Ok(dict.into())
    }
}

// ---------------- Reward Calculator with Next Obs Check ----------------

#[pyclass]
struct RustRewardCalculator {
    num_agents: usize,
    total_episode_reward: Vec<f32>,
}

#[derive(FromPyObject)]
struct AlignmentConfig {
    #[pyo3(item)]
    use_alignment: bool,
    #[pyo3(item)]
    value: Option<f32>,
}

#[derive(FromPyObject)]
struct StopPenaltyConfig {
    #[pyo3(item)]
    use_stop_penalty: bool,
    #[pyo3(item)]
    value: Option<f32>,
}

#[derive(FromPyObject)]
struct StepPenaltyConfig {
    #[pyo3(item)]
    use_step_penalty: bool,
    #[pyo3(item)]
    value: Option<f32>,
}

#[pymethods]
impl RustRewardCalculator {
    #[new]
    fn new(num_agents: usize) -> Self {
        RustRewardCalculator {
            num_agents,
            total_episode_reward: vec![0.0; num_agents],
        }
    }

    fn reset(&mut self) {
        for x in &mut self.total_episode_reward {
            *x = 0.0;
        }
    }

    fn calculate<'py>(
        &mut self,
        py: Python<'py>,
        rewards: PyReadonlyArray1<f32>,
        obs: PyReadonlyArray4<f32>,
        next_obs: PyReadonlyArray4<f32>,
        actions: PyReadonlyArray1<i64>,
        alignment_config: AlignmentConfig,
        stop_penalty_config: StopPenaltyConfig,
        step_penalty_config: StepPenaltyConfig,
        goal_reward_multiple: Option<f32>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let rewards = rewards.as_array();
        let obs = obs.as_array();
        let next_obs = next_obs.as_array();
        let actions = actions.as_array();

        let n = self.num_agents;
        let shape = obs.shape();
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let cy = (h / 2) as f32;
        let cx = (w / 2) as f32;

        let target_channel_idx = if c >= 4 { 3 } else { 2 };

        let mut total_rewards = Vec::with_capacity(n);

        for i in 0..n {
            let mut r_val = 0.0;
            let action = actions[i];

            // 1. 撞墙检测 (Wall Collision Check)
            // 原理: 如果动作不是 Stay (0)，但 obs 和 next_obs 完全一样，说明撞墙了。
            let mut did_move = true;
            if action != 0 {
                // 简单的差异检测：检查 Target Channel 或者全图
                // 这里为了性能，我们快速检查 Target Channel 的中心区域差异
                let obs_slice = obs.slice(numpy::ndarray::s![i, .., .., ..]);
                let next_obs_slice = next_obs.slice(numpy::ndarray::s![i, .., .., ..]);

                let mut diff_sum = 0.0;
                // 遍历检查差异
                for channel in 0..c {
                    for r in 0..h {
                        for col in 0..w {
                            let d =
                                obs_slice[[channel, r, col]] - next_obs_slice[[channel, r, col]];
                            diff_sum += d.abs();
                            if diff_sum > 0.001 {
                                break;
                            }
                        }
                        if diff_sum > 0.001 {
                            break;
                        }
                    }
                    if diff_sum > 0.001 {
                        break;
                    }
                }

                if diff_sum < 0.001 {
                    did_move = false; // 确实没动
                }
                if action != 0 && !did_move {
                    // r_val -= 0.15;也许应该允许模型去犯错
                }
            }

            // 2. Alignment Reward
            if alignment_config.use_alignment {
                let target_channel = obs.slice(numpy::ndarray::s![i, target_channel_idx, .., ..]);
                let mut target_y_sum = 0.0;
                let mut target_x_sum = 0.0;
                let mut count = 0.0;

                // 计算目标重心
                if let Some(slice) = target_channel.as_slice() {
                    for (idx, &val) in slice.iter().enumerate() {
                        if val > 0.0 {
                            let r = idx / w;
                            let c = idx % w;
                            target_y_sum += r as f32;
                            target_x_sum += c as f32;
                            count += 1.0;
                        }
                    }
                } else {
                    for r in 0..h {
                        for c in 0..w {
                            if target_channel[[r, c]] > 0.0 {
                                target_y_sum += r as f32;
                                target_x_sum += c as f32;
                                count += 1.0;
                            }
                        }
                    }
                }

                if count > 0.0 {
                    let ty = target_y_sum / count;
                    let tx = target_x_sum / count;

                    // 矩阵坐标系: Target - Center
                    // 如果目标在上方(Row 0)，dy 为负；如果目标在右方(Col Max)，dx 为正
                    let dy = ty - cy;
                    let dx = tx - cx;

                    // [动作映射修正]
                    // 但修正 Y 轴符号以匹配矩阵坐标 (Up是负，Down是正)
                    let (ay, ax) = match action {
                        1 => (-1.0, 0.0), // Up:    Row-1, Col不变
                        2 => (1.0, 0.0),  // Down:  Row+1, Col不变
                        3 => (0.0, -1.0), // Left:  Row不变, Col-1
                        4 => (0.0, 1.0),  // Right: Row不变, Col+1
                        _ => (0.0, 0.0),  // Stay
                    };

                    // 只有真的移动了才给奖励！
                    if did_move {
                        let dist = (dy * dy + dx * dx).sqrt();
                        if dist > 0.001 {
                            let ndy = dy / dist;
                            let ndx = dx / dist;
                            let alignment = ndy * ay + ndx * ax;

                            r_val += alignment * alignment_config.value.unwrap_or(0.1);
                        }
                    }
                }
            }

            // 3. Penalties
            if step_penalty_config.use_step_penalty {
                r_val -= step_penalty_config.value.unwrap_or(0.01);
            }
            if stop_penalty_config.use_stop_penalty && action == 0 {
                r_val -= stop_penalty_config.value.unwrap_or(0.01);
            }

            // 4. Env Reward
            let env_reward = rewards[i];
            r_val += env_reward * goal_reward_multiple.unwrap_or(50.0);

            total_rewards.push(r_val);
            self.total_episode_reward[i] += r_val;
        }

        let out_array = PyArray1::from_vec(py, total_rewards);
        Ok(out_array)
    }
    fn get_total_reward<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
        let arr = PyArray1::from_vec(py, self.total_episode_reward.clone());
        Ok(arr)
    }
}

#[pymodule]
fn rust_buffer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustReplayBuffer>()?;
    m.add_class::<RustRewardCalculator>()?;
    Ok(())
}
