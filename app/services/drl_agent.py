"""
DQN 在线学习层 - 渐进式强化学习增强 LightGBM
架构: Q-Network + Experience Replay Buffer + 动态奖励调整
纯 NumPy 实现，无需 PyTorch 依赖
"""
import json
import os
import sys
import random
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg

# ─────────────────────────────────────────────────────────────
# 特征定义（与 LightGBM 保持一致）
# ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "crystal_ppm", "ct_ratio_err", "clock_accuracy_sd", "factory_fault",
    "initial_error_max", "final_error_max", "error_degradation_rate", "clock_drift_rate",
    "clock_daily_error_max", "error_variability_max", "error_consistency_max",
    "energy_balance_ratio", "energy_peak_ratio", "rs485_total_errors",
]


class ReplayBuffer:
    """经验回放缓冲区 - 存储 (state, action, reward, next_state, done)"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNetwork:
    """
    深度 Q 网络 - 纯 NumPy 实现
    三层全连接网络
    """
    
    def __init__(self, state_dim: int, action_dim: int = 2, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_torch = False
        
        # Xavier 初始化
        scale = np.sqrt(2.0 / state_dim)
        self.W1 = np.random.randn(state_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, action_dim) * scale
        self.b3 = np.zeros(action_dim)
        
        # 目标网络
        self.target_W1, self.target_b1 = self.W1.copy(), self.b1.copy()
        self.target_W2, self.target_b2 = self.W2.copy(), self.b2.copy()
        self.target_W3, self.target_b3 = self.W3.copy(), self.b3.copy()
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def predict(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        """预测 Q 值"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        if target:
            h1 = self._relu(state @ self.target_W1 + self.target_b1)
            h2 = self._relu(h1 @ self.target_W2 + self.target_b2)
            return h2 @ self.target_W3 + self.target_b3
        else:
            h1 = self._relu(state @ self.W1 + self.b1)
            h2 = self._relu(h1 @ self.W2 + self.b2)
            return h2 @ self.W3 + self.b3
    
    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """更新网络参数"""
        batch_size = len(states)
        lr = 0.001
        total_loss = 0.0
        
        for i in range(batch_size):
            s = states[i]
            a = int(actions[i])
            r = float(rewards[i])
            s_next = next_states[i]
            done = float(dones[i])
            
            # 前向传播
            h1 = self._relu(s @ self.W1 + self.b1)
            h2 = self._relu(h1 @ self.W2 + self.b2)
            q_vals = h2 @ self.W3 + self.b3
            
            # 目标 Q 值
            h1_t = self._relu(s_next @ self.target_W1 + self.target_b1)
            h2_t = self._relu(h1_t @ self.target_W2 + self.target_b2)
            next_q_vals = h2_t @ self.target_W3 + self.target_b3
            max_next_q = float(np.max(next_q_vals))
            target_q = r + gamma * (1 - done) * max_next_q
            
            # 梯度更新
            delta = target_q - q_vals[a]
            self.W3[:, a] += lr * delta * h2
            self.b3[a] += lr * delta
            
            grad_h2 = self.W3[:, a] * delta
            self.W2 += lr * np.outer(h1, grad_h2 * self._relu_deriv(h1 @ self.W2 + self.b2))
            self.b2 += lr * grad_h2 * self._relu_deriv(h1 @ self.W2 + self.b2)
            
            grad_h1 = self.W2 @ grad_h2 * self._relu_deriv(h1)
            self.W1 += lr * np.outer(s, grad_h1 * self._relu_deriv(s @ self.W1 + self.b1))
            self.b1 += lr * grad_h1 * self._relu_deriv(s @ self.W1 + self.b1)
            
            total_loss += delta ** 2
        
        return total_loss / max(batch_size, 1)
    
    def update_target_network(self):
        """更新目标网络 (soft update)"""
        tau = 0.005
        self.target_W1 = tau * self.W1 + (1 - tau) * self.target_W1
        self.target_b1 = tau * self.b1 + (1 - tau) * self.target_b1
        self.target_W2 = tau * self.W2 + (1 - tau) * self.target_W2
        self.target_b2 = tau * self.b2 + (1 - tau) * self.target_b2
        self.target_W3 = tau * self.W3 + (1 - tau) * self.target_W3
        self.target_b3 = tau * self.b3 + (1 - tau) * self.target_b3
    
    def save(self, path: Path):
        """保存模型"""
        joblib.dump({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'target_W1': self.target_W1, 'target_b1': self.target_b1,
            'target_W2': self.target_W2, 'target_b2': self.target_b2,
            'target_W3': self.target_W3, 'target_b3': self.target_b3,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
    
    def load(self, path: Path):
        """加载模型"""
        data = joblib.load(path)
        self.W1, self.b1 = data['W1'], data['b1']
        self.W2, self.b2 = data['W2'], data['b2']
        self.W3, self.b3 = data['W3'], data['b3']
        self.target_W1, self.target_b1 = data['target_W1'], data['target_b1']
        self.target_W2, self.target_b2 = data['target_W2'], data['target_b2']
        self.target_W3, self.target_b3 = data['target_W3'], data['target_b3']


class DRQLearningAgent:
    """
    DQN 在线学习智能体
    
    工作流程:
    1. 接收新样本 (特征, 真实标签)
    2. 计算奖励 (与 LightGBM 预测对比)
    3. 存入经验回放
    4. 定期从回放中采样学习
    5. 输出置信度调整后的预测
    """
    
    def __init__(
        self,
        state_dim: int = 14,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 32,
        update_freq: int = 4,
        target_update_freq: int = 100,
    ):
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        
        self.q_network = DQNetwork(state_dim, action_dim=2, hidden_dim=64)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        self.total_steps = 0
        self.episode_rewards = []
        self.reward_history = deque(maxlen=100)
        self.target_reward = 5.0
        
        self.model_dir = cfg.ML_MODELS_DIR / "drl"
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "dqn_agent.pkl"
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy 策略选择动作"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 1)
        q_values = self.q_network.predict(state)
        return int(np.argmax(q_values))
    
    def compute_reward(
        self,
        lgbm_prob: float,
        drl_action: int,
        true_label: int,
        state: np.ndarray,
        lgbm_confidence: float,
    ) -> float:
        """动态奖励函数"""
        reward = 0.0
        
        # 分类准确性奖励
        if drl_action == true_label:
            reward += 10.0
        else:
            reward -= 10.0
        
        # 一致性奖励
        lgbm_pred = 1 if lgbm_prob >= 0.5 else 0
        if drl_action == lgbm_pred:
            reward += 2.0
        
        # 置信度调整奖励
        if lgbm_confidence > 0.8:
            if drl_action == lgbm_pred:
                reward += 3.0 * lgbm_confidence
            else:
                reward -= 2.0 * lgbm_confidence
        
        # 特征异常奖励
        anomaly_indicators = self._check_anomaly_indicators(state)
        if anomaly_indicators > 2:
            if drl_action == 1:
                reward += 1.0 * anomaly_indicators
            else:
                reward -= 0.5 * anomaly_indicators
        
        # 探索奖励
        if self.epsilon > 0.05:
            reward += 0.5
        
        return reward
    
    def _check_anomaly_indicators(self, state: np.ndarray) -> int:
        """检查特征异常指标数量"""
        indicators = 0
        if state[0] > 15: indicators += 1
        if state[1] > 0.8: indicators += 1
        if state[6] > 0.1: indicators += 1
        if state[7] > 0.3: indicators += 1
        if state[13] > 0: indicators += 1
        return indicators
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
        self.episode_rewards.append(reward)
    
    def update(self) -> Optional[float]:
        """执行一次学习更新"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        loss = self.q_network.update(states, actions, rewards, next_states, dones, self.gamma)
        
        if self.total_steps % self.target_update_freq == 0:
            self.q_network.update_target_network()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.reward_history.append(np.mean(self.episode_rewards[-self.batch_size:]))
        return loss
    
    def learn_from_labeled_sample(self, features, true_label, lgbm_prob, lgbm_confidence) -> dict:
        """从标注样本中学习"""
        state = features.astype(np.float32)
        action = true_label
        drl_action = self.get_action(state, training=True)
        reward = self.compute_reward(lgbm_prob, drl_action, true_label, state, lgbm_confidence)
        
        self.store_transition(state, action, reward, state, done=True)
        loss = self.update()
        
        return {
            "reward": reward,
            "loss": loss,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_reward": float(np.mean(self.reward_history)) if self.reward_history else 0,
        }
    
    def predict_with_confidence(self, features, lgbm_prob, lgbm_confidence) -> dict:
        """使用 DQN 预测并输出置信度调整"""
        state = features.astype(np.float32)
        
        drl_action = self.get_action(state, training=False)
        q_values = self.q_network.predict(state)[0]
        q_confidence = float(abs(q_values[1] - q_values[0]))
        
        lgbm_pred = 1 if lgbm_prob >= 0.5 else 0
        
        if drl_action == lgbm_pred:
            final_pred = lgbm_pred
            confidence = 0.7 * lgbm_confidence + 0.3 * min(q_confidence, 1.0)
            agreement = True
        else:
            final_pred = lgbm_pred
            confidence = 0.4 * lgbm_confidence
            agreement = False
        
        level = "高" if confidence >= 0.7 else ("中" if confidence >= 0.4 else "不确定")
        
        return {
            "drl_action": drl_action,
            "drl_q_values": q_values.tolist(),
            "drl_confidence": q_confidence,
            "lgbm_pred": lgbm_pred,
            "final_pred": final_pred,
            "confidence": confidence,
            "risk_level": level,
            "agreement": agreement,
        }
    
    def get_agent_stats(self) -> dict:
        """获取智能体统计信息"""
        return {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "avg_reward": float(np.mean(self.reward_history)) if self.reward_history else 0,
            "reward_std": float(np.std(self.reward_history)) if len(self.reward_history) > 1 else 0,
            "target_update_freq": self.target_update_freq,
            "has_torch": False,
        }
    
    def save(self):
        """保存智能体状态"""
        self.q_network.save(self.model_path)
        meta = {
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "reward_history": list(self.reward_history),
        }
        with open(self.model_dir / "dqn_meta.json", "w") as f:
            json.dump(meta, f)
    
    def load(self):
        """加载智能体状态"""
        if self.model_path.exists():
            self.q_network.load(self.model_path)
        meta_path = self.model_dir / "dqn_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.total_steps = meta.get("total_steps", 0)
            self.epsilon = meta.get("epsilon", self.epsilon)
            self.reward_history = deque(meta.get("reward_history", []), maxlen=100)


# ─────────────────────────────────────────────────────────────
# 全局智能体实例
# ─────────────────────────────────────────────────────────────
_agent: Optional[DRQLearningAgent] = None


def get_agent() -> DRQLearningAgent:
    """获取全局 DQN 智能体实例"""
    global _agent
    if _agent is None:
        _agent = DRQLearningAgent()
        _agent.load()
    return _agent


def reset_agent():
    """重置智能体"""
    global _agent
    _agent = DRQLearningAgent()
    return _agent


def features_to_array(features: dict) -> np.ndarray:
    """将特征字典转换为 numpy 数组"""
    return np.array([
        features.get("crystal_ppm", 9.0),
        features.get("ct_ratio_err", 0.5),
        features.get("clock_accuracy_sd", 0.018),
        float(features.get("factory_fault", False)),
        features.get("initial_error_max", 0.0),
        features.get("final_error_max", 0.0),
        features.get("error_degradation_rate", 0.0),
        features.get("clock_drift_rate", 0.0),
        features.get("clock_daily_error_max", 0.22),
        features.get("error_variability_max", 0.006),
        features.get("error_consistency_max", 0.086),
        features.get("energy_balance_ratio", 1.0),
        features.get("energy_peak_ratio", 0.246),
        float(features.get("rs485_total_errors", 0)),
    ], dtype=np.float32)


def predict_hybrid(features: dict, lgbm_prob: float, lgbm_confidence: float) -> dict:
    """LightGBM + DQN 混合预测"""
    agent = get_agent()
    x = features_to_array(features)
    result = agent.predict_with_confidence(x, lgbm_prob, lgbm_confidence)
    
    return {
        "drl_risk_prob": float(result["drl_q_values"][1] / sum(result["drl_q_values"])),
        "drl_confidence": result["drl_confidence"],
        "final_confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "agree_with_lgbm": result["agreement"],
        "agent_stats": agent.get_agent_stats(),
    }


def learn_from_new_label(features: dict, true_label: int, lgbm_prob: float, lgbm_confidence: float) -> dict:
    """从新标注样本学习"""
    agent = get_agent()
    x = features_to_array(features)
    result = agent.learn_from_labeled_sample(x, true_label, lgbm_prob, lgbm_confidence)
    
    if agent.total_steps % 100 == 0:
        agent.save()
    
    return {
        "learn_success": True,
        "reward": result["reward"],
        "loss": result["loss"],
        "total_learned": agent.total_steps,
        "epsilon": result["epsilon"],
    }
