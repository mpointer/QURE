"""
QURE Learning Agent - Thompson Sampling Policy Optimization

Implements contextual multi-armed bandit using Thompson Sampling to continuously
optimize policy weights based on real-world outcomes.
"""

from .learning_agent import LearningAgent
from .thompson_sampling import ThompsonSamplingBandit
from .logging_pipeline import DecisionLogger
from .reward_shaper import RewardShaper, get_reward_shaper
from .context_clusterer import ContextClusterer
from .drift_detector import DriftDetector
from .counterfactual_evaluator import CounterfactualEvaluator

__version__ = "1.0.0"

__all__ = [
    'LearningAgent',
    'ThompsonSamplingBandit',
    'DecisionLogger',
    'RewardShaper',
    'get_reward_shaper',
    'ContextClusterer',
    'DriftDetector',
    'CounterfactualEvaluator',
]
