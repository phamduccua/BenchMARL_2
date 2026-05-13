#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .ippo_vi import IppoVI, IppoVIConfig, VIOptimizer, VIOptimizerCallback
from .ippo_extragradient import IppoExtragradient, IppoExtragradientConfig, ExtraAdamOptimizer, ExtraGradientCallback
from .ippo_extragradient_adaptive import IppoAdaptiveExtragradient, IppoAdaptiveExtragradientConfig, AdaptiveExtraAdamOptimizer, AdaptiveExtragradientCallback
from .ippo_vi_no_norm import IppoVINoNorm, IppoVINoNormConfig, VIOptimizerNoNorm, VIOptimizerNoNormCallback
from .ippo_vi_no_anchor import IppoVINoAnchor, IppoVINoAnchorConfig, VIOptimizerNoAnchor, VIOptimizerNoAnchorCallback
from .nashconv_callback import NashConvCallback
from .iql import Iql, IqlConfig
from .isac import Isac, IsacConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig
from .qmix import Qmix, QmixConfig
from .vdn import Vdn, VdnConfig

classes = [
    "Iddpg",
    "IddpgConfig",
    "Ippo",
    "IppoConfig",
    "IppoVI",
    "IppoVIConfig",
    "VIOptimizer",
    "VIOptimizerCallback",
    "IppoExtragradient",
    "IppoExtragradientConfig",
    "ExtraAdamOptimizer",
    "ExtraGradientCallback",
    "IppoAdaptiveExtragradient",
    "IppoAdaptiveExtragradientConfig",
    "AdaptiveExtraAdamOptimizer",
    "AdaptiveExtragradientCallback",
    "IppoVINoNorm",
    "IppoVINoNormConfig",
    "VIOptimizerNoNorm",
    "VIOptimizerNoNormCallback",
    "IppoVINoAnchor",
    "IppoVINoAnchorConfig",
    "VIOptimizerNoAnchor",
    "VIOptimizerNoAnchorCallback",
    "NashConvCallback",
    "Iql",
    "IqlConfig",
    "Isac",
    "IsacConfig",
    "Maddpg",
    "MaddpgConfig",
    "Mappo",
    "MappoConfig",
    "Masac",
    "MasacConfig",
    "Qmix",
    "QmixConfig",
    "Vdn",
    "VdnConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "ippo_vi": IppoVIConfig,
    "ippo_extragradient": IppoExtragradientConfig,
    "ippo_extragradient_adaptive": IppoAdaptiveExtragradientConfig,
    "ippo_vi_no_norm": IppoVINoNormConfig,
    "ippo_vi_no_anchor": IppoVINoAnchorConfig,
    "maddpg": MaddpgConfig,
    "iddpg": IddpgConfig,
    "masac": MasacConfig,
    "isac": IsacConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
}
