# Third-Party Notices

This repository includes or adapts software and experimental protocols from the projects listed below. The project-level `LICENSE` applies to the original contributions of the LLM-PeerReview authors. Third-party components remain subject to their respective terms.

## GaC

- Upstream: https://github.com/yaoching0/GaC
- Revision: `bbb2567b9445504b2fa0421d440c95123ed9bf07`
- Local components: `GaC/gac_api_server.py` and `GaC/utils/`
- License: MIT; see `LICENSES/GaC-MIT.txt`

## Smoothie

- Upstream: https://github.com/HazyResearch/smoothie
- Revision: `8f8bd153058d3551955cf9fd62a501ee1e6a324f`
- Local component: `Utils/smoothie_model.py`, with project-specific integration in `Src/baseline/smoothie.py` and `Utils/smoothie_util.py`
- License: MIT; see `LICENSES/Smoothie-MIT.txt`

## CoRE

- Upstream: https://github.com/zhichenz98/CoRE-EACL26
- Revision: `2da4da7814cd16d8178caa970a28245664697bfc`
- Local adaptation: `Src/baseline/core_generate.py`, `Utils/core_decode.py`, and `Utils/core_token_map.py`
- License: MIT; see `LICENSES/CoRE-MIT.txt`

## Speculative Decoding

- Upstream: https://github.com/romsto/Speculative-Decoding
- Revision: `db7db67604359d0380c330c240c276ec1f85ba65`
- Local adaptation: `SAFE/caching.py`
- License: MIT; see `LICENSES/Speculative-Decoding-MIT.txt`

## SAFE

- Upstream: https://github.com/yoon6503/SAFE
- Revision: `1c8dc8d25e3cebc299402fb1de913cae0db3c87c`
- Local components: `SAFE/`, with project-specific integration in `Src/baseline/safe_adapter.py`, `Src/baseline/safe_core.py`, and `Src/baseline/safe_generate.py`
- The upstream repository does not publish a license. No license is asserted here on behalf of its authors.

## Multi-Agent Debate

- Upstream: https://github.com/composable-models/llm_multiagent_debate
- Revision: `9846749350eb917ae5bfaaff4c645fc705b8d3af`
- Local adaptation: `Src/baseline/debate.py` and `Src/evaluate/evaluate_debate.py`
- The upstream repository does not publish a license. No license is asserted here on behalf of its authors.

## Agent-Forest

- Upstream: https://github.com/MoreAgentsIsAllYouNeed/AgentForest
- Revision: `efb0eb4ea4fc86e9c6cf022294ce632960c7afbc`
- The local implementation in `Src/baseline/agent_forest.py` and `Utils/agent_forest_util.py` is project-specific; upstream source files are not included.
