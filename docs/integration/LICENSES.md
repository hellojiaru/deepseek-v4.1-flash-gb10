# 许可与来源清单

保留原作者版权声明、NOTICE及固定来源引用；独立发布采用经隐私检查的新根提交，旧集成历史仅在本地保留。本项目沿用上游许可，不将第三方实现改署为本项目原创。文件来源及内容 SHA 以 BASELINE.json 为准。

| 材料 | 许可/保留文件 |
|---|---|
| dsgb10/dsgx10 原生引擎、前端及本地整合工具 | 根目录 LICENSE（MIT）；原作者版权保留 |
| DeepSeek 参考代码、编码、vision 衍生文件 | `ds41f-4xgx10-switchless/ref/LICENSE-deepseek`（MIT） |
| 参考 DeepInfer 重实现的概率采样部分 | `LICENSES/Apache-2.0-DeepInfer.txt`，具体来源和修改见 NOTICE |
| token-map 生成器参考的 NumPy/PCG64 逻辑 | `LICENSES/BSD-3-Clause-NumPy.txt`、`LICENSES/PCG64.txt` |
| tokenizer 的 Unicode 16 数据表 | `LICENSES/Unicode-V3.txt`，原数据版权及生成器归属保留 |
| 历史 golden transcript 中 CPython 标准库摘录 | `LICENSES/PSF-Python-3.12.txt` |
| pi coding-agent 的历史基准材料 | `ds41f-4xgx10-switchless/bench/THIRD_PARTY.md`（MIT） |

本次补入的许可证原文 URL、取得日期和 SHA 在 LICENSE_SOURCES.json。它们补充既有归属，不替换源码中的声明。Apache 文件来自已有分支，未修改其条款。

Rust crate 由 Cargo.lock 固定，PyTorch、CUDA、libibverbs、Python/Pillow 等依赖需另行安装并遵守各自条款；本仓库没有重新分发这些运行库。模型权重和tokenizer由使用者单独提供，随附的可选Creative v1方向按其目录内MIT许可证提供，保留DeepSeek归属，见资产说明与清单。用户自带或重新拟合的资产分别保留其许可，不因本项目代码许可而改变。steering转换入口只处理本地已有资产，不自动下载或训练。

完整DeepSeek权重、tokenizer和CUDA运行库不随仓库分发；随附的小型方向二进制不是完整模型权重。预发布构建与隐私检查见[发布检查](../PUBLICATION.md)。
