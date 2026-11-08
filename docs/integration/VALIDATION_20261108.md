# C4 独立项目整合验证（2026-11-08）

源码、配对前端、构建和许可已整合。四rank整模型资格、A/B/A及原服务恢复通过；本次没有引入新的推理优化。全部结果仅本地保存，未公开推送。

| 工作负载 | 并发 | 原C4前后均值 tok/s | 重建版本 tok/s | 差值 |
|---|---:|---:|---:|---:|
| 代码 | C1 | 141.98 | 142.21 | +0.16% |
| 代码 | C4 | 174.01 | 173.43 | -0.33% |
| 文章 | C1 | 93.65 | 94.06 | +0.43% |
| 文章 | C4 | 152.83 | 152.57 | -0.17% |

本轮样本表现接近，支持作为现有基线继续维护；不是新提速或统计等价证明。每阶段每负载仅两次。单路分别生成2048/1024tokens，四路各512tokens，T=0、seed42、thinking关闭，仅Normal做性能测量。C1按首内容到DONE的输出速率计，C4按最早首内容至最后DONE的整体输出速率计，均减去首token；具体公式、逐样本和输出SHA在同名JSON。

重建版本对前、后两轮原版分别比较，单路4/4和四路16/16的输出SHA全部相同。此结论仅限受检采样与工作负载，不扩展为冷/热缓存的位级等价。

通过范围：5项host CTest；含真实tokenizer及ignored项的50项前端测试；11项vision CPU检查；3,359,678项spec-batch控制；168项native24 GPU exact检查（0失败）；Creative GPU的Normal位级恒等、N/C/N回放及FP64参考；vision GPU的embedding、chunk/replay/routing检查。前端58张全分辨率图片测试使用mock encoder/engine，不是58张真实整模型视觉质量评测。

整模型通过17个模式/工具/图像/缓存请求、四路并发、混合长短完成与取消隔离。三个短请求的DONE尾延迟为0.126s, 0.098s, 0.155s；另一个请求取消后同场peer完整完成。四个65,013-token前缀在Normal→Creative→Normal后各复用65,012tokens。

原服务恢复：原引擎与前端SHA、argv、cwd、环境恢复，网关账户维持C4，native held=0、engine=1；vision和四节点温控PID未变。所有本次试验rank/frontend/probe/guard已退出。模型停止至核心恢复约372.1秒，账户准入围栏约588.6秒，两种时长不混用。维护造成活跃KV丢失；既有共享collector未对账状态未改写。

源码归一化：现有快照存在CRLF，Git规范化为LF。290项引擎、前端、vision及相关支持文件输入逐文件核对，原始受检输入与LF SHA均保留在BASELINE.json。LF源码再次从空目录构建并包含完整前端测试，耗时139.47秒。两次可执行文件SHA不同；引擎/前端的CPU .text与.rodata相同，引擎完整GPU SASS dump逐字节相同。变化涉及build-id、符号/字符串表和内嵌CUDA镜像；不宣称完整文件逐字节可复现，没有为LF另做一次同口径C4停服重跑；随后用户要求的[C8补测](C8_REFERENCE_20261108.md)已用这份LF构建实际运行四rank并通过。

steering转换器从原SHA固定JSON得到819,216B的DSCR0001文件，与当前实际资产SHA相同。数据仍在私有路径，仓库只含转换代码。构建工具链：CUDA13.0.88、g++13.3、CMake3.28.3、Rust/Cargo1.99.1；vision使用torch2.13.0+cu130、numpy2.3.5、Pillow12.3.0、safetensors0.8.0。

保留的失败与未测项：vision测试容器最初一次权限错误、一次缺少/ref只读挂载导致2项报错，修正挂载后11项全过，原日志保留。Rust有一条既有fetch_update弃用警告。C2/C8本轮未复测；完整4×1M耐久、广泛长图质量、长agent任务及冷/热位级等价没有新增验收。scale回收、W4A4、NVFP4权重和后续10%优化不属于本次基线。

原始日志、intent、完整环境和恢复回执保存在Git忽略的 `.local/`；同名JSON固定其SHA。纳入Git的维护源码不含Key、家庭fleet、模型、steering数据或私有工作负载。
