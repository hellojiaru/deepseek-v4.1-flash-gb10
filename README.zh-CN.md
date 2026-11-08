# DeepSeek GB10 Runtime

**在四台GB10上运行完整DeepSeek-V4.1-Flash，为长上下文Agent提供快速、本地的推理服务。**

[English](README.md) · [安装运行](docs/BUILD_AND_RUN.md) · [性能与测试方法](docs/BENCHMARKS.md) · [同类方案对比](docs/COMPARISON.md)

使用官方1017权重，保留原生混合精度，通过OpenAI兼容接口接入编程Agent或聊天客户端。支持图片、工具调用和历史缓存，默认启动原始模型，不加载额外向量，也不要求使用特定Agent框架。

| 日常Agent生成 | 读取全新长输入 | 同时保留的工作上下文 |
|---|---|---|
| **典型约110 tokens/s** | **约9,600 tokens/s** | **8路×1M tokens**，已实际填充并生成 |

约110反映日常长响应的典型生成体验；长输入速度包含前端处理时间。测试条件和完整平均值在下面分别说明。

## 适合怎样使用

- **让Agent阅读大型项目。** 262,145 tokens源码输入，在没有缓存命中的情况下，27.28秒开始回答；后续工具往返可以继续复用已有历史。
- **一个人持续做长任务。** 既有实跑包含编程、三维与Canvas应用、文件处理和方案研究。受控单路代码生成约142 tokens/s。
- **让几个Agent共享一套模型。** 默认四路，也可启用八路。八路代码／文章总吞吐约212／181 tokens/s，单独使用时速度基本保持。
- **保留完整工作流。** 流式输出、思考、工具调用、图片输入和缓存统计通过同一个API提供。工具由客户端执行，推理引擎本身不依赖DSH。

目标硬件是**4台GB10、每台128GB统一内存**，通过支持的ConnectX-7 RDMA ring连接。项目面向这套硬件和模型做优化，暂不承诺其它拓扑即装即用。

## 性能

C1表示一个请求，C4／C8表示四个／八个活跃请求。并发成绩是所有请求的合计输出速度。

| 使用状态 | 代码 | 文章 |
|---|---:|---:|
| 默认C4配置，单路使用 | **142.2 tokens/s** | **94.1 tokens/s** |
| 两路同时使用，总吞吐 | **148.8 tokens/s** | **127.6 tokens/s** |
| 四路同时使用，总吞吐 | **173.4 tokens/s** | **152.6 tokens/s** |
| 八路同时使用，总吞吐 | **212.2 tokens/s** | **180.8 tokens/s** |
| C8配置，单路使用 | 142.3 tokens/s | 93.9 tokens/s |

这组受控测试使用temperature 0、关闭思考、两次重复（C2三次）。单路代码／文章输出分别为2,048／1,024 tokens，并发每路512 tokens。推测解码的接受率随内容变化，因此代码、文章与真实Agent任务分别报告。八路提高的是合计吞吐，不是让八个人都获得单人独占的速度。

### 长输入读得有多快

| 全新源码输入 | 开始回答的等待时间 | 输入处理速度 |
|---|---:|---:|
| 65,537 tokens | 7.06秒 | 9,281 tokens/s |
| 262,145 tokens | 27.28秒 | 9,611 tokens/s |

每个长度重复三次，缓存命中均为0。这里从请求发出计到首个内容，包含前端、tokenization和通信，更接近用户实际等待的时间。[测试细节](docs/BENCHMARKS.md)保留原始数字和重复结果。

### 真实长任务的体验

向前审计10月18日至31日的8个普通长任务，共1,589条有效模型响应、255万输出tokens。1,056条至少256 tokens的较长响应中，中位速度为**108.7 tokens/s**，232条达到130 tokens/s以上。全部有效响应按生成时间加权为**103.4 tokens/s**，各任务整体为**95.9～122.2 tokens/s**。这些任务与近期CORAL多并发、四Agent同时开发测试分开统计，慢响应也保留。

另一个C4生产实例在计划维护前连续运行**123.6小时**，对应16个会话、5,905条响应和397万输出tokens，覆盖工具往返、长历史复用和自动压缩。进程持续时间包含空闲，不把它称为五天持续满载压测。

[日常单任务明细](docs/benchmarks/ordinary-agents-20261109.md) · [后续生产实跑记录](docs/benchmarks/real-workloads-20261108.md)

## 上下文容量

| 配置 | 活跃请求 | 每路窗口 | 总活跃窗口 |
|---|---:|---:|---:|
| 默认配置 | 4 | 1,048,576 tokens | 4,194,304 tokens |
| 可选C8 | 8 | 1,048,576 tokens | **8,388,608 tokens** |

C8实测每路输入1,048,320 tokens，再生成256 tokens，八路全部完成，24处位置标记全部召回。最紧张节点仍有12.6 GiB可用host内存，无新增swap、OOM kill或磁盘缓存spill。

**8M是八个1M窗口的合计，单条请求仍是1M。** 输入、图片展开、工具历史和输出共同使用窗口。

八条完全冷的1M输入整组耗时约14.6分钟。当前prefill依次处理，最后一路约862秒开始输出；已有流也可能等待长prefill结束。日常Agent的缓存续接与这种全冷同时到达不同。一次满容量测试已完成，长期满窗耐久和全面1M答案质量仍待扩大验证。

[满8M容量实测](docs/benchmarks/capacity-8m-20261108.md)

## 从原始模型开始

需要Linux/aarch64、CUDA 13、CMake≥3.28、C++20、libibverbs、Rust/Cargo、Python和已配置的四机ring。图片功能还需vision worker的PyTorch依赖。先按[模型准备](docs/MODEL_SETUP.md)生成各rank的DSW与Engram文件。

在仓库根目录构建，再将构建产物和启动入口分发到四个节点：

```sh
python3 tools/build.py --build-dir /opt/deepseek-build --jobs 2
```

复制[示例fleet](ds41f-4xgx10-switchless/fleet/homelab.conf)，填写自己的主机、RDMA和模型路径。在对应节点依次启动rank 1、2、3，最后启动rank 0。例如在rank 1节点：

```sh
python3 tools/serve_original.py rank \
  --rank 1 --fleet /opt/deepseek/fleet.conf \
  --build-dir /opt/deepseek-build --runtime-dir /opt/deepseek-runtime \
  --spill-dir /opt/deepseek-spill --disk-tag my-deepseek
```

默认四槽。需要八槽时，确认各节点余量后，为每个rank增加`--profile c8-reference`。先排空已有模型，避免两套完整模型同时驻留。

在rank 0启动vision worker与配对API前端，完整命令见[安装运行](docs/BUILD_AND_RUN.md)。原始模型入口拒绝向量参数，并清除父环境中残留的向量配置，不需要任何额外向量资产。

安装OpenAI Python SDK后可这样请求，也可以配置支持Chat Completions的Agent客户端：

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:18893/v1",
    api_key=Path("/opt/deepseek/api-key").read_text().strip(),
)
reply = client.chat.completions.create(
    model="deepseek-v4.1-flash-dsgb10-official-1017",
    messages=[{"role": "user", "content": "帮我分析这个项目。"}],
    max_tokens=4096,
)
print(reply.choices[0].message.content)
```

服务本身不依赖该SDK。[API说明](docs/API.md)列出采样、思考、工具、图片和缓存参数。

## 与其它方案怎么选

如果最在意长资料读取、单人编程Agent，以及少量Agent共享服务，这个方案值得考虑。若需求是大量并发、强制JSON/schema或多种硬件部署，同类项目也有更成熟的能力。详见[当前对比与待补项](docs/COMPARISON.md)。没有同提示、同硬件的配对测试时，不把公开数字的并排展示说成全面领先。

目前需要留意：

- 大段冷prefill会影响同池其它请求，尚无完整的分块让路调度。
- 可生成JSON、解析工具调用，但不提供grammar强制的JSON schema、regex或工具参数约束。
- 长图、长图片历史和全面1M质量需要专门评测。
- 活跃KV不能跨进程重启保留。本基线没有启用额外量化、W4A4或scale回收。

## 可选：用向量调整模型行为

前面的默认启动保留原始模型行为。希望调整回答风格或拒答倾向时，可以自愿启用**激活向量调节**：用少量方向向量改变模型内部激活，权重文件保持不变，也不需要额外驻留一套完整模型。

提供并显式启用资产后，同一API可以按请求选择：

- **Normal**：关闭向量作用，使用原始路径。
- **Creative**：应用用户提供的方向。这个名称表示一条调节路径，不是更高temperature，也不保证所有创作任务更好。

模式切换无需重新加载完整模型，在安全的执行边界生效，历史缓存按模式分开。仓库另附**实验性可选[Creative v1资产](assets/steering/creative-v1/README.md)**，同时提供小型二进制和可读数值JSON，默认入口不会加载。资产说明如实记录历史校准来源与局限。用户也可以在本地生成、检查并自愿启用自己的方向，观察它对回答与拒答的影响。

[向量调节方案详解](docs/STEERING.zh-CN.md)说明投影原理、方向拟合、透明格式、接入、缓存身份及效果验证。当前开启强度固定为1.5；更换向量需使用新的资产身份并重启，尚无按请求任意调强度的接口。

## 来源与许可证

基于[blaesus/dsgb10](https://github.com/blaesus/dsgb10)，保留原生C++/CUDA引擎、RDMA架构与原作者归属。本项目整合配对前端、native24、完成通知修复、概率采样及视觉支持，提供可独立构建的源码。

详见[LICENSE](LICENSE)、[NOTICE](NOTICE)与[第三方许可证](docs/integration/LICENSES.md)。DeepInfer派生采样代码保留Apache-2.0条款；模型权重和外部资产分别遵守其许可。
