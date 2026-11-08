# 维护这条 C4 基线

项目主入口是根目录 README、`profiles/c4.json`、`tools/build.py` 和 `tools/serve_original.py`。默认不加载向量；可选调节必须通过`tools/run.py --enable-steering`显式启用并绑定资产scope。`ds41f-4xgx10-switchless`保留原生项目布局，便于追溯上游；其它历史目录、旧benchmark数字和service脚本保留为历史材料，不能直接代表当前受检配置。

源代码更新时同时核对 C++ `engine/common/frontend_protocol.h` 和 Rust `frontend/src/protocol.rs`，当前为协议5。引擎、前端、tokenizer、steering scope、vision revision 是配对关系。换掉其中之一，就不能沿用旧的整组能力验收结果。

本轮没有改变推理算法。native24、四槽、单槽 1,048,576 tokens、概率 SPEC、原版权重和图片历史 tail-exact 策略均来自当前服务。scale 回收、W4A4、NVFP4 整包权重仍在另行验证范围，不混入基线。

构建从空目录开始，使用固定 Cargo.lock，保存 BUILD_RESULT.json 中的命令、源 SHA、产物 SHA、工具链版本及测试退出。复用下载的依赖源码不等于复用旧的二进制；不允许把历史对象库作为干净构建的输入。真实 tokenizer 测试和 vision CPU 测试另外记录，避免将缺少数据后提前返回的测试误计为完整覆盖。

四机测试应保留以下证据：

- 原服务每个进程的 PID、创建时间、完整 argv、cwd、环境及二进制 SHA，相关配对前端和 vision 身份。
- 新鲜的原生与网关在途/排队状态，以及已知其它任务的终态；目录可见或暂时没有输出不代表空闲。
- 每节点有限内存、CPU、磁盘和绝对超时预算；完整模型不双驻留，停止与恢复只作用于本次归属明确的服务。
- 原 C4 → 重建 C4 → 原 C4 的同口径数据，分开报告单路、四路、TTFT、完成尾延迟和输出 SHA。
- 原服务恢复后的参数、SHA、模式、目录和准入状态，以及所有失败/部分日志。重启导致的活跃 KV 丢失必须记录。

`tests/integration/probe.py` 只发合成任务到明确指定的端点：

```sh
python3 tests/integration/probe.py http://127.0.0.1:18894/v1 /abs/api-key-file /abs/result.json candidate
```

`baseline-before` 和 `baseline-after` 仅运行同口径性能样本，`candidate` 额外检查混合长短请求、取消、工具、图像、模式切换和四个约65K前缀的复用。它是有界资格检查，不等于长 agent 全覆盖、完整4×1M耐久或图像质量基准。正常和 Creative 不重复做完整性能矩阵，只做切换及必要功能检查。

家庭集群地址、私有 fleet、API Key、原始标定记录和停止恢复适配器放在 Git 忽略的 `.local/`，不进入维护源码。独立维护不改变原作者归属；更新来源时同步更新 NOTICE、许可证副本及 BASELINE.json。新代码、模型权重与外部依赖各自遵守其许可，不将它们统一改署为本项目原创。

向量发布范围：公开方案、源码、格式及本地转换方法，随附Creative v1作为实验性可选资产，二进制、可读JSON、清单和许可证成对维护。默认原始模型保持，任何资产均需用户主动启用。原始私有提示和采集记录不公开，历史拟合不可完整重做的限制必须披露。说明文档须区分已实现的加载/投影与尚未提供的一键整模型激活采集、任意强度API及新拟合方向的质量验证。

2026-11-09新增验收：原始入口真实四rank启动与前后Normal对照、历史普通单任务审计、三次冷prefill、长prefill短请求干扰、C2参考。结果索引在[BENCHMARKS.md](BENCHMARKS.md)。CPU CI仅覆盖新启动策略和资产格式，不把它写成CUDA或整模型资格。双语README的摘要必须与这些机器可读结果同步。

发布入口是独立仓库`hellojiaru/deepseek-v4.1-flash-gb10`。只推送经过检查的发布分支与明确的release标签，私有集成历史保留在本地，不推送全部分支或旧标签。发布不等于部署新优化；未验证优化仍需单独评测。
