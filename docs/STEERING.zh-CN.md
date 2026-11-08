# 可选激活向量调节：原理、生成与接入

[English](STEERING.md)

## 本项目发布什么

发布**方案说明、投影与加载源码、数据格式和本地转换工具**，另附需要主动启用的[Creative v1实验资产](../assets/steering/creative-v1/README.md)：二进制、可读数值JSON、SHA清单及MIT许可证。不会自动下载或加载；用户也可以用自己的数据生成和检查本地向量。

默认仍使用`tools/serve_original.py`启动原始模型，不加载向量。该入口拒绝向量参数，并清除父环境中残留的向量配置。

随附JSON可以逐字节复现二进制。原始校准提示与激活采集记录没有随附，因此不能从本仓库完整重做历史拟合。方案与源码说明参数怎样产生、在哪里起作用。SHA只能证明文件身份，不能证明向量的含义、质量或使用许可。

## Normal和Creative实际做什么

它调整的是推理过程中的attention输出，不改写模型权重文件，也不驻留第二套完整模型。设激活为`y`、单位方向为`r`、强度为`λ`，实际运算是：

```text
y_new = y - λ × dot(y, r) × r
```

| 模式 | 当前强度 | 行为 |
|---|---:|---|
| Normal | 0 | 投影kernel在读取或改写激活前直接返回 |
| Creative | 1.5 | 使用用户提供的方向进行投影 |

把激活分为沿着方向的部分和垂直部分，就有`y_new = y_垂直 + (1-λ)×y_平行`。当λ=1.5，平行分量变为原来的−0.5倍。这不是提高temperature，也不自动代表创作能力更强或拒答率必然降低。λ=0和1.5是当前支持的两个状态，公式不表示API已经开放任意强度。

当前prefill与decode都使用方向。decode在本rank的attention `wo_b`输出之后、TP求和与mHC融合之前投影；prefill在四rank求和之后投影。线性投影与求和在精确算术下可交换，但BF16舍入顺序可以不同。不能在别的隐藏状态位置提取方向后，未经验证就认为作用完全相同。

模型原始权重保持不变，这套机制不为DSpark草稿另拟一组方向。两种模式在安全执行边界切换，共享同一物理计算池；历史缓存按模式分开，保留的原模式历史可继续复用。

## 怎样从自己的数据获得方向

历史标定采用“配对激活差逐项归一化，再提取未中心化第一主方向”的方法。下面说明可复现的实验流程，不提供现成校准集或预先保证效果的参数。

1. **先定义要改变的行为。** 例如减少对本来可以正常作答的问题的误拒答。人工核对实际回答，不能把更长的回答直接记为成功；事实正确性与行为变化分别评价。
2. **准备匹配的提示对。** 使用自写或取得适当许可的数据，尽量匹配任务、语言、长度和结构，覆盖多个领域并保留无须改变的对照。不能仅凭不同主题的激活差就认定找到了某种行为方向。按主题和措辞保留独立验证集。
3. **在原始模型上采集激活。** 关闭向量，在相同hook、token位置或池化规则、数值配置下采集每个提示的激活摘要。记录checkpoint、tokenizer/模板、hook位置、池化方法、dtype、rank语义及源码身份。普通聊天API只返回文本，不返回这些内部激活；需要本地参考模型或明确的采集构建。**当前仓库没有提供一条命令完成整模型激活采集的入口。**
4. **计算并归一化差方向。** 对提示对p和层l，计算`d[p,l] = h[A,p,l] - h[B,p,l]`。剔除零差和无效值，再将每个有效的pair-layer差方向归一化。
5. **提取共享主方向。** 历史方法汇总中层10～29，不减均值，取第一右奇异向量并归一化，再复制到40层条目。格式也允许每层各用不同方向，但那是需要另行验证的新实验。
6. **保留可读参数与记录。** 保存JSON、模型与样本身份、独立提示数量和拟合选择。多个层的向量行不等于更多独立问题；第一主成分能量高低也不能单独证明行为更好。
7. **通过独立任务决定是否采用。** 对照原始与调节后的回答，再检查日常准确性、代码执行、工具、图片、思考、长历史和模式切换，完整保留退化项。误拒答减少不等于事实更准确，也不等于整体创作水平提高。

以下是离线拟合示意。`H`应当已经是按一致规则采集的激活摘要，形状为`[提示对数量, 2, 40, 5120]`：

```python
import numpy as np

H = np.asarray(H, dtype=np.float64)
D = (H[:, 0, 10:30, :] - H[:, 1, 10:30, :]).reshape(-1, 5120)
norms = np.linalg.norm(D, axis=1)
valid = np.isfinite(D).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
D = D[valid] / norms[valid, None]
if len(D) == 0:
    raise ValueError("no valid pair-layer differences")
_, singular_values, Vt = np.linalg.svd(D, full_matrices=False)  # 不减均值
r = Vt[0] / np.linalg.norm(Vt[0])
asset = {"hidden_size": 5120,
         "directions": {str(layer): r.tolist() for layer in range(40)}}
```

它只说明拟合步骤，不包含激活采集和已验收的数据集。由于投影同时使用`dot(y,r)`和`r`，**单纯把r翻转符号不会改变结果**；改长度则会被单位范数校验拒绝。当前接口要改变行为，需要拟合不同的方向。任意强度需要另外修改引擎/接口并验证，不是现成请求参数。

## 可读JSON与本地运行文件

本地JSON须包含0～39全部层，每层5120个有限数，平方范数与1之差不超过`2e-4`。下面是结构示意，不能直接作为有效参数使用：

```json
{
  "hidden_size": 5120,
  "directions": {
    "0": ["5120 numeric components"],
    "1": ["5120 numeric components"],
    "39": ["5120 numeric components"]
  }
}
```

检查自己生成的JSON及SHA后，在本地转换：

```sh
python3 tools/pack_creative.py /abs/my-directions.json /abs/my-directions.bin \
  --source-sha256 <sha256-of-json> \
  --checkpoint-revision 1bae0e1088993b9b82de08f59de5be5a34e42170
```

转换器验证形状、有限值和单位范数，拒绝覆盖已有输出，生成DSCR0001文件及清单。它不训练、不下载方向。

| 字节位置 | 内容 |
|---|---|
| 0～7 | ASCII标识`DSCR0001` |
| 8～11 | 小端uint32层数：40 |
| 12～15 | 小端uint32维度：5120 |
| 16起 | 按层排列的40×5120个小端float32数值 |

完整大小为819,216字节。它是原始数值数据，加载器不会反序列化Python/pickle可执行对象。[转换源码](../tools/pack_creative.py)和[加载校验源码](../ds41f-4xgx10-switchless/engine/model/model.cu)均可直接检查。用户可以查看自己的JSON、重算SHA，再复现本地二进制；可选示例同时附带两种表示，CI检查转换结果逐字节一致。

## 显式启用随附资产或自己的资产

随附示例位于`assets/steering/creative-v1`。先阅读[资产说明](../assets/steering/creative-v1/README.md)，把`directions.bin`复制到**每个节点**的绝对路径，并读取`manifest.json`中的`suggested_scope`。沿用[启动指南](BUILD_AND_RUN.md)的rank、前端、tokenizer与vision参数，换用`tools/run.py`，再追加下面的参数；四rank和前端必须使用完全相同的scope。

使用`tools/run.py`，不使用原始模型专用入口。为**每一个rank**增加：

```sh
--enable-steering --creative-file /abs/my-directions.bin \
  --creative-scope <suggested_scope-from-manifest>
```

为配对**前端**增加：

```sh
--enable-steering --creative-scope <相同scope>
```

引擎检查四rank的形状、数值、范数和身份是否一致。前端列出`<model-name>-normal`和`<model-name>-creative`，请求时选择即可。切换这两个状态不重载完整模型；更换向量则需要新资产scope并重启配对实例，避免旧缓存错误复用。

当前开启强度固定1.5，尚无热替换向量或任意强度API。改变方向后要重新检查行为与正常能力，不能仅凭成功加载或缓存身份正确就认定质量保持。

## 源码与验证边界

- [creative.cu](../ds41f-4xgx10-switchless/engine/model/creative.cu)：投影运算与Normal直接返回。
- [model.cu](../ds41f-4xgx10-switchless/engine/model/model.cu)：decode作用位置、资产加载校验与模式状态。
- [prefill.cu](../ds41f-4xgx10-switchless/engine/model/prefill.cu)：prefill作用位置。
- [run.py](../tools/run.py)：显式启用与环境处理。
- [原始入口实测](benchmarks/original-launch-20261109.md)：原始模型与既有Normal路径的受检输出一致性、工具/图片/缓存检查。

随附方向是历史实验资产，不是新行为评测，也不保证所有任务获益。二进制身份保持，私有提示标签与采集记录不公开。可用的旧记录没有保留精确的激活采集checkpoint修订，不能说它是在1017上重新拟合，也不能以成功加载推断质量提升。外部资产的来源与许可不由本项目源码许可自动覆盖。
