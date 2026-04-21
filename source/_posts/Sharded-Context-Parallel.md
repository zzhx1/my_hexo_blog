---
title: Sharded Context Parallelism：极致挖掘 CP 潜力
tags:
  - DeepSeek
  - Context Parallelism
  - Ascend
  - Optimization
date: 2025-12-04 03:11:56
updated: 2026-02-03 00:11:56
category: MLsys
description: 提出 Sharded Context Parallelism 方案，通过单卡粒度 CP、Shard Linear 机制及通信掩盖技术，大幅削减冗余计算，通信开销被计算完全掩盖。在 DeepSeek-v3.2 上实现 336% 的吞吐提升。
keywords: Context Parallelism, Sharded CP, DeepSeek, vllm, Parallel Inference
---


## 1. 背景与挑战

### 1.1 Context Parallelism 的背景
上下文并行（Context Parallelism, CP）已成为扩展长上下文（Long Context）及稀疏注意力（Sparse Attention）模型推理的关键技术。相较于传统的张量并行（TP），CP 具备显著优势（参考 [MLSys 2025](https://mlsys.org/virtual/2025/3329)）：
- **低延迟**：通过序列维度的并行计算显著降低首字延迟（TTFT）。
- **低通信**：节点间通信量远低于 TP，适合跨节点扩展。
- **分布式 KV Cache**：KV 缓存容量随设备数线性扩展，有效支撑超长序列。

### 1.2 现有 CP+TP 架构的瓶颈
然而，当前的 CP 主流实现（如 RFC #22693）通常采用 **CP+TP 混合架构**。在处理 DeepSeek-V3/V3.2 等稀疏模型时，这种架构暴露出了三大核心局限：

#### (1) 显存瓶颈：权重冗余存储
在 CP 组内，尽管序列被切分，但每个 Rank 仍需持有**完整的权重副本**（或 TP 分片）。
- **后果**：总显存占用随 CP Rank 数线性增长，严重限制了 CP 在细粒度（如单卡单 Rank）场景下的扩展能力。

#### (2) 计算瓶颈：稀疏逻辑冗余
对于 DeepSeek DSA 等稀疏模型，**Indexer 模块**（负责动态选择活跃 Token）由于仅包含单次矩阵乘 + top-k，无法通过 Megatron 式 TP 并行化。
- **后果**：TP 组内每张卡都会在**相同的本地序列分片**上独立运行完整的 Indexer 逻辑，产生大量重复计算，冗余量随 **TP 规模**线性增加。

#### (3) 存储瓶颈：KV Cache 去重不彻底
虽然 PCP/DCP 实现了 CP 间的 KV Cache 去重，但在 Rank 内部的 TP 设备组之间，KV Cache 仍然是**完全复制**的。

---

## 2. 核心方案：Sharded Context Parallelism

为了突破上述限制，我们提出了 **Sharded Context Parallelism（分片上下文并行）**。该方案支持**单 GPU 粒度**的 CP，通过极致的软硬协同设计，实现了显存、计算与通信的三重优化。

### 2.1 核心创新
1.  **Shard Linear 机制**：借鉴 FSDP 思想，按需加载权重，消除权重的冗余显存占用。
2.  **消除冗余计算**：每张卡仅计算其负责的序列片段（SeqLen），Indexer 等无法 TP 并行的模块不再在 TP 组内重复计算。
3.  **通信完全掩盖**：通过深度流水线优化，将 Attention 阶段的通信隐藏在计算之下，等效通信开销为零。

### 2.2 实验成果
我们在昇腾 910C NPU 的 `vllm-ascend` 框架中实现了该方案。
- **吞吐提升**：在 DeepSeek-v3.2 上获得 **336%** 的吞吐量提升。
- **长文性能**：在 128K 上下文场景下，性能提升高达 **500%**。
- **开源贡献**：[vllm-project/vllm-ascend#4702](https://github.com/vllm-project/vllm-ascend/pull/4702)

---

## 3. Sharded CP for DeepSeek V3.2 

### 3.1 整体设计架构
DeepSeek V3.2 的 Sharded CP 架构设计如下图所示：

<img src="/images/Sharded-Context-Parallel/543207039-e199ca08-7637-4efa-8934-2946e0423e39.png" alt="Architecture Overview" style="zoom:50%;" />

#### 通信消除策略
由于 Attention 权重通常是全量或 TP 冗余存储，我们采取了以下通信消除策略：
1.  **消除 Output All-Reduce**：方案摒弃了 TP row-parallel 的 `o_proj`（改由 Shard Linear 提供完整权重），自然无需 `o_proj` 后的 All-Reduce；每张卡仅输出其负责的序列片段的结果。
2.  **KV Cache 聚合与掩盖**：
    - 计算 Attention 需要完整的 KV。我们将主模型的 KV Cache、RoPE 以及 Indexer 模块的 K Cache 进行横向拼接。
    - 发起一次 **All-Gather** 操作，将其与计算密集型的 `q_up_proj` 并行执行，实现**通信完全被计算掩盖**。
    - **显存优化**：KV Cache 按层通信，用完即存入 Block 或释放，峰值显存占用被限制在单层 KV 的规模。

#### MoE 部分优化
对 MoE 模块同样应用序列并行（SP）。Attention 结束后，每张卡持有部分 Token 的完整结果。在进入 MoE 之前的 Quant 及 Route Logits 计算后，对结果进行 All-Gather，该过程的通信同样被计算完全掩盖。

### 3.2 Shard Linear：权重的极致分片

针对占用 MLA 大部分显存的 **Q_proj** 和 **O_proj**，我们引入了 **Shard Linear** 特性。

#### 设计理念
受 FSDP 启发，Shard Linear 改变了权重的生命周期管理：**静态时分片存储，计算前动态聚合出完整权重完成矩阵乘，计算后立即释放。**

<img src="/images/Sharded-Context-Parallel/522353220-6b15a757-e1ac-4d92-a03b-b7b8bf063e27.png" alt="Shard Linear Concept" style="zoom: 33%;" />

#### 针对 NPU 的 Layer Sharding + Broadcast 方案
FSDP 的原生做法是**层内切分**（每卡持有每层权重的一部分），通过 All-Gather 在首维拼接。但昇腾 NPU 存在私有权重格式（NZ 格式）及量化属性（Quant Scale 等），按首维 All-Gather 会破坏这些格式与元数据。因此我们放弃层内切分，改用**层间 Layer Sharding + Broadcast** 的方案：

1.  **层级分片（Layer Sharding）**：将各层权重按 `Layer ID % Device Count` 策略**完整地**分配到不同设备，每层权重只由一张卡持有，不做层内再切分。
2.  **启动预热**：每张卡额外冗余缓存前 $K$ 层的完整权重，确保首 token 推理无延迟启动。
3.  **异步预取（Asynchronous Prefetch）**：
    - 在 SP 第一阶段 All-Gather 完成后（进入 Attention 前），定位 $L+K$ 层权重所在的源设备。
    - 源设备发起 **Broadcast**，将完整权重（连同 NZ/量化元数据）一次性分发至所有卡，格式天然保留。
    - 由于 Prefill 阶段 `q_up_proj` 和 `FlashAttention` 耗时较长，Broadcast 通信可被**完全掩盖**。
4.  **及时释放**：前向计算完成后，立即释放该层权重。

<img src="/images/Sharded-Context-Parallel/layer_sharding.png" alt="layer_sharding" style="zoom:50%;" />

> 此特性已合入 vllm-ascend 主分支：[PR #2931](https://github.com/vllm-project/vllm-ascend/pull/2931)

---

## 4. 性能提升分析⭐️

这个性能提升幅度超出了预期 —— Prefill 阶段的单项优化通常能拿到 10% 左右就已经算相当不错的关键特性了。下面针对性能来源做详细拆解。

**性能提升benchmark图** && **gsm8k精度图**：

<img src="/images/Sharded-Context-Parallel/image-20260203005548367.png" alt="image-20260203005548367" style="zoom:33%;" />

<img src="/images/Sharded-Context-Parallel/image-20260203005633972.png" alt="image-20260203005633972" style="zoom:33%;" />

### 算子性能分析：

首先看性能差异比较大的算子：

#### 1. SparseFlashAttention

**profile：**

![image-20260203005907133](/images/Sharded-Context-Parallel/image-20260203005907133.png)

**kernel_details**：

| 方案         | Device_id | Model ID   | Task ID | Stream ID | Name                 | Type                 | OP State | Accelerator Core | Start Time(us)       | Duration(us) | Wait Time(us) | Block Dim | Mix Block Dim | **HF32 Eligible** | **Input Shapes**                                             | **Input Data Types**                                         | **Input Formats**          | **Output Shapes** | **Output Data Types** | **Output Formats** | **Context ID** | **aicore_time(us)** | **aic_total_cycles** | **aic_mac_time(us)** | **aic_mac_ratio** | **aic_scalar_time(us)** | **aic_scalar_ratio** | **aic_mte1_time(us)** | **aic_mte1_ratio** | **aic_mte2_time(us)** | **aic_mte2_ratio** | **aic_fixpipe_time(us)** | **aic_fixpipe_ratio** | **aic_icache_miss_rate** | **aiv_time(us)** | **aiv_total_cycles** | **aiv_vec_time(us)** | **aiv_vec_ratio** | **aiv_scalar_time(us)** | **aiv_scalar_ratio** | **aiv_mte2_time(us)** | **aiv_mte2_ratio** | **aiv_mte3_time(us)** | **aiv_mte3_ratio** | **aiv_icache_miss_rate** | **cube_utilization(%)** |
| ------------ | --------- | ---------- | ------- | --------- | -------------------- | -------------------- | -------- | ---------------- | -------------------- | ------------ | ------------- | --------- | ------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------- | ----------------- | --------------------- | ------------------ | -------------- | ------------------- | -------------------- | -------------------- | ----------------- | ----------------------- | -------------------- | --------------------- | ------------------ | --------------------- | ------------------ | ------------------------ | --------------------- | ------------------------ | ---------------- | -------------------- | -------------------- | ----------------- | ----------------------- | -------------------- | --------------------- | ------------------ | --------------------- | ------------------ | ------------------------ | ----------------------- |
| **cp**       | 0         | 4294967295 | 58111   | 2         | SparseFlashAttention | SparseFlashAttention | dynamic  | MIX_AIC          | 1764761459549479.309 | 3524.631     | 1.055         | 24        | 48            | NO                | "1024,128,512;485,128,1,512;485,128,1,512;1024,1,2048;5,128;5;5;1024,128,64;485,128,1,64" | DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32;INT32;DT_BF16;DT_BF16 | ND;ND;ND;ND;ND;ND;ND;ND;ND | "1024,128,512"    | DT_BF16               | ND                 | 0              | 2767.033            | 122856248            | 1270.05              | 0.459             | 1090.331                | 0.394                | 1124.983              | 0.407              | 1901.353              | 0.687              | 1908.977                 | 0.69                  | 0                        | 2769.781         | 245956577            | 802.586              | 0.29              | 2338.385                | 0.844                | 1725.753              | 0.623              | 714.356               | 0.258              | 0                        | 75.365                  |
| **baseline** | 0         | 4294967295 | 32713   | 2         | SparseFlashAttention | SparseFlashAttention | dynamic  | MIX_AIC          | 1764762104173838.715 | 39846.596    | 1.135         | 24        | 48            | NO                | "16384,8,512;699,128,1,512;699,128,1,512;16384,1,2048;4,128;4;4;16384,8,64;699,128,1,64" | DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32;INT32;DT_BF16;DT_BF16 | ND;ND;ND;ND;ND;ND;ND;ND;ND | "16384,8,512"     | DT_BF16               | ND                 | 0              | 30812.793           | 1368088000           | 5191.721             | 0.168             | 12573.224               | 0.408                | 13614.999             | 0.442              | 24602.17              | 0.798              | 23509.344                | 0.763                 | 0                        | 30813.542        | 2736242537           | 1104.192             | 0.036             | 28802.779               | 0.935                | 24316.9               | 0.789              | 5973.504              | 0.194              | 0                        | 74.235                  |

**我们来看一下其中的一些关键指标**

| 维度 | CP | Baseline | 差异分析 |
| :--- | :--- | :--- | ---- |
| **执行耗时 (Duration)** | 3.52 ms | 39.85 ms | cp 快 11.3 倍 |
| **input_shape** | `[1024, 128, 512]...` | `[16384, 8, 512]...` | CP 方案切分 seq，但保留完整的 head；虽维度不同，单卡数据量相同 |
| **output_shape** | `[1024, 128, 512]` | `[16384, 8, 512]` | 同上（可以推断 SFA 的计算量完全相同） |
| **cube_utilization(%)** | 75.365% | 74.235% | 利用率相近，说明计算效率相当 |
| **aic_mte1_time**<br />**aic_mte2_time**<br />**aiv_mte2_time**<br />**aiv_mte3_time** | 1124.98 μs<br />1901.35 μs<br />1725.75 μs<br />714.36 μs | 13614.99 μs<br />24602.17 μs<br />24316.9 μs<br />5973.5 μs | 基本都是 12 倍左右，MTE 是性能差异的主因 |

在华为昇腾 AI 处理器（如 Ascend 910）中，MTE 是专门负责数据搬运的硬件单元，与计算单元（AI Core 中的 Cube、Vector 单元）解耦并行工作。CP 方案显著减少了 KV Cache 的访存量，有效缓解了 MTE 的访存瓶颈，使得 SFA 算子中的计算能够更高效地流水执行。

**为什么 CP 能减少 kv cache 访存 ?**

要理解这一点，关键是看清 **Sparse Attention 的访存模式与稠密 MLA 的本质区别**。

**稠密 MLA** 中，所有 Q token 共享同一份完整 KV，内核可以将 KV 一次性、连续地从 HBM 读入片上并被所有 Q token 复用，访存是**规整的大块顺序读**，MTE 带宽利用率高。

**DeepSeek-V3.2 的 SFA** 则不同：Indexer 为**每个 Q token 独立**筛选出 top-2048 个活跃 KV —— 不同 token 命中的 KV 位置**不同、不连续、随机分布**在完整 KV Cache 中。内核只能**按 token 粒度**做**稀疏、非连续的随机访存**：每个 Q token 都要触发一次对其 2048 个 KV 位置的 gather，彼此之间几乎无法复用同一次 HBM 读取。这种访存模式下，MTE 的开销**正比于 Q token 的数量**，而不是像稠密场景那样与总 KV 大小线性相关。

- **Baseline (TP16)**：TP 切的是 head 而非 seq，每张卡都持有完整 seqlen 的 Q，需要对**全部 token**做稀疏 gather。更关键的是，MLA 的 latent KV 在 head 维度是共享的，TP 无法连带把 KV 切开 —— 每张卡实际仍在**完整的 KV Cache** 上做稀疏随机读。最终大量时间耗在 MTE 上，SFA 内核表现为严重的访存瓶颈与流水线停顿。
- **CP16**：序列沿 CP 维度被切成 16 份，每张卡只持有 1/CP 的 Q token，稀疏 gather 的次数也只有 baseline 的 1/CP。MTE 总工作量按序列切分比例线性下降，内核从访存 bound 回归到更健康的计算-访存平衡，这正是 SFA 算子提速 11× 的根本原因。

理论上 baseline 单卡 KV 读取量应达 CP 方案的 16 倍，但实测 MTE 时间差距约 11~12 倍（见前文 `aic_mte` / `aiv_mte` 指标）。推测原因：不同 Q token 的 top-k 结果存在位置重叠，底层 cache 能命中一部分重复访问，摊薄了 baseline 的访存开销。

#### 2. LightningIndexer 的计算

**profile：**

![image-20260203011606196](/images/Sharded-Context-Parallel/image-20260203011606196.png)

**kernel_details**：

| 方案     | Device_id | Model ID   | Task ID | Stream ID | Name             | Type             | OP State | Accelerator Core | Start Time(us)       | Duration(us) | Wait Time(us) | Block Dim | Mix Block Dim | HF32 Eligible | Input Shapes                                    | Input Data Types                          | Input Formats     | Output Shapes  | Output Data Types | Output Formats | Context ID | aicore_time(us) | aic_total_cycles | aic_mac_time(us) | aic_mac_ratio | aic_scalar_time(us) | aic_scalar_ratio | aic_mte1_time(us) | aic_mte1_ratio | aic_mte2_time(us) | aic_mte2_ratio | aic_fixpipe_time(us) | aic_fixpipe_ratio | aic_icache_miss_rate | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio | aiv_mte2_time(us) | aiv_mte2_ratio | aiv_mte3_time(us) | aiv_mte3_ratio | aiv_icache_miss_rate | cube_utilization(%) |
| -------- | --------- | ---------- | ------- | --------- | ---------------- | ---------------- | -------- | ---------------- | -------------------- | ------------ | ------------- | --------- | ------------- | ------------- | ----------------------------------------------- | ----------------------------------------- | ----------------- | -------------- | ----------------- | -------------- | ---------- | --------------- | ---------------- | ---------------- | ------------- | ------------------- | ---------------- | ----------------- | -------------- | ----------------- | -------------- | -------------------- | ----------------- | -------------------- | ------------ | ---------------- | ---------------- | ------------- | ------------------- | ---------------- | ----------------- | -------------- | ----------------- | -------------- | -------------------- | ------------------- |
| cp       | 0         | 4294967295 | 58109   | 2         | LightningIndexer | LightningIndexer | dynamic  | MIX_AIC          | 1764761459548934.118 | 282.446      | 0.789         | 24        | 48            | NO            | "1024,64,128;485,128,1,128;1024,64;5;5;5,128"   | DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32 | ND;ND;ND;ND;ND;ND | "1024,1,2048"  | INT32             | ND             | 0          | 261.516         | 11611319         | 84.575           | 0.323         | 99.695              | 0.381            | 70.886            | 0.271          | 29.714            | 0.114          | 139.095              | 0.532             | 0.005                | 278.619      | 24741398         | 213.992          | 0.768         | 74.65               | 0.268            | 96.997            | 0.348          | 6.258             | 0.022          | 0.005                | 88.886              |
| baseline | 0         | 4294967295 | 32711   | 2         | LightningIndexer | LightningIndexer | dynamic  | MIX_AIC          | 1764762104168502.129 | 5092.241     | 2.098         | 24        | 48            | NO            | "16384,64,128;699,128,1,128;16384,64;4;4;4,128" | DT_BF16;DT_BF16;DT_BF16;INT32;INT32;INT32 | ND;ND;ND;ND;ND;ND | "16384,1,2048" | INT32             | ND             | 0          | 4554.882        | 202236777        | 1736.127         | 0.381         | 1781.882            | 0.391            | 1451.462          | 0.319          | 549.574           | 0.121          | 2590.513             | 0.569             | 0                    | 5085.976     | 451634633        | 4250.194         | 0.836         | 1419.064            | 0.279            | 1844.032          | 0.363          | 65.617            | 0.013          | 0                    | 85.87               |

**关键指标: **

| 维度                | CP               | baseline           | 差异分析                                                     |
| ------------------- | ---------------- | ------------------ | ------------------------------------------------------------ |
| 执行耗时 (Duration) | 282.4us          | 5092.2us           | cp 比 baseline 快 18 倍                                      |
| input_shape         | [1024,64,128]... | [16384,64,128].... | **关键：seq 维度相差 16 倍**                                 |
| output_shape        | [1024, 1, 2048]  | [16384, 1, 2048]   | 同上（可以推断 CP 方案下 Indexer 的计算量远小于 baseline）   |
| aic_mte             | ....             | ....               | cp 远小于 baseline（20x）                                    |
| aiv_mte             | ....             | ....               | cp 远小于 baseline(18x)                                      |

**LightningIndexer 模块的核心收益来自于消除冗余计算。**

**为什么 baseline 中 Indexer 模块的三个矩阵无法在 TP 维度切分？**

Megatron 式 TP 依赖 column-parallel + row-parallel 的成对组合，通过两次矩阵乘在内部消化掉激活的切分。而 LightningIndexer 只包含一次矩阵乘，其输出激活直接送入 top-k —— top-k 需要看到**完整激活**才能做全局选择。在不引入额外 All-Gather 通信的前提下，这里无法做 TP 切分。

## 5. 总结与展望

Sharded Context Parallelism 是一次将 CP 理念推向极致的尝试：
1.  ✅ **打破粒度限制**：实现单卡级 CP，最大化硬件利用率。
2.  ✅ **打破显存限制**：Shard Linear 消除权重冗余。
3.  ✅ **打破计算限制**：彻底消除稀疏模型的 Indexer 冗余。
4.  ✅ **打破通信限制**：通过全流程计算-通信掩盖，Attention 阶段的通信开销等效为零。

**实测效果**：DeepSeek-v3.2 吞吐量提升 **336%**，验证了该架构在处理复杂稀疏大模型时的卓越效能。

**未来计划**：
- 将 Sharded CP 推广至更多 Transformer 架构模型。
