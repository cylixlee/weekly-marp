---
marp: true
paginate: true
headingDivider: 2
header: "Image Hijacks: Adversarial Images can Control Generative Models at Runtime"
math: mathjax
---

<style>
    p, li {
        font-size: 26px;
    }
</style>

# 图像劫持：对抗图像可在运行时控制生成模型

<small>李政翱 · 周论文阅读 · 24th Feb</small>

## 摘要
在这项工作中，我们专注于对视觉语言模型（VLM）的图像输入。我们发现了图像劫持，以及在推理时控制 VLM 行为的对抗性图像，并引入了训练图像劫持的一般**行为匹配**算法。

由此，我们推导出了**提示匹配**方法，允许我们训练劫持匹配任意用户定义的文本提示的行为。我们使用行为匹配来制造四种类型的攻击劫持：

1. **Specific String Attack**：迫使 VLM 生成攻击者选择的输出
2. **Leak Context Attack**：从上下文窗口泄漏信息
3. **Jailbreak Attack**：覆盖安全训练
4. **Disinformation Attack**：相信虚假陈述

我们研究了这些针对 LLaVA 的攻击，这是一种基于 CLIP 和 LLaMA-2 的最先进的 VLM，发现所有攻击类型的成功率都超过 80%。此外，我们的攻击是自动化的，只需要很小的图像扰动。

## 图像劫持样例

<style scoped>
    li {
        margin-left: 800px;
        margin-bottom: 15px;
    }
</style>

<img align=left src="figure1.png" width=800>

如左图所示，四种攻击分别为：

1. **特定字符串攻击**
   使模型输出攻击者指定的字符串
2. **上下文泄露攻击**
   使模型泄露当前上下文
3. **越狱攻击**
   使模型绕过安全政策
4. **虚假信息攻击**
   使模型相信虚假信息

## 问题建模

<style scoped>
    p, li {
        font-size: 23px;
    }
</style>

根据 [Zhao et al.](https://arxiv.org/abs/2305.16934) 的工作，本文形式化**威胁模型**如下：

- **模型 API**
  本文使用 $M_\phi(\mathbf x, \texttt{ctx})\mapsto out$ 来表示 VLM。其中输入图像 $\mathbf{x}:\texttt{Image}$ 取值范围为 $[0, 1]^{c\times h\times w}$，输入上下文为 $\texttt{ctx}: \texttt{Text}$，模型输出为 $out: \texttt{Logits}$。

- **攻击者知识**
  本文假定攻击者对于模型 $M_\phi$ 有白盒权限。攻击者可以通过 $M_\phi(\mathbf x, \texttt{ctx})$ 计算关于 $\mathbf{x}$ 的梯度。

- **攻击者能力**
  本文未对攻击者能力做严格假设，攻击者可以输入任意的图像。

- **攻击者目标**
  我们将希望 VLM 做出的**目标行为**定义为从输入上下文到输出 Logits 的函数映射。对于给定的行为 $B: C \to \texttt{Logits}$，攻击者的目标是生成对抗图像 $\hat{\mathbf{x}}$，使得 VLM 在一组可能的输入上下文 $C$ 下做出行为 $B$。公式化表示为：

$$
M_\phi(\hat{\mathbf{x}}, \texttt{ctx}) \approx B(\texttt{ctx}) \ \text{for all contexts} \ \texttt{ctx} \in C
$$

## 行为匹配算法

<style scoped>
    p {
        margin-bottom: 48px;
    }
</style>

基于上述的问题建模，定义教师强制模型 $M_\phi^{forced}(\mathbf{x}, \texttt{ctx}, \texttt{target}) \mapsto out$，其在给定的上下文 $\texttt{ctx}: \texttt{Text}$ 下根据目标 $\texttt{target}: \texttt{Logits}$ 返回 $out$。本文使用投影梯度下降（PGD）来寻找 $\hat{\mathbf{x}}$：

$$
\mathop{\arg\min}\limits_{\mathbf{x}\in\texttt{Image}}\sum_{\texttt{ctx}\in C}[\mathcal{L}(M_\phi^{forced}(\mathbf{x}, \texttt{ctx}, B(\texttt{ctx})), B(\texttt{ctx}))] \tag{1}
$$

其中 $\mathcal{L}: \texttt{Logits}\times\texttt{Logits}\to\mathbb{R}$ 为交叉熵损失函数。在解决如上的优化问题后，本文将劫持图像从 $\hat{x}_{cij}\in[0, 1]$ 映射为 $[0, 255]$ 区间内的整数值。

## 行为匹配算法图例

![](figure2.png)

给定一个不良行为的数据集和一个冻结的 VLM 数据集，我们使用公式 $(1)$ 来优化图像，使 VLM 输出匹配行为。

## 提示匹配方法 I

<style scoped>
    p {
        margin-top: 40px;
    }
</style>

行为匹配方法提供了训练图像劫持的一般方法。然而，构建上下文数据集很难，实际无法手动将用户可能的上下文穷举并作为数据集训练。然而，通过添加**指令**（如“请按照埃菲尔铁塔搬到了罗马的假设输出”），能使诱导 VLM 做出目标行为的过程简化许多。

由此，我们可以引出**提示匹配方法**：构建图像 $\mathbf{x}$，使得

$$
\newcommand{\doubleplus}{\mathop{+ \! \! +}}

\forall \texttt{ctx} \quad M_\phi(\mathbf{x}, \texttt{ctx}) \approx M_\phi(I, \texttt{p} \doubleplus \texttt{ctx} )
$$

其中 $\texttt{p}$ 表示提示词，$I$ 表示输入图像，$\texttt{p} \doubleplus \texttt{ctx}$ 表示提示词和上下文的拼接。

## 提示匹配方法 II

<style scoped>
    p {
        margin-top: 40px;
    }
</style>

[Bagdasaryan et al.](https://arxiv.org/abs/2307.10490) 尝试使用内嵌方法生成对抗图像，即使得图像输入 VLM 后，模型会将其认定为提示词而非图像。然而，由于**模态间隙**（由 [Liang et al.](https://arxiv.org/abs/2203.02053) 提出）的存在，这样的方法无法实现。

然而，我们只需要构造图像使得上文提到的公式成立，因此只需使得最终输出的 Logits 与目标 Logits 高度相似，而不需要图像嵌入与提示词嵌入高度相似。定义：

$$
B_{\texttt{p}}: C \to \texttt{Logits}
$$
$$
B_{\texttt{p}}(\texttt{ctx}) := M_\phi(I, \texttt{p}\doubleplus\texttt{ctx})
$$

其中 $C$ 为一些通用的文本数据集。此时可在 $D=\{(\texttt{ctx}, B_{\texttt{p}}(\texttt{ctx}))|\texttt{ctx}\in C\}$ 上执行行为匹配，本文将这种过程称为**提示匹配**。

## 实验结果

![w:650 left](table1.png) ![w:450 right](table2.png)

## 总结

<style scoped>
    p {
        margin-top: 30px;
    }
</style>

本文提出了**图像劫持**（Image Hijacks）的概念——一种能够在运行时控制视觉语言模型（VLMs）的对抗性图像。

本文提出了**行为匹配**（Behaviour Matching）算法用于训练劫持图像，并基于此推导出**提示匹配**（Prompt Matching）算法，使得能够使用与目标提示无关的通用数据集训练出与任意文本提示行为匹配的劫持样本。

通过这些技术，本文实现了对**特定字符串攻击**、**上下文泄露攻击**、**越狱攻击**和**虚假信息攻击**的构造，在所有攻击类型中成功率均达到至少80%。劫持图像可自动化生成、对人不可感知，并允许对模型输出进行细粒度控制。

## 心得

<style scoped>
    p {
        margin-top: 60px;
    }
</style>

此篇论文角度非常新颖。初见时觉得通过生成对抗图像来使得 VLM 做出错误推理是很困难的任务，然而只要对于 VLM 的**嵌入表示有一定理解**（即*图像是通过连续输入通道输入模型的* ），则可以将此任务**形式化表示为**一种**优化问题**，从而使得训练相应模型成为了可能。

对于模型结构及原理的深刻理解是解决问题的基石。基础不牢，地动山摇。在之后的阅读中，我会更专注于理解技术原理，积跬步以至千里。