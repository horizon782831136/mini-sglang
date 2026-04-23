# CLAUDE.md — Project Conventions for Claude

## LaTeX / Math Formula Rules (Markdown docs)

When writing mathematical formulas in Markdown files (`.md`), follow these rules
to ensure correct rendering in GitHub, KaTeX, and common Markdown viewers.

### 1. No `\mathbb`, `\mathcal`, or `amssymb` commands
These are not supported in all renderers. Replace with shape/tuple notation.

**Wrong:**
```
$X \in \mathbb{R}^{n \times d}$
$W \in \mathbb{R}^{d_{model} \times d_k}$
```
**Correct:**
```
$X$ with shape $(n, d)$
$W$ with shape $(d_{model}, d_k)$
```

### 2. No Chinese (or non-ASCII) inside `\text{}`
LaTeX `\text{}` is for ASCII text only. Put Chinese explanations outside the formula.

**Wrong:**
```
$$\text{显存} = 2 \times N \times B$$
$$\approx 1\,\text{GB（单请求）}$$
```
**Correct:**
```
显存计算公式：
$$\text{Memory} = 2 \times N \times B$$
即单个请求约占 1 GB 显存。
```

### 3. No multiple inline `$...$` blocks separated by Chinese punctuation
Chinese punctuation (，。) between two `$` blocks breaks inline math parsing.

**Wrong:**
```
$Q \in \mathbb{R}^{n \times d}$，$K \in \mathbb{R}^{n \times d}$，$n$ 可能很大
```
**Correct:**
```
Q 和 K 的形状均为 $(n, d)$，其中 $n$ 可能很大
```

### 4. Thousands separators in math mode use `{,}` not `,`
A bare `,` in math mode is treated as an operator, not a digit separator.

**Wrong:**
```
$$1,073,741,824 \approx 1\ \text{GB}$$
```
**Correct:**
```
$$1{,}073{,}741{,}824 \approx 1\ \text{GB}$$
```

### 5. No `\_` inside subscripts in math mode
Inside `$...$` or `$$...$$`, underscores do not need escaping. Use `-` or
omit the underscore for multi-word subscript labels.

**Wrong:**
```
$n_{kv\_heads}$, $\text{max\_seq\_len}$
```
**Correct:**
```
$n_{kv}$, $S_{max}$
```
Explain variable meanings in surrounding text rather than encoding them in subscripts.

### 6. Preferred shape notation for tensors
Use tuple notation consistently for tensor shapes.

```
Shape $(n, h, d_k)$ instead of $\in \mathbb{R}^{n \times h \times d_k}$
```
