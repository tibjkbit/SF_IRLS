# 随机前沿分析项目

## 动机

本项目的灵感来源于阅读一篇关于随机生产前沿的文章。出于好奇，回顾了1977年的最初随机前沿设定，并将其与现代数值优化方法进行了对比。

## 概述

此仓库包含了对随机前沿模型的全面分析，比较了其原始版本与使用内点法的IRLS（迭代加权最小二乘法）估计的更现代实现。分析采用R语言执行，在相同问题情境下进行了详细对比。

## 内容

- **引言和讨论**：
  - `Introduction and Discussion.pdf` 和 `Introduction and Discussion.md`：这些文件详细介绍了随机前沿模型的原始版本和动机，并使用R语言复现。同时与在相同问题情境下的IRLS估计的现代实现进行了对比。
  
- **图片**：
  - `fig`：该目录包含文档中提到的图片。

- **代码**：
  - `code`：包含样本数据和SF（随机前沿）以及IRLS算法的具体实现，提供了实践理解和分析模型的方法。

- **论文**：
  - `paper`：包括Aigner, D.J.等人在1977年发表的原始论文，这是关于随机前沿设定和估计最早的系统性讨论之一。目录中还包括其前几章的翻译，提供了模型基础方面的洞察。

## 数据和估计

项目采用基于B-spline基函数的原始一维数据转换作为设计矩阵。此转换有助于在二维空间中更好地可视化“前沿”或“边界”概念，方便对随机前沿模型的讨论。

## 如何使用本仓库

- 要了解理论背景和动机：请参考 `Introduction and Discussion.pdf` 或 `.md` 文件。
- 获取可视化文件：查看 `fig` 目录。
- 深入代码和数据：浏览 `code` 目录。
- 阅读基础文献：访问 `paper` 目录。
