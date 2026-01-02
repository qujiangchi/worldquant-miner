# 中文版论文 (Chinese Version)

这是论文的中文版本，位于 `paper-zh` 文件夹中。

## 编译说明

使用 XeLaTeX 编译（支持中文）：

```bash
xelatex main.tex
xelatex main.tex  # 第二次编译以生成目录
```

或者使用 `latexmk`：

```bash
latexmk -xelatex main.tex
```

## 注意事项

- TikZ 图表保持不变（与英文版相同）
- 代码示例保持不变
- 仅文本内容翻译为中文
- 使用 `ctex` 包支持中文排版

## 章节文件

所有章节文件位于 `chapters/` 目录下，与英文版结构相同。
