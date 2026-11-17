# 第三方数据说明

本插件内置的鸟类名称对照表（`bird/data/bird_names.tsv.gz`）由以下来源合并而成，
仅保留「学名 → 中文名 / 英文名」三列事实数据，不含任何源代码。

## 1. Fork-huashu-bird-sticker

- 项目：https://github.com/Sileec/Fork-huashu-bird-sticker
- 文件：`data/names.csv`
- 许可：仓库 `data` 目录整体为 MIT；英文学名与英文俗名来自 AviList（CC BY 4.0）；
  其余语言列（含简体中文）来自 IOC World Bird List v15.2（CC BY 3.0）。

## 2. SuperPicky

- 项目：https://github.com/jamesphotography/SuperPicky
- 文件：`birdid/data/bird_reference.sqlite` 中的物种名称列
- 许可：项目整体 AGPL-3.0。此处仅提取物种名称这类事实性数据用于交叉补全，
  未使用或改编其源代码，本插件亦不包含该项目的任何代码。

## 合并方式

以学名为键取并集：两个来源都收录的学名以第一个来源为准，只被第二个来源收录的学名
用于补齐。部分学名在名录版本间存在异名，合并时已按学名归一化（小写、下划线转空格、
压缩空白）后匹配。

如权利人认为此处使用不当，请提交 issue，我们会移除相应数据。