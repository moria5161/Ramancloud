# RamanCloud

RamanCloud 是由厦门大学 Ren Research Group 开发的在线拉曼光谱处理与分析平台。本仓库包含基于 Streamlit 的网站代码，提供光谱预处理、高光谱成像与时间序列处理、分子光谱预测及常用文件工具。

**[访问 RamanCloud](https://ramancloud.xmu.edu.cn/)**

## 功能模块

| 模块 | 功能 |
| :--- | :--- |
| **Homepage · 主页** | 提供功能入口、项目介绍及课题组近期研究成果。 |
| **Process Spectra · 光谱处理** | 对单条或批量光谱进行波数截取、降噪和基线校正，预览处理结果并下载光谱及基线。 |
| **Process Mapping · 成像与时间序列处理** | 对高光谱成像和时间序列数据进行截取、降噪及基线校正，查看热图、选取位置或时间点的光谱并导出结果。 |
| **Spectral Analysis · 光谱分析** | 提供多光谱 PCA 降维分析及拉曼光谱相关问答入口。 |
| **Structure to Spectrum · 结构到光谱** | 通过 XYZ 文件、SMILES 或分子绘图输入结构，查看三维构象并预测、下载拉曼光谱。 |
| **Spectrum to Structure · 光谱到结构** | 功能建设中，目前展示预告页面。 |
| **Extra Tools · 附加工具** | 将成像或时间序列文件拆分为单谱、合并光谱文件，以及进行 Horiba 与 Nanophoton 成像格式转换。 |
| **Tutorial · 使用教程** | 介绍数据导入、操作流程、算法及参数。 |
| **Changelog · 更新记录** | 查看网站功能更新与改进记录。 |
| **Contributors · 贡献者** | 展示开发团队、指导老师及项目致谢。 |
| **Privacy Policy · 隐私政策** | 说明用户协议、数据使用与隐私政策。 |

## 新版网站

**新版 RamanCloud 已上线：[进入新版工作区](https://ramancloud.xmu.edu.cn/preprocessing/)。**

新版专注于单光谱、批量光谱、高光谱成像和时间序列的预处理，支持自由排列截取、降噪与基线校正步骤，并提供交互式对比、处理记录导出、中英文界面及深浅主题。

[查看新版代码仓库](https://github.com/moria5161/RamancloudV2)

## Contact

**Ren Research Group, Xiamen University · 厦门大学**

- **科研合作：** [bren@xmu.edu.cn](mailto:bren@xmu.edu.cn)
- **使用与技术问题：** [xinyulu@stu.xmu.edu.cn](mailto:xinyulu@stu.xmu.edu.cn)
- **问题反馈与建议：** [GitHub Issues](https://github.com/moria5161/Ramancloud/issues)

## 许可证

RamanCloud 源代码采用 [Apache License 2.0](LICENSE) 许可证。

第三方组件保留其原有许可证。模型权重、数据集及论文插图不包含在本软件许可证的授权范围内，除非另有明确说明。
