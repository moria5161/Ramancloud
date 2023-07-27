# 去噪算法的整合与部署

## 面临的问题
+ 目前的光谱分析软件无法批量化和自动化地进行去噪和去基线等预处理；  
+ 课题组内新开发的算法需要部署给大家使用，并利用反馈意见进行迭代；  

## 目前已整合的功能
### 光谱
+ S-G滤波
+ airPLS 去基线
### 成像
+ ALRMA 和 CLRMA 去噪

---
## 备注
网页应用基于streamlit开发，主体内容为python，ALRMA和CLRMA部分为MATLAB脚本，
已通过MATLAB Compiler集成到主体应用。目前MATLAB部分仅搭建了整体工作流，运行时仍存在许多bug亟待修复。

## 数据收集
目前上传至改平台的所有数据均存放在/home/room/flask/received

