# 整合的头像生成系统使用说明

## 概述

`integrated_avatar_generation.py` 是一个完整的端到端头像生成系统，整合了以下功能：
1. VHAP数据转换为LAM兼容格式
2. 多视角头像视频生成
3. 自动视频处理和音频提取
4. 统一的工作流程管理

## 功能特性

### 🔄 数据转换
- 自动转换VHAP输出为LAM兼容格式
- 支持多种视频格式输入（.mp4/.mov/.avi/.mkv等）
- 自动提取音频文件
- 生成完整的驱动数据结构

### 🎬 多视角生成
- 支持多个偏航角（yaw）和俯仰角（pitch）组合
- 批量生成不同视角的头像视频
- 自动音频同步
- 高质量视频输出

### 🛠️ 自动化流程
- 一键完成从原始数据到最终视频的全流程
- 智能错误处理和状态检查
- 详细的进度显示和日志输出

## 安装要求

```bash
# 基础依赖
pip install torch torchvision
pip install opencv-python pillow
pip install moviepy numpy
pip install omegaconf argparse
pip install tqdm pathlib

# LAM相关依赖
# 请确保已正确安装LAM框架和相关模型
```

## 使用方法

### 基本用法

```bash
python integrated_avatar_generation.py \
    --input_image /path/to/your/image.jpg \
    --source_video /path/to/your/video.mp4 \
    --output_dir ./output_results/
```

### 完整参数示例

```bash
python integrated_avatar_generation.py \
    --input_image /root/autodl-tmp/datasets/mono_jp/apple_jp_3/images_4/000000.jpg \
    --source_video /root/autodl-tmp/datasets/mono_jp/apple_jp_3.mov \
    --vhap_dir /root/autodl-tmp/VHAP_track/mono_jp/export_epoch0 \
    --output_dir ./output_integrated/ \
    --drive_name custom_jp_drive \
    --model_name /root/autodl-tmp/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500 \
    --yaw_angles -30 -20 -10 0 10 20 30 \
    --pitch_angles -10 -5 0 5 10 \
    --max_frames 264
```

## 参数说明

### 必需参数
- `--input_image`: 输入的人像图片路径
- `--source_video`: 源视频文件路径（用于驱动生成）

### 可选参数
- `--vhap_dir`: VHAP输出目录路径（如果需要转换VHAP数据）
- `--output_dir`: 输出目录路径（默认：./output_integrated/）
- `--drive_name`: 驱动数据名称（默认：custom_integrated_drive）
- `--model_name`: LAM模型路径
- `--yaw_angles`: 偏航角度列表（默认：-30到30度）
- `--pitch_angles`: 俯仰角度列表（默认：-10到10度）
- `--max_frames`: 最大帧数（默认：264）

## 工作流程

### 1. 数据准备阶段
如果提供了`--vhap_dir`参数：
```
VHAP数据 → 格式转换 → LAM兼容格式
源视频 → 格式转换 → MP4 + 音频提取
```

### 2. 模型初始化
```
预处理器初始化 → FLAME跟踪器初始化 → LAM模型加载
```

### 3. 多视角生成
```
输入图片预处理 → FLAME参数估计 → 多角度视频生成 → 音频合成
```

## 输出结果

生成的文件结构：
```
output_dir/
├── output_yaw-30_pitch-10.png          # 处理后的参考图片
├── output_yaw-30_pitch-10_audio.mp4    # 带音频的最终视频
├── output_yaw-20_pitch-5_audio.mp4
├── ...
└── debug_first_frame.png               # 调试用第一帧图片
```

## 使用场景

### 场景1：从VHAP数据生成多视角视频
```bash
# 适用于已有VHAP处理结果的情况
python integrated_avatar_generation.py \
    --input_image portrait.jpg \
    --source_video driving_video.mp4 \
    --vhap_dir ./vhap_output/ \
    --output_dir ./results/
```

### 场景2：使用现有驱动数据
```bash
# 适用于已有LAM格式驱动数据的情况
python integrated_avatar_generation.py \
    --input_image portrait.jpg \
    --drive_name Joe_Biden \
    --output_dir ./results/
```

### 场景3：自定义角度范围
```bash
# 只生成正面和轻微转动的视频
python integrated_avatar_generation.py \
    --input_image portrait.jpg \
    --source_video video.mp4 \
    --yaw_angles -10 0 10 \
    --pitch_angles -5 0 5
```

## 性能优化建议

### 1. GPU内存优化
- 如果GPU内存不足，可以减小`batch_size`（在代码中调整）
- 考虑减少同时生成的角度数量

### 2. 速度优化
- 使用SSD存储以提高I/O性能
- 确保CUDA环境正确配置
- 可以调整`max_frames`参数控制视频长度

### 3. 质量优化
- 使用高质量的输入图片（建议512x512或更高）
- 确保输入图片人脸清晰且居中
- 选择合适的驱动视频（表情丰富、清晰度高）

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案：减少batch_size或降低render_size
   ```

2. **视频格式不支持**
   ```
   错误：不支持的视频格式
   解决方案：使用支持的格式（.mp4/.mov/.avi/.mkv等）
   ```

3. **FLAME跟踪失败**
   ```
   错误：flametracking preprocess failed
   解决方案：检查输入图片质量，确保人脸清晰可见
   ```

4. **模型加载失败**
   ```
   错误：模型路径不存在
   解决方案：检查--model_name参数，确保模型文件存在
   ```

### 调试模式

在代码中设置调试标志以获取更详细的输出：
```python
# 在main()函数开始处添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 更新日志

### v1.0.0
- 整合VHAP转换和多视角生成功能
- 支持多种视频格式输入
- 自动音频处理和同步
- 完整的错误处理和状态检查

## 贡献

如有问题或建议，请提交Issue或Pull Request。

## 许可证

遵循原始LAM项目的Apache License 2.0许可证。 