# VHAP转LAM转换工具使用说明

这个工具用于将VHAP_track的输出转换为LAM兼容的驱动文件格式，并自动处理视频文件。

## 新增功能

- ✅ 自动处理视频文件（支持.mp4/.mov/.avi/.mkv/.wmv/.flv/.webm等格式）
- ✅ 自动提取音频文件
- ✅ 生成完全符合multi_view_image_generation.py要求的输入数据配置
- ✅ 智能文件检查，确保所有必需文件完整

## 使用方法

### 1. 基本使用（使用默认路径）

```bash
python convert_vhap_to_lam.py
```

### 2. 自定义参数

```bash
python convert_vhap_to_lam.py \
    --vhap_dir "/path/to/vhap/output" \
    --output_dir "/path/to/lam/assets/sample_motion/export" \
    --target_name "my_custom_drive" \
    --max_frames 264 \
    --source_video_path "/path/to/source/video.mov"
```

### 3. 参数说明

- `--vhap_dir`: VHAP输出目录路径
- `--output_dir`: LAM的assets/sample_motion/export目录路径
- `--target_name`: 目标驱动名称（生成的文件将以此命名）
- `--max_frames`: 最大帧数（默认264帧）
- `--source_video_path`: 源视频文件路径（支持多种格式）

## 输出文件结构

转换完成后，将在`{output_dir}/{target_name}/`目录下生成以下文件：

```
custom_jp_drive/
├── custom_jp_drive.mp4      # 处理后的驱动视频
├── custom_jp_drive.wav      # 提取的音频文件
├── flame_param/             # FLAME参数目录
│   ├── 00000.npz
│   ├── 00001.npz
│   └── ...
├── flame_params.json        # 合并的FLAME参数文件
├── transforms.json          # 变换矩阵文件
└── canonical_flame_param.npz # 规范化FLAME参数
```

## 与multi_view_image_generation.py集成

转换完成后，直接在`multi_view_image_generation.py`中设置：

```python
# 驱动数据名称
drive_name = 'custom_jp_drive'  # 与convert_vhap_to_lam.py中的target_name保持一致

# 无需再设置source_video_path，因为视频和音频已经处理完毕
```

## 支持的视频格式

- ✅ MP4 (.mp4)
- ✅ QuickTime (.mov)
- ✅ AVI (.avi)
- ✅ Matroska (.mkv)
- ✅ Windows Media (.wmv)
- ✅ Flash Video (.flv)
- ✅ WebM (.webm)

## 错误处理

脚本会自动检查：
- 输入文件是否存在
- 视频格式是否支持
- 音频提取是否成功
- 所有必需文件是否完整

如果某些步骤失败，会提供详细的错误信息和建议。

## 示例

```bash
# 转换VHAP数据并处理视频
python convert_vhap_to_lam.py \
    --vhap_dir "/root/autodl-tmp/VHAP_track/mono_jp/export_epoch0" \
    --source_video_path "/root/autodl-tmp/datasets/mono_jp/apple_jp_3.mov" \
    --target_name "apple_jp_drive"
```

转换完成后，您可以直接在`multi_view_image_generation.py`中使用：

```python
drive_name = 'apple_jp_drive'
``` 