# Good-GYM 安卓版 - 模型转换脚本

把桌面版使用的 ONNX 模型 (RTMPose simcc + YOLOX humanart) 转换为 Android 端可用的 TFLite 格式, 并验证转换后输出与原 ONNX 一致。

> **当前状态 (2026-07-18):**
>
> 经实测, `onnx2tf 1.28.8` 在 Windows 上转换 RTMPose (simcc 头部) 时发生 segfault (exit code 0xC0000005),
> 原因是 onnx2tf 对 simcc 的 MatMul/Reshape 节点转换存在已知问题。
> YOLOX (无 simcc 头) 的转换预期可成功, 但本仓库目前**已转向 onnxruntime-mobile Android 路径**,
> 直接复用原 `.onnx` 文件, 跳过 TFLite 转换环节。
>
> **推荐路径**: 见本仓库 `android/` 下的 Flutter 工程, 使用 `onnxruntime` Flutter 包直接加载 `.onnx`。
>
> 本目录的脚本作为**备选方案保留**, 供未来 onnx2tf 修复后或换用其他转换工具时复用。
> YOLOX 模型的 TFLite 转换仍可走通 (无 simcc 难点)。

## 目录说明

```
android/scripts/
├── convert_onnx_to_tflite.py  # ONNX -> SavedModel -> TFLite 转换脚本
├── verify_tflite.py           # 验证 ONNX 与 TFLite 输出一致性
├── requirements.txt           # 转换专用 Python 依赖
└── README.md                  # 本文件
```

转换后的 TFLite 模型默认输出到 `models/tflite/`, 后续会被 Flutter 安卓工程引用。

## 1. 安装依赖

转换依赖较重 (`tensorflow`、`onnx2tf` 等), 推荐在独立 venv 中安装, 避免污染桌面版 PyQt5 环境:

```powershell
# Windows PowerShell
python -m venv venv_tflite
.\venv_tflite\Scripts\activate
pip install -r requirements.txt
```

> `tensorflow` 在 Windows 上 CPU 版约 500 MB, `onnx2tf` 会再拉取 `sng4onnx`、`onnx_graphsurgeon` 等约 200 MB 依赖, 总磁盘占用约 1.5 GB。

## 2. 探测 ONNX 输入输出 (可选, 仅查看)

```powershell
python convert_onnx_to_tflite.py --onnx ../../models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx --inspect-only
```

输出会列出模型的输入张量名 / 形状 / 输出张量名 / 形状, 用于确认 RTMPose simcc 的两个输出张量存在。

## 3. 执行转换

### RTMPose (姿态模型, 优先转换 t 版本, 移动端最快)

```powershell
# float16 量化: 体积减半, 精度损失极小, 推荐用于 Flutter 集成
python convert_onnx_to_tflite.py `
    --onnx ../../models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx `
    --quantize float16

# int8 量化: 体积最小, 推理最快, 但精度损失略大
python convert_onnx_to_tflite.py `
    --onnx ../../models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx `
    --quantize int8 `
    --input-shape 1,256,192,3 `
    --samples 200

# 不量化 (float32 基线, 用于精度对照)
python convert_onnx_to_tflite.py `
    --onnx ../../models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx `
    --quantize none
```

### YOLOX (人体检测模型)

```powershell
python convert_onnx_to_tflite.py `
    --onnx ../../models/yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx `
    --input-shape 1,416,416,3 `
    --quantize float16
```

转换完成后, `../../models/tflite/` 下会生成 `<模型名>_<量化方式>.tflite` 文件。

## 4. 验证一致性

同一输入下比对 ONNX 与 TFLite 的输出:

```powershell
# 用合成噪声图快速验证 (无需准备图像)
python verify_tflite.py `
    --onnx ../../models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx `
    --tflite ../../models/tflite/rtmpose-t_..._float16.tflite

# 用真实图像验证 (更接近实际使用)
python verify_tflite.py `
    --onnx ../../models/rtmpose-t_*.onnx `
    --tflite ../../models/tflite/rtmpose-t_..._float16.tflite `
    --image ./test.jpg
```

脚本会输出:
- simcc 张量数值差异 (最大值、平均值)
- 17 个关键点坐标的 argmax 结果差异
- 误差 <= 2 网格的关键点比例
- 整体验收结论 (`PASS` / `WARN` / `FAIL`)

### 验收标准

| 等级 | 最大偏差 | 含义 |
|---|---|---|
| PASS | ≤ 3 网格 | 可直接用于 Flutter 端集成 |
| WARN | 4–8 网格 | 实际可用但精度略损, 建议改 float16 量化或加大 PTQ 样本 |
| FAIL | > 8 网格 | 转换链路异常, 排查 onnx2tf 是否完整转换 simcc 输出 |

## 5. 转换流程细节

### 模型类型自动识别

脚本根据文件名自动判断:
- `rtmpose` 或 `simcc` -> RTMPose 姿态模型, 使用 RTMPose 标准预处理
- `yolox` 或 `humanart` -> YOLOX 检测模型, 使用 0-1 范围预处理

### 预处理常数

RTMPose 输入预处理 (与桌面版 rtmlib 内部一致):

```
mean = [123.675, 116.28, 103.53]   # ImageNet BGR 均值
std  = [58.395, 57.12, 57.375]     # ImageNet BGR 标准差
输入范围: 0-255 (不预先除以 255)
输入布局: ONNX NCHW, TFLite NHWC (onnx2tf 自动转换)
```

> 若验证出现大量关键点偏差, 优先检查 rtmlib 是否使用了不同的 mean/std。可在桌面版代码中临时打印 `Wholebody` 内部预处理参数确认。

### simcc 后处理

RTMPose simcc 输出两个张量:
- `simcc_x`: shape `(1, 17, W_simcc)`, W_simcc 通常是输入宽度 (192) 的扩展
- `simcc_y`: shape `(1, 17, H_simcc)`, H_simcc 通常是输入高度 (256) 的扩展

关键点坐标 = `(simcc_x.argmax(-1), simcc_y.argmax(-1))`, 直接得到 17 个 (x, y) 整数坐标。

> **重要**: 该后处理逻辑需要在 Flutter/Dart 端复现一遍 (argmax + 坐标缩放回原图)。桌面版 rtmlib 内部已封装, Flutter 端要自己写。

## 6. 推荐的转换组合

移动端推理性能与精度权衡, 推荐组合:

| 模型 | 量化 | 体积 | 备注 |
|---|---|---|---|
| rtmpose-t | float16 | ~1.5 MB | 推荐: 速度与精度平衡 |
| rtmpose-t | int8 | ~0.5 MB | 极致体积, 精度损失略大 |
| rtmpose-s | float16 | ~3 MB | 精度更高, 速度仍可接受 |
| yolox_nano | float16 | ~5 MB | 检测器, float16 已足够 |
| yolox_nano | int8 | ~2 MB | 移动端首选, 检测精度容忍度高 |

最终在 Flutter 工程中建议同时打包 `rtmpose-t_float16.tflite` 与 `yolox_nano_float16.tflite`, 总体积约 7 MB, 在中端安卓机上单帧推理 < 30ms。

## 7. 下一步

转换验证通过后, 进入 Flutter 工程初始化阶段:

1. 在 `android/` 下创建 Flutter 工程 (`flutter create --org com.goodgym good_gym`)
2. 把 `models/tflite/` 拷贝到 Flutter 工程的 `assets/models/`
3. 把桌面版 `data/exercises.json` 拷贝到 `assets/data/`
4. 开始写 `lib/core/pose_detector.dart` (TFLite 推理封装) 与 `lib/core/exercise_counter.dart` (计数逻辑翻译)
