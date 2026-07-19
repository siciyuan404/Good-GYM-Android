"""
ONNX -> TF SavedModel -> TFLite 转换脚本
适配 Good-GYM 使用的 RTMPose (simcc) 与 YOLOX (humanart) 模型。

用法:
    python convert_onnx_to_tflite.py --onnx ../../models/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.onnx
    python convert_onnx_to_tflite.py --onnx ../../models/yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx --input-shape 1,416,416,3

可选:
    --quantize float16|int8|none   量化方式，默认 float16
    --samples N                    int8 量化代表性数据集样本数，默认 100
    --keep-savedmodel              转换后保留 SavedModel（默认清理）
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# 转换流程依赖: onnx / onnx2tf / tensorflow
# 验证 onnx 输入输出张量
try:
    import onnx
    from onnx import shape_inference
except ImportError:
    print("[ERROR] 缺少 onnx 包, 请执行: pip install onnx")
    sys.exit(1)

# onnx2tf 在导入时即初始化, 这里做软依赖: 真正转换时才报错
_onnx2tf = None
def _ensure_onnx2tf():
    global _onnx2tf
    if _onnx2tf is not None:
        return _onnx2tf
    try:
        import onnx2tf as _m
    except ImportError:
        print("[ERROR] 缺少 onnx2tf, 请执行: pip install onnx2tf")
        sys.exit(1)
    _onnx2tf = _m
    return _onnx2tf


# tensorflow 同样懒加载, 仅 SavedModel -> TFLite 阶段需要
_tf = None
def _ensure_tf():
    global _tf
    if _tf is not None:
        return _tf
    try:
        import tensorflow as _m
    except ImportError:
        print("[ERROR] 缺少 tensorflow, 请执行: pip install tensorflow==2.15.* (CPU 版即可)")
        sys.exit(1)
    _tf = _m
    return _tf


# RTMPose 标准预处理常数 (输入 0-255, 不先 /255)
RTMPOSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RTMPOSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# YOLOX 标准预处理 (0-1 范围, 不减 mean)
# 注意: YOLOX 默认输入是 BGR -> RGB 后 / 255, 不做 mean/std normalization


def inspect_onnx(onnx_path: str):
    """打印 ONNX 模型的输入输出张量信息"""
    print(f"\n[INFO] 探测 ONNX 模型: {onnx_path}")
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)

    print("  输入张量:")
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    name={inp.name}, shape={dims}")

    print("  输出张量:")
    for out in model.graph.output:
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    name={out.name}, shape={dims}")

    return model


def onnx_to_savedmodel(onnx_path: str, output_dir: str):
    """用 onnx2tf 把 ONNX 转 TF SavedModel (NHWC)"""
    onnx2tf = _ensure_onnx2tf()
    print(f"\n[INFO] ONNX -> TF SavedModel")
    print(f"  源: {onnx_path}")
    print(f"  目标: {output_dir}")

    if os.path.exists(output_dir):
        print(f"  清理已有目录: {output_dir}")
        shutil.rmtree(output_dir)

    # onnx2tf 1.28.x 内部无条件调用 download_test_image_data() 加载 .npy 测试图像,
    # 但 numpy 1.26 默认 allow_pickle=False 会拒绝加载. 这里临时 patch np.load 默认允许 pickle.
    _orig_np_load = np.load
    def _patched_np_load(*args, **kwargs):
        if "allow_pickle" not in kwargs:
            kwargs["allow_pickle"] = True
        return _orig_np_load(*args, **kwargs)
    np.load = _patched_np_load
    try:
        # onnx2tf 默认输出 NHWC, 适合后续 TFLite
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            non_verbose=True,
        )
    finally:
        np.load = _orig_np_load

    # 检查输出
    savedmodel_path = os.path.join(output_dir, "saved_model.pb")
    if not os.path.exists(savedmodel_path):
        # onnx2tf 也可能直接放在 output_dir 根下
        alt = os.path.join(output_dir, "model_float32.pb")
        if os.path.exists(alt):
            print(f"  SavedModel 已生成 (alt 形式): {alt}")
        else:
            raise RuntimeError(f"SavedModel 未生成于: {savedmodel_path}")

    print(f"  SavedModel 生成完毕")
    return output_dir


def make_representative_dataset(input_shape, samples, model_kind="rtmpose"):
    """生成 int8 量化用的代表性数据集生成器"""
    def gen():
        for _ in range(samples):
            if model_kind == "rtmpose":
                # 模拟自然图像: BGR 噪声 -> RGB -> 0-255 范围
                # TFLite 输入约定为已 normalize 后的 float32 NHWC
                raw = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
                arr = raw.astype(np.float32)[..., ::-1]  # BGR -> RGB
                arr = (arr - RTMPOSE_MEAN) / RTMPOSE_STD
                yield [arr]
            else:
                # YOLOX: 0-1 范围
                arr = np.random.rand(*input_shape).astype(np.float32)
                yield [arr]
    return gen


def savedmodel_to_tflite(savedmodel_dir, output_tflite, quantize="float16",
                         input_shape=None, samples=100, model_kind="rtmpose"):
    """TF SavedModel -> TFLite, 支持不同量化精度"""
    tf = _ensure_tf()
    print(f"\n[INFO] SavedModel -> TFLite ({quantize})")
    print(f"  源: {savedmodel_dir}")
    print(f"  目标: {output_tflite}")

    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)

    if quantize == "none":
        # 不做优化, 保持 float32
        pass
    elif quantize == "float16":
        # float16 量化: 体积减半, 精度损失极小
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        # int8 PTQ 量化: 体积最小, 推理最快, 需要 representative dataset
        if input_shape is None:
            raise ValueError("int8 量化需要 --input-shape")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(
            input_shape, samples, model_kind
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError(f"未知量化方式: {quantize}")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_tflite) or ".", exist_ok=True)
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite 生成完毕, 体积: {size_kb:.1f} KB")
    return output_tflite


def detect_model_kind(onnx_filename: str, input_shape):
    """根据文件名与输入形状推断模型类型, 决定预处理方式"""
    name = os.path.basename(onnx_filename).lower()
    if "rtmpose" in name or "simcc" in name:
        return "rtmpose"
    if "yolox" in name or "humanart" in name:
        return "yolox"
    # 兜底: 若输入是 256x192x3, 当作 rtmpose
    if input_shape and len(input_shape) == 4:
        _, h, w, _ = input_shape
        if (h, w) == (256, 192):
            return "rtmpose"
        if (h, w) == (416, 416):
            return "yolox"
    return "rtmpose"


def parse_shape(s: str):
    """解析 '1,256,192,3' 为 tuple"""
    return tuple(int(x) for x in s.split(","))


def main():
    parser = argparse.ArgumentParser(description="ONNX -> TFLite 转换 (Good-GYM 模型专用)")
    parser.add_argument("--onnx", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output-dir", default=None, help="输出目录, 默认与 ONNX 同级 tflite/")
    parser.add_argument("--input-shape", default=None,
                        help="NHWC 输入形状, 如 '1,256,192,3' (int8 量化必须)")
    parser.add_argument("--quantize", default="float16",
                        choices=["none", "float16", "int8"],
                        help="量化方式, 默认 float16")
    parser.add_argument("--samples", type=int, default=100,
                        help="int8 量化代表性数据集样本数, 默认 100")
    parser.add_argument("--keep-savedmodel", action="store_true",
                        help="转换后保留 SavedModel 目录")
    parser.add_argument("--inspect-only", action="store_true",
                        help="仅探测 ONNX 输入输出, 不执行转换")
    args = parser.parse_args()

    onnx_path = args.onnx
    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX 文件不存在: {onnx_path}")
        sys.exit(1)

    # 第一步: 探测 ONNX
    model = inspect_onnx(onnx_path)

    if args.inspect_only:
        return

    # 推断输入形状 (用于 int8 量化与预处理推断)
    input_shape = parse_shape(args.input_shape) if args.input_shape else None
    if input_shape is None:
        # 从 ONNX 输入张量自动推断 (NCHW -> NHWC)
        first_input = model.graph.input[0]
        dims = [d.dim_value for d in first_input.type.tensor_type.shape.dim]
        if len(dims) == 4 and all(d > 0 for d in dims):
            n, c, h, w = dims
            input_shape = (n, h, w, c)
            print(f"[INFO] 自动推断 NHWC 输入形状: {input_shape}")

    if args.quantize == "int8" and input_shape is None:
        print("[ERROR] int8 量化必须提供 --input-shape")
        sys.exit(1)

    model_kind = detect_model_kind(onnx_path, input_shape)
    print(f"[INFO] 推断模型类型: {model_kind} (决定预处理方式)")

    # 输出目录与文件名
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(onnx_path) or ".", "tflite")
    os.makedirs(args.output_dir, exist_ok=True)

    base = Path(onnx_path).stem
    savedmodel_dir = os.path.join(args.output_dir, f"{base}_savedmodel")

    # 第二步: ONNX -> SavedModel
    onnx_to_savedmodel(onnx_path, savedmodel_dir)

    # 第三步: SavedModel -> TFLite
    tflite_path = os.path.join(args.output_dir, f"{base}_{args.quantize}.tflite")
    savedmodel_to_tflite(
        savedmodel_dir, tflite_path,
        quantize=args.quantize,
        input_shape=input_shape,
        samples=args.samples,
        model_kind=model_kind,
    )

    # 清理 SavedModel
    if not args.keep_savedmodel:
        print(f"\n[INFO] 清理 SavedModel: {savedmodel_dir}")
        shutil.rmtree(savedmodel_dir, ignore_errors=True)
    else:
        print(f"\n[INFO] 已保留 SavedModel: {savedmodel_dir}")

    print(f"\n[OK] 转换完成")
    print(f"  TFLite: {tflite_path}")
    print(f"  下一步: python verify_tflite.py --onnx {onnx_path} --tflite {tflite_path}")


if __name__ == "__main__":
    main()
