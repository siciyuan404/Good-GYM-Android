"""
验证脚本: 比对同一输入下 ONNX 与 TFLite 模型输出的一致性。

工作原理:
  1. 用同一张图像 (或合成噪声) 做预处理
  2. 分别喂给 ONNX (onnxruntime) 与 TFLite (tflite_runtime / tensorflow.lite)
  3. 处理两路输出:
     - RTMPose (simcc): 对 simcc_x / simcc_y 做 argmax 得到 17 个关键点坐标
     - YOLOX (检测): 比对原始 logits 张量数值差异
  4. 报告关键点坐标绝对误差 (像素)、最大偏差、匹配率

用法:
    python verify_tflite.py --onnx ../../models/rtmpose-t_*.onnx --tflite ../../models/tflite/rtmpose-t_float16.tflite
    python verify_tflite.py --onnx ... --tflite ... --image ./test.jpg
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] 缺少 onnxruntime, 请执行: pip install onnxruntime")
    sys.exit(1)

# TFLite 推理: 优先用 tflite_runtime (轻量), 否则回退到 tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_BACKEND = "tflite_runtime"
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_BACKEND = "tensorflow.lite"
    except ImportError:
        print("[ERROR] 缺少 TFLite 推理库, 请执行: pip install tflite-runtime 或 pip install tensorflow")
        sys.exit(1)


# RTMPose 标准预处理常数 (与 convert 脚本保持一致)
RTMPOSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RTMPOSE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ---------- ONNX 侧 ----------

def load_onnx_session(onnx_path, device="cpu"):
    """加载 ONNX 推理会话, 返回 (session, input_names, output_names)"""
    providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inputs = [(i.name, [d if isinstance(d, int) else d for d in i.shape]) for i in sess.get_inputs()]
    outputs = [(o.name, [d if isinstance(d, int) else d for d in o.shape]) for o in sess.get_outputs()]
    print(f"[ONNX] 输入: {inputs}")
    print(f"[ONNX] 输出: {outputs}")
    return sess, inputs, outputs


def run_onnx(sess, input_tensor, input_name):
    """跑 ONNX 推理, 返回输出张量字典"""
    feeds = {input_name: input_tensor}
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outs))


# ---------- TFLite 侧 ----------

def load_tflite_interpreter(tflite_path):
    """加载 TFLite 解释器, 返回 (interpreter, input_details, output_details)"""
    interp = tflite.Interpreter(model_path=tflite_path, num_threads=1)
    interp.allocate_tensors()
    in_details = interp.get_input_details()
    out_details = interp.get_output_details()
    print(f"[TFLite] 输入: {[(d['name'], d['shape'], d['dtype']) for d in in_details]}")
    print(f"[TFLite] 输出: {[(d['name'], d['shape'], d['dtype']) for d in out_details]}")
    return interp, in_details, out_details


def run_tflite(interp, input_array):
    """跑 TFLite 推理, 返回输出张量列表 (按 output_details 顺序)"""
    in_detail = interp.get_input_details()[0]
    in_dtype = in_detail["dtype"]

    # int8 量化模型需要把 float 输入量化到 int8
    if in_dtype == np.int8:
        scale, zero_point = in_detail["quantization"]
        input_array = (input_array / scale + zero_point).round().astype(np.int8)
    elif in_dtype == np.uint8:
        scale, zero_point = in_detail["quantization"]
        input_array = (input_array / scale + zero_point).round().astype(np.uint8)
    elif in_dtype == np.float16:
        input_array = input_array.astype(np.float16)
    else:
        input_array = input_array.astype(np.float32)

    # 确保形状匹配
    expected_shape = tuple(in_detail["shape"])
    if input_array.shape != expected_shape:
        # 兼容 batch=1 (TFLite 可能固定 batch)
        if input_array.shape[1:] == expected_shape[1:]:
            input_array = input_array[:expected_shape[0]]
        else:
            raise ValueError(f"输入形状不匹配: 提供 {input_array.shape}, 期望 {expected_shape}")

    interp.set_tensor(in_detail["index"], input_array)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


# ---------- 预处理 ----------

def preprocess_rtmpose(image_hwc_bgr, target_h=256, target_w=192):
    """RTMPose 预处理: BGR -> RGB -> resize -> normalize -> NCHW"""
    import cv2
    img = cv2.cvtColor(image_hwc_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h))  # cv2: (w, h)
    img = img.astype(np.float32)
    img = (img - RTMPOSE_MEAN) / RTMPOSE_STD
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # NCHW
    return img


def preprocess_rtmpose_tflite(image_hwc_bgr, target_h=256, target_w=192):
    """RTMPose 预处理为 TFLite 的 NHWC 格式"""
    import cv2
    img = cv2.cvtColor(image_hwc_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_w, target_h))
    img = img.astype(np.float32)
    img = (img - RTMPOSE_MEAN) / RTMPOSE_STD
    img = np.expand_dims(img, axis=0)  # NHWC
    return img


def preprocess_yolox(image_hwc_bgr, target_size=416, for_tflite=False):
    """YOLOX 预处理: letterbox 到方形, /255"""
    import cv2
    h, w = image_hwc_bgr.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(image_hwc_bgr, (nw, nh))
    pad_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_img[:nh, :nw] = img
    img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if for_tflite:
        return np.expand_dims(img, axis=0)  # NHWC
    return np.transpose(np.expand_dims(img, axis=0), (0, 3, 1, 2))  # NCHW


# ---------- simcc 后处理 ----------

def simcc_to_keypoints(simcc_x, simcc_y):
    """对 simcc 输出做 argmax 得到 17 个关键点 (x, y) 坐标

    Args:
        simcc_x: shape (1, 17, W_simcc)
        simcc_y: shape (1, 17, H_simcc)
    Returns:
        keypoints: shape (17, 2), 单位是 simcc 网格坐标
    """
    # 在最后一个维度做 argmax
    x = simcc_x[0].argmax(axis=-1)  # (17,)
    y = simcc_y[0].argmax(axis=-1)  # (17,)
    return np.stack([x, y], axis=1).astype(np.float32)  # (17, 2)


def detect_model_kind(onnx_path):
    """根据文件名推断模型类型"""
    name = os.path.basename(onnx_path).lower()
    if "rtmpose" in name or "simcc" in name:
        return "rtmpose"
    if "yolox" in name or "humanart" in name:
        return "yolox"
    return "rtmpose"


# ---------- 验证主流程 ----------

def verify_rtmpose(onnx_path, tflite_path, image=None, target_h=256, target_w=192):
    """验证 RTMPose (simcc) 模型"""
    print("\n" + "=" * 60)
    print(" 验证 RTMPose (simcc 输出)")
    print("=" * 60)

    # 准备测试输入
    if image is not None:
        import cv2
        if not os.path.exists(image):
            print(f"[ERROR] 图像不存在: {image}")
            sys.exit(1)
        raw = cv2.imread(image)
        print(f"[INFO] 使用真实图像: {image}, 原始尺寸 {raw.shape}")
    else:
        # 合成噪声图: 模拟自然图像分布
        raw = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        print("[INFO] 使用合成噪声图 (480x640x3) 作为输入")

    # ONNX 路径
    print("\n[1/2] ONNX 推理...")
    sess, onnx_inputs, onnx_outputs = load_onnx_session(onnx_path)
    onnx_input_name = onnx_inputs[0][0]
    input_nchw = preprocess_rtmpose(raw, target_h, target_w)
    onnx_out = run_onnx(sess, input_nchw, onnx_input_name)
    onnx_out_names = list(onnx_out.keys())
    print(f"  ONNX 输出键: {onnx_out_names}")

    # TFLite 路径
    print("\n[2/2] TFLite 推理...")
    interp, in_details, out_details = load_tflite_interpreter(tflite_path)
    input_nhwc = preprocess_rtmpose_tflite(raw, target_h, target_w)
    tflite_out = run_tflite(interp, input_nhwc)
    tflite_out_names = [d["name"] for d in out_details]
    print(f"  TFLite 输出键: {tflite_out_names}")

    # 找 simcc 输出 (通常是两个张量, 名称含 x/y 或 0/1)
    # ONNX: 找最长维度的两个 (simcc 通常长度 192/256)
    def find_simcc_pair(outs_dict_or_list, is_dict):
        candidates = []
        if is_dict:
            for k, v in outs_dict_or_list.items():
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[1] == 17:
                    candidates.append((k, v))
        else:
            for i, v in enumerate(outs_dict_or_list):
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[1] == 17:
                    candidates.append((i, v))
        # 按 simcc 长度从大到小排序: simcc_x (W_simcc) > simcc_y (H_simcc)
        candidates.sort(key=lambda kv: kv[1].shape[-1], reverse=True)
        return candidates

    onnx_simcc = find_simcc_pair(onnx_out, is_dict=True)
    tflite_simcc = find_simcc_pair(tflite_out, is_dict=False)

    if len(onnx_simcc) < 2 or len(tflite_simcc) < 2:
        print("\n[WARN] 未能识别到两个 simcc 张量, 直接比对原始输出数值")
        # 回退: 直接对比第一个输出张量
        for i, (name, arr) in enumerate(onnx_simcc or []):
            t_arr = tflite_simcc[i][1] if i < len(tflite_simcc) else None
            if t_arr is not None:
                diff = np.abs(arr.astype(np.float32) - t_arr.astype(np.float32))
                print(f"  [{name}] 输出张量均值: ONNX={arr.mean():.4f} TFLite={t_arr.mean():.4f}, "
                      f"最大绝对差={diff.max():.4f}, 平均绝对差={diff.mean():.4f}")
        return

    onnx_x, onnx_y = onnx_simcc[0][1], onnx_simcc[1][1]
    tflite_x, tflite_y = tflite_simcc[0][1], tflite_simcc[1][1]

    # simcc 张量数值差异 (反映量化误差)
    diff_x = np.abs(onnx_x.astype(np.float32) - tflite_x.astype(np.float32))
    diff_y = np.abs(onnx_y.astype(np.float32) - tflite_y.astype(np.float32))
    print(f"\n[数值差异] simcc_x 最大={diff_x.max():.4f} 平均={diff_x.mean():.4f}")
    print(f"[数值差异] simcc_y 最大={diff_y.max():.4f} 平均={diff_y.mean():.4f}")

    # argmax 后的关键点坐标差异 (这才是真正影响应用的指标)
    kp_onnx = simcc_to_keypoints(onnx_x, onnx_y)
    kp_tflite = simcc_to_keypoints(tflite_x, tflite_y)
    kp_diff = np.abs(kp_onnx - kp_tflite)
    kp_names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
                "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
                "l_knee", "r_knee", "l_ankle", "r_ankle"]

    print("\n[关键点坐标差异] (simcc 网格单位, 1 像素 = ~1 网格)")
    print(f"  {'关键点':<12} {'ONNX(x,y)':<18} {'TFLite(x,y)':<18} {'差值(x,y)':<14}")
    for i, name in enumerate(kp_names):
        ox, oy = kp_onnx[i]
        tx, ty = kp_tflite[i]
        dx, dy = kp_diff[i]
        flag = " OK" if (dx <= 2 and dy <= 2) else (" !! 偏差大" if (dx > 5 or dy > 5) else " ~")
        print(f"  {name:<12} ({int(ox):>3},{int(oy):>3})       ({int(tx):>3},{int(ty):>3})       ({dx:>4.1f},{dy:>4.1f}){flag}")

    # 汇总
    max_diff = kp_diff.max()
    mean_diff = kp_diff.mean()
    matched = np.all(kp_diff <= 2, axis=1).sum()
    print(f"\n[汇总]")
    print(f"  最大偏差: {max_diff:.1f} 网格")
    print(f"  平均偏差: {mean_diff:.2f} 网格")
    print(f"  误差<=2 的关键点: {matched}/17 ({matched/17*100:.1f}%)")

    # 验收标准
    if max_diff <= 3:
        print(f"\n[PASS] 关键点偏差在可接受范围 (<=3 网格), 可用于 Flutter 端集成")
    elif max_diff <= 8:
        print(f"\n[WARN] 偏差偏大, 实际使用可能略有不准确. 建议改用 float16 量化或检查预处理")
    else:
        print(f"\n[FAIL] 偏差过大, 模型转换链路存在异常. 请检查:")
        print(f"   - onnx2tf 是否完整转换了 simcc 输出")
        print(f"   - 预处理是否一致 (mean/std, 输入范围 0-255 vs 0-1)")
        print(f"   - 是否使用了过度激进的量化")
        sys.exit(2)


def verify_yolox(onnx_path, tflite_path, image=None, target_size=416):
    """验证 YOLOX 检测模型"""
    print("\n" + "=" * 60)
    print(" 验证 YOLOX (检测输出)")
    print("=" * 60)

    if image is not None:
        import cv2
        raw = cv2.imread(image)
    else:
        raw = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

    print("\n[1/2] ONNX 推理...")
    sess, onnx_inputs, onnx_outputs = load_onnx_session(onnx_path)
    onnx_input_name = onnx_inputs[0][0]
    input_nchw = preprocess_yolox(raw, target_size, for_tflite=False)
    onnx_out = run_onnx(sess, input_nchw, onnx_input_name)

    print("\n[2/2] TFLite 推理...")
    interp, in_details, out_details = load_tflite_interpreter(tflite_path)
    input_nhwc = preprocess_yolox(raw, target_size, for_tflite=True)
    tflite_out = run_tflite(interp, input_nhwc)

    # 比对原始输出张量
    print("\n[输出数值差异]")
    onnx_arr = list(onnx_out.values())[0]
    tflite_arr = tflite_out[0]
    # 形状对齐
    if onnx_arr.shape != tflite_arr.shape:
        # ONNX 是 NCHW, TFLite 是 NHWC
        if onnx_arr.ndim == 4 and tflite_arr.ndim == 4:
            onnx_arr = np.transpose(onnx_arr, (0, 2, 3, 1))

    diff = np.abs(onnx_arr.astype(np.float32) - tflite_arr.astype(np.float32))
    print(f"  ONNX 输出形状: {onnx_arr.shape}, 均值 {onnx_arr.mean():.4f}")
    print(f"  TFLite 输出形状: {tflite_arr.shape}, 均值 {tflite_arr.mean():.4f}")
    print(f"  最大绝对差: {diff.max():.4f}")
    print(f"  平均绝对差: {diff.mean():.4f}")
    print(f"  相对误差: {(diff.mean() / (np.abs(onnx_arr).mean() + 1e-6) * 100):.2f}%")

    if diff.max() < 0.5:
        print("\n[PASS] 输出数值差异可接受")
    elif diff.max() < 2.0:
        print("\n[WARN] 输出数值略有差异, 实际检测影响应有限")
    else:
        print("\n[FAIL] 数值差异过大, 需排查量化与预处理")


def main():
    parser = argparse.ArgumentParser(description="验证 ONNX 与 TFLite 输出一致性")
    parser.add_argument("--onnx", required=True, help="ONNX 模型路径")
    parser.add_argument("--tflite", required=True, help="TFLite 模型路径")
    parser.add_argument("--image", default=None, help="测试图像 (可选, 缺省用合成噪声)")
    parser.add_argument("--input-h", type=int, default=256, help="RTMPose 输入高度, 默认 256")
    parser.add_argument("--input-w", type=int, default=192, help="RTMPose 输入宽度, 默认 192")
    parser.add_argument("--det-size", type=int, default=416, help="YOLOX 输入尺寸, 默认 416")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"[ERROR] ONNX 不存在: {args.onnx}")
        sys.exit(1)
    if not os.path.exists(args.tflite):
        print(f"[ERROR] TFLite 不存在: {args.tflite}")
        sys.exit(1)

    print(f"[INFO] ONNX: {args.onnx}")
    print(f"[INFO] TFLite: {args.tflite}")
    print(f"[INFO] TFLite 推理后端: {TFLITE_BACKEND}")

    model_kind = detect_model_kind(args.onnx)
    if model_kind == "rtmpose":
        verify_rtmpose(args.onnx, args.tflite, args.image, args.input_h, args.input_w)
    else:
        verify_yolox(args.onnx, args.tflite, args.image, args.det_size)


if __name__ == "__main__":
    main()
