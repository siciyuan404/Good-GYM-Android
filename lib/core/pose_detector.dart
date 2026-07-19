/// 姿态检测核心 - onnxruntime 推理封装
///
/// 直接复用桌面版 [models/](file:///f:/git/Good-GYM/models) 下的两个 ONNX 模型:
/// - YOLOX nano: 人体检测 (416x416 输入, 已内置 NMS, 输出 dets+labels)
/// - RTMPose-t: 17 关键点姿态估计 (256x192 输入, simcc 输出)
///
/// 与桌面版 [core/rtmpose_processor.py](file:///f:/git/Good-GYM/core/rtmpose_processor.py) 的推理流程对齐:
///   1. YOLOX 检测人体 bbox
///   2. 取最高置信度的 bbox, crop + resize 到 256x192
///   3. RTMPose 推理得到 simcc_x (1,17,384) + simcc_y (1,17,512)
///   4. argmax 得到 17 个关键点坐标, 缩放回原图坐标系
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart' as ort;

/// 单个检测框 (YOLOX 输出)
class Detection {
  final double x1, y1, x2, y2; // 原图坐标系下的 bbox
  final double score;
  const Detection(this.x1, this.y1, this.x2, this.y2, this.score);
}

/// 姿态检测结果: 17 关键点 + 检测框
class PoseResult {
  /// 17 个关键点, 每个为 [x, y] (原图坐标系), 已过滤低置信度点为 (0, 0)
  final List<Float64List> keypoints;
  final Detection? detection;
  PoseResult(this.keypoints, this.detection);
}

/// YOLOX + RTMPose 推理封装
///
/// 资源管理约定:
/// - [init] 创建 session 与 options, [dispose] 释放
/// - 每次 [run] 调用后, 输入张量 / RunOptions / 输出张量都需要显式 release
class PoseDetector {
  static const int _yoloxInputSize = 416;
  static const int _rtmposeInputW = 192; // 宽
  static const int _rtmposeInputH = 256; // 高
  static const double _confThreshold = 0.5; // 关键点置信度阈值, 与桌面版一致

  // RTMPose 标准预处理常数 (与桌面版 rtmlib 内部一致)
  static final _mean = Float32List.fromList([123.675, 116.28, 103.53]);
  static final _std = Float32List.fromList([58.395, 57.12, 57.375]);

  // onnxruntime 会话与配置 (使用后需 release)
  ort.OrtSession? _yoloxSession;
  ort.OrtSession? _rtmposeSession;
  ort.OrtSessionOptions? _yoloxOptions;
  ort.OrtSessionOptions? _rtmposeOptions;
  bool _initialized = false;

  bool get isInitialized => _initialized;

  /// 从 Flutter assets 加载 ONNX 字节并创建会话
  ///
  /// 使用 [OrtSession.fromBuffer] 直接从内存加载, 无需写临时文件
  /// (onnxruntime 1.4.1 的 OrtSession.fromFile 也行, 但 fromBuffer 更省事)
  Future<void> init() async {
    if (_initialized) return;

    // 1. 初始化 onnxruntime 全局环境 (幂等, 内部用单例保护)
    ort.OrtEnv.instance.init();

    // 2. 创建会话选项: CPU + 全图优化 + 单线程 (避免移动端线程争抢)
    _yoloxOptions = ort.OrtSessionOptions()
      ..setIntraOpNumThreads(1)
      ..setInterOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(
          ort.GraphOptimizationLevel.ortEnableAll);
    _rtmposeOptions = ort.OrtSessionOptions()
      ..setIntraOpNumThreads(1)
      ..setInterOpNumThreads(1)
      ..setSessionGraphOptimizationLevel(
          ort.GraphOptimizationLevel.ortEnableAll);

    // 3. 从 assets 读字节, 用 fromBuffer 创建会话
    final yoloxBytes =
        (await rootBundle.load('assets/models/yolox_nano.onnx'))
            .buffer
            .asUint8List();
    final rtmposeBytes =
        (await rootBundle.load('assets/models/rtmpose_t.onnx'))
            .buffer
            .asUint8List();

    _yoloxSession = ort.OrtSession.fromBuffer(yoloxBytes, _yoloxOptions!);
    _rtmposeSession =
        ort.OrtSession.fromBuffer(rtmposeBytes, _rtmposeOptions!);
    _initialized = true;
  }

  /// 释放资源
  void dispose() {
    _yoloxSession?.release();
    _rtmposeSession?.release();
    _yoloxOptions?.release();
    _rtmposeOptions?.release();
    _yoloxSession = null;
    _rtmposeSession = null;
    _yoloxOptions = null;
    _rtmposeOptions = null;
    _initialized = false;
  }

  /// 完整推理: 给定一帧图像, 返回 17 关键点 (RGB 输入)
  ///
  /// 内部调用同步的 [ort.OrtSession.run] - 由于一帧推理本身耗时,
  /// 调用方应通过 isolate 或 async 包装以避免阻塞 UI (此处 detect 返回 Future
  /// 是为了让调用方可以 await, 实际推理仍在调用线程上运行)
  Future<PoseResult?> detect(img.Image rgbImage) async {
    if (!_initialized || _yoloxSession == null || _rtmposeSession == null) {
      return null;
    }

    // 1. YOLOX 检测
    final dets = _detectPersons(rgbImage);
    if (dets.isEmpty) return null;
    final best = dets.first; // YOLOX 输出已按 score 排序

    // 2. RTMPose 关键点
    final keypoints = _estimatePose(rgbImage, best);
    if (keypoints == null) return null;
    return PoseResult(keypoints, best);
  }

  /// YOLOX 检测: letterbox 到 416x416, /255, 输出 `List<Detection>`
  List<Detection> _detectPersons(img.Image rgb) {
    final origW = rgb.width;
    final origH = rgb.height;

    // letterbox 到 416x416
    final scale = math.min(_yoloxInputSize / origW, _yoloxInputSize / origH);
    final newW = (origW * scale).round();
    final newH = (origH * scale).round();
    final resized = img.copyResize(rgb, width: newW, height: newH);
    final padded = img.Image(width: _yoloxInputSize, height: _yoloxInputSize);
    img.compositeImage(padded, resized, dstX: 0, dstY: 0);

    // NCHW float32 [1, 3, 416, 416], /255
    final input = Float32List(1 * 3 * _yoloxInputSize * _yoloxInputSize);
    var idx = 0;
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < _yoloxInputSize; y++) {
        for (var x = 0; x < _yoloxInputSize; x++) {
          final px = padded.getPixel(x, y);
          // RGB 通道顺序
          input[idx++] = px[c] / 255.0;
        }
      }
    }

    final inputOrt = ort.OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, _yoloxInputSize, _yoloxInputSize],
    );
    final inputs = <String, ort.OrtValue>{'input': inputOrt};
    final runOptions = ort.OrtRunOptions();

    // run() 同步返回 List<OrtValue?>, 顺序按 _outputNames: ['dets', 'labels']
    final outputs = _yoloxSession!.run(runOptions, inputs);

    // 显式释放输入张量与 run 选项
    inputOrt.release();
    runOptions.release();

    if (outputs.isEmpty) return [];
    final detsRaw = outputs[0]?.value as List;
    final labelsRaw = outputs.length > 1 ? outputs[1]?.value as List : null;
    // 释放输出张量
    for (final o in outputs) {
      o?.release();
    }

    if (detsRaw.isEmpty) return [];
    // dets 形状 [1, N, 5], 取 [0] 得 [N, 5]
    final detsList = detsRaw[0] as List;
    // labels 形状 [1, N], 取 [0] 得 [N]
    final labelsList =
        (labelsRaw != null && labelsRaw.isNotEmpty) ? labelsRaw[0] as List : <List>[];

    final result = <Detection>[];
    for (var i = 0; i < detsList.length; i++) {
      final det = detsList[i] as List;
      final label = labelsList.isEmpty ? 0 : labelsList[i] as int;
      // humanart 数据集, label=0 是 person
      if (label != 0) continue;
      final score = (det[4] as num).toDouble();
      if (score < 0.3) continue; // 置信度阈值
      // 还原到原图坐标系 (除以 scale)
      result.add(Detection(
        (det[0] as num).toDouble() / scale,
        (det[1] as num).toDouble() / scale,
        (det[2] as num).toDouble() / scale,
        (det[3] as num).toDouble() / scale,
        score,
      ));
    }
    // 按 score 降序
    result.sort((a, b) => b.score.compareTo(a.score));
    return result;
  }

  /// RTMPose 推理: 取 bbox 区域 crop + resize 到 256x192, simcc 后处理得到 17 关键点
  List<Float64List>? _estimatePose(img.Image rgb, Detection det) {
    // 1. crop bbox 区域, resize 到 256x192
    final x1 = math.max(0, det.x1.floor());
    final y1 = math.max(0, det.y1.floor());
    final x2 = math.min(rgb.width, det.x2.ceil());
    final y2 = math.min(rgb.height, det.y2.ceil());
    final boxW = x2 - x1;
    final boxH = y2 - y1;
    if (boxW <= 0 || boxH <= 0) return null;

    final cropped = img.copyCrop(rgb, x: x1, y: y1, width: boxW, height: boxH);
    final resized =
        img.copyResize(cropped, width: _rtmposeInputW, height: _rtmposeInputH);

    // 2. NCHW float32 [1, 3, 256, 192], (x - mean) / std
    final input = Float32List(1 * 3 * _rtmposeInputH * _rtmposeInputW);
    var idx = 0;
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < _rtmposeInputH; y++) {
        for (var x = 0; x < _rtmposeInputW; x++) {
          final px = resized.getPixel(x, y);
          input[idx++] = (px[c] - _mean[c]) / _std[c];
        }
      }
    }

    final inputOrt = ort.OrtValueTensor.createTensorWithDataList(
      input,
      [1, 3, _rtmposeInputH, _rtmposeInputW],
    );
    final inputs = <String, ort.OrtValue>{'input': inputOrt};
    final runOptions = ort.OrtRunOptions();

    // run() 同步返回 List<OrtValue?>, 顺序按 _outputNames: ['simcc_x', 'simcc_y']
    final outputs = _rtmposeSession!.run(runOptions, inputs);

    inputOrt.release();
    runOptions.release();
    if (outputs.length < 2) return null;

    // simcc_x [1, 17, 384], simcc_y [1, 17, 512]
    final simccXRaw = outputs[0]?.value as List;
    final simccYRaw = outputs[1]?.value as List;
    for (final o in outputs) {
      o?.release();
    }
    if (simccXRaw.isEmpty || simccYRaw.isEmpty) return null;
    final simccX = (simccXRaw[0] as List).cast<List>();
    final simccY = (simccYRaw[0] as List).cast<List>();
    // simcc_split = 2.0, 即 simcc 长度 = 输入尺寸 * 2
    const simccSplit = 2.0;

    // 3. argmax 得到 (x, y) 坐标 (输入图像坐标系), 再缩放回 bbox 区域, 最后回到原图
    final keypoints = <Float64List>[];
    for (var i = 0; i < 17; i++) {
      final xs = simccX[i].cast<num>();
      final ys = simccY[i].cast<num>();
      var maxX = 0, maxY = 0;
      var maxXVal = -double.infinity;
      var maxYVal = -double.infinity;
      for (var j = 0; j < xs.length; j++) {
        if (xs[j].toDouble() > maxXVal) {
          maxXVal = xs[j].toDouble();
          maxX = j;
        }
      }
      for (var j = 0; j < ys.length; j++) {
        if (ys[j].toDouble() > maxYVal) {
          maxYVal = ys[j].toDouble();
          maxY = j;
        }
      }
      // 把 simcc 网格坐标缩放回输入图像坐标, 再缩放回 bbox 区域
      final inputX = maxX / simccSplit; // 0~192
      final inputY = maxY / simccSplit; // 0~256
      // 映射回原图 (bbox 区域)
      final origX = x1 + inputX / _rtmposeInputW * boxW;
      final origY = y1 + inputY / _rtmposeInputH * boxH;

      // 关键点置信度: 取 simcc 最大值作为 score (粗略, 桌面版用的是单独 score 张量)
      final score = math.max(maxXVal, maxYVal);
      if (score < _confThreshold * 10) {
        // 桌面版会把低置信度点设为 (0,0)
        keypoints.add(Float64List.fromList([0, 0]));
      } else {
        keypoints.add(Float64List.fromList([origX, origY]));
      }
    }
    return keypoints;
  }
}
