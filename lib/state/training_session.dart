/// 训练会话状态 - ChangeNotifier 模式
///
/// 集中持有姿态检测器与计数器, 暴露给 UI 通过 Provider 监听:
/// - 当前选中运动类型 [ExerciseConfig]
/// - 实时关键点列表 (供 [SkeletonPainter] 绘制)
/// - 当前累计次数与角度
/// - 当前是否在运行推理 (避免重复处理同一帧)
///
/// UI 调用流程:
///   1. 启动时 [load] 加载配置 + 初始化检测器
///   2. 选择运动类型 [selectExercise]
///   3. 拿到 CameraImage 后调用 [processFrame]
///   4. 退出时 [dispose]
library;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import '../core/exercise_config.dart';
import '../core/exercise_counter.dart';
import '../core/pose_detector.dart';

/// 推理最小间隔 (毫秒)
///
/// 相机 30fps, 推理跟不上; 姿态计数 7fps 已足够流畅
/// 此值控制 [processFrame] 节流: 距上次推理 < 此值则跳过
const int _kMinInferenceIntervalMs = 130; // ~7-8 fps

class TrainingSession extends ChangeNotifier {
  final PoseDetector _detector = PoseDetector();
  final ExerciseCounter _counter = ExerciseCounter();

  PoseDetector get detector => _detector;

  ExerciseRepository get repository => ExerciseRepository.instance;

  bool _loading = false;
  bool get isLoading => _loading;

  ExerciseConfig? _current;
  ExerciseConfig? get currentExercise => _current;

  /// 最近一帧的关键点 (供 [SkeletonPainter] 绘制)
  List<Float64List> _keypoints = const [];
  List<Float64List> get keypoints => _keypoints;

  /// 最近一帧的输入图像尺寸 (供绘制器做坐标映射)
  img.Image? _lastFrame;
  img.Image? get lastFrame => _lastFrame;

  /// 当前角度 (度数, 可能为 null 表示无效)
  double? _angle;
  double? get angle => _angle;

  int get count => _counter.count;

  /// 是否正在处理一帧 (UI 据此跳过新帧, 避免堆积)
  bool _processing = false;
  bool get isProcessing => _processing;

  /// 上次推理开始时间 (用于节流)
  DateTime _lastInferenceTime = DateTime.fromMillisecondsSinceEpoch(0);

  /// 启动: 加载运动配置 + 初始化模型会话
  Future<void> load() async {
    if (_loading) return;
    _loading = true;
    notifyListeners();
    try {
      if (!repository.isLoaded) {
        await repository.load();
      }
      await _detector.init();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  /// 选择当前运动类型 (key 对应 exercises.json 中的 key)
  void selectExercise(String key) {
    final cfg = repository[key];
    if (cfg == null) {
      debugPrint('Unknown exercise key: $key');
      return;
    }
    _current = cfg;
    _counter.reset();
    _keypoints = const [];
    _angle = null;
    notifyListeners();
  }

  /// 处理一帧: 给定已转好的 RGB [img.Image], 推理并更新计数
  ///
  /// 节流策略:
  /// 1. 如果正在推理中, 直接跳过 (避免堆积)
  /// 2. 距上次推理 < [_kMinInferenceIntervalMs] 也跳过 (限制推理频率)
  ///
  /// 推理本身用 [PoseDetector.detect] 内部的 runAsync 推到后台 isolate,
  /// 主 isolate 不阻塞, UI 保持 60fps
  Future<void> processFrame(img.Image rgb) async {
    if (_processing || !_detector.isInitialized) return;

    // 时间节流
    final now = DateTime.now();
    final elapsed = now.difference(_lastInferenceTime).inMilliseconds;
    if (elapsed < _kMinInferenceIntervalMs) return;
    _lastInferenceTime = now;
    _processing = true;

    try {
      final result = await _detector.detect(rgb);
      _lastFrame = rgb;

      if (result == null) {
        _keypoints = const [];
        _angle = null;
      } else {
        _keypoints = result.keypoints;
        if (_current != null) {
          _angle = _counter.update(result.keypoints, _current!);
        }
      }
      notifyListeners();
    } catch (e, st) {
      debugPrint('processFrame error: $e\n$st');
    } finally {
      _processing = false;
    }
  }

  /// 重置计数 (保留当前选中的运动类型)
  void resetCount() {
    _counter.reset();
    _keypoints = const [];
    _angle = null;
    notifyListeners();
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }
}

