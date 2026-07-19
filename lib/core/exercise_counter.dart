/// 运动计数核心 - 翻译自桌面版 [exercise_counters.py](file:///f:/git/Good-GYM/exercise_counters.py)
///
/// 逻辑保持与桌面版一致:
/// - 三点夹角 (COCO 17 关键点)
/// - 滑动窗口平滑 (中位数 + 2σ 离群点过滤)
/// - 状态机计数 (up/down 阶段切换)
/// - 腿部动作左右腿独立计数
/// - 最小 rep 间隔 (0.5s) 防误触
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'exercise_config.dart';

/// 关键点坐标, 形如 [[x0, y0], [x1, y1], ...] 共 17 个
typedef Keypoints = List<Float64List>;

/// 计数阶段
enum Stage { up, down, none }

class ExerciseCounter {
  /// 当前累计次数
  int count = 0;

  /// 当前阶段 (用于状态机判断)
  Stage _stage = Stage.none;

  /// 腿部动作的左右腿独立阶段
  Stage _leftLegStage = Stage.none;
  Stage _rightLegStage = Stage.none;

  /// 平滑窗口大小, 与桌面版默认 5 一致
  final int smoothingWindow;

  /// 角度历史 (滑动窗口)
  final List<double> _angleHistory = [];

  /// 上次计数时间戳 (毫秒)
  int _lastCountTime = 0;

  /// 最小 rep 间隔 (毫秒), 桌面版为 0.5s
  static const int _minRepIntervalMs = 500;

  ExerciseCounter({this.smoothingWindow = 5});

  /// 重置计数状态
  void reset() {
    count = 0;
    _stage = Stage.none;
    _leftLegStage = Stage.none;
    _rightLegStage = Stage.none;
    _angleHistory.clear();
    _lastCountTime = 0;
  }

  /// 三点夹角 (b 为顶点), 单位度数
  /// 返回 null 表示点无效 (含 NaN 或 (0,0))
  static double? calculateAngle(Float64List a, Float64List b, Float64List c) {
    // 检查点有效性 (与桌面版一致: 任意点为 (0,0) 视为无效)
    if ((a[0] == 0 && a[1] == 0) ||
        (b[0] == 0 && b[1] == 0) ||
        (c[0] == 0 && c[1] == 0)) {
      return null;
    }

    final baX = a[0] - b[0];
    final baY = a[1] - b[1];
    final bcX = c[0] - b[0];
    final bcY = c[1] - b[1];

    final baNorm = math.sqrt(baX * baX + baY * baY);
    final bcNorm = math.sqrt(bcX * bcX + bcY * bcY);
    if (baNorm == 0 || bcNorm == 0) {
      return null;
    }

    var cosine = (baX * bcX + baY * bcY) / (baNorm * bcNorm);
    // 裁剪到 [-1, 1] 防止数值误差导致 acos 返回 NaN
    cosine = cosine.clamp(-1.0, 1.0);
    return math.acos(cosine) * 180.0 / math.pi;
  }

  /// 对角度做滑动窗口平滑: 中位数过滤 + 2σ 离群点去除后取均值
  /// 与桌面版 smooth_angle 完全一致
  double? smoothAngle(double? angle) {
    if (angle == null) return null;

    _angleHistory.add(angle);
    if (_angleHistory.length > smoothingWindow) {
      _angleHistory.removeAt(0);
    }
    if (_angleHistory.length < 3) return angle;

    final sorted = List<double>.of(_angleHistory)..sort();
    final median = _medianOfSorted(sorted);

    final std = _stdDev(_angleHistory);
    final filtered = _angleHistory.where((v) => (v - median).abs() <= 2 * std).toList();
    return filtered.isEmpty ? angle : filtered.reduce((a, b) => a + b) / filtered.length;
  }

  /// 通用计数入口: 给定 17 个关键点与运动类型, 返回当前角度 (供 UI 显示)
  /// 内部维护 count 状态
  ///
  /// (方法名与字段 `count` 冲突, 故改用 `update` - 与 ChangeNotifier 惯例一致)
  double? update(Keypoints keypoints, ExerciseConfig config) {
    final kp = config.keypoints;

    final leftAngle = calculateAngle(
      keypoints[kp.left[0]],
      keypoints[kp.left[1]],
      keypoints[kp.left[2]],
    );
    final rightAngle = calculateAngle(
      keypoints[kp.right[0]],
      keypoints[kp.right[1]],
      keypoints[kp.right[2]],
    );

    if (leftAngle == null || rightAngle == null) return null;

    // 腿部动作走独立逻辑
    if (config.isLegExercise) {
      return _countLegExercise(leftAngle, rightAngle, config);
    }

    final avg = (leftAngle + rightAngle) / 2;
    final smoothed = smoothAngle(avg);
    if (smoothed == null) return null;

    final direction = config.inferDirection;
    final upThreshold = config.upAngle;
    final downThreshold = config.downAngle;

    if (direction == 'decreasing') {
      // 角度减小算一次 (如深蹲、俯卧撑: 起身大角度, 下蹲/下压小角度)
      if (smoothed > downThreshold) {
        _stage = Stage.down;
      } else if (smoothed < upThreshold &&
          _stage == Stage.down &&
          _checkRepTiming()) {
        _stage = Stage.up;
        count++;
        _lastCountTime = _nowMs();
      }
    } else {
      // 角度增大算一次 (如侧平举: 起始小角度, 抬起大角度)
      if (smoothed > upThreshold) {
        _stage = Stage.up;
      } else if (smoothed < downThreshold &&
          _stage == Stage.up &&
          _checkRepTiming()) {
        _stage = Stage.down;
        count++;
        _lastCountTime = _nowMs();
      }
    }
    return smoothed;
  }

  /// 腿部动作: 左右腿独立计数, 任意一条腿完成一次 up-down 循环即 +1
  double _countLegExercise(double leftAngle, double rightAngle, ExerciseConfig config) {
    final direction = config.inferDirection;
    final upThreshold = config.upAngle;
    final downThreshold = config.downAngle;

    if (_checkRepTiming()) {
      // 左腿
      _trackLeg(leftAngle, direction, upThreshold, downThreshold, (s) => _leftLegStage = s, () => _leftLegStage);
      // 右腿
      _trackLeg(rightAngle, direction, upThreshold, downThreshold, (s) => _rightLegStage = s, () => _rightLegStage);
    }
    return (leftAngle + rightAngle) / 2;
  }

  /// 单条腿的阶段跟踪与计数 (辅助 _countLegExercise)
  void _trackLeg(
    double angle,
    String direction,
    double upThreshold,
    double downThreshold,
    void Function(Stage) setStage,
    Stage Function() getStage,
  ) {
    bool ready;
    bool counted;
    Stage readyStage;
    Stage countedStage;

    if (direction == 'decreasing') {
      ready = angle > downThreshold;
      counted = angle < upThreshold && getStage() == Stage.down;
      readyStage = Stage.down;
      countedStage = Stage.up;
    } else {
      ready = angle > upThreshold;
      counted = angle < downThreshold && getStage() == Stage.up;
      readyStage = Stage.up;
      countedStage = Stage.down;
    }

    if (ready) {
      setStage(readyStage);
    } else if (counted) {
      count++;
      _lastCountTime = _nowMs();
      setStage(countedStage);
    }
  }

  bool _checkRepTiming() {
    final now = _nowMs();
    return now - _lastCountTime >= _minRepIntervalMs;
  }

  static int _nowMs() => DateTime.now().millisecondsSinceEpoch;

  static double _medianOfSorted(List<double> sorted) {
    final n = sorted.length;
    final mid = n ~/ 2;
    return n.isEven ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  static double _stdDev(List<double> xs) {
    if (xs.isEmpty) return 0;
    final mean = xs.reduce((a, b) => a + b) / xs.length;
    final variance = xs.map((x) => (x - mean) * (x - mean)).reduce((a, b) => a + b) / xs.length;
    return math.sqrt(variance);
  }
}
