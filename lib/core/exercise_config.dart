/// 运动配置加载器 - 加载并解析 assets/data/exercises.json
///
/// 与桌面版 [data/exercises.json](file:///f:/git/Good-GYM/data/exercises.json) 完全复用,
/// 字段结构保持一致, 便于桌面版与移动版同步更新。
library;

import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

/// 单个运动类型的配置 (对应 JSON 中的一项)
class ExerciseConfig {
  final String key;            // 配置 key, 如 "squat"
  final String nameZh;
  final String nameEn;
  final double downAngle;      // "下"姿态角度阈值
  final double upAngle;        // "上"姿态角度阈值
  final KeypointIndices keypoints;
  final bool isLegExercise;    // 腿部运动需要左右腿独立计数
  final List<int> anglePoint;  // 用于在画面上绘制角度的三个关键点索引
  final String? angleDirection; // "increasing" / "decreasing", 缺省时按 up/down 自动推断

  const ExerciseConfig({
    required this.key,
    required this.nameZh,
    required this.nameEn,
    required this.downAngle,
    required this.upAngle,
    required this.keypoints,
    required this.isLegExercise,
    required this.anglePoint,
    this.angleDirection,
  });

  factory ExerciseConfig.fromJson(String key, Map<String, dynamic> json) {
    final kp = json['keypoints'] as Map<String, dynamic>;
    return ExerciseConfig(
      key: key,
      nameZh: json['name_zh'] as String? ?? key,
      nameEn: json['name_en'] as String? ?? key,
      downAngle: (json['down_angle'] as num).toDouble(),
      upAngle: (json['up_angle'] as num).toDouble(),
      keypoints: KeypointIndices(
        left: ((kp['left'] as List).cast<num>()).map((e) => e.toInt()).toList(),
        right: ((kp['right'] as List).cast<num>()).map((e) => e.toInt()).toList(),
      ),
      isLegExercise: json['is_leg_exercise'] as bool? ?? false,
      anglePoint: ((json['angle_point'] as List).cast<num>()).map((e) => e.toInt()).toList(),
      angleDirection: json['angle_direction'] as String?,
    );
  }

  /// 推断动作方向: "increasing" = 角度增大算一次, "decreasing" = 角度减小算一次
  /// 当配置未指定时, 按 up < down 推断为 decreasing, 反之 increasing
  String get inferDirection {
    if (angleDirection == 'increasing' || angleDirection == 'decreasing') {
      return angleDirection!;
    }
    return upAngle < downAngle ? 'decreasing' : 'increasing';
  }
}

/// 运动用到的关键点索引 (COCO 17 格式)
class KeypointIndices {
  final List<int> left;   // [pt1, pt2, pt3] 用于计算左侧角度
  final List<int> right;  // [pt1, pt2, pt3] 用于计算右侧角度
  const KeypointIndices({required this.left, required this.right});
}

/// 运动配置仓库 - 单例, 启动时加载一次
class ExerciseRepository {
  final Map<String, ExerciseConfig> _configs = {};

  ExerciseRepository._();

  static final ExerciseRepository instance = ExerciseRepository._();

  bool get isLoaded => _configs.isNotEmpty;

  List<ExerciseConfig> get all => _configs.values.toList(growable: false);

  ExerciseConfig? operator [](String key) => _configs[key];

  /// 从 Flutter assets 加载 exercises.json
  /// assetPath 对应 pubspec.yaml 中注册的路径
  Future<void> load({String assetPath = 'assets/data/exercises.json'}) async {
    final raw = await rootBundle.loadString(assetPath);
    final data = jsonDecode(raw) as Map<String, dynamic>;
    final exercises = data['exercises'] as Map<String, dynamic>;
    _configs.clear();
    exercises.forEach((key, value) {
      _configs[key] = ExerciseConfig.fromJson(key, value as Map<String, dynamic>);
    });
  }
}
