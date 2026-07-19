/// COCO 17 关键点骨架绘制器
///
/// 在相机预览层上叠加绘制 17 个关键点 + 16 条骨骼连线
/// 坐标系: 接收原图坐标, 内部按 [renderSize] 与 [imageSize] 做缩放映射
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

/// COCO 17 关键点标准连线表
const List<List<int>> cocoSkeletonPairs = [
  [0, 1], // nose - left_eye
  [0, 2], // nose - right_eye
  [1, 3], // left_eye - left_ear
  [2, 4], // right_eye - right_ear
  [5, 6], // left_shoulder - right_shoulder
  [5, 11], // left_shoulder - left_hip
  [6, 12], // right_shoulder - right_hip
  [11, 12], // left_hip - right_hip
  [5, 7], // left_shoulder - left_elbow
  [7, 9], // left_elbow - left_wrist
  [6, 8], // right_shoulder - right_elbow
  [8, 10], // right_elbow - right_wrist
  [11, 13], // left_hip - left_knee
  [13, 15], // left_knee - left_ankle
  [12, 14], // right_hip - right_knee
  [14, 16], // right_knee - right_ankle
];

/// 关键点颜色 (与桌面版配色一致: 关键点用亮黄, 骨骼用亮青)
const Color _kPointColor = Color(0xFFFFEB3B);
const Color _kLineColor = Color(0xFF00E5FF);

/// 骨架绘制器
///
/// [keypoints] 原始图像坐标系下的 17 个点 (每点 [x, y] 或 [0, 0] 表示无效)
/// [imageSize] 输入图像的尺寸 (来自 [PoseDetector.detect])
/// [renderSize] 实际绘制区域 (相机预览的 widget 尺寸)
/// [mirror] 是否水平镜像 (前置相机镜像, 后置不镜像)
class SkeletonPainter extends CustomPainter {
  final List<Float64List> keypoints;
  final Size imageSize;
  final Size renderSize;
  final bool mirror;

  const SkeletonPainter({
    required this.keypoints,
    required this.imageSize,
    required this.renderSize,
    this.mirror = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (keypoints.isEmpty || imageSize.isEmpty) return;

    // 计算从图像坐标到绘制坐标的等比缩放 (contain 模式)
    final scale = math.min(
      renderSize.width / imageSize.width,
      renderSize.height / imageSize.height,
    );
    final offsetX = (renderSize.width - imageSize.width * scale) / 2;
    final offsetY = (renderSize.height - imageSize.height * scale) / 2;

    Offset transform(double x, double y) {
      var tx = offsetX + x * scale;
      if (mirror) tx = renderSize.width - tx;
      return Offset(tx, offsetY + y * scale);
    }

    // 1. 绘制骨骼连线
    final linePaint = Paint()
      ..color = _kLineColor
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    for (final pair in cocoSkeletonPairs) {
      if (pair[0] >= keypoints.length || pair[1] >= keypoints.length) continue;
      final a = keypoints[pair[0]];
      final b = keypoints[pair[1]];
      // 跳过无效点 (0, 0)
      if ((a[0] == 0 && a[1] == 0) || (b[0] == 0 && b[1] == 0)) continue;
      canvas.drawLine(
        transform(a[0], a[1]),
        transform(b[0], b[1]),
        linePaint,
      );
    }

    // 2. 绘制关键点
    final pointPaint = Paint()..color = _kPointColor;
    const pointRadius = 4.0;

    for (final kp in keypoints) {
      if (kp[0] == 0 && kp[1] == 0) continue;
      final pos = transform(kp[0], kp[1]);
      canvas.drawCircle(pos, pointRadius, pointPaint);
    }
  }

  @override
  bool shouldRepaint(covariant SkeletonPainter old) {
    // 关键点变化时才重绘; 尺寸变化也要重绘
    if (old.imageSize != imageSize ||
        old.renderSize != renderSize ||
        old.mirror != mirror) {
      return true;
    }
    if (old.keypoints.length != keypoints.length) return true;
    for (var i = 0; i < keypoints.length; i++) {
      if (old.keypoints[i][0] != keypoints[i][0] ||
          old.keypoints[i][1] != keypoints[i][1]) {
        return true;
      }
    }
    return false;
  }
}
