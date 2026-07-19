/// 相机训练屏幕 - 实时预览 + 骨架叠加 + 计数显示
///
/// 主要职责:
/// 1. 申请相机权限, 初始化 [CameraController]
/// 2. 启动 [CameraController.startImageStream] 取流
/// 3. 每帧 CameraImage → img.Image → [TrainingSession.processFrame]
/// 4. 用 [SkeletonPainter] 在预览上层绘制关键点
/// 5. 在屏幕顶部显示次数与角度
library;

import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';

import '../state/training_session.dart';
import '../util/yuv_to_rgb.dart';
import '../widgets/skeleton_painter.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _cameraReady = false;
  String? _errorText;
  bool _mirror = true; // 默认前置镜像, 后置时改为 false

  // 帧节流: 上一帧处理完成后才接收下一帧
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopImageStream();
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // 应用进入后台或被遮挡时暂停相机, 节省资源
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _stopImageStream();
    } else if (state == AppLifecycleState.resumed) {
      _startImageStream();
    }
  }

  Future<void> _initCamera() async {
    // 1. 权限检查
    final status = await Permission.camera.status;
    if (!status.isGranted) {
      final result = await Permission.camera.request();
      if (!result.isGranted) {
        setState(() => _errorText = '相机权限被拒绝');
        return;
      }
    }

    // 2. 查找相机, 默认前置 (user-facing)
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      setState(() => _errorText = '未找到可用相机');
      return;
    }
    final front = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );
    _mirror = front.lensDirection == CameraLensDirection.front;

    // 3. 初始化 CameraController
    // ResolutionPreset.low (480p, 640x480) - 推理性能优先
    // 480p 推理速度比 720p 快 ~2x, 关键点精度无明显差异
    _controller = CameraController(
      front,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      setState(() => _cameraReady = true);
      await _startImageStream();
    } on CameraException catch (e) {
      setState(() => _errorText = '相机初始化失败: ${e.description ?? e.code}');
    }
  }

  Future<void> _startImageStream() async {
    if (_controller == null ||
        !_controller!.value.isInitialized ||
        _controller!.value.isStreamingImages) {
      return;
    }
    await _controller!.startImageStream(_onImage);
  }

  Future<void> _stopImageStream() async {
    if (_controller?.value.isStreamingImages ?? false) {
      await _controller?.stopImageStream();
    }
  }

  /// CameraImage 回调: 限流后转发给 [TrainingSession.processFrame]
  void _onImage(CameraImage image) {
    if (_busy) return;
    final session = context.read<TrainingSession>();
    if (!session.detector.isInitialized) return;

    _busy = true;

    // 在 Dart 主 isolate 上做 YUV→RGB 与推理, 简单但会阻塞 UI 短暂时间
    // 后续可考虑用 compute() 把 detect 推到后台 isolate
    final rgb = yuv420ToImage(image);
    if (rgb == null) {
      _busy = false;
      return;
    }
    session.processFrame(rgb).whenComplete(() {
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final session = context.watch<TrainingSession>();
    final exercise = session.currentExercise;

    Widget body;
    if (_errorText != null) {
      body = _CenterMessage(
        icon: Icons.error_outline,
        text: _errorText!,
        action: TextButton(
          onPressed: () {
            setState(() => _errorText = null);
            _initCamera();
          },
          child: const Text('重试'),
        ),
      );
    } else if (!_cameraReady) {
      body = const Center(child: CircularProgressIndicator());
    } else {
      body = _buildCameraStack(session);
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(exercise?.nameZh ?? '训练'),
        actions: [
          IconButton(
            icon: const Icon(Icons.cameraswitch_outlined),
            tooltip: '切换前后置相机',
            onPressed: _switchCamera,
          ),
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: '重置计数',
            onPressed: () => session.resetCount(),
          ),
        ],
      ),
      body: body,
    );
  }

  Widget _buildCameraStack(TrainingSession session) {
    final controller = _controller!;
    // CameraPreview 的固有尺寸来自相机传感器 (previewSize)
    // 用 AspectRatio 自动按 sensor aspect ratio 缩放到屏幕, 避免变形
    // 注意: previewSize 是 (width, height) 但 CameraPreview 期望 aspectRatio = height/width
    // 因为 Android 相机传感器是横置的, 屏幕竖向时预览旋转 90°
    final previewSize = controller.value.previewSize ?? const Size(640, 480);
    final aspectRatio = previewSize.height / previewSize.width;

    return Stack(
      children: [
        // 相机预览层 - 用 AspectRatio 防变形
        Positioned.fill(
          child: RepaintBoundary(
            child: Center(
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: CameraPreview(controller),
              ),
            ),
          ),
        ),
        // 骨架叠加 - 独立重绘边界, 关键点变化才重绘
        Positioned.fill(
          child: RepaintBoundary(
            child: IgnorePointer(
              child: CustomPaint(
                painter: SkeletonPainter(
                  keypoints: session.keypoints,
                  imageSize: Size(
                    session.lastFrame?.width.toDouble() ??
                        previewSize.width,
                    session.lastFrame?.height.toDouble() ??
                        previewSize.height,
                  ),
                  renderSize: MediaQuery.of(context).size,
                  mirror: _mirror,
                ),
              ),
            ),
          ),
        ),
        // 顶部信息条: 次数 + 角度
        Positioned(
          top: 0,
          left: 0,
          right: 0,
          child: _StatsBar(session: session),
        ),
        // 调试 HUD: 显示推理状态 (后续可移除)
        Positioned(
          bottom: 0,
          left: 0,
          right: 0,
          child: _DebugHud(session: session),
        ),
      ],
    );
  }

  Future<void> _switchCamera() async {
    if (_controller == null) return;
    final currentDir = _controller!.description.lensDirection;
    final cameras = await availableCameras();
    final next = cameras.firstWhere(
      (c) => c.lensDirection != currentDir,
      orElse: () => _controller!.description,
    );
    if (next == _controller!.description) return;

    await _stopImageStream();
    await _controller!.dispose();
    _mirror = next.lensDirection == CameraLensDirection.front;
    _controller = CameraController(
      next,
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    try {
      await _controller!.initialize();
      await _startImageStream();
      setState(() {});
    } on CameraException catch (e) {
      setState(() => _errorText = '切换相机失败: ${e.description ?? e.code}');
    }
  }
}

/// 顶部信息条: 大号次数 + 当前角度 + 运动名
class _StatsBar extends StatelessWidget {
  final TrainingSession session;
  const _StatsBar({required this.session});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final exercise = session.currentExercise;
    final angle = session.angle;
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Colors.black54, Colors.transparent],
        ),
      ),
      padding: const EdgeInsets.only(top: 12, left: 16, right: 16, bottom: 16),
      child: SafeArea(
        bottom: false,
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    exercise?.nameZh ?? '',
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: Colors.white,
                    ),
                  ),
                  if (angle != null)
                    Text(
                      '角度: ${angle.toStringAsFixed(1)}°',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: Colors.white70,
                      ),
                    ),
                ],
              ),
            ),
            Text(
              '${session.count}',
              style: theme.textTheme.displayLarge?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// 居中错误/提示 widget
class _CenterMessage extends StatelessWidget {
  final IconData icon;
  final String text;
  final Widget? action;

  const _CenterMessage({
    required this.icon,
    required this.text,
    this.action,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 64, color: Colors.grey),
            const SizedBox(height: 16),
            Text(
              text,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            if (action != null) ...[const SizedBox(height: 16), action!],
          ],
        ),
      ),
    );
  }
}

/// 调试 HUD - 显示推理状态帮助定位问题
/// 显示: 帧尺寸 / 关键点数量 / 检测状态
class _DebugHud extends StatelessWidget {
  final TrainingSession session;
  const _DebugHud({required this.session});

  @override
  Widget build(BuildContext context) {
    final frame = session.lastFrame;
    final kpCount = session.keypoints.length;
    final status = session.isProcessing
        ? '推理中'
        : (kpCount > 0 ? '已检测' : '等待');
    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [Colors.black54, Colors.transparent],
        ),
      ),
      padding: const EdgeInsets.only(
          top: 16, left: 16, right: 16, bottom: 16),
      child: SafeArea(
        top: false,
        child: Text(
          'frame: ${frame?.width ?? 0}x${frame?.height ?? 0}  '
          'kp: $kpCount  '
          'status: $status',
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 12,
            fontFamily: 'monospace',
          ),
        ),
      ),
    );
  }
}
