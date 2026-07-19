/// 首页 - 运动类型选择网格
///
/// 显示所有 [ExerciseRepository] 中加载的运动卡片, 点击进入相机训练页
library;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../state/training_session.dart';
import 'camera_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Good-GYM'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: '关于',
            onPressed: () => _showAbout(context),
          ),
        ],
      ),
      body: Consumer<TrainingSession>(
        builder: (context, session, _) {
          if (session.isLoading || !session.repository.isLoaded) {
            return const Center(child: CircularProgressIndicator());
          }
          final exercises = session.repository.all;
          if (exercises.isEmpty) {
            return const Center(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text(
                  '未加载到运动配置, 请检查 assets/data/exercises.json',
                  textAlign: TextAlign.center,
                ),
              ),
            );
          }
          return GridView.builder(
            padding: const EdgeInsets.all(16),
            gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
              maxCrossAxisExtent: 180,
              mainAxisSpacing: 12,
              crossAxisSpacing: 12,
              childAspectRatio: 1.0,
            ),
            itemCount: exercises.length,
            itemBuilder: (context, index) {
              final cfg = exercises[index];
              return _ExerciseCard(
                title: cfg.nameZh,
                subtitle: cfg.nameEn,
                onTap: () => _enter(context, session, cfg.key),
              );
            },
          );
        },
      ),
    );
  }

  Future<void> _enter(
    BuildContext context,
    TrainingSession session,
    String key,
  ) async {
    session.selectExercise(key);
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const CameraScreen()),
    );
  }

  void _showAbout(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Good-GYM 安卓版'),
        content: const Text(
          '基于桌面版 Good-GYM 移植\n'
          '模型: RTMPose-t + YOLOX nano (onnxruntime-mobile)\n'
          '17 关键点实时姿态检测 + 三点夹角计数',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('确定'),
          ),
        ],
      ),
    );
  }
}

/// 单个运动卡片
class _ExerciseCard extends StatelessWidget {
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _ExerciseCard({
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.fitness_center,
                  size: 36, color: theme.colorScheme.primary),
              const SizedBox(height: 8),
              Text(
                title,
                style: theme.textTheme.titleMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 2),
              Text(
                subtitle,
                style: theme.textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
