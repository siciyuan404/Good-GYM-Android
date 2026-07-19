/// Good-GYM 安卓版入口
///
/// 启动时创建 [TrainingSession] 并通过 Provider 注入,
/// 等待模型加载完成后进入首页
library;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/home_screen.dart';
import 'state/training_session.dart';

void main() {
  runApp(const GoodGymApp());
}

class GoodGymApp extends StatelessWidget {
  const GoodGymApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => TrainingSession()..load(),
      child: MaterialApp(
        title: 'Good-GYM',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF00E676),
            brightness: Brightness.light,
          ),
          useMaterial3: true,
        ),
        home: const _SplashGate(child: HomeScreen()),
      ),
    );
  }
}

/// 启动闸门: 在 [TrainingSession] 完成加载前显示 splash, 加载完成后切换到主页
class _SplashGate extends StatelessWidget {
  final Widget child;
  const _SplashGate({required this.child});

  @override
  Widget build(BuildContext context) {
    final session = context.watch<TrainingSession>();
    if (session.isLoading && !session.repository.isLoaded) {
      return _Splash(message: '加载运动配置...');
    }
    if (!session.detector.isInitialized) {
      return _Splash(message: '加载姿态检测模型...');
    }
    return child;
  }
}

class _Splash extends StatelessWidget {
  final String message;
  const _Splash({required this.message});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.surface,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.fitness_center, size: 80, color: Colors.green),
            const SizedBox(height: 24),
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text(message, style: Theme.of(context).textTheme.titleMedium),
          ],
        ),
      ),
    );
  }
}
