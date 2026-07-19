package com.goodgym.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * 应用顶层 Composable - 简单路由: Home ↔ Camera
 *
 * 用 [remember] 维护当前路由状态, 没有引入 Navigation Compose (省依赖, 项目很小)
 */
@Composable
fun GoodGymApp() {
    val viewModel: CameraViewModel = viewModel()
    val appState by viewModel.appState.collectAsState()

    // 当前路由: null = 主页, 非 null = 对应运动 key
    var route by remember { mutableStateOf<String?>(null) }

    when {
        // 错误提示
        appState.modelLoadError != null -> {
            ErrorScreen(appState.modelLoadError!!)
        }
        // 模型 / 配置未加载完
        !appState.isModelsLoaded || !appState.isRepositoryLoaded -> {
            LoadingScreen()
        }
        // 进入相机页
        route != null -> {
            val key = route!!
            CameraScreen(
                exerciseKey = key,
                config = viewModel.getExerciseConfig(key),
                viewModel = viewModel,
                onBack = { route = null }
            )
        }
        // 主页: 运动列表
        else -> {
            HomeScreen(
                exerciseKeys = viewModel.exerciseKeys.toList(),
                getConfig = { viewModel.getExerciseConfig(it) },
                onSelect = { key -> route = key }
            )
        }
    }
}

@Composable
private fun LoadingScreen() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            CircularProgressIndicator()
            Text(
                text = "加载模型中...",
                modifier = Modifier.padding(top = 16.dp),
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
private fun ErrorScreen(message: String) {
    Box(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = message,
            color = MaterialTheme.colorScheme.error
        )
    }
}
