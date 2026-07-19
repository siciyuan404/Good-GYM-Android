package com.goodgym.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.goodgym.core.ExerciseConfig

/**
 * 相机训练页 - 预览 + 骨架 + 计数 HUD
 *
 * 进入流程:
 *   1. 检查相机权限, 未授权则弹请求
 *   2. 授权后启动 [CameraManager] 绑定相机
 *   3. 实时显示骨架 + 关节角度 + 计数
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    exerciseKey: String,
    config: ExerciseConfig?,
    viewModel: CameraViewModel,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    // 权限请求 launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // 选择当前运动 (重置计数器)
    LaunchedEffect(exerciseKey) {
        viewModel.selectExercise(exerciseKey)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = config?.name_zh ?: exerciseKey,
                        fontWeight = FontWeight.SemiBold
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "返回"
                        )
                    }
                },
                actions = {
                    // 切换前后置摄像头
                    TextButton(
                        onClick = { viewModel.switchCamera() }
                    ) {
                        Text(
                            text = "翻转",
                            color = MaterialTheme.colorScheme.onSurface
                        )
                    }
                    IconButton(onClick = { viewModel.resetCount() }) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "重置计数"
                        )
                    }
                }
            )
        }
    ) { inner ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(inner)
        ) {
            if (!hasCameraPermission) {
                PermissionRequestPanel(onRequest = {
                    permissionLauncher.launch(Manifest.permission.CAMERA)
                })
            } else {
                // 相机预览 + 骨架 + HUD
                CameraPreviewWithOverlay(viewModel = viewModel)
            }
        }
    }
}

@Composable
private fun CameraPreviewWithOverlay(viewModel: CameraViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val frameState by viewModel.frameState.collectAsState()

    // 创建 PreviewView
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    // 启动相机 (DisposableEffect 保证离开页面时解绑)
    DisposableEffect(Unit) {
        viewModel.startCamera(lifecycleOwner, previewView)
        onDispose {
            viewModel.stopCamera()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        // 1. 相机预览 (AndroidView 包装 CameraX PreviewView)
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // 2. 骨架叠加层 (前置摄像头需要镜像)
        SkeletonOverlay(
            keypoints = frameState.keypoints,
            imageW = frameState.frameW,
            imageH = frameState.frameH,
            rotationDegrees = frameState.rotationDegrees,
            mirror = !frameState.lensFacingBack
        )

        // 3. 计数 HUD (顶部 + 左下角调试信息)
        HudOverlay(
            count = frameState.count,
            angle = frameState.angle,
            fps = frameState.fps,
            inferenceMs = frameState.inferenceMs,
            hasKeypoints = frameState.keypoints != null
        )
    }
}

@Composable
private fun HudOverlay(
    count: Int,
    angle: Float?,
    fps: Float,
    inferenceMs: Long,
    hasKeypoints: Boolean
) {
    // 顶部中央: 大号计数
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 24.dp),
        contentAlignment = Alignment.TopCenter
    ) {
        Surface(
            color = Color.Black.copy(alpha = 0.55f),
            shape = RoundedCornerShape(24.dp)
        ) {
            Text(
                text = "$count",
                modifier = Modifier.padding(horizontal = 32.dp, vertical = 8.dp),
                color = Color.White,
                style = MaterialTheme.typography.displayLarge,
                fontWeight = FontWeight.Bold
            )
        }
    }

    // 底部调试信息 (左下角)
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Bottom,
        horizontalAlignment = Alignment.Start
    ) {
        val angleText = angle?.let { "角度: ${it.toInt()}°" } ?: "角度: --"
        val statusText = if (hasKeypoints) "已检测" else "等待人体"
        Text(
            text = angleText,
            color = Color.White,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            text = "状态: $statusText",
            color = Color.White.copy(alpha = 0.8f),
            style = MaterialTheme.typography.bodySmall
        )
        Text(
            text = "FPS: ${"%.1f".format(fps)}   推理: ${inferenceMs}ms",
            color = Color.White.copy(alpha = 0.6f),
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
private fun PermissionRequestPanel(onRequest: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "需要相机权限",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = "用于实时姿态检测, 数据仅在本地推理, 不会上传",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(vertical = 8.dp)
        )
        Surface(
            color = MaterialTheme.colorScheme.primary,
            shape = RoundedCornerShape(24.dp),
            onClick = onRequest,
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text(
                text = "授予相机权限",
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 12.dp),
                color = MaterialTheme.colorScheme.onPrimary
            )
        }
    }
}
