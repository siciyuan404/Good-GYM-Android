package com.goodgym.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import androidx.camera.view.PreviewView
import com.goodgym.camera.CameraManager
import com.goodgym.core.ExerciseConfig
import com.goodgym.core.ExerciseCounter
import com.goodgym.core.ExerciseRepository
import com.goodgym.core.PoseDetector
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * 顶层 UI 状态 - 当前选中的运动 + 是否在训练中 + 仓库是否就绪
 */
data class AppState(
    val currentExerciseKey: String? = null,
    val currentExercise: ExerciseConfig? = null,
    val isRepositoryLoaded: Boolean = false,
    val isModelsLoaded: Boolean = false,
    val modelLoadError: String? = null
)

/**
 * 全局 ViewModel - 持有 PoseDetector + CameraManager + ExerciseCounter + ExerciseRepository
 *
 * 单例: [CameraManager] 与 [PoseDetector] 跨 Activity 重建保持存活, 避免重复加载模型
 */
class CameraViewModel(app: Application) : AndroidViewModel(app) {
    private val poseDetector = PoseDetector(app)
    private val counter = ExerciseCounter()
    private val repository = ExerciseRepository(app)

    val cameraManager: CameraManager = CameraManager(app, poseDetector, counter)

    private val _appState = MutableStateFlow(AppState())
    val appState: StateFlow<AppState> = _appState.asStateFlow()

    val frameState: StateFlow<com.goodgym.camera.FrameState> = cameraManager.state

    val exerciseKeys: Set<String> get() = repository.keys

    init {
        // 1. 加载运动配置 (assets/data/exercises.json)
        viewModelScope.launch {
            try {
                repository.load()
                _appState.value = _appState.value.copy(isRepositoryLoaded = true)
            } catch (e: Exception) {
                _appState.value = _appState.value.copy(
                    isRepositoryLoaded = false,
                    modelLoadError = "Failed to load exercises.json: ${e.message}"
                )
            }
        }

        // 2. 加载 ONNX 模型 (assets/models/*.onnx)
        viewModelScope.launch {
            try {
                poseDetector.init()
                _appState.value = _appState.value.copy(isModelsLoaded = true)
            } catch (e: Exception) {
                _appState.value = _appState.value.copy(
                    isModelsLoaded = false,
                    modelLoadError = "Failed to load ONNX models: ${e.message}"
                )
            }
        }
    }

    /**
     * 选择运动项目 (从 HomeScreen 跳到 CameraScreen 之前调用)
     */
    fun selectExercise(key: String) {
        val config = repository[key] ?: return
        cameraManager.switchExercise(config)
        _appState.value = _appState.value.copy(
            currentExerciseKey = key,
            currentExercise = config
        )
    }

    /**
     * 启动相机 (供 CameraScreen 通过 LaunchedEffect 调用)
     */
    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        viewModelScope.launch {
            cameraManager.start(lifecycleOwner, previewView)
        }
    }

    /**
     * 停止相机 (供 CameraScreen 在 dispose 时调用)
     */
    fun stopCamera() {
        cameraManager.stop()
    }

    /**
     * 重置当前运动计数
     */
    fun resetCount() {
        cameraManager.resetCount()
    }

    fun getExerciseConfig(key: String): ExerciseConfig? = repository[key]

    override fun onCleared() {
        super.onCleared()
        cameraManager.release()
    }
}
