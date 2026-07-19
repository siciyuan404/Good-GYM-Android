package com.goodgym.camera

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import com.goodgym.core.ExerciseConfig
import com.goodgym.core.ExerciseCounter
import com.goodgym.core.PoseDetector
import com.goodgym.core.PoseResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume

/**
 * 相机帧数据 + 推理结果 + 计数状态 (一次性 immutable snapshot)
 *
 * [keypoints] 长度 34 的 FloatArray (17 点 × 2), 已 [copyOf] 防止下一帧覆盖
 */
data class FrameState(
    val keypoints: FloatArray? = null,
    val detectionScore: Float = 0f,
    val angle: Float? = null,
    val count: Int = 0,
    val frameW: Int = 0,
    val frameH: Int = 0,
    val rotationDegrees: Int = 0,
    val fps: Float = 0f,
    val inferenceMs: Long = 0L,
    val lensFacingBack: Boolean = true
)

/**
 * CameraX 封装 - 串联: 相机帧 → YUV→RGB → YOLOX+RTMPose 推理 → 计数
 *
 * 设计要点:
 *   - 单后台线程推理 ([inferenceExecutor]), 不阻塞主线程
 *   - [ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST] 自动 backpressure: 上一帧没处理完时新帧直接丢弃
 *   - [Mutex] 二次保险: 即使 executor 重入也只允许一帧在推理
 *   - 状态通过 [state] StateFlow 推送到 UI (Compose 自动 recomposition)
 *
 * 生命周期:
 *   [start] 绑定相机 use case, [stop] 解绑, [release] 关闭推理资源
 */
class CameraManager(
    private val context: Context,
    private val poseDetector: PoseDetector,
    private val counter: ExerciseCounter
) {
    companion object {
        private const val TAG = "CameraManager"
        private const val TARGET_RESOLUTION_W = 640
        private const val TARGET_RESOLUTION_H = 480
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var boundPreviewView: PreviewView? = null
    private var boundLifecycleOwner: LifecycleOwner? = null

    // 当前镜头方向 (true=后置, false=前置)
    @Volatile
    var lensFacingBack: Boolean = true
        private set

    // 单线程后台执行器: 保证推理串行, 不抢 UI 资源
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val inferenceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val yuvConverter = YuvToRgbConverter()
    private val inferenceLock = Mutex()
    private val isInferring = AtomicBoolean(false)

    private val _state = MutableStateFlow(FrameState())
    val state: StateFlow<FrameState> = _state.asStateFlow()

    private var lastFrameTimeMs = 0L
    private var fpsSmoothed = 0f

    // 当前运动配置 (由 ViewModel 注入)
    @Volatile
    var currentConfig: ExerciseConfig? = null

    /**
     * 绑定相机 use case 到 [lifecycleOwner], 预览渲染到 [previewView]
     *
     * 必须在相机权限已授予后调用
     */
    @SuppressLint("RestrictedApi")
    suspend fun start(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val provider = ProcessCameraProvider.getInstance(context).await()
        cameraProvider = provider
        boundPreviewView = previewView
        boundLifecycleOwner = lifecycleOwner
        bindCamera(lensFacingBack)
    }

    /**
     * 切换前后置摄像头 (运行时切换, 不需要重新 start)
     */
    suspend fun switchCamera() {
        lensFacingBack = !lensFacingBack
        val provider = cameraProvider ?: return
        val pv = boundPreviewView ?: return
        val owner = boundLifecycleOwner ?: return
        try {
            provider.unbindAll()
            bindCamera(lensFacingBack)
            Log.i(TAG, "Camera switched: ${if (lensFacingBack) "back" else "front"}")
        } catch (e: Exception) {
            Log.e(TAG, "switchCamera failed", e)
        }
    }

    /**
     * 实际绑定 use case 到指定方向摄像头
     */
    @SuppressLint("RestrictedApi")
    private suspend fun bindCamera(back: Boolean) {
        val provider = cameraProvider ?: return
        val pv = boundPreviewView ?: return
        val owner = boundLifecycleOwner ?: return

        // 1. Preview use case - 渲染到 PreviewView
        preview = Preview.Builder()
            .build()
            .also { it.setSurfaceProvider(pv.surfaceProvider) }

        // 2. ImageAnalysis use case - ML 推理
        // STRATEGY_KEEP_ONLY_LATEST: 上一帧没处理完时, 新帧直接 drop (CameraX 自带 backpressure)
        imageAnalysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetResolution(
                android.util.Size(TARGET_RESOLUTION_W, TARGET_RESOLUTION_H)
            )
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(inferenceExecutor) { imageProxy ->
                    processFrame(imageProxy)
                }
            }

        // 3. 绑定到生命周期
        val selector = if (back) CameraSelector.DEFAULT_BACK_CAMERA
                       else CameraSelector.DEFAULT_FRONT_CAMERA
        try {
            provider.unbindAll()
            provider.bindToLifecycle(
                owner,
                selector,
                preview,
                imageAnalysis
            )
            Log.i(TAG, "Camera bound (${if (back) "back" else "front"}): ${TARGET_RESOLUTION_W}x${TARGET_RESOLUTION_H}")
        } catch (e: Exception) {
            Log.e(TAG, "bindToLifecycle failed", e)
        }
    }

    /**
     * 停止相机, 解绑所有 use case
     */
    fun stop() {
        cameraProvider?.unbindAll()
    }

    /**
     * 释放推理资源 (PoseDetector 的 ONNX session)
     */
    fun release() {
        stop()
        inferenceScope.cancel()
        inferenceExecutor.shutdown()
        poseDetector.close()
    }

    /**
     * 切换运动项目 - 重置计数器
     */
    fun switchExercise(config: ExerciseConfig) {
        counter.reset()
        currentConfig = config
        _state.value = _state.value.copy(count = 0, angle = null)
    }

    /**
     * 重置计数 (不切换 config)
     */
    fun resetCount() {
        counter.reset()
        _state.value = _state.value.copy(count = 0, angle = null)
    }

    /**
     * 处理一帧: YUV→RGB → YOLOX+RTMPose → 计数器
     *
     * CameraX 保证: 调用线程 = [inferenceExecutor] (单线程)
     * 内部仍加 [Mutex] 防御叠加, 加 [AtomicBoolean] 做快路径跳帧
     */
    private fun processFrame(imageProxy: ImageProxy) {
        val t0 = System.currentTimeMillis()

        // FPS 平滑 (指数加权)
        val now = t0
        if (lastFrameTimeMs > 0) {
            val dt = now - lastFrameTimeMs
            if (dt in 1..999) {
                val inst = 1000f / dt
                fpsSmoothed = if (fpsSmoothed == 0f) inst else fpsSmoothed * 0.85f + inst * 0.15f
            }
        }
        lastFrameTimeMs = now

        val w = imageProxy.width
        val h = imageProxy.height

        // 拿到 YUV 三 plane, 复制到自己的 ByteArray (ImageProxy buffer 会被复用)
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        val yBytes = ByteArray(yPlane.buffer.remaining()).also { yPlane.buffer.get(it) }
        val uBytes = ByteArray(uPlane.buffer.remaining()).also { uPlane.buffer.get(it) }
        val vBytes = ByteArray(vPlane.buffer.remaining()).also { vPlane.buffer.get(it) }

        // 拿到旋转信息 (后置摄像头 sensor 默认横屏, UI 是 portrait 需要旋转 90°)
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        imageProxy.close()

        // 快路径跳帧: 上一帧还在推理就丢弃当前帧
        if (!isInferring.compareAndSet(false, true)) {
            return
        }

        inferenceScope.launch {
            try {
                inferenceLock.withLock {
                    runInference(
                        yBytes, uBytes, vBytes,
                        yRowStride, uRowStride, uPixelStride,
                        vRowStride, vPixelStride,
                        w, h, rotationDegrees, t0
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "inference pipeline error", e)
            } finally {
                isInferring.set(false)
            }
        }
    }

    private suspend fun runInference(
        yBytes: ByteArray, uBytes: ByteArray, vBytes: ByteArray,
        yRowStride: Int, uRowStride: Int, uPixelStride: Int,
        vRowStride: Int, vPixelStride: Int,
        w: Int, h: Int, rotationDegrees: Int, frameStartMs: Long
    ) {
        // 1. YUV → RGB (CPU 整数算术, 480p ~2ms)
        val rgb = try {
            yuvConverter.yuv420ToRgb(
                yBytes, uBytes, vBytes,
                yRowStride = yRowStride,
                uvRowStride = uRowStride,
                uvPixelStride = uPixelStride,
                w = w, h = h
            )
        } catch (e: Exception) {
            Log.w(TAG, "YUV→RGB failed: ${e.message}")
            return
        }

        // 2. YOLOX + RTMPose 推理 (NNAPI EP 加速)
        val result: PoseResult? = try {
            poseDetector.detect(rgb, w, h)
        } catch (e: Exception) {
            Log.e(TAG, "inference error", e)
            null
        }

        val inferenceMs = System.currentTimeMillis() - frameStartMs

        // 3. 推送状态 (keypoints 拷贝, 防止下一帧覆盖)
        if (result == null) {
            _state.value = _state.value.copy(
                keypoints = null,
                detectionScore = 0f,
                angle = null,
                frameW = w,
                frameH = h,
                rotationDegrees = rotationDegrees,
                fps = fpsSmoothed,
                inferenceMs = inferenceMs,
                lensFacingBack = lensFacingBack
            )
            return
        }

        // 4. 计数器更新
        val currentConfig = currentConfig
        var angle: Float? = null
        if (currentConfig != null) {
            angle = counter.update(result.keypoints, currentConfig)
        }

        _state.value = FrameState(
            keypoints = result.keypoints.copyOf(),
            detectionScore = result.detection?.score ?: 0f,
            angle = angle,
            count = counter.count,
            frameW = w,
            frameH = h,
            rotationDegrees = rotationDegrees,
            fps = fpsSmoothed,
            inferenceMs = inferenceMs,
            lensFacingBack = lensFacingBack
        )
    }
}

/**
 * 等待 [ProcessCameraProvider] 异步就绪 (替代 Task.addOnSuccessListener)
 *
 * 用 [suspendCancellableCoroutine] 桥接 ListenableFuture, 不需要 guava-ktx 依赖
 */
private suspend fun <T> com.google.common.util.concurrent.ListenableFuture<T>.await(): T =
    suspendCancellableCoroutine { cont ->
        addListener({
            try {
                cont.resume(get())
            } catch (e: Exception) {
                cont.cancel(e)
            }
        }, Runnable::run)
    }
