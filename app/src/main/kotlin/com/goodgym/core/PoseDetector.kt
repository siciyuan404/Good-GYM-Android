package com.goodgym.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.max
import kotlin.math.min

/**
 * 单个 YOLOX 检测框 (原图坐标系, 已反 letterbox 缩放)
 */
data class Detection(
    val x1: Float, val y1: Float,
    val x2: Float, val y2: Float,
    val score: Float
)

/**
 * 姿态检测结果: 17 关键点 + 人体检测框
 *
 * [keypoints] 是 FloatArray 长度 34 (17 个点 × 2 坐标), 顺序为 (x0, y0, x1, y1, ...)
 * 低置信度的点已置为 (0, 0)
 */
data class PoseResult(
    val keypoints: FloatArray,    // 长度 34
    val detection: Detection?
)

/**
 * YOLOX nano + RTMPose-t 双模型推理封装
 *
 * 与桌面版 [core/rtmpose_processor.py] 的推理流程对齐:
 *   1. YOLOX 检测人体 bbox (416x416 输入, /255 归一化)
 *   2. 取最高置信度 bbox, crop + resize 到 256x192
 *   3. RTMPose 推理得到 simcc_x [1,17,384] + simcc_y [1,17,512]
 *   4. argmax 得到 17 个关键点, 除以 simccSplit=2.0 缩放回输入坐标
 *   5. 映射回原图坐标系
 *
 * 性能要点:
 *   - 使用 ONNX Runtime Android 官方 AAR, 推理在调用线程 (主线程或调度线程)
 *   - 启用 NNAPI EP 加速 (Android 8.1+)
 *   - FloatBuffer 直接喂给 OnnxTensor, 避免重复拷贝
 */
class PoseDetector(private val context: Context) {
    companion object {
        private const val TAG = "PoseDetector"
        private const val YOLOX_INPUT_SIZE = 416
        private const val RTMPOSE_INPUT_W = 192
        private const val RTMPOSE_INPUT_H = 256
        private const val SIMCC_SPLIT = 2.0f
        private const val CONF_THRESHOLD = 0.5f
        private const val YOLOX_SCORE_THRESHOLD = 0.1f  // 降到 0.1 方便调试

        // RTMPose 标准预处理常数 (与 rtmlib 内部一致, RGB 顺序)
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var yoloxSession: OrtSession? = null
    private var rtmposeSession: OrtSession? = null

    /**
     * 最近一次推理的调试信息 (供 UI HUD 显示)
     * 包含 YOLOX 原始输出类型、检测数、最佳分数等
     */
    @Volatile
    var lastDebugInfo: String = ""
        private set

    /**
     * 从 assets 加载两个 .onnx 模型并创建会话
     * 启用 NNAPI EP 加速 (Android 8.1+)
     */
    suspend fun init(): Unit = withContext(Dispatchers.IO) {
        val opts = OrtSession.SessionOptions().apply {
            // CPU 线程数: 移动端 4 核已足够
            setInterOpNumThreads(1)
            setIntraOpNumThreads(4)
            // 全图优化
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // NNAPI EP (Android Neural Networks API, 利用 NPU 加速)
            try {
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI unavailable, fallback to CPU: ${e.message}")
            }
        }

        yoloxSession = env.createSession(
            readAssetBytes("models/yolox_nano.onnx"),
            opts
        )
        rtmposeSession = env.createSession(
            readAssetBytes("models/rtmpose_t.onnx"),
            opts
        )
        Log.i(TAG, "Models loaded: yolox=${yoloxSession != null}, rtmpose=${rtmposeSession != null}")
    }

    /**
     * 完整推理: 给定 RGB 像素 buffer + 尺寸, 返回 17 关键点
     *
     * @param rgbPixels RGB 紧凑字节 (3 字节/像素, 顺序 R, G, B), 长度 = w*h*3
     * @param w 宽度
     * @param h 高度
     */
    suspend fun detect(rgbPixels: ByteArray, w: Int, h: Int): PoseResult? =
        withContext(Dispatchers.Default) {
            val session = yoloxSession ?: return@withContext null
            val poseSession = rtmposeSession ?: return@withContext null

            // 1. YOLOX 检测人体 bbox
            val dets = detectPersons(rgbPixels, w, h)
            if (dets.isEmpty()) {
                Log.d(TAG, "no person detected")
                return@withContext null
            }
            val best = dets.first()
            Log.d(TAG, "best det: score=${best.score}")

            // 2. RTMPose 关键点
            val kps = estimatePose(rgbPixels, w, h, best) ?: return@withContext null
            PoseResult(kps, best)
        }

    /**
     * YOLOX 推理: letterbox 到 416x416, 模型已内置 NMS
     *
     * 与 rtmlib.tools.object_detection.yolox.YOLOX 完全对齐:
     *   - top-left 对齐 (不居中!), pad 114 灰
     *   - **不除以 255!** 直接喂 0-255 的 float32 (rtmlib 也是这样)
     *   - 输入 buffer 是 RGB, 模型训练用 OpenCV BGR, 通道反过来送 BGR
     *   - 反 letterbox: 直接 /scale, 不减 dx/dy (因为本来就是 top-left)
     */
    private fun detectPersons(rgb: ByteArray, w: Int, h: Int): List<Detection> {
        val session = yoloxSession ?: run {
            lastDebugInfo = "yolox session null"
            return emptyList()
        }

        // letterbox 缩放: 保持长宽比
        val scale = min(YOLOX_INPUT_SIZE.toFloat() / w, YOLOX_INPUT_SIZE.toFloat() / h)
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()

        // NCHW float32 [1, 3, 416, 416], 填充 114 (YOLOX 标准 padding 灰度)
        // ⚠️ 不除 255! rtmlib 也是直接送 0-255 的 float32
        val input = FloatArray(3 * YOLOX_INPUT_SIZE * YOLOX_INPUT_SIZE)
        for (i in input.indices) input[i] = 114f

        for (y in 0 until newH) {
            val srcY = (y / scale).toInt().coerceIn(0, h - 1)
            for (x in 0 until newW) {
                val srcX = (x / scale).toInt().coerceIn(0, w - 1)
                val srcIdx = (srcY * w + srcX) * 3
                val dstIdx = y * YOLOX_INPUT_SIZE + x
                // RGB → BGR (匹配 YOLOX 训练的 OpenCV BGR 顺序)
                input[0 * YOLOX_INPUT_SIZE * YOLOX_INPUT_SIZE + dstIdx] = (rgb[srcIdx + 2].toInt() and 0xFF).toFloat()  // B
                input[1 * YOLOX_INPUT_SIZE * YOLOX_INPUT_SIZE + dstIdx] = (rgb[srcIdx + 1].toInt() and 0xFF).toFloat()  // G
                input[2 * YOLOX_INPUT_SIZE * YOLOX_INPUT_SIZE + dstIdx] = (rgb[srcIdx].toInt() and 0xFF).toFloat()      // R
            }
        }

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1L, 3L, YOLOX_INPUT_SIZE.toLong(), YOLOX_INPUT_SIZE.toLong())
        )

        // YOLOX 输出: ['dets', 'labels']
        // dets [1, N, 5] - 已内置 NMS (按 score 降序)
        // Note: OrtSession.run() returns OrtSession.Result which implements
        // Iterable<Map.Entry<String, OnnxValue>> + AutoCloseable
        val result = session.run(mapOf("input" to inputTensor))
        inputTensor.close()
        try {
            val detsEntry = result.first { it.key == "dets" }
            val detsValue = detsEntry.value.value
            val valueClassName = detsValue?.javaClass?.name ?: "null"
            Log.d(TAG, "dets value class: $valueClassName")

            // 兼容多种返回类型: Array<FloatArray>, Array<DoubleArray>, Array<Array<Float>>, ...
            val detsRaw: Array<*> = when (detsValue) {
                is Array<*> -> detsValue[0] as Array<*>
                else -> {
                    lastDebugInfo = "dets type=$valueClassName (not Array)"
                    return emptyList()
                }
            }
            Log.d(TAG, "dets[0] len=${detsRaw.size}, first=${detsRaw.firstOrNull()}")

            val list = mutableListOf<Detection>()
            var bestScore = 0f
            var unknownTypeCount = 0
            var unknownTypeName = ""

            for (det in detsRaw) {
                // YOLOX 输出: [x1, y1, x2, y2, score]
                val floats: FloatArray? = when (det) {
                    is FloatArray -> det
                    is DoubleArray -> FloatArray(det.size) { det[it].toFloat() }
                    is Array<*> -> {
                        // boxed Float/Double 数组
                        if (det.isEmpty()) null
                        else when (det[0]) {
                            is Float -> FloatArray(det.size) { (det[it] as Float) }
                            is Double -> FloatArray(det.size) { (det[it] as Double).toFloat() }
                            is Number -> FloatArray(det.size) { (det[it] as Number).toFloat() }
                            else -> {
                                unknownTypeCount++
                                if (unknownTypeName.isEmpty()) unknownTypeName = det[0]?.javaClass?.name ?: "?"
                                null
                            }
                        }
                    }
                    else -> {
                        unknownTypeCount++
                        if (unknownTypeName.isEmpty()) unknownTypeName = det?.javaClass?.name ?: "?"
                        null
                    }
                }
                if (floats == null || floats.size < 5) continue
                val score = floats[4]
                if (score > bestScore) bestScore = score
                if (score < YOLOX_SCORE_THRESHOLD) continue
                // 反 letterbox: top-left 对齐, 只需 /scale (不减 dx/dy)
                list.add(Detection(
                    x1 = (floats[0] / scale).coerceIn(0f, w.toFloat()),
                    y1 = (floats[1] / scale).coerceIn(0f, h.toFloat()),
                    x2 = (floats[2] / scale).coerceIn(0f, w.toFloat()),
                    y2 = (floats[3] / scale).coerceIn(0f, h.toFloat()),
                    score = score
                ))
            }
            Log.d(TAG, "detected ${list.size} persons (raw=${detsRaw.size}, best=$bestScore, unknown=$unknownTypeCount/$unknownTypeName)")
            lastDebugInfo = "raw=${detsRaw.size} best=${"%.2f".format(bestScore)} ok=${list.size}" +
                (if (unknownTypeCount > 0) " unk=$unknownTypeCount/$unknownTypeName" else "") +
                " type=${valueClassName.substringAfterLast('.')}"
            // YOLOX 输出可能已按 score 排序, 但保险起见再排一次
            list.sortByDescending { it.score }
            return list
        } finally {
            result.close()
        }
    }

    /**
     * RTMPose 推理: crop bbox 区域 + resize 到 256x192, simcc 后处理
     */
    private fun estimatePose(
        rgb: ByteArray, w: Int, h: Int, det: Detection
    ): FloatArray? {
        val session = rtmposeSession ?: return null

        val x1 = max(0, det.x1.toInt())
        val y1 = max(0, det.y1.toInt())
        val x2 = min(w, det.x2.toInt())
        val y2 = min(h, det.y2.toInt())
        val boxW = x2 - x1
        val boxH = y2 - y1
        if (boxW <= 0 || boxH <= 0) return null

        // NCHW float32 [1, 3, 256, 192], (x - mean) / std, 双线性缩放
        val input = FloatArray(3 * RTMPOSE_INPUT_H * RTMPOSE_INPUT_W)
        val sx = boxW.toFloat() / RTMPOSE_INPUT_W
        val sy = boxH.toFloat() / RTMPOSE_INPUT_H
        for (y in 0 until RTMPOSE_INPUT_H) {
            val srcY = (y1 + (y * sy).toInt()).coerceIn(0, h - 1)
            for (x in 0 until RTMPOSE_INPUT_W) {
                val srcX = (x1 + (x * sx).toInt()).coerceIn(0, w - 1)
                val srcIdx = (srcY * w + srcX) * 3
                for (c in 0 until 3) {
                    val v = (rgb[srcIdx + c].toInt() and 0xFF)
                    input[c * RTMPOSE_INPUT_H * RTMPOSE_INPUT_W + y * RTMPOSE_INPUT_W + x] =
                        (v - MEAN[c]) / STD[c]
                }
            }
        }

        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1L, 3L, RTMPOSE_INPUT_H.toLong(), RTMPOSE_INPUT_W.toLong())
        )

        val result = session.run(mapOf("input" to inputTensor))
        inputTensor.close()

        // simcc_x [1, 17, 384], simcc_y [1, 17, 512]
        try {
            val simccXEntry = result.first { it.key == "simcc_x" }
            val simccYEntry = result.first { it.key == "simcc_y" }
            @Suppress("UNCHECKED_CAST")
            val simccX = (simccXEntry.value.value as Array<Array<FloatArray>>)[0]
            @Suppress("UNCHECKED_CAST")
            val simccY = (simccYEntry.value.value as Array<Array<FloatArray>>)[0]

            // argmax + 缩放回输入坐标
            // 与 rtmlib get_simcc_maximum 对齐: 低置信度点 (vals<=0) 置 (0,0)
            // 否则 ExerciseCounter 的 (0,0) 无效点过滤永远不触发
            val keypoints = FloatArray(17 * 2)
            for (kp in 0 until 17) {
                val xArr = simccX[kp]
                val yArr = simccY[kp]
                var maxX = 0
                var maxY = 0
                var maxVX = Float.NEGATIVE_INFINITY
                var maxVY = Float.NEGATIVE_INFINITY
                for (i in xArr.indices) {
                    if (xArr[i] > maxVX) { maxVX = xArr[i]; maxX = i }
                }
                for (i in yArr.indices) {
                    if (yArr[i] > maxVY) { maxVY = yArr[i]; maxY = i }
                }
                // 置信度 = (maxVX + maxVY) / 2, 与 rtmlib 一致
                val conf = (maxVX + maxVY) / 2f
                if (conf <= 0f) {
                    // 低置信度, 标记为 (0,0) (ExerciseCounter 用此判断无效点)
                    keypoints[kp * 2] = 0f
                    keypoints[kp * 2 + 1] = 0f
                } else {
                    // simccSplit=2.0, 除以得输入坐标系下的位置
                    val poseX = maxX / SIMCC_SPLIT
                    val poseY = maxY / SIMCC_SPLIT
                    // 映射回原图 bbox 区域
                    keypoints[kp * 2] = x1 + poseX / RTMPOSE_INPUT_W * boxW
                    keypoints[kp * 2 + 1] = y1 + poseY / RTMPOSE_INPUT_H * boxH
                }
            }
            return keypoints
        } finally {
            result.close()
        }
    }

    private fun readAssetBytes(name: String): ByteArray {
        context.assets.open(name).use { input ->
            return input.readBytes()
        }
    }

    fun close() {
        yoloxSession?.close()
        rtmposeSession?.close()
    }
}
