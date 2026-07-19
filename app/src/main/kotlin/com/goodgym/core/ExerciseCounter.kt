package com.goodgym.core

import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * 运动计数器 - 翻译自桌面版 [exercise_counters.py]
 *
 * 核心算法:
 *   1. 三点夹角 = atan2(cross, dot) 计算关节角度
 *   2. 滑动窗口平滑: 中位数 + 2σ 离群点过滤
 *   3. up/down 阶段切换 + 0.5s 最小 rep 间隔
 *   4. 腿部动作独立计数: 左右腿各自维护 stage 状态
 *
 * 状态机:
 *   down -> up: 关节伸过 up_angle 阈值
 *   up -> down: 关节弯过 down_angle 阈值
 *   一次完整 down -> up -> down 算 1 rep (或方向相反)
 */
class ExerciseCounter(
    private val smoothingWindow: Int = 5
) {
    enum class Stage { NONE, UP, DOWN }

    var count: Int = 0
        private set

    private var stage: Stage = Stage.NONE
    private var leftLegStage: Stage = Stage.NONE
    private var rightLegStage: Stage = Stage.NONE

    private val angleHistory = ArrayDeque<Float>()
    private var lastCountTimeMs: Long = 0

    companion object {
        private const val MIN_REP_INTERVAL_MS = 500L
    }

    /**
     * 处理一帧关键点, 返回当前平滑角度 (度数, 可能为 null)
     *
     * [keypoints] 长度 34 的 FloatArray (17 个点 × 2 坐标)
     */
    fun update(keypoints: FloatArray, config: ExerciseConfig): Float? {
        val kp = config.keypoints

        // 计算左右两侧角度
        val leftAngle = calculateAngle(
            keypoints, kp.left[0], kp.left[1], kp.left[2]
        ) ?: return null
        val rightAngle = calculateAngle(
            keypoints, kp.right[0], kp.right[1], kp.right[2]
        ) ?: return null

        // 腿部动作: 双腿独立计数
        if (config.is_leg_exercise) {
            return countLegExercise(leftAngle, rightAngle, config)
        }

        // 其他动作: 左右平均
        val avg = (leftAngle + rightAngle) / 2
        val smoothed = smoothAngle(avg) ?: return null

        val upThr = config.up_angle
        val downThr = config.down_angle
        val direction = getAngleDirection(config)

        if (direction == "decreasing") {
            when {
                smoothed > downThr -> stage = Stage.DOWN
                smoothed < upThr && stage == Stage.DOWN && checkRepTiming() -> {
                    stage = Stage.UP
                    count++
                    lastCountTimeMs = System.currentTimeMillis()
                }
            }
        } else {
            when {
                smoothed > upThr -> stage = Stage.UP
                smoothed < downThr && stage == Stage.UP && checkRepTiming() -> {
                    stage = Stage.DOWN
                    count++
                    lastCountTimeMs = System.currentTimeMillis()
                }
            }
        }

        return smoothed
    }

    private fun countLegExercise(
        leftAngle: Float, rightAngle: Float, config: ExerciseConfig
    ): Float {
        val upThr = config.up_angle
        val downThr = config.down_angle
        val direction = getAngleDirection(config)

        if (checkRepTiming()) {
            // 左腿
            if (direction == "decreasing") {
                val leftReady = leftAngle > downThr
                val leftCounted = leftAngle < upThr && leftLegStage == Stage.DOWN
                if (leftReady) {
                    leftLegStage = Stage.DOWN
                } else if (leftCounted) {
                    count++
                    lastCountTimeMs = System.currentTimeMillis()
                    leftLegStage = Stage.UP
                }
            } else {
                val leftReady = leftAngle > upThr
                val leftCounted = leftAngle < downThr && leftLegStage == Stage.UP
                if (leftReady) {
                    leftLegStage = Stage.UP
                } else if (leftCounted) {
                    count++
                    lastCountTimeMs = System.currentTimeMillis()
                    leftLegStage = Stage.DOWN
                }
            }

            // 右腿
            if (direction == "decreasing") {
                val rightReady = rightAngle > downThr
                val rightCounted = rightAngle < upThr && rightLegStage == Stage.DOWN
                if (rightReady) {
                    rightLegStage = Stage.DOWN
                } else if (rightCounted) {
                    count++
                    lastCountTimeMs = System.currentTimeMillis()
                    rightLegStage = Stage.UP
                }
            } else {
                val rightReady = rightAngle > upThr
                val rightCounted = rightAngle < downThr && rightLegStage == Stage.UP
                if (rightReady) {
                    rightLegStage = Stage.UP
                } else if (rightCounted) {
                    count++
                    lastCountTimeMs = System.currentTimeMillis()
                    rightLegStage = Stage.DOWN
                }
            }
        }

        return (leftAngle + rightAngle) / 2
    }

    private fun getAngleDirection(config: ExerciseConfig): String {
        config.angle_direction?.let { if (it == "increasing" || it == "decreasing") return it }
        return if (config.up_angle < config.down_angle) "decreasing" else "increasing"
    }

    private fun checkRepTiming(): Boolean {
        val now = System.currentTimeMillis()
        return now - lastCountTimeMs >= MIN_REP_INTERVAL_MS
    }

    private fun smoothAngle(angle: Float): Float? {
        angleHistory.addLast(angle)
        if (angleHistory.size > smoothingWindow) angleHistory.removeFirst()
        if (angleHistory.size < 3) return angle

        // 中位数
        val sorted = angleHistory.sorted()
        val median = sorted[sorted.size / 2]
        // 标准差
        val mean = angleHistory.toFloatArray().sum() / angleHistory.size
        val variance = angleHistory.toFloatArray().map { (it - mean) * (it - mean) }.sum() / angleHistory.size
        val std = sqrt(variance)
        // 过滤 2σ 离群点
        val filtered = angleHistory.filter { kotlin.math.abs(it - median) <= 2 * std }
        return if (filtered.isNotEmpty()) filtered.toFloatArray().sum() / filtered.size else angle
    }

    /**
     * 计算三点夹角 - 端点 a, 关节 b, 端点 c
     *
     * 返回 (0, 180] 度数, 任意点为 (0, 0) 视为无效返回 null
     */
    private fun calculateAngle(
        kps: FloatArray, aIdx: Int, bIdx: Int, cIdx: Int
    ): Float? {
        val ax = kps[aIdx * 2]; val ay = kps[aIdx * 2 + 1]
        val bx = kps[bIdx * 2]; val by = kps[bIdx * 2 + 1]
        val cx = kps[cIdx * 2]; val cy = kps[cIdx * 2 + 1]
        // 任意一点为 (0, 0) 视为无效 (低置信度被过滤)
        if (ax == 0f && ay == 0f) return null
        if (bx == 0f && by == 0f) return null
        if (cx == 0f && cy == 0f) return null

        val bax = ax - bx; val bay = ay - by
        val bcx = cx - bx; val bcy = cy - by
        val dot = bax * bcx + bay * bcy
        val cross = bax * bcy - bay * bcx
        var ang = Math.toDegrees(atan2(cross.toDouble(), dot.toDouble())).toFloat()
        if (ang < 0) ang += 180f
        return ang
    }

    fun reset() {
        count = 0
        stage = Stage.NONE
        leftLegStage = Stage.NONE
        rightLegStage = Stage.NONE
        angleHistory.clear()
        lastCountTimeMs = 0
    }
}
