package com.goodgym.core

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * 单个运动配置 - 对应 data/exercises.json 中的一项
 *
 * [keypoints] left/right 三点: [端点, 中点(关节), 端点]
 *   端点-中点-端点的夹角即为运动角度
 *
 * [isLegExercise] true: 双腿独立计数 (左/右各算一次 rep)
 *                false: 左右平均, 单计数
 *
 * [angleDirection] 推理时关节角度的变化方向
 *   "decreasing": 起始位角度高, 动作完成角度低 (如深蹲: 160° → 110°)
 *   "increasing": 反之
 *   不填时按 up/down_angle 自动推断
 */
@Serializable
data class ExerciseConfig(
    val name_zh: String,
    val name_en: String? = null,
    val down_angle: Float,
    val up_angle: Float,
    val keypoints: KeypointIndices,
    val is_leg_exercise: Boolean = false,
    val angle_direction: String? = null
)

@Serializable
data class KeypointIndices(
    val left: List<Int>,
    val right: List<Int>
)

@Serializable
data class ExercisesFile(
    val exercises: Map<String, ExerciseConfig>
)

/**
 * 单例仓库 - 加载 assets/data/exercises.json
 */
class ExerciseRepository(private val context: Context) {
    private val json = Json { ignoreUnknownKeys = true }
    private var data: Map<String, ExerciseConfig> = emptyMap()

    val isLoaded: Boolean get() = data.isNotEmpty()
    val keys: Set<String> get() = data.keys

    operator fun get(key: String): ExerciseConfig? = data[key]
    fun values(): List<ExerciseConfig> = data.values.toList()

    fun load() {
        val text = context.assets.open("data/exercises.json").bufferedReader().use { it.readText() }
        val parsed = json.decodeFromString(ExercisesFile.serializer(), text)
        data = parsed.exercises
    }
}
