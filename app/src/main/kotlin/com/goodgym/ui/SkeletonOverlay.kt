package com.goodgym.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.unit.dp
import com.goodgym.ui.theme.AccentCyan
import com.goodgym.ui.theme.AccentYellow

/**
 * COCO 17 关键点标准连线表
 * (nose-eyes, ears, shoulders, hips, arms, legs)
 */
val COCO_SKELETON_PAIRS = listOf(
    0 to 1, 0 to 2, 1 to 3, 2 to 4,       // 面部
    5 to 6,                                // 双肩
    5 to 11, 6 to 12, 11 to 12,            // 躯干
    5 to 7, 7 to 9,                        // 左臂
    6 to 8, 8 to 10,                       // 右臂
    11 to 13, 13 to 15,                    // 左腿
    12 to 14, 14 to 16                     // 右腿
)

/**
 * 骨架绘制层 - 在相机预览上叠加 17 关键点 + 16 条骨骼连线
 *
 * [keypoints] portrait 坐标系下的 17 个点 (FloatArray 长度 34, 顺序 x0,y0,x1,y1,...)
 * [imageW]/[imageH] 推理输入图像尺寸 (已是 portrait 旋转后, 用于做坐标缩放映射)
 * [rotationDegrees] 已废弃 (旋转在 CameraManager 推理前完成), 保留参数兼容旧调用
 * [mirror] 是否水平镜像 (前置相机 true, 后置 false)
 *
 * 内部做 cover 缩放: 把 image 坐标系映射到当前 Canvas 尺寸, 保持长宽比 + 裁剪超出部分
 * (与 PreviewView 的 FILL_CENTER 一致, 否则骨架和画面会对不上)
 */
@Composable
fun SkeletonOverlay(
    keypoints: FloatArray?,
    imageW: Int,
    imageH: Int,
    @Suppress("UNUSED_PARAMETER") rotationDegrees: Int,
    mirror: Boolean,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        if (keypoints == null || keypoints.size < 34 || imageW <= 0 || imageH <= 0) return@Canvas

        val canvasW = size.width
        val canvasH = size.height

        // cover 缩放: 把 imageW × imageH 等比铺满 canvas (会裁剪超出部分)
        // 必须用 maxOf 而非 minOf, 否则 contain 缩放会留黑边, 和 PreviewView FILL_CENTER 不一致
        val scale = maxOf(canvasW / imageW.toFloat(), canvasH / imageH.toFloat())
        val offsetX = (canvasW - imageW * scale) / 2f
        val offsetY = (canvasH - imageH * scale) / 2f

        // 直接映射, 不再做坐标旋转 (旋转已在 CameraManager 推理前完成)
        fun mapToCanvas(idx: Int): Offset {
            val ox = keypoints[idx * 2]
            val oy = keypoints[idx * 2 + 1]
            var tx = offsetX + ox * scale
            if (mirror) tx = canvasW - tx
            val ty = offsetY + oy * scale
            return Offset(tx, ty)
        }

        // 1. 绘制骨骼连线 (亮青色)
        val lineStrokeWidth = 4.dp.toPx()
        for ((a, b) in COCO_SKELETON_PAIRS) {
            if (a >= 17 || b >= 17) continue
            val ax = keypoints[a * 2]; val ay = keypoints[a * 2 + 1]
            val bx = keypoints[b * 2]; val by = keypoints[b * 2 + 1]
            // 跳过无效点 (低置信度置 0)
            if ((ax == 0f && ay == 0f) || (bx == 0f && by == 0f)) continue
            drawLine(
                color = AccentCyan,
                start = mapToCanvas(a),
                end = mapToCanvas(b),
                strokeWidth = lineStrokeWidth,
                cap = StrokeCap.Round
            )
        }

        // 2. 绘制 17 个关键点 (亮黄色实心圆)
        val pointRadius = 6.dp.toPx()
        for (i in 0 until 17) {
            val x = keypoints[i * 2]
            val y = keypoints[i * 2 + 1]
            if (x == 0f && y == 0f) continue
            drawCircle(
                color = AccentYellow,
                radius = pointRadius,
                center = mapToCanvas(i)
            )
        }
    }
}
