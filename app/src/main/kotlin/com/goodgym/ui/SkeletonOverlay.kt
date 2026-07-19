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
 * [keypoints] 原图坐标系下的 17 个点 (FloatArray 长度 34, 顺序 x0,y0,x1,y1,...)
 * [imageW]/[imageH] 推理输入图像尺寸 (用于做坐标缩放映射)
 * [rotationDegrees] 相机帧旋转角度 (通常后置 portrait UI 是 90°)
 * [mirror] 是否水平镜像 (前置相机 true, 后置 false)
 *
 * 内部做 contain 缩放: 把 image 坐标系映射到当前 Canvas 尺寸, 保持长宽比 + 居中
 */
@Composable
fun SkeletonOverlay(
    keypoints: FloatArray?,
    imageW: Int,
    imageH: Int,
    rotationDegrees: Int,
    mirror: Boolean,
    modifier: Modifier = Modifier
) {
    Canvas(modifier = modifier.fillMaxSize()) {
        if (keypoints == null || keypoints.size < 34 || imageW <= 0 || imageH <= 0) return@Canvas

        val canvasW = size.width
        val canvasH = size.height

        // 把原图坐标系旋转到 portrait (与 PreviewView 一致)
        // 后置相机 sensor 横屏, 预览被旋转 90° 显示成 portrait
        // 推理图像坐标系仍是 sensor 原始方向, 这里做坐标变换
        data class Pt(val x: Float, val y: Float)
        val transformed = Array(17) { i ->
            val ox = keypoints[i * 2]
            val oy = keypoints[i * 2 + 1]
            if (ox == 0f && oy == 0f) {
                Pt(0f, 0f) // 无效点 (低置信度被过滤)
            } else {
                when (rotationDegrees) {
                    90 -> Pt(imageH - oy, ox)         // 顺时针 90°
                    180 -> Pt(imageW - ox, imageH - oy)
                    270 -> Pt(oy, imageW - ox)
                    else -> Pt(ox, oy)
                }
            }
        }
        // 旋转后的图像"逻辑"尺寸 (用于 contain 缩放)
        val logicalW = if (rotationDegrees == 90 || rotationDegrees == 270) imageH.toFloat() else imageW.toFloat()
        val logicalH = if (rotationDegrees == 90 || rotationDegrees == 270) imageW.toFloat() else imageH.toFloat()

        // contain 缩放: 把 logicalW × logicalH 等比塞进 canvasW × canvasH
        val scale = minOf(canvasW / logicalW, canvasH / logicalH)
        val offsetX = (canvasW - logicalW * scale) / 2f
        val offsetY = (canvasH - logicalH * scale) / 2f

        fun mapToCanvas(p: Pt): Offset {
            var tx = offsetX + p.x * scale
            if (mirror) tx = canvasW - tx
            val ty = offsetY + p.y * scale
            return Offset(tx, ty)
        }

        // 1. 绘制骨骼连线 (亮青色)
        val lineStrokeWidth = 4.dp.toPx()
        for ((a, b) in COCO_SKELETON_PAIRS) {
            if (a >= 17 || b >= 17) continue
            val pa = transformed[a]
            val pb = transformed[b]
            if ((pa.x == 0f && pa.y == 0f) || (pb.x == 0f && pb.y == 0f)) continue
            drawLine(
                color = AccentCyan,
                start = mapToCanvas(pa),
                end = mapToCanvas(pb),
                strokeWidth = lineStrokeWidth,
                cap = StrokeCap.Round
            )
        }

        // 2. 绘制 17 个关键点 (亮黄色实心圆)
        val pointRadius = 6.dp.toPx()
        for (kp in transformed) {
            if (kp.x == 0f && kp.y == 0f) continue
            drawCircle(
                color = AccentYellow,
                radius = pointRadius,
                center = mapToCanvas(kp)
            )
        }
    }
}
