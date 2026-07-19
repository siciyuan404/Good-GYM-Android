allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// 强制所有 Android 子项目 (含 Flutter 插件如 onnxruntime/camera) 使用 compileSdk=36
// 解决 onnxruntime 1.4.1 用 android-33 编译但其 androidx 依赖要求 34+ 的冲突
// 用 gradle.afterProject 钩子: 在子项目 evaluate 完毕后强制覆盖 android.compileSdk
// (evaluationDependsOn 让 :app 先 evaluate, 但其他插件子项目仍然走正常流程)
gradle.afterProject {
    val androidExt = extensions.findByName("android")
    if (androidExt is com.android.build.gradle.LibraryExtension) {
        androidExt.compileSdkVersion(36)
    } else if (androidExt is com.android.build.gradle.AppExtension) {
        androidExt.compileSdkVersion(36)
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
