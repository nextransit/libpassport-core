import org.gradle.api.tasks.testing.logging.TestLogEvent

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    ndkVersion = "26.1.10909125"
    namespace = "com.nextransit.mrzbench"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.nextransit.mrzbench"
        minSdk = 28           // Android 9 (192.168.26.123)
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"
        ndk {
            abiFilters += listOf("armeabi-v7a")
        }
        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_static"
                arguments += "-DANDROID_PLATFORM=android-28"
                cppFlags += listOf("-std=c++17")
            }
        }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
        buildConfig = true
    }
}

dependencies {
        implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.5")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.8.5")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
