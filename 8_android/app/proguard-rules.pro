# keep native JNI entry points
-keepclasseswithmembers class * { native <methods>; }
-keep class com.nextransit.mrzbench.jni.** { *; }
