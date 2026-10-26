package com.nextransit.mrzbench

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import android.view.View
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.nextransit.mrzbench.photo.PhotoMrzPipeline
import com.nextransit.mrzbench.photo.verdict
import com.nextransit.mrzbench.ui.PhotoMrzFragment
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * UI 层验证：把一张能命中 issuerSuspect / countryMismatch 的照片灌进
 * PhotoMrzFragment，确认两行展示确实被渲染出来。
 *
 * 为什么要有这个测试：
 *
 * 设备侧的无界面测试（PhotoMrzDeviceTest）只验证了「JNI -> Kotlin ->
 * 三态判定」这条链路上的**数据**是对的。但 render() 里那两行文本是
 * 独立的代码路径：它读的是 output.issuerSuspect / output.countryMismatch，
 * 拼成 SpannableString 写进 TextView。数据对了不代表字符串就一定
 * 被拼出来——写错了 key、写死了文案、把条件写反，设备测试全都发现
 * 不了，因为那些测试根本不看 UI。
 *
 * 所以这里必须真的把 Fragment 挂起来、真的跑一遍 render()、真的读回
 * TextView 的文本。
 */
@RunWith(AndroidJUnit4::class)
class PhotoMrzUiTest {

    private val instrumentation get() = InstrumentationRegistry.getInstrumentation()

    /**
     * 直接调Fragment 的 render 路径。
     *
     * 不走「选图 -> 点 Run」这套 UI 交互（那需要 Espresso，测试 APK
     * 里没有这个依赖，而且点sample 会弹 AlertDialog，跨进程选条目很
     * 脆）。而是把 Fragment 挂到一个真实的 Activity 上、灌入图片、
     * 然后用反射调私有的 render()。
     *
     * 这样测的是**渲染逻辑本身**，而不是点击链路——后者由别的测试
     * 负责。
     */
    @Test
    fun issuerAndCountryRowsAreRendered() {
        val target = instrumentation.targetContext
        val testCtx = instrumentation.context

        // 找到一张能命中新信号的图：Photo/3/IMG_1792133000_003.jpg 在
        // 真机上 issuer=true、country=true、verdict=SUSPECT。
        val path = "Photo/3/IMG_1792133000_003.jpg"
        val bmp = target.assets.open(path).use { BitmapFactory.decodeStream(it) }
            ?: error("cannot decode $path")

        // 先跑一次 pipeline 拿到 PhotoMrzResult——这与设备测试用的是
        // 同一条路径，所以 UI 显示的内容与设备测试断言的内容必然同源。
        val out = PhotoMrzPipeline.process(target, bmp, "auto")
        Log.i(
            TAG,
            "PIPELINE decode=${out.decodeOk} struct=${out.structOk} nat=${out.natSuspect}" +
                " issuer=${out.issuerSuspect} country=${out.countryMismatch}" +
                " verdict=${out.verdict} backend=${out.backend}",
        )
        Log.i(TAG, "PIPELINE line1=${out.line1} line2=${out.line2}")

        // 断言前置条件：这张图必须真的命中新信号，否则后面的 UI 断言
        // 是空转（会「通过」但什么都没测）。
        assertTrue("样本必须命中 issuerSuspect，实际 issuer=${out.issuerSuspect}", out.issuerSuspect)
        assertTrue("样本必须命中 countryMismatch，实际 country=${out.countryMismatch}", out.countryMismatch)
        // verdict 是 MrzVerdict 枚举，不是 String——用 toString 比。
        assertEquals("样本应判为 SUSPECT", "SUSPECT", out.verdict.toString())

        // 挂Fragment
        val activity = launchFragment()
        val frag = activity.supportFragmentManager.findFragmentByTag("f3") as PhotoMrzFragment
        val view = frag.requireView()
        val resultView = view.findViewById<android.widget.TextView>(
            resourcesId(target, "photo_result")
        )
        val statusView = view.findViewById<android.widget.TextView>(
            resourcesId(target, "photo_status")
        )

        // 反射调私有 render()：它是private的，但 UI 渲染路径本身就是
        // 我们要验证的对象，用反射比给它开internal 更贴近真实。
        val render = PhotoMrzFragment::class.java.getDeclaredMethod(
            "render", com.nextransit.mrzbench.photo.PhotoMrzResult::class.java
        )
        render.isAccessible = true
        render.invoke(frag, out)

        val rendered = resultView.text.toString()
        val status = statusView.text.toString()
        Log.i(TAG, "UI result text >>>\n$rendered")
        Log.i(TAG, "UI status text >>> $status")

        // --- 断言 1：issuer 行必须出现，且文案正确 ---
        assertTrue(
            "结果区必须出现 issuer 行，实际文本：\n$rendered",
            rendered.contains("issuer"),
        )
        assertTrue(
            "issuer 行必须是 SUSPECT，实际文本：\n$rendered",
            rendered.contains("SUSPECT  (签发国码不在 ISO-3166 内)"),
        )

        // --- 断言 2：country 行必须出现，且文案正确 ---
        assertTrue(
            "结果区必须出现 country 行，实际文本：\n$rendered",
            rendered.contains("country"),
        )
        assertTrue(
            "country 行必须是 MISMATCH，实际文本：\n$rendered",
            rendered.contains("MISMATCH  (签发国与国籍不一致)"),
        )

        // --- 断言 3：两行都出现时，行序必须对（issuer 在 country 之前）---
        val issuerIdx = rendered.indexOf("issuer")
        val countryIdx = rendered.indexOf("country")
        assertTrue(
            "issuer 行应排在 country 行之前，实际：\n$rendered",
            issuerIdx in 0 until countryIdx,
        )

        // --- 断言 4：状态栏也必须反映新信号 ---
        assertTrue(
            "状态栏应提示签发国码可疑，实际：$status",
            status.contains("签发国码可疑"),
        )

        // --- 断言 5：不得把最终判定显示成 PASS ---
        //
        // 注意区分两层含义：
        //   - 「decode  PASS」是校验位闭合（decodeOk），与最终判定无关，
        //     一个 SUSPECT 结果本来就会是 decode PASS（校验位凑齐了但
        //     内容不可能为真）。所以这里不能断言结果区不含 "PASS"。
        //   - 最终判定由 status 行给出，status 必须是「签发国码可疑」而
        //     不是 "PASS"。
        assertTrue(
            "状态栏不应显示最终 PASS，实际：$status",
            !status.contains("PASS"),
        )

        // 截图留证
        try {
            val dir = File(target.filesDir, "uishot").apply { mkdirs() }
            val (w, h) = layoutView(view)
            val shot = android.graphics.Bitmap.createBitmap(
                w, h, android.graphics.Bitmap.Config.ARGB_8888
            )
            val canvas = android.graphics.Canvas(shot)
            view.draw(canvas)
            val f = File(dir, "issuer_country_render.png")
            f.outputStream().use { shot.compress(Bitmap.CompressFormat.PNG, 100, it) }
            Log.i(TAG, "UI screenshot=${f.absolutePath} ${shot.width}x${shot.height}")
        } catch (t: Throwable) {
            Log.w(TAG, "screenshot failed: ${t.message}")
        }

        bmp.recycle()
        Log.i(TAG, "UI_OK issuer+country rendered")
    }

    /**
     * 干净样本不应出现 issuer / country 两行。
     *
     * 反向断言很重要：如果这两行是无条件显示的，上面那条测试照样会
     * 「通过」。
     */
    @Test
    fun cleanSampleHasNoIssuerOrCountryRows() {
        val target = instrumentation.targetContext
        // Photo/lv/2c793efe12276f9b_0258.jpg 在真机上四个信号全为false。
        val path = "Photo/lv/2c793efe12276f9b_0258.jpg"
        val bmp = target.assets.open(path).use { BitmapFactory.decodeStream(it) }
            ?: error("cannot decode $path")
        val out = PhotoMrzPipeline.process(target, bmp, "auto")
        Log.i(
            TAG,
            "PIPELINE(clean) decode=${out.decodeOk} struct=${out.structOk} nat=${out.natSuspect}" +
                " issuer=${out.issuerSuspect} country=${out.countryMismatch}" +
                " verdict=${out.verdict}",
        )
        assertFalse("干净样本不应命中 issuer", out.issuerSuspect)
        assertFalse("干净样本不应命中 country", out.countryMismatch)

        val activity = launchFragment()
        val frag = activity.supportFragmentManager.findFragmentByTag("f3") as PhotoMrzFragment
        val view = frag.requireView()
        val resultView = view.findViewById<android.widget.TextView>(
            resourcesId(target, "photo_result")
        )
        val render = PhotoMrzFragment::class.java.getDeclaredMethod(
            "render", com.nextransit.mrzbench.photo.PhotoMrzResult::class.java
        )
        render.isAccessible = true
        render.invoke(frag, out)
        val rendered = resultView.text.toString()
        Log.i(TAG, "UI(clean) result text >>>\n$rendered")
        assertFalse(
            "干净样本不应显示 issuer 行，实际：\n$rendered",
            rendered.contains("issuer  "),
        )
        assertFalse(
            "干净样本不应显示 country 行，实际：\n$rendered",
            rendered.contains("country "),
        )
        try {
            val dir = File(target.filesDir, "uishot").apply { mkdirs() }
            val (w, h) = layoutView(view)
            val shot = android.graphics.Bitmap.createBitmap(
                w, h, android.graphics.Bitmap.Config.ARGB_8888
            )
            view.draw(android.graphics.Canvas(shot))
            val f = File(dir, "clean_render.png")
            f.outputStream().use { shot.compress(Bitmap.CompressFormat.PNG, 100, it) }
            Log.i(TAG, "UI screenshot=${f.absolutePath} ${shot.width}x${shot.height}")
        } catch (t: Throwable) {
            Log.w(TAG, "screenshot failed: ${t.message}")
        }
        bmp.recycle()
        Log.i(TAG, "UI_OK clean sample clean rows")
    }

    /**
     * 让 view 有一个可用的尺寸并完成一次 layout。
     *
     * 无头 instrumentation 里Activity 的窗口可能从未真正 layout，view 的
     * width/height 与 measuredWidth/measuredHeight 全是 0——直接
     * Bitmap.createBitmap(0, 0, ...) 会抛 "width and height must be > 0"，
     * 而 view.draw() 在 0 尺寸上会抛 "Index 0 out of bounds"。
     *
     * 所以这里用固定尺寸强行 layout，不去读测量结果（测量结果在某些
     * 设备上会是 0，即便 measure() 已经被调用）。
     */
    /**
     * 截图尺寸刻意取小（360x640）。
     *
     * 原来用 1080x1920 时，在低内存真机（p18q_dual_extend_64）上分配这张
     * 8 MB 的 ARGB bitmap 会被 lowmemorykiller 直接杀掉进程，测试报成
     * "Process crashed" 而不是断言失败——排查这个假失败花掉的时间比
     * 截图本身的价值还高。360x640 只要 0.9 MB，足够看清两行文本的排版，
     * 而且不会触发 LMK。
     */
    private fun layoutView(view: View, w: Int = 360, h: Int = 640): Pair<Int, Int> {
        view.measure(
            View.MeasureSpec.makeMeasureSpec(w, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(h, View.MeasureSpec.EXACTLY),
        )
        view.layout(0, 0, w, h)
        return w to h
    }

    private fun launchFragment(): androidx.appcompat.app.AppCompatActivity {
        val intent = android.content.Intent(
            instrumentation.targetContext,
            com.nextransit.mrzbench.MainActivity::class.java
        )
        intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
        val monitor = instrumentation.addMonitor(
            com.nextransit.mrzbench.MainActivity::class.java.name, null, false
        )
        instrumentation.targetContext.startActivity(intent)
        val activity = monitor.waitForActivityWithTimeout(20_000) as androidx.appcompat.app.AppCompatActivity
        // 等 FragmentManager 把 PhotoMrzFragment 挂上来
        val deadline = System.currentTimeMillis() + 20_000
        while (System.currentTimeMillis() < deadline) {
            val f = activity.supportFragmentManager.findFragmentByTag("f3")
            if (f != null && f.isAdded && f.view != null) return activity
            Thread.sleep(100)
        }
        error("PhotoMrzFragment not attached within timeout")
    }

    private fun resourcesId(ctx: android.content.Context, name: String): Int =
        ctx.resources.getIdentifier(name, "id", ctx.packageName)

    private fun assertTrue(msg: String, cond: Boolean) {
        if (!cond) throw AssertionError(msg)
    }

    private fun assertFalse(msg: String, cond: Boolean) {
        if (cond) throw AssertionError(msg)
    }

    private fun assertEquals(msg: String, expected: String, actual: String) {
        if (expected != actual) throw AssertionError("$msg: expected=$expected actual=$actual")
    }

    private companion object {
        const val TAG = "MrzUi"
    }
}
