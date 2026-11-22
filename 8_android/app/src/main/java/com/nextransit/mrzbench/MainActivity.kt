package com.nextransit.mrzbench

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.google.android.material.tabs.TabLayout
import com.nextransit.mrzbench.bench.Benchmark
import com.nextransit.mrzbench.bench.HeadlessRunner
import com.nextransit.mrzbench.jni.MrzNative
import com.nextransit.mrzbench.photo.PaddleOcrAndroid
import com.nextransit.mrzbench.photo.PhotoMrzPipeline
import com.nextransit.mrzbench.ui.RunFragment
import com.nextransit.mrzbench.ui.StreamFragment
import com.nextransit.mrzbench.ui.CompareFragment
import com.nextransit.mrzbench.ui.PhotoMrzFragment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var tabs: TabLayout
    private lateinit var status: TextView
    private val fragments = listOf(
        RunFragment(),
        StreamFragment(),
        CompareFragment(),
        PhotoMrzFragment(),
    )
    private val titles = listOf("RUN", "STREAM", "COMPARE", "PHOTO -> MRZ")

    private var benchReceiver: BroadcastReceiver? = null
    private var photoBatchReceiver: BroadcastReceiver? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tabs = findViewById(R.id.tabs)
        status = findViewById(R.id.status_bar)

        for (t in titles) tabs.addTab(tabs.newTab().setText(t))

        // Pre-attach all tabs so back-stack isn't a question. Show
        // only one at a time to keep memory flat (each fragment is
        // tiny but <5 MB matters on Android 9 / 2 GB-class handsets).
        supportFragmentManager.beginTransaction().apply {
            fragments.forEachIndexed { i, f ->
                add(R.id.host, f, "f$i")
                hide(f)
            }
        }.commit()
        show(0)

        tabs.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab) = show(tab.position)
            override fun onTabUnselected(tab: TabLayout.Tab) {}
            override fun onTabReselected(tab: TabLayout.Tab) {}
        })

        // Print native build hash to the status bar so the on-device vs
        // host traceability is one tap away (see Compare tab too).
        status.text = try {
            "native: ${MrzNative.versionString()}"
        } catch (t: Throwable) { "native: (unavailable: ${t.javaClass.simpleName})" }

        // Pre-warm the PaddleOcr ORT session on a worker thread so the
        // first photo does not pay 2-3s of model copy + session build.
        PaddleOcrAndroid.preload(applicationContext)
        PhotoMrzPipeline.warm(applicationContext)

        // Test-driver hook: a host broadcast (e.g. `adb shell am
        // broadcast -a com.nextransit.mrzbench.action.RUN_BENCH -p
        // com.nextransit.mrzbench --es backend both`) kicks a batch
        // run without needing tap coordinates. We register at the
        // activity level so it survives fragment-view recreation.
        val filter = IntentFilter("com.nextransit.mrzbench.action.RUN_BENCH")
        benchReceiver = object : BroadcastReceiver() {
            override fun onReceive(ctx: Context?, intent: Intent?) {
                Log.i("MrzBench", "RUN_BENCH broadcast received: " + intent?.extras)
                val backend = intent?.getStringExtra("backend") ?: "both"
                val backends = when (backend.lowercase()) {
                    "trad", "template" -> listOf(Benchmark.Backend.TRAD)
                    "cnn"              -> listOf(Benchmark.Backend.CNN)
                    "crnn"             -> listOf(Benchmark.Backend.CRNN)
                    "fuse"             -> listOf(Benchmark.Backend.FUSE)
                    else               -> listOf(Benchmark.Backend.TRAD, Benchmark.Backend.CNN)
                }
                // Also flip the on-screen toggle so the UI matches.
                val runFrag = fragments[0] as RunFragment
                runFrag.setBackendSelection(backend)
                // HeadlessRunner runs independently of fragment view
                // lifecycle; it writes metrics/records/report to the
                // external results dir so they survive even if the UI
                // is hidden when the run completes.
                val ext = getExternalFilesDir("corpus")
                if (ext == null) {
                    Toast.makeText(this@MainActivity, "no external dir", Toast.LENGTH_SHORT).show()
                    return
                }
                val json = java.io.File(ext, "corpus.json")
                if (!json.exists()) {
                    Toast.makeText(this@MainActivity, "no corpus.json at " + json.absolutePath, Toast.LENGTH_LONG).show()
                    return
                }
                val imgDir = java.io.File(ext, "img").let { if (it.exists()) it else ext }
                setStatus("host command: run backend=" + backend + " (n=" + backends.size + ")")
                val profile = intent?.getStringExtra("profile") == "1"
                com.nextransit.mrzbench.jni.MrzNative.setProfiling(profile)
                val parallel = intent?.getStringExtra("parallel") == "1"
                com.nextransit.mrzbench.jni.MrzNative.setParallel(parallel)
                Toast.makeText(this@MainActivity, "RUN_BENCH " + backend, Toast.LENGTH_SHORT).show()
                Log.i("MrzBench", "HeadlessRunner.run start backends=" + backends.size + " imgsDir=" + imgDir.absolutePath + " jsonpath=" + json.absolutePath)
                HeadlessRunner.run(
                    ctx = applicationContext,
                    backends = backends,
                    imagesDir = imgDir,
                    corpusJson = json,
                    onComplete = { totals ->
                        val msg = totals.entries.joinToString { (b, m) ->
                            b.displayName + "=" + "%.2f".format(m.l2Acc * 100) + "%" }
                        runStatusSink?.invoke("done: " + msg)
                        Log.i("MrzBench", "HeadlessRunner.onComplete: " + msg)
                    },
                )
            }
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(benchReceiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(benchReceiver, filter)
        }

        // PHOTO->MRZ batch test driver: allows ADB to trigger a photo
        // batch test with a specific algorithm and image source without
        // UI interaction. Results are written to a JSON file in the
        // external files dir so the host can pull them via adb.
        //
        // Usage:
        //   adb shell am broadcast \
        //     -a com.nextransit.mrzbench.action.PHOTO_BATCH \
        //     -p com.nextransit.mrzbench \
        //     --es method crnn \
        //     --es source tmp/PassPortPhoto
        //
        // method: auto/cnn/crnn/fuse/traditional/tesseract/paddle/indra/mlkit
        // source: all/assets/pic/Photo/tmp/PassPortPhoto
        //
        // Results are written to:
        //   /storage/emulated/0/Android/data/com.nextransit.mrzbench/files/photo_batch_result.json
        val photoFilter = IntentFilter("com.nextransit.mrzbench.action.PHOTO_BATCH")
        photoBatchReceiver = object : BroadcastReceiver() {
            override fun onReceive(ctx: Context?, intent: Intent?) {
                val method = intent?.getStringExtra("method") ?: "auto"
                val source = intent?.getStringExtra("source") ?: "all"
                val loader = intent?.getStringExtra("loader")
                Log.i("MrzBench", "PHOTO_BATCH method=$method source=$source loader=$loader")
                Toast.makeText(this@MainActivity, "PHOTO_BATCH $method/$source/$loader", Toast.LENGTH_SHORT).show()

                val photoFrag = fragments[3] as PhotoMrzFragment
                photoFrag.runBatchFromBroadcast(method, source, loader)
            }
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(photoBatchReceiver, photoFilter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            registerReceiver(photoBatchReceiver, photoFilter)
        }

        // Push headless-run progress events into the visible fragment
        // (best-effort — the headless run completes regardless of UI).
        setRunStatusSink { s -> runOnUiThread { setStatus(s) } }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (tabs.selectedTabPosition > 0) {
                    tabs.getTabAt(0)?.select()
                } else {
                    finish()
                }
            }
        })
    }

    override fun onDestroy() {
        benchReceiver?.let { unregisterReceiver(it) }
        benchReceiver = null
        photoBatchReceiver?.let { unregisterReceiver(it) }
        photoBatchReceiver = null
        super.onDestroy()
    }

    private fun show(idx: Int) {
        val tx = supportFragmentManager.beginTransaction()
        fragments.forEachIndexed { i, f -> if (i == idx) tx.show(f) else tx.hide(f) }
        tx.commit()
    }

    fun setStatus(s: String) { status.text = s }

    /** Receiver can update the status bar even when no fragment owns it. */
    private var runStatusSink: ((String) -> Unit)? = null
    fun setRunStatusSink(fn: (String) -> Unit) { runStatusSink = fn }
}
