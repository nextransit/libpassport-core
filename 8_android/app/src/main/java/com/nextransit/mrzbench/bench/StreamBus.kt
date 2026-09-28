package com.nextransit.mrzbench.bench

import java.util.concurrent.CopyOnWriteArrayList

/**
 * Tiny pub/sub used by RunFragment to push per-image log lines to
 * StreamFragment without holding a fragment reference. CopyOnWrite so
 * additions during iteration (publish on Main, consume on Main) are
 * safe.
 */
object StreamBus {
    private val subs = CopyOnWriteArrayList<(String) -> Unit>()

    fun subscribe(fn: (String) -> Unit): () -> Unit {
        subs.add(fn)
        return { subs.remove(fn) }
    }

    fun publish(line: String) {
        for (s in subs) s(line)
    }
}
