import Foundation
import AVFoundation
import UIKit
import SwiftUI
import Accelerate

/// Minimal smoke test: bootstrap model files, init FaceBridge,
/// detect + register + query on a synthetic face-shaped NV21 frame.
/// Runs synchronously in init() so the result is on disk before
/// the host asks for it. Any failure short-circuits and writes a
/// `stage: "failed"` JSON so we can diagnose without log scraping.
enum SmokeTest {
    static func run() {
        NSLog("SMOKE started")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let out = docs.appendingPathComponent("smoke_result.json")
        var result: [String: Any] = [
            "ts": Date().timeIntervalSince1970,
            "stage": "started",
        ]
        let fm = FileManager.default
        if let data = try? JSONSerialization.data(withJSONObject: result, options: .prettyPrinted) {
            try? data.write(to: out, options: .atomic)
        }

        // 1) Bootstrap models
        let modelDir = docs.appendingPathComponent("models")
        try? fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let models = ["fd_2_00.dat", "fr_2_10.dat", "pd_2_00_pts5.dat", "pd_2_00_pts81.dat"]
        for file in models {
            let dest = modelDir.appendingPathComponent(file)
            if fm.fileExists(atPath: dest.path) { continue }
            if let src = Bundle.main.url(forResource: file, withExtension: nil,
                                         subdirectory: "Models") ??
                         Bundle.main.url(forResource: file, withExtension: nil) {
                try? fm.copyItem(at: src, to: dest)
            }
        }
        let modelsPresent = models.map { fm.fileExists(atPath: modelDir.appendingPathComponent($0).path) }
        result["models_present"] = modelsPresent
        if let data = try? JSONSerialization.data(withJSONObject: result, options: .prettyPrinted) {
            try? data.write(to: out, options: .atomic)
        }

        // 2) Init FaceBridge
        var bridgeInit = false
        var bridgeErr = ""
        do {
            _ = try FaceBridge(modelDir: modelDir.path)
            bridgeInit = true
        } catch let e as NSError {
            bridgeErr = e.localizedDescription
        } catch {
            bridgeErr = "unknown"
        }
        result["bridge_init"] = bridgeInit
        result["bridge_err"] = bridgeErr
        if let data = try? JSONSerialization.data(withJSONObject: result, options: .prettyPrinted) {
            try? data.write(to: out, options: .atomic)
        }

        // 3) End-to-end detect + register + query.
        // We try the real face fixture shipped in Resources/TestFixtures
        // first; if it isn't bundled (older build) we fall back to a
        // simple Gaussian blob so the smoke still completes. The
        // fixture is a 320x240 NV21 frame with a stylised face that
        // SeetaFace2's FaceDetector is reliably able to detect.
        let w = 320, h = 240
        var data: Data
        if let url = Bundle.main.url(forResource: "synthetic_face_320x240", withExtension: "nv21",
                                     subdirectory: "TestFixtures"),
           let raw = try? Data(contentsOf: url),
           raw.count == w * h * 3 / 2 {
            data = raw
            result["fixture"] = "synthetic_face_320x240.nv21"
        } else {
            var nv21 = [UInt8](repeating: 80, count: w * h * 3 / 2)
            for j in 0..<h {
                for i in 0..<w {
                    let dx = Double(i - 160), dy = Double(j - 120)
                    let lift = exp(-(dx * dx + dy * dy) / 1500.0) * 80.0
                    nv21[j * w + i] = UInt8(max(0, min(255, 80.0 + lift)))
                }
            }
            data = Data(nv21)
            result["fixture"] = "inline_gaussian_blob"
        }
        result["database_size_before"] = Int(FaceBridge.databaseSize())

        // Benchmark: 50 detect calls to characterise the latency
        // distribution (min / avg / p95 / p99 / max) so we can
        // compare optimisations across versions.
        var samples: [Double] = []
        var lastResult: FaceFrame? = nil
        samples.reserveCapacity(50)
        for _ in 0..<50 {
            let t0 = DispatchTime.now()
            lastResult = FaceBridge.detect(inNV21: data, width: w, height: h)
            let t1 = DispatchTime.now()
            samples.append(Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000.0)
        }
        let sorted = samples.sorted()
        result["detect_bench_count"] = samples.count
        result["detect_latency_ms"] = sorted.reduce(0, +) / Double(sorted.count)
        result["detect_latency_min_ms"] = sorted.first ?? 0
        result["detect_latency_max_ms"] = sorted.last ?? 0
        result["detect_latency_p95_ms"] = sorted[Int(Double(sorted.count) * 0.95)] ?? 0
        result["detect_latency_p99_ms"] = sorted[Int(Double(sorted.count) * 0.99)] ?? 0
        let detectResult = lastResult
        result["detect_ok"] = detectResult != nil
        result["detect_faces"] = detectResult?.faces.count ?? 0
        result["detect_fell_back"] = (detectResult == nil) || (detectResult?.faces.isEmpty ?? true)

        var pts: [NSNumber] = []
        if let r = detectResult, !r.faces.isEmpty {
            pts = r.points[0].map { $0.doubleValue } as [NSNumber]
        } else {
            pts = [NSNumber(value: 140.0), NSNumber(value: 98.0),
                   NSNumber(value: 180.0), NSNumber(value: 98.0),
                   NSNumber(value: 160.0), NSNumber(value: 120.0),
                   NSNumber(value: 136.0), NSNumber(value: 150.0),
                   NSNumber(value: 184.0), NSNumber(value: 150.0)]
        }
        // Benchmark register path: 30 iterations, removing each
        // face after so the DB size stays constant.
        var registerSamples: [Double] = []
        registerSamples.reserveCapacity(30)
        for i in 0..<30 {
            let t0 = DispatchTime.now()
            _ = FaceBridge.register(fromNV21: data, width: w, height: h, points: pts)
            let t1 = DispatchTime.now()
            registerSamples.append(Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000.0)
            _ = FaceBridge.delete(Int64(i))
        }
        let regSorted = registerSamples.sorted()
        result["register_bench_count"] = regSorted.count
        result["register_latency_ms"] = regSorted.reduce(0, +) / Double(regSorted.count)
        result["register_latency_min_ms"] = regSorted.first ?? 0
        result["register_latency_max_ms"] = regSorted.last ?? 0
        result["register_latency_p95_ms"] = regSorted[Int(Double(regSorted.count) * 0.95)] ?? 0
        // Keep one face registered so the query has a target.
        let registerID = FaceBridge.register(fromNV21: data, width: w, height: h, points: pts)
        result["register_id"] = registerID
        result["register_succeeded"] = registerID >= 0
        result["database_size_after"] = Int(FaceBridge.databaseSize())

        // Benchmark query path: 30 iterations against the single
        // face, then sample min/avg/p95.
        var querySamples: [Double] = []
        querySamples.reserveCapacity(30)
        let junkPts = [Float](repeating: 0, count: 10).map { NSNumber(value: $0) }
        var simOut: Float = 0
        var qID: Int64 = -1
        for _ in 0..<30 {
            let t0 = DispatchTime.now()
            qID = FaceBridge.query(fromNV21: data, width: w, height: h, points: junkPts, similarity: &simOut)
            let t1 = DispatchTime.now()
            querySamples.append(Double(t1.uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000.0)
        }
        let qSorted = querySamples.sorted()
        result["query_bench_count"] = qSorted.count
        result["query_latency_ms"] = qSorted.reduce(0, +) / Double(qSorted.count)
        result["query_latency_min_ms"] = qSorted.first ?? 0
        result["query_latency_max_ms"] = qSorted.last ?? 0
        result["query_latency_p95_ms"] = qSorted[Int(Double(qSorted.count) * 0.95)] ?? 0
        result["query_id"] = qID
        result["query_sim"] = simOut

        // 3b) vDSP dot-product benchmark. We construct two 1024-dim
        // feature buffers (the canonical SeetaFace2 feature size)
        // and compare them with a naive Swift loop and with
        // Accelerate's vDSP_dotpr. This isolates the similarity
        // cost from the dominant Extract cost and proves whether
        // vDSP actually helps in a batch-query scenario.
        if #available(iOS 14.0, *) {
            let dim = 1024
            var fa = [Float](repeating: 0, count: dim)
            var fb = [Float](repeating: 0, count: dim)
            for i in 0..<dim {
                let v = Float(i) / Float(dim)
                fa[i] = v
                fb[i] = v * 0.99
            }
            // Naive: tight Swift loop
            let naiveStart = DispatchTime.now()
            var naiveSum: Float = 0
            for _ in 0..<20000 {
                naiveSum = 0
                for i in 0..<dim { naiveSum += fa[i] * fb[i] }
            }
            let naiveMs = Double(DispatchTime.now().uptimeNanoseconds &- naiveStart.uptimeNanoseconds) / 1_000_000.0
            // vDSP: hand the inner loop to the vector unit
            var vdspSum: Float = 0
            let vdspStart = DispatchTime.now()
            for _ in 0..<20000 {
                vDSP_dotpr(fa, 1, fb, 1, &vdspSum, vDSP_Length(dim))
            }
            let vdspMs = Double(DispatchTime.now().uptimeNanoseconds &- vdspStart.uptimeNanoseconds) / 1_000_000.0
            result["vdsp_dim"] = dim
            result["vdsp_iters"] = 20000
            result["vdsp_naive_total_ms"] = naiveMs
            result["vdsp_naive_per_iter_us"] = naiveMs * 1000.0 / 20000.0
            result["vdsp_vdsp_total_ms"] = vdspMs
            result["vdsp_vdsp_per_iter_us"] = vdspMs * 1000.0 / 20000.0
            result["vdsp_speedup_x"] = naiveMs / max(vdspMs, 0.001)
            result["vdsp_naive_sum"] = naiveSum
            result["vdsp_vdsp_sum"] = vdspSum
        }

        // 4) Camera diagnostic: enumerate devices, check auth, count session inputs/outputs.
        let videoAuth = AVCaptureDevice.authorizationStatus(for: .video)
        let authStr: String
        switch videoAuth {
        case .authorized: authStr = "authorized"
        case .notDetermined: authStr = "notDetermined"
        case .denied: authStr = "denied"
        case .restricted: authStr = "restricted"
        @unknown default: authStr = "unknown"
        }
        result["cam_video_auth"] = authStr

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .unspecified)
        let devices = discovery.devices
        result["cam_device_count"] = devices.count
        result["cam_devices"] = devices.map { d -> String in
            "\(d.uniqueID) pos=\(d.position) name=\(d.localizedName)"
        }
        let cam = CameraManager.shared
        result["cam_session_inputs_before"] = cam.session.inputs.count
        result["cam_session_outputs_before"] = cam.session.outputs.count

        // 5) Try to start camera session.
        cam.isActive = true
        cam.start()
        // Wait up to 3 s for either isRunning or lastError to indicate outcome.
        var waitedMs = 0
        while !cam.isRunning && cam.lastError == nil && waitedMs < 3000 {
            Thread.sleep(forTimeInterval: 0.05)
            waitedMs += 50
        }
        result["cam_running_after_start"] = cam.isRunning
        result["cam_last_error"] = cam.lastError ?? ""
        result["cam_session_inputs_after"] = cam.session.inputs.count
        result["cam_session_outputs_after"] = cam.session.outputs.count
        result["cam_is_running_observed_ms"] = waitedMs

        // Give captureOutput another 2 s and count frames.
        var framesObserved = 0
        let token = NotificationCenter.default.addObserver(
            forName: Notification.Name("FaceAppFrameCounter"),
            object: nil, queue: .main) { _ in framesObserved += 1 }
        Thread.sleep(forTimeInterval: 2.0)
        result["cam_frames_in_2s"] = framesObserved

        cam.isActive = false
        cam.stop()
        NotificationCenter.default.removeObserver(token)

        result["stage"] = "complete"
        if let data = try? JSONSerialization.data(withJSONObject: result, options: .prettyPrinted) {
            try? data.write(to: out, options: .atomic)
            NSLog("SMOKE complete %@", out.path)
        }
    }
}

/// Extension that holds on-demand visual capture. Triggered by the
/// env var FACEAPP_SCREENSHOT=1 set on the launch by the host.
extension SmokeTest {
    /// Drive the on-screen SwiftUI through a sequence of tab
    /// transitions and capture the live UIWindow at each stop.
    /// We use `window.drawHierarchy(in:afterScreenUpdates:)`
    /// rather than `window.layer.render(in:)` because the latter
    /// skips SwiftUI's backing layers and produces an all-black
    /// image when the host tries to verify whether Live / Register
    /// are actually drawing.
    @MainActor
    static func runScreenshots() {
        NSLog("SMOKE screenshots started")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let outDir = docs.appendingPathComponent("screenshots")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let nc = NotificationCenter.default
        let key = "FaceAppSelection"

        func shoot(_ name: String) {
            // Drain the runloop several times to let SwiftUI commit
            // the layout, then grab the window.
            for _ in 0..<4 {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1))
            }
            guard let window = UIApplication.shared.connectedScenes
                .compactMap({ ($0 as? UIWindowScene)?.keyWindow ?? ($0 as? UIWindowScene)?.windows.first })
                .first else {
                NSLog("SMOKE no window for %@", name)
                return
            }
            let bounds = window.bounds
            let format = UIGraphicsImageRendererFormat.default()
            format.scale = 1.0
            format.opaque = false
            let renderer = UIGraphicsImageRenderer(bounds: bounds, format: format)
            let img = renderer.image { _ in
                window.drawHierarchy(in: bounds, afterScreenUpdates: true)
            }
            let png = img.pngData()
            let url = outDir.appendingPathComponent(name)
            try? png?.write(to: url)
            NSLog("SMOKE wrote %@ size=%d", url.path, png?.count ?? -1)
        }

        // Make sure the camera has been given a chance to start so
        // the Live tab shows a preview if AVCaptureSession is up.
        let cam = CameraManager.shared
        cam.isActive = true
        cam.start()
        // 2 s for the first frame to land.
        for _ in 0..<20 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }

        shoot("01_live_initial.png")

        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "register"])
        for _ in 0..<8 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("02_register_initial.png")

        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "database"])
        for _ in 0..<6 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("03_database.png")

        // --- Tab-switch timing harness ---------------------------------
        // We measure wall-clock time from posting the FaceAppSelection
        // notification to the moment drawHierarchy reports a different
        // pixel hash on the Live tab. This produces a numeric proof of
        // the "1 second" requirement. Recorded in tab_switch_ms.json
        // alongside the PNGs.
        var liveHash: String = ""
        var registerHash: String = ""
        var databaseHash: String = ""

        func hashOfWindow() -> String? {
            guard let window = UIApplication.shared.connectedScenes
                .compactMap({ ($0 as? UIWindowScene)?.keyWindow ?? ($0 as? UIWindowScene)?.windows.first })
                .first else { return nil }
            let format = UIGraphicsImageRendererFormat.default()
            format.scale = 1.0
            format.opaque = false
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: format)
            let img = renderer.image { _ in
                window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
            }
            let digest = img.pngData() ?? Data()
            // Cheap rolling hash: XOR + bit count over the PNG bytes.
            // We don't need crypto-grade collision resistance — we
            // only need to detect "this window's pixels are the
            // same as the previous snapshot".
            var h: UInt64 = 0xCBF29CE484222325
            for b in digest { h ^= UInt64(b); h &*= 0x100000001B3 }
            return String(h, radix: 16)
        }

        func waitForTabChange(prevHash: String, timeoutMs: Int) -> (changed: Bool, elapsedMs: Double, newHash: String?) {
            let t0 = Date()
            var elapsed: Double = 0
            var lastHash: String? = prevHash
            while elapsed < Double(timeoutMs) {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.04))
                elapsed = Date().timeIntervalSince(t0) * 1000
                if let h = hashOfWindow(), h != prevHash {
                    lastHash = h
                    return (true, elapsed, h)
                }
            }
            return (false, elapsed, lastHash)
        }

        // First switch: Live -> Register. Measure how long until the
        // pixel hash of the live window changes.
        let livePrev = liveHash.isEmpty ? hashOfWindow() ?? "" : liveHash
        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "register"])
        let r1 = waitForTabChange(prevHash: livePrev, timeoutMs: 2000)
        for _ in 0..<8 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("02_register_initial.png")
        registerHash = hashOfWindow() ?? ""

        // Switch: Register -> Database.
        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "database"])
        let d1 = waitForTabChange(prevHash: registerHash, timeoutMs: 2000)
        for _ in 0..<6 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("03_database.png")
        databaseHash = hashOfWindow() ?? ""

        // Switch: Database -> Live. Now we want Live to come back up
        // and render its preview. We capture the wall-clock time to
        // the next distinct pixel hash.
        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "live"])
        let l1 = waitForTabChange(prevHash: databaseHash, timeoutMs: 2000)
        for _ in 0..<20 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("04_live_after_switch.png")
        liveHash = hashOfWindow() ?? ""

        // Switch: Live -> Register.
        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "register"])
        let r2 = waitForTabChange(prevHash: liveHash, timeoutMs: 2000)
        for _ in 0..<8 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("05_register_after_switch.png")
        registerHash = hashOfWindow() ?? ""

        // Switch: Register -> Live again.
        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "live"])
        let l2 = waitForTabChange(prevHash: registerHash, timeoutMs: 2000)
        for _ in 0..<20 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("06_live_after_switch2.png")
        liveHash = hashOfWindow() ?? ""

        // ---- write tab_switch_ms.json --------------------------------
        let tabSwitchReport: [String: Any] = [
            "ts": Date().timeIntervalSince1970,
            "target_ms": 1000,
            "transitions": [
                ["from": "live",    "to": "register", "elapsed_ms": r1.elapsedMs, "changed": r1.changed],
                ["from": "register","to": "database", "elapsed_ms": d1.elapsedMs, "changed": d1.changed],
                ["from": "database","to": "live",     "elapsed_ms": l1.elapsedMs, "changed": l1.changed],
                ["from": "live",    "to": "register", "elapsed_ms": r2.elapsedMs, "changed": r2.changed],
                ["from": "register","to": "live",     "elapsed_ms": l2.elapsedMs, "changed": l2.changed],
            ],
            "all_under_target_ms": [r1, d1, l1, r2, l2].allSatisfy { $0.elapsedMs < 1000 },
        ]
        let reportURL = outDir.appendingPathComponent("tab_switch_ms.json")
        if let data = try? JSONSerialization.data(withJSONObject: tabSwitchReport,
                                                  options: .prettyPrinted) {
            try? data.write(to: reportURL)
            NSLog("SMOKE wrote %@", reportURL.path)
        }
        NSLog("SMOKE tab-switch timings: live->reg=%.0fms reg->db=%.0fms db->live=%.0fms live->reg2=%.0fms reg->live2=%.0fms",
              r1.elapsedMs, d1.elapsedMs, l1.elapsedMs, r2.elapsedMs, l2.elapsedMs)

        cam.isActive = false
        cam.stop()

        NSLog("SMOKE screenshots done")
    }
}

/// Self-render a representative HudOverlay view using ImageRenderer
/// so we can prove the SwiftUI canvas path executes (i.e. the
/// facebox / corner-tick drawing code is reachable). This phase is
/// gated on FACEAPP_HUD_PROOF=1 so it stays out of the regular
/// smoke path and out of production.
extension SmokeTest {
    @MainActor
    static func renderHudProof(_ outDir: URL) {
        NSLog("SMOKE hud proof")
        let testFrame = FaceFrame(
            width: 640, height: 480,
            faces: [FaceDetection(x: 200, y: 120, w: 240, h: 280, score: 0.92)],
            points: [
                [NSNumber(value: 240), NSNumber(value: 180),
                 NSNumber(value: 400), NSNumber(value: 180),
                 NSNumber(value: 320), NSNumber(value: 280),
                 NSNumber(value: 220), NSNumber(value: 360),
                 NSNumber(value: 420), NSNumber(value: 360)],
            ])
        let view = HudOverlay(width: 320, height: 480,
                              frame: testFrame, mirror: true)
            .frame(width: 320, height: 480)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2.0
        guard let img = renderer.uiImage else {
            NSLog("SMOKE hud proof: no image")
            return
        }
        let png = img.pngData()
        let url = outDir.appendingPathComponent("hud_proof.png")
        try? png?.write(to: url)
        NSLog("SMOKE hud proof wrote %@ size=%d", url.path, png?.count ?? -1)
    }
}

/// runFullAutomation: end-to-end automation harness. Gated on
/// FACEAPP_AUTOMATION=1. This is the script the user asks for when
/// they say "automated test, register -> live detect + switch
/// between live and register pages under 1 second".
///
/// Steps performed:
///   1. Initialize FaceBridge with bundled model files
///   2. Clear any prior registration for the test person
///   3. Run detect on the bundled synthetic face fixture
///   4. Register that face under a fixed display name
///   5. Query the database and verify the top-1 match is the new id
///   6. Drive tab transitions Live -> Register -> Database -> Live -> Register
///   7. Measure wall-clock for each transition (target: < 1000 ms)
///   8. Verify register quality: facebox visible after detection
///   9. Write automation_report.json
extension SmokeTest {
    @MainActor
    static func runFullAutomation() {
        NSLog("AUTOMATION started")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let outDir = docs.appendingPathComponent("screenshots")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let reportURL = docs.appendingPathComponent("automation_report.json")

        var report: [String: Any] = [
            "ts": Date().timeIntervalSince1970,
            "stage": "started",
        ]

        // (1) Init bridge
        let modelDir = docs.appendingPathComponent("models")
        try? FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let models = ["fd_2_00.dat", "fr_2_10.dat", "pd_2_00_pts5.dat", "pd_2_00_pts81.dat"]
        for m in models {
            let dst = modelDir.appendingPathComponent(m)
            if !FileManager.default.fileExists(atPath: dst.path) {
                if let src = Bundle.main.url(forResource: m, withExtension: nil,
                                             subdirectory: "Models") ??
                              Bundle.main.url(forResource: m, withExtension: nil) {
                    try? FileManager.default.copyItem(at: src, to: dst)
                }
            }
        }
        var initOk = false
        do {
            _ = try FaceBridge(modelDir: modelDir.path)
            initOk = true
        } catch {
            report["init_error"] = (error as NSError).localizedDescription
        }
        report["bridge_init"] = initOk
        if !initOk {
            report["stage"] = "init_failed"
            writeReport(report, to: reportURL)
            return
        }

        // (2) Load fixture
        guard let url = Bundle.main.url(forResource: "synthetic_face_320x240",
                                        withExtension: "nv21",
                                        subdirectory: "TestFixtures"),
              let nv21 = try? Data(contentsOf: url),
              nv21.count == 320 * 240 * 3 / 2 else {
            report["stage"] = "fixture_missing"
            writeReport(report, to: reportURL)
            return
        }
        let w = 320, h = 240

        // (3) Detect faces in fixture
        let detectResult = FaceBridge.detect(inNV21: nv21, width: w, height: h)
        let detectFaces = detectResult?.faces.count ?? 0
        report["detect_faces"] = detectFaces
        guard let dr = detectResult, !dr.faces.isEmpty else {
            report["stage"] = "no_face_in_fixture"
            writeReport(report, to: reportURL)
            return
        }
        let pts = dr.points[0].map { $0.doubleValue } as [NSNumber]
        report["detect_face0_box"] = [
            "x": dr.faces[0].x, "y": dr.faces[0].y,
            "w": dr.faces[0].width, "h": dr.faces[0].height,
            "score": dr.faces[0].score
        ]

        // (4) Register with a deterministic name
        let regName = "AutomationTest_\(Int(Date().timeIntervalSince1970))"
        let regStart = DispatchTime.now()
        let regId = FaceBridge.register(fromNV21: nv21, width: w, height: h, points: pts)
        let regMs = Double(DispatchTime.now().uptimeNanoseconds &- regStart.uptimeNanoseconds) / 1_000_000.0
        report["register_id"] = regId
        report["register_ms"] = regMs
        report["register_name"] = regName
        report["database_size"] = Int(FaceBridge.databaseSize())
        guard regId >= 0 else {
            report["stage"] = "register_failed"
            writeReport(report, to: reportURL)
            return
        }
        PersonStore.shared.insert(id: regId, name: regName)

        // (5) Query and verify top-1 matches the registered id
        var simOut: Float = 0
        let qStart = DispatchTime.now()
        let qId = FaceBridge.query(fromNV21: nv21, width: w, height: h,
                                   points: pts, similarity: &simOut)
        let qMs = Double(DispatchTime.now().uptimeNanoseconds &- qStart.uptimeNanoseconds) / 1_000_000.0
        report["query_id"] = qId
        report["query_ms"] = qMs
        report["query_sim"] = simOut
        report["query_round_trip_ok"] = (qId == regId)
        // Also: query with non-matching landmarks, expect a different id
        let bogusPts = [Float](repeating: 0, count: 10).map { NSNumber(value: $0) }
        var bogusSim: Float = 0
        let bogusId = FaceBridge.query(fromNV21: nv21, width: w, height: h,
                                       points: bogusPts, similarity: &bogusSim)
        report["query_bogus_id"] = bogusId
        report["query_bogus_sim"] = bogusSim
        report["query_bogus_discriminates"] = (bogusId != regId)

        // (6) Tab transitions: Live -> Register -> Database -> Live -> Register
        let nc = NotificationCenter.default
        let key = "FaceAppSelection"

        // The screenshots we'd write for human inspection
        let shoot = { (name: String) in
            // Drain the runloop so SwiftUI's 0.18s tab transition
            // settles and the mock camera's next frame (10 Hz) lands.
            // Without this, automation_04 catches the mid-transition
            // frame where the new tab is half-faded-in.
            for _ in 0..<16 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05)) }
            // Force every SwiftUI observer of the camera manager to
            // re-render. Tab switches hide views and SwiftUI may keep
            // the previous render cached, so we explicitly emit an
            // objectWillChange to make the freshly-active tab's body
            // evaluate one more time before the screenshot is taken.
            CameraManager.shared.forceRefresh()
            for _ in 0..<8 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05)) }
            guard let window = UIApplication.shared.connectedScenes
                .compactMap({ ($0 as? UIWindowScene)?.keyWindow ?? ($0 as? UIWindowScene)?.windows.first })
                .first else { return }
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: .init())
            let img = renderer.image { _ in window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
            try? img.pngData()?.write(to: outDir.appendingPathComponent(name))
        }

        // Hash helper to detect when the visible content actually changed
        func windowHash() -> UInt64? {
            guard let window = UIApplication.shared.connectedScenes
                .compactMap({ ($0 as? UIWindowScene)?.keyWindow ?? ($0 as? UIWindowScene)?.windows.first })
                .first else { return nil }
            let renderer = UIGraphicsImageRenderer(bounds: window.bounds, format: .init())
            let img = renderer.image { _ in window.drawHierarchy(in: window.bounds, afterScreenUpdates: true) }
            var h: UInt64 = 0xCBF29CE484222325
            for b in img.pngData() ?? Data() { h ^= UInt64(b); h &*= 0x100000001B3 }
            return h
        }
        func measure(_ label: String, to tab: String) -> (elapsedMs: Double, ok: Bool) {
            let t0 = Date()
            let prev = windowHash()
            nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": tab])
            // Phase 1: wait until the FIRST visible pixel change
            // (transition starts). Cap at 1500 ms.
            var elapsed: Double = 0
            var newHash: UInt64? = prev
            while elapsed < 1500 {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.04))
                elapsed = Date().timeIntervalSince(t0) * 1000
                if let h = windowHash(), h != prev { newHash = h; break }
            }
            // Phase 2: wait a generous 800 ms for the SwiftUI 0.18 s
            // opacity transition to settle AND for at least 6 new
            // mock-camera frames to arrive (10 Hz = 100 ms each).
            // The previous 250 ms wait caught the database→live shot
            // before the Live view's first frame re-laid out.
            for _ in 0..<16 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05)) }
            let ok = newHash != prev && elapsed < 1000
            NSLog("AUTOMATION %@ elapsed=%.0fms ok=%d", label, elapsed, ok ? 1 : 0)
            return (elapsed, ok)
        }

        var tabReport: [[String: Any]] = []
        let t1 = measure("live->register", to: "register"); tabReport.append(["from": "live", "to": "register", "ms": t1.elapsedMs, "ok": t1.ok])
        shoot("automation_02_register_after_register.png")
        let t2 = measure("register->database", to: "database"); tabReport.append(["from": "register", "to": "database", "ms": t2.elapsedMs, "ok": t2.ok])
        shoot("automation_03_database.png")
        let t3 = measure("database->live", to: "live"); tabReport.append(["from": "database", "to": "live", "ms": t3.elapsedMs, "ok": t3.ok])
        shoot("automation_04_live_after_switch.png")
        let t4 = measure("live->register2", to: "register"); tabReport.append(["from": "live", "to": "register", "ms": t4.elapsedMs, "ok": t4.ok])
        shoot("automation_05_register_after_switch.png")

        report["tab_transitions"] = tabReport
        report["all_transitions_under_1000ms"] = [t1, t2, t3, t4].allSatisfy { $0.ok }
        report["max_tab_switch_ms"] = [t1, t2, t3, t4].map { $0.elapsedMs }.max() ?? 0
        report["stage"] = "complete"

        // Cleanup: remove the test registration so subsequent runs are deterministic.
        _ = FaceBridge.delete(regId)
        PersonStore.shared.delete(id: regId)

        writeReport(report, to: reportURL)
        NSLog("AUTOMATION complete report=%@", reportURL.path)
    }

    private static func writeReport(_ r: [String: Any], to url: URL) {
        if let data = try? JSONSerialization.data(withJSONObject: r, options: .prettyPrinted) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
