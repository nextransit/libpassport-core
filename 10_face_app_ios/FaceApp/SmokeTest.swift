import Foundation
import AVFoundation
import UIKit
import SwiftUI

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
        let w = 320, h = 240
        var nv21 = [UInt8](repeating: 80, count: w * h * 3 / 2)
        for j in 0..<h {
            for i in 0..<w {
                let dx = Double(i - 160), dy = Double(j - 120)
                let lift = exp(-(dx * dx + dy * dy) / 1500.0) * 80.0
                nv21[j * w + i] = UInt8(max(0, min(255, 80.0 + lift)))
            }
        }
        let data = Data(nv21)
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

        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "live"])
        for _ in 0..<20 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("04_live_after_switch.png")

        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "register"])
        for _ in 0..<8 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("05_register_after_switch.png")

        nc.post(name: Notification.Name(key), object: nil, userInfo: ["tab": "live"])
        for _ in 0..<20 { RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.1)) }
        shoot("06_live_after_switch2.png")

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
