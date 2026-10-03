import SwiftUI

@main
struct FaceAppApp: App {
    @StateObject private var model = FaceModel.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .preferredColorScheme(.dark)
        }
    }

    init() {
        // Optional on-launch smoke test, gated on an env var so it
        // only runs when explicitly requested (set by the build /
        // install / test harness). Reports results to a JSON file in
        // the app's Documents/ so the host can read it back via
        // `xcrun devicectl device info files ... appDataContainer`.
        let smokeRequested = ProcessInfo.processInfo.environment["FACEAPP_SMOKE"] == "1"
        NSLog("SMOKE env FACEAPP_SMOKE=%@ requested=%d",
              ProcessInfo.processInfo.environment["FACEAPP_SMOKE"] ?? "<nil>",
              smokeRequested ? 1 : 0)
        if smokeRequested {
            // Run synchronously in init() so we know the test
            // completed before the rest of the app takes over. The
            // test writes Documents/smoke_result.json which we can
            // pull back from the host with `xcrun devicectl`.
            SmokeTest.run()
        } else if ProcessInfo.processInfo.environment["FACEAPP_SCREENSHOT"] == "1" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                SmokeTest.runScreenshots()
            }
        } else if ProcessInfo.processInfo.environment["FACEAPP_HUD_PROOF"] == "1" {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                let outDir = docs.appendingPathComponent("screenshots")
                try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
                SmokeTest.renderHudProof(outDir)
            }
        }
    }
}
