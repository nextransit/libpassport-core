import AVFoundation
import UIKit
import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

/// RegisterView — guides the operator through a face-registration
/// flow with explicit state feedback at every step.
///
/// UX rules (2026-10-03):
///   - the display-name TextField highlights red when empty
///   - the "From Camera" button shows a *reason label* above it
///     whenever it is disabled, so the operator never has to guess
///   - status text is colour-coded by category (green ok, orange
///     hint, red error); we never display a bare "failed" line
///   - a transient success state triggers haptic feedback
struct RegisterView: View {
    @EnvironmentObject var model: FaceModel
    @ObservedObject private var cam: CameraManager = .shared
    @State private var name: String = ""
    @State private var status: RegisterStatus = .hint("Aim camera at a single, well-lit face, then Register.")
    @State private var quality: FaceQuality.Verdict?
    @State private var pickerItem: PhotosPickerItem?
    @State private var successTick: Date? = nil  // for haptic + checkmark

    var body: some View {
        ZStack {
            CameraPreview(session: cam.session)
                .ignoresSafeArea()

            HudOverlay(width: 320, height: 480,
                       frame: cam.frameResult?.detected,
                       mirror: true)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 8) {
                Text("REGISTER NEW FACE")
                    .font(.system(.title3, design: .monospaced))
                    .foregroundStyle(.cyan)

                TextField("Display name", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    .submitLabel(.done)
                    .onSubmit { UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil) }

                Text(status.text)
                    .font(.footnote)
                    .foregroundStyle(status.color)
                    .animation(.easeInOut(duration: 0.15), value: status)

                if let q = quality, !q.pass {
                    Text(q.summary)
                        .font(.footnote)
                        .foregroundStyle(.orange)
                }

                if let err = cam.lastError {
                    cameraErrorBlock(err)
                }

                // Reason label for the disabled From Camera button.
                if !canRegisterFromCamera {
                    Text(cameraButtonHint)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 2)
                }

                HStack(spacing: 12) {
                    Button {
                        registerFromCamera()
                    } label: {
                        Label("From Camera", systemImage: "camera.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canRegisterFromCamera)

                    PhotosPicker(selection: $pickerItem, matching: .images) {
                        Label("From Image", systemImage: "photo.fill")
                    }
                    .buttonStyle(.bordered)
                    .tint(.pink)
                    .disabled(name.isEmpty)
                }

                if let tick = successTick, Date().timeIntervalSince(tick) < 1.5 {
                    Label("Saved", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                        .font(.subheadline.bold())
                        .transition(.scale.combined(with: .opacity))
                }

                Spacer()
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(LinearGradient(colors: [.black.opacity(0.6), .clear],
                                       startPoint: .top, endPoint: .bottom))
        }
        .onAppear {
            cam.isActive = true
            cam.start()
        }
        .onDisappear {
            cam.isActive = false
            cam.stop()
        }
        .background(
            Color.clear
                .contentShape(Rectangle())
                .onTapGesture {
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
        )
        .onChange(of: pickerItem) { item in
            guard let item = item else { return }
            Task {
                let maybeData = (try? await item.loadTransferable(type: Data.self)) ?? nil
                if let data = maybeData,
                   let ui = UIImage(data: data),
                   let cg = ui.cgImage {
                    registerFromCGImage(cg)
                }
            }
        }
        .onReceive(cam.$frameResult.compactMap { $0 }) { frame in
            quality = FaceQuality.assess(frame: frame)
        }
    }

    /// Status as a typed enum so the colour is decided in one place.
    private enum RegisterStatus: Equatable {
        case hint(String)
        case ok(String)
        case warn(String)
        case err(String)

        var text: String {
            switch self {
            case .hint(let s), .ok(let s), .warn(let s), .err(let s):
                return s
            }
        }
        var color: Color {
            switch self {
            case .hint: return .white.opacity(0.85)
            case .ok: return .green
            case .warn: return .orange
            case .err: return .red
            }
        }
    }

    /// Friendly explanation for *why* the From Camera button is
    /// disabled right now. The operator never has to guess.
    private var cameraButtonHint: String {
        if name.isEmpty { return "Enter a display name above to enable." }
        guard let frame = cam.frameResult, let det = frame.detected else {
            return "Waiting for the camera to start…"
        }
        if det.faces.isEmpty { return "No face detected — aim the camera at one face." }
        if det.faces.count > 1 { return "Multiple faces in frame — keep only one." }
        if (det.points.first ?? []).count < 10 {
            return "Face detected, but landmarks incomplete — try moving closer."
        }
        if let q = quality, !q.pass {
            return "Quality too low: \(q.summary)"
        }
        return "Ready to register."
    }

    /// When the camera fails to start, surface a one-tap "Open
    /// Settings" link so the operator can grant permission without
    /// having to leave the app and remember to come back.
    @ViewBuilder
    private func cameraErrorBlock(_ msg: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("camera: \(msg)")
                .font(.footnote)
                .foregroundStyle(.red)
            if msg.lowercased().contains("permission") ||
               msg.lowercased().contains("denied") {
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                .font(.caption)
                .buttonStyle(.bordered)
            }
        }
    }

    /// Enable From Camera as soon as we have a face; missing
    /// landmarks fall back to face-box estimation.
    private var canRegisterFromCamera: Bool {
        guard !name.isEmpty, let frame = cam.frameResult,
              let det = frame.detected, !det.faces.isEmpty else { return false }
        if det.faces.count > 1 { return false }
        return true
    }

    private func registerFromCamera() {
        guard let frame = cam.frameResult, let det = frame.detected else {
            status = .warn("Waiting for the first camera frame…")
            return
        }
        guard det.faces.count == 1 else {
            status = .warn("Multiple faces in frame — keep only one.")
            return
        }
        var pts = det.points.first ?? []
        var synthUsed = false
        if pts.count < 10 {
            let face = det.faces[0]
            let cx = Double(face.x + face.width / 2)
            let cy = Double(face.y + face.height / 2)
            let w = Double(max(face.width, 1))
            let h = Double(max(face.height, 1))
            let synth: [Double] = [
                cx - w * 0.18, cy - h * 0.18,
                cx + w * 0.18, cy - h * 0.18,
                cx,            cy,
                cx - w * 0.22, cy + h * 0.30,
                cx + w * 0.22, cy + h * 0.30,
            ]
            pts = synth.map { NSNumber(value: $0) }
            synthUsed = true
            status = .warn("Landmarks incomplete — using face-box estimate.")
        }
        if let q = quality, !q.pass {
            status = .warn("Quality: \(q.summary) — registering anyway.")
        }
        let points = pts.map { $0.doubleValue } as [NSNumber]
        let registerName = name.trimmingCharacters(in: .whitespaces)
        guard !registerName.isEmpty else {
            status = .err("Display name is empty.")
            return
        }
        status = .hint("Registering…")
        DispatchQueue.global(qos: .userInitiated).async {
            let id = FaceBridge.register(fromNV21: frame.nv21,
                                         width: frame.width,
                                         height: frame.height,
                                         points: points)
            DispatchQueue.main.async {
                if id >= 0 {
                    PersonStore.shared.insert(id: id, name: registerName)
                    let msg = synthUsed
                        ? "Registered id=\(id) (estimated landmarks)."
                        : "Registered id=\(id) · total=\(FaceBridge.databaseSize())."
                    status = .ok(msg)
                    UINotificationFeedbackGeneratorWrapper.success()
                    successTick = Date()
                    model.refreshDatabaseSize()
                } else {
                    status = .err("Register failed — SeetaFace2 rejected the face; try a different angle or distance.")
                }
            }
        }
    }

    private func registerFromCGImage(_ cg: CGImage) {
        let w = cg.width, h = cg.height
        let maxDim = 640
        let scale = min(1.0, Double(maxDim) / Double(max(w, h)))
        let sw = max(1, Int(Double(w) * scale))
        let sh = max(1, Int(Double(h) * scale))
        let cs = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = sw * 4
        var pixels = [UInt8](repeating: 0, count: sw * sh * 4)
        guard let ctx = CGContext(data: &pixels, width: sw, height: sh,
                                  bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                  space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            status = .err("Could not decode the picked image.")
            return
        }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: sw, height: sh))
        let ySize = sw * sh
        var nv21 = Data(count: ySize + (sw * sh / 2))
        nv21.withUnsafeMutableBytes { destRaw in
            let d = destRaw.baseAddress!
            for j in 0..<sh {
                for i in 0..<sw {
                    let off = (j * sw + i) * 4
                    let r = Int(pixels[off + 0])
                    let g = Int(pixels[off + 1])
                    let b = Int(pixels[off + 2])
                    let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
                    d.storeBytes(of: UInt8(max(0, min(255, y))),
                                 toByteOffset: j * sw + i,
                                 as: UInt8.self)
                }
            }
            var dst = ySize
            for j in stride(from: 0, to: sh, by: 2) {
                for i in stride(from: 0, to: sw, by: 2) {
                    let off = (j * sw + i) * 4
                    let r = Int(pixels[off + 0])
                    let g = Int(pixels[off + 1])
                    let b = Int(pixels[off + 2])
                    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128
                    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128
                    d.storeBytes(of: UInt8(max(0, min(255, v))),
                                 toByteOffset: dst,
                                 as: UInt8.self)
                    d.storeBytes(of: UInt8(max(0, min(255, u))),
                                 toByteOffset: dst + 1,
                                 as: UInt8.self)
                    dst += 2
                }
            }
        }
        let registerName = name.trimmingCharacters(in: .whitespaces)
        guard !registerName.isEmpty else {
            status = .err("Display name is empty.")
            return
        }
        status = .hint("Registering image…")
        DispatchQueue.global(qos: .userInitiated).async {
            let det = FaceBridge.detect(inNV21: nv21, width: sw, height: sh)
            DispatchQueue.main.async {
                guard let det = det, !det.faces.isEmpty else {
                    status = .err("No face found in the picked image — try a clearer photo.")
                    return
                }
                if det.faces.count > 1 {
                    status = .err("Multiple faces in image — pick a single-face photo.")
                    return
                }
                let pts = det.points[0].map { $0.doubleValue } as [NSNumber]
                let id = FaceBridge.register(fromNV21: nv21, width: sw, height: sh, points: pts)
                if id >= 0 {
                    PersonStore.shared.insert(id: id, name: registerName)
                    status = .ok("Registered id=\(id) · total=\(FaceBridge.databaseSize()).")
                    UINotificationFeedbackGeneratorWrapper.success()
                    successTick = Date()
                    model.refreshDatabaseSize()
                } else {
                    status = .err("Register failed — SeetaFace2 rejected the image.")
                }
            }
        }
    }
}

/// Thin Swift wrapper so views don't import CoreHaptics directly.
enum UINotificationFeedbackGeneratorWrapper {
    @MainActor
    static func success() {
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(.success)
    }
    @MainActor
    static func warning() {
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(.warning)
    }
    @MainActor
    static func error() {
        let g = UINotificationFeedbackGenerator()
        g.prepare()
        g.notificationOccurred(.error)
    }
}
