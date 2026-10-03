import AVFoundation
import SwiftUI

struct LiveDetectView: View {
    @EnvironmentObject var model: FaceModel
    @ObservedObject private var cam: CameraManager = .shared
    @State private var faceCount: Int = 0
    @State private var fps: Double = 0
    @State private var showOverlay: Bool = true
    /// Rolling window of frame timestamps for an honest FPS readout.
    @State private var fpsTimestamps: [Double] = []

    var body: some View {
        ZStack {
            CameraPreview(session: cam.session)
                .ignoresSafeArea()
                .onTapGesture { showOverlay.toggle() }

            // Show the latest NV21 frame as a SwiftUI Image so the
            // mock-camera simulator path actually displays the face.
            // On a real device the AVCaptureVideoPreviewLayer does
            // this for us and the Image stays drawn out (Camerapreview
            // wins the Z-order). On the simulator there is no preview
            // layer content so we fall back to a manual render.
            if let frame = cam.frameResult {
                NV21Preview(nv21: frame.nv21,
                            width: frame.width,
                            height: frame.height)
                    .ignoresSafeArea()
                    .opacity(cam.isRunning ? 1 : 0)
            }

            // Placeholder visible whenever the camera session is not
            // actually running. This covers (a) the iOS Simulator
            // where AVCaptureDevice.DiscoverySession returns 0 devices
            // AND we have no frame yet, and (b) the moment after launch
            // before captureOutput fires its first frame. Without this
            // the central area shows a black void, which the operator
            // reads as a bug.
            if !cam.isRunning {
                CameraUnavailablePlaceholder(reason: cam.lastError ?? "camera session is not running")
                    .allowsHitTesting(false)
            }

            if showOverlay {
                HudOverlay(width: 320, height: 480,
                           frame: cam.frameResult?.detected,
                           mirror: true)
                    .ignoresSafeArea()

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Label("LIVE", systemImage: "circle.fill")
                            .labelStyle(.titleAndIcon)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.cyan)
                        Spacer()
                        statBadge
                    }
                    .padding(.horizontal)
                    .padding(.top, 8)

                    if let err = cam.lastError {
                        Text("camera: \(err)")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.red)
                            .padding(.horizontal)
                    } else if faceCount == 0 && cam.isRunning {
                        Text("Scanning for faces…")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.6))
                            .padding(.horizontal)
                    }

                    Spacer()

                    Text("Tap to \(showOverlay ? "hide" : "show") overlay")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.45))
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.bottom, 8)
                }
            }
        }
        .onAppear {
            cam.isActive = true
            cam.start()
            fpsTimestamps.removeAll()
        }
        .onDisappear {
            cam.isActive = false
            cam.stop()
        }
        .onReceive(cam.$frameResult.compactMap { $0 }) { frame in
            let n = frame.detected?.faces.count ?? 0
            if n != faceCount {
                faceCount = n
            }
            // FPS rolling average over the last second.
            let now = CACurrentMediaTime()
            fpsTimestamps.append(now)
            fpsTimestamps = fpsTimestamps.filter { now - $0 < 1.0 }
            let newFps = Double(fpsTimestamps.count)
            if abs(newFps - fps) > 0.5 {
                fps = newFps
            }
        }
    }

    /// Compact status pill that shows face count + capture rate so
    /// the operator always knows the camera is live and at what FPS.
    @ViewBuilder
    private var statBadge: some View {
        HStack(spacing: 8) {
            faceCountPill
            fpsPill
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(Capsule().fill(Color.black.opacity(0.55)))
    }

    private var faceCountPill: some View {
        HStack(spacing: 4) {
            Image(systemName: faceCount == 0 ? "person.crop.circle.badge.questionmark" : "person.crop.circle.fill")
                .foregroundStyle(faceCount > 0 ? .green : .yellow)
            Text("\(faceCount) face\(faceCount == 1 ? "" : "s")")
                .font(.caption2.monospaced())
                .foregroundStyle(.white)
        }
    }

    private var fpsPill: some View {
        HStack(spacing: 4) {
            Image(systemName: "speedometer")
                .foregroundStyle(fps >= 8 ? .green : (fps >= 4 ? .yellow : .red))
            Text(String(format: "%.0f fps", fps))
                .font(.caption2.monospaced())
                .foregroundStyle(.white)
        }
    }
}

/// SwiftUI wrapper around AVCaptureVideoPreviewLayer. We
/// intentionally use `makeUIView` only — never `updateUIView` —
/// because rebinding `layer.session = session` on every SwiftUI
/// re-render is exactly what previously caused the preview to
/// freeze when switching back from Register to Live.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    func makeUIView(context: Context) -> PreviewView { PreviewView(session: session) }
    func updateUIView(_ uiView: PreviewView, context: Context) {
        // No-op: session is the shared singleton and is owned by
        // CameraManager, not by this view. Re-binding would race
        // with the AVCaptureSession state machine.
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    init(session: AVCaptureSession) {
        super.init(frame: .zero)
        guard let layer = self.layer as? AVCaptureVideoPreviewLayer else { return }
        layer.session = session
        layer.videoGravity = .resizeAspectFill
    }
    required init?(coder: NSCoder) { fatalError() }
}

/// HUD overlay drawn on top of the camera preview. Wrapped in a
/// `.equatable()`-friendly view so SwiftUI's diff can skip it when
/// the underlying FrameResult hasn't actually changed.
struct HudOverlay: View {
    let width: CGFloat
    let height: CGFloat
    let frame: FaceFrame?
    let mirror: Bool

    var body: some View {
        Canvas { ctx, size in
            guard let frame = frame, width > 0, height > 0 else { return }
            let sx = size.width / CGFloat(frame.width)
            let sy = size.height / CGFloat(frame.height)
            for (i, face) in frame.faces.enumerated() {
                let r = CGRect(
                    x: CGFloat(face.x) * sx,
                    y: CGFloat(face.y) * sy,
                    width: CGFloat(face.width) * sx,
                    height: CGFloat(face.height) * sy
                )
                ctx.stroke(Path(r), with: .color(.cyan), lineWidth: 3)
                let t: CGFloat = 16
                let path = Path { p in
                    p.move(to: CGPoint(x: r.minX, y: r.minY))
                    p.addLine(to: CGPoint(x: r.minX + t, y: r.minY))
                    p.move(to: CGPoint(x: r.minX, y: r.minY))
                    p.addLine(to: CGPoint(x: r.minX, y: r.minY + t))
                    p.move(to: CGPoint(x: r.maxX, y: r.minY))
                    p.addLine(to: CGPoint(x: r.maxX - t, y: r.minY))
                    p.move(to: CGPoint(x: r.maxX, y: r.minY))
                    p.addLine(to: CGPoint(x: r.maxX, y: r.minY + t))
                    p.move(to: CGPoint(x: r.minX, y: r.maxY))
                    p.addLine(to: CGPoint(x: r.minX + t, y: r.maxY))
                    p.move(to: CGPoint(x: r.minX, y: r.maxY))
                    p.addLine(to: CGPoint(x: r.minX, y: r.maxY - t))
                    p.move(to: CGPoint(x: r.maxX, y: r.maxY))
                    p.addLine(to: CGPoint(x: r.maxX - t, y: r.maxY))
                    p.move(to: CGPoint(x: r.maxX, y: r.maxY))
                    p.addLine(to: CGPoint(x: r.maxX, y: r.maxY - t))
                }
                ctx.stroke(path, with: .color(.cyan), lineWidth: 6)
                if frame.points.indices.contains(i) {
                    let pts = frame.points[i]
                    var j = 0
                    while j + 1 < pts.count {
                        let cx = CGFloat(pts[j].doubleValue) * sx
                        let cy = CGFloat(pts[j + 1].doubleValue) * sy
                        ctx.fill(Path(ellipseIn: CGRect(x: cx - 3, y: cy - 3, width: 6, height: 6)),
                                 with: .color(.pink))
                        j += 2
                    }
                }
            }
        }
    }
}

/// Placeholder shown when no camera frames are available. We render
/// a dark gradient with a centred icon + message so the operator
/// doesn't read the absence of a preview layer as a SwiftUI hang.
/// This view never blocks taps — the tap-to-toggle overlay gesture
/// keeps working above it.
struct CameraUnavailablePlaceholder: View {
    let reason: String

    var body: some View {
        ZStack {
            // A clearly-visible dark gradient so the operator doesn't
            // read the empty preview as "the app is hung". 0.18..0.32
            // is well above the perceptual threshold on a real OLED
            // display and renders as ~46..82 RGB on a 0-255 scale.
            LinearGradient(
                colors: [Color(white: 0.18), Color(white: 0.32), Color(white: 0.18)],
                startPoint: .top, endPoint: .bottom
            )
            VStack(spacing: 16) {
                ZStack {
                    Circle()
                        .fill(Color.cyan.opacity(0.12))
                        .frame(width: 120, height: 120)
                    Image(systemName: "camera.metering.unknown")
                        .font(.system(size: 64, weight: .light))
                        .foregroundStyle(.cyan.opacity(0.9))
                }
                Text("Camera unavailable")
                    .font(.system(.title3, design: .monospaced).bold())
                    .foregroundStyle(.white)
                Text(reason)
                    .font(.caption.monospaced())
                    .foregroundStyle(.white.opacity(0.75))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                Text("Live preview requires a physical iPhone.\nThe iOS Simulator does not expose AVCaptureDevice.")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.55))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
                    .padding(.top, 6)
            }
        }
    }
}

/// SwiftUI view that renders an NV21 buffer as a stretched Image.
/// Used only as a fallback when the AVCaptureVideoPreviewLayer
/// has no input (i.e. iOS Simulator with the mock camera). On a
/// real device, CameraPreview wins the Z-order and this view is
/// behind it, effectively hidden.
struct NV21Preview: View {
    let nv21: Data
    let width: Int
    let height: Int

    var body: some View {
        if let img = Self.makeImage(nv21: nv21, w: width, h: height) {
            Image(uiImage: img)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            Color.black
        }
    }

    /// NV21 → CGImage (RGB) → UIImage. Mirrors the conversion in
    /// face_pipeline.mm but at Swift level. We deliberately avoid
    /// Accelerate here — this runs once per published frame at
    /// ~10 Hz and the cost is dominated by UIGraphics, not the
    /// pixel math.
    private static func makeImage(nv21: Data, w: Int, h: Int) -> UIImage? {
        guard nv21.count == w * h * 3 / 2 else { return nil }
        let ySize = w * h
        let yp = [UInt8](nv21.prefix(ySize))
        let vp = [UInt8](nv21.suffix(ySize / 2))
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        for j in 0..<h {
            for i in 0..<w {
                let Y = Int(yp[j * w + i])
                let vui = (j >> 1) * w + (i & ~1)
                let V = Int(vp[vui + 0]) - 128
                let U = Int(vp[vui + 1]) - 128
                let r = Y + ((1471 * V) >> 10)
                let g = Y - ((352 * U + 731 * V) >> 10)
                let b = Y + ((1815 * U) >> 10)
                let off = (j * w + i) * 4
                pixels[off + 0] = UInt8(max(0, min(255, r)))
                pixels[off + 1] = UInt8(max(0, min(255, g)))
                pixels[off + 2] = UInt8(max(0, min(255, b)))
                pixels[off + 3] = 255
            }
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &pixels,
                            width: w, height: h,
                            bitsPerComponent: 8,
                            bytesPerRow: w * 4,
                            space: cs,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        guard let cg = ctx?.makeImage() else { return nil }
        return UIImage(cgImage: cg)
    }
}
