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
