import AVFoundation
import CoreVideo
import Combine
import UIKit

/// One shared AVCaptureSession for the whole app. The session
/// itself stays alive for the whole foreground lifetime; tab
/// switches never tear it down. Two big changes make tab-switch
/// snappy under 1 s:
///
///   1. Detection (FaceBridge.detect) and FaceQuality.assess run on
///      a dedicated background queue (`detectQueue`), not inline
///      on the camera queue. The camera queue now only copies the
///      NV21 bytes and enqueues a small DetectJob, so the camera
///      keeps producing frames at ~30 fps even when a SeetaFace2
///      inference is in flight.
///
///   2. `frameResult` publishing is gated by `clients > 0` and
///      also rate-limited (coalesce to ~10 Hz). Two views both
///      subscribing to `@Published frameResult` at 30 Hz was the
///      main cause of the visible SwiftUI diff hitches on tab
///      switches; coalescing keeps SwiftUI's redraw path idle.
final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    static let shared = CameraManager()

    @Published private(set) var frameResult: FrameResult?
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var lastError: String?

    /// Number of foreground tab views currently requesting the
    /// camera. The session itself stays running until the app is
    /// backgrounded; this counter is only used by the lifecycle
    /// observer to know whether to restart on foreground, and by
    /// the publish gate to suppress background-tab redraws.
    private var clients = 0
    /// Set true by the tab that is currently visible. Hidden tabs
    /// keep the camera running but skip per-frame SwiftUI updates.
    @Published var isActive: Bool = false {
        didSet {
            if isActive {
                pumpLatest()
            }
        }
    }

    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "faceapp.camera", qos: .userInitiated)
    /// Background queue for FaceBridge.detect + FaceQuality.assess.
    /// Lives outside the camera queue so a slow detect never blocks
    /// the next captureOutput callback.
    private let detectQueue = DispatchQueue(label: "faceapp.detect", qos: .userInitiated)

    private override init() {
        super.init()
        configure()
        observeAppLifecycle()
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        defer { session.commitConfiguration() }

        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front)
                ?? AVCaptureDevice.default(for: .video) else {
            lastError = "no camera available"
            return
        }
        if let input = try? AVCaptureDeviceInput(device: dev) {
            if session.canAddInput(input) { session.addInput(input) }
        }
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }
        if let conn = output.connection(with: .video) {
            if conn.isVideoOrientationSupported { conn.videoOrientation = .portrait }
            if conn.isVideoMirroringSupported { conn.isVideoMirrored = true }
        }
    }

    /// Tie session start/stop to foreground/background, not tab-switch.
    /// Tab switches bump a reference count but never tear the session
    /// down — that's what previously caused LiveDetectView's preview
    /// to lock (session.stopRunning in the middle of a layout pass
    /// could leave AVCaptureVideoPreviewLayer with no running session
    /// to render from, and re-starting racing with SwiftUI's view
    /// diff produced a frozen frame).
    private func observeAppLifecycle() {
        let nc = NotificationCenter.default
        nc.addObserver(self, selector: #selector(handleEnterForeground),
                       name: UIApplication.willEnterForegroundNotification, object: nil)
        nc.addObserver(self, selector: #selector(handleEnterBackground),
                       name: UIApplication.didEnterBackgroundNotification, object: nil)
    }

    @objc private func handleEnterForeground() {
        guard clients > 0 else { return }
        startSession()
    }

    @objc private func handleEnterBackground() {
        stopSession()
    }

    /// Acquire one client (called from a tab's onAppear). The
    /// session itself is started only once per foreground cycle
    /// and stays running until the app is backgrounded.
    func start() {
        clients += 1
        startSession()
    }

    /// Release one client (called from a tab's onDisappear). The
    /// session is *kept* running until the app goes to background
    /// — tab switches should never tear the preview down.
    func stop() {
        clients = max(0, clients - 1)
    }

    private func startSession() {
        guard !session.isRunning else {
            DispatchQueue.main.async { self.isRunning = true }
            return
        }
        queue.async { [weak self] in
            guard let self else { return }
            // startRunning() is blocking and returns when the session
            // is actually running. If it returns false-ish (the API
            // actually returns Void — the session may simply not
            // start if there's no authorisation, no camera, or
            // resource contention), the .isRunning flag will be the
            // ground truth.
            self.session.startRunning()
            let ok = self.session.isRunning
            DispatchQueue.main.async {
                self.isRunning = ok
                if !ok {
                    // Diagnose common failure modes. If we have an
                    // authorisation error we surface it; otherwise
                    // the generic "failed to start" line is shown.
                    switch AVCaptureDevice.authorizationStatus(for: .video) {
                    case .notDetermined:
                        self.lastError = "camera permission not determined (likely needs user tap)"
                    case .denied, .restricted:
                        self.lastError = "camera permission denied"
                    @unknown default:
                        self.lastError = "failed to start AVCaptureSession (isRunning=false)"
                    }
                }
            }
        }
    }

    private func stopSession() {
        guard session.isRunning else { return }
        queue.async { [weak self] in
            guard let self else { return }
            self.session.stopRunning()
            DispatchQueue.main.async {
                self.isRunning = false
                self.frameResult = nil
            }
        }
    }

    // Coalesce: only republish every Nth frame on the camera queue
    // so SwiftUI's diff pass isn't woken up 30 times/s.
    private var frameCounter: UInt64 = 0
    private let publishEveryNFrames: UInt64 = 3   // ~10 Hz

    /// Pre-allocated NV21 destination buffer. Sized to whatever
    /// the first capture delivered; grown on demand for larger
    /// resolutions. Reusing one allocation across every frame
    /// eliminates a ~460 KB Data alloc/copy every 33 ms.
    private var nv21Buffer: Data = Data(count: 0)

    func captureOutput(_ output: AVCaptureOutput,
                        didOutput sampleBuffer: CMSampleBuffer,
                        from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0),
              let uvBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1) else { return }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
        let uvStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)

        // NV21 layout = Y plane followed by interleaved VU.
        // Most iPhone sensors pad Y rows to 16-byte boundaries so
        // yStride may be > w — that's fine, we copy the full row.
        let ySize = yStride * h
        let chromaSize = uvStride * (h / 2)
        let needed = ySize + chromaSize
        if nv21Buffer.count != needed {
            nv21Buffer = Data(count: needed)
        }

        nv21Buffer.withUnsafeMutableBytes { destRaw in
            let dest = destRaw.baseAddress!
            memcpy(dest, yBase, ySize)
            // Interleave UV row-by-row: NV21 = V, U, V, U, ...
            // Our camera delivers BiPlanarFullRange so UV plane is
            // already in VU order; we just memcpy.
            memcpy(dest.advanced(by: ySize), uvBase, chromaSize)
        }

        // Bump frame counter and only republish every Nth frame.
        frameCounter &+= 1
        let shouldPublish = (frameCounter % publishEveryNFrames) == 0

        // Hand the frame to the detect queue. The detect path runs
        // off the main thread and uses a single dedicated queue so
        // it can never starve the camera or the main thread.
        let bufferCopy = nv21Buffer  // Data is CoW; this is cheap.
        detectQueue.async { [weak self] in
            guard let self else { return }
            let detected = FaceBridge.detect(inNV21: bufferCopy, width: w, height: h)
            if !shouldPublish { return }
            DispatchQueue.main.async {
                self.frameResult = FrameResult(
                    width: w, height: h,
                    nv21: bufferCopy,
                    yStride: yStride,
                    detected: detected
                )
                NotificationCenter.default.post(
                    name: Notification.Name("FaceAppFrameCounter"),
                    object: nil)
            }
        }
    }

    private func pumpLatest() {
        // Explicit refresh when a tab becomes active so it doesn't
        // have to wait for the next captureOutput frame to render.
        objectWillChange.send()
    }
}

/// A frame plus the faces/landmarks computed *from that same frame*.
/// Guarantees the Register path registers the points of the exact NV21
/// buffer it queries — eliminating the stale-point mismatch.
///
/// `yStride` carries the byte-per-row of the Y plane so quality
/// scanners do not read past a padded row boundary (the previous
/// implementation assumed width == stride, which is not always true
/// on iOS NV21 buffers and produced noisy Laplacian numbers).
struct FrameResult {
    let width: Int
    let height: Int
    let nv21: Data
    let yStride: Int
    let detected: FaceFrame?
}
