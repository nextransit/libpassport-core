import Foundation
import CoreVideo

/// Registration-quality gate. Runs purely on the frame's Y plane (the
/// luma channel), so it is fast and needs no extra dependencies:
///   - sharpness: 3x3 Laplacian variance on the Y plane (blur check)
///   - coverage:   the detected face box must occupy enough of the frame
///   - integrity:  all five landmarks must lie inside the frame and
///                 have plausible geometry (left/right eyes apart, etc.)
enum FaceQuality {

    struct Verdict {
        let pass: Bool
        let reasons: [String]
        var summary: String { reasons.joined(separator: " · ") }
    }

    /// Empirical: iPhone front-camera VGA frames with a single in-focus
    /// face at 30–50 cm deliver a Laplacian variance of ~150–800 over a
    /// 80×60 grid. The previous 60 threshold was so easy to clear that
    /// every frame passed, masking real registration problems. We now
    /// require at least 80 to keep the gate meaningful without
    /// rejecting slightly noisy close-up selfies.
    static let minSharpness = 80.0
    /// Face box vs frame area (front cam, 640×480 preset).
    static let minCoverage = 0.18
    static let maxCoverage = 0.95

    static func assess(frame: FrameResult) -> Verdict {
        var reasons: [String] = []

        let sharp = sharpness(of: frame)
        if sharp < minSharpness {
            reasons.append("image is blurry (sharpness \(Int(sharp)))")
        }

        guard let det = frame.detected, let face = det.faces.first else {
            reasons.append("no face detected")
            return Verdict(pass: false, reasons: reasons)
        }
        if det.faces.count > 1 {
            reasons.append("multiple faces")
            return Verdict(pass: false, reasons: reasons)
        }

        let area = Double(frame.width * frame.height)
        let faceArea = Double(face.width * face.height)
        let coverage = area > 0 ? faceArea / area : 0
        if coverage < minCoverage {
            reasons.append("face too small (move closer)")
        } else if coverage > maxCoverage {
            reasons.append("face too close")
        }

        if let points = det.points.first {
            let ok = landmarkIntegrity(points: points, frame: frame)
            if !ok { reasons.append("eyes not fully visible") }
        }

        return Verdict(pass: reasons.isEmpty, reasons: reasons)
    }

    /// Variance of the 3x3 Laplacian response over the Y plane,
    /// computed on a downsampled grid so it stays cheap per frame.
    /// Honors `yStride` so padded rows do not bleed into the next
    /// scan line — the previous version assumed stride == width and
    /// produced noisy numbers on iPhone NV21 buffers.
    private static func sharpness(of frame: FrameResult) -> Double {
        let w = frame.width, h = frame.height
        let stride = max(frame.yStride, w)
        let stepX = max(1, w / 80), stepY = max(1, h / 80)
        var sum = 0.0, sumSq = 0.0, n = 0
        frame.nv21.withUnsafeBytes { raw in
            let y = raw.bindMemory(to: UInt8.self)
            var yy = stepY
            while yy < h - stepY {
                var xx = stepX
                while xx < w - stepX {
                    // 3x3 Laplacian (no divide to keep it integer-ish)
                    let center = Int(y[yy * stride + xx])
                    let l = Int(y[yy * stride + (xx - stepX)])
                    let r = Int(y[yy * stride + (xx + stepX)])
                    let u = Int(y[(yy - stepY) * stride + xx])
                    let d = Int(y[(yy + stepY) * stride + xx])
                    let lap = 4 * center - l - r - u - d
                    sum += Double(lap)
                    sumSq += Double(lap * lap)
                    n += 1
                    xx += stepX
                }
                yy += stepY
            }
        }
        guard n > 0 else { return 0 }
        let mean = sum / Double(n)
        let variance = (sumSq / Double(n)) - (mean * mean)
        return max(0, variance)
    }

    /// Landmarks are [x0,y0,x1,y1,...,x4,y4] where 0=left eye, 1=right
    /// eye, 2=nose tip, 3=left mouth corner, 4=right mouth corner.
    /// All must be inside the frame and the eye-to-eye distance must
    /// be non-trivial (an occlusion often collapses it to ~0).
    private static func landmarkIntegrity(points: [NSNumber], frame: FrameResult) -> Bool {
        guard points.count >= 10 else { return false }
        var coords: [Double] = points.map { $0.doubleValue }
        for i in stride(from: 0, to: 10, by: 2) {
            let x = coords[i], y = coords[i + 1]
            if x < 0 || y < 0 || x >= Double(frame.width) || y >= Double(frame.height) {
                return false
            }
        }
        let dx = coords[2] - coords[0]
        let dy = coords[3] - coords[1]
        let eyeDist = sqrt(dx * dx + dy * dy)
        return eyeDist > 8.0
    }
}
