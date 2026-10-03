import Foundation
import Combine

/// Singleton view-model. Owns one FaceBridge process-wide. All
/// `FaceBridge` methods are blocking — callers should always dispatch
/// the call onto a background queue and update published state from
/// the main queue when it returns.
final class FaceModel: ObservableObject {
    static let shared = FaceModel()

    enum InitState: Equatable { case unknown, ok, noModels, failed(String) }

    @Published private(set) var initState: InitState = .unknown
    @Published private(set) var databaseSize: Int = 0
    @Published var lastError: String? = nil

    static let modelFiles = [ "fd_2_00.dat", "pd_2_00_pts5.dat", "pd_2_00_pts81.dat", "fr_2_10.dat" ]

    private init() {}

    func bootstrap() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            let modelDir = docs.appendingPathComponent("models")
            if !self.ensureModels(at: modelDir) {
                DispatchQueue.main.async { self.initState = .noModels }
                return
            }
            do {
                _ = try FaceBridge(modelDir: modelDir.path)
                DispatchQueue.main.async {
                    self.initState = .ok
                    self.databaseSize = Int(FaceBridge.databaseSize())
                }
            } catch let err as NSError {
                DispatchQueue.main.async { self.initState = .failed(err.localizedDescription) }
            } catch {
                DispatchQueue.main.async { self.initState = .failed("unknown error") }
            }
        }
    }

    /// Copy the bundled model weights into Documents/models on first
    /// launch. This makes the app work offline with zero manual file
    /// pushing (the .dat files ship inside the app bundle's
    /// Resources/Models folder).
    private func ensureModels(at dir: URL) -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in Self.modelFiles {
            let dest = dir.appendingPathComponent(file)
            if fm.fileExists(atPath: dest.path) { continue }
            guard let src = Bundle.main.url(forResource: file, withExtension: nil,
                                            subdirectory: "Models") ??
                            Bundle.main.url(forResource: file, withExtension: nil) else {
                return false
            }
            do {
                try fm.copyItem(at: src, to: dest)
            } catch {
                return false
            }
        }
        return true
    }

    func refreshDatabaseSize() {
        DispatchQueue.main.async { self.databaseSize = Int(FaceBridge.databaseSize()) }
    }
}
