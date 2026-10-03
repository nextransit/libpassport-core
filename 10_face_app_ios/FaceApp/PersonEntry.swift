import Foundation

/// Swift-side mirror of a registered person. The actual 1024-dim
/// embedding lives in the native FaceDatabase; this struct only
/// carries the metadata used by the Database tab UI.
struct PersonEntry: Identifiable, Codable, Equatable {
    let id: Int64
    var name: String
    var createdAt: Date
}
