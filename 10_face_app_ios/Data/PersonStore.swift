import Foundation

/// Tiny JSON-on-disk store for face metadata. The actual feature
/// embeddings live in the native FaceDatabase; this store just keeps
/// the user-visible name + creation timestamp for each FaceDatabase
/// index.
final class PersonStore {
    static let shared = PersonStore()

    private let url: URL
    private(set) var entries: [PersonEntry] = []

    private init() {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.url = docs.appendingPathComponent("people.json")
        load()
    }

    private func load() {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([PersonEntry].self, from: data) else { return }
        entries = decoded
    }

    private func save() {
        let data = try? JSONEncoder().encode(entries)
        try? data?.write(to: url, options: .atomic)
    }

    func all() -> [PersonEntry] { entries }

    func insert(id: Int64, name: String) {
        entries.removeAll { $0.id == id }
        entries.append(PersonEntry(id: id, name: name, createdAt: Date()))
        save()
    }

    func delete(id: Int64) {
        entries.removeAll { $0.id == id }
        save()
    }
}
