import SwiftUI

struct DatabaseView: View {
    @EnvironmentObject var model: FaceModel
    @State private var entries: [PersonEntry] = []

    var body: some View {
        NavigationView {
            List {
                Section(footer: Text("SeetaFace2 entries: \(model.databaseSize) · SwiftUI list: \(entries.count)")) {
                    ForEach(entries) { e in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(e.name).font(.headline)
                            Text("id=\(e.id) · \(e.createdAt.formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .onDelete(perform: delete)
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Database")
            .toolbar { EditButton() }
            .onAppear { refresh() }
        }
    }

    private func refresh() {
        entries = PersonStore.shared.all()
    }

    private func delete(at offsets: IndexSet) {
        offsets.forEach { idx in
            let id = entries[idx].id
            _ = FaceBridge.delete(id)
            PersonStore.shared.delete(id: id)
        }
        refresh()
        model.refreshDatabaseSize()
    }
}

