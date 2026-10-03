import SwiftUI

/// Two-tab container that keeps both tabs mounted at all times
/// so that AVCaptureVideoPreviewLayer and SwiftUI body diff stay
/// bounded across tab switches (the original `TabView` lazy mount
/// was the source of the multi-second freeze).
///
/// Selection can be driven two ways:
///   - tapping a `CustomTabBar` button (with animated transition)
///   - a `Notification.Name("FaceAppSelection")` carrying
///     `userInfo["tab"]` of "live" / "register" / "database"
///     (used by SmokeTest.runScreenshots()).
struct ContentView: View {
    @EnvironmentObject var model: FaceModel
    @State private var selection: AppTab = .live
    /// Drives the smooth slide+fade between tabs.
    @State private var transition: AppTab = .live

    enum AppTab: Hashable {
        case live, register, database
        init?(_ raw: String) {
            switch raw.lowercased() {
            case "live": self = .live
            case "register": self = .register
            case "database": self = .database
            default: return nil
            }
        }
        var rawValue: String {
            switch self {
            case .live: return "live"
            case .register: return "register"
            case .database: return "database"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                tabLayer(.live)
                tabLayer(.register)
                tabLayer(.database)
            }
            .frame(maxHeight: .infinity)

            CustomTabBar(selection: $selection)
        }
        .tint(.cyan)
        .onAppear { model.bootstrap() }
        .onChange(of: selection) { newValue in
            withAnimation(.easeInOut(duration: 0.18)) {
                transition = newValue
            }
        }
        .onAppear { transition = selection }
        .onReceive(NotificationCenter.default.publisher(for: Notification.Name("FaceAppSelection"))) { note in
            if let raw = note.userInfo?["tab"] as? String, let t = AppTab(raw) {
                selection = t
            }
        }
    }

    /// A single tab's view: it stays mounted (good for camera) but
    /// only the visible one takes hits and has full opacity. A
    /// short transition tag lets SwiftUI animate between them.
    @ViewBuilder
    private func tabLayer(_ tab: AppTab) -> some View {
        Group {
            switch tab {
            case .live: LiveDetectView()
            case .register: RegisterView()
            case .database: DatabaseView()
            }
        }
        .opacity(transition == tab ? 1 : 0)
        .allowsHitTesting(transition == tab)
        // Pure opacity transitions so the CameraPreview /
        // NV21Preview inside each tab stays laid out at its full
        // frame and the mock camera frames keep rendering
        // throughout. The original .move(edge: .leading) layered
        // a translation on top, which clipped the live preview
        // during the 0.18 s animation and made the second back-
        // switch land on an all-black LiveDetectView.
        .transition(.opacity)
    }
}

/// Hand-rolled tab bar. The active button shows a coloured pill
/// background that animates between buttons so the operator
/// always knows where they are.
struct CustomTabBar: View {
    @Binding var selection: ContentView.AppTab
    @Namespace private var pillNamespace

    var body: some View {
        HStack(spacing: 0) {
            tabButton(.live, icon: "camera.viewfinder", label: "Live")
            tabButton(.register, icon: "person.crop.circle.badge.plus", label: "Register")
            tabButton(.database, icon: "internaldrive", label: "Database")
        }
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Color.cyan.opacity(0.3))
                .frame(height: 1)
        }
    }

    @ViewBuilder
    private func tabButton(_ tab: ContentView.AppTab, icon: String, label: String) -> some View {
        Button {
            // Light haptic on every tab change so the operator
            // feels the switch even if the visual transition
            // is sub-perceptual.
            UISelectionFeedbackGenerator().selectionChanged()
            selection = tab
        } label: {
            VStack(spacing: 2) {
                Image(systemName: icon)
                    .font(.system(size: 22, weight: .semibold))
                Text(label)
                    .font(.system(.caption2, design: .monospaced))
            }
            .frame(maxWidth: .infinity)
            .foregroundStyle(selection == tab ? Color.cyan : Color.white.opacity(0.5))
            .padding(.vertical, 4)
            .background(
                Group {
                    if selection == tab {
                        Capsule()
                            .fill(Color.cyan.opacity(0.18))
                            .matchedGeometryEffect(id: "tab-pill", in: pillNamespace)
                    }
                }
            )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
        .accessibilityAddTraits(selection == tab ? [.isSelected] : [])
    }
}

#Preview {
    ContentView().environmentObject(FaceModel.shared)
}
