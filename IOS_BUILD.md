# iOS 工程编译与安装指南 (CodexMobile / Decard Remote)

> 本文档描述 **remodex_rust** 仓库内 iOS 应用 (`CodexMobile`, bundle id
> `com.decard.remoteagent`) 的全链路 build / ad-hoc 签名 / OTA 发布 /
> iPhone 安装方法。所有敏感凭证（Apple ID、Team ID、远端机器 SSH 密码
> 等）均来自 `passport/.env`，**本文档不包含任何明文凭证**。

仓库: <https://github.com/nextransit/remodex_rust>
本地路径: `/Users/zhouyong/Desktop/work/Decard/gitlab/ai/remodex_rust`

---

## 0. 加载凭证

```sh
# 只在当前 shell 内生效，绝不写入 history / git
set -a
source /Users/zhouyong/gitlab/passport/.env
set +a
```

脚本中需要的变量（已自动来自 `passport/.env`）：

| 变量 | 用途 |
| --- | --- |
| `IOS_DEVELOPMENT_TEAM` | Xcode `DEVELOPMENT_TEAM`（如 `D4LM897A5P`）|
| `APPLE_DEVELOPER`     | Apple ID（用于解锁 keychain / 开发者后台）|
| `APPLE_PASSWORD`      | Apple ID 专用 app password（**不是登录密码**）|
| `MACMINI_IP`          | macmini 的 Tailscale / LAN IP |
| `MACMINI_USER`        | macmini 的 SSH 用户 |
| `MACMINI_PASSWORD`    | macmini 的 SSH 密码（keychain 解锁备用）|
| `MACMINI_WORKDIR`     | macmini 上的工作根（如 `/Volumes/Extended/macmini`）|
| `IOS_REMOTE_WORKDIR`  | 远端 build 目录（与 `MACMINI_WORKDIR` 通常一致）|
| `IOS_XCODE_PATH`      | 远端 Xcode.app 完整路径 |
| `IOS_RELAY_URL`       | 中继 WebSocket URL（如 `ws://43.134.2.104:5080/relay`）|

---

## 1. 远端机器拓扑

| 角色 | 主机 | 说明 |
| --- | --- | --- |
| 开发者本机 (MacBook Pro) | `100.77.223.85` (TS), `192.168.26.110` (LAN) | 跑 Rust bridge / Tailscale Serve |
| macmini (iOS 构建机) | `100.82.48.23` (TS), `192.168.26.108` (LAN) | 跑 Xcode build + serve OTA |
| ben-ubuntu (Linux bridge) | `100.97.199.25` (TS), `192.168.26.106` (LAN) | Linux bridge 交叉编译 |
| 43 服务器 (生产 relay) | `43.134.2.104` | 远端中继 `systemd` 服务 |

---

## 2. 在 macmini 上 SSH 登录

macmini 默认允许 SSH key + 密码双认证。自动化脚本里用
`sshpass`（密码来自 `MACMINI_PASSWORD`）：

```sh
sshpass -p "$MACMINI_PASSWORD" ssh "${MACMINI_USER}@${MACMINI_IP}"
```

> 如果偏好 SSH key，把 `~/.ssh/decard` 拷过去并加入 `ssh-agent` 即可，
> 脚本里直接 `ssh ben@${MACMINI_IP}` 也能工作。

---

## 3. 解锁 macmini 的 signing keychain（可选）

正常 Xcode 自动签名流程需要 keychain 里有 `Apple Development` 证书。
当证书无法解锁时（例如 `decard-signing.keychain-db` 密码遗失），用
**ad-hoc 签名**绕过（见 §5）。

```sh
sshpass -p "$MACMINI_PASSWORD" ssh "${MACMINI_USER}@${MACMINI_IP}" bash -s <<'REMOTE'
security unlock-keychain -p "$APPLE_PASSWORD" ~/Library/Keychains/login.keychain-db
REMOTE
```

> 仅在确实要走真机调试且 Apple ID + 专用密码可解锁 `login.keychain` 时
> 才执行。**当前生产 OTA 流程使用 ad-hoc，不依赖 keychain**。

---

## 4. iOS 源码结构

remodex_rust 仓库内 **只保存 overlay（差量补丁）**：

```
remodex_rust/
└── ios/self_hosted/
    ├── prepare.py            # 给上游源码打补丁（overlay 同步）
    ├── overlay/CodexMobile/  # 真正的 diff 源
    │   └── CodexMobile/
    │       └── Services/SelfHostedUpdateService.swift
    └── tests/test_prepare.py
```

macmini 上的**完整上游源码**在：

```
${IOS_REMOTE_WORKDIR}/remodex-ios/source/CodexMobile/
```

> **永远在 macmini 上编辑 Swift，再把 diff 同步回 `overlay/`**。
> 这样 `prepare.py` 重新打补丁不会冲掉你的工作。

把 macmini 的改动同步回 repo：

```sh
scp "${MACMINI_USER}@${MACMINI_IP}:${IOS_REMOTE_WORKDIR}/remodex-ios/source/CodexMobile/CodexMobile/Services/SelfHostedUpdateService.swift" \
    ios/self_hosted/overlay/CodexMobile/CodexMobile/Services/SelfHostedUpdateService.swift

git -c user.email=auto@local -c user.name=Auto commit -am "feat(ios-overlay): <change>"
git push origin master
```

---

## 5. 完整 build → ipa → ad-hoc 签名流程

> 在 macmini 上一次性完成。全部命令默认从本机通过 SSH 触发。

```sh
sshpass -p "$MACMINI_PASSWORD" ssh "${MACMINI_USER}@${MACMINI_IP}" bash -s <<REMOTE
set -euo pipefail

# 1) archive
xcodebuild \
  -project CodexMobile/CodexMobile.xcodeproj \
  -scheme CodexMobile \
  -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "${IOS_REMOTE_WORKDIR}/remodex-ios/build/DerivedData-\$(date +%Y%m%d%H%M)" \
  -archivePath   "${IOS_REMOTE_WORKDIR}/remodex-ios/build/DecardRemote.xcarchive" \
  CODE_SIGN_STYLE=Automatic \
  DEVELOPMENT_TEAM="${IOS_DEVELOPMENT_TEAM}" \
  PRODUCT_BUNDLE_IDENTIFIER=com.decard.remoteagent \
  -skipPackagePluginValidation -skipMacroValidation \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO \
  archive

# 2) export ipa
xcodebuild -exportArchive \
  -archivePath "${IOS_REMOTE_WORKDIR}/remodex-ios/build/DecardRemote.xcarchive" \
  -exportPath   "${IOS_REMOTE_WORKDIR}/remodex-ios/build/export" \
  -exportOptionsPlist "${IOS_REMOTE_WORKDIR}/remodex-ios/build/ExportOptions.plist"

# 3) ad-hoc 签名（绕过 keychain / 真证书缺失）
codesign --force --sign - --timestamp=none --deep \
  "${IOS_REMOTE_WORKDIR}/remodex-ios/build/export/CodexMobile.ipa"

# 4) 校验
codesign -dv "${IOS_REMOTE_WORKDIR}/remodex-ios/build/export/CodexMobile.ipa" \
  | grep -E 'Signature|Identifier|TeamIdentifier'
ls -lh  "${IOS_REMOTE_WORKDIR}/remodex-ios/build/export/CodexMobile.ipa"
REMOTE
```

> ⚠️ `DEVELOPMENT_TEAM` 在 archive 阶段写一次只是为了保持 `Info.plist`
> 里 `CFBundleIdentifier` 一致；真正的签名因为 `CODE_SIGNING_ALLOWED=NO`
> 被禁掉，由 §5.3 的 ad-hoc `codesign` 完成。

### 5.1 何时用真签名

仅当：

- Apple Developer `decardemv@decard.com` 的 专用 app password 已知，且
- macmini 上 `~/Library/Keychains/login.keychain-db` 能用该密码解锁，
- 仍能看到 Apple Development 证书。

否则一律 ad-hoc。**当前默认走 ad-hoc。**

### 5.2 已知限制（ad-hoc）

- 安装后 **7 天内** 必须重装（iOS 不信任过期 ad-hoc 签名）。
- iPhone 必须**信任企业证书**或保持开发者模式开启（首次安装需要
  Settings → General → VPN & Device Management → 信任）。
- 一个 ipa **最多绑定 100 个设备 UUID**（ad-hoc 上限）。

### 5.3 重新打签名（如果将来 Apple Development 证书恢复）

```sh
sshpass -p "$MACMINI_PASSWORD" ssh "${MACMINI_USER}@${MACMINI_IP}" bash -s <<REMOTE
set -euo pipefail
security unlock-keychain -p "\$APPLE_PASSWORD" \
  ~/Library/Keychains/login.keychain-db
xcodebuild \
  -project CodexMobile/CodexMobile.xcodeproj \
  -scheme CodexMobile -configuration Debug \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "${IOS_REMOTE_WORKDIR}/remodex-ios/build/DerivedData-real" \
  -archivePath     "${IOS_REMOTE_WORKDIR}/remodex-ios/build/DecardRemote.xcarchive" \
  CODE_SIGN_STYLE=Automatic \
  DEVELOPMENT_TEAM="${IOS_DEVELOPMENT_TEAM}" \
  PRODUCT_BUNDLE_IDENTIFIER=com.decard.remoteagent \
  -skipPackagePluginValidation -skipMacroValidation \
  archive
REMOTE
```

无需再 `codesign --force --sign -`，archive 阶段会处理。

---

## 6. 发布 OTA：写 release.json / manifest.plist

> Swift 端 `SelfHostedRelease` struct 对 `release.json` 的 schema 要求：
>
> ```swift
> struct SelfHostedRelease: Decodable, Equatable, Sendable {
>     let version: String
>     let build: Int            // 必须是 int，不是 string！
>     let installURL: URL
>     let publishedAt: Date     // ISO8601
>     let expiresAt: Date?      // ISO8601，可选
>     let ipaSizeBytes: Int64?  // 可选
> }
> ```
>
> **任何 schema 不匹配都会让 iPhone 报 `Update feed is malformed`。**

### 6.1 用  一键生成（推荐）

remodex 仓库里已经有现成的脚本：

```sh
set -a
source /Users/zhouyong/gitlab/passport/.env
set +a

sshpass -p "$MACMINI_PASSWORD" ssh "${MACMINI_USER}@${MACMINI_IP}"   python3 \/remodex_rust/ios/self_hosted/scripts/build_ota_artifacts.py     --ipa        \/remodex-ios/build/export/CodexMobile.ipa     --publish-dir /Users/\/.remodex/ios-self-hosted/publish     --base-url   https://benzhous-macbook-pro.tail6788c8.ts.net/decard-remote
```

脚本会自动算 SHA256 / size / ISO8601 时间，生成与
`SelfHostedRelease` Swift struct 严格对齐的 release.json + manifest.plist。

支持的参数（`--help` 查看全部）：

- `--build YYYYMMDDHHMM` 覆盖默认 build 时间戳
- `--expires-at ISO8601` 覆盖默认过期时间
- `--upstream-ref <sha>` 覆盖默认上游 commit 引用
- `--dry-run` 只打印 release.json 不写文件

### 6.2 单元测试

```sh
cd remodex_rust
python3 -m pytest ios/self_hosted/tests/test_build_ota_artifacts.py -v
# 5 passed: schema 严格校验、plist 资产一致性、dry-run 不写文件、
#           非法 build 字符串拒绝、缺 ipa 拒绝
```

### 6.3 同步到 MacBook Pro (Tailscale Serve 源)

```sh
scp "${MACMINI_USER}@${MACMINI_IP}:/Users/${MACMINI_USER}/.remodex/ios-self-hosted/publish/*.{ipa,plist,json}"     ~/.remodex/ios-self-hosted/publish/
```

### 6.4 验证可达性

```sh
# release.json schema 正确
curl -sS https://benzhous-macbook-pro.tail6788c8.ts.net/decard-remote/release.json | python3 -m json.tool

# ipa 完整下载
curl -sS -o /dev/null -w 'status=%{http_code} size=%{size_download}
'   "https://benzhous-macbook-pro.tail6788c8.ts.net/decard-remote/decard-remote-3.0-${BUILD}.ipa"

# manifest.plist 也得 200
curl -sS -o /dev/null -w 'manifest_status=%{http_code}
'   https://benzhous-macbook-pro.tail6788c8.ts.net/decard-remote/manifest.plist
```

## 7. iPhone 安装

### 7.1 OTA (推荐，Tailscale / Wi-Fi 都行)

iPhone app 启动时会自动：

1. 拉 `release.json`；
2. 若 `build > CFBundleVersion`，自动弹 **Install / Later** 系统弹窗；
3. 用户点 Install → iOS 接管 `itms-services://` → 弹系统级安装确认。

所以只要 iPhone 在 Tailscale 上、app 没被 force kill，新版本 ipa 一
放到 `publish/` 就会**两秒内弹 alert**。

### 7.2 USB + Xcode（紧急 / 调试用）

```sh
sshpass -p "$MACMINI_PASSWORD" ssh "${MACMINI_USER}@${MACMINI_IP}" \
  xcrun devicectl device install app \
    --device <DEVICE_UDID> \
    ${IOS_REMOTE_WORKDIR}/remodex-ios/build/export/CodexMobile.ipa
```

> iPhone 必须先解锁屏幕并信任 macmini。第一次 `devicectl list devices`
> 会显示 `unavailable`，解锁 + USB 接入后会变 `available`。

### 7.3 卸载 + 重装

```sh
xcrun devicectl device uninstall app \
  --device <DEVICE_UDID> com.decard.remoteagent
```

---

## 8. 排错

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `error: "...keychain-db" couldn't be unlocked` | keychain 密码不在 `.env` | 改用 §5 的 ad-hoc 签名（无需 keychain）|
| Xcode archive 卡在 `Compiling SecureStore.swift` 5 分钟以上 | DerivedData 损坏 | `rm -rf DerivedData-*` 后重 archive |
| `xcrun devicectl list devices` 显示 `unavailable` | iPhone 锁屏 / 没接 USB | 解锁屏幕 + 重新插线 |
| iPhone 装完启动后 alert **不弹** | `currentBuild >= latestRelease.build` 或 dismissedBuild == latestBuild | 删旧 build / Settings 里 Check for Updates force |
| OTA `release.json` 报 `malformed` | schema 错（build 是 string） | 重新跑 §6.1 脚本确认 `build` 是 int |
| `tailscaled` 与 `tailscale` CLI 版本不匹配 | 升级到相同版本 | `brew upgrade tailscale` |
| iPhone `itms-services` 弹窗点了 Install 后一直 "Verifying..." | ad-hoc 签名过期 (>7d) | 重 build + 重发 ipa |

---

## 9. 相关文档

- `remodex_rust/README.md`
- `remodex_rust/ios/self_hosted/prepare.py` (overlay patcher)
- `remodex_rust/AGENTS.md` (multi-machine deploy rules)
- `passport/.env` (本文档引用的全部凭证)
