# 📦 Champei Spa Intelligence — Customer Download & Installation Review

## 🚀 Package Overview

| Attribute | Details |
| :--- | :--- |
| **Package File** | `Champei_Spa_Intelligence.zip` |
| **File Size** | **~710 MB** (Complete offline bundle) |
| **Target OS** | Windows 10 / Windows 11 (64-bit) |
| **Included AI Models** | YOLOv8 person detection, Role classification, ReID models (`.onnx`, `.pt`) |
| **Setup Script** | `INSTALL_WINDOWS.bat` (1-Click Automated Setup) |
| **Launcher** | `start_champei.bat` / Desktop Shortcut |

---

## 🎁 Why This Package is Ideal for Non-Technical Customers

1. **Pre-Bundled AI Models**: Vision weights are already stored inside the package — no large model downloads during client setup.
2. **Zero Developer Tooling Needed**: Non-technical clients **do not need** Git, Node.js, npm, or terminal tools.
3. **Automated Desktop Shortcut**: Running the setup automatically creates a **"Champei Spa Intelligence"** launcher on their Windows Desktop.
4. **Isolated Python Environment**: Automatically provisions an isolated virtual environment (`edge\.venv`) without conflicting with any existing software.
5. **Self-Verifying Diagnostic**: Setup includes an automated component test to verify models, camera connections, and web server readiness before concluding.

---

## 🌐 Hosting & Distribution Options

### 🏆 Option 1: GitHub Release (Recommended for Public Direct Download)
Publishing as a **GitHub Release** allows customers to download with a single click without needing a GitHub account:
1. Go to: [New GitHub Release](https://github.com/sothunly-alt/Ibound-survillence-V2/releases/new)
2. Create a tag (e.g., `v1.0.0`).
3. Drag & drop `Champei_Spa_Intelligence.zip` into the **Attach binaries** section.
4. Provide the customer with your direct download link:
   ```text
   https://github.com/sothunly-alt/Ibound-survillence-V2/releases/latest/download/Champei_Spa_Intelligence.zip
   ```

### ☁️ Option 2: Cloud Storage (Google Drive / OneDrive / Dropbox)
1. Upload `Champei_Spa_Intelligence.zip` to Google Drive or OneDrive.
2. Set permissions to **"Anyone with the link can view/download"**.
3. Paste the share link in the customer instructions below.

### 👥 Option 3: GitHub Actions Build Artifact (Internal Team Access)
If the recipient has repository permissions and is logged into GitHub:
- [Actions Run #36809678776 Artifact #11139208855](https://github.com/sothunly-alt/Ibound-survillence-V2/actions/runs/36809678776/artifacts/11139208855)

---

## 📋 Instructions to Send to the Customer

### 1. Download & Extract
* **Download link:** `[Insert your GitHub Release or Drive Link here]` (or `Champei_Spa_Intelligence.zip`)
* Right-click the `.zip` file and select **Extract All...** (or unzip to a folder of choice, e.g., `C:\Champei_Spa_Intelligence`).

### 2. Install Python (If not already installed)
* Download Python (v3.10, 3.11, or 3.12) from [python.org/downloads](https://www.python.org/downloads/).
* ⚠️ **IMPORTANT**: During installation, check the box:
  `[✔] Add Python to PATH`

### 3. Run 1-Click Setup
* Open the extracted folder.
* Double-click **`INSTALL_WINDOWS.bat`**.
* The installer will automatically:
  - Configure the isolated environment
  - Install dependencies
  - Run diagnostic verification
  - Create a **"Champei Spa Intelligence"** shortcut on the Desktop

### 4. Start Using the System
* Double-click the **"Champei Spa Intelligence"** shortcut on the Desktop (or `start_champei.bat` inside the folder) anytime to launch the surveillance & analytics interface.

---

## 🔄 For Developers / Admins: Regenerating the ZIP
Whenever updates or code adjustments are made, regenerate the distribution package with:

```bash
python3 tools/package_release_zip.py
```
