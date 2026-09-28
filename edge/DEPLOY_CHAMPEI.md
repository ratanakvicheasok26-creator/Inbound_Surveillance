# Champei Spa — Edge Laptop Deployment Checklist

Software hardening and defensive validation are complete for this tree. Use this checklist when setting up the back-room laptop. Do **not** implement per-user Telegram ID whitelists on the edge; staff ACL is Telegram group membership (see [`FEATURE_AUDIT.md`](FEATURE_AUDIT.md) §6).

## Prerequisites

- Windows laptop with camera / NVR access on the local network
- Python 3.12+ and a local venv
- Bot token from [@BotFather](https://t.me/BotFather)
- Staff **group** chat ID (`TELEGRAM_STAFF_CHAT_ID`, often `-100…`)
- Owner **1:1** chat ID (`TELEGRAM_OWNER_CHAT_ID`)

## 1. Environment setup

1. Copy or clone the repo to `C:\Champei\Inbound_Surveillance`.
2. Create and activate a virtual environment at the repo root; install edge deps:

```bat
cd C:\Champei\Inbound_Surveillance
python -m venv venv
venv\Scripts\activate
pip install -r edge\requirements.txt
```

3. Copy [`.env.champei.example`](../.env.champei.example) to `.env` at the repo root and fill:

```env
TELEGRAM_BOT_TOKEN=<new_token_from_botfather>
TELEGRAM_STAFF_CHAT_ID=-100xxxxxxxxxx
TELEGRAM_OWNER_CHAT_ID=yyyyyyyyy
BRANCH_ID=champei-pp-01
HUB_PORT=8000
```

4. Smoke-check (optional):

```bat
cd edge
python run_champei.py --mock
```

Confirm `[DB] SQLite WAL mode confirmed` and that the Telegram controller starts. Ctrl-C to stop.

## 2. Lockdown and permissions

1. Restrict `.env` to Administrators (or the service account that runs the task). On PowerShell (elevated):

```powershell
icacls C:\Champei\Inbound_Surveillance\.env /inheritance:r
icacls C:\Champei\Inbound_Surveillance\.env /grant:r Administrators:F
```

2. Prefer mode `600`-equivalent for SQLite under the data dir (owner-only). Boot path uses `db.connect` with `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout = 30000`; `run_champei` verifies WAL at start.

3. Keep the hub on loopback only (`127.0.0.1`). Do not rebind to `0.0.0.0`.

## 3. Auto-boot via Task Scheduler (`run_silent.vbs`)

`run_silent.vbs` is **not** checked into the repo (see also [`start_champei.bat`](../start_champei.bat) for an interactive console start). Create it once on the laptop:

**`C:\Champei\Inbound_Surveillance\run_silent.vbs`**

```vb
' Silent Champei edge boot — no console window.
Option Explicit
Dim sh, root, py, script
Set sh = CreateObject("WScript.Shell")
root = "C:\Champei\Inbound_Surveillance"
py = root & "\venv\Scripts\pythonw.exe"
If CreateObject("Scripting.FileSystemObject").FileExists(py) = False Then
  py = root & "\venv\Scripts\python.exe"
End If
script = root & "\edge\run_champei.py"
sh.CurrentDirectory = root & "\edge"
' 0 = hidden window; False = do not wait
sh.Run """" & py & """ """ & script & """", 0, False
```

**Task Scheduler**

1. Create Basic Task → trigger **At startup** (or **At log on**).
2. Action: Start a program  
   - Program/script: `wscript.exe`  
   - Arguments: `"C:\Champei\Inbound_Surveillance\run_silent.vbs"`
3. Properties: *Run whether user is logged on or not*, *Run with highest privileges*.
4. After a power cut, the engine should return without opening a visible console.

## 4. Remote management (Tailscale)

1. Install Tailscale on the laptop and authorize it to your private tailnet.
2. Enable unattended / always-on access for maintenance (SSH, RDP, or file pull).
3. Do **not** port-forward the hub or expose Telegram tokens to the public internet; use Tailscale only.

## RBAC reminder (accepted)

| Role | Bound to | Can run |
| --- | --- | --- |
| Owner | `TELEGRAM_OWNER_CHAT_ID` (1:1 DM) | `/scorecard`, `/churn`, staff cmds |
| Staff | `TELEGRAM_STAFF_CHAT_ID` (group) | `/name`, `/info`, `/help` — **any group member** |
| Stranger | other chats | Silent drop |

Hire/fire ACL = add/remove from the staff Telegram group. No numeric user-ID list on the laptop.
