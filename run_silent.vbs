' Silent Champei edge boot — no console window.
' Place at repo root. Task Scheduler: wscript.exe "<repo>\run_silent.vbs"
Option Explicit
Dim sh, fso, root, py, script
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)

If fso.FileExists(root & "\edge\.venv\Scripts\pythonw.exe") Then
  py = root & "\edge\.venv\Scripts\pythonw.exe"
ElseIf fso.FileExists(root & "\edge\.venv\Scripts\python.exe") Then
  py = root & "\edge\.venv\Scripts\python.exe"
ElseIf fso.FileExists(root & "\.venv\Scripts\pythonw.exe") Then
  py = root & "\.venv\Scripts\pythonw.exe"
ElseIf fso.FileExists(root & "\.venv\Scripts\python.exe") Then
  py = root & "\.venv\Scripts\python.exe"
ElseIf fso.FileExists(root & "\venv\Scripts\pythonw.exe") Then
  py = root & "\venv\Scripts\pythonw.exe"
ElseIf fso.FileExists(root & "\venv\Scripts\python.exe") Then
  py = root & "\venv\Scripts\python.exe"
Else
  py = "pythonw.exe"
End If

script = root & "\edge\run_champei.py"
sh.CurrentDirectory = root & "\edge"
' 0 = hidden window; False = do not wait
sh.Run """" & py & """ """ & script & """", 0, False

