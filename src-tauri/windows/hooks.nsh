; Installer hooks for Inbound Surveillance.
; Detects a missing Visual C++ 2015–2022 runtime and installs the copy
; bundled next to the app so Windows users do not have to hunt for it.

!macro NSIS_HOOK_PREINSTALL
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ReadRegDWord $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  ${If} $0 == 1
    DetailPrint "Visual C++ Redistributable already installed"
    Goto vcredist_done
  ${EndIf}

  ReadRegDWord $0 HKLM "SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  ${If} $0 == 1
    DetailPrint "Visual C++ Redistributable already installed"
    Goto vcredist_done
  ${EndIf}

  ReadRegDWord $0 HKCU "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
  ${If} $0 == 1
    DetailPrint "Visual C++ Redistributable already installed"
    Goto vcredist_done
  ${EndIf}

  StrCpy $1 "$INSTDIR\resources\vc_redist.x64.exe"
  ${IfNot} ${FileExists} "$1"
    StrCpy $1 "$INSTDIR\vc_redist.x64.exe"
  ${EndIf}

  ${If} ${FileExists} "$1"
    DetailPrint "Installing Visual C++ Redistributable (required by the camera engine)..."
    ; Quiet install first. Exit 0 = success, 1638 = already present, 3010 = reboot suggested.
    ; currentUser installers are not elevated, so quiet install often fails with
    ; access denied. Fall back to runas so Windows can prompt for permission.
    ExecWait '"$1" /install /quiet /norestart' $0
    ${If} $0 == 0
    ${OrIf} $0 == 1638
    ${OrIf} $0 == 3010
      DetailPrint "Visual C++ Redistributable is ready"
    ${Else}
      DetailPrint "Quiet install failed (code $0); requesting administrator permission..."
      ExecShellWait "runas" "$1" "/install /passive /norestart"
      DetailPrint "Visual C++ Redistributable install finished (or was skipped)"
    ${EndIf}
  ${Else}
    DetailPrint "Visual C++ Redistributable was not bundled in this installer"
  ${EndIf}

  vcredist_done:
!macroend

!macro NSIS_HOOK_PREUNINSTALL
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
!macroend
