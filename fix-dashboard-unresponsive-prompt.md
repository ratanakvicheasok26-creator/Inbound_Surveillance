# Task: Fix Unresponsive Dashboard in Inbound Garage Hub (`edge/hub.html`)

## Problem Description
When opening the Inbound Garage dashboard (`edge/hub.html`, served by `edge/launcher.py` at `http://127.0.0.1:8765/`), none of the buttons, tabs, navigation links, or interactive controls are responsive.

## Root Cause
In [`edge/hub.html`](file:///home/george/Documents/Inbound-Surveillance/edge/hub.html), inside the primary `<script>` tag at lines 1876–1879, there is an orphaned, duplicate `catch` block immediately following the `hitBay()` function:

```javascript
// edge/hub.html (around lines 1869–1882)
    function hitBay(nx, ny) {
      for (let i = liveBays.length - 1; i >= 0; i--) {
        const [x, y, w, h] = liveBays[i].roi;
        if (nx >= x && nx <= x + w && ny >= y && ny <= y + h) return liveBays[i];
      }
      return null;
    }
      } catch (err) {
        /* ignore poll glitches */
      }
    }

    function clamp01(v) { return Math.max(0, Math.min(1, v)); }
```

This stray block causes an immediate **`Uncaught SyntaxError: Unexpected token '}'`** during browser JavaScript parsing.

Because the `<script>` tag fails to parse:
1. The browser aborts execution of the entire script.
2. Global functions (`setNav`, `showConfigTab`, `startAddCamera`, `connectAndStreamCamera`, `saveCameraOnly`, `setRotate`, `toggleFlip`, `testTelegram`, `onRoiDown`, `renderScorecard`, etc.) are never defined on `window`.
3. Every button with an `onclick` attribute throws an `Uncaught ReferenceError` when clicked.
4. The `DOMContentLoaded` event listener never registers, so `loadInitialConfig()`, `paintOrient()`, and all telemetry polling loops (`pollTelemetry`, `pollGarage`) never run.

## Instructions to Fix

1. Open [`edge/hub.html`](file:///home/george/Documents/Inbound-Surveillance/edge/hub.html).
2. Delete the 4 orphaned lines (lines 1876–1879):
   ```diff
   -      } catch (err) {
   -        /* ignore poll glitches */
   -      }
   -    }
   ```
   So that `hitBay` directly precedes `clamp01`:
   ```javascript
       function hitBay(nx, ny) {
         for (let i = liveBays.length - 1; i >= 0; i--) {
           const [x, y, w, h] = liveBays[i].roi;
           if (nx >= x && nx <= x + w && ny >= y && ny <= y + h) return liveBays[i];
         }
         return null;
       }

       function clamp01(v) { return Math.max(0, Math.min(1, v)); }
   ```

3. Verification:
   Run a JavaScript syntax check via Node.js:
   ```bash
   node -e '
   const fs = require("fs");
   const vm = require("node:vm");
   const html = fs.readFileSync("edge/hub.html", "utf8");
   const scriptStart = html.indexOf("<script>") + 8;
   const scriptEnd = html.lastIndexOf("</script>");
   const code = html.substring(scriptStart, scriptEnd);
   new vm.Script(code, { filename: "edge/hub.html" });
   console.log("hub.html JavaScript syntax is 100% VALID");
   '
   ```
4. Run the garage unit tests:
   ```bash
   edge/.venv/bin/python edge/test_garage.py
   ```
