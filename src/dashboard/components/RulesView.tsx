import { toggleObjectType, useOps } from "../store";
import type { ObjectClass, TimeWindow } from "../types";

export function RulesView() {
  const { state, setRules } = useOps();
  const { rules } = state;

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Rule engine</h2>
          <p>
            Four checks decide whether a detection is important. Toggles apply immediately to
            the live feed. This is configuration, not a polygon editor.
          </p>
        </div>
      </header>
      <div className="rules">
        <label className="rule">
          <div>
            <h3>Region of interest</h3>
            <p>Require the subject to cross a restricted ROI before packaging an alert.</p>
          </div>
          <span className="switch">
            <input
              type="checkbox"
              checked={rules.roiEnabled}
              onChange={(event) => setRules({ ...rules, roiEnabled: event.target.checked })}
            />
            <span />
          </span>
        </label>

        <label className="rule">
          <div>
            <h3>Time window</h3>
            <p>Limit promotion to after-hours events, or allow all hours.</p>
          </div>
          <select
            value={rules.timeWindow}
            onChange={(event) =>
              setRules({ ...rules, timeWindow: event.target.value as TimeWindow })
            }
          >
            <option value="all">All hours</option>
            <option value="after-hours">After hours only</option>
          </select>
        </label>

        <label className="rule">
          <div>
            <h3>Cooldown / debounce</h3>
            <p>Suppress repeat alerts from the same camera inside this window.</p>
          </div>
          <select
            value={rules.cooldownSec}
            onChange={(event) =>
              setRules({ ...rules, cooldownSec: Number(event.target.value) })
            }
          >
            <option value={30}>30 seconds</option>
            <option value={60}>60 seconds</option>
            <option value={120}>120 seconds</option>
          </select>
        </label>

        <div className="rule">
          <div>
            <h3>Object type filter</h3>
            <p>Human is the Safety Wedge / watchlist path. Vehicle is parking only.</p>
          </div>
          <div className="checks">
            {(["human", "vehicle"] as ObjectClass[]).map((value) => (
              <label key={value}>
                <input
                  type="checkbox"
                  checked={rules.objectTypes.includes(value)}
                  onChange={() => setRules(toggleObjectType(rules, value))}
                />
                {value}
              </label>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
