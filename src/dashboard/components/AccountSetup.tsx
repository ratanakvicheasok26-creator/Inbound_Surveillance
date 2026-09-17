import { useEffect, useMemo, useState, type FormEvent } from "react";
import { PROTOCOLS, hostFromUrl, initials, type CameraInput, type CrewWithPhoto, type RoiInput } from "../account";
import { useAccount } from "../auth";
import type { CameraProtocol } from "../types";
import { parseWorkplaceId, parseZoneKind, workplaceOf, type ZoneKind } from "../../workplaces";

const ALL_STEPS = [
  { id: "profile", label: "Profile" },
  { id: "crew", label: "Crew" },
  { id: "cameras", label: "Cameras" },
  { id: "roi", label: "ROI" },
] as const;

type StepId = (typeof ALL_STEPS)[number]["id"];

function asProtocol(value: string): CameraProtocol {
  return (PROTOCOLS as string[]).includes(value) ? (value as CameraProtocol) : "rtsp";
}

export function AccountSetup({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { snapshot, updateProfile, uploadAvatar, saveCrew, removeCrew, saveCamera, removeCamera, saveRois } =
    useAccount();
  const [step, setStep] = useState<StepId>("profile");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [displayName, setDisplayName] = useState(snapshot?.profile.display_name || "");
  const [venueName, setVenueName] = useState(snapshot?.profile.venue_name || "");
  const [crewName, setCrewName] = useState("");
  const [crewRole, setCrewRole] = useState("technician");
  const [crewPhoto, setCrewPhoto] = useState<File | null>(null);
  const [camName, setCamName] = useState("");
  const [camZone, setCamZone] = useState("");
  const [camProtocol, setCamProtocol] = useState<CameraProtocol>("rtsp");
  const [camUrl, setCamUrl] = useState("");
  const [camUser, setCamUser] = useState("");
  const [camPass, setCamPass] = useState("");
  const [roiName, setRoiName] = useState("");
  const [roiType, setRoiType] = useState<ZoneKind>("vehicle_bay");
  const [roiCamera, setRoiCamera] = useState("");
  const [roiX, setRoiX] = useState("0.10");
  const [roiY, setRoiY] = useState("0.20");
  const [roiW, setRoiW] = useState("0.35");
  const [roiH, setRoiH] = useState("0.60");

  const workplace = workplaceOf(snapshot?.profile.workplace_type);
  const STEPS = useMemo(
    () => ALL_STEPS.filter((item) => (item.id === "crew" ? workplace.includeCrewSetup : true)),
    [workplace.includeCrewSetup],
  );
  const stepIndex = Math.max(0, STEPS.findIndex((item) => item.id === step));
  const crews = snapshot?.crews || [];
  const cameras = snapshot?.cameras || [];
  const rois = snapshot?.rois || [];

  const roiDrafts: RoiInput[] = useMemo(
    () =>
      rois.map((row, index) => ({
        id: row.id,
        name: row.name,
        bay_type: row.bay_type,
        zone_kind: row.zone_kind || row.bay_type,
        roi: row.roi,
        camera_id: row.camera_id,
        external_id: row.external_id,
        sort_order: row.sort_order ?? index,
      })),
    [rois],
  );

  useEffect(() => {
    if (!open || !snapshot) return;
    setDisplayName(snapshot.profile.display_name || "");
    setVenueName(snapshot.profile.venue_name || "");
  }, [open, snapshot?.profile.display_name, snapshot?.profile.venue_name]);

  useEffect(() => {
    if (open) setStep("profile");
    if (open && snapshot) setRoiType(workplace.defaultZoneKind);
  }, [open, snapshot, workplace.defaultZoneKind]);

  if (!open || !snapshot) return null;
  const account = snapshot;

  async function run(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed.");
    } finally {
      setBusy(false);
    }
  }

  async function saveProfile(event: FormEvent) {
    event.preventDefault();
    await run(() => updateProfile({ display_name: displayName.trim(), venue_name: venueName.trim() }));
  }

  async function addCrew(event: FormEvent) {
    event.preventDefault();
    if (!crewName.trim()) return;
    await run(async () => {
      await saveCrew({ display_name: crewName.trim(), role: crewRole.trim(), photo: crewPhoto });
      setCrewName("");
      setCrewRole("technician");
      setCrewPhoto(null);
    });
  }

  async function addCamera(event: FormEvent) {
    event.preventDefault();
    const payload: CameraInput = {
      name: camName.trim(),
      zone: camZone.trim(),
      protocol: camProtocol,
      source_url: camUrl.trim(),
      username: camUser.trim(),
      password: camPass,
    };
    if (!payload.name || !payload.source_url) return;
    await run(async () => {
      await saveCamera(payload);
      setCamName("");
      setCamZone("");
      setCamUrl("");
      setCamUser("");
      setCamPass("");
    });
  }

  async function addRoi(event: FormEvent) {
    event.preventDefault();
    const roi = [roiX, roiY, roiW, roiH].map((value) => Number(value));
    if (!roiName.trim() || roi.some((value) => Number.isNaN(value))) return;
    await run(async () => {
      await saveRois([
        ...roiDrafts,
        {
          name: roiName.trim(),
          bay_type: roiType,
          zone_kind: roiType,
          roi,
          camera_id: roiCamera || null,
          sort_order: roiDrafts.length,
        },
      ]);
      setRoiName("");
    });
  }

  async function finish() {
    await run(async () => {
      await updateProfile({
        display_name: displayName.trim() || account.profile.display_name,
        venue_name: venueName.trim() || account.profile.venue_name,
        setup_completed: true,
      });
      onClose();
    });
  }

  return (
    <div className="setup-scrim" role="presentation">
      <section className="setup-panel" role="dialog" aria-labelledby="setup-title">
        <header className="setup-panel__head">
          <div>
            <h2 id="setup-title">Account setup</h2>
            <p>These settings stay private to your login. Other accounts cannot see them.</p>
          </div>
          <button className="btn btn--ghost btn--sm" type="button" onClick={() => void finish()}>
            Close
          </button>
        </header>
        <nav className="setup-steps" aria-label="Setup steps">
          {STEPS.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={item.id === step ? "is-on" : ""}
              onClick={() => setStep(item.id)}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              {item.label}
            </button>
          ))}
        </nav>
        {error ? <p className="auth-error">{error}</p> : null}

        {step === "profile" ? (
          <form className="form" onSubmit={(event) => void saveProfile(event)}>
            <label className="avatar-picker">
              {snapshot.avatarUrl ? (
                <img src={snapshot.avatarUrl} alt="" />
              ) : (
                <span>{initials(displayName || snapshot.profile.display_name)}</span>
              )}
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp,image/gif"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void run(() => uploadAvatar(file));
                }}
              />
              <em>Private profile picture</em>
            </label>
            <label className="field">
              <span>Display name</span>
              <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} required />
            </label>
            <label className="field">
              <span>Venue</span>
              <input value={venueName} onChange={(event) => setVenueName(event.target.value)} />
            </label>
            <label className="field">
              <span>Workplace</span>
              <input value={workplace.label} readOnly />
            </label>
            <button className="btn btn--primary" type="submit" disabled={busy}>
              Save profile
            </button>
          </form>
        ) : null}

        {step === "crew" ? (
          <div className="setup-grid">
            <form className="form" onSubmit={(event) => void addCrew(event)}>
              <label className="field">
                <span>Crew name</span>
                <input value={crewName} onChange={(event) => setCrewName(event.target.value)} placeholder="Sothun" required />
              </label>
              <label className="field">
                <span>Role</span>
                <input value={crewRole} onChange={(event) => setCrewRole(event.target.value)} placeholder="technician" />
              </label>
              <label className="field">
                <span>Identity photo</span>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/gif"
                  onChange={(event) => setCrewPhoto(event.target.files?.[0] || null)}
                />
              </label>
              <button className="btn btn--primary" type="submit" disabled={busy}>
                Add crew identity
              </button>
            </form>
            <div className="setup-list">
              {crews.length ? crews.map((crew) => <CrewCard key={crew.id} crew={crew} onDelete={() => void run(() => removeCrew(crew))} />) : <p>No crew identities yet.</p>}
            </div>
          </div>
        ) : null}

        {step === "cameras" ? (
          <div className="setup-grid">
            <form className="form" onSubmit={(event) => void addCamera(event)}>
              <label className="field">
                <span>Camera name</span>
                <input value={camName} onChange={(event) => setCamName(event.target.value)} placeholder={workplace.cameraPlaceholder} required />
              </label>
              <label className="field">
                <span>Zone</span>
                <input value={camZone} onChange={(event) => setCamZone(event.target.value)} placeholder="Front shop" />
              </label>
              <label className="field">
                <span>Connecting protocol</span>
                <select value={camProtocol} onChange={(event) => setCamProtocol(asProtocol(event.target.value))}>
                  {PROTOCOLS.map((item) => (
                    <option key={item} value={item}>
                      {item.toUpperCase()}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Stream URL</span>
                <input
                  value={camUrl}
                  onChange={(event) => setCamUrl(event.target.value)}
                  placeholder="rtsp://192.168.1.64:554/Streaming/Channels/101"
                  required
                />
              </label>
              <div className="row-2">
                <label className="field">
                  <span>Username</span>
                  <input value={camUser} onChange={(event) => setCamUser(event.target.value)} autoComplete="off" />
                </label>
                <label className="field">
                  <span>Password</span>
                  <input type="password" value={camPass} onChange={(event) => setCamPass(event.target.value)} autoComplete="off" />
                </label>
              </div>
              <button className="btn btn--primary" type="submit" disabled={busy}>
                Save camera
              </button>
            </form>
            <div className="setup-list">
              {cameras.length ? (
                cameras.map((camera) => (
                  <article className="setup-item" key={camera.id}>
                    <div>
                      <h3>{camera.name}</h3>
                      <p className="mono">
                        {camera.protocol.toUpperCase()} · {hostFromUrl(camera.source_url)}
                      </p>
                    </div>
                    <button className="btn btn--danger btn--sm" type="button" onClick={() => void run(() => removeCamera(camera.id))}>
                      Remove
                    </button>
                  </article>
                ))
              ) : (
                <p>No private cameras yet.</p>
              )}
            </div>
          </div>
        ) : null}

        {step === "roi" ? (
          <div className="setup-grid">
            <form className="form" onSubmit={(event) => void addRoi(event)}>
              <label className="field">
                <span>{workplace.roiNameLabel}</span>
                <input value={roiName} onChange={(event) => setRoiName(event.target.value)} placeholder={workplace.roiPlaceholder} required />
              </label>
              <label className="field">
                <span>Type</span>
                <select
                  value={roiType}
                  onChange={(event) => setRoiType(parseZoneKind(event.target.value, parseWorkplaceId(workplace.id)))}
                >
                  {workplace.zoneKinds.map((kind) => (
                    <option key={kind.id} value={kind.id}>
                      {kind.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Camera (optional)</span>
                <select value={roiCamera} onChange={(event) => setRoiCamera(event.target.value)}>
                  <option value="">Any / unassigned</option>
                  {cameras.map((camera) => (
                    <option key={camera.id} value={camera.id}>
                      {camera.name}
                    </option>
                  ))}
                </select>
              </label>
              <div className="row-2">
                <label className="field">
                  <span>X</span>
                  <input value={roiX} onChange={(event) => setRoiX(event.target.value)} />
                </label>
                <label className="field">
                  <span>Y</span>
                  <input value={roiY} onChange={(event) => setRoiY(event.target.value)} />
                </label>
                <label className="field">
                  <span>W</span>
                  <input value={roiW} onChange={(event) => setRoiW(event.target.value)} />
                </label>
                <label className="field">
                  <span>H</span>
                  <input value={roiH} onChange={(event) => setRoiH(event.target.value)} />
                </label>
              </div>
              <button className="btn btn--primary" type="submit" disabled={busy}>
                Add ROI
              </button>
            </form>
            <div className="setup-list">
              {rois.length ? (
                rois.map((bay) => (
                  <article className="setup-item" key={bay.id}>
                    <div>
                      <h3>{bay.name}</h3>
                      <p className="mono">
                        {bay.zone_kind || bay.bay_type} · [{bay.roi.map((n) => n.toFixed(2)).join(", ")}]
                      </p>
                    </div>
                    <button
                      className="btn btn--danger btn--sm"
                      type="button"
                      onClick={() => void run(() => saveRois(roiDrafts.filter((item) => item.id !== bay.id)))}
                    >
                      Remove
                    </button>
                  </article>
                ))
              ) : (
                <p>No ROI boxes yet. You can also draw them later on the live feed.</p>
              )}
            </div>
          </div>
        ) : null}

        <footer className="setup-panel__foot">
          <button
            className="btn btn--ghost"
            type="button"
            disabled={stepIndex === 0}
            onClick={() => setStep(STEPS[Math.max(0, stepIndex - 1)].id)}
          >
            Back
          </button>
          {stepIndex < STEPS.length - 1 ? (
            <button className="btn btn--primary" type="button" onClick={() => setStep(STEPS[stepIndex + 1].id)}>
              Next
            </button>
          ) : (
            <button className="btn btn--primary" type="button" disabled={busy} onClick={() => void finish()}>
              Finish setup
            </button>
          )}
        </footer>
      </section>
    </div>
  );
}

function CrewCard({ crew, onDelete }: { crew: CrewWithPhoto; onDelete: () => void }) {
  return (
    <article className="setup-item">
      {crew.photoUrl ? <img src={crew.photoUrl} alt="" className="thumb" /> : <div className="thumb" />}
      <div>
        <h3>{crew.display_name}</h3>
        <p>{crew.role}</p>
      </div>
      <button className="btn btn--danger btn--sm" type="button" onClick={onDelete}>
        Remove
      </button>
    </article>
  );
}
