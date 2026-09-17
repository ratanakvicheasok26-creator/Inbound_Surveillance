import { useEffect, useRef, useState } from "react";
import { cameraToOps } from "./account";
import { useAccount } from "./auth";
import { AccountSetup } from "./components/AccountSetup";
import { AlertsView } from "./components/AlertsView";
import { CasesView } from "./components/CasesView";
import { ComplaintsView } from "./components/ComplaintsView";
import { LiveView } from "./components/LiveView";
import { PipelineView } from "./components/PipelineView";
import { ProfileChip } from "./components/ProfileChip";
import { RulesView } from "./components/RulesView";
import { ScanAndGoView } from "./components/ScanAndGoView";
import { TelegramPanel } from "./components/TelegramPanel";
import { VisitsView } from "./components/VisitsView";
import { useOps } from "./store";
import type { ViewId } from "./types";
import { workplaceOf } from "../workplaces";

function PipelineStrip() {
  const { state } = useOps();
  const { snapshot } = useAccount();
  const workplace = workplaceOf(snapshot?.profile.workplace_type);
  const recentDetect = state.detections.some((item) => Date.now() - item.ts < 60_000);
  const recentDispatch = state.alerts.some((item) => !item.dismissed && item.telegramState === "sent");

  const steps = [
    { n: "01", title: "Ingest", note: `${state.cameras.length} streams`, live: true, scan: false },
    {
      n: "02",
      title: "Infer",
      note: state.scanning ? "Sampling frames…" : recentDetect ? workplace.pipelineIdle : "Idle",
      live: recentDetect || state.scanning,
      scan: state.scanning,
    },
    { n: "03", title: "Rules", note: `Cooldown ${state.rules.cooldownSec}s`, live: true, scan: false },
    {
      n: "04",
      title: "Dispatch",
      note: recentDispatch ? "Telegram sendMessage" : "No outbound",
      live: recentDispatch,
      scan: false,
    },
  ];

  return (
    <div className="pipeline" aria-label="CCTV AI Telegram pipeline">
      {steps.map((step) => (
        <div
          key={step.n}
          className={`pipeline__step${step.live ? " is-live" : ""}${step.scan ? " is-scan" : ""}`}
        >
          <span className="pipeline__n">{step.n}</span>
          <strong>{step.title}</strong>
          <em>{step.note}</em>
        </div>
      ))}
    </div>
  );
}

function Shell() {
  const { state, hydrateAccount } = useOps();
  const { snapshot } = useAccount();
  const workplace = workplaceOf(snapshot?.profile.workplace_type);
  const [view, setView] = useState<ViewId>("live");
  const [setupOpen, setSetupOpen] = useState(false);
  const hydrateRef = useRef(hydrateAccount);
  hydrateRef.current = hydrateAccount;
  const mainView = view === "bot" ? "live" : view;
  const tabs = workplace.tabs;

  useEffect(() => {
    if (!snapshot) return;
    hydrateRef.current(
      snapshot.profile.venue_name,
      snapshot.cameras.map(cameraToOps),
    );
    if (!snapshot.profile.setup_completed) setSetupOpen(true);
  }, [snapshot]);

  useEffect(() => {
    if (!tabs.some((tab) => tab.id === view)) setView("live");
  }, [tabs, view]);

  return (
    <div className={`ops${view === "bot" ? " is-bot" : ""}`}>
      <header className="ops__top">
        <ProfileChip onOpenSetup={() => setSetupOpen(true)} />
        <p className="venue">
          {snapshot?.profile.venue_name || state.venue}
          <span className="venue-type"> · {workplace.label}</span>
        </p>
      </header>
      <PipelineStrip />
      <div className="ops__body">
        <div className="ops__main">
          <nav className="tabs" aria-label="Console sections">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={`${view === tab.id ? "is-on" : ""}${tab.id === "bot" ? " tab-bot" : ""}`}
                onClick={() => setView(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          {mainView === "live" && <LiveView />}
          {mainView === "rules" && <RulesView />}
          {mainView === "cases" && <CasesView />}
          {mainView === "alerts" && <AlertsView />}
          {mainView === "visits" && <VisitsView />}
          {mainView === "complaints" && <ComplaintsView />}
          {mainView === "pipeline" && <PipelineView />}
          {mainView === "scan-and-go" && <ScanAndGoView />}
        </div>
        <TelegramPanel />
      </div>
      {state.toast && (
        <div className="toast" role="status">
          {state.toast}
        </div>
      )}
      <AccountSetup open={setupOpen} onClose={() => setSetupOpen(false)} />
    </div>
  );
}

export function App() {
  return <Shell />;
}
