import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { AccountProvider, useAccount } from "./auth";
import { AuthLoading, AuthScreen } from "./components/AuthScreen";
import { StoreProvider } from "./store";
import "./dashboard.css";

function Root() {
  const { configured, status, session, passwordRecovery } = useAccount();
  if (status === "loading") return <AuthLoading />;
  if (!configured || !session || passwordRecovery) return <AuthScreen />;
  return (
    <StoreProvider>
      <App />
    </StoreProvider>
  );
}

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root");

createRoot(root).render(
  <StrictMode>
    <AccountProvider>
      <Root />
    </AccountProvider>
  </StrictMode>,
);
