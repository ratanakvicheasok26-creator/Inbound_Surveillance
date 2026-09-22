import { useMemo, useState, type FormEvent } from "react";
import { useAccount } from "../auth";
import { supabase, supabaseConfigured } from "../../lib/supabase";
import { WORKPLACES, parseWorkplaceId, type WorkplaceId } from "../../workplaces";

type AuthMode = "signin" | "signup" | "forgot" | "code" | "confirm";

function modeFromUrl(): AuthMode {
  const params = new URLSearchParams(window.location.search);
  const mode = params.get("mode");
  if (mode === "signup") return "signup";
  if (mode === "reset") return "forgot";
  if (mode === "confirm") return "confirm";
  return "signin";
}

function isEmailUnconfirmed(error: { code?: string; message?: string }) {
  return error.code === "email_not_confirmed" || /email not confirmed/i.test(error.message || "");
}

export function AuthScreen() {
  const { passwordRecovery, beginPasswordRecovery, finishPasswordRecovery } = useAccount();
  const [mode, setMode] = useState<AuthMode>(modeFromUrl);
  const [displayName, setDisplayName] = useState("");
  const [venueName, setVenueName] = useState("");
  const [workplaceType, setWorkplaceType] = useState<WorkplaceId>("garage");
  const [email, setEmail] = useState(() => localStorage.getItem("hub_remembered_email") || "");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [code, setCode] = useState("");
  const [remember, setRemember] = useState(() => localStorage.getItem("hub_remember_me") !== "0");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const settingNewPassword = passwordRecovery;
  const view: AuthMode | "set-password" = settingNewPassword ? "set-password" : mode;

  const title =
    view === "set-password"
      ? "Set a new password"
      : view === "forgot" || view === "code"
        ? "Reset password"
        : view === "confirm"
          ? "Confirm your email"
          : view === "signup"
            ? "Create account"
            : "Sign in";

  const copy =
    view === "set-password"
      ? "Choose a new password for this operator account. Use at least 8 characters."
      : view === "confirm"
        ? "Enter the 6-digit code from your email to confirm this account. Type the code yourself — inbox scanners often burn one-click links."
        : view === "code"
          ? "Enter the 6-digit code from your email, then choose a new password. Type the code yourself — inbox scanners often burn one-click reset links."
          : view === "forgot"
            ? "We email a 6-digit code instead of a magic link. Codes survive email security scanners; one-click links often do not."
            : "Each account keeps profile, workplace type, crew identities, ROI, camera protocol, and stream URLs private. Other operators cannot read them.";

  const redirectTo = useMemo(() => `${window.location.origin}/dashboard.html?mode=reset`, []);

  function go(next: AuthMode, nextNotice = "") {
    setMode(next);
    setError("");
    setNotice(nextNotice);
    setPassword("");
    setConfirmPassword("");
    if (next !== "code" && next !== "confirm") setCode("");
    const url = new URL(window.location.href);
    if (next === "signup") url.searchParams.set("mode", "signup");
    else if (next === "forgot" || next === "code") url.searchParams.set("mode", "reset");
    else if (next === "confirm") url.searchParams.set("mode", "confirm");
    else url.searchParams.delete("mode");
    window.history.replaceState({}, "", url);
  }

  async function resendConfirmation() {
    setError("");
    if (!supabaseConfigured) {
      setError("Add VITE_SUPABASE_ANON_KEY to .env, then restart the dev server.");
      return;
    }
    setBusy(true);
    try {
      const { error: resendError } = await supabase.auth.resend({
        type: "signup",
        email: email.trim(),
      });
      if (resendError) throw resendError;
      setNotice("A new confirmation code is on its way.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send a new code.");
    } finally {
      setBusy(false);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError("");
    if (view !== "forgot") setNotice("");
    if (!supabaseConfigured) {
      setError("Add VITE_SUPABASE_ANON_KEY to .env, then restart the dev server.");
      return;
    }
    setBusy(true);
    try {
      if (view === "set-password") {
        if (password.length < 8) throw new Error("Password must be at least 8 characters.");
        if (password !== confirmPassword) throw new Error("Passwords do not match.");
        const { error: updateError } = await supabase.auth.updateUser({ password });
        if (updateError) throw updateError;
        finishPasswordRecovery();
        return;
      }
      if (view === "signup") {
        if (password.length < 8) throw new Error("Password must be at least 8 characters.");
        const { data, error: signError } = await supabase.auth.signUp({
          email: email.trim(),
          password,
          options: {
            data: {
              display_name: displayName.trim(),
              venue_name: venueName.trim(),
              workplace_type: workplaceType,
            },
          },
        });
        if (signError) throw signError;
        if (!data.session) {
          go("confirm", "Account created. Enter the 6-digit code we just emailed.");
        }
        return;
      }
      if (view === "confirm") {
        const token = code.replace(/\s/g, "");
        if (!/^\d{6}$/.test(token)) throw new Error("Enter the 6-digit code from the email.");
        const { error: otpError } = await supabase.auth.verifyOtp({
          email: email.trim(),
          token,
          type: "signup",
        });
        if (otpError) throw otpError;
        return;
      }
      if (view === "forgot") {
        const { error: resetError } = await supabase.auth.resetPasswordForEmail(email.trim(), {
          redirectTo,
        });
        if (resetError) throw resetError;
        go("code", "If that email has an account, a 6-digit code is on its way.");
        return;
      }
      if (view === "code") {
        const token = code.replace(/\s/g, "");
        if (!/^\d{6}$/.test(token)) throw new Error("Enter the 6-digit code from the email.");
        if (password.length < 8) throw new Error("Password must be at least 8 characters.");
        if (password !== confirmPassword) throw new Error("Passwords do not match.");
        beginPasswordRecovery();
        const { error: otpError } = await supabase.auth.verifyOtp({
          email: email.trim(),
          token,
          type: "recovery",
        });
        if (otpError) {
          finishPasswordRecovery();
          throw otpError;
        }
        const { error: updateError } = await supabase.auth.updateUser({ password });
        if (updateError) throw updateError;
        finishPasswordRecovery();
        return;
      }
      if (view === "signin") {
        if (remember) {
          localStorage.setItem("hub_remembered_email", email.trim());
          localStorage.setItem("hub_remember_me", "1");
        } else {
          localStorage.removeItem("hub_remembered_email");
          localStorage.setItem("hub_remember_me", "0");
        }
      }
      const { error: signError } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password,
      });
      if (signError) {
        if (isEmailUnconfirmed(signError)) {
          go(
            "confirm",
            "This account is not confirmed yet. Enter the 6-digit code from your email, or send a new one.",
          );
          return;
        }
        throw signError;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  const submitLabel =
    busy
      ? "Working…"
      : view === "set-password"
        ? "Save password"
        : view === "signup"
          ? "Create account"
          : view === "forgot"
            ? "Email reset code"
            : view === "confirm"
              ? "Confirm account"
              : view === "code"
                ? "Set new password"
                : "Sign in";

  return (
    <div className="auth-screen">
      <form className="auth-card" onSubmit={(event) => void onSubmit(event)}>
        <div className="auth-card__brand">
          <img
            src="/inb_surveillance.png"
            alt="Inbound Surveillance"
            width={36}
            height={36}
            onError={(e) => {
              const target = e.currentTarget;
              if (!target.dataset.fallback) {
                target.dataset.fallback = "1";
                target.src = "/inb_surveillance-removebg-preview.png";
              }
            }}
          />
          <div>
            <strong>Inbound Surveillance</strong>
            <span>Private operator console</span>
          </div>
        </div>
        <h1>{title}</h1>
        <p>{copy}</p>
        {!supabaseConfigured ? (
          <p className="auth-error">
            Supabase keys are missing. Set <code>VITE_SUPABASE_URL</code> and{" "}
            <code>VITE_SUPABASE_ANON_KEY</code> in <code>.env</code>.
          </p>
        ) : null}

        {view === "signup" ? (
          <>
            <label className="field">
              <span>Display name</span>
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                placeholder="Operator name"
                autoComplete="name"
                required
              />
            </label>
            <label className="field">
              <span>Venue</span>
              <input
                value={venueName}
                onChange={(event) => setVenueName(event.target.value)}
                placeholder="Shop or site name"
                autoComplete="organization"
              />
            </label>
            <fieldset className="field workplace-picker">
              <legend>Workplace</legend>
              <div className="workplace-picker__grid">
                {(["garage", "massage"] as WorkplaceId[]).map((id) => {
                  const profile = WORKPLACES[id];
                  return (
                    <button
                      key={id}
                      type="button"
                      className={workplaceType === id ? "is-on" : ""}
                      onClick={() => setWorkplaceType(id)}
                    >
                      <strong>{profile.label}</strong>
                      <span>
                        {id === "garage"
                          ? "Track employees, bays, and wrench time."
                          : "Track anonymous customer visits by day and week."}
                      </span>
                    </button>
                  );
                })}
              </div>
            </fieldset>
          </>
        ) : null}

        {view !== "set-password" ? (
          <label className="field">
            <span>Email</span>
            <input
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder={WORKPLACES[parseWorkplaceId(workplaceType)].emailPlaceholder}
              autoComplete="email"
              required
              readOnly={view === "code" || view === "confirm"}
            />
          </label>
        ) : null}

        {view === "code" || view === "confirm" ? (
          <label className="field">
            <span>{view === "confirm" ? "Confirmation code" : "Reset code"}</span>
            <input
              className="otp-input"
              value={code}
              onChange={(event) => setCode(event.target.value.replace(/[^\d]/g, "").slice(0, 6))}
              placeholder="6-digit code"
              inputMode="numeric"
              autoComplete="one-time-code"
              required
            />
          </label>
        ) : null}

        {view === "signin" || view === "signup" || view === "code" || view === "set-password" ? (
          <label className="field">
            <span>{view === "signin" ? "Password" : "New password"}</span>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={view === "signin" ? "Password" : "At least 8 characters"}
              autoComplete={view === "signin" ? "current-password" : "new-password"}
              required
            />
          </label>
        ) : null}

        {view === "signin" ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", margin: "2px 0 6px" }}>
            <label style={{ display: "inline-flex", alignItems: "center", gap: "8px", cursor: "pointer", userSelect: "none" }}>
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
                style={{
                  width: "16px",
                  height: "16px",
                  minWidth: "16px",
                  minHeight: "16px",
                  margin: 0,
                  accentColor: "var(--green)",
                  cursor: "pointer",
                }}
              />
              <span style={{ fontSize: "0.8125rem", color: "var(--muted)", whiteSpace: "nowrap" }}>
                Keep me signed in
              </span>
            </label>
          </div>
        ) : null}

        {view === "code" || view === "set-password" ? (
          <label className="field">
            <span>Confirm password</span>
            <input
              type="password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              placeholder="Repeat new password"
              autoComplete="new-password"
              required
            />
          </label>
        ) : null}

        {error ? <p className="auth-error">{error}</p> : null}
        {notice ? <p className="auth-notice">{notice}</p> : null}
        <button className="btn btn--primary" type="submit" disabled={busy || !supabaseConfigured}>
          {submitLabel}
        </button>
        {view === "signin" ? (
          <button className="btn btn--ghost" type="button" onClick={() => go("forgot")}>
            Forgot password?
          </button>
        ) : null}
        {view === "code" ? (
          <button className="btn btn--ghost" type="button" onClick={() => go("forgot")}>
            Send a new code
          </button>
        ) : null}
        {view === "confirm" ? (
          <button className="btn btn--ghost" type="button" onClick={() => void resendConfirmation()}>
            Send a new code
          </button>
        ) : null}
        {view !== "set-password" ? (
          <button
            className="btn btn--ghost"
            type="button"
            onClick={() => go(view === "signup" ? "signin" : view === "signin" ? "signup" : "signin")}
          >
            {view === "signup"
              ? "Already have an account? Sign in"
              : view === "signin"
                ? "Create account"
                : "Back to sign in"}
          </button>
        ) : null}
      </form>
    </div>
  );
}

export function AuthLoading() {
  return (
    <div className="auth-screen">
      <p className="auth-loading">Loading operator console…</p>
    </div>
  );
}
